"""
IoT Gateway Communication

This module handles communication with IoT edge gateways that collect
sensor data from farms and transmit it to the central system.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import socket
import ssl
import paho.mqtt.client as mqtt
from enum import Enum

from .sensors import SensorReading, SensorType, SensorStatus


class GatewayProtocol(Enum):
    """Supported gateway communication protocols"""
    MQTT = "mqtt"
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    LORA_WAN = "lora_wan"
    MODBUS_TCP = "modbus_tcp"


class GatewayStatus(Enum):
    """Gateway connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    CONNECTING = "connecting"
    MAINTENANCE = "maintenance"


@dataclass
class GatewayConfig:
    """Gateway configuration parameters"""
    gateway_id: str
    name: str
    location: tuple[float, float]  # lat, lon
    protocol: GatewayProtocol
    connection_params: Dict[str, Any]
    sensors: List[str]  # List of sensor IDs
    data_collection_interval: int = 300  # seconds
    heartbeat_interval: int = 60  # seconds
    max_offline_duration: int = 3600  # seconds


@dataclass
class GatewayData:
    """Data packet from IoT gateway"""
    gateway_id: str
    timestamp: datetime
    sensors_data: List[SensorReading]
    gateway_status: Dict[str, Any]
    signal_strength: float
    battery_level: Optional[float] = None
    error_codes: Optional[List[str]] = None


class BaseGateway(ABC):
    """Abstract base class for IoT gateway implementations"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.status = GatewayStatus.OFFLINE
        self.last_seen = None
        self.data_callbacks: List[Callable] = []
        self.logger = logging.getLogger(f"gateway.{config.gateway_id}")
        self.connection = None
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the gateway"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the gateway"""
        pass
    
    @abstractmethod
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to the gateway"""
        pass
    
    @abstractmethod
    async def receive_data(self) -> Optional[GatewayData]:
        """Receive data from the gateway"""
        pass
    
    def add_data_callback(self, callback: Callable[[GatewayData], None]) -> None:
        """Add callback for received data"""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable[[GatewayData], None]) -> None:
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def _notify_callbacks(self, data: GatewayData) -> None:
        """Notify all registered callbacks"""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in data callback: {e}")
    
    def is_online(self) -> bool:
        """Check if gateway is currently online"""
        return self.status == GatewayStatus.ONLINE and self.is_connected
    
    def get_offline_duration(self) -> Optional[timedelta]:
        """Get duration since gateway went offline"""
        if self.last_seen and self.status == GatewayStatus.OFFLINE:
            return datetime.now() - self.last_seen
        return None


class MQTTGateway(BaseGateway):
    """MQTT-based IoT gateway implementation"""
    
    def __init__(self, config: GatewayConfig):
        super().__init__(config)
        self.client = mqtt.Client(client_id=f"gateway_{config.gateway_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # MQTT topics
        self.data_topic = f"sensors/{config.gateway_id}/data"
        self.status_topic = f"sensors/{config.gateway_id}/status"
        self.command_topic = f"sensors/{config.gateway_id}/commands"
    
    async def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.status = GatewayStatus.CONNECTING
            
            # Set connection parameters
            params = self.config.connection_params
            host = params.get('host', 'localhost')
            port = params.get('port', 1883)
            keepalive = params.get('keepalive', 60)
            
            # Set credentials if provided
            if 'username' in params and 'password' in params:
                self.client.username_pw_set(params['username'], params['password'])
            
            # Enable SSL/TLS if configured
            if params.get('use_ssl', False):
                self.client.tls_set()
            
            # Connect to broker
            self.client.connect(host, port, keepalive)
            self.client.loop_start()
            
            # Subscribe to topics
            self.client.subscribe(self.data_topic)
            self.client.subscribe(self.status_topic)
            
            self.is_connected = True
            self.status = GatewayStatus.ONLINE
            self.last_seen = datetime.now()
            
            self.logger.info(f"Connected to MQTT gateway {self.config.gateway_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT gateway: {e}")
            self.status = GatewayStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            self.status = GatewayStatus.OFFLINE
            self.logger.info(f"Disconnected from MQTT gateway {self.config.gateway_id}")
    
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command to gateway via MQTT"""
        try:
            if not self.is_connected:
                return False
            
            command_json = json.dumps(command)
            result = self.client.publish(self.command_topic, command_json)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            return False
    
    async def receive_data(self) -> Optional[GatewayData]:
        """Receive data from MQTT (handled by callback)"""
        # Data is received via callback, this method is not used for MQTT
        return None
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.is_connected = True
            self.status = GatewayStatus.ONLINE
            self.logger.info(f"MQTT gateway {self.config.gateway_id} connected")
        else:
            self.status = GatewayStatus.ERROR
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message received callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            if topic == self.data_topic:
                # Parse sensor data
                gateway_data = self._parse_sensor_data(payload)
                if gateway_data:
                    self._notify_callbacks(gateway_data)
                    self.last_seen = datetime.now()
            
            elif topic == self.status_topic:
                # Handle status updates
                self._handle_status_update(payload)
                
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.is_connected = False
        self.status = GatewayStatus.OFFLINE
        self.logger.warning(f"MQTT gateway {self.config.gateway_id} disconnected")
    
    def _parse_sensor_data(self, payload: Dict[str, Any]) -> Optional[GatewayData]:
        """Parse sensor data from MQTT payload"""
        try:
            sensors_data = []
            
            for sensor_data in payload.get('sensors', []):
                sensor = SensorReading(
                    sensor_id=sensor_data['sensor_id'],
                    sensor_type=SensorType(sensor_data['sensor_type']),
                    timestamp=datetime.fromisoformat(sensor_data['timestamp']),
                    value=sensor_data['value'],
                    unit=sensor_data['unit'],
                    quality_score=sensor_data.get('quality_score', 1.0),
                    location=tuple(sensor_data.get('location', [0, 0])),
                    metadata=sensor_data.get('metadata', {})
                )
                sensors_data.append(sensor)
            
            gateway_data = GatewayData(
                gateway_id=self.config.gateway_id,
                timestamp=datetime.fromisoformat(payload['timestamp']),
                sensors_data=sensors_data,
                gateway_status=payload.get('gateway_status', {}),
                signal_strength=payload.get('signal_strength', 0.0),
                battery_level=payload.get('battery_level'),
                error_codes=payload.get('error_codes')
            )
            
            return gateway_data
            
        except Exception as e:
            self.logger.error(f"Error parsing sensor data: {e}")
            return None
    
    def _handle_status_update(self, payload: Dict[str, Any]) -> None:
        """Handle gateway status updates"""
        try:
            status = payload.get('status', 'unknown')
            if status in [s.value for s in GatewayStatus]:
                self.status = GatewayStatus(status)
                self.last_seen = datetime.now()
        except Exception as e:
            self.logger.error(f"Error handling status update: {e}")


class HTTPRestGateway(BaseGateway):
    """HTTP REST API based IoT gateway implementation"""
    
    def __init__(self, config: GatewayConfig):
        super().__init__(config)
        self.base_url = config.connection_params.get('base_url', 'http://localhost:8080')
        self.api_key = config.connection_params.get('api_key')
        self.session = None
    
    async def connect(self) -> bool:
        """Establish HTTP session"""
        try:
            import aiohttp
            
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(headers=headers)
            
            # Test connection with health check
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    self.is_connected = True
                    self.status = GatewayStatus.ONLINE
                    self.last_seen = datetime.now()
                    return True
                else:
                    self.status = GatewayStatus.ERROR
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to HTTP gateway: {e}")
            self.status = GatewayStatus.ERROR
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.is_connected = False
            self.status = GatewayStatus.OFFLINE
    
    async def send_command(self, command: Dict[str, Any]) -> bool:
        """Send command via HTTP POST"""
        try:
            if not self.session:
                return False
            
            url = f"{self.base_url}/gateway/{self.config.gateway_id}/command"
            async with self.session.post(url, json=command) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Failed to send HTTP command: {e}")
            return False
    
    async def receive_data(self) -> Optional[GatewayData]:
        """Poll for data via HTTP GET"""
        try:
            if not self.session:
                return None
            
            url = f"{self.base_url}/gateway/{self.config.gateway_id}/data"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_http_data(data)
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to receive HTTP data: {e}")
            return None
    
    def _parse_http_data(self, data: Dict[str, Any]) -> Optional[GatewayData]:
        """Parse data from HTTP response"""
        # Similar to MQTT parsing but adapted for HTTP response format
        try:
            sensors_data = []
            
            for sensor_data in data.get('sensors', []):
                sensor = SensorReading(
                    sensor_id=sensor_data['sensor_id'],
                    sensor_type=SensorType(sensor_data['sensor_type']),
                    timestamp=datetime.fromisoformat(sensor_data['timestamp']),
                    value=sensor_data['value'],
                    unit=sensor_data['unit'],
                    quality_score=sensor_data.get('quality_score', 1.0),
                    location=tuple(sensor_data.get('location', [0, 0])),
                    metadata=sensor_data.get('metadata', {})
                )
                sensors_data.append(sensor)
            
            return GatewayData(
                gateway_id=self.config.gateway_id,
                timestamp=datetime.fromisoformat(data['timestamp']),
                sensors_data=sensors_data,
                gateway_status=data.get('gateway_status', {}),
                signal_strength=data.get('signal_strength', 0.0),
                battery_level=data.get('battery_level'),
                error_codes=data.get('error_codes')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing HTTP data: {e}")
            return None


class GatewayManager:
    """Manager for multiple IoT gateways"""
    
    def __init__(self):
        self.gateways: Dict[str, BaseGateway] = {}
        self.data_handlers: List[Callable[[GatewayData], None]] = []
        self.logger = logging.getLogger("gateway_manager")
        self.is_running = False
    
    def add_gateway(self, gateway: BaseGateway) -> None:
        """Add a gateway to the manager"""
        gateway_id = gateway.config.gateway_id
        self.gateways[gateway_id] = gateway
        
        # Subscribe to gateway data
        gateway.add_data_callback(self._handle_gateway_data)
        
        self.logger.info(f"Added gateway {gateway_id}")
    
    def remove_gateway(self, gateway_id: str) -> None:
        """Remove a gateway from the manager"""
        if gateway_id in self.gateways:
            gateway = self.gateways[gateway_id]
            gateway.remove_data_callback(self._handle_gateway_data)
            del self.gateways[gateway_id]
            self.logger.info(f"Removed gateway {gateway_id}")
    
    async def start_all_gateways(self) -> None:
        """Start all gateways"""
        self.logger.info("Starting all gateways...")
        
        for gateway_id, gateway in self.gateways.items():
            try:
                success = await gateway.connect()
                if success:
                    self.logger.info(f"Gateway {gateway_id} started successfully")
                else:
                    self.logger.error(f"Failed to start gateway {gateway_id}")
            except Exception as e:
                self.logger.error(f"Error starting gateway {gateway_id}: {e}")
        
        self.is_running = True
    
    async def stop_all_gateways(self) -> None:
        """Stop all gateways"""
        self.logger.info("Stopping all gateways...")
        
        for gateway_id, gateway in self.gateways.items():
            try:
                await gateway.disconnect()
                self.logger.info(f"Gateway {gateway_id} stopped")
            except Exception as e:
                self.logger.error(f"Error stopping gateway {gateway_id}: {e}")
        
        self.is_running = False
    
    def add_data_handler(self, handler: Callable[[GatewayData], None]) -> None:
        """Add handler for gateway data"""
        self.data_handlers.append(handler)
    
    def remove_data_handler(self, handler: Callable[[GatewayData], None]) -> None:
        """Remove data handler"""
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)
    
    def _handle_gateway_data(self, data: GatewayData) -> None:
        """Handle data from any gateway"""
        # Notify all registered handlers
        for handler in self.data_handlers:
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Error in data handler: {e}")
    
    def get_gateway_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all gateways"""
        status = {}
        
        for gateway_id, gateway in self.gateways.items():
            status[gateway_id] = {
                'status': gateway.status.value,
                'is_connected': gateway.is_connected,
                'last_seen': gateway.last_seen.isoformat() if gateway.last_seen else None,
                'offline_duration': str(gateway.get_offline_duration()) if gateway.get_offline_duration() else None,
                'protocol': gateway.config.protocol.value,
                'location': gateway.config.location
            }
        
        return status
    
    async def send_broadcast_command(self, command: Dict[str, Any]) -> Dict[str, bool]:
        """Send command to all gateways"""
        results = {}
        
        for gateway_id, gateway in self.gateways.items():
            if gateway.is_online():
                results[gateway_id] = await gateway.send_command(command)
            else:
                results[gateway_id] = False
        
        return results


def create_gateway(config: GatewayConfig) -> BaseGateway:
    """Factory function to create gateway instances"""
    if config.protocol == GatewayProtocol.MQTT:
        return MQTTGateway(config)
    elif config.protocol == GatewayProtocol.HTTP_REST:
        return HTTPRestGateway(config)
    else:
        raise ValueError(f"Unsupported gateway protocol: {config.protocol}")


# Example usage and sample configurations
def create_sample_gateway_configs() -> List[GatewayConfig]:
    """Create sample gateway configurations for testing"""
    return [
        GatewayConfig(
            gateway_id="farm_001_gateway",
            name="Kampala Region Farm Gateway",
            location=(0.3476, 32.5825),  # Kampala coordinates
            protocol=GatewayProtocol.MQTT,
            connection_params={
                'host': 'mqtt.agriculture.ug',
                'port': 1883,
                'username': 'farm_001',
                'password': 'sensor_password',
                'use_ssl': True
            },
            sensors=['soil_temp_001', 'soil_moisture_001', 'air_temp_001', 'humidity_001'],
            data_collection_interval=300,
            heartbeat_interval=60
        ),
        GatewayConfig(
            gateway_id="farm_002_gateway",
            name="Masaka Region Farm Gateway",
            location=(-0.3476, 31.7414),  # Masaka coordinates
            protocol=GatewayProtocol.HTTP_REST,
            connection_params={
                'base_url': 'https://api.agriculture.ug/v1',
                'api_key': 'farm_002_api_key'
            },
            sensors=['weather_station_002', 'soil_analyzer_002'],
            data_collection_interval=600,
            heartbeat_interval=120
        )
    ]