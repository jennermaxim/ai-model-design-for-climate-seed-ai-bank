"""
IoT Communication Protocols

This module implements various communication protocols used for IoT sensor
networks in agricultural applications.
"""

import asyncio
import struct
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import socket
import serial
from crccheck.crc import Crc16, Crc32


class ProtocolType(Enum):
    """Supported IoT communication protocols"""
    MODBUS_RTU = "modbus_rtu"
    MODBUS_TCP = "modbus_tcp" 
    LORA_WAN = "lora_wan"
    ZIGBEE = "zigbee"
    WIFI_UDP = "wifi_udp"
    BLUETOOTH_LE = "bluetooth_le"
    CUSTOM_SERIAL = "custom_serial"


class MessageType(Enum):
    """Types of protocol messages"""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    COMMAND = "command"
    ACKNOWLEDGMENT = "acknowledgment"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CONFIGURATION = "configuration"


@dataclass
class ProtocolMessage:
    """Generic protocol message structure"""
    message_type: MessageType
    source_id: str
    destination_id: str
    timestamp: datetime
    payload: Dict[str, Any]
    sequence_number: int = 0
    checksum: Optional[str] = None
    retry_count: int = 0


@dataclass 
class ProtocolConfig:
    """Configuration for protocol handlers"""
    protocol_type: ProtocolType
    connection_params: Dict[str, Any]
    message_timeout: float = 5.0
    max_retries: int = 3
    heartbeat_interval: float = 30.0
    buffer_size: int = 1024


class BaseProtocol(ABC):
    """Abstract base class for IoT communication protocols"""
    
    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.logger = logging.getLogger(f"protocol.{config.protocol_type.value}")
        self.is_connected = False
        self.message_handlers: Dict[MessageType, callable] = {}
        self.sequence_counter = 0
        self.pending_messages: Dict[int, ProtocolMessage] = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection using the protocol"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close protocol connection"""
        pass
    
    @abstractmethod
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send message using the protocol"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive message using the protocol"""
        pass
    
    def add_message_handler(self, message_type: MessageType, handler: callable) -> None:
        """Add handler for specific message types"""
        self.message_handlers[message_type] = handler
    
    def _get_next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_counter += 1
        return self.sequence_counter
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity"""
        crc = Crc16.calc(data)
        return f"{crc:04x}"
    
    def _verify_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """Verify data integrity using checksum"""
        calculated = self._calculate_checksum(data)
        return calculated.lower() == expected_checksum.lower()


class ModbusTCPProtocol(BaseProtocol):
    """Modbus TCP protocol implementation for industrial sensors"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.socket = None
        self.server_host = config.connection_params.get('host', 'localhost')
        self.server_port = config.connection_params.get('port', 502)
        self.unit_id = config.connection_params.get('unit_id', 1)
        self.transaction_id = 0
    
    async def connect(self) -> bool:
        """Connect to Modbus TCP server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.message_timeout)
            self.socket.connect((self.server_host, self.server_port))
            self.is_connected = True
            self.logger.info(f"Connected to Modbus TCP server {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Modbus TCP server: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Modbus TCP server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.is_connected = False
            self.logger.info("Disconnected from Modbus TCP server")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send Modbus TCP message"""
        try:
            if not self.is_connected:
                return False
            
            # Build Modbus TCP frame
            modbus_frame = self._build_modbus_frame(message)
            self.socket.send(modbus_frame)
            
            # Wait for response
            response_data = self.socket.recv(self.config.buffer_size)
            response = self._parse_modbus_response(response_data)
            
            if response:
                # Handle response in message handler
                if MessageType.DATA_RESPONSE in self.message_handlers:
                    self.message_handlers[MessageType.DATA_RESPONSE](response)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Modbus TCP message: {e}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive Modbus TCP message"""
        try:
            if not self.is_connected:
                return None
            
            data = self.socket.recv(self.config.buffer_size)
            if data:
                return self._parse_modbus_response(data)
            
        except Exception as e:
            self.logger.error(f"Error receiving Modbus TCP message: {e}")
        
        return None
    
    def _build_modbus_frame(self, message: ProtocolMessage) -> bytes:
        """Build Modbus TCP frame from message"""
        self.transaction_id += 1
        
        # Modbus TCP header (7 bytes) + PDU
        payload = message.payload
        function_code = payload.get('function_code', 3)  # Read holding registers
        start_address = payload.get('start_address', 0)
        quantity = payload.get('quantity', 1)
        
        # Build PDU
        if function_code == 3:  # Read holding registers
            pdu = struct.pack('>BBH', function_code, start_address, quantity)
        else:
            pdu = b'\x03\x00\x00\x00\x01'  # Default read
        
        # Build MBAP header
        protocol_id = 0
        length = len(pdu) + 1  # PDU + unit identifier
        mbap = struct.pack('>HHHB', self.transaction_id, protocol_id, length, self.unit_id)
        
        return mbap + pdu
    
    def _parse_modbus_response(self, data: bytes) -> Optional[ProtocolMessage]:
        """Parse Modbus TCP response"""
        try:
            if len(data) < 8:
                return None
            
            # Parse MBAP header
            transaction_id, protocol_id, length, unit_id = struct.unpack('>HHHB', data[:7])
            
            # Parse PDU
            pdu = data[7:]
            if len(pdu) >= 2:
                function_code = pdu[0]
                if function_code == 3:  # Read holding registers response
                    byte_count = pdu[1]
                    register_values = []
                    for i in range(0, byte_count, 2):
                        if i + 1 < len(pdu) - 2:
                            value = struct.unpack('>H', pdu[2+i:4+i])[0]
                            register_values.append(value)
                    
                    return ProtocolMessage(
                        message_type=MessageType.DATA_RESPONSE,
                        source_id=f"modbus_unit_{unit_id}",
                        destination_id="gateway",
                        timestamp=datetime.now(),
                        payload={
                            'function_code': function_code,
                            'values': register_values,
                            'transaction_id': transaction_id
                        },
                        sequence_number=transaction_id
                    )
            
        except Exception as e:
            self.logger.error(f"Error parsing Modbus response: {e}")
        
        return None


class LoRaWANProtocol(BaseProtocol):
    """LoRaWAN protocol implementation for long-range sensor networks"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.device_eui = config.connection_params.get('device_eui')
        self.app_eui = config.connection_params.get('app_eui')
        self.app_key = config.connection_params.get('app_key')
        self.gateway_connection = None
        self.frame_counter = 0
    
    async def connect(self) -> bool:
        """Connect to LoRaWAN network"""
        try:
            # In a real implementation, this would involve LoRaWAN join procedure
            # For now, simulate connection
            self.is_connected = True
            self.logger.info(f"Connected to LoRaWAN network with DevEUI: {self.device_eui}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to LoRaWAN network: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from LoRaWAN network"""
        self.is_connected = False
        self.logger.info("Disconnected from LoRaWAN network")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send LoRaWAN message"""
        try:
            if not self.is_connected:
                return False
            
            # Build LoRaWAN frame
            lorawan_frame = self._build_lorawan_frame(message)
            
            # In real implementation, this would send via LoRa radio
            # For simulation, just log the frame
            self.logger.info(f"Sending LoRaWAN frame: {lorawan_frame.hex()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending LoRaWAN message: {e}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive LoRaWAN message"""
        # In real implementation, this would listen for incoming frames
        # For now, return None as LoRaWAN is typically uplink-only for sensors
        return None
    
    def _build_lorawan_frame(self, message: ProtocolMessage) -> bytes:
        """Build LoRaWAN data frame"""
        # Simplified LoRaWAN frame structure
        # Real implementation would include proper encryption and formatting
        
        payload_data = json.dumps(message.payload).encode('utf-8')
        
        # LoRaWAN frame header (simplified)
        mhdr = 0x40  # Unconfirmed data up
        dev_addr = b'\x01\x02\x03\x04'  # Device address
        fctrl = 0x00  # Frame control
        fcnt = self.frame_counter.to_bytes(2, 'little')
        
        # Frame header
        fhdr = dev_addr + bytes([fctrl]) + fcnt
        
        # Frame payload
        frm_payload = payload_data
        
        # MIC calculation (simplified - real implementation would use AES-CMAC)
        mic = self._calculate_mic(mhdr.to_bytes(1, 'big') + fhdr + frm_payload)
        
        self.frame_counter += 1
        
        return mhdr.to_bytes(1, 'big') + fhdr + frm_payload + mic
    
    def _calculate_mic(self, data: bytes) -> bytes:
        """Calculate Message Integrity Code (simplified)"""
        # In real LoRaWAN, this would use AES-CMAC with network session key
        # For simulation, use CRC32
        crc = Crc32.calc(data)
        return crc.to_bytes(4, 'little')


class CustomSerialProtocol(BaseProtocol):
    """Custom serial protocol for proprietary sensor devices"""
    
    def __init__(self, config: ProtocolConfig):
        super().__init__(config)
        self.serial_port = None
        self.port_name = config.connection_params.get('port', '/dev/ttyUSB0')
        self.baud_rate = config.connection_params.get('baud_rate', 9600)
        self.data_bits = config.connection_params.get('data_bits', 8)
        self.stop_bits = config.connection_params.get('stop_bits', 1)
        self.parity = config.connection_params.get('parity', 'N')
        
        # Custom protocol parameters
        self.start_byte = 0xAA
        self.end_byte = 0x55
        self.escape_byte = 0xCC
    
    async def connect(self) -> bool:
        """Connect to serial port"""
        try:
            self.serial_port = serial.Serial(
                port=self.port_name,
                baudrate=self.baud_rate,
                bytesize=self.data_bits,
                stopbits=self.stop_bits,
                parity=self.parity,
                timeout=self.config.message_timeout
            )
            
            if self.serial_port.is_open:
                self.is_connected = True
                self.logger.info(f"Connected to serial port {self.port_name}")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to serial port: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from serial port"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.is_connected = False
            self.logger.info(f"Disconnected from serial port {self.port_name}")
    
    async def send_message(self, message: ProtocolMessage) -> bool:
        """Send message via custom serial protocol"""
        try:
            if not self.is_connected:
                return False
            
            frame = self._build_custom_frame(message)
            self.serial_port.write(frame)
            self.serial_port.flush()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending serial message: {e}")
            return False
    
    async def receive_message(self) -> Optional[ProtocolMessage]:
        """Receive message via custom serial protocol"""
        try:
            if not self.is_connected:
                return None
            
            # Wait for start byte
            while True:
                byte = self.serial_port.read(1)
                if len(byte) == 0:  # Timeout
                    return None
                if byte[0] == self.start_byte:
                    break
            
            # Read frame
            frame_data = bytearray()
            escaped = False
            
            while True:
                byte = self.serial_port.read(1)
                if len(byte) == 0:  # Timeout
                    return None
                
                byte_val = byte[0]
                
                if escaped:
                    frame_data.append(byte_val)
                    escaped = False
                elif byte_val == self.escape_byte:
                    escaped = True
                elif byte_val == self.end_byte:
                    break
                else:
                    frame_data.append(byte_val)
            
            # Parse frame
            return self._parse_custom_frame(bytes(frame_data))
            
        except Exception as e:
            self.logger.error(f"Error receiving serial message: {e}")
            return None
    
    def _build_custom_frame(self, message: ProtocolMessage) -> bytes:
        """Build custom serial protocol frame"""
        # Frame structure: START | TYPE | SEQ | LENGTH | PAYLOAD | CRC | END
        
        message_type_byte = list(MessageType).index(message.message_type)
        seq_bytes = message.sequence_number.to_bytes(2, 'big')
        
        payload_json = json.dumps({
            'source': message.source_id,
            'destination': message.destination_id,
            'timestamp': message.timestamp.isoformat(),
            'payload': message.payload
        }).encode('utf-8')
        
        length_bytes = len(payload_json).to_bytes(2, 'big')
        
        # Build frame data
        frame_data = (
            bytes([message_type_byte]) +
            seq_bytes +
            length_bytes +
            payload_json
        )
        
        # Calculate CRC
        crc = Crc16.calc(frame_data)
        crc_bytes = crc.to_bytes(2, 'big')
        
        # Escape special bytes in frame data and CRC
        escaped_data = self._escape_bytes(frame_data + crc_bytes)
        
        # Build complete frame
        return bytes([self.start_byte]) + escaped_data + bytes([self.end_byte])
    
    def _parse_custom_frame(self, frame_data: bytes) -> Optional[ProtocolMessage]:
        """Parse custom serial protocol frame"""
        try:
            if len(frame_data) < 7:  # Minimum frame size
                return None
            
            # Extract components
            message_type_byte = frame_data[0]
            seq_number = int.from_bytes(frame_data[1:3], 'big')
            payload_length = int.from_bytes(frame_data[3:5], 'big')
            
            if len(frame_data) < 5 + payload_length + 2:
                return None
            
            payload_data = frame_data[5:5+payload_length]
            crc_received = int.from_bytes(frame_data[5+payload_length:7+payload_length], 'big')
            
            # Verify CRC
            crc_calculated = Crc16.calc(frame_data[:5+payload_length])
            if crc_calculated != crc_received:
                self.logger.warning("CRC mismatch in received frame")
                return None
            
            # Parse payload
            payload_json = json.loads(payload_data.decode('utf-8'))
            
            # Get message type
            message_type = list(MessageType)[message_type_byte]
            
            return ProtocolMessage(
                message_type=message_type,
                source_id=payload_json['source'],
                destination_id=payload_json['destination'],
                timestamp=datetime.fromisoformat(payload_json['timestamp']),
                payload=payload_json['payload'],
                sequence_number=seq_number
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing custom frame: {e}")
            return None
    
    def _escape_bytes(self, data: bytes) -> bytes:
        """Escape special bytes in frame data"""
        escaped = bytearray()
        
        for byte in data:
            if byte in [self.start_byte, self.end_byte, self.escape_byte]:
                escaped.append(self.escape_byte)
                escaped.append(byte)
            else:
                escaped.append(byte)
        
        return bytes(escaped)


class ProtocolManager:
    """Manager for multiple protocol handlers"""
    
    def __init__(self):
        self.protocols: Dict[str, BaseProtocol] = {}
        self.logger = logging.getLogger("protocol_manager")
        self.message_router: Dict[str, str] = {}  # device_id -> protocol_name
    
    def add_protocol(self, name: str, protocol: BaseProtocol) -> None:
        """Add a protocol handler"""
        self.protocols[name] = protocol
        self.logger.info(f"Added protocol handler: {name}")
    
    def remove_protocol(self, name: str) -> None:
        """Remove a protocol handler"""
        if name in self.protocols:
            del self.protocols[name]
            self.logger.info(f"Removed protocol handler: {name}")
    
    def route_device_to_protocol(self, device_id: str, protocol_name: str) -> None:
        """Route device to specific protocol"""
        if protocol_name in self.protocols:
            self.message_router[device_id] = protocol_name
            self.logger.info(f"Routed device {device_id} to protocol {protocol_name}")
    
    async def send_to_device(self, device_id: str, message: ProtocolMessage) -> bool:
        """Send message to device using appropriate protocol"""
        protocol_name = self.message_router.get(device_id)
        if protocol_name and protocol_name in self.protocols:
            protocol = self.protocols[protocol_name]
            return await protocol.send_message(message)
        else:
            self.logger.error(f"No protocol route found for device {device_id}")
            return False
    
    async def start_all_protocols(self) -> None:
        """Connect all protocols"""
        for name, protocol in self.protocols.items():
            try:
                success = await protocol.connect()
                if success:
                    self.logger.info(f"Protocol {name} connected successfully")
                else:
                    self.logger.error(f"Failed to connect protocol {name}")
            except Exception as e:
                self.logger.error(f"Error connecting protocol {name}: {e}")
    
    async def stop_all_protocols(self) -> None:
        """Disconnect all protocols"""
        for name, protocol in self.protocols.items():
            try:
                await protocol.disconnect()
                self.logger.info(f"Protocol {name} disconnected")
            except Exception as e:
                self.logger.error(f"Error disconnecting protocol {name}: {e}")


# Factory function for creating protocol instances
def create_protocol(config: ProtocolConfig) -> BaseProtocol:
    """Factory function to create protocol instances"""
    if config.protocol_type == ProtocolType.MODBUS_TCP:
        return ModbusTCPProtocol(config)
    elif config.protocol_type == ProtocolType.LORA_WAN:
        return LoRaWANProtocol(config)
    elif config.protocol_type == ProtocolType.CUSTOM_SERIAL:
        return CustomSerialProtocol(config)
    else:
        raise ValueError(f"Unsupported protocol type: {config.protocol_type}")


# Sample configurations for testing
def create_sample_protocol_configs() -> List[ProtocolConfig]:
    """Create sample protocol configurations"""
    return [
        ProtocolConfig(
            protocol_type=ProtocolType.MODBUS_TCP,
            connection_params={
                'host': '192.168.1.100',
                'port': 502,
                'unit_id': 1
            },
            message_timeout=5.0,
            max_retries=3
        ),
        ProtocolConfig(
            protocol_type=ProtocolType.LORA_WAN,
            connection_params={
                'device_eui': '0004A30B001B7AD2',
                'app_eui': '70B3D57ED00001A6',
                'app_key': '36AB03413F4573F4A973A6BFEAB18A09'
            },
            message_timeout=30.0,  # Longer timeout for LoRaWAN
            max_retries=2
        ),
        ProtocolConfig(
            protocol_type=ProtocolType.CUSTOM_SERIAL,
            connection_params={
                'port': '/dev/ttyUSB0',
                'baud_rate': 9600,
                'data_bits': 8,
                'stop_bits': 1,
                'parity': 'N'
            },
            message_timeout=2.0,
            max_retries=3
        )
    ]