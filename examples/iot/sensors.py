"""
IoT Sensor Data Processing

This module handles IoT sensor data ingestion, processing, and real-time monitoring
for the Climate-Adaptive Seed AI Bank.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from enum import Enum


class SensorType(Enum):
    """Enumeration of supported sensor types"""
    WEATHER_STATION = "weather"
    SOIL_SENSOR = "soil"  
    PLANT_MONITOR = "plant"
    WATER_SENSOR = "water"
    PEST_TRAP = "pest"


class DataQuality(Enum):
    """Data quality indicators"""
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class SensorStatus(Enum):
    """Sensor operational status"""
    NORMAL = "normal"
    WARNING = "warning"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class SensorLocation:
    """Geographic location of sensor"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    field_id: Optional[str] = None


@dataclass
class DeviceStatus:
    """Device health and status information"""
    battery_level: float  # 0-100%
    signal_strength: float  # dBm
    internal_temperature: float  # Celsius
    uptime_hours: float
    last_calibration: Optional[datetime] = None


@dataclass
class SensorReading:
    """Individual sensor measurement"""
    sensor_id: str
    timestamp: datetime
    location: SensorLocation
    sensor_type: SensorType
    measurements: Dict[str, float]
    quality_flags: Dict[str, DataQuality]
    device_status: DeviceStatus
    metadata: Optional[Dict[str, Any]] = None


class BaseSensor(ABC):
    """Abstract base class for all sensor types"""
    
    def __init__(self, sensor_id: str, location: SensorLocation):
        self.sensor_id = sensor_id
        self.location = location
        self.calibration_date = datetime.now()
        self.measurement_interval = 300  # seconds (5 minutes default)
        self.is_active = True
        self.error_count = 0
        self.last_reading = None
    
    @abstractmethod
    def read_sensors(self) -> Dict[str, float]:
        """Read raw sensor values"""
        pass
    
    @abstractmethod
    def validate_reading(self, measurements: Dict[str, float]) -> Dict[str, DataQuality]:
        """Validate sensor readings and assign quality flags"""
        pass
    
    @abstractmethod
    def get_sensor_type(self) -> SensorType:
        """Return the sensor type"""
        pass
    
    def get_device_status(self) -> DeviceStatus:
        """Get current device status"""
        return DeviceStatus(
            battery_level=self._read_battery_level(),
            signal_strength=self._read_signal_strength(),
            internal_temperature=self._read_internal_temperature(),
            uptime_hours=self._calculate_uptime(),
            last_calibration=self.calibration_date
        )
    
    def take_reading(self) -> SensorReading:
        """Take a complete sensor reading with validation"""
        try:
            # Read sensor values
            measurements = self.read_sensors()
            
            # Validate readings
            quality_flags = self.validate_reading(measurements)
            
            # Get device status
            device_status = self.get_device_status()
            
            # Create sensor reading
            reading = SensorReading(
                sensor_id=self.sensor_id,
                timestamp=datetime.now(),
                location=self.location,
                sensor_type=self.get_sensor_type(),
                measurements=measurements,
                quality_flags=quality_flags,
                device_status=device_status
            )
            
            self.last_reading = reading
            self.error_count = 0  # Reset error count on successful reading
            
            return reading
            
        except Exception as e:
            self.error_count += 1
            self._handle_sensor_error(e)
            raise
    
    def _read_battery_level(self) -> float:
        """Read battery level (simulated)"""
        # Simulate battery drain over time
        hours_since_start = self._calculate_uptime()
        base_level = 100 - (hours_since_start * 0.01)  # 1% per 100 hours
        return max(base_level + np.random.normal(0, 2), 0)
    
    def _read_signal_strength(self) -> float:
        """Read signal strength (simulated)"""
        return np.random.normal(-70, 10)  # Typical cellular signal strength
    
    def _read_internal_temperature(self) -> float:
        """Read internal device temperature (simulated)"""
        return np.random.normal(35, 5)  # Device runs warm
    
    def _calculate_uptime(self) -> float:
        """Calculate device uptime in hours"""
        return (datetime.now() - self.calibration_date).total_seconds() / 3600
    
    def _handle_sensor_error(self, error: Exception):
        """Handle sensor errors"""
        print(f"Sensor {self.sensor_id} error: {error}")
        if self.error_count > 5:
            self.is_active = False


class WeatherStationSensor(BaseSensor):
    """Weather station with multiple meteorological sensors"""
    
    def __init__(self, sensor_id: str, location: SensorLocation):
        super().__init__(sensor_id, location)
        self.measurement_interval = 900  # 15 minutes for weather data
    
    def read_sensors(self) -> Dict[str, float]:
        """Read weather sensor values"""
        # Simulate realistic weather readings for Uganda
        base_temp = 25 + 5 * np.sin((datetime.now().hour - 12) * np.pi / 12)  # Daily temp cycle
        
        measurements = {
            'temperature_celsius': base_temp + np.random.normal(0, 2),
            'humidity_percent': min(max(np.random.normal(75, 15), 20), 100),
            'rainfall_mm': max(np.random.exponential(0.1), 0),  # Most readings are 0
            'wind_speed_ms': max(np.random.exponential(2), 0),
            'wind_direction_degrees': np.random.uniform(0, 360),
            'pressure_hpa': np.random.normal(1013, 10),
            'solar_radiation_wm2': max(
                800 * np.sin(max((datetime.now().hour - 6) * np.pi / 12, 0)) + np.random.normal(0, 50), 0
            ),
            'uv_index': max(np.random.normal(6, 2), 0)
        }
        
        return measurements
    
    def validate_reading(self, measurements: Dict[str, float]) -> Dict[str, DataQuality]:
        """Validate weather measurements"""
        quality_flags = {}
        
        # Temperature validation
        temp = measurements.get('temperature_celsius', 0)
        if 10 <= temp <= 45:
            quality_flags['temperature_celsius'] = DataQuality.GOOD
        elif 5 <= temp <= 50:
            quality_flags['temperature_celsius'] = DataQuality.FAIR
        else:
            quality_flags['temperature_celsius'] = DataQuality.POOR
        
        # Humidity validation
        humidity = measurements.get('humidity_percent', 0)
        if 0 <= humidity <= 100:
            quality_flags['humidity_percent'] = DataQuality.GOOD
        else:
            quality_flags['humidity_percent'] = DataQuality.INVALID
        
        # Rainfall validation
        rainfall = measurements.get('rainfall_mm', 0)
        if rainfall >= 0 and rainfall <= 200:  # Max reasonable hourly rainfall
            quality_flags['rainfall_mm'] = DataQuality.GOOD
        else:
            quality_flags['rainfall_mm'] = DataQuality.POOR
        
        # Wind speed validation
        wind_speed = measurements.get('wind_speed_ms', 0)
        if 0 <= wind_speed <= 50:  # Reasonable wind speeds
            quality_flags['wind_speed_ms'] = DataQuality.GOOD
        else:
            quality_flags['wind_speed_ms'] = DataQuality.POOR
        
        return quality_flags
    
    def get_sensor_type(self) -> SensorType:
        return SensorType.WEATHER_STATION


class SoilSensor(BaseSensor):
    """Multi-parameter soil monitoring sensor"""
    
    def __init__(self, sensor_id: str, location: SensorLocation, depth_cm: float = 30):
        super().__init__(sensor_id, location)
        self.depth_cm = depth_cm
        self.measurement_interval = 3600  # 1 hour for soil data
    
    def read_sensors(self) -> Dict[str, float]:
        """Read soil sensor values"""
        measurements = {
            'soil_moisture_percent': min(max(np.random.normal(45, 15), 0), 100),
            'soil_temperature_celsius': np.random.normal(22, 3),
            'soil_ph': np.random.normal(6.5, 0.5),
            'electrical_conductivity_dsm': np.random.exponential(1.5),
            'nitrogen_ppm': np.random.exponential(50),
            'phosphorus_ppm': np.random.exponential(20),
            'potassium_ppm': np.random.exponential(100),
            'soil_oxygen_percent': min(max(np.random.normal(18, 3), 5), 21)
        }
        
        return measurements
    
    def validate_reading(self, measurements: Dict[str, float]) -> Dict[str, DataQuality]:
        """Validate soil measurements"""
        quality_flags = {}
        
        # Soil moisture validation
        moisture = measurements.get('soil_moisture_percent', 0)
        if 0 <= moisture <= 100:
            quality_flags['soil_moisture_percent'] = DataQuality.GOOD
        else:
            quality_flags['soil_moisture_percent'] = DataQuality.INVALID
        
        # pH validation
        ph = measurements.get('soil_ph', 7)
        if 3 <= ph <= 10:
            quality_flags['soil_ph'] = DataQuality.GOOD
        else:
            quality_flags['soil_ph'] = DataQuality.POOR
        
        # Temperature validation
        temp = measurements.get('soil_temperature_celsius', 20)
        if 5 <= temp <= 35:
            quality_flags['soil_temperature_celsius'] = DataQuality.GOOD
        else:
            quality_flags['soil_temperature_celsius'] = DataQuality.FAIR
        
        # Conductivity validation
        ec = measurements.get('electrical_conductivity_dsm', 0)
        if ec >= 0 and ec <= 10:
            quality_flags['electrical_conductivity_dsm'] = DataQuality.GOOD
        else:
            quality_flags['electrical_conductivity_dsm'] = DataQuality.POOR
        
        return quality_flags
    
    def get_sensor_type(self) -> SensorType:
        return SensorType.SOIL_SENSOR


class PlantMonitorSensor(BaseSensor):
    """Plant health and growth monitoring sensor"""
    
    def __init__(self, sensor_id: str, location: SensorLocation):
        super().__init__(sensor_id, location)
        self.measurement_interval = 86400  # Daily measurements
    
    def read_sensors(self) -> Dict[str, float]:
        """Read plant monitoring values"""
        # Simulate plant growth over time
        days_since_planting = max((datetime.now() - self.calibration_date).days, 0)
        
        measurements = {
            'plant_height_cm': min(days_since_planting * 2 + np.random.normal(0, 5), 250),
            'leaf_area_index': min(days_since_planting * 0.02 + np.random.normal(0, 0.2), 6),
            'ndvi': min(max(np.random.normal(0.7, 0.1), 0), 1),  # Normalized Difference Vegetation Index
            'canopy_temperature_celsius': np.random.normal(28, 3),
            'chlorophyll_content': np.random.normal(45, 8),  # SPAD units
            'stem_diameter_mm': min(days_since_planting * 0.1 + np.random.normal(0, 1), 30)
        }
        
        return measurements
    
    def validate_reading(self, measurements: Dict[str, float]) -> Dict[str, DataQuality]:
        """Validate plant measurements"""
        quality_flags = {}
        
        # Plant height validation
        height = measurements.get('plant_height_cm', 0)
        if height >= 0 and height <= 300:
            quality_flags['plant_height_cm'] = DataQuality.GOOD
        else:
            quality_flags['plant_height_cm'] = DataQuality.POOR
        
        # NDVI validation
        ndvi = measurements.get('ndvi', 0)
        if 0 <= ndvi <= 1:
            quality_flags['ndvi'] = DataQuality.GOOD
        else:
            quality_flags['ndvi'] = DataQuality.INVALID
        
        # LAI validation
        lai = measurements.get('leaf_area_index', 0)
        if 0 <= lai <= 8:
            quality_flags['leaf_area_index'] = DataQuality.GOOD
        else:
            quality_flags['leaf_area_index'] = DataQuality.POOR
        
        return quality_flags
    
    def get_sensor_type(self) -> SensorType:
        return SensorType.PLANT_MONITOR


class SensorNetwork:
    """Manages a network of IoT sensors"""
    
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.sensors: Dict[str, BaseSensor] = {}
        self.readings_history: List[SensorReading] = []
        self.alerts: List[Dict[str, Any]] = []
    
    def add_sensor(self, sensor: BaseSensor) -> None:
        """Add a sensor to the network"""
        self.sensors[sensor.sensor_id] = sensor
        print(f"Added sensor {sensor.sensor_id} to network {self.network_id}")
    
    def remove_sensor(self, sensor_id: str) -> None:
        """Remove a sensor from the network"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            print(f"Removed sensor {sensor_id} from network {self.network_id}")
    
    def collect_all_readings(self) -> List[SensorReading]:
        """Collect readings from all active sensors"""
        current_readings = []
        
        for sensor in self.sensors.values():
            if sensor.is_active:
                try:
                    reading = sensor.take_reading()
                    current_readings.append(reading)
                    self.readings_history.append(reading)
                    
                    # Check for alerts
                    self._check_alerts(reading)
                    
                except Exception as e:
                    print(f"Failed to read sensor {sensor.sensor_id}: {e}")
        
        return current_readings
    
    def get_sensor_status_summary(self) -> Dict[str, Any]:
        """Get summary of sensor network status"""
        active_sensors = sum(1 for s in self.sensors.values() if s.is_active)
        total_sensors = len(self.sensors)
        
        sensor_types = {}
        for sensor in self.sensors.values():
            sensor_type = sensor.get_sensor_type().value
            sensor_types[sensor_type] = sensor_types.get(sensor_type, 0) + 1
        
        return {
            'network_id': self.network_id,
            'total_sensors': total_sensors,
            'active_sensors': active_sensors,
            'offline_sensors': total_sensors - active_sensors,
            'sensor_types': sensor_types,
            'total_readings': len(self.readings_history),
            'recent_alerts': len([a for a in self.alerts if 
                                (datetime.now() - datetime.fromisoformat(a['timestamp'])).hours < 24])
        }
    
    def get_recent_readings(self, hours: int = 24) -> List[SensorReading]:
        """Get readings from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [r for r in self.readings_history if r.timestamp > cutoff_time]
    
    def get_readings_by_sensor_type(self, sensor_type: SensorType, hours: int = 24) -> List[SensorReading]:
        """Get readings filtered by sensor type"""
        recent_readings = self.get_recent_readings(hours)
        return [r for r in recent_readings if r.sensor_type == sensor_type]
    
    def _check_alerts(self, reading: SensorReading) -> None:
        """Check for alert conditions in sensor readings"""
        alerts = []
        
        # Temperature alerts
        if reading.sensor_type == SensorType.WEATHER_STATION:
            temp = reading.measurements.get('temperature_celsius')
            if temp is not None:
                if temp > 40:
                    alerts.append("High temperature alert")
                elif temp < 5:
                    alerts.append("Low temperature alert")
        
        # Soil moisture alerts
        if reading.sensor_type == SensorType.SOIL_SENSOR:
            moisture = reading.measurements.get('soil_moisture_percent')
            if moisture is not None:
                if moisture < 10:
                    alerts.append("Low soil moisture alert")
                elif moisture > 90:
                    alerts.append("High soil moisture alert")
        
        # Battery alerts
        if reading.device_status.battery_level < 20:
            alerts.append("Low battery alert")
        
        # Data quality alerts
        poor_quality_sensors = [
            key for key, quality in reading.quality_flags.items()
            if quality in [DataQuality.POOR, DataQuality.INVALID]
        ]
        if poor_quality_sensors:
            alerts.append(f"Poor data quality for: {', '.join(poor_quality_sensors)}")
        
        # Store alerts
        for alert_msg in alerts:
            alert = {
                'sensor_id': reading.sensor_id,
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'sensor_alert',
                'message': alert_msg,
                'severity': 'warning'
            }
            self.alerts.append(alert)
    
    def export_readings_to_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """Export recent readings to pandas DataFrame"""
        recent_readings = self.get_recent_readings(hours)
        
        if not recent_readings:
            return pd.DataFrame()
        
        # Flatten readings into tabular format
        rows = []
        for reading in recent_readings:
            base_row = {
                'sensor_id': reading.sensor_id,
                'timestamp': reading.timestamp,
                'sensor_type': reading.sensor_type.value,
                'latitude': reading.location.latitude,
                'longitude': reading.location.longitude,
                'altitude': reading.location.altitude,
                'field_id': reading.location.field_id,
                'battery_level': reading.device_status.battery_level,
                'signal_strength': reading.device_status.signal_strength
            }
            
            # Add measurements
            base_row.update(reading.measurements)
            rows.append(base_row)
        
        return pd.DataFrame(rows)
    
    def save_readings_to_json(self, filepath: str, hours: int = 24) -> None:
        """Save recent readings to JSON file"""
        recent_readings = self.get_recent_readings(hours)
        
        # Convert readings to JSON-serializable format
        readings_data = []
        for reading in recent_readings:
            reading_dict = {
                'sensor_id': reading.sensor_id,
                'timestamp': reading.timestamp.isoformat(),
                'location': asdict(reading.location),
                'sensor_type': reading.sensor_type.value,
                'measurements': reading.measurements,
                'quality_flags': {k: v.value for k, v in reading.quality_flags.items()},
                'device_status': asdict(reading.device_status)
            }
            # Handle datetime serialization in device_status
            if reading_dict['device_status']['last_calibration']:
                reading_dict['device_status']['last_calibration'] = \
                    reading_dict['device_status']['last_calibration'].isoformat()
            readings_data.append(reading_dict)
        
        with open(filepath, 'w') as f:
            json.dump({
                'network_id': self.network_id,
                'export_timestamp': datetime.now().isoformat(),
                'readings_count': len(readings_data),
                'readings': readings_data
            }, f, indent=2)


def create_sample_sensor_network() -> SensorNetwork:
    """Create a sample sensor network for testing"""
    network = SensorNetwork("UGANDA-CENTRAL-001")
    
    # Add weather station
    weather_location = SensorLocation(
        latitude=0.3476, longitude=32.5825, altitude=1190, field_id="FIELD-0001"
    )
    weather_sensor = WeatherStationSensor("WEATHER-001", weather_location)
    network.add_sensor(weather_sensor)
    
    # Add soil sensors at different depths
    for depth in [10, 30, 60]:
        soil_location = SensorLocation(
            latitude=0.3476, longitude=32.5825, altitude=1190, field_id="FIELD-0001"
        )
        soil_sensor = SoilSensor(f"SOIL-{depth:03d}", soil_location, depth)
        network.add_sensor(soil_sensor)
    
    # Add plant monitoring sensor
    plant_location = SensorLocation(
        latitude=0.3476, longitude=32.5825, altitude=1190, field_id="FIELD-0001"
    )
    plant_sensor = PlantMonitorSensor("PLANT-001", plant_location)
    network.add_sensor(plant_sensor)
    
    return network


if __name__ == "__main__":
    # Example usage
    print("Creating sample sensor network...")
    network = create_sample_sensor_network()
    
    print(f"\nNetwork Status:")
    status = network.get_sensor_status_summary()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print(f"\nCollecting readings from all sensors...")
    readings = network.collect_all_readings()
    
    print(f"Collected {len(readings)} readings:")
    for reading in readings:
        print(f"  {reading.sensor_id} ({reading.sensor_type.value}): "
              f"{len(reading.measurements)} measurements")
        
        # Show a few key measurements
        key_measurements = {}
        if reading.sensor_type == SensorType.WEATHER_STATION:
            key_measurements = {k: v for k, v in reading.measurements.items() 
                              if k in ['temperature_celsius', 'humidity_percent', 'rainfall_mm']}
        elif reading.sensor_type == SensorType.SOIL_SENSOR:
            key_measurements = {k: v for k, v in reading.measurements.items() 
                              if k in ['soil_moisture_percent', 'soil_ph', 'soil_temperature_celsius']}
        elif reading.sensor_type == SensorType.PLANT_MONITOR:
            key_measurements = {k: v for k, v in reading.measurements.items() 
                              if k in ['plant_height_cm', 'ndvi', 'leaf_area_index']}
        
        for key, value in key_measurements.items():
            print(f"    {key}: {value:.2f}")
    
    print(f"\nExporting readings to DataFrame...")
    df = network.export_readings_to_dataframe(hours=1)
    print(f"Exported {len(df)} readings with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Check for alerts
    if network.alerts:
        print(f"\nActive Alerts ({len(network.alerts)}):")
        for alert in network.alerts[-5:]:  # Show last 5 alerts
            print(f"  {alert['timestamp']}: {alert['message']}")
    else:
        print(f"\nNo active alerts")