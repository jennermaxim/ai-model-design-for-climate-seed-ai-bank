"""
Configuration Management

This module provides configuration management for the Climate-Adaptive Seed AI Bank system.
Handles loading, validation, and access to configuration settings from various sources.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "seed_ai_bank"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class APIConfig:
    """API server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 300  # 5 minutes
    rate_limit_per_minute: int = 60
    enable_swagger: bool = True


@dataclass
class MLModelConfig:
    """Machine learning model configuration"""
    models_directory: str = "./models"
    enable_training: bool = True
    training_schedule: str = "daily"  # daily, weekly, monthly
    model_versions_to_keep: int = 5
    prediction_batch_size: int = 100
    feature_cache_ttl: int = 3600  # 1 hour
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'climate_compatibility': 0.35,
        'yield_prediction': 0.40,
        'risk_assessment': 0.25
    })


@dataclass
class IoTConfig:
    """IoT integration configuration"""
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_use_ssl: bool = False
    data_collection_interval: int = 300  # 5 minutes
    sensor_timeout: int = 600  # 10 minutes
    gateway_heartbeat_interval: int = 60  # 1 minute
    max_offline_duration: int = 3600  # 1 hour


@dataclass
class ExternalDataConfig:
    """External data source configuration"""
    weather_api_key: str = ""
    weather_api_url: str = "https://api.openweathermap.org/data/2.5"
    soil_data_api_url: str = ""
    market_data_api_url: str = ""
    government_data_api_url: str = ""
    data_refresh_interval: int = 3600  # 1 hour
    api_timeout: int = 30  # 30 seconds
    max_retries: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    file_path: str = "./logs/seed_ai_bank.log"
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    console_handler: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_https: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""


@dataclass
class ApplicationConfig:
    """Main application configuration"""
    app_name: str = "Climate-Adaptive Seed AI Bank"
    app_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    timezone: str = "UTC"
    default_language: str = "en"
    supported_languages: list = field(default_factory=lambda: ["en", "sw", "lg"])
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ml_models: MLModelConfig = field(default_factory=MLModelConfig)
    iot: IoTConfig = field(default_factory=IoTConfig)
    external_data: ExternalDataConfig = field(default_factory=ExternalDataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


class ConfigManager:
    """
    Configuration manager that handles loading, validation, and access
    to application configuration from various sources.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[ApplicationConfig] = None
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables"""
        try:
            # Start with default configuration
            config_dict = self._get_default_config()
            
            # Load from configuration file if provided
            if self.config_path and self.config_path.exists():
                file_config = self._load_config_file(self.config_path)
                config_dict = self._merge_configs(config_dict, file_config)
            
            # Override with environment variables
            env_config = self._load_env_config()
            config_dict = self._merge_configs(config_dict, env_config)
            
            # Create configuration object
            self.config = self._dict_to_config(config_dict)
            
            # Validate configuration
            self._validate_config()
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use default configuration as fallback
            self.config = ApplicationConfig()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        default_config = ApplicationConfig()
        return self._config_to_dict(default_config)
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        except Exception as e:
            self.logger.warning(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Define environment variable mappings
        env_mappings = {
            # Database
            'DB_HOST': 'database.host',
            'DB_PORT': 'database.port',
            'DB_NAME': 'database.database',
            'DB_USER': 'database.username',
            'DB_PASSWORD': 'database.password',
            
            # API
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'API_DEBUG': 'api.debug',
            
            # Security
            'SECRET_KEY': 'security.secret_key',
            'JWT_ALGORITHM': 'security.jwt_algorithm',
            
            # IoT
            'MQTT_HOST': 'iot.mqtt_broker_host',
            'MQTT_PORT': 'iot.mqtt_broker_port',
            'MQTT_USER': 'iot.mqtt_username',
            'MQTT_PASS': 'iot.mqtt_password',
            
            # External APIs
            'WEATHER_API_KEY': 'external_data.weather_api_key',
            
            # General
            'ENVIRONMENT': 'environment',
            'LOG_LEVEL': 'logging.level'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                env_value = self._convert_env_value(env_value)
                self._set_nested_value(env_config, config_path, env_value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean values
        if value.lower() in ['true', 'yes', '1', 'on']:
            return True
        elif value.lower() in ['false', 'no', '0', 'off']:
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # String value (default)
        return value
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ApplicationConfig:
        """Convert configuration dictionary to ApplicationConfig object"""
        try:
            # Create sub-configurations
            database_config = DatabaseConfig(**config_dict.get('database', {}))
            api_config = APIConfig(**config_dict.get('api', {}))
            ml_config = MLModelConfig(**config_dict.get('ml_models', {}))
            iot_config = IoTConfig(**config_dict.get('iot', {}))
            external_config = ExternalDataConfig(**config_dict.get('external_data', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            security_config = SecurityConfig(**config_dict.get('security', {}))
            
            # Create main configuration
            main_config_dict = {k: v for k, v in config_dict.items() 
                              if k not in ['database', 'api', 'ml_models', 'iot', 
                                         'external_data', 'logging', 'security']}
            
            return ApplicationConfig(
                **main_config_dict,
                database=database_config,
                api=api_config,
                ml_models=ml_config,
                iot=iot_config,
                external_data=external_config,
                logging=logging_config,
                security=security_config
            )
        except Exception as e:
            self.logger.error(f"Failed to create config object: {e}")
            return ApplicationConfig()
    
    def _config_to_dict(self, config: ApplicationConfig) -> Dict[str, Any]:
        """Convert ApplicationConfig object to dictionary"""
        result = {}
        
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Nested configuration object
                result[field_name] = field_value.__dict__.copy()
            else:
                result[field_name] = field_value
        
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        if not self.config:
            raise ValueError("Configuration is not loaded")
        
        # Validate required fields
        if not self.config.security.secret_key or self.config.security.secret_key == "change-me-in-production":
            if self.config.environment == "production":
                raise ValueError("SECRET_KEY must be set in production environment")
            else:
                self.logger.warning("Using default SECRET_KEY - not suitable for production")
        
        # Validate port numbers
        if not (1 <= self.config.api.port <= 65535):
            raise ValueError(f"Invalid API port: {self.config.api.port}")
        
        if not (1 <= self.config.database.port <= 65535):
            raise ValueError(f"Invalid database port: {self.config.database.port}")
        
        # Validate directories
        models_dir = Path(self.config.ml_models.models_directory)
        if not models_dir.exists():
            self.logger.info(f"Creating models directory: {models_dir}")
            models_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate logging configuration
        log_dir = Path(self.config.logging.file_path).parent
        if not log_dir.exists():
            self.logger.info(f"Creating log directory: {log_dir}")
            log_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        if not self.config:
            return default
        
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def reload(self) -> None:
        """Reload configuration"""
        self._load_config()
    
    def save_config(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file"""
        if not self.config:
            raise ValueError("No configuration to save")
        
        output_path = Path(file_path) if file_path else self.config_path
        if not output_path:
            raise ValueError("No output path specified")
        
        config_dict = self._config_to_dict(self.config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            'app_name': self.config.app_name if self.config else 'Unknown',
            'app_version': self.config.app_version if self.config else 'Unknown',
            'environment': self.config.environment if self.config else 'Unknown',
            'python_version': os.sys.version,
            'config_loaded': self.config is not None,
            'config_path': str(self.config_path) if self.config_path else None,
            'timestamp': datetime.now().isoformat()
        }


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> ApplicationConfig:
    """Load configuration and return config object"""
    global _config_manager
    
    if _config_manager is None or config_path:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.config


def get_config() -> Optional[ApplicationConfig]:
    """Get current configuration"""
    return _config_manager.config if _config_manager else None


def reload_config() -> ApplicationConfig:
    """Reload configuration"""
    global _config_manager
    if _config_manager:
        _config_manager.reload()
        return _config_manager.config
    else:
        return load_config()


# Example usage and default configuration file generation
if __name__ == "__main__":
    # Create a sample configuration file
    config_manager = ConfigManager()
    config_manager.save_config("config/config.example.yaml")
    print("Example configuration saved to config/config.example.yaml")