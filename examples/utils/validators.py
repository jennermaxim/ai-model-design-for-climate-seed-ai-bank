"""
Data Validators

This module provides comprehensive validation functions for various data types
used in the Climate-Adaptive Seed AI Bank system.
"""

import re
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class SeverityLevel(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def add_error(self, message: str) -> None:
        """Add an error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message"""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str) -> None:
        """Add a suggestion message"""
        self.suggestions.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result"""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.suggestions.extend(other.suggestions)


def validate_coordinates(latitude: float, longitude: float) -> ValidationResult:
    """
    Validate geographic coordinates
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Validate latitude range
    if not (-90 <= latitude <= 90):
        result.add_error(f"Latitude {latitude} is out of range [-90, 90]")
    
    # Validate longitude range
    if not (-180 <= longitude <= 180):
        result.add_error(f"Longitude {longitude} is out of range [-180, 180]")
    
    # Check if coordinates are in Uganda (approximate bounds)
    uganda_bounds = {
        'lat_min': -1.5,
        'lat_max': 4.2,
        'lon_min': 29.5,
        'lon_max': 35.0
    }
    
    if not (uganda_bounds['lat_min'] <= latitude <= uganda_bounds['lat_max'] and
            uganda_bounds['lon_min'] <= longitude <= uganda_bounds['lon_max']):
        result.add_warning("Coordinates appear to be outside Uganda")
        result.add_suggestion("Verify coordinates are correct for Uganda-based farms")
    
    # Check for common invalid coordinates
    if latitude == 0.0 and longitude == 0.0:
        result.add_error("Coordinates (0, 0) are likely invalid")
    
    return result


def validate_date_range(
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date],
    max_range_days: Optional[int] = None,
    allow_future: bool = True
) -> ValidationResult:
    """
    Validate date range
    
    Args:
        start_date: Start date (string or date object)
        end_date: End date (string or date object)
        max_range_days: Maximum allowed range in days
        allow_future: Whether to allow future dates
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Convert strings to date objects
    try:
        if isinstance(start_date, str):
            start_date = datetime.datetime.fromisoformat(start_date).date()
        if isinstance(end_date, str):
            end_date = datetime.datetime.fromisoformat(end_date).date()
    except ValueError as e:
        result.add_error(f"Invalid date format: {e}")
        return result
    
    # Check if start_date is before end_date
    if start_date >= end_date:
        result.add_error("Start date must be before end date")
    
    # Check future dates if not allowed
    today = datetime.date.today()
    if not allow_future:
        if start_date > today:
            result.add_error("Start date cannot be in the future")
        if end_date > today:
            result.add_error("End date cannot be in the future")
    
    # Check maximum range
    if max_range_days and result.is_valid:
        range_days = (end_date - start_date).days
        if range_days > max_range_days:
            result.add_error(f"Date range ({range_days} days) exceeds maximum allowed ({max_range_days} days)")
    
    # Historical data warnings
    years_ago = (today - start_date).days / 365.25
    if years_ago > 10:
        result.add_warning(f"Start date is {years_ago:.1f} years ago - very old data")
    elif years_ago > 5:
        result.add_warning(f"Start date is {years_ago:.1f} years ago - old data")
    
    return result


def validate_farm_data(farm_data: Dict[str, Any]) -> ValidationResult:
    """
    Validate farm data structure and values
    
    Args:
        farm_data: Dictionary containing farm information
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Required fields
    required_fields = ['farmer_id', 'field_id', 'location', 'field_size_hectares']
    for field in required_fields:
        if field not in farm_data:
            result.add_error(f"Missing required field: {field}")
        elif farm_data[field] is None:
            result.add_error(f"Field {field} cannot be None")
    
    if not result.is_valid:
        return result
    
    # Validate farmer and field IDs
    farmer_id = farm_data.get('farmer_id', '')
    if not isinstance(farmer_id, str) or len(farmer_id) < 3:
        result.add_error("farmer_id must be a string with at least 3 characters")
    elif not re.match(r'^[A-Za-z0-9_-]+$', farmer_id):
        result.add_error("farmer_id contains invalid characters")
    
    field_id = farm_data.get('field_id', '')
    if not isinstance(field_id, str) or len(field_id) < 3:
        result.add_error("field_id must be a string with at least 3 characters")
    
    # Validate location
    location = farm_data.get('location')
    if location and len(location) == 2:
        coord_result = validate_coordinates(location[0], location[1])
        result.merge(coord_result)
    else:
        result.add_error("location must be a tuple/list of [latitude, longitude]")
    
    # Validate field size
    field_size = farm_data.get('field_size_hectares')
    if field_size is not None:
        if not isinstance(field_size, (int, float)) or field_size <= 0:
            result.add_error("field_size_hectares must be a positive number")
        elif field_size > 10000:
            result.add_warning(f"Field size {field_size} hectares is very large")
        elif field_size < 0.01:
            result.add_warning(f"Field size {field_size} hectares is very small")
    
    # Validate altitude if present
    altitude = farm_data.get('altitude')
    if altitude is not None:
        if not isinstance(altitude, (int, float)):
            result.add_error("altitude must be a number")
        elif altitude < -500 or altitude > 6000:
            result.add_error(f"altitude {altitude}m is out of reasonable range")
        elif altitude < 0:
            result.add_warning("Negative altitude (below sea level)")
    
    # Validate soil properties
    soil_properties = farm_data.get('soil_properties', {})
    if soil_properties:
        soil_result = validate_soil_properties(soil_properties)
        result.merge(soil_result)
    
    # Validate climate data
    climate_data = farm_data.get('climate_data', {})
    if climate_data:
        climate_result = validate_climate_data(climate_data)
        result.merge(climate_result)
    
    # Validate infrastructure data
    infrastructure = farm_data.get('infrastructure', {})
    if infrastructure:
        for key, value in infrastructure.items():
            if not isinstance(value, bool):
                result.add_warning(f"infrastructure.{key} should be a boolean value")
    
    return result


def validate_soil_properties(soil_properties: Dict[str, float]) -> ValidationResult:
    """
    Validate soil property values
    
    Args:
        soil_properties: Dictionary of soil measurements
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Define valid ranges for soil properties
    soil_ranges = {
        'ph': (3.0, 10.0),
        'organic_matter': (0.0, 20.0),  # percentage
        'nitrogen': (0.0, 200.0),  # ppm
        'phosphorus': (0.0, 200.0),  # ppm
        'potassium': (0.0, 2000.0),  # ppm
        'calcium': (0.0, 5000.0),  # ppm
        'magnesium': (0.0, 1000.0),  # ppm
        'sulfur': (0.0, 100.0),  # ppm
        'soil_depth': (0.0, 300.0),  # cm
        'water_holding_capacity': (0.0, 1.0),  # fraction
        'drainage_score': (0.0, 1.0)  # score
    }
    
    # Optimal ranges for key properties
    optimal_ranges = {
        'ph': (6.0, 7.5),
        'organic_matter': (2.0, 5.0),
        'nitrogen': (20.0, 50.0),
        'phosphorus': (10.0, 30.0),
        'potassium': (150.0, 300.0)
    }
    
    for property_name, value in soil_properties.items():
        if not isinstance(value, (int, float)):
            result.add_error(f"soil_properties.{property_name} must be a number")
            continue
        
        # Check valid range
        if property_name in soil_ranges:
            min_val, max_val = soil_ranges[property_name]
            if value < min_val or value > max_val:
                result.add_error(
                    f"soil_properties.{property_name} ({value}) is outside valid range [{min_val}, {max_val}]"
                )
        
        # Check optimal range
        if property_name in optimal_ranges and result.is_valid:
            opt_min, opt_max = optimal_ranges[property_name]
            if value < opt_min:
                result.add_warning(f"soil_properties.{property_name} ({value}) is below optimal range")
            elif value > opt_max:
                result.add_warning(f"soil_properties.{property_name} ({value}) is above optimal range")
    
    # Check for missing critical properties
    critical_properties = ['ph', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium']
    missing_critical = [prop for prop in critical_properties if prop not in soil_properties]
    if missing_critical:
        result.add_warning(f"Missing critical soil properties: {missing_critical}")
    
    return result


def validate_climate_data(climate_data: Dict[str, float]) -> ValidationResult:
    """
    Validate climate data values
    
    Args:
        climate_data: Dictionary of climate measurements
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Define valid ranges for climate properties (Uganda context)
    climate_ranges = {
        'avg_temperature': (10.0, 40.0),  # Celsius
        'min_temperature': (5.0, 30.0),   # Celsius
        'max_temperature': (15.0, 45.0),  # Celsius
        'annual_rainfall': (200.0, 3000.0),  # mm
        'humidity': (20.0, 100.0),  # percentage
        'wind_speed': (0.0, 100.0),  # km/h
        'sunshine_hours': (1000.0, 4000.0),  # hours/year
        'solar_radiation': (10.0, 30.0)  # MJ/m²/day
    }
    
    # Typical ranges for Uganda
    typical_ranges = {
        'avg_temperature': (18.0, 28.0),
        'annual_rainfall': (600.0, 1800.0),
        'humidity': (40.0, 80.0)
    }
    
    for property_name, value in climate_data.items():
        if not isinstance(value, (int, float)):
            result.add_error(f"climate_data.{property_name} must be a number")
            continue
        
        # Check valid range
        if property_name in climate_ranges:
            min_val, max_val = climate_ranges[property_name]
            if value < min_val or value > max_val:
                result.add_error(
                    f"climate_data.{property_name} ({value}) is outside valid range [{min_val}, {max_val}]"
                )
        
        # Check typical range for Uganda
        if property_name in typical_ranges and result.is_valid:
            typ_min, typ_max = typical_ranges[property_name]
            if value < typ_min or value > typ_max:
                result.add_warning(
                    f"climate_data.{property_name} ({value}) is outside typical range for Uganda [{typ_min}, {typ_max}]"
                )
    
    # Logical consistency checks
    if 'min_temperature' in climate_data and 'max_temperature' in climate_data:
        if climate_data['min_temperature'] >= climate_data['max_temperature']:
            result.add_error("min_temperature must be less than max_temperature")
    
    if 'min_temperature' in climate_data and 'avg_temperature' in climate_data:
        if climate_data['min_temperature'] > climate_data['avg_temperature']:
            result.add_error("min_temperature cannot be greater than avg_temperature")
    
    if 'max_temperature' in climate_data and 'avg_temperature' in climate_data:
        if climate_data['max_temperature'] < climate_data['avg_temperature']:
            result.add_error("max_temperature cannot be less than avg_temperature")
    
    return result


def validate_seed_variety(seed_data: Dict[str, Any]) -> ValidationResult:
    """
    Validate seed variety data structure and values
    
    Args:
        seed_data: Dictionary containing seed variety information
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Required fields
    required_fields = ['seed_id', 'name', 'crop_type']
    for field in required_fields:
        if field not in seed_data:
            result.add_error(f"Missing required field: {field}")
        elif not seed_data[field]:
            result.add_error(f"Field {field} cannot be empty")
    
    if not result.is_valid:
        return result
    
    # Validate seed_id
    seed_id = seed_data.get('seed_id', '')
    if not re.match(r'^[A-Za-z0-9_-]+$', seed_id):
        result.add_error("seed_id contains invalid characters (use only letters, numbers, underscore, hyphen)")
    
    # Validate crop_type
    valid_crops = [
        'maize', 'beans', 'rice', 'wheat', 'cassava', 'sweet_potato',
        'banana', 'coffee', 'cotton', 'groundnuts', 'sorghum', 'millet',
        'soybeans', 'sunflower', 'vegetables'
    ]
    crop_type = seed_data.get('crop_type', '').lower()
    if crop_type not in valid_crops:
        result.add_warning(f"crop_type '{crop_type}' is not in common crop list")
        result.add_suggestion(f"Common crops include: {', '.join(valid_crops[:5])}, etc.")
    
    # Validate agronomic properties
    agro_props = seed_data.get('agronomic_properties', {})
    if agro_props:
        agro_result = validate_agronomic_properties(agro_props)
        result.merge(agro_result)
    
    # Validate genetic traits
    genetic_traits = seed_data.get('genetic_traits', {})
    if genetic_traits:
        traits_result = validate_genetic_traits(genetic_traits)
        result.merge(traits_result)
    
    # Validate environmental requirements
    env_reqs = seed_data.get('environmental_requirements', {})
    if env_reqs:
        env_result = validate_environmental_requirements(env_reqs)
        result.merge(env_result)
    
    return result


def validate_agronomic_properties(agro_props: Dict[str, Any]) -> ValidationResult:
    """Validate agronomic properties of seed varieties"""
    result = ValidationResult(True, [], [], [])
    
    # Define valid ranges
    ranges = {
        'maturity_days': (30, 300),
        'yield_potential': (0.5, 15.0),  # tons/hectare
        'plant_height': (0.3, 5.0),  # meters
        'seed_weight': (10.0, 1000.0),  # grams per 1000 seeds
        'drought_tolerance_score': (0.0, 1.0),
        'heat_tolerance_score': (0.0, 1.0),
        'disease_resistance_score': (0.0, 1.0)
    }
    
    for prop, value in agro_props.items():
        if prop in ranges:
            if not isinstance(value, (int, float)):
                result.add_error(f"agronomic_properties.{prop} must be a number")
                continue
            
            min_val, max_val = ranges[prop]
            if value < min_val or value > max_val:
                result.add_error(f"agronomic_properties.{prop} ({value}) outside range [{min_val}, {max_val}]")
    
    return result


def validate_genetic_traits(genetic_traits: Dict[str, str]) -> ValidationResult:
    """Validate genetic traits descriptions"""
    result = ValidationResult(True, [], [], [])
    
    valid_tolerance_levels = ['low', 'medium', 'high', 'very_high']
    tolerance_traits = [
        'drought_tolerance', 'heat_tolerance', 'cold_tolerance',
        'disease_resistance', 'pest_resistance', 'salt_tolerance'
    ]
    
    for trait, value in genetic_traits.items():
        if trait in tolerance_traits:
            if value.lower() not in valid_tolerance_levels:
                result.add_error(
                    f"genetic_traits.{trait} must be one of: {', '.join(valid_tolerance_levels)}"
                )
    
    return result


def validate_environmental_requirements(env_reqs: Dict[str, Any]) -> ValidationResult:
    """Validate environmental requirements"""
    result = ValidationResult(True, [], [], [])
    
    # Check that ranges are tuples with min < max
    for req, value in env_reqs.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            min_val, max_val = value
            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                result.add_error(f"environmental_requirements.{req} values must be numbers")
            elif min_val >= max_val:
                result.add_error(f"environmental_requirements.{req} minimum must be less than maximum")
        elif req.endswith(('_min', '_max')):
            if not isinstance(value, (int, float)):
                result.add_error(f"environmental_requirements.{req} must be a number")
    
    return result


def validate_api_request(request_data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate API request against a schema
    
    Args:
        request_data: Request data to validate
        schema: Validation schema
        
    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(True, [], [], [])
    
    # Check required fields
    required = schema.get('required', [])
    for field in required:
        if field not in request_data:
            result.add_error(f"Missing required field: {field}")
    
    # Check field types and constraints
    properties = schema.get('properties', {})
    for field, value in request_data.items():
        if field in properties:
            field_schema = properties[field]
            field_result = validate_field_against_schema(field, value, field_schema)
            result.merge(field_result)
    
    return result


def validate_field_against_schema(field_name: str, value: Any, field_schema: Dict[str, Any]) -> ValidationResult:
    """Validate a single field against its schema"""
    result = ValidationResult(True, [], [], [])
    
    # Check type
    expected_type = field_schema.get('type')
    if expected_type:
        if expected_type == 'string' and not isinstance(value, str):
            result.add_error(f"{field_name} must be a string")
        elif expected_type == 'number' and not isinstance(value, (int, float)):
            result.add_error(f"{field_name} must be a number")
        elif expected_type == 'integer' and not isinstance(value, int):
            result.add_error(f"{field_name} must be an integer")
        elif expected_type == 'boolean' and not isinstance(value, bool):
            result.add_error(f"{field_name} must be a boolean")
        elif expected_type == 'array' and not isinstance(value, list):
            result.add_error(f"{field_name} must be an array")
        elif expected_type == 'object' and not isinstance(value, dict):
            result.add_error(f"{field_name} must be an object")
    
    # Check constraints
    if isinstance(value, str):
        min_length = field_schema.get('minLength')
        if min_length and len(value) < min_length:
            result.add_error(f"{field_name} must be at least {min_length} characters long")
        
        max_length = field_schema.get('maxLength')
        if max_length and len(value) > max_length:
            result.add_error(f"{field_name} cannot exceed {max_length} characters")
        
        pattern = field_schema.get('pattern')
        if pattern and not re.match(pattern, value):
            result.add_error(f"{field_name} does not match required pattern")
    
    elif isinstance(value, (int, float)):
        minimum = field_schema.get('minimum')
        if minimum is not None and value < minimum:
            result.add_error(f"{field_name} must be at least {minimum}")
        
        maximum = field_schema.get('maximum')
        if maximum is not None and value > maximum:
            result.add_error(f"{field_name} cannot exceed {maximum}")
    
    elif isinstance(value, list):
        min_items = field_schema.get('minItems')
        if min_items and len(value) < min_items:
            result.add_error(f"{field_name} must have at least {min_items} items")
        
        max_items = field_schema.get('maxItems')
        if max_items and len(value) > max_items:
            result.add_error(f"{field_name} cannot have more than {max_items} items")
    
    return result


# Utility functions for common validations
def is_valid_email(email: str) -> bool:
    """Check if email address is valid"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_phone_number(phone: str) -> bool:
    """Check if phone number is valid (Uganda format)"""
    # Uganda phone numbers: +256XXXXXXXXX or 0XXXXXXXXX
    pattern = r'^(\+256|0)[0-9]{9}$'
    return re.match(pattern, phone) is not None


def sanitize_string(input_string: str, max_length: int = 255) -> str:
    """Sanitize string input"""
    if not isinstance(input_string, str):
        return str(input_string)
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>&"\'`]', '', input_string)
    
    # Limit length
    return sanitized[:max_length]