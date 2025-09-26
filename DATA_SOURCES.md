# Data Sources and Locations in the Climate-Adaptive Seed AI Bank

Based on analysis of the project, here's a complete breakdown of what data is being used and where you can find it:

## Data Categories & Sources

### 1. Synthetic Training Data (Generated)
**Location**: `examples/models/core_models.py` → `create_sample_data()` function

**What it contains:**
- 1,000 synthetic farm records for training AI models
- 27 features covering location, soil, climate, and management data
- 3 target variables for climate compatibility, yield, and risks

**Features Generated:**
```python
# Location Data (Uganda coordinates)
'latitude': -1.5 to 4.5 degrees
'longitude': 29.5 to 35.5 degrees  
'altitude': 500-2000 meters
'slope': exponential distribution

# Soil Properties
'soil_ph': normal distribution around 6.5
'soil_organic_matter': exponential distribution
'nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm': nutrient levels

# Climate Variables
'temperature_avg': normal around 25°C
'rainfall_annual': gamma distribution (realistic precipitation)
'rainfall_dry_season', 'rainfall_wet_season': seasonal patterns
'humidity_avg': normal around 70%
'evapotranspiration': normal around 1500mm

# Management & Variety Data
'irrigation_available': 70% rain-fed, 30% irrigated
'fertilizer_applied', 'variety_max_yield', etc.
```

### 2. Seed Variety Database (Hard-coded)
**Location**: `examples/models/ensemble.py` → `_create_sample_seed_database()` function

**Contains 5 seed varieties:**
```python
1. 'LONGE-10H' - Maize hybrid, drought-tolerant
2. 'UH-5051' - High-yield maize hybrid  
3. 'KAWANDA-1' - Open-pollinated maize
4. 'K132' - Improved bean variety
5. 'WITA-9' - Improved rice variety
```

**Seed Data Fields:**
- Crop type, variety type, maturity days
- Max yield potential (tons/hectare)
- Drought tolerance, disease resistance
- Optimal temperature range
- Water requirements, seed costs

### 3. Real-time IoT Sensor Data (Simulated)
**Location**: `examples/iot/sensors.py` → Sensor classes

**Sensor Types:**
- **Weather Stations**: Temperature, humidity, rainfall, wind, pressure, solar radiation, UV index
- **Soil Sensors**: Moisture, temperature, pH, nutrients (N-P-K), electrical conductivity
- **Plant Monitors**: Height, leaf area index, NDVI, canopy temperature, chlorophyll

**Generated Data Format:**
```python
# Weather readings every 15 minutes
temperature: 15-35°C range
humidity: 30-95% range
rainfall: 0-50mm/hour

# Soil measurements at 3 depths (10cm, 30cm, 60cm)
moisture: 20-80% range
pH: 4.5-8.5 range
nutrients: realistic PPM ranges

# Plant health indicators
NDVI: 0.1-0.9 vegetation index
growth_rate: cm/day measurements
```

### 4. Demo Monitoring Data (Generated during runtime)
**Location**: `monitoring_data.csv` (created when demo runs)

**Contains:**
- 35+ sensor readings from 24-hour simulation
- Timestamps, sensor IDs, GPS coordinates
- All sensor measurements in CSV format
- Battery levels, signal strength, data quality scores

### 5. Farm Scenario Data (Hard-coded examples)
**Location**: `examples/seed_recommendation_demo.py` and `examples/models/ensemble.py`

**3 Farm Scenarios:**
```python
1. Small Scale Maize Farmer
   - Location: (1.37, 32.29) - Central Uganda
   - 2.5 hectares, rain-fed
   - pH 6.2, moderate conditions
   - $200 budget

2. Commercial Rice Producer  
   - Location: (1.50, 33.20) - High rainfall area
   - 10 hectares, irrigated
   - pH 5.8, high rainfall (1800mm/year)
   - $500 budget

3. Drought-Prone Region Farmer
   - Location: (2.00, 34.00) - Northern Uganda
   - 1.5 hectares, no irrigation
   - pH 7.2, low rainfall (800mm/year)
   - $80 budget
```

## File Structure & Data Locations

```
ai-model-design/
├── DATA GENERATION
│   ├── examples/models/core_models.py     # create_sample_data() - 1000 training samples
│   ├── examples/models/ensemble.py        # seed database + farm scenarios
│   └── examples/iot/sensors.py           # IoT sensor data simulation
│
├── RUNTIME DATA
│   ├── monitoring_data.csv              # Generated during demo runs
│   └── tmp/                             # Temporary ML model files
│
├── DATA PROCESSING
│   ├── examples/data/processors/        # Climate data processing classes
│   ├── examples/data/schemas/           # Data validation schemas
│   └── examples/data/connectors/        # Database connection utilities
│
└── CONFIGURATION
    └── config/config.example.yaml      # System configuration template
```

## How to Access the Data

### 1. Training Data (Python)
```python
from examples.models.core_models import create_sample_data

# Generate 1000 synthetic farm records
data, targets = create_sample_data()
print(f"Training data shape: {data.shape}")
print(f"Available features: {list(data.columns)}")
```

### 2. Seed Database (Python)
```python
from examples.models.ensemble import SeedRecommendationEnsemble

ensemble = SeedRecommendationEnsemble()
seed_db = ensemble._create_sample_seed_database()
print(seed_db[['name', 'crop_type', 'max_yield_tons_per_hectare']])
```

### 3. IoT Sensor Data (Python)
```python
from examples.iot.sensors import create_sample_sensor_network

network = create_sample_sensor_network()
readings = network.collect_all_readings()
for reading in readings:
    print(f"{reading.sensor_id}: {reading.value} {reading.unit}")
```

### 4. Monitoring Data (CSV)
```bash
# Generated after running demo
head -10 monitoring_data.csv
```

## Data Characteristics

### Realism Level:
- **Synthetic but Realistic**: All data uses appropriate statistical distributions
- **Uganda-Specific**: Coordinates, climate ranges, and crop varieties match Uganda
- **Agriculture-Focused**: Based on real agricultural research and practices

### Data Quality:
- **Consistent**: All datasets use the same coordinate system and units
- **Complete**: No missing values in generated datasets
- **Validated**: Data passes through quality checks and validation

### Scalability:
- **Parameterized**: Easy to change sample sizes and distributions
- **Extensible**: New sensors, crops, and features can be added
- **Modular**: Each data source is independent and replaceable

## Data Generation Details

### Statistical Distributions Used:
```python
# Location data uses uniform distributions within Uganda bounds
np.random.uniform(-1.5, 4.5, n_samples)  # Latitude
np.random.uniform(29.5, 35.5, n_samples)  # Longitude

# Climate data uses appropriate distributions
np.random.normal(25, 3, n_samples)        # Temperature (°C)
np.random.gamma(2, 500, n_samples)        # Annual rainfall (mm)
np.random.exponential(5, n_samples)       # Slope (degrees)

# Soil properties use realistic ranges
np.random.normal(6.5, 1.0, n_samples)     # pH
np.random.exponential(50, n_samples)      # Nitrogen (PPM)
```

### Seed Variety Data Schema:
```python
{
    'seed_id': str,           # Unique identifier
    'name': str,              # Commercial name
    'crop_type': str,         # maize, beans, rice, etc.
    'variety_type': str,      # hybrid, open_pollinated, improved
    'maturity_days': int,     # Days to harvest
    'max_yield_tons_per_hectare': float,  # Maximum yield potential
    'drought_tolerance': str, # low, moderate, high
    'disease_resistance': str,# low, moderate, high
    'optimal_temp_min': int,  # Minimum temperature (°C)
    'optimal_temp_max': int,  # Maximum temperature (°C)
    'water_requirement_mm': int, # Water needs (mm/season)
    'seed_cost_per_kg': float   # Cost per kilogram
}
```

### IoT Sensor Data Schema:
```python
{
    'sensor_id': str,         # Unique sensor identifier
    'sensor_type': str,       # weather, soil, plant
    'timestamp': datetime,    # Reading timestamp
    'value': float,           # Primary measurement value
    'unit': str,              # Measurement unit
    'quality_score': float,   # Data quality (0.0-1.0)
    'location': tuple,        # (latitude, longitude)
    'metadata': dict          # Additional sensor information
}
```

## Next Steps for Real Data

### Production Data Sources (Future):
1. **Weather APIs**: Uganda Meteorological Authority, OpenWeatherMap
2. **Satellite Data**: NASA, ESA climate monitoring satellites
3. **Government Databases**: Uganda Bureau of Statistics agricultural data
4. **Research Institutions**: NARO (National Agricultural Research Organisation)
5. **Farmer Surveys**: Real farm condition data collection
6. **IoT Deployment**: Physical sensor networks on actual farms

### Data Integration Strategy:
1. **API Connections**: Connect to external weather and climate APIs
2. **Database Integration**: Set up PostgreSQL/MongoDB for real data storage
3. **Real-time Streaming**: Implement MQTT/WebSocket for live sensor data
4. **Data Validation**: Enhance validation schemas for real-world data quality
5. **Historical Data**: Import existing agricultural databases and research data

### How to Replace Synthetic Data:
1. **Keep the same data structure** (column names and types)
2. **Replace generation functions** with real data loading
3. **Update validation schemas** for real data quality checks
4. **Maintain the same API interfaces** for seamless integration
5. **Add data cleaning pipelines** for real-world data inconsistencies

## Data Quality and Validation

### Current Validation:
- Data type checking for all numeric and categorical fields
- Range validation for sensor readings (e.g., temperature, pH)
- Geographic coordinate validation for Uganda boundaries
- Temporal validation for sensor timestamps

### Quality Metrics:
- **Completeness**: 100% for synthetic data (no missing values)
- **Consistency**: All data follows the same schemas and formats
- **Accuracy**: Synthetic data uses realistic ranges and distributions
- **Timeliness**: Sensor data includes proper timestamps and intervals

The current synthetic data provides a solid foundation for development and testing, with realistic patterns that will make the transition to real data straightforward when ready for production deployment.