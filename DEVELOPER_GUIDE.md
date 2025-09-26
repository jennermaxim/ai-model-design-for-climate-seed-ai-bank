# Developer Quick Start Guide

## **Getting Up and Running**

### **Prerequisites**
- Python 3.8+ (tested on 3.13)
- 4GB+ RAM (8GB+ recommended for ML models)
- 2GB+ free disk space
- Internet connection for package downloads

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/jennermaxim/ai-model-design-for-climate-seed-ai-bank.git
cd ai-model-design-for-climate-seed-ai-bank

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 4. Install dependencies (choose one approach)

# Option A: Install all at once (may require large temp space)
pip install -r requirements.txt

# Option B: Install in stages (recommended if space is limited)
pip install -r requirements-light.txt  # Basic packages
pip install tensorflow                  # ML packages separately

# Option C: Custom temp directory (if you get space errors)
export TMPDIR=/path/to/large/temp/dir
pip install --cache-dir /path/to/large/cache/dir -r requirements.txt
```

### **Quick Test**
```bash
# Run the complete demo
python examples/seed_recommendation_demo.py

# Test individual components
python examples/models/core_models.py
python -c "from examples.models import *; print('Models imported successfully')"
```

## **Key Files to Know**

### **Main Demo**
- `examples/seed_recommendation_demo.py` - **START HERE**: Complete system demonstration

### **AI Models**
- `examples/models/core_models.py` - Base classes and data structures
- `examples/models/seed_matching.py` - Climate compatibility AI
- `examples/models/yield_prediction.py` - Harvest forecasting
- `examples/models/risk_assessment.py` - Threat analysis
- `examples/models/ensemble.py` - Combined recommendation engine

### **IoT System**
- `examples/iot/sensors.py` - Sensor implementations
- `examples/iot/gateways.py` - Communication gateways
- `examples/iot/protocols.py` - Communication protocols

### **Data Processing**
- `examples/data/processors/climate_processor.py` - Weather data processing
- `examples/utils/config.py` - Configuration management
- `examples/utils/validators.py` - Data validation

## **Development Workflow**

### **1. Understanding the System**
```bash
# Start with the demo to see everything working
python examples/seed_recommendation_demo.py

# Study the output to understand:
# - How IoT sensors collect data
# - How AI models make predictions
# - How recommendations are generated
```

### **2. Working with Individual Components**

**AI Models:**
```python
from examples.models.seed_matching import ClimateCompatibilityModel
from examples.models.core_models import create_sample_data

# Load sample data
data, targets = create_sample_data()

# Train a model
model = ClimateCompatibilityModel()
model.train(data, targets['climate_compatibility'])

# Make predictions
predictions = model.predict(data.head(5))
```

**IoT Sensors:**
```python
from examples.iot.sensors import create_sample_sensor_network

# Create sensor network
network = create_sample_sensor_network()

# Collect readings
readings = network.collect_all_readings()
for reading in readings:
    print(f"{reading.sensor_id}: {reading.value} {reading.unit}")
```

**Data Processing:**
```python
from examples.data.processors.climate_processor import ClimateDataProcessor

# Process climate data
processor = ClimateDataProcessor()
processed_data = processor.process_daily_data({
    'temperature': [20, 25, 30],
    'humidity': [60, 70, 80],
    'rainfall': [0, 5, 10]
})
```

### **3. Common Development Tasks**

**Adding a New AI Model:**
1. Create new file in `examples/models/`
2. Inherit from `BaseModel` in `core_models.py`
3. Implement `train()` and `predict()` methods
4. Add to `ensemble.py` if needed
5. Update `__init__.py` imports

**Adding New Sensor Type:**
1. Add sensor type to `SensorType` enum in `sensors.py`
2. Create new sensor class inheriting from `BaseSensor`
3. Implement `take_reading()` method
4. Update sensor network creation functions

**Adding Data Processing:**
1. Create processor in `examples/data/processors/`
2. Inherit from appropriate base class
3. Implement data transformation methods
4. Add validation schemas in `examples/data/schemas/`

## **Common Issues & Solutions**

### **Installation Issues**

**Problem**: `No space left on device` during pip install
```bash
# Solution: Use custom temp directory
export TMPDIR=/path/to/large/temp/directory
mkdir -p $TMPDIR
pip install --cache-dir /path/to/cache -r requirements.txt
```

**Problem**: TensorFlow installation fails
```bash
# Solution: Install separately with specific version
pip install tensorflow==2.20.0
# Or use CPU-only version
pip install tensorflow-cpu
```

**Problem**: Package conflicts
```bash
# Solution: Use fresh virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-light.txt
```

## **Testing**

### **Unit Tests**
```bash
# Run all tests (when implemented)
pytest

# Test specific component
python examples/models/core_models.py
python examples/iot/sensors.py
```

### **Manual Testing**
```bash
# Test AI models individually
python -c "
from examples.models.seed_matching import ClimateCompatibilityModel
from examples.models.core_models import create_sample_data
data, targets = create_sample_data()
model = ClimateCompatibilityModel()
model.train(data, targets['climate_compatibility'])
print('Model training successful')
"

# Test IoT system
python -c "
from examples.iot.sensors import create_sample_sensor_network
network = create_sample_sensor_network()
readings = network.collect_all_readings()
print(f'Collected {len(readings)} sensor readings')
"
```

## **Performance Optimization**

### **For Development**
- Use subset of training data for faster iterations
- Cache processed data to avoid recomputation
- Use CPU-only TensorFlow for development if no GPU

### **For Production**
- Implement data streaming for large datasets
- Use model checkpointing for training
- Optimize database queries
- Implement caching layers

## **Learning Resources**

### **Agriculture & Climate**
- Understanding crop varieties and growing conditions
- Climate data interpretation
- Soil science basics
- Agricultural economics

### **Technical Skills**
- Machine Learning with scikit-learn and TensorFlow
- IoT protocols (MQTT, HTTP)
- Time-series data analysis
- API development with FastAPI
- Database design (PostgreSQL, MongoDB)