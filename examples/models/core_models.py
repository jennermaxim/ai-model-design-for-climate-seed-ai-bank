"""
Climate-Adaptive Seed AI Bank - Core Models

This module contains the core AI models for seed recommendation,
yield prediction, and risk assessment.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class FarmData:
    """Data structure for farm characteristics"""
    farmer_id: str
    field_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    altitude: float
    field_size_hectares: float
    soil_properties: Dict[str, float]  # pH, organic_matter, nutrients, etc.
    climate_data: Dict[str, float]  # temperature, rainfall, humidity, etc.
    infrastructure: Dict[str, bool]  # irrigation, storage, etc.
    farmer_preferences: Dict[str, Union[str, float]]


@dataclass
class SeedVariety:
    """Data structure for seed variety characteristics"""
    seed_id: str
    name: str
    crop_type: str
    genetic_traits: Dict[str, str]  # drought_tolerance, disease_resistance, etc.
    agronomic_properties: Dict[str, float]  # maturity_days, yield_potential, etc.
    environmental_requirements: Dict[str, Tuple[float, float]]  # min/max ranges


@dataclass
class Recommendation:
    """Data structure for seed recommendations"""
    seed_id: str
    confidence_score: float
    expected_yield: float
    risk_score: float
    explanation: List[str]
    cost_benefit_analysis: Dict[str, float]


class BaseModel(ABC):
    """Abstract base class for all AI models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.feature_names = []
        self.model_version = "1.0.0"
    
    @abstractmethod
    def train(self, training_data: pd.DataFrame, target: pd.Series) -> None:
        """Train the model with provided data"""
        pass
    
    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        import joblib
        model_data = {
            'model': self,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'trained_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load trained model from file"""
        import joblib
        model_data = joblib.load(filepath)
        return model_data['model']


class ClimateCompatibilityModel(BaseModel):
    """Model for assessing seed-climate compatibility"""
    
    def __init__(self):
        super().__init__("climate_compatibility")
        self.climate_zones = {}
        self.compatibility_matrix = None
    
    def train(self, training_data: pd.DataFrame, target: pd.Series) -> None:
        """
        Train climate compatibility model
        
        Args:
            training_data: Features including climate and seed data
            target: Compatibility scores (0-1)
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Feature engineering for climate compatibility
        climate_features = self._extract_climate_features(training_data)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(climate_features)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(scaled_features, target)
        
        self.feature_names = climate_features.columns.tolist()
        self.is_trained = True
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Predict climate compatibility scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        climate_features = self._extract_climate_features(input_data)
        scaled_features = self.scaler.transform(climate_features)
        
        compatibility_scores = self.model.predict(scaled_features)
        return np.clip(compatibility_scores, 0, 1)  # Ensure scores are between 0-1
    
    def _extract_climate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer climate-related features"""
        climate_features = pd.DataFrame()
        
        # Temperature features
        climate_features['avg_temperature'] = data['temperature_avg']
        climate_features['temp_variability'] = data['temperature_std']
        climate_features['growing_degree_days'] = np.maximum(
            data['temperature_avg'] - 10, 0
        ).cumsum()
        
        # Precipitation features
        climate_features['annual_rainfall'] = data['rainfall_annual']
        climate_features['dry_season_rainfall'] = data['rainfall_dry_season']
        climate_features['wet_season_rainfall'] = data['rainfall_wet_season']
        climate_features['drought_index'] = self._calculate_drought_index(data)
        
        # Humidity and evapotranspiration
        climate_features['humidity_avg'] = data['humidity_avg']
        climate_features['evapotranspiration'] = data['evapotranspiration']
        
        # Seasonal patterns
        climate_features['seasonality_index'] = self._calculate_seasonality_index(data)
        
        return climate_features
    
    def _calculate_drought_index(self, data: pd.DataFrame) -> pd.Series:
        """Calculate standardized precipitation index (SPI)"""
        rainfall_mean = data['rainfall_annual'].mean()
        rainfall_std = data['rainfall_annual'].std()
        spi = (data['rainfall_annual'] - rainfall_mean) / rainfall_std
        return spi
    
    def _calculate_seasonality_index(self, data: pd.DataFrame) -> pd.Series:
        """Calculate climate seasonality index"""
        # Simple seasonality measure based on temperature and rainfall variation
        temp_cv = data['temperature_std'] / data['temperature_avg']
        rainfall_cv = data['rainfall_std'] / data['rainfall_annual']
        seasonality = (temp_cv + rainfall_cv) / 2
        return seasonality
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return {}
        
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class YieldPredictionModel(BaseModel):
    """Model for predicting crop yields"""
    
    def __init__(self):
        super().__init__("yield_prediction")
        self.crop_models = {}  # Separate models for different crops
    
    def train(self, training_data: pd.DataFrame, target: pd.Series) -> None:
        """
        Train yield prediction model
        
        Args:
            training_data: Features including farm, climate, and management data
            target: Historical yield data (tons per hectare)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Extract and engineer features
        yield_features = self._extract_yield_features(training_data)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(yield_features)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(scaled_features, target)
        
        self.feature_names = yield_features.columns.tolist()
        self.is_trained = True
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Predict crop yields"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        yield_features = self._extract_yield_features(input_data)
        scaled_features = self.scaler.transform(yield_features)
        
        yield_predictions = self.model.predict(scaled_features)
        return np.maximum(yield_predictions, 0)  # Ensure non-negative yields
    
    def _extract_yield_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer yield-related features"""
        yield_features = pd.DataFrame()
        
        # Soil features
        yield_features['soil_ph'] = data['soil_ph']
        yield_features['organic_matter'] = data['soil_organic_matter']
        yield_features['soil_fertility_index'] = (
            data['nitrogen_ppm'] * 0.4 + 
            data['phosphorus_ppm'] * 0.3 + 
            data['potassium_ppm'] * 0.3
        )
        
        # Climate features
        yield_features['growing_season_rainfall'] = data['growing_season_rainfall']
        yield_features['temperature_optimality'] = self._calculate_temperature_optimality(data)
        yield_features['water_stress_index'] = self._calculate_water_stress(data)
        
        # Management features
        yield_features['irrigation_score'] = data['irrigation_available'].astype(int)
        yield_features['fertilizer_score'] = data.get('fertilizer_applied', 0)
        
        # Variety features
        yield_features['variety_yield_potential'] = data['variety_max_yield']
        yield_features['variety_maturity'] = data['variety_maturity_days']
        
        # Geographic features
        yield_features['altitude'] = data['altitude']
        yield_features['slope'] = data['slope']
        
        return yield_features
    
    def _calculate_temperature_optimality(self, data: pd.DataFrame) -> pd.Series:
        """Calculate temperature optimality score for crop growth"""
        # Assuming optimal temperature range is stored in data
        optimal_min = data['crop_temp_min']
        optimal_max = data['crop_temp_max']
        actual_temp = data['temperature_avg']
        
        # Calculate distance from optimal range
        temp_optimality = np.where(
            (actual_temp >= optimal_min) & (actual_temp <= optimal_max),
            1.0,  # Optimal
            np.where(
                actual_temp < optimal_min,
                actual_temp / optimal_min,  # Too cold
                optimal_max / actual_temp   # Too hot
            )
        )
        
        return pd.Series(temp_optimality, index=data.index)
    
    def _calculate_water_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate water stress index"""
        water_requirement = data['crop_water_requirement']
        water_available = data['rainfall_growing_season'] + data.get('irrigation_amount', 0)
        
        water_stress = 1.0 - (water_available / water_requirement).clip(0, 1)
        return water_stress
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return {}
        
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))


class RiskAssessmentModel(BaseModel):
    """Model for assessing agricultural risks"""
    
    def __init__(self):
        super().__init__("risk_assessment")
        self.risk_categories = ['drought', 'flood', 'pest', 'disease', 'market']
    
    def train(self, training_data: pd.DataFrame, target: pd.Series) -> None:
        """
        Train risk assessment model
        
        Args:
            training_data: Features including climate, pest, disease, market data
            target: Historical risk outcomes (0-1 for each risk category)
        """
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Extract risk features
        risk_features = self._extract_risk_features(training_data)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(risk_features)
        
        # Train multi-output classifier for different risk types
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model = MultiOutputClassifier(base_classifier)
        self.model.fit(scaled_features, target)
        
        self.feature_names = risk_features.columns.tolist()
        self.is_trained = True
    
    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Predict risk probabilities for different categories"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        risk_features = self._extract_risk_features(input_data)
        scaled_features = self.scaler.transform(risk_features)
        
        risk_probabilities = self.model.predict_proba(scaled_features)
        
        # Extract positive class probabilities for each risk category
        risk_scores = np.array([
            prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
            for prob in risk_probabilities
        ]).T
        
        return risk_scores
    
    def _extract_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer risk-related features"""
        risk_features = pd.DataFrame()
        
        # Drought risk features
        risk_features['rainfall_deficit'] = np.maximum(
            data['crop_water_requirement'] - data['rainfall_annual'], 0
        )
        risk_features['drought_history'] = data.get('historical_droughts', 0)
        risk_features['consecutive_dry_days'] = data.get('max_dry_days', 0)
        
        # Flood risk features
        risk_features['excess_rainfall'] = np.maximum(
            data['rainfall_annual'] - data['optimal_rainfall'], 0
        )
        risk_features['drainage_score'] = data['field_drainage'].map({
            'poor': 0, 'moderate': 1, 'good': 2, 'excellent': 3
        })
        
        # Pest risk features
        risk_features['pest_pressure_history'] = data.get('historical_pest_damage', 0)
        risk_features['temperature_pest_risk'] = self._calculate_pest_temperature_risk(data)
        risk_features['humidity_pest_risk'] = np.where(
            data['humidity_avg'] > 70, 1, 0
        )
        
        # Disease risk features
        risk_features['disease_pressure_history'] = data.get('historical_disease_damage', 0)
        risk_features['humidity_disease_risk'] = np.where(
            data['humidity_avg'] > 80, 1, 0
        )
        risk_features['leaf_wetness_duration'] = data.get('leaf_wetness_hours', 0)
        
        # Market risk features (if available)
        risk_features['price_volatility'] = data.get('price_volatility_index', 0)
        risk_features['market_access_score'] = data.get('market_access_score', 1)
        
        return risk_features
    
    def _calculate_pest_temperature_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate temperature-based pest risk"""
        # Most pests thrive in moderate to warm temperatures
        temp_risk = np.where(
            (data['temperature_avg'] >= 20) & (data['temperature_avg'] <= 35),
            1.0,  # High risk temperature range
            np.where(
                data['temperature_avg'] < 20,
                data['temperature_avg'] / 20,  # Lower risk when cold
                1.0 - ((data['temperature_avg'] - 35) / 15)  # Lower risk when very hot
            )
        )
        
        return pd.Series(np.clip(temp_risk, 0, 1), index=data.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance averaged across all risk categories"""
        if not self.is_trained:
            return {}
        
        # Average feature importance across all estimators
        importance_scores = np.mean([
            estimator.feature_importances_ 
            for estimator in self.model.estimators_
        ], axis=0)
        
        return dict(zip(self.feature_names, importance_scores))


def create_sample_data() -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Create sample data for testing models"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic farm data
    data = pd.DataFrame({
        # Location data
        'latitude': np.random.uniform(-1.5, 4.5, n_samples),
        'longitude': np.random.uniform(29.5, 35.5, n_samples),
        'altitude': np.random.uniform(500, 2000, n_samples),
        'slope': np.random.exponential(5, n_samples),
        
        # Soil data
        'soil_ph': np.random.normal(6.5, 1.0, n_samples),
        'soil_organic_matter': np.random.exponential(3, n_samples),
        'nitrogen_ppm': np.random.exponential(50, n_samples),
        'phosphorus_ppm': np.random.exponential(20, n_samples),
        'potassium_ppm': np.random.exponential(100, n_samples),
        
        # Climate data
        'temperature_avg': np.random.normal(25, 3, n_samples),
        'temperature_std': np.random.exponential(2, n_samples),
        'rainfall_annual': np.random.gamma(2, 500, n_samples),
        'rainfall_dry_season': np.random.gamma(1, 100, n_samples),
        'rainfall_wet_season': np.random.gamma(2, 400, n_samples),
        'rainfall_std': np.random.exponential(200, n_samples),
        'humidity_avg': np.random.normal(70, 10, n_samples),
        'evapotranspiration': np.random.normal(1500, 200, n_samples),
        
        # Management data
        'irrigation_available': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'fertilizer_applied': np.random.exponential(2, n_samples),
        
        # Variety data
        'variety_max_yield': np.random.normal(4, 1, n_samples),
        'variety_maturity_days': np.random.normal(120, 20, n_samples),
        'crop_temp_min': np.random.normal(18, 2, n_samples),
        'crop_temp_max': np.random.normal(32, 3, n_samples),
        'crop_water_requirement': np.random.normal(800, 150, n_samples),
        
        # Additional features
        'growing_season_rainfall': np.random.gamma(2, 300, n_samples),
        'field_drainage': np.random.choice(['poor', 'moderate', 'good'], n_samples),
        'optimal_rainfall': np.random.normal(1000, 200, n_samples),
    })
    
    # Generate target variables
    targets = {
        'climate_compatibility': pd.Series(np.random.beta(2, 2, n_samples)),
        'yield': pd.Series(np.random.gamma(3, 1.5, n_samples)),
        'risks': pd.DataFrame({
            'drought': np.random.binomial(1, 0.2, n_samples),
            'flood': np.random.binomial(1, 0.1, n_samples),
            'pest': np.random.binomial(1, 0.3, n_samples),
            'disease': np.random.binomial(1, 0.25, n_samples),
            'market': np.random.binomial(1, 0.15, n_samples),
        })
    }
    
    return data, targets


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    data, targets = create_sample_data()
    
    print("Training Climate Compatibility Model...")
    climate_model = ClimateCompatibilityModel()
    climate_model.train(data, targets['climate_compatibility'])
    
    print("Training Yield Prediction Model...")
    yield_model = YieldPredictionModel()
    yield_model.train(data, targets['yield'])
    
    print("Training Risk Assessment Model...")
    risk_model = RiskAssessmentModel()
    risk_model.train(data, targets['risks'])
    
    # Make sample predictions
    sample_input = data.head(10)
    
    climate_pred = climate_model.predict(sample_input)
    yield_pred = yield_model.predict(sample_input)
    risk_pred = risk_model.predict(sample_input)
    
    print(f"\nSample Predictions:")
    print(f"Climate Compatibility: {climate_pred[:5]}")
    print(f"Yield Prediction: {yield_pred[:5]}")
    print(f"Risk Assessment: {risk_pred[:5]}")
    
    # Print feature importance
    print(f"\nClimate Model Feature Importance:")
    for feature, importance in climate_model.get_feature_importance().items():
        print(f"  {feature}: {importance:.3f}")