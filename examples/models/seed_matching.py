"""
Seed-Climate Matching Models

This module implements AI models for matching seed varieties to climate conditions,
soil properties, and environmental factors for optimal crop production.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from .core_models import FarmData, SeedVariety, BaseModel


class ClimateCompatibilityModel(BaseModel):
    """
    Model for assessing climate compatibility between seeds and farm conditions.
    Uses machine learning to predict how well different seed varieties will
    perform under specific climate conditions.
    """
    
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.climate_thresholds = {
            'temperature_optimal': (20, 30),  # Celsius
            'rainfall_optimal': (600, 1200),  # mm/year
            'humidity_optimal': (40, 70),     # %
            'drought_tolerance_scores': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9
            }
        }
    
    def _extract_features(self, farm_data: FarmData, seed_variety: SeedVariety) -> np.ndarray:
        """Extract features for climate compatibility prediction"""
        features = []
        
        # Climate features
        climate = farm_data.climate_data
        features.extend([
            climate.get('avg_temperature', 25),
            climate.get('min_temperature', 15),
            climate.get('max_temperature', 35),
            climate.get('annual_rainfall', 800),
            climate.get('humidity', 60),
            climate.get('sunshine_hours', 2500)
        ])
        
        # Soil features
        soil = farm_data.soil_properties
        features.extend([
            soil.get('ph', 6.5),
            soil.get('organic_matter', 2.5),
            soil.get('nitrogen', 20),
            soil.get('phosphorus', 15),
            soil.get('potassium', 150)
        ])
        
        # Location features
        features.extend([
            farm_data.location[0],  # latitude
            farm_data.location[1],  # longitude
            farm_data.altitude
        ])
        
        # Seed characteristics
        agro = seed_variety.agronomic_properties
        features.extend([
            agro.get('maturity_days', 120),
            agro.get('yield_potential', 3.0),
            agro.get('drought_tolerance_score', 0.5),
            agro.get('heat_tolerance_score', 0.5),
            agro.get('disease_resistance_score', 0.5)
        ])
        
        return np.array(features)
    
    def _calculate_climate_score(self, farm_data: FarmData, seed_variety: SeedVariety) -> Dict[str, float]:
        """Calculate detailed climate compatibility scores"""
        climate = farm_data.climate_data
        requirements = seed_variety.environmental_requirements
        
        scores = {}
        
        # Temperature compatibility
        temp_avg = climate.get('avg_temperature', 25)
        temp_range = requirements.get('temperature', (15, 35))
        if temp_range[0] <= temp_avg <= temp_range[1]:
            temp_score = 1.0
        else:
            temp_deviation = min(abs(temp_avg - temp_range[0]), abs(temp_avg - temp_range[1]))
            temp_score = max(0.0, 1.0 - temp_deviation / 10.0)
        scores['temperature'] = temp_score
        
        # Rainfall compatibility
        rainfall = climate.get('annual_rainfall', 800)
        rainfall_range = requirements.get('rainfall', (400, 1500))
        if rainfall_range[0] <= rainfall <= rainfall_range[1]:
            rain_score = 1.0
        else:
            rain_deviation = min(abs(rainfall - rainfall_range[0]), abs(rainfall - rainfall_range[1]))
            rain_score = max(0.0, 1.0 - rain_deviation / 500.0)
        scores['rainfall'] = rain_score
        
        # Humidity compatibility
        humidity = climate.get('humidity', 60)
        humidity_range = requirements.get('humidity', (30, 80))
        if humidity_range[0] <= humidity <= humidity_range[1]:
            humidity_score = 1.0
        else:
            humidity_deviation = min(abs(humidity - humidity_range[0]), abs(humidity - humidity_range[1]))
            humidity_score = max(0.0, 1.0 - humidity_deviation / 20.0)
        scores['humidity'] = humidity_score
        
        # Overall climate score
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def train(self, training_data: pd.DataFrame, targets: pd.Series) -> None:
        """Train the climate compatibility model"""
        self.feature_columns = [col for col in training_data.columns if col != 'target']
        
        # Prepare features
        X = training_data[self.feature_columns].fillna(0)
        y = targets
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels if they're categorical
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y_encoded, y_pred)
        print(f"Climate compatibility model training accuracy: {accuracy:.3f}")
    
    def predict_compatibility(
        self, 
        farm_data: FarmData, 
        seed_varieties: List[SeedVariety]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict climate compatibility for multiple seed varieties
        
        Args:
            farm_data: Farm characteristics
            seed_varieties: List of seed varieties to evaluate
            
        Returns:
            Dictionary mapping seed_id to compatibility scores
        """
        if not self.is_trained:
            # Use rule-based approach if not trained
            return self._rule_based_compatibility(farm_data, seed_varieties)
        
        results = {}
        
        for seed in seed_varieties:
            # Extract features
            features = self._extract_features(farm_data, seed)
            
            if hasattr(self, 'scaler'):
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                ml_score = self.model.predict_proba(features_scaled)[0].max()
            else:
                ml_score = 0.7  # Default score
            
            # Calculate detailed climate scores
            climate_scores = self._calculate_climate_score(farm_data, seed)
            
            # Combine ML prediction with rule-based scores
            combined_score = 0.6 * ml_score + 0.4 * climate_scores['overall']
            
            results[seed.seed_id] = {
                'overall_compatibility': combined_score,
                'ml_prediction': ml_score,
                **climate_scores
            }
        
        return results
    
    def _rule_based_compatibility(
        self, 
        farm_data: FarmData, 
        seed_varieties: List[SeedVariety]
    ) -> Dict[str, Dict[str, float]]:
        """Fallback rule-based compatibility assessment"""
        results = {}
        
        for seed in seed_varieties:
            scores = self._calculate_climate_score(farm_data, seed)
            results[seed.seed_id] = scores
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class SoilCompatibilityModel(BaseModel):
    """
    Model for assessing soil compatibility between seeds and farm soil conditions.
    Evaluates pH, nutrient levels, and soil structure compatibility.
    """
    
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.soil_requirements = {
            'ph_optimal': (6.0, 7.5),
            'organic_matter_min': 2.0,
            'nitrogen_min': 15,
            'phosphorus_min': 10,
            'potassium_min': 100
        }
    
    def assess_soil_compatibility(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety
    ) -> Dict[str, float]:
        """Assess soil compatibility for a specific seed variety"""
        soil = farm_data.soil_properties
        requirements = seed_variety.environmental_requirements
        
        scores = {}
        
        # pH compatibility
        ph = soil.get('ph', 6.5)
        ph_range = requirements.get('ph', (5.5, 8.0))
        if ph_range[0] <= ph <= ph_range[1]:
            ph_score = 1.0
        else:
            ph_deviation = min(abs(ph - ph_range[0]), abs(ph - ph_range[1]))
            ph_score = max(0.0, 1.0 - ph_deviation / 2.0)
        scores['ph'] = ph_score
        
        # Nutrient compatibility
        nutrients = ['nitrogen', 'phosphorus', 'potassium']
        nutrient_scores = []
        
        for nutrient in nutrients:
            level = soil.get(nutrient, 0)
            min_req = requirements.get(f'{nutrient}_min', 0)
            if level >= min_req:
                nutrient_score = min(1.0, level / (min_req + 20))
            else:
                nutrient_score = level / min_req if min_req > 0 else 0.5
            scores[nutrient] = nutrient_score
            nutrient_scores.append(nutrient_score)
        
        # Organic matter
        om = soil.get('organic_matter', 2.0)
        om_min = requirements.get('organic_matter_min', 1.5)
        om_score = min(1.0, om / max(om_min, 1.0))
        scores['organic_matter'] = om_score
        
        # Overall soil score
        all_scores = [ph_score] + nutrient_scores + [om_score]
        scores['overall'] = np.mean(all_scores)
        
        return scores


class AdaptabilityModel(BaseModel):
    """
    Model for assessing seed adaptability to changing climate conditions.
    Considers climate variability, extreme weather tolerance, and future projections.
    """
    
    def __init__(self):
        super().__init__()
        self.climate_scenarios = {
            'current': 1.0,
            'moderate_change': 0.8,
            'severe_change': 0.6
        }
    
    def assess_adaptability(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety,
        climate_scenario: str = 'current'
    ) -> Dict[str, float]:
        """Assess seed adaptability under different climate scenarios"""
        traits = seed_variety.genetic_traits
        climate = farm_data.climate_data
        
        # Get adaptability scores from genetic traits
        drought_tolerance = self._parse_tolerance(traits.get('drought_tolerance', 'medium'))
        heat_tolerance = self._parse_tolerance(traits.get('heat_tolerance', 'medium'))
        disease_resistance = self._parse_tolerance(traits.get('disease_resistance', 'medium'))
        
        # Climate stress factors
        temp_stress = self._calculate_temperature_stress(climate)
        water_stress = self._calculate_water_stress(climate)
        
        # Base adaptability score
        base_score = np.mean([drought_tolerance, heat_tolerance, disease_resistance])
        
        # Adjust for climate scenario
        scenario_factor = self.climate_scenarios.get(climate_scenario, 1.0)
        
        # Adjust for specific stresses
        stress_adjustment = 1.0 - 0.3 * (temp_stress + water_stress) / 2.0
        
        final_score = base_score * scenario_factor * stress_adjustment
        
        return {
            'overall_adaptability': final_score,
            'drought_tolerance': drought_tolerance,
            'heat_tolerance': heat_tolerance,
            'disease_resistance': disease_resistance,
            'temperature_stress': temp_stress,
            'water_stress': water_stress,
            'scenario_factor': scenario_factor
        }
    
    def _parse_tolerance(self, tolerance_str: str) -> float:
        """Parse tolerance string to numeric score"""
        tolerance_map = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9,
            'very_high': 1.0
        }
        return tolerance_map.get(tolerance_str.lower(), 0.5)
    
    def _calculate_temperature_stress(self, climate: Dict[str, float]) -> float:
        """Calculate temperature stress factor"""
        max_temp = climate.get('max_temperature', 30)
        if max_temp > 35:
            return min(1.0, (max_temp - 35) / 10.0)
        return 0.0
    
    def _calculate_water_stress(self, climate: Dict[str, float]) -> float:
        """Calculate water stress factor"""
        rainfall = climate.get('annual_rainfall', 800)
        if rainfall < 600:
            return min(1.0, (600 - rainfall) / 400.0)
        return 0.0