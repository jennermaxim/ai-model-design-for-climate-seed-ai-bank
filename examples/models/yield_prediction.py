"""
Yield Prediction Models

This module implements AI models for predicting crop yields based on
seed varieties, environmental conditions, and farming practices.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from .core_models import FarmData, SeedVariety, BaseModel


class YieldPredictionModel(BaseModel):
    """
    Advanced yield prediction model using ensemble methods and multiple
    environmental and genetic factors to predict crop yields.
    """
    
    def __init__(self):
        super().__init__()
        self.primary_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.secondary_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        self.baseline_model = LinearRegression()
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True)
        
        self.feature_columns = []
        self.model_weights = {
            'random_forest': 0.5,
            'gradient_boost': 0.3,
            'linear': 0.2
        }
        
        # Yield factors and their typical ranges
        self.yield_factors = {
            'base_yield': 2.5,  # tons/hectare
            'climate_multiplier': (0.5, 1.5),
            'soil_multiplier': (0.6, 1.3),
            'management_multiplier': (0.7, 1.4),
            'genetic_multiplier': (0.8, 1.6)
        }
    
    def _extract_yield_features(self, farm_data: FarmData, seed_variety: SeedVariety) -> np.ndarray:
        """Extract comprehensive features for yield prediction"""
        features = []
        
        # Climate features (major yield drivers)
        climate = farm_data.climate_data
        features.extend([
            climate.get('avg_temperature', 25),
            climate.get('max_temperature', 32),
            climate.get('min_temperature', 18),
            climate.get('temperature_variance', 5),
            climate.get('annual_rainfall', 800),
            climate.get('rainfall_distribution_score', 0.7),  # How evenly distributed
            climate.get('humidity', 60),
            climate.get('sunshine_hours', 2500),
            climate.get('wind_speed', 10),
            climate.get('frost_days', 0)
        ])
        
        # Soil features (crucial for nutrient availability)
        soil = farm_data.soil_properties
        features.extend([
            soil.get('ph', 6.5),
            soil.get('organic_matter', 2.5),
            soil.get('nitrogen', 20),
            soil.get('phosphorus', 15),
            soil.get('potassium', 150),
            soil.get('calcium', 200),
            soil.get('magnesium', 50),
            soil.get('sulfur', 10),
            soil.get('soil_depth', 50),  # cm
            soil.get('water_holding_capacity', 0.15),
            soil.get('drainage_score', 0.7)
        ])
        
        # Farm management features
        infrastructure = farm_data.infrastructure
        features.extend([
            infrastructure.get('irrigation', False) * 1.0,
            infrastructure.get('fertilizer_access', False) * 1.0,
            infrastructure.get('pest_control', False) * 1.0,
            infrastructure.get('storage_facilities', False) * 1.0,
            infrastructure.get('mechanization_level', 0.3),  # 0-1 scale
            farm_data.field_size_hectares,
            farm_data.altitude / 1000.0  # normalize altitude
        ])
        
        # Seed/genetic features
        agro = seed_variety.agronomic_properties
        features.extend([
            agro.get('yield_potential', 3.0),
            agro.get('maturity_days', 120),
            agro.get('disease_resistance_score', 0.5),
            agro.get('drought_tolerance_score', 0.5),
            agro.get('nutrient_efficiency_score', 0.5),
            agro.get('lodging_resistance_score', 0.5),
            agro.get('grain_quality_score', 0.7)
        ])
        
        # Geographic features
        features.extend([
            farm_data.location[0],  # latitude
            farm_data.location[1],  # longitude
            np.sin(2 * np.pi * farm_data.location[0] / 360),  # seasonal patterns
            np.cos(2 * np.pi * farm_data.location[0] / 360)
        ])
        
        return np.array(features)
    
    def train(self, training_data: pd.DataFrame, targets: pd.Series) -> None:
        """Train the yield prediction ensemble"""
        self.feature_columns = [col for col in training_data.columns if col not in ['target', 'yield']]
        
        # Prepare features
        X = training_data[self.feature_columns].fillna(0)
        y = targets.fillna(targets.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create polynomial features for linear model
        X_poly = self.poly_features.fit_transform(X_scaled)
        
        # Train ensemble models
        print("Training Random Forest yield model...")
        self.primary_model.fit(X_scaled, y)
        
        print("Training Gradient Boosting yield model...")
        self.secondary_model.fit(X_scaled, y)
        
        print("Training Linear yield model...")
        self.baseline_model.fit(X_poly, y)
        
        self.is_trained = True
        
        # Evaluate models
        self._evaluate_models(X_scaled, X_poly, y)
    
    def _evaluate_models(self, X_scaled: np.ndarray, X_poly: np.ndarray, y: pd.Series) -> None:
        """Evaluate trained models and print metrics"""
        # Random Forest
        rf_pred = self.primary_model.predict(X_scaled)
        rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))
        rf_r2 = r2_score(y, rf_pred)
        
        # Gradient Boosting
        gb_pred = self.secondary_model.predict(X_scaled)
        gb_rmse = np.sqrt(mean_squared_error(y, gb_pred))
        gb_r2 = r2_score(y, gb_pred)
        
        # Linear Model
        lr_pred = self.baseline_model.predict(X_poly)
        lr_rmse = np.sqrt(mean_squared_error(y, lr_pred))
        lr_r2 = r2_score(y, lr_pred)
        
        print(f"\nModel Performance:")
        print(f"Random Forest    - RMSE: {rf_rmse:.3f}, R²: {rf_r2:.3f}")
        print(f"Gradient Boost   - RMSE: {gb_rmse:.3f}, R²: {gb_r2:.3f}")
        print(f"Linear Model     - RMSE: {lr_rmse:.3f}, R²: {lr_r2:.3f}")
    
    def predict_yield(
        self, 
        farm_data: FarmData, 
        seed_varieties: List[SeedVariety],
        confidence_interval: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict yields for multiple seed varieties
        
        Args:
            farm_data: Farm characteristics
            seed_varieties: List of seed varieties to evaluate
            confidence_interval: Whether to include confidence intervals
            
        Returns:
            Dictionary mapping seed_id to yield predictions and metrics
        """
        results = {}
        
        for seed in seed_varieties:
            if self.is_trained:
                prediction = self._ml_predict_yield(farm_data, seed, confidence_interval)
            else:
                prediction = self._rule_based_yield(farm_data, seed)
            
            results[seed.seed_id] = prediction
        
        return results
    
    def _ml_predict_yield(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety,
        confidence_interval: bool
    ) -> Dict[str, float]:
        """Machine learning-based yield prediction"""
        # Extract features
        features = self._extract_yield_features(farm_data, seed_variety)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_poly = self.poly_features.transform(features_scaled)
        
        # Get predictions from all models
        rf_pred = self.primary_model.predict(features_scaled)[0]
        gb_pred = self.secondary_model.predict(features_scaled)[0]
        lr_pred = self.baseline_model.predict(features_poly)[0]
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.model_weights['random_forest'] * rf_pred +
            self.model_weights['gradient_boost'] * gb_pred +
            self.model_weights['linear'] * lr_pred
        )
        
        result = {
            'predicted_yield': max(0.0, ensemble_pred),
            'rf_prediction': max(0.0, rf_pred),
            'gb_prediction': max(0.0, gb_pred),
            'lr_prediction': max(0.0, lr_pred)
        }
        
        # Add confidence interval if requested
        if confidence_interval:
            predictions = [rf_pred, gb_pred, lr_pred]
            std_dev = np.std(predictions)
            result.update({
                'confidence_lower': max(0.0, ensemble_pred - 1.96 * std_dev),
                'confidence_upper': ensemble_pred + 1.96 * std_dev,
                'prediction_std': std_dev
            })
        
        return result
    
    def _rule_based_yield(self, farm_data: FarmData, seed_variety: SeedVariety) -> Dict[str, float]:
        """Fallback rule-based yield estimation"""
        base_yield = seed_variety.agronomic_properties.get('yield_potential', 3.0)
        
        # Climate adjustment
        climate = farm_data.climate_data
        temp_optimal = 25  # °C
        rain_optimal = 800  # mm
        
        temp_factor = 1.0 - abs(climate.get('avg_temperature', 25) - temp_optimal) / 20.0
        rain_factor = min(1.0, climate.get('annual_rainfall', 800) / rain_optimal)
        climate_adj = np.mean([temp_factor, rain_factor])
        
        # Soil adjustment
        soil = farm_data.soil_properties
        ph_optimal = 6.5
        ph_factor = 1.0 - abs(soil.get('ph', 6.5) - ph_optimal) / 2.0
        nutrient_factor = min(1.0, (
            soil.get('nitrogen', 15) / 20 +
            soil.get('phosphorus', 10) / 15 +
            soil.get('potassium', 100) / 150
        ) / 3.0)
        soil_adj = np.mean([ph_factor, nutrient_factor])
        
        # Management adjustment
        infrastructure = farm_data.infrastructure
        mgmt_factor = 0.7 + 0.3 * np.mean([
            infrastructure.get('irrigation', False),
            infrastructure.get('fertilizer_access', False),
            infrastructure.get('pest_control', False)
        ])
        
        # Final yield estimate
        estimated_yield = base_yield * climate_adj * soil_adj * mgmt_factor
        
        return {
            'predicted_yield': max(0.1, estimated_yield),
            'climate_adjustment': climate_adj,
            'soil_adjustment': soil_adj,
            'management_adjustment': mgmt_factor,
            'base_yield_potential': base_yield
        }
    
    def get_yield_limiting_factors(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety
    ) -> Dict[str, Dict[str, float]]:
        """Identify factors that may limit yield"""
        climate = farm_data.climate_data
        soil = farm_data.soil_properties
        infrastructure = farm_data.infrastructure
        
        limiting_factors = {
            'climate': {},
            'soil': {},
            'management': {},
            'overall_limitations': {}
        }
        
        # Climate limitations
        temp = climate.get('avg_temperature', 25)
        if temp < 18 or temp > 32:
            limiting_factors['climate']['temperature'] = abs(temp - 25) / 10.0
        
        rainfall = climate.get('annual_rainfall', 800)
        if rainfall < 400:
            limiting_factors['climate']['water_deficit'] = (400 - rainfall) / 400.0
        elif rainfall > 1500:
            limiting_factors['climate']['excess_water'] = (rainfall - 1500) / 1000.0
        
        # Soil limitations
        ph = soil.get('ph', 6.5)
        if ph < 5.5 or ph > 8.0:
            limiting_factors['soil']['ph_stress'] = abs(ph - 6.5) / 2.0
        
        if soil.get('nitrogen', 15) < 10:
            limiting_factors['soil']['nitrogen_deficiency'] = (10 - soil.get('nitrogen', 15)) / 10.0
        
        if soil.get('phosphorus', 10) < 5:
            limiting_factors['soil']['phosphorus_deficiency'] = (5 - soil.get('phosphorus', 10)) / 5.0
        
        # Management limitations
        if not infrastructure.get('irrigation', False) and rainfall < 600:
            limiting_factors['management']['irrigation_needed'] = (600 - rainfall) / 600.0
        
        if not infrastructure.get('fertilizer_access', False):
            limiting_factors['management']['fertilizer_access'] = 0.3
        
        # Calculate overall limitation score
        all_limitations = []
        for category in limiting_factors.values():
            if isinstance(category, dict):
                all_limitations.extend(category.values())
        
        if all_limitations:
            limiting_factors['overall_limitations']['total_score'] = np.mean(all_limitations)
            limiting_factors['overall_limitations']['max_limitation'] = max(all_limitations)
            limiting_factors['overall_limitations']['num_factors'] = len(all_limitations)
        
        return limiting_factors
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the primary model"""
        if not self.is_trained:
            return {}
        
        if hasattr(self.primary_model, 'feature_importances_'):
            importance = dict(zip(self.feature_columns, self.primary_model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}


class SeasonalYieldModel(BaseModel):
    """
    Model for predicting yields across different seasons and planting times.
    Considers seasonal weather patterns and crop calendar optimization.
    """
    
    def __init__(self):
        super().__init__()
        self.seasonal_factors = {
            'dry_season': {'temp_boost': 0.1, 'water_penalty': -0.3},
            'wet_season': {'temp_penalty': -0.1, 'water_boost': 0.2},
            'transition': {'stability_boost': 0.05}
        }
    
    def predict_seasonal_yield(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety,
        planting_month: int
    ) -> Dict[str, float]:
        """Predict yield for specific planting season"""
        base_yield = seed_variety.agronomic_properties.get('yield_potential', 3.0)
        
        # Determine season
        if planting_month in [12, 1, 2, 3]:
            season = 'dry_season'
        elif planting_month in [6, 7, 8, 9]:
            season = 'wet_season'
        else:
            season = 'transition'
        
        # Apply seasonal adjustments
        seasonal_adj = 1.0
        climate = farm_data.climate_data
        
        if season == 'dry_season':
            seasonal_adj += self.seasonal_factors['dry_season']['temp_boost']
            if not farm_data.infrastructure.get('irrigation', False):
                seasonal_adj += self.seasonal_factors['dry_season']['water_penalty']
        elif season == 'wet_season':
            seasonal_adj += self.seasonal_factors['wet_season']['water_boost']
            if climate.get('max_temperature', 30) > 30:
                seasonal_adj += self.seasonal_factors['wet_season']['temp_penalty']
        else:
            seasonal_adj += self.seasonal_factors['transition']['stability_boost']
        
        seasonal_yield = base_yield * max(0.3, seasonal_adj)
        
        return {
            'seasonal_yield': seasonal_yield,
            'season': season,
            'seasonal_adjustment': seasonal_adj,
            'base_yield': base_yield,
            'optimal_planting_month': planting_month
        }