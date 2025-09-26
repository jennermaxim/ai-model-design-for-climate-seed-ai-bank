"""
Seed Recommendation Ensemble Model

This module implements the main ensemble model that combines individual
models to provide comprehensive seed recommendations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from .core_models import (
    ClimateCompatibilityModel, 
    YieldPredictionModel, 
    RiskAssessmentModel,
    FarmData,
    SeedVariety,
    Recommendation
)


@dataclass
class ModelWeights:
    """Weights for combining different model outputs"""
    climate_compatibility: float = 0.35
    yield_prediction: float = 0.40
    risk_assessment: float = 0.25
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = self.climate_compatibility + self.yield_prediction + self.risk_assessment
        self.climate_compatibility /= total
        self.yield_prediction /= total
        self.risk_assessment /= total


class SeedRecommendationEnsemble:
    """
    Ensemble model combining climate compatibility, yield prediction,
    and risk assessment for comprehensive seed recommendations.
    """
    
    def __init__(self, seed_database: Optional[pd.DataFrame] = None):
        """
        Initialize the ensemble model
        
        Args:
            seed_database: DataFrame containing seed variety information
        """
        self.climate_model = ClimateCompatibilityModel()
        self.yield_model = YieldPredictionModel()
        self.risk_model = RiskAssessmentModel()
        
        self.model_weights = ModelWeights()
        self.seed_database = seed_database or self._create_sample_seed_database()
        self.is_trained = False
        
        # Economic parameters for cost-benefit analysis
        self.economic_params = {
            'seed_cost_per_kg': 5.0,  # USD
            'yield_price_per_ton': 200.0,  # USD
            'production_cost_per_hectare': 300.0,  # USD
            'risk_penalty_factor': 0.1  # Penalty for high-risk recommendations
        }
    
    def train(self, training_data: pd.DataFrame, targets: Dict[str, pd.Series]) -> None:
        """
        Train all component models
        
        Args:
            training_data: Training features
            targets: Dictionary with 'climate_compatibility', 'yield', and 'risks' targets
        """
        print("Training climate compatibility model...")
        self.climate_model.train(training_data, targets['climate_compatibility'])
        
        print("Training yield prediction model...")
        self.yield_model.train(training_data, targets['yield'])
        
        print("Training risk assessment model...")
        self.risk_model.train(training_data, targets['risks'])
        
        self.is_trained = True
        print("Ensemble model training completed!")
    
    def recommend_seeds(
        self, 
        farm_data: FarmData,
        top_n: int = 5,
        risk_tolerance: str = 'moderate'
    ) -> List[Recommendation]:
        """
        Generate seed recommendations for a specific farm
        
        Args:
            farm_data: Farm characteristics and constraints
            top_n: Number of top recommendations to return
            risk_tolerance: 'low', 'moderate', or 'high'
        
        Returns:
            List of seed recommendations ranked by suitability
        """
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making recommendations")
        
        # Convert farm data to DataFrame format expected by models
        farm_features = self._prepare_farm_features(farm_data)
        
        # Get predictions from all models for each seed variety
        recommendations = []
        
        for _, seed_row in self.seed_database.iterrows():
            # Combine farm and seed data
            combined_features = self._combine_farm_seed_features(farm_features, seed_row)
            
            # Get predictions from component models
            climate_score = self.climate_model.predict(combined_features)[0]
            expected_yield = self.yield_model.predict(combined_features)[0]
            risk_scores = self.risk_model.predict(combined_features)[0]
            
            # Calculate overall risk score
            overall_risk = np.mean(risk_scores)
            
            # Apply risk tolerance filtering
            if not self._meets_risk_tolerance(overall_risk, risk_tolerance):
                continue
            
            # Calculate composite recommendation score
            composite_score = self._calculate_composite_score(
                climate_score, expected_yield, overall_risk
            )
            
            # Perform cost-benefit analysis
            cost_benefit = self._calculate_cost_benefit_analysis(
                expected_yield, seed_row, farm_data
            )
            
            # Create recommendation object
            recommendation = Recommendation(
                seed_id=seed_row['seed_id'],
                confidence_score=composite_score,
                expected_yield=expected_yield,
                risk_score=overall_risk,
                explanation=self._generate_explanation(
                    climate_score, expected_yield, risk_scores, seed_row
                ),
                cost_benefit_analysis=cost_benefit
            )
            
            recommendations.append(recommendation)
        
        # Sort recommendations by composite score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return recommendations[:top_n]
    
    def _prepare_farm_features(self, farm_data: FarmData) -> pd.DataFrame:
        """Convert FarmData to DataFrame format for models"""
        features = pd.DataFrame([{
            # Location features
            'latitude': farm_data.location[0],
            'longitude': farm_data.location[1],
            'altitude': farm_data.altitude,
            'slope': farm_data.soil_properties.get('slope', 0),
            
            # Soil features
            'soil_ph': farm_data.soil_properties.get('ph', 6.5),
            'soil_organic_matter': farm_data.soil_properties.get('organic_matter', 3.0),
            'nitrogen_ppm': farm_data.soil_properties.get('nitrogen', 50),
            'phosphorus_ppm': farm_data.soil_properties.get('phosphorus', 20),
            'potassium_ppm': farm_data.soil_properties.get('potassium', 100),
            
            # Climate features
            'temperature_avg': farm_data.climate_data.get('temperature_avg', 25),
            'temperature_std': farm_data.climate_data.get('temperature_std', 2),
            'rainfall_annual': farm_data.climate_data.get('rainfall_annual', 1200),
            'rainfall_dry_season': farm_data.climate_data.get('rainfall_dry_season', 100),
            'rainfall_wet_season': farm_data.climate_data.get('rainfall_wet_season', 800),
            'rainfall_std': farm_data.climate_data.get('rainfall_std', 200),
            'humidity_avg': farm_data.climate_data.get('humidity_avg', 70),
            'evapotranspiration': farm_data.climate_data.get('evapotranspiration', 1500),
            
            # Management features
            'irrigation_available': int(farm_data.infrastructure.get('irrigation', False)),
            'fertilizer_applied': farm_data.farmer_preferences.get('fertilizer_budget', 100) / 50,  # Convert to application rate
            
            # Field features
            'field_drainage': farm_data.soil_properties.get('drainage', 'moderate'),
            'optimal_rainfall': 1000,  # Default optimal rainfall
        }])
        
        return features
    
    def _combine_farm_seed_features(self, farm_features: pd.DataFrame, seed_row: pd.Series) -> pd.DataFrame:
        """Combine farm features with seed variety characteristics"""
        combined = farm_features.copy()
        
        # Add seed variety features
        combined['variety_max_yield'] = seed_row['max_yield_tons_per_hectare']
        combined['variety_maturity_days'] = seed_row['maturity_days']
        combined['crop_temp_min'] = seed_row['optimal_temp_min']
        combined['crop_temp_max'] = seed_row['optimal_temp_max']
        combined['crop_water_requirement'] = seed_row['water_requirement_mm']
        combined['growing_season_rainfall'] = combined['rainfall_wet_season']
        
        return combined
    
    def _meets_risk_tolerance(self, risk_score: float, risk_tolerance: str) -> bool:
        """Check if recommendation meets farmer's risk tolerance"""
        risk_thresholds = {
            'low': 0.3,      # Very conservative
            'moderate': 0.5,  # Balanced approach
            'high': 0.8      # Willing to take risks for higher returns
        }
        
        return risk_score <= risk_thresholds.get(risk_tolerance, 0.5)
    
    def _calculate_composite_score(
        self, 
        climate_score: float, 
        expected_yield: float, 
        risk_score: float
    ) -> float:
        """Calculate weighted composite recommendation score"""
        # Normalize yield to 0-1 scale (assuming max yield of 10 tons/hectare)
        normalized_yield = min(expected_yield / 10.0, 1.0)
        
        # Risk score should be inverted (lower risk = higher score)
        risk_benefit_score = 1.0 - risk_score
        
        # Calculate weighted combination
        composite_score = (
            self.model_weights.climate_compatibility * climate_score +
            self.model_weights.yield_prediction * normalized_yield +
            self.model_weights.risk_assessment * risk_benefit_score
        )
        
        return composite_score
    
    def _calculate_cost_benefit_analysis(
        self,
        expected_yield: float,
        seed_row: pd.Series,
        farm_data: FarmData
    ) -> Dict[str, float]:
        """Calculate economic cost-benefit analysis"""
        
        # Calculate costs
        seed_cost = seed_row.get('seed_cost_per_kg', 10.0) * farm_data.field_size_hectares
        production_cost = self.economic_params['production_cost_per_hectare'] * farm_data.field_size_hectares
        total_cost = seed_cost + production_cost
        
        # Calculate expected revenue
        expected_revenue = (
            expected_yield * 
            farm_data.field_size_hectares * 
            self.economic_params['yield_price_per_ton']
        )
        
        # Calculate profit and ROI
        expected_profit = expected_revenue - total_cost
        roi = (expected_profit / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'seed_cost': seed_cost,
            'production_cost': production_cost,
            'total_cost': total_cost,
            'expected_revenue': expected_revenue,
            'expected_profit': expected_profit,
            'roi_percent': roi
        }
    
    def _generate_explanation(
        self,
        climate_score: float,
        expected_yield: float,
        risk_scores: np.ndarray,
        seed_row: pd.Series
    ) -> List[str]:
        """Generate human-readable explanation for recommendation"""
        explanations = []
        
        # Climate suitability explanation
        if climate_score >= 0.8:
            explanations.append("Excellent climate match for this variety")
        elif climate_score >= 0.6:
            explanations.append("Good climate suitability with minor adaptations needed")
        elif climate_score >= 0.4:
            explanations.append("Moderate climate fit - consider with caution")
        else:
            explanations.append("Poor climate match - high risk of poor performance")
        
        # Yield potential explanation
        if expected_yield >= seed_row.get('max_yield_tons_per_hectare', 0) * 0.8:
            explanations.append("High yield potential under current conditions")
        elif expected_yield >= seed_row.get('max_yield_tons_per_hectare', 0) * 0.6:
            explanations.append("Moderate yield potential with proper management")
        else:
            explanations.append("Lower yield potential - consider alternatives")
        
        # Risk assessment explanation
        risk_categories = ['drought', 'flood', 'pest', 'disease', 'market']
        high_risks = [
            risk_categories[i] for i, risk in enumerate(risk_scores) 
            if risk > 0.6
        ]
        
        if high_risks:
            explanations.append(f"Higher risk areas: {', '.join(high_risks)}")
        else:
            explanations.append("Low to moderate risk across all categories")
        
        # Variety-specific features
        if seed_row.get('drought_tolerance') == 'high':
            explanations.append("Excellent drought tolerance")
        
        if seed_row.get('disease_resistance') == 'high':
            explanations.append("Strong disease resistance")
        
        return explanations
    
    def _create_sample_seed_database(self) -> pd.DataFrame:
        """Create a sample seed database for testing"""
        seeds = pd.DataFrame([
            {
                'seed_id': 'LONGE-10H',
                'name': 'Longe 10H',
                'crop_type': 'maize',
                'variety_type': 'hybrid',
                'maturity_days': 120,
                'max_yield_tons_per_hectare': 8.5,
                'drought_tolerance': 'high',
                'disease_resistance': 'moderate',
                'optimal_temp_min': 18,
                'optimal_temp_max': 30,
                'water_requirement_mm': 600,
                'seed_cost_per_kg': 12.0
            },
            {
                'seed_id': 'UH-5051',
                'name': 'UH 5051',
                'crop_type': 'maize',
                'variety_type': 'hybrid',
                'maturity_days': 110,
                'max_yield_tons_per_hectare': 9.0,
                'drought_tolerance': 'moderate',
                'disease_resistance': 'high',
                'optimal_temp_min': 20,
                'optimal_temp_max': 32,
                'water_requirement_mm': 700,
                'seed_cost_per_kg': 15.0
            },
            {
                'seed_id': 'KAWANDA-1',
                'name': 'Kawanda Composite A',
                'crop_type': 'maize',
                'variety_type': 'open_pollinated',
                'maturity_days': 135,
                'max_yield_tons_per_hectare': 6.0,
                'drought_tolerance': 'moderate',
                'disease_resistance': 'moderate',
                'optimal_temp_min': 16,
                'optimal_temp_max': 28,
                'water_requirement_mm': 550,
                'seed_cost_per_kg': 5.0
            },
            {
                'seed_id': 'K132',
                'name': 'K132 Bean',
                'crop_type': 'beans',
                'variety_type': 'improved',
                'maturity_days': 90,
                'max_yield_tons_per_hectare': 2.5,
                'drought_tolerance': 'high',
                'disease_resistance': 'high',
                'optimal_temp_min': 18,
                'optimal_temp_max': 27,
                'water_requirement_mm': 400,
                'seed_cost_per_kg': 8.0
            },
            {
                'seed_id': 'WITA-9',
                'name': 'WITA-9',
                'crop_type': 'rice',
                'variety_type': 'improved',
                'maturity_days': 125,
                'max_yield_tons_per_hectare': 7.0,
                'drought_tolerance': 'low',
                'disease_resistance': 'high',
                'optimal_temp_min': 20,
                'optimal_temp_max': 35,
                'water_requirement_mm': 1200,
                'seed_cost_per_kg': 10.0
            }
        ])
        
        return seeds
    
    def update_model_weights(self, weights: ModelWeights) -> None:
        """Update the weights for model combination"""
        weights.normalize()
        self.model_weights = weights
    
    def get_model_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all component models"""
        return {
            'climate_model': self.climate_model.get_feature_importance(),
            'yield_model': self.yield_model.get_feature_importance(),
            'risk_model': self.risk_model.get_feature_importance()
        }
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the entire ensemble model"""
        import joblib
        
        ensemble_data = {
            'climate_model': self.climate_model,
            'yield_model': self.yield_model,
            'risk_model': self.risk_model,
            'model_weights': self.model_weights,
            'seed_database': self.seed_database,
            'economic_params': self.economic_params,
            'ensemble_version': '1.0.0',
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, filepath)
    
    @classmethod
    def load_ensemble(cls, filepath: str):
        """Load a saved ensemble model"""
        import joblib
        
        ensemble_data = joblib.load(filepath)
        
        # Create new instance
        ensemble = cls(ensemble_data['seed_database'])
        ensemble.climate_model = ensemble_data['climate_model']
        ensemble.yield_model = ensemble_data['yield_model']
        ensemble.risk_model = ensemble_data['risk_model']
        ensemble.model_weights = ensemble_data['model_weights']
        ensemble.economic_params = ensemble_data['economic_params']
        ensemble.is_trained = True
        
        return ensemble


def create_sample_farm_data() -> FarmData:
    """Create sample farm data for testing"""
    return FarmData(
        farmer_id="UG-12345678",
        field_id="FIELD-0000000001",
        location=(1.3733, 32.2903),  # Near Kampala
        altitude=1200,
        field_size_hectares=2.5,
        soil_properties={
            'ph': 6.2,
            'organic_matter': 3.5,
            'nitrogen': 45,
            'phosphorus': 18,
            'potassium': 95,
            'drainage': 'moderate',
            'slope': 3.2
        },
        climate_data={
            'temperature_avg': 24.5,
            'temperature_std': 2.1,
            'rainfall_annual': 1300,
            'rainfall_dry_season': 150,
            'rainfall_wet_season': 850,
            'rainfall_growing_season': 650,
            'rainfall_std': 180,
            'humidity_avg': 75,
            'evapotranspiration': 1400
        },
        infrastructure={
            'irrigation': False,
            'storage': True,
            'road_access': 'good'
        },
        farmer_preferences={
            'risk_tolerance': 'moderate',
            'yield_priority': 'high',
            'fertilizer_budget': 200,
            'sustainability_focus': True
        }
    )


if __name__ == "__main__":
    # Example usage
    from .core_models import create_sample_data
    
    print("Creating ensemble model...")
    ensemble = SeedRecommendationEnsemble()
    
    print("Generating training data...")
    training_data, targets = create_sample_data()
    
    print("Training ensemble...")
    ensemble.train(training_data, targets)
    
    print("Creating sample farm data...")
    farm_data = create_sample_farm_data()
    
    print("Generating recommendations...")
    recommendations = ensemble.recommend_seeds(
        farm_data, 
        top_n=3, 
        risk_tolerance='moderate'
    )
    
    print(f"\nTop 3 Seed Recommendations for Farmer {farm_data.farmer_id}:")
    print("=" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Seed ID: {rec.seed_id}")
        print(f"   Confidence Score: {rec.confidence_score:.3f}")
        print(f"   Expected Yield: {rec.expected_yield:.1f} tons/hectare")
        print(f"   Risk Score: {rec.risk_score:.3f}")
        print(f"   Expected ROI: {rec.cost_benefit_analysis['roi_percent']:.1f}%")
        print(f"   Key Points:")
        for explanation in rec.explanation:
            print(f"     • {explanation}")
    
    print(f"\nModel Performance Insights:")
    performance = ensemble.get_model_performance_metrics()
    
    for model_name, features in performance.items():
        print(f"\n{model_name.replace('_', ' ').title()} - Top Features:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:3]:
            print(f"  {feature}: {importance:.3f}")