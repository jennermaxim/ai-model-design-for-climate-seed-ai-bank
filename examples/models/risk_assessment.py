"""
Risk Assessment Models

This module implements comprehensive risk assessment models for agricultural
decision-making, including climate risks, market risks, and production risks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .core_models import FarmData, SeedVariety, BaseModel


class RiskAssessmentModel(BaseModel):
    """
    Comprehensive risk assessment model that evaluates multiple types of risks
    including climate, production, market, and economic risks.
    """
    
    def __init__(self):
        super().__init__()
        self.climate_risk_model = ClimateRiskModel()
        self.production_risk_model = ProductionRiskModel()
        self.market_risk_model = MarketRiskModel()
        self.economic_risk_model = EconomicRiskModel()
        
        self.risk_weights = {
            'climate': 0.35,
            'production': 0.30,
            'market': 0.20,
            'economic': 0.15
        }
        
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }
    
    def assess_comprehensive_risk(
        self, 
        farm_data: FarmData, 
        seed_varieties: List[SeedVariety],
        market_data: Optional[Dict] = None,
        time_horizon: int = 12  # months
    ) -> Dict[str, Dict[str, Union[float, str, List[str]]]]:
        """
        Assess comprehensive risk for multiple seed varieties
        
        Args:
            farm_data: Farm characteristics and conditions
            seed_varieties: List of seed varieties to evaluate
            market_data: Optional market price and demand data
            time_horizon: Risk assessment time horizon in months
            
        Returns:
            Dictionary mapping seed_id to comprehensive risk assessment
        """
        results = {}
        
        for seed in seed_varieties:
            # Assess individual risk components
            climate_risks = self.climate_risk_model.assess_climate_risk(farm_data, seed, time_horizon)
            production_risks = self.production_risk_model.assess_production_risk(farm_data, seed)
            market_risks = self.market_risk_model.assess_market_risk(seed, market_data)
            economic_risks = self.economic_risk_model.assess_economic_risk(farm_data, seed)
            
            # Calculate weighted overall risk
            overall_risk = (
                self.risk_weights['climate'] * climate_risks['overall_score'] +
                self.risk_weights['production'] * production_risks['overall_score'] +
                self.risk_weights['market'] * market_risks['overall_score'] +
                self.risk_weights['economic'] * economic_risks['overall_score']
            )
            
            # Determine risk category
            risk_category = self._categorize_risk(overall_risk)
            
            # Identify top risk factors
            top_risks = self._identify_top_risks(climate_risks, production_risks, market_risks, economic_risks)
            
            # Generate risk mitigation recommendations
            mitigation_strategies = self._generate_mitigation_strategies(
                farm_data, seed, climate_risks, production_risks, market_risks, economic_risks
            )
            
            results[seed.seed_id] = {
                'overall_risk_score': overall_risk,
                'risk_category': risk_category,
                'climate_risk': climate_risks['overall_score'],
                'production_risk': production_risks['overall_score'],
                'market_risk': market_risks['overall_score'],
                'economic_risk': economic_risks['overall_score'],
                'top_risk_factors': top_risks,
                'mitigation_strategies': mitigation_strategies,
                'detailed_assessment': {
                    'climate': climate_risks,
                    'production': production_risks,
                    'market': market_risks,
                    'economic': economic_risks
                }
            }
        
        return results
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk levels"""
        if risk_score <= self.risk_thresholds['low']:
            return 'low'
        elif risk_score <= self.risk_thresholds['moderate']:
            return 'moderate'
        elif risk_score <= self.risk_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def _identify_top_risks(self, *risk_assessments) -> List[str]:
        """Identify the top risk factors across all categories"""
        all_risks = []
        
        for assessment in risk_assessments:
            if isinstance(assessment, dict):
                for key, value in assessment.items():
                    if isinstance(value, (int, float)) and key != 'overall_score':
                        all_risks.append((key, value))
        
        # Sort by risk score and return top 3
        all_risks.sort(key=lambda x: x[1], reverse=True)
        return [risk[0] for risk in all_risks[:3]]
    
    def _generate_mitigation_strategies(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety,
        climate_risks: Dict,
        production_risks: Dict,
        market_risks: Dict,
        economic_risks: Dict
    ) -> List[str]:
        """Generate risk mitigation strategies based on risk assessment"""
        strategies = []
        
        # Climate risk mitigation
        if climate_risks['drought_risk'] > 0.6:
            strategies.append("Implement drought-resistant farming techniques and water conservation")
        if climate_risks['flood_risk'] > 0.5:
            strategies.append("Improve drainage systems and consider raised bed cultivation")
        if climate_risks['temperature_stress'] > 0.5:
            strategies.append("Use shade nets or adjust planting timing to avoid extreme temperatures")
        
        # Production risk mitigation
        if production_risks['pest_disease_risk'] > 0.6:
            strategies.append("Implement integrated pest management and regular crop monitoring")
        if production_risks['nutrient_deficiency_risk'] > 0.5:
            strategies.append("Conduct soil testing and apply targeted fertilization")
        
        # Market risk mitigation
        if market_risks['price_volatility'] > 0.6:
            strategies.append("Consider crop insurance and forward contracting")
        if market_risks['demand_uncertainty'] > 0.5:
            strategies.append("Diversify crops and explore value-added processing")
        
        # Economic risk mitigation
        if economic_risks['input_cost_risk'] > 0.6:
            strategies.append("Bulk purchase inputs and explore cooperative buying")
        if economic_risks['financing_risk'] > 0.5:
            strategies.append("Secure agricultural credit and explore microfinance options")
        
        return strategies[:5]  # Limit to top 5 strategies


class ClimateRiskModel(BaseModel):
    """
    Specialized model for assessing climate-related risks including
    drought, floods, temperature extremes, and weather variability.
    """
    
    def __init__(self):
        super().__init__()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.extreme_weather_thresholds = {
            'drought': {'rainfall_percentile': 20, 'severity_months': 3},
            'flood': {'rainfall_percentile': 90, 'duration_days': 7},
            'heat_stress': {'temperature_threshold': 35, 'duration_days': 5},
            'cold_stress': {'temperature_threshold': 10, 'duration_days': 3}
        }
    
    def assess_climate_risk(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety,
        time_horizon: int = 12
    ) -> Dict[str, float]:
        """Assess various climate risks for the farm and seed variety"""
        climate = farm_data.climate_data
        requirements = seed_variety.environmental_requirements
        
        risks = {}
        
        # Drought risk assessment
        annual_rainfall = climate.get('annual_rainfall', 800)
        min_rainfall_req = requirements.get('rainfall', (400, 1200))[0]
        if annual_rainfall < min_rainfall_req * 1.2:  # 20% buffer
            drought_risk = min(1.0, (min_rainfall_req * 1.2 - annual_rainfall) / min_rainfall_req)
        else:
            drought_risk = 0.0
        risks['drought_risk'] = drought_risk
        
        # Flood risk assessment
        max_rainfall_tolerance = requirements.get('rainfall', (400, 1200))[1]
        rainfall_variability = climate.get('rainfall_variability', 0.3)
        if annual_rainfall > max_rainfall_tolerance * 0.8:
            flood_risk = min(1.0, (annual_rainfall - max_rainfall_tolerance * 0.8) / max_rainfall_tolerance)
            flood_risk *= (1 + rainfall_variability)  # Higher variability increases flood risk
        else:
            flood_risk = rainfall_variability * 0.5
        risks['flood_risk'] = min(1.0, flood_risk)
        
        # Temperature stress risks
        avg_temp = climate.get('avg_temperature', 25)
        temp_range = requirements.get('temperature', (15, 35))
        
        if avg_temp > temp_range[1]:
            heat_stress = (avg_temp - temp_range[1]) / 10.0
        else:
            heat_stress = 0.0
        risks['temperature_stress'] = min(1.0, heat_stress)
        
        if avg_temp < temp_range[0]:
            cold_stress = (temp_range[0] - avg_temp) / 10.0
        else:
            cold_stress = 0.0
        risks['cold_stress'] = min(1.0, cold_stress)
        
        # Weather variability risk
        temp_variability = climate.get('temperature_variability', 0.2)
        weather_variability_risk = (temp_variability + rainfall_variability) / 2.0
        risks['weather_variability'] = min(1.0, weather_variability_risk)
        
        # Extreme weather frequency risk
        extreme_weather_frequency = climate.get('extreme_weather_days_per_year', 10) / 365.0
        risks['extreme_weather_frequency'] = min(1.0, extreme_weather_frequency * 5)
        
        # Climate change adaptation risk
        climate_trend = climate.get('temperature_trend_per_decade', 0.2)  # °C per decade
        adaptation_difficulty = abs(climate_trend) / 2.0  # Normalize
        risks['climate_change_adaptation'] = min(1.0, adaptation_difficulty)
        
        # Overall climate risk
        risk_values = list(risks.values())
        risks['overall_score'] = np.mean(risk_values)
        
        return risks
    
    def predict_extreme_weather_probability(
        self, 
        farm_data: FarmData, 
        event_type: str,
        months_ahead: int = 3
    ) -> float:
        """Predict probability of extreme weather events"""
        climate = farm_data.climate_data
        
        if event_type == 'drought':
            base_prob = 0.1 if climate.get('annual_rainfall', 800) > 600 else 0.3
            seasonal_factor = 1.5 if months_ahead in [1, 2, 3] else 1.0  # Dry season
            return min(1.0, base_prob * seasonal_factor)
        
        elif event_type == 'flood':
            base_prob = 0.15 if climate.get('annual_rainfall', 800) > 1000 else 0.05
            seasonal_factor = 2.0 if months_ahead in [6, 7, 8] else 1.0  # Wet season
            return min(1.0, base_prob * seasonal_factor)
        
        elif event_type == 'heatwave':
            base_prob = 0.2 if climate.get('max_temperature', 30) > 32 else 0.1
            return min(1.0, base_prob)
        
        return 0.1  # Default low probability


class ProductionRiskModel(BaseModel):
    """
    Model for assessing production-related risks including pest/disease
    outbreaks, input supply issues, and yield variability.
    """
    
    def __init__(self):
        super().__init__()
        self.pest_disease_patterns = {
            'high_humidity_pests': ['aphids', 'fungal_diseases'],
            'drought_stress_pests': ['spider_mites', 'thrips'],
            'temperature_diseases': ['bacterial_wilt', 'viral_diseases']
        }
    
    def assess_production_risk(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety
    ) -> Dict[str, float]:
        """Assess production risks for the farm and seed variety"""
        risks = {}
        
        # Pest and disease risk
        pest_risk = self._assess_pest_disease_risk(farm_data, seed_variety)
        risks.update(pest_risk)
        
        # Yield variability risk
        yield_potential = seed_variety.agronomic_properties.get('yield_potential', 3.0)
        historical_variance = 0.3  # Assume 30% coefficient of variation
        yield_risk = min(1.0, historical_variance * (4.0 - yield_potential) / 4.0)
        risks['yield_variability_risk'] = yield_risk
        
        # Input supply risk
        infrastructure = farm_data.infrastructure
        supply_risk = 1.0 - np.mean([
            infrastructure.get('fertilizer_access', False),
            infrastructure.get('seed_access', True),
            infrastructure.get('pest_control', False)
        ])
        risks['input_supply_risk'] = supply_risk
        
        # Soil degradation risk
        soil = farm_data.soil_properties
        organic_matter = soil.get('organic_matter', 2.5)
        soil_health_risk = max(0.0, (2.5 - organic_matter) / 2.5)
        risks['soil_degradation_risk'] = soil_health_risk
        
        # Nutrient deficiency risk
        nutrient_levels = [
            soil.get('nitrogen', 15) / 20.0,
            soil.get('phosphorus', 10) / 15.0,
            soil.get('potassium', 100) / 150.0
        ]
        nutrient_adequacy = np.mean(nutrient_levels)
        nutrient_risk = max(0.0, 1.0 - nutrient_adequacy)
        risks['nutrient_deficiency_risk'] = nutrient_risk
        
        # Technology adoption risk
        tech_level = infrastructure.get('mechanization_level', 0.3)
        tech_risk = 1.0 - tech_level
        risks['technology_risk'] = tech_risk
        
        # Overall production risk
        risk_values = [v for k, v in risks.items() if k != 'overall_score']
        risks['overall_score'] = np.mean(risk_values)
        
        return risks
    
    def _assess_pest_disease_risk(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety
    ) -> Dict[str, float]:
        """Assess specific pest and disease risks"""
        climate = farm_data.climate_data
        traits = seed_variety.genetic_traits
        
        # Base resistance levels
        disease_resistance = self._parse_resistance(traits.get('disease_resistance', 'medium'))
        pest_resistance = self._parse_resistance(traits.get('pest_resistance', 'medium'))
        
        # Climate-induced pest/disease pressure
        humidity = climate.get('humidity', 60)
        temperature = climate.get('avg_temperature', 25)
        
        # Fungal disease risk (high humidity)
        if humidity > 70:
            fungal_risk = (humidity - 70) / 30.0 * (1.0 - disease_resistance)
        else:
            fungal_risk = 0.0
        
        # Insect pest risk (temperature dependent)
        if 25 <= temperature <= 30:  # Optimal for many pests
            insect_risk = 0.8 * (1.0 - pest_resistance)
        else:
            insect_risk = 0.4 * (1.0 - pest_resistance)
        
        # Viral disease risk (vector-borne, related to pest activity)
        viral_risk = insect_risk * 0.7 * (1.0 - disease_resistance)
        
        return {
            'fungal_disease_risk': min(1.0, fungal_risk),
            'insect_pest_risk': min(1.0, insect_risk),
            'viral_disease_risk': min(1.0, viral_risk),
            'pest_disease_risk': min(1.0, np.mean([fungal_risk, insect_risk, viral_risk]))
        }
    
    def _parse_resistance(self, resistance_str: str) -> float:
        """Parse resistance string to numeric value"""
        resistance_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'very_high': 0.9
        }
        return resistance_map.get(resistance_str.lower(), 0.5)


class MarketRiskModel(BaseModel):
    """
    Model for assessing market-related risks including price volatility,
    demand fluctuations, and market access issues.
    """
    
    def __init__(self):
        super().__init__()
        self.price_volatility_factors = {
            'staple_crops': 0.3,
            'cash_crops': 0.6,
            'specialty_crops': 0.8
        }
    
    def assess_market_risk(
        self, 
        seed_variety: SeedVariety,
        market_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Assess market risks for the seed variety"""
        crop_type = seed_variety.crop_type.lower()
        
        # Determine crop category for risk assessment
        if crop_type in ['maize', 'rice', 'wheat', 'beans']:
            crop_category = 'staple_crops'
        elif crop_type in ['coffee', 'cotton', 'tobacco']:
            crop_category = 'cash_crops'
        else:
            crop_category = 'specialty_crops'
        
        risks = {}
        
        # Price volatility risk
        base_volatility = self.price_volatility_factors[crop_category]
        if market_data and 'price_history' in market_data:
            # Calculate actual price volatility if data is available
            prices = market_data['price_history']
            if len(prices) > 1:
                price_changes = np.diff(prices) / prices[:-1]
                actual_volatility = np.std(price_changes)
                volatility_risk = min(1.0, actual_volatility / 0.5)
            else:
                volatility_risk = base_volatility
        else:
            volatility_risk = base_volatility
        risks['price_volatility'] = volatility_risk
        
        # Market demand risk
        if market_data and 'demand_trend' in market_data:
            demand_trend = market_data['demand_trend']  # -1 to 1
            demand_risk = max(0.0, -demand_trend * 0.5 + 0.3)
        else:
            # Default based on crop type
            demand_risk = 0.3 if crop_category == 'staple_crops' else 0.5
        risks['demand_uncertainty'] = demand_risk
        
        # Market access risk
        if market_data and 'market_distance' in market_data:
            distance = market_data['market_distance']  # km
            access_risk = min(1.0, distance / 50.0)  # Risk increases with distance
        else:
            access_risk = 0.4  # Default moderate risk
        risks['market_access_risk'] = access_risk
        
        # Competition risk
        market_saturation = market_data.get('market_saturation', 0.5) if market_data else 0.5
        competition_risk = market_saturation
        risks['competition_risk'] = competition_risk
        
        # Export dependency risk (for cash crops)
        if crop_category == 'cash_crops':
            export_risk = 0.6  # Higher risk due to international market exposure
        else:
            export_risk = 0.2
        risks['export_dependency_risk'] = export_risk
        
        # Overall market risk
        risk_values = [v for k, v in risks.items() if k != 'overall_score']
        risks['overall_score'] = np.mean(risk_values)
        
        return risks


class EconomicRiskModel(BaseModel):
    """
    Model for assessing economic and financial risks including input costs,
    financing availability, and profitability risks.
    """
    
    def __init__(self):
        super().__init__()
        self.cost_inflation_rates = {
            'fertilizer': 0.15,  # 15% annual inflation
            'seeds': 0.08,
            'pesticides': 0.12,
            'labor': 0.10,
            'fuel': 0.20
        }
    
    def assess_economic_risk(
        self, 
        farm_data: FarmData, 
        seed_variety: SeedVariety
    ) -> Dict[str, float]:
        """Assess economic and financial risks"""
        risks = {}
        
        # Input cost inflation risk
        cost_risk = np.mean(list(self.cost_inflation_rates.values()))
        risks['input_cost_risk'] = min(1.0, cost_risk)
        
        # Financing access risk
        farm_size = farm_data.field_size_hectares
        if farm_size < 1:  # Small-scale farmers
            financing_risk = 0.7
        elif farm_size < 5:  # Medium-scale farmers
            financing_risk = 0.4
        else:  # Large-scale farmers
            financing_risk = 0.2
        risks['financing_risk'] = financing_risk
        
        # Profitability risk
        yield_potential = seed_variety.agronomic_properties.get('yield_potential', 3.0)
        input_intensity = self._calculate_input_intensity(seed_variety)
        profit_margin = (yield_potential * 200 - input_intensity * 500) / (yield_potential * 200)
        profitability_risk = max(0.0, 1.0 - profit_margin / 0.5)  # Risk if margin < 50%
        risks['profitability_risk'] = min(1.0, profitability_risk)
        
        # Cash flow risk
        maturity_days = seed_variety.agronomic_properties.get('maturity_days', 120)
        cash_flow_risk = min(1.0, maturity_days / 200.0)  # Longer maturity = higher risk
        risks['cash_flow_risk'] = cash_flow_risk
        
        # Insurance availability risk
        insurance_risk = 0.6 if farm_size < 2 else 0.3  # Smaller farms have less access
        risks['insurance_access_risk'] = insurance_risk
        
        # Overall economic risk
        risk_values = [v for k, v in risks.items() if k != 'overall_score']
        risks['overall_score'] = np.mean(risk_values)
        
        return risks
    
    def _calculate_input_intensity(self, seed_variety: SeedVariety) -> float:
        """Calculate relative input intensity score (0-1)"""
        # High-yielding varieties typically require more inputs
        yield_potential = seed_variety.agronomic_properties.get('yield_potential', 3.0)
        maturity_days = seed_variety.agronomic_properties.get('maturity_days', 120)
        
        # Normalize factors
        yield_factor = min(1.0, yield_potential / 5.0)
        maturity_factor = min(1.0, maturity_days / 150.0)
        
        return (yield_factor + maturity_factor) / 2.0