"""
AI Models Package

This package contains all the AI model implementations for the Climate-Adaptive Seed AI Bank.
"""

from .core_models import (
    FarmData,
    SeedVariety, 
    Recommendation,
    BaseModel
)

from .seed_matching import (
    ClimateCompatibilityModel,
    SoilCompatibilityModel,
    AdaptabilityModel
)

from .yield_prediction import (
    YieldPredictionModel,
    SeasonalYieldModel
)

from .risk_assessment import (
    RiskAssessmentModel,
    ClimateRiskModel,
    ProductionRiskModel,
    MarketRiskModel,
    EconomicRiskModel
)

from .ensemble import (
    SeedRecommendationEnsemble,
    ModelWeights
)

__all__ = [
    'FarmData',
    'SeedVariety',
    'Recommendation',
    'BaseModel',
    'ClimateCompatibilityModel',
    'SoilCompatibilityModel', 
    'AdaptabilityModel',
    'YieldPredictionModel',
    'SeasonalYieldModel',
    'RiskAssessmentModel',
    'ClimateRiskModel',
    'ProductionRiskModel',
    'MarketRiskModel',
    'EconomicRiskModel',
    'SeedRecommendationEnsemble',
    'ModelWeights'
]