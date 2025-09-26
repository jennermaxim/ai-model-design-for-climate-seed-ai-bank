"""
API Service Layer

This package provides REST API endpoints for the Climate-Adaptive Seed AI Bank system.
"""

from .endpoints import (
    RecommendationAPI,
    FarmDataAPI,
    SeedVarietyAPI,
    IoTDataAPI,
    AnalyticsAPI
)

from .middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    CORSMiddleware
)

from .utils import (
    ResponseBuilder,
    RequestValidator,
    ErrorHandler,
    APIDocumentationGenerator
)

__all__ = [
    'RecommendationAPI',
    'FarmDataAPI', 
    'SeedVarietyAPI',
    'IoTDataAPI',
    'AnalyticsAPI',
    'AuthenticationMiddleware',
    'RateLimitMiddleware',
    'LoggingMiddleware',
    'CORSMiddleware',
    'ResponseBuilder',
    'RequestValidator',
    'ErrorHandler',
    'APIDocumentationGenerator'
]