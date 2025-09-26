"""
Climate Data Processor

Handles processing, cleaning, and feature engineering for climate and weather data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ClimateDataProcessor:
    """
    Processor for climate and weather data with advanced feature engineering
    and data quality assessment capabilities.
    """
    
    def __init__(self):
        self.required_columns = [
            'date', 'temperature', 'rainfall', 'humidity', 
            'wind_speed', 'solar_radiation'
        ]
        self.optional_columns = [
            'pressure', 'evapotranspiration', 'soil_temperature',
            'leaf_wetness', 'uv_index'
        ]
        self.outlier_thresholds = {
            'temperature': (-10, 50),  # Celsius
            'rainfall': (0, 300),      # mm/day
            'humidity': (0, 100),      # %
            'wind_speed': (0, 100),    # km/h
            'solar_radiation': (0, 35) # MJ/m²/day
        }
    
    def process_raw_climate_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw climate data with cleaning and feature engineering
        
        Args:
            raw_data: DataFrame with raw climate measurements
            
        Returns:
            Processed DataFrame with engineered features
        """
        print("Processing raw climate data...")
        
        # Validate input data
        processed_data = self._validate_and_clean_data(raw_data.copy())
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Remove outliers
        processed_data = self._remove_outliers(processed_data)
        
        # Engineer features
        processed_data = self._engineer_climate_features(processed_data)
        
        # Calculate derived metrics
        processed_data = self._calculate_derived_metrics(processed_data)
        
        print(f"Climate data processing complete. Shape: {processed_data.shape}")
        return processed_data
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data structure and clean basic issues"""
        # Ensure date column exists and is datetime
        if 'date' not in data.columns:
            raise ValueError("Date column is required")
        
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        
        # Convert numeric columns
        numeric_cols = [col for col in self.required_columns + self.optional_columns 
                       if col in data.columns and col != 'date']
        
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using various interpolation methods"""
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().sum() > 0:
                # Use different methods based on data characteristics
                if col in ['temperature', 'humidity']:
                    # Linear interpolation for smooth variables
                    data[col] = data[col].interpolate(method='linear')
                elif col in ['rainfall']:
                    # Forward fill for precipitation (often zero)
                    data[col] = data[col].fillna(0)
                else:
                    # Use median for other variables
                    data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on predefined thresholds"""
        original_length = len(data)
        
        for col, (min_val, max_val) in self.outlier_thresholds.items():
            if col in data.columns:
                data = data[(data[col] >= min_val) & (data[col] <= max_val)]
        
        removed_count = original_length - len(data)
        if removed_count > 0:
            print(f"Removed {removed_count} outlier records")
        
        return data.reset_index(drop=True)
    
    def _engineer_climate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced climate features"""
        # Time-based features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_year'] = data['date'].dt.dayofyear
        data['season'] = data['month'].map(self._get_season)
        
        # Temperature features
        if 'temperature' in data.columns:
            data['temp_rolling_7day'] = data['temperature'].rolling(window=7, center=True).mean()
            data['temp_rolling_30day'] = data['temperature'].rolling(window=30, center=True).mean()
            data['temp_deviation_from_seasonal'] = (
                data['temperature'] - data.groupby('month')['temperature'].transform('mean')
            )
        
        # Rainfall features
        if 'rainfall' in data.columns:
            data['rainfall_cumulative_7day'] = data['rainfall'].rolling(window=7).sum()
            data['rainfall_cumulative_30day'] = data['rainfall'].rolling(window=30).sum()
            data['dry_days_streak'] = self._calculate_dry_streak(data['rainfall'])
            data['wet_days_streak'] = self._calculate_wet_streak(data['rainfall'])
        
        # Growing degree days (base 10°C)
        if 'temperature' in data.columns:
            data['gdd_base10'] = np.maximum(0, data['temperature'] - 10)
            data['gdd_cumulative'] = data['gdd_base10'].cumsum()
        
        # Vapor pressure deficit (if humidity available)
        if 'temperature' in data.columns and 'humidity' in data.columns:
            data['vpd'] = self._calculate_vpd(data['temperature'], data['humidity'])
        
        # Climate stress indicators
        data['heat_stress_days'] = (data['temperature'] > 35).astype(int) if 'temperature' in data.columns else 0
        data['drought_stress_indicator'] = self._calculate_drought_stress(data)
        
        return data
    
    def _calculate_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived climate metrics"""
        # Monthly aggregations
        monthly_stats = data.groupby(['year', 'month']).agg({
            'temperature': ['mean', 'min', 'max', 'std'],
            'rainfall': ['sum', 'count', 'max'],
            'humidity': ['mean', 'std'] if 'humidity' in data.columns else ['mean']
        }).round(2)
        
        # Flatten column names
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
        monthly_stats = monthly_stats.reset_index()
        
        # Merge back to main data
        data = data.merge(monthly_stats, on=['year', 'month'], how='left', suffixes=('', '_monthly'))
        
        # Annual aggregations
        annual_stats = data.groupby('year').agg({
            'rainfall': 'sum',
            'temperature': 'mean',
            'heat_stress_days': 'sum'
        }).add_suffix('_annual').reset_index()
        
        data = data.merge(annual_stats, on='year', how='left')
        
        return data
    
    def _get_season(self, month: int) -> str:
        """Map month to season (Uganda context)"""
        if month in [12, 1, 2]:
            return 'dry_season'
        elif month in [3, 4, 5]:
            return 'first_rains'
        elif month in [6, 7, 8]:
            return 'dry_spell'
        else:
            return 'second_rains'
    
    def _calculate_dry_streak(self, rainfall: pd.Series, threshold: float = 1.0) -> pd.Series:
        """Calculate consecutive dry days"""
        dry_days = (rainfall <= threshold).astype(int)
        streaks = dry_days.groupby((dry_days != dry_days.shift()).cumsum()).cumsum()
        return streaks * dry_days  # Zero out when not in dry streak
    
    def _calculate_wet_streak(self, rainfall: pd.Series, threshold: float = 1.0) -> pd.Series:
        """Calculate consecutive wet days"""
        wet_days = (rainfall > threshold).astype(int)
        streaks = wet_days.groupby((wet_days != wet_days.shift()).cumsum()).cumsum()
        return streaks * wet_days  # Zero out when not in wet streak
    
    def _calculate_vpd(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate Vapor Pressure Deficit"""
        # Saturated vapor pressure (kPa)
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        # Actual vapor pressure
        ea = es * (humidity / 100.0)
        # Vapor pressure deficit
        vpd = es - ea
        return vpd
    
    def _calculate_drought_stress(self, data: pd.DataFrame) -> pd.Series:
        """Calculate drought stress indicator"""
        if 'rainfall_cumulative_30day' in data.columns:
            # Define drought as less than 50mm in 30 days
            drought_threshold = 50
            return (data['rainfall_cumulative_30day'] < drought_threshold).astype(int)
        else:
            return pd.Series(0, index=data.index)
    
    def generate_climate_summary(self, data: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Generate comprehensive climate summary statistics"""
        summary = {}
        
        if 'temperature' in data.columns:
            summary.update({
                'avg_temperature': round(data['temperature'].mean(), 2),
                'min_temperature': round(data['temperature'].min(), 2),
                'max_temperature': round(data['temperature'].max(), 2),
                'temperature_variability': round(data['temperature'].std(), 2)
            })
        
        if 'rainfall' in data.columns:
            summary.update({
                'annual_rainfall': round(data['rainfall'].sum(), 2),
                'max_daily_rainfall': round(data['rainfall'].max(), 2),
                'rainy_days': int((data['rainfall'] > 1).sum()),
                'rainfall_variability': round(data['rainfall'].std(), 2)
            })
        
        if 'humidity' in data.columns:
            summary.update({
                'avg_humidity': round(data['humidity'].mean(), 2),
                'humidity_range': round(data['humidity'].max() - data['humidity'].min(), 2)
            })
        
        # Growing season metrics
        if 'gdd_cumulative' in data.columns:
            summary['total_growing_degree_days'] = round(data['gdd_cumulative'].iloc[-1], 2)
        
        if 'heat_stress_days' in data.columns:
            summary['heat_stress_days'] = int(data['heat_stress_days'].sum())
        
        if 'drought_stress_indicator' in data.columns:
            summary['drought_stress_periods'] = int(data['drought_stress_indicator'].sum())
        
        return summary
    
    def assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Union[float, int, str]]:
        """Assess quality of climate data"""
        quality_report = {
            'total_records': len(data),
            'date_range': f"{data['date'].min()} to {data['date'].max()}",
            'completeness_score': 0.0,
            'quality_issues': []
        }
        
        # Check data completeness
        total_possible_values = len(data) * len(self.required_columns)
        missing_values = data[self.required_columns].isnull().sum().sum()
        completeness = 1.0 - (missing_values / total_possible_values)
        quality_report['completeness_score'] = round(completeness, 3)
        
        # Check for data gaps
        date_diff = data['date'].diff().dt.days
        gaps = (date_diff > 1).sum()
        if gaps > 0:
            quality_report['quality_issues'].append(f"{gaps} date gaps found")
        
        # Check for suspect values
        for col in ['temperature', 'humidity', 'rainfall']:
            if col in data.columns:
                zero_values = (data[col] == 0).sum()
                if col == 'temperature' and zero_values > len(data) * 0.1:
                    quality_report['quality_issues'].append(f"High number of zero temperatures: {zero_values}")
                elif col == 'humidity' and zero_values > len(data) * 0.05:
                    quality_report['quality_issues'].append(f"Suspect zero humidity values: {zero_values}")
        
        # Overall quality score
        if completeness > 0.95 and len(quality_report['quality_issues']) == 0:
            quality_report['overall_quality'] = 'Excellent'
        elif completeness > 0.90 and len(quality_report['quality_issues']) < 2:
            quality_report['overall_quality'] = 'Good'
        elif completeness > 0.80:
            quality_report['overall_quality'] = 'Fair'
        else:
            quality_report['overall_quality'] = 'Poor'
        
        return quality_report