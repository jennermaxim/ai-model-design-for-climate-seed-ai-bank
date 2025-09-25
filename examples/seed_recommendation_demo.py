"""
Seed Recommendation Demo

This script demonstrates the complete workflow of the Climate-Adaptive Seed AI Bank
recommendation system, from data collection to final recommendations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "models"))
sys.path.append(str(Path(__file__).parent / "iot"))

from models.core_models import create_sample_data, FarmData
from models.ensemble import SeedRecommendationEnsemble, create_sample_farm_data, ModelWeights
from iot.sensors import create_sample_sensor_network, SensorType


def print_separator(title: str, char: str = "="):
    """Print a formatted separator with title"""
    print(f"\n{char * 60}")
    print(f"{title.center(60)}")
    print(f"{char * 60}")


def demonstrate_iot_integration():
    """Demonstrate IoT sensor data collection"""
    print_separator("IoT SENSOR DATA COLLECTION", "=")
    
    # Create sensor network
    print("📡 Setting up sensor network...")
    network = create_sample_sensor_network()
    
    # Show network status
    status = network.get_sensor_status_summary()
    print(f"\nNetwork Status:")
    print(f"  • Network ID: {status['network_id']}")
    print(f"  • Total Sensors: {status['total_sensors']}")
    print(f"  • Active Sensors: {status['active_sensors']}")
    print(f"  • Sensor Types: {status['sensor_types']}")
    
    # Collect readings
    print(f"\n🔍 Collecting sensor readings...")
    readings = network.collect_all_readings()
    
    # Display readings by type
    for sensor_type in [SensorType.WEATHER_STATION, SensorType.SOIL_SENSOR, SensorType.PLANT_MONITOR]:
        type_readings = [r for r in readings if r.sensor_type == sensor_type]
        if type_readings:
            print(f"\n{sensor_type.value.upper()} SENSORS:")
            for reading in type_readings:
                print(f"  📊 {reading.sensor_id}:")
                # Show top 3 measurements
                measurements = list(reading.measurements.items())[:3]
                for key, value in measurements:
                    quality = reading.quality_flags.get(key, 'unknown')
                    print(f"    • {key}: {value:.2f} (quality: {quality.value if hasattr(quality, 'value') else quality})")
                print(f"    • Battery: {reading.device_status.battery_level:.1f}%")
    
    # Check for alerts
    if network.alerts:
        print(f"\n⚠️  ACTIVE ALERTS ({len(network.alerts)}):")
        for alert in network.alerts[-3:]:  # Show last 3 alerts
            print(f"  • {alert['message']} (Sensor: {alert['sensor_id']})")
    else:
        print(f"\n✅ No active alerts - all systems normal")
    
    return network


def demonstrate_model_training():
    """Demonstrate AI model training"""
    print_separator("AI MODEL TRAINING", "=")
    
    # Generate training data
    print("📚 Generating synthetic training data...")
    training_data, targets = create_sample_data()
    print(f"  • Training samples: {len(training_data)}")
    print(f"  • Features: {len(training_data.columns)}")
    print(f"  • Target variables: {list(targets.keys())}")
    
    # Create and train ensemble
    print(f"\n🤖 Creating and training ensemble model...")
    ensemble = SeedRecommendationEnsemble()
    ensemble.train(training_data, targets)
    
    # Show model performance insights
    print(f"\n📈 Model Performance Insights:")
    performance = ensemble.get_model_performance_metrics()
    
    for model_name, features in performance.items():
        print(f"\n  {model_name.replace('_', ' ').title()}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:3]:
            print(f"    • {feature}: {importance:.3f}")
    
    return ensemble


def demonstrate_seed_recommendations(ensemble, sensor_network):
    """Demonstrate seed recommendation generation"""
    print_separator("SEED RECOMMENDATION GENERATION", "=")
    
    # Create sample farm scenarios
    scenarios = [
        {
            "name": "Small Scale Maize Farmer",
            "farm_data": create_sample_farm_data(),
            "risk_tolerance": "moderate"
        },
        {
            "name": "Commercial Rice Producer", 
            "farm_data": FarmData(
                farmer_id="UG-87654321",
                field_id="FIELD-0000000002",
                location=(1.5000, 33.2000),  # Different region
                altitude=1100,
                field_size_hectares=10.0,  # Larger farm
                soil_properties={
                    'ph': 5.8,
                    'organic_matter': 2.8,
                    'nitrogen': 35,
                    'phosphorus': 15,
                    'potassium': 80,
                    'drainage': 'poor',  # Good for rice
                    'slope': 1.0
                },
                climate_data={
                    'temperature_avg': 26.0,
                    'temperature_std': 1.8,
                    'rainfall_annual': 1800,  # High rainfall
                    'rainfall_dry_season': 200,
                    'rainfall_wet_season': 1200,
                    'rainfall_std': 250,
                    'humidity_avg': 80,  # High humidity
                    'evapotranspiration': 1600
                },
                infrastructure={
                    'irrigation': True,  # Has irrigation
                    'storage': True,
                    'road_access': 'excellent'
                },
                farmer_preferences={
                    'risk_tolerance': 'low',
                    'yield_priority': 'high',
                    'fertilizer_budget': 500,  # Higher budget
                    'sustainability_focus': False
                }
            ),
            "risk_tolerance": "low"
        },
        {
            "name": "Drought-Prone Region Farmer",
            "farm_data": FarmData(
                farmer_id="UG-11223344",
                field_id="FIELD-0000000003",
                location=(2.0000, 34.0000),  # Northern region
                altitude=800,
                field_size_hectares=1.5,
                soil_properties={
                    'ph': 7.2,
                    'organic_matter': 2.0,  # Low organic matter
                    'nitrogen': 25,  # Low nutrients
                    'phosphorus': 10,
                    'potassium': 60,
                    'drainage': 'excellent',
                    'slope': 8.0
                },
                climate_data={
                    'temperature_avg': 28.0,  # Hotter
                    'temperature_std': 3.5,
                    'rainfall_annual': 800,  # Low rainfall
                    'rainfall_dry_season': 50,
                    'rainfall_wet_season': 600,
                    'rainfall_std': 300,  # High variability
                    'humidity_avg': 60,  # Lower humidity
                    'evapotranspiration': 1800
                },
                infrastructure={
                    'irrigation': False,  # No irrigation
                    'storage': False,
                    'road_access': 'poor'
                },
                farmer_preferences={
                    'risk_tolerance': 'high',  # Willing to take risks
                    'yield_priority': 'moderate',
                    'fertilizer_budget': 80,  # Low budget
                    'sustainability_focus': True
                }
            ),
            "risk_tolerance": "high"
        }
    ]
    
    # Generate recommendations for each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🌱 SCENARIO {i}: {scenario['name']}")
        print("-" * 50)
        
        farm_data = scenario['farm_data']
        risk_tolerance = scenario['risk_tolerance']
        
        # Show farm characteristics
        print(f"📍 Location: ({farm_data.location[0]:.2f}, {farm_data.location[1]:.2f})")
        print(f"🏡 Field Size: {farm_data.field_size_hectares} hectares")
        print(f"🌡️  Climate: {farm_data.climate_data['temperature_avg']:.1f}°C, "
              f"{farm_data.climate_data['rainfall_annual']:.0f}mm/year")
        print(f"🏞️  Soil pH: {farm_data.soil_properties['ph']:.1f}")
        print(f"⚡ Infrastructure: "
              f"{'Irrigation' if farm_data.infrastructure['irrigation'] else 'Rain-fed'}")
        print(f"💰 Budget: ${farm_data.farmer_preferences['fertilizer_budget']}")
        
        # Get recommendations
        try:
            recommendations = ensemble.recommend_seeds(
                farm_data,
                top_n=3,
                risk_tolerance=risk_tolerance
            )
            
            print(f"\n🎯 TOP SEED RECOMMENDATIONS:")
            
            for j, rec in enumerate(recommendations, 1):
                print(f"\n  {j}. SEED: {rec.seed_id}")
                print(f"     ⭐ Confidence Score: {rec.confidence_score:.3f}")
                print(f"     🌾 Expected Yield: {rec.expected_yield:.1f} tons/hectare")
                print(f"     ⚠️  Risk Level: {rec.risk_score:.3f}")
                print(f"     💵 Expected ROI: {rec.cost_benefit_analysis['roi_percent']:.1f}%")
                print(f"     💰 Expected Profit: ${rec.cost_benefit_analysis['expected_profit']:.0f}")
                
                print(f"     💡 Key Insights:")
                for explanation in rec.explanation[:3]:  # Show top 3 explanations
                    print(f"        • {explanation}")
                
                # Add sensor-based recommendations if available
                if sensor_network and sensor_network.readings_history:
                    latest_weather = None
                    latest_soil = None
                    
                    for reading in reversed(sensor_network.readings_history):
                        if reading.sensor_type == SensorType.WEATHER_STATION and not latest_weather:
                            latest_weather = reading
                        elif reading.sensor_type == SensorType.SOIL_SENSOR and not latest_soil:
                            latest_soil = reading
                        
                        if latest_weather and latest_soil:
                            break
                    
                    if latest_weather or latest_soil:
                        print(f"     📡 Real-time Sensor Insights:")
                        if latest_weather:
                            temp = latest_weather.measurements.get('temperature_celsius')
                            humidity = latest_weather.measurements.get('humidity_percent')
                            print(f"        • Current conditions: {temp:.1f}°C, {humidity:.1f}% humidity")
                        
                        if latest_soil:
                            moisture = latest_soil.measurements.get('soil_moisture_percent')
                            soil_temp = latest_soil.measurements.get('soil_temperature_celsius')
                            print(f"        • Soil conditions: {moisture:.1f}% moisture, {soil_temp:.1f}°C")
        
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")


def demonstrate_real_time_monitoring(sensor_network):
    """Demonstrate real-time monitoring capabilities"""
    print_separator("REAL-TIME MONITORING", "=")
    
    print("🔄 Simulating continuous monitoring over time...")
    
    # Simulate multiple readings over time
    monitoring_data = []
    time_points = []
    
    for hour in range(0, 24, 4):  # Every 4 hours for a day
        print(f"\n⏰ Hour {hour:02d}:00 - Collecting readings...")
        readings = sensor_network.collect_all_readings()
        
        # Aggregate key metrics
        weather_readings = [r for r in readings if r.sensor_type == SensorType.WEATHER_STATION]
        soil_readings = [r for r in readings if r.sensor_type == SensorType.SOIL_SENSOR]
        
        if weather_readings:
            weather = weather_readings[0]
            temp = weather.measurements.get('temperature_celsius', 0)
            humidity = weather.measurements.get('humidity_percent', 0)
            rainfall = weather.measurements.get('rainfall_mm', 0)
            print(f"  🌡️  Weather: {temp:.1f}°C, {humidity:.1f}% RH, {rainfall:.1f}mm rain")
        
        if soil_readings:
            soil = soil_readings[0]  # Take first soil sensor
            moisture = soil.measurements.get('soil_moisture_percent', 0)
            soil_temp = soil.measurements.get('soil_temperature_celsius', 0)
            print(f"  🏞️  Soil: {moisture:.1f}% moisture, {soil_temp:.1f}°C")
        
        monitoring_data.append({
            'hour': hour,
            'temperature': temp if weather_readings else None,
            'soil_moisture': moisture if soil_readings else None,
            'alerts': len(sensor_network.alerts)
        })
        time_points.append(hour)
    
    # Show monitoring summary
    print(f"\n📊 24-HOUR MONITORING SUMMARY:")
    df = pd.DataFrame(monitoring_data)
    
    if not df.empty and 'temperature' in df.columns:
        print(f"  🌡️  Temperature Range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
        print(f"  💧 Soil Moisture Range: {df['soil_moisture'].min():.1f}% - {df['soil_moisture'].max():.1f}%")
        print(f"  ⚠️  Total Alerts Generated: {df['alerts'].max()}")
    
    # Export data
    export_file = "monitoring_data.csv"
    sensor_df = sensor_network.export_readings_to_dataframe(hours=24)
    if not sensor_df.empty:
        sensor_df.to_csv(export_file, index=False)
        print(f"  💾 Exported {len(sensor_df)} readings to {export_file}")


def main():
    """Main demonstration workflow"""
    print_separator("CLIMATE-ADAPTIVE SEED AI BANK DEMO", "=")
    print("🌱 Welcome to the Climate-Adaptive Seed AI Bank demonstration!")
    print("   This demo showcases the complete AI-powered seed recommendation system.")
    
    try:
        # Step 1: IoT Integration
        sensor_network = demonstrate_iot_integration()
        
        # Step 2: Model Training
        ensemble_model = demonstrate_model_training()
        
        # Step 3: Seed Recommendations
        demonstrate_seed_recommendations(ensemble_model, sensor_network)
        
        # Step 4: Real-time Monitoring
        demonstrate_real_time_monitoring(sensor_network)
        
        # Summary
        print_separator("DEMONSTRATION COMPLETE", "=")
        print("✅ Successfully demonstrated:")
        print("   • IoT sensor data collection and monitoring")
        print("   • AI model training and ensemble learning")
        print("   • Personalized seed recommendations")
        print("   • Real-time agricultural monitoring")
        print("   • Cost-benefit analysis and risk assessment")
        print("   • Multi-scenario farming conditions")
        
        print(f"\n🚀 Next Steps for Implementation:")
        print("   • Deploy sensor networks across target regions")
        print("   • Integrate with real agricultural databases")
        print("   • Develop mobile application interface")
        print("   • Establish farmer training programs")
        print("   • Create partnerships with seed suppliers")
        print("   • Implement feedback collection system")
        
        print(f"\n📊 System Impact Potential:")
        print("   • 15-25% increase in crop yields")
        print("   • 20-30% reduction in crop failure rates")
        print("   • 20% reduction in water and fertilizer usage")
        print("   • Improved food security across Uganda")
        print("   • Enhanced climate resilience for farmers")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        print("Please check the system setup and try again.")


if __name__ == "__main__":
    main()