#!/usr/bin/env python3
"""
Sample Crime Data Analysis Script
Demonstrates how to use the enhanced crime dataset with the simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_analyze_crime_data():
    """Load and perform basic analysis on crime datasets"""
    
    # Load both datasets
    print("Loading crime datasets...")
    
    # Original dataset
    original_path = 'data/real_crime_data.csv'
    if os.path.exists(original_path):
        original_data = pd.read_csv(original_path)
        print(f"Original dataset: {len(original_data)} records")
    else:
        print("Original dataset not found")
        original_data = None
    
    # Enhanced dataset
    enhanced_path = 'data/enhanced_crime_data.csv'
    if os.path.exists(enhanced_path):
        enhanced_data = pd.read_csv(enhanced_path)
        print(f"Enhanced dataset: {len(enhanced_data)} records")
    else:
        print("Enhanced dataset not found")
        enhanced_data = None
    
    return original_data, enhanced_data

def analyze_temporal_patterns(data):
    """Analyze temporal crime patterns"""
    print("\n=== TEMPORAL ANALYSIS ===")
    
    if 'time' in data.columns:
        # Convert time to hour for analysis
        data['hour'] = pd.to_datetime(data['time'], format='%H:%M').dt.hour
        
        # Crime by hour
        hourly_crimes = data.groupby('hour').size()
        print(f"Peak crime hour: {hourly_crimes.idxmax()}:00 ({hourly_crimes.max()} incidents)")
        print(f"Lowest crime hour: {hourly_crimes.idxmin()}:00 ({hourly_crimes.min()} incidents)")
        
        # Plot hourly distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        hourly_crimes.plot(kind='bar')
        plt.title('Crime Distribution by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=45)
        
    # Crime by month (if date column exists)
    if 'date' in data.columns:
        data['month'] = pd.to_datetime(data['date']).dt.month
        monthly_crimes = data.groupby('month').size()
        
        plt.subplot(1, 2, 2)
        monthly_crimes.plot(kind='bar', color='orange')
        plt.title('Crime Distribution by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Incidents')
        plt.xticks(rotation=0)
        
    plt.tight_layout()
    plt.savefig('output/crime_temporal_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_spatial_patterns(data):
    """Analyze spatial crime patterns"""
    print("\n=== SPATIAL ANALYSIS ===")
    
    # Crime by location type
    location_crimes = data['location_type'].value_counts()
    print("Crime distribution by location type:")
    for loc, count in location_crimes.items():
        print(f"  {loc}: {count} incidents ({count/len(data)*100:.1f}%)")
    
    # Create spatial visualization
    plt.figure(figsize=(15, 5))
    
    # Location type distribution
    plt.subplot(1, 3, 1)
    location_crimes.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Crime Distribution by Location Type')
    plt.ylabel('')
    
    # Geographic scatter plot
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(data['longitude'], data['latitude'], 
                         c=data['severity'], cmap='Reds', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Severity Level')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Crime Locations by Severity')
    
    # Crime type distribution
    plt.subplot(1, 3, 3)
    crime_types = data['crime_type'].value_counts()
    crime_types.plot(kind='barh')
    plt.title('Crime Type Distribution')
    plt.xlabel('Number of Incidents')
    
    plt.tight_layout()
    plt.savefig('output/crime_spatial_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_environmental_factors(data):
    """Analyze environmental factors (enhanced dataset only)"""
    print("\n=== ENVIRONMENTAL ANALYSIS ===")
    
    if 'environmental_factors' not in data.columns:
        print("Environmental factors not available in this dataset")
        return
    
    # Parse environmental factors
    all_factors = []
    for factors_str in data['environmental_factors'].dropna():
        factors = factors_str.split(';')
        all_factors.extend(factors)
    
    factor_counts = pd.Series(all_factors).value_counts()
    print("Most common environmental factors:")
    for factor, count in factor_counts.head(10).items():
        print(f"  {factor}: {count} incidents")
    
    # Correlation analysis
    numeric_cols = ['severity', 'socioeconomic_level', 'lighting_level', 
                   'surveillance_level', 'population_density']
    
    if all(col in data.columns for col in numeric_cols):
        correlation_matrix = data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 5))
        
        # Environmental factors frequency
        plt.subplot(1, 2, 1)
        factor_counts.head(10).plot(kind='barh')
        plt.title('Top Environmental Factors')
        plt.xlabel('Frequency')
        
        # Correlation heatmap
        plt.subplot(1, 2, 2)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Factor Correlations')
        
        plt.tight_layout()
        plt.savefig('output/environmental_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def analyze_crime_outcomes(data):
    """Analyze crime resolution and law enforcement response"""
    print("\n=== OUTCOME ANALYSIS ===")
    
    # Resolution rates
    if 'resolved' in data.columns:
        resolution_rate = data['resolved'].mean() * 100
        print(f"Overall resolution rate: {resolution_rate:.1f}%")
        
        # Resolution by crime type
        resolution_by_type = data.groupby('crime_type')['resolved'].mean() * 100
        print("Resolution rates by crime type:")
        for crime_type, rate in resolution_by_type.sort_values(ascending=False).items():
            print(f"  {crime_type}: {rate:.1f}%")
    
    # Arrest rates
    if 'arrest_made' in data.columns:
        arrest_rate = data['arrest_made'].mean() * 100
        print(f"\nOverall arrest rate: {arrest_rate:.1f}%")
        
        # Arrest by severity
        if 'severity' in data.columns:
            arrest_by_severity = data.groupby('severity')['arrest_made'].mean() * 100
            print("Arrest rates by severity level:")
            for severity, rate in arrest_by_severity.items():
                print(f"  Severity {severity}: {rate:.1f}%")
    
    # Response times
    if 'response_time_minutes' in data.columns:
        avg_response = data['response_time_minutes'].mean()
        print(f"\nAverage response time: {avg_response:.1f} minutes")
        
        # Response time by location type
        response_by_location = data.groupby('location_type')['response_time_minutes'].mean()
        print("Average response time by location:")
        for location, time in response_by_location.sort_values().items():
            print(f"  {location}: {time:.1f} minutes")

def generate_simulation_parameters(data):
    """Generate simulation parameters based on real data patterns"""
    print("\n=== SIMULATION PARAMETERS ===")
    
    # Crime type probabilities
    crime_probs = data['crime_type'].value_counts(normalize=True)
    print("Suggested crime type probabilities:")
    for crime_type, prob in crime_probs.items():
        print(f"  {crime_type}: {prob:.3f}")
    
    # Location type distribution
    location_probs = data['location_type'].value_counts(normalize=True)
    print("\nSuggested location type distribution:")
    for location, prob in location_probs.items():
        print(f"  {location}: {prob:.3f}")
    
    # Severity distribution
    severity_probs = data['severity'].value_counts(normalize=True).sort_index()
    print("\nSuggested severity distribution:")
    for severity, prob in severity_probs.items():
        print(f"  Level {severity}: {prob:.3f}")
    
    # Time-based patterns
    if 'time' in data.columns:
        # Convert to hours and get distribution
        hours = pd.to_datetime(data['time'], format='%H:%M').dt.hour
        peak_hours = hours.value_counts().head(6).index.tolist()
        print(f"\nPeak crime hours: {sorted(peak_hours)}")
    
    # Generate configuration snippet
    config_snippet = f"""
# Suggested simulation configuration based on data analysis
CRIME_TYPE_PROBABILITIES = {{
{chr(10).join(f'    "{crime_type}": {prob:.3f},' for crime_type, prob in crime_probs.items())}
}}

LOCATION_TYPE_DISTRIBUTION = {{
{chr(10).join(f'    "{location}": {prob:.3f},' for location, prob in location_probs.items())}
}}

SEVERITY_WEIGHTS = {severity_probs.to_dict()}
"""
    
    # Save configuration
    with open('output/suggested_config.py', 'w') as f:
        f.write(config_snippet)
    print("\nConfiguration suggestions saved to 'output/suggested_config.py'")

def main():
    """Main analysis function"""
    print("Criminal Behavior Modeling - Crime Data Analysis")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Load datasets
    original_data, enhanced_data = load_and_analyze_crime_data()
    
    # Use enhanced dataset if available, otherwise original
    data = enhanced_data if enhanced_data is not None else original_data
    
    if data is None:
        print("No crime data found. Please ensure datasets are in the data/ directory.")
        return
    
    print(f"\nAnalyzing {len(data)} crime incidents...")
    print(f"Crime types: {sorted(data['crime_type'].unique())}")
    print(f"Location types: {sorted(data['location_type'].unique())}")
    print(f"Severity range: {data['severity'].min()} - {data['severity'].max()}")
    
    # Perform analyses
    analyze_temporal_patterns(data)
    analyze_spatial_patterns(data)
    analyze_environmental_factors(data)
    analyze_crime_outcomes(data)
    generate_simulation_parameters(data)
    
    print("\nAnalysis complete! Check the output/ directory for visualizations and configuration files.")

if __name__ == "__main__":
    main()
