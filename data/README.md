# Crime Data Documentation

## Overview
This directory contains sample crime datasets for testing and research with the Criminal Behavior Modeling simulation. The data represents realistic crime incidents with comprehensive environmental and contextual factors.

## Dataset Files

### 1. `real_crime_data.csv`
**Format**: Original format with detailed victim demographics and weather conditions
**Records**: 50 crime incidents from January-March 2024
**Columns**:
- `incident_id`: Unique identifier for each crime
- `crime_type`: Type of crime (Theft, Assault, Burglary, Robbery, Vandalism, Drug_Offense)
- `latitude`, `longitude`: Geographic coordinates (NYC area)
- `date_time`: Combined date and time of incident
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of the week
- `month`: Month name
- `severity`: Crime severity (1-4 scale)
- `victim_age`: Age of victim
- `victim_gender`: Gender of victim
- `location_type`: Type of location (Commercial, Residential, Public)
- `arrest_made`: Binary flag for arrest (0/1)
- `patrol_present`: Binary flag for patrol presence (0/1)
- `lighting_quality`: Lighting conditions (Poor, Medium, Good)
- `population_density`: Density level (Low, Medium, High)
- `socioeconomic_index`: Index value (0.0-1.0)
- `surveillance_coverage`: Coverage level (None, Low, Medium, High)
- `weather_condition`: Weather at time of incident
- `temperature`: Temperature in Fahrenheit

### 2. `enhanced_crime_data.csv`
**Format**: Enhanced format optimized for simulation analysis
**Records**: 50 crime incidents from January-June 2024
**Columns**:
- `incident_id`: Unique identifier
- `date`: Date in YYYY-MM-DD format
- `time`: Time in HH:MM format
- `crime_type`: Type of crime
- `latitude`, `longitude`: Geographic coordinates
- `location_type`: Location category (commercial, residential, public, industrial)
- `severity`: Crime severity (1-5 scale)
- `environmental_factors`: Semicolon-separated environmental conditions
- `socioeconomic_level`: Numeric scale (1-5)
- `lighting_level`: Numeric scale (1-5)
- `surveillance_level`: Numeric scale (1-5)
- `population_density`: Numeric scale (1-10)
- `resolved`: Boolean flag for case resolution
- `arrest_made`: Boolean flag for arrest
- `response_time_minutes`: Police response time in minutes

## Usage in Simulation

### Data Loading
```python
import pandas as pd

# Load original format
crime_data = pd.read_csv('data/real_crime_data.csv')

# Load enhanced format
enhanced_data = pd.read_csv('data/enhanced_crime_data.csv')
```

### Integration with Analytics Module
The `analytics/predictive_modeling.py` module can use this data for:
- Training crime prediction models
- Validating environmental factor correlations
- Testing spatial-temporal pattern recognition
- Evaluating intervention strategies

### Environmental Factor Analysis
The enhanced dataset includes detailed environmental factors:
- **Lighting conditions**: poor_lighting, dark, daylight, moderate_lighting
- **Activity levels**: high_foot_traffic, crowded, busy_area, isolated
- **Security presence**: cameras_present, security_present, no_surveillance
- **Temporal contexts**: business_hours, late_night, evening_rush, dawn
- **Location specifics**: tourist_area, school_zone, bar_district, warehouse_district

### Simulation Configuration
Use this data to configure realistic simulation parameters:
```python
# Extract location distribution
location_types = enhanced_data['location_type'].value_counts()

# Analyze crime patterns by time
time_patterns = enhanced_data.groupby('time').size()

# Environmental factor frequency
env_factors = enhanced_data['environmental_factors'].str.split(';').explode().value_counts()
```

## Data Quality Notes

### Coordinate System
- All coordinates use WGS84 decimal degrees
- Covers Manhattan and surrounding NYC boroughs
- Coordinates are realistic for urban crime simulation

### Temporal Coverage
- **Original**: January-March 2024 (3 months)
- **Enhanced**: January-June 2024 (6 months)
- Includes varied temporal patterns across seasons

### Crime Type Distribution
- **Theft**: Most common (~30% of incidents)
- **Assault**: High severity incidents (~25%)
- **Burglary**: Residential focus (~20%)
- **Robbery**: High-impact crimes (~15%)
- **Vandalism**: Low severity (~5%)
- **Drug Offense**: Context-dependent (~5%)

### Environmental Realism
- Lighting correlates with time of day
- Population density reflects location type
- Surveillance levels match commercial vs residential areas
- Response times vary by location and severity

## Research Applications

### Criminological Theory Testing
- **Routine Activity Theory**: Offender-victim-guardian convergence patterns
- **Environmental Criminology**: Spatial crime concentration
- **Temporal Patterns**: Time-of-day and seasonal variations
- **Social Disorganization**: Neighborhood-level factors

### Policy Analysis
- Patrol deployment optimization
- Surveillance system placement
- Lighting improvement initiatives
- Community policing effectiveness

### Model Validation
- Compare simulation outputs to real patterns
- Test intervention scenario impacts
- Validate agent behavior models
- Assess prediction accuracy

## Data Extension

To extend these datasets:
1. Maintain consistent column formats
2. Ensure geographic coordinates stay within simulation bounds
3. Keep environmental factors realistic and consistent
4. Balance crime type distributions
5. Include seasonal and temporal variations

## Privacy and Ethics

This is **simulated data** created for research purposes:
- No real crime incidents or personal information
- Coordinates are realistic but not tied to actual events
- Designed to support academic and policy research
- Follows ethical guidelines for crime simulation research
