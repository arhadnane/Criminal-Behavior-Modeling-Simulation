"""
Configuration settings for Criminal Behavior Modeling Simulation
"""

# Simulation Parameters
SIMULATION_CONFIG = {
    'grid_size': {
        'width': 100,
        'height': 100
    },
    'agent_populations': {
        'criminals': 50,
        'victims': 200,
        'law_enforcement': 10
    },
    'simulation_steps': 1000,
    'random_seed': 42
}

# Environmental Factors
ENVIRONMENTAL_CONFIG = {
    'lighting_quality': 0.7,
    'surveillance_coverage': 0.5,
    'police_presence': 0.3,
    'socioeconomic_status': 0.6,
    'population_density': 0.8
}

# Crime Types
CRIME_TYPES = [
    'theft',
    'burglary', 
    'assault',
    'drug_related',
    'vandalism'
]

# Agent Behavior Parameters
AGENT_CONFIG = {
    'criminal': {
        'risk_tolerance_range': (0.1, 1.0),
        'motivation_range': (0.3, 1.0),
        'opportunity_threshold_range': (0.4, 0.8),
        'experience_scaling': 2.0
    },
    'victim': {
        'awareness_range': (0.3, 0.9),
        'vulnerability_range': (0.2, 0.8),
        'protective_behavior_range': (0.3, 0.7),
        'routine_predictability_range': (0.3, 0.9)
    },
    'law_enforcement': {
        'patrol_efficiency_range': (0.5, 0.9),
        'investigation_skill_range': (0.4, 0.8),
        'response_time_range': (0.6, 0.95),
        'detection_radius': 3
    }
}

# Machine Learning Parameters
ML_CONFIG = {
    'training': {
        'test_size': 0.3,
        'random_state': 42,
        'min_samples': 20
    },
    'models': {
        'crime_classifier': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'hotspot_predictor': {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42,
            'learning_rate': 0.1
        }
    },
    'prediction': {
        'hotspot_threshold': 0.7,
        'time_horizon': 100,
        'grid_sample_step': 5
    }
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    'color_schemes': {
        'crime_heat': 'Reds',
        'police_presence': 'Blues',
        'socioeconomic': 'RdYlGn', 
        'risk_assessment': 'YlOrRd'
    },
    'figure_size': {
        'default': (12, 10),
        'dashboard': (16, 12),
        'timeline': (14, 10)
    },
    'dpi': 300,
    'alpha': 0.7
}

# Data Export Settings
EXPORT_CONFIG = {
    'output_directory': 'output',
    'data_formats': ['csv', 'json'],
    'visualization_formats': ['png', 'html'],
    'compression': True
}

# Intervention Settings
INTERVENTION_CONFIG = {
    'types': [
        'lighting_improvement',
        'surveillance_installation', 
        'increased_patrol',
        'community_programs',
        'socioeconomic_improvement'
    ],
    'default_intensity': 0.3,
    'default_radius': 3,
    'effectiveness_thresholds': {
        'low': 0.1,
        'medium': 0.25,
        'high': 0.4
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_output': True,
    'console_output': True,
    'log_file': 'simulation.log'
}
