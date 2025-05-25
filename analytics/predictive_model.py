"""
Predictive Analytics for Criminal Behavior Modeling

This module implements machine learning models to predict criminal activity
patterns, identify hotspots, and assess intervention effectiveness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class PredictiveModel:
    """
    Machine learning models for criminal behavior prediction
    """
    
    def __init__(self, model):
        self.model = model
        self.crime_classifier = None
        self.hotspot_predictor = None
        self.intervention_analyzer = None
        self.scaler = StandardScaler()
        
        # Feature importance tracking
        self.feature_importance = {}
        self.model_performance = {}
        
        # Historical data storage
        self.historical_features = []
        self.historical_targets = []
        
    def extract_features(self, step_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for machine learning from simulation data"""
        features = []
        
        # Environmental features
        env_data = step_data.get('environmental_factors', {})
        features.extend([
            env_data.get('lighting_quality', 0.5),
            env_data.get('surveillance_coverage', 0.5),
            env_data.get('police_presence', 0.3),
            env_data.get('socioeconomic_status', 0.6),
            env_data.get('population_density', 0.8)
        ])
        
        # Temporal features
        time_of_day = step_data.get('time_of_day', 12)
        day_of_week = step_data.get('day_of_week', 0)
        
        # Convert time to cyclical features
        features.extend([
            np.sin(2 * np.pi * time_of_day / 24),
            np.cos(2 * np.pi * time_of_day / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        # Agent-based features
        agent_data = step_data.get('agent_data', {})
        if agent_data:
            # Criminal agent features
            criminal_agents = [agent for agent in self.model.schedule.agents 
                             if hasattr(agent, 'agent_type') and agent.agent_type == 'criminal']
            
            if criminal_agents:
                avg_motivation = np.mean([agent.motivation_level for agent in criminal_agents])
                avg_heat_level = np.mean([agent.heat_level for agent in criminal_agents])
                avg_experience = np.mean([agent.criminal_experience for agent in criminal_agents])
                total_crimes = sum([agent.crimes_committed for agent in criminal_agents])
            else:
                avg_motivation = avg_heat_level = avg_experience = total_crimes = 0
            
            features.extend([avg_motivation, avg_heat_level, avg_experience, total_crimes])
            
            # Law enforcement features
            le_agents = [agent for agent in self.model.schedule.agents 
                        if hasattr(agent, 'agent_type') and agent.agent_type == 'law_enforcement']
            
            if le_agents:
                total_arrests = sum([agent.arrests_made for agent in le_agents])
                avg_patrol_efficiency = np.mean([agent.patrol_efficiency for agent in le_agents])
            else:
                total_arrests = avg_patrol_efficiency = 0
            
            features.extend([total_arrests, avg_patrol_efficiency])
        else:
            # Default values when agent data not available
            features.extend([0.5, 0.0, 2.0, 0, 0, 0.7])
        
        # Historical crime features (last N steps)
        recent_crimes = step_data.get('recent_crime_count', 0)
        crime_trend = step_data.get('crime_trend', 0.0)
        features.extend([recent_crimes, crime_trend])
        
        return np.array(features)
    
    def collect_training_data(self):
        """Collect training data from simulation history"""
        if not hasattr(self.model, 'crime_incidents') or len(self.model.crime_incidents) < 10:
            return False
        
        # Create training samples from simulation data
        step_interval = 10  # Sample every 10 steps
        
        for step in range(0, self.model.step_count, step_interval):
            # Get crime incidents for this time window
            step_crimes = [crime for crime in self.model.crime_incidents 
                          if step <= crime['step'] < step + step_interval]
            
            # Create features for this time step
            step_data = {
                'environmental_factors': self.model.environmental_factors,
                'time_of_day': (step // 10) % 24,  # Simulate time progression
                'day_of_week': (step // 240) % 7,  # Simulate day progression
                'recent_crime_count': len([crime for crime in self.model.crime_incidents 
                                         if step - 50 <= crime['step'] < step]),
                'crime_trend': self._calculate_crime_trend(step)
            }
            
            features = self.extract_features(step_data)
            
            # Target: crime occurrence (binary) and crime count
            crime_occurred = len(step_crimes) > 0
            crime_count = len(step_crimes)
            
            self.historical_features.append(features)
            self.historical_targets.append({'occurred': crime_occurred, 'count': crime_count})
        
        return len(self.historical_features) > 20  # Need minimum samples
    
    def _calculate_crime_trend(self, current_step: int) -> float:
        """Calculate crime trend over recent period"""
        if current_step < 100:
            return 0.0
        
        recent_period = [crime for crime in self.model.crime_incidents 
                        if current_step - 100 <= crime['step'] < current_step]
        earlier_period = [crime for crime in self.model.crime_incidents 
                         if current_step - 200 <= crime['step'] < current_step - 100]
        
        recent_count = len(recent_period)
        earlier_count = len(earlier_period)
        
        if earlier_count == 0:
            return 0.0
        
        return (recent_count - earlier_count) / earlier_count
    
    def train_crime_classifier(self) -> bool:
        """Train model to predict crime occurrence"""
        if not self.collect_training_data():
            return False
        
        X = np.array(self.historical_features)
        y = np.array([target['occurred'] for target in self.historical_targets])
        
        if len(X) < 20 or len(np.unique(y)) < 2:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.crime_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.crime_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.crime_classifier.score(X_train_scaled, y_train)
        test_score = self.crime_classifier.score(X_test_scaled, y_test)
        
        self.model_performance['crime_classifier'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        # Feature importance
        feature_names = [
            'lighting', 'surveillance', 'police_presence', 'socioeconomic', 'population_density',
            'time_sin', 'time_cos', 'day_sin', 'day_cos',
            'avg_motivation', 'avg_heat', 'avg_experience', 'total_crimes',
            'total_arrests', 'avg_patrol_efficiency', 'recent_crimes', 'crime_trend'
        ]
        
        importance_dict = dict(zip(feature_names, self.crime_classifier.feature_importances_))
        self.feature_importance['crime_classifier'] = importance_dict
        
        return True
    
    def train_hotspot_predictor(self) -> bool:
        """Train model to predict crime hotspots"""
        if not self.collect_training_data():
            return False
        
        X = np.array(self.historical_features)
        y = np.array([target['count'] for target in self.historical_targets])
        
        if len(X) < 20:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regression model
        self.hotspot_predictor = GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=6
        )
        self.hotspot_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.hotspot_predictor.predict(X_train_scaled)
        test_pred = self.hotspot_predictor.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        self.model_performance['hotspot_predictor'] = {
            'train_mse': train_mse,
            'test_mse': test_mse
        }
        
        return True
    
    def predict_crime_probability(self, location: Tuple[int, int], 
                                 time_step: int) -> float:
        """Predict probability of crime at location and time"""
        if self.crime_classifier is None:
            if not self.train_crime_classifier():
                return 0.5  # Default probability
        
        # Get environmental factors for location
        env_factors = self.model.urban_environment.get_environmental_factors(
            location[0], location[1]
        )
        
        # Create feature vector
        step_data = {
            'environmental_factors': env_factors,
            'time_of_day': (time_step // 10) % 24,
            'day_of_week': (time_step // 240) % 7,
            'recent_crime_count': len([crime for crime in self.model.crime_incidents 
                                     if time_step - 50 <= crime['step'] < time_step]),
            'crime_trend': self._calculate_crime_trend(time_step)
        }
        
        features = self.extract_features(step_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probability
        prob = self.crime_classifier.predict_proba(features_scaled)[0][1]
        return prob
    
    def predict_crime_count(self, location: Tuple[int, int], 
                          time_step: int) -> float:
        """Predict expected number of crimes at location and time"""
        if self.hotspot_predictor is None:
            if not self.train_hotspot_predictor():
                return 0.0
        
        # Get environmental factors for location
        env_factors = self.model.urban_environment.get_environmental_factors(
            location[0], location[1]
        )
        
        # Create feature vector
        step_data = {
            'environmental_factors': env_factors,
            'time_of_day': (time_step // 10) % 24,
            'day_of_week': (time_step // 240) % 7,
            'recent_crime_count': len([crime for crime in self.model.crime_incidents 
                                     if time_step - 50 <= crime['step'] < time_step]),
            'crime_trend': self._calculate_crime_trend(time_step)
        }
        
        features = self.extract_features(step_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        count = max(0, self.hotspot_predictor.predict(features_scaled)[0])
        return count
    
    def identify_crime_hotspots(self, time_step: int, 
                              threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Identify locations with high crime probability"""
        hotspots = []
        
        # Sample grid locations
        step_size = 5
        for x in range(0, self.model.width, step_size):
            for y in range(0, self.model.height, step_size):
                prob = self.predict_crime_probability((x, y), time_step)
                if prob > threshold:
                    hotspots.append((x, y))
        
        return hotspots
    
    def analyze_intervention_effectiveness(self, intervention_data: List[Dict]) -> Dict[str, float]:
        """Analyze effectiveness of different interventions"""
        if not intervention_data:
            return {}
        
        effectiveness = {}
        
        for intervention in intervention_data:
            intervention_type = intervention['type']
            before_crimes = intervention['crimes_before']
            after_crimes = intervention['crimes_after']
            
            if before_crimes > 0:
                reduction_rate = (before_crimes - after_crimes) / before_crimes
                effectiveness[intervention_type] = effectiveness.get(intervention_type, [])
                effectiveness[intervention_type].append(reduction_rate)
        
        # Calculate average effectiveness
        avg_effectiveness = {}
        for intervention_type, reductions in effectiveness.items():
            avg_effectiveness[intervention_type] = np.mean(reductions)
        
        return avg_effectiveness
    
    def generate_predictive_report(self, time_horizon: int = 100) -> Dict[str, Any]:
        """Generate comprehensive predictive analytics report"""
        current_step = self.model.step_count
        
        # Predict future hotspots
        future_hotspots = []
        for future_step in range(current_step, current_step + time_horizon, 10):
            hotspots = self.identify_crime_hotspots(future_step)
            future_hotspots.extend(hotspots)
        
        # Most frequent hotspot locations
        hotspot_frequency = {}
        for location in future_hotspots:
            area_key = f"{location[0]//5}_{location[1]//5}"
            hotspot_frequency[area_key] = hotspot_frequency.get(area_key, 0) + 1
        
        top_hotspots = sorted(hotspot_frequency.items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        # Crime trend analysis
        recent_crime_trend = self._calculate_crime_trend(current_step)
        
        # Risk assessment by area type
        area_risks = self._assess_area_type_risks(current_step)
        
        # Recommended interventions
        recommendations = self._generate_intervention_recommendations(top_hotspots)
        
        report = {
            'prediction_timestamp': current_step,
            'time_horizon': time_horizon,
            'top_predicted_hotspots': top_hotspots,
            'crime_trend': recent_crime_trend,
            'area_type_risks': area_risks,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'intervention_recommendations': recommendations
        }
        
        return report
    
    def _assess_area_type_risks(self, time_step: int) -> Dict[str, float]:
        """Assess crime risk by different area types"""
        area_risks = {}
        
        # Sample locations of different area types
        building_types = self.model.urban_environment.building_types
        
        for area_type, locations in building_types.items():
            if locations:
                sample_locations = locations[:min(10, len(locations))]  # Sample up to 10
                risks = []
                
                for location in sample_locations:
                    risk = self.predict_crime_probability(location, time_step)
                    risks.append(risk)
                
                area_risks[area_type] = np.mean(risks)
        
        return area_risks
    
    def _generate_intervention_recommendations(self, top_hotspots: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """Generate intervention recommendations based on predictions"""
        recommendations = []
        
        for area_key, frequency in top_hotspots[:5]:  # Top 5 hotspots
            try:
                area_x, area_y = map(int, area_key.split('_'))
                center_x, center_y = area_x * 5 + 2, area_y * 5 + 2
                
                # Get environmental factors for the area
                env_factors = self.model.urban_environment.get_environmental_factors(
                    center_x, center_y
                )
                
                # Determine best intervention based on weakest factors
                interventions = []
                
                if env_factors['lighting_quality'] < 0.5:
                    interventions.append({
                        'type': 'lighting_improvement',
                        'priority': 'high',
                        'expected_impact': 0.3
                    })
                
                if env_factors['surveillance_coverage'] < 0.4:
                    interventions.append({
                        'type': 'surveillance_installation',
                        'priority': 'high',
                        'expected_impact': 0.4
                    })
                
                if env_factors['police_presence'] < 0.3:
                    interventions.append({
                        'type': 'increased_patrol',
                        'priority': 'medium',
                        'expected_impact': 0.25
                    })
                
                if env_factors['socioeconomic_status'] < 0.3:
                    interventions.append({
                        'type': 'community_programs',
                        'priority': 'low',
                        'expected_impact': 0.15
                    })
                
                recommendation = {
                    'location': (center_x, center_y),
                    'hotspot_frequency': frequency,
                    'interventions': interventions,
                    'total_expected_impact': sum([i['expected_impact'] for i in interventions])
                }
                
                recommendations.append(recommendation)
                
            except (ValueError, IndexError):
                continue
        
        return recommendations
    
    def export_model_data(self) -> Dict[str, Any]:
        """Export model data for external analysis"""
        return {
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'training_samples': len(self.historical_features),
            'models_trained': {
                'crime_classifier': self.crime_classifier is not None,
                'hotspot_predictor': self.hotspot_predictor is not None
            }
        }
