"""
Unit tests for analytics and predictive modeling
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.predictive_model import PredictiveModel


class TestPredictiveModel(unittest.TestCase):
    """Test predictive modeling functionality"""
    
    def setUp(self):
        """Set up test data and model"""
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                self.width = 10
                self.height = 10
                
        self.mock_model = MockModel()
        self.model = PredictiveModel(self.mock_model)
        
        # Create sample training data
        np.random.seed(42)  # For reproducible tests
        self.n_samples = 100
        
        # Features: [hour, day_of_week, lighting, surveillance, socioeconomic, population_density]
        self.features = pd.DataFrame({
            'hour': np.random.randint(0, 24, self.n_samples),
            'day_of_week': np.random.randint(0, 7, self.n_samples),
            'lighting': np.random.random(self.n_samples),
            'surveillance': np.random.random(self.n_samples),
            'socioeconomic': np.random.random(self.n_samples),
            'population_density': np.random.random(self.n_samples),
            'x': np.random.randint(0, 10, self.n_samples),
            'y': np.random.randint(0, 10, self.n_samples)
        })
        
        # Target: binary crime occurrence
        self.targets = np.random.randint(0, 2, self.n_samples)
        
        # Crime count data for hotspot prediction
        self.crime_counts = np.random.poisson(2, self.n_samples)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.crime_model)
        self.assertIsNotNone(self.model.hotspot_model)
        self.assertFalse(self.model.is_trained)
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        engineered_features = self.model.engineer_features(self.features)
        
        # Should have additional features
        self.assertGreater(len(engineered_features.columns), len(self.features.columns))
        
        # Check for engineered features
        expected_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'risk_score']
        for feature in expected_features:
            self.assertIn(feature, engineered_features.columns)
    
    def test_model_training(self):
        """Test model training"""
        # Train the model
        metrics = self.model.train(self.features, self.targets)
        
        # Model should be trained
        self.assertTrue(self.model.is_trained)
        
        # Should return metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Metrics should be valid
        for metric_name, metric_value in metrics.items():
            self.assertIsInstance(metric_value, (int, float))
            self.assertTrue(0 <= metric_value <= 1)
    
    def test_crime_prediction(self):
        """Test crime prediction"""
        # Train model first
        self.model.train(self.features, self.targets)
        
        # Make predictions
        test_features = self.features.iloc[:10]  # First 10 samples
        predictions = self.model.predict_crime_probability(test_features)
        
        # Should return probabilities
        self.assertEqual(len(predictions), 10)
        for prob in predictions:
            self.assertTrue(0 <= prob <= 1)
    
    def test_hotspot_prediction(self):
        """Test crime hotspot prediction"""
        # Train hotspot model
        self.model.train_hotspot_model(self.features, self.crime_counts)
        
        # Predict hotspots
        grid_features = self.features.iloc[:25]  # 5x5 grid
        hotspots = self.model.predict_hotspots(grid_features)
        
        # Should return hotspot scores
        self.assertEqual(len(hotspots), 25)
        for score in hotspots:
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0)
    
    def test_risk_score_calculation(self):
        """Test risk score calculation"""
        # Test individual location
        risk = self.model.calculate_risk_score(
            lighting=0.5, surveillance=0.3, socioeconomic=0.7, 
            population_density=0.8, police_presence=0.2
        )
        
        self.assertIsInstance(risk, float)
        self.assertTrue(0 <= risk <= 1)
        
        # Test multiple locations
        risks = self.model.calculate_risk_scores(self.features.iloc[:5])
        self.assertEqual(len(risks), 5)
        for risk in risks:
            self.assertTrue(0 <= risk <= 1)
    
    def test_model_validation(self):
        """Test model validation"""
        # Perform cross-validation
        cv_scores = self.model.validate_model(self.features, self.targets, cv_folds=3)
        
        # Should return validation scores
        self.assertIn('accuracy', cv_scores)
        self.assertIn('precision', cv_scores)
        self.assertIn('recall', cv_scores)
        self.assertIn('f1_score', cv_scores)
        
        # Each should be a list of scores
        for metric_name, scores in cv_scores.items():
            self.assertIsInstance(scores, list)
            self.assertEqual(len(scores), 3)  # 3-fold CV
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train model first
        self.model.train(self.features, self.targets)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Should return importance scores
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # All importance values should be non-negative
        for feature_name, importance_score in importance.items():
            self.assertIsInstance(importance_score, (int, float))
            self.assertGreaterEqual(importance_score, 0)
    
    def test_prediction_explanation(self):
        """Test prediction explanation"""
        # Train model first
        self.model.train(self.features, self.targets)
        
        # Get explanation for a prediction
        test_sample = self.features.iloc[0:1]
        explanation = self.model.explain_prediction(test_sample)
        
        # Should return explanation
        self.assertIsInstance(explanation, dict)
        self.assertIn('prediction', explanation)
        self.assertIn('confidence', explanation)
        self.assertIn('feature_contributions', explanation)
    
    def test_temporal_patterns(self):
        """Test temporal pattern analysis"""
        # Create temporal crime data
        temporal_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'crime_count': np.random.poisson(3, 100)
        })
        
        patterns = self.model.analyze_temporal_patterns(temporal_data)
        
        # Should return pattern analysis
        self.assertIn('daily_pattern', patterns)
        self.assertIn('weekly_pattern', patterns)
        self.assertIn('trend', patterns)
    
    def test_spatial_patterns(self):
        """Test spatial pattern analysis"""
        # Create spatial crime data
        spatial_data = pd.DataFrame({
            'x': np.random.randint(0, 10, 50),
            'y': np.random.randint(0, 10, 50),
            'crime_count': np.random.poisson(2, 50)
        })
        
        patterns = self.model.analyze_spatial_patterns(spatial_data, grid_size=(10, 10))
        
        # Should return spatial analysis
        self.assertIn('hotspot_locations', patterns)
        self.assertIn('spatial_autocorrelation', patterns)
        self.assertIn('clustering_metrics', patterns)


class TestAnalyticsIntegration(unittest.TestCase):
    """Test integration of analytics with simulation"""
    
    def setUp(self):
        """Set up integration test"""
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                self.width = 10
                self.height = 10
                
        self.mock_model = MockModel()
        self.model = PredictiveModel(self.mock_model)
        
        # Create realistic simulation data
        np.random.seed(123)
        self.simulation_data = self.create_simulation_data()
    
    def create_simulation_data(self):
        """Create realistic simulation data for testing"""
        n_steps = 50
        grid_size = 10
        
        data = []
        for step in range(n_steps):
            for x in range(grid_size):
                for y in range(grid_size):
                    # Simulate environmental conditions
                    hour = (step % 24)
                    day = step // 24
                    
                    # Environmental factors (some locations are riskier)
                    base_risk = 0.1 if (x < 3 and y < 3) else 0.05  # High-risk area
                    lighting = 0.9 if hour < 18 else 0.3  # Lower at night
                    surveillance = 0.7 if (x == 5 and y == 5) else 0.3  # Central surveillance
                    
                    # Crime occurrence (probabilistic)
                    crime_prob = base_risk * (1 - lighting * 0.5) * (1 - surveillance * 0.3)
                    crime_occurred = np.random.random() < crime_prob
                    
                    data.append({
                        'step': step,
                        'x': x,
                        'y': y,
                        'hour': hour,
                        'day_of_week': day % 7,
                        'lighting': lighting,
                        'surveillance': surveillance,
                        'socioeconomic': 0.5 + 0.3 * (x + y) / (2 * grid_size),
                        'population_density': 0.8 - 0.3 * abs(x - 5) / 5,
                        'police_presence': 0.6 if surveillance > 0.5 else 0.2,
                        'crime_occurred': crime_occurred
                    })
        
        return pd.DataFrame(data)
    
    def test_simulation_data_training(self):
        """Test training on simulation-generated data"""
        # Prepare features and targets
        feature_cols = ['hour', 'day_of_week', 'lighting', 'surveillance', 
                       'socioeconomic', 'population_density', 'police_presence', 'x', 'y']
        features = self.simulation_data[feature_cols]
        targets = self.simulation_data['crime_occurred'].astype(int)
        
        # Train model
        metrics = self.model.train(features, targets)
        
        # Should achieve reasonable performance
        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertTrue(self.model.is_trained)
    
    def test_hotspot_identification(self):
        """Test hotspot identification from simulation data"""
        # Aggregate crime counts by location
        crime_counts = self.simulation_data.groupby(['x', 'y'])['crime_occurred'].sum().reset_index()
        crime_counts.columns = ['x', 'y', 'crime_count']
        
        # Add environmental features (using mean values)
        env_features = self.simulation_data.groupby(['x', 'y'])[
            ['lighting', 'surveillance', 'socioeconomic', 'population_density', 'police_presence']
        ].mean().reset_index()
        
        # Merge data
        hotspot_data = crime_counts.merge(env_features, on=['x', 'y'])
        
        # Train hotspot model
        feature_cols = ['lighting', 'surveillance', 'socioeconomic', 
                       'population_density', 'police_presence', 'x', 'y']
        self.model.train_hotspot_model(hotspot_data[feature_cols], hotspot_data['crime_count'])
        
        # Predict hotspots
        hotspots = self.model.predict_hotspots(hotspot_data[feature_cols])
        
        # Verify predictions
        self.assertEqual(len(hotspots), len(hotspot_data))
        
        # High-crime areas should have higher hotspot scores
        high_crime_mask = hotspot_data['crime_count'] > hotspot_data['crime_count'].median()
        high_crime_scores = np.array(hotspots)[high_crime_mask]
        low_crime_scores = np.array(hotspots)[~high_crime_mask]
        
        if len(high_crime_scores) > 0 and len(low_crime_scores) > 0:
            self.assertGreater(np.mean(high_crime_scores), np.mean(low_crime_scores))
    
    def test_prediction_accuracy_validation(self):
        """Test prediction accuracy on simulation data"""
        # Split data into train/test
        train_size = int(0.8 * len(self.simulation_data))
        train_data = self.simulation_data.iloc[:train_size]
        test_data = self.simulation_data.iloc[train_size:]
        
        # Prepare features and targets
        feature_cols = ['hour', 'day_of_week', 'lighting', 'surveillance', 
                       'socioeconomic', 'population_density', 'police_presence', 'x', 'y']
        
        # Train on training data
        train_features = train_data[feature_cols]
        train_targets = train_data['crime_occurred'].astype(int)
        self.model.train(train_features, train_targets)
        
        # Test on test data
        test_features = test_data[feature_cols]
        test_targets = test_data['crime_occurred'].astype(int)
        predictions = self.model.predict_crime_probability(test_features)
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = (binary_predictions == test_targets).mean()
        
        # Should achieve reasonable accuracy on this controlled data
        self.assertGreater(accuracy, 0.5)


if __name__ == '__main__':
    unittest.main()
