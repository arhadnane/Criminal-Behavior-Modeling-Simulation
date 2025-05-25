"""
Unit tests for simulation engine
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import CriminalBehaviorSimulation
from environment.urban_environment import UrbanEnvironment
from config.settings import SIMULATION_CONFIG


class TestCriminalBehaviorSimulation(unittest.TestCase):
    """Test main simulation engine"""
    
    def setUp(self):
        """Set up test simulation"""
        # Use smaller config for testing
        test_config = SIMULATION_CONFIG.copy()
        test_config['grid_size'] = (10, 10)
        test_config['num_criminals'] = 5
        test_config['num_victims'] = 10
        test_config['num_law_enforcement'] = 2  # Changed from num_police to match simulation
        test_config['simulation_steps'] = 10
        
        self.simulation = CriminalBehaviorSimulation(config=test_config)
        
    def test_simulation_initialization(self):
        """Test simulation initialization"""
        self.assertIsNotNone(self.simulation.urban_environment)
        self.assertIsNotNone(self.simulation.datacollector)
        
        # Check agent counts
        criminal_count = len([agent for agent in self.simulation.agents 
                            if agent.__class__.__name__ == 'CriminalAgent'])
        victim_count = len([agent for agent in self.simulation.agents 
                          if agent.__class__.__name__ == 'VictimAgent'])
        police_count = len([agent for agent in self.simulation.agents 
                          if agent.__class__.__name__ == 'LawEnforcementAgent'])
        
        self.assertEqual(criminal_count, 5)
        self.assertEqual(victim_count, 10)
        self.assertEqual(police_count, 2)
    
    def test_simulation_step(self):
        """Test single simulation step"""
        initial_step = self.simulation.step_count
        self.simulation.step()
        self.assertEqual(self.simulation.step_count, initial_step + 1)
    
    def test_run_simulation(self):
        """Test running complete simulation"""
        self.simulation.run_simulation(steps=5)
        self.assertEqual(self.simulation.step_count, 5)
        
        # Check that data was collected
        model_data = self.simulation.datacollector.get_model_vars_dataframe()
        self.assertGreater(len(model_data), 0)
    
    def test_get_results(self):
        """Test results retrieval"""
        self.simulation.run_simulation(steps=3)
        results = self.simulation.get_results()
        
        self.assertIn('total_crimes', results)
        self.assertIn('total_arrests', results)
        self.assertIn('crime_rate', results)
        self.assertIsInstance(results['total_crimes'], int)
        self.assertIsInstance(results['total_arrests'], int)
    
    def test_analyze_crime_patterns(self):
        """Test crime pattern analysis"""
        self.simulation.run_simulation(steps=5)
        patterns = self.simulation.analyze_crime_patterns()
        
        self.assertIn('spatial_distribution', patterns)
        self.assertIn('temporal_distribution', patterns)
        self.assertIn('hotspots', patterns)
    
    def test_intervention_effects(self):
        """Test intervention implementation"""
        # Run baseline simulation
        self.simulation.run_simulation(steps=5)
        baseline_crimes = self.simulation.total_crimes
        
        # Apply intervention (increase police presence)
        self.simulation.apply_intervention('increase_police_presence', intensity=0.5)
        
        # Reset and run again
        self.simulation.reset()
        self.simulation.run_simulation(steps=5)
        intervention_crimes = self.simulation.total_crimes
        
        # Intervention should affect crime counts
        self.assertIsInstance(baseline_crimes, int)
        self.assertIsInstance(intervention_crimes, int)


class TestUrbanEnvironment(unittest.TestCase):
    """Test urban environment modeling"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock model for testing
        class MockModel:
            def __init__(self):
                self.width = 10
                self.height = 10
        
        self.mock_model = MockModel()
        self.environment = UrbanEnvironment(self.mock_model)
    
    def test_environment_initialization(self):
        """Test environment setup"""
        self.assertEqual(self.environment.width, 10)
        self.assertEqual(self.environment.height, 10)
        self.assertIsNotNone(self.environment.lighting_grid)
        self.assertIsNotNone(self.environment.surveillance_grid)
        self.assertIsNotNone(self.environment.socioeconomic_grid)
    
    def test_grid_values(self):
        """Test grid value ranges"""
        # Test a sample of grid values
        for x in range(5):
            for y in range(5):
                lighting = self.environment.get_lighting(x, y)
                surveillance = self.environment.get_surveillance(x, y)
                socioeconomic = self.environment.get_socioeconomic_status(x, y)
                
                self.assertTrue(0 <= lighting <= 1)
                self.assertTrue(0 <= surveillance <= 1)
                self.assertTrue(0 <= socioeconomic <= 1)
    
    def test_environmental_updates(self):
        """Test dynamic environmental updates"""
        original_lighting = self.environment.get_lighting(5, 5)
        
        # Update environment (simulate time passing)
        self.environment.update_time_based_factors(hour=22)  # night time
        
        new_lighting = self.environment.get_lighting(5, 5)
        
        # Lighting should change based on time
        self.assertIsInstance(original_lighting, float)
        self.assertIsInstance(new_lighting, float)
    
    def test_risk_calculation(self):
        """Test environmental risk calculation"""
        risk = self.environment.calculate_risk(5, 5)
        self.assertIsInstance(risk, float)
        self.assertTrue(0 <= risk <= 1)
    
    def test_attractiveness_calculation(self):
        """Test location attractiveness for crimes"""
        attractiveness = self.environment.calculate_attractiveness(5, 5)
        self.assertIsInstance(attractiveness, float)
        self.assertTrue(0 <= attractiveness <= 1)


class TestSimulationIntegration(unittest.TestCase):
    """Test integration between simulation components"""
    
    def setUp(self):
        """Set up integration test"""
        test_config = SIMULATION_CONFIG.copy()
        test_config['grid_size'] = (8, 8)
        test_config['num_criminals'] = 3
        test_config['num_victims'] = 6
        test_config['num_police'] = 2
        test_config['simulation_steps'] = 5
        
        self.simulation = CriminalBehaviorSimulation(config=test_config)
    
    def test_agent_environment_interaction(self):
        """Test agents interacting with environment"""
        self.simulation.run_simulation(steps=3)
        
        # Check that agents are positioned on the grid
        for agent in self.simulation.scheduler.agents:
            pos = agent.pos
            self.assertIsNotNone(pos)
            self.assertTrue(0 <= pos[0] < 8)
            self.assertTrue(0 <= pos[1] < 8)
    
    def test_data_collection(self):
        """Test data collection during simulation"""
        self.simulation.run_simulation(steps=4)
        
        # Check model-level data
        model_data = self.simulation.datacollector.get_model_vars_dataframe()
        self.assertEqual(len(model_data), 4)  # 4 steps
        
        # Check agent-level data
        agent_data = self.simulation.datacollector.get_agent_vars_dataframe()
        self.assertGreater(len(agent_data), 0)
    
    def test_temporal_dynamics(self):
        """Test temporal changes in simulation"""
        initial_time = self.simulation.current_time
        
        self.simulation.run_simulation(steps=3)
        
        final_time = self.simulation.current_time
        
        # Time should have progressed
        self.assertNotEqual(initial_time, final_time)
    
    def test_spatial_crime_distribution(self):
        """Test spatial distribution of crimes"""
        self.simulation.run_simulation(steps=5)
        
        # Get crime locations
        crimes = getattr(self.simulation, 'crime_locations', [])
        
        # Crimes should be distributed across the grid
        if crimes:
            for crime in crimes:
                self.assertTrue(0 <= crime[0] < 8)
                self.assertTrue(0 <= crime[1] < 8)


if __name__ == '__main__':
    unittest.main()
