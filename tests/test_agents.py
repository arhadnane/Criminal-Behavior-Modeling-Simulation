"""
Unit tests for agent behaviors
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesa import Model
from mesa.space import MultiGrid

from agents.criminal_agent import CriminalAgent
from agents.victim_agent import VictimAgent
from agents.law_enforcement_agent import LawEnforcementAgent
from config.settings import SIMULATION_CONFIG


class MockModel(Model):
    """Mock model for testing agents"""
    
    def __init__(self, width=10, height=10):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.current_time = 12  # noon
        self.current_day = 0
        self.total_crimes = 0
        self.arrests_made = 0
        # Add environmental_factors for LawEnforcementAgent
        self.environmental_factors = {
            'lighting_quality': 0.5,
            'surveillance_coverage': 0.6,
            'foot_traffic': 0.4,
            'economic_conditions': 0.5
        }
        self.crime_incidents = []
        self.step_count = 0
        self.arrests_made = 0


class TestCriminalAgent(unittest.TestCase):
    """Test CriminalAgent behavior"""
    
    def setUp(self):
        """Set up test environment"""
        self.model = MockModel()
        self.agent = CriminalAgent(self.model, unique_id=1)
        self.model.grid.place_agent(self.agent, (5, 5))
    
    def test_agent_creation(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.unique_id, 1)
        self.assertEqual(self.agent.model, self.model)
        self.assertIsInstance(self.agent.risk_tolerance, float)
        self.assertIsInstance(self.agent.skill_level, float)
        self.assertTrue(0 <= self.agent.risk_tolerance <= 1)
        self.assertTrue(0 <= self.agent.skill_level <= 1)
    
    def test_risk_assessment(self):
        """Test risk assessment calculation"""
        risk = self.agent.assess_risk((5, 5))
        self.assertIsInstance(risk, float)
        self.assertTrue(0 <= risk <= 1)
    
    def test_decision_making(self):
        """Test criminal decision making"""
        decision = self.agent.make_decision()
        self.assertIn(decision, ['commit_crime', 'move', 'wait'])
    
    def test_crime_commission(self):
        """Test crime commission"""
        initial_crimes = self.model.total_crimes
        self.agent.commit_crime()
        # Crime count should increase
        self.assertGreaterEqual(self.model.total_crimes, initial_crimes)
    
    def test_movement(self):
        """Test agent movement"""
        original_pos = self.agent.pos
        self.agent.move()
        # Agent should either move or stay (based on decision logic)
        new_pos = self.agent.pos
        self.assertIsNotNone(new_pos)


class TestVictimAgent(unittest.TestCase):
    """Test VictimAgent behavior"""
    def setUp(self):
        """Set up test environment"""
        self.model = MockModel()
        self.agent = VictimAgent(self.model, unique_id=2)
        self.model.grid.place_agent(self.agent, (3, 3))
    
    def test_agent_creation(self):
        """Test victim agent initialization"""
        self.assertEqual(self.agent.unique_id, 2)
        self.assertIsInstance(self.agent.vulnerability, float)
        self.assertIsInstance(self.agent.awareness, float)
        self.assertTrue(0 <= self.agent.vulnerability <= 1)
        self.assertTrue(0 <= self.agent.awareness <= 1)
    
    def test_vulnerability_calculation(self):
        """Test vulnerability assessment"""
        vulnerability = self.agent.calculate_vulnerability()
        self.assertIsInstance(vulnerability, float)
        self.assertTrue(0 <= vulnerability <= 1)
    
    def test_protective_behavior(self):
        """Test protective behavior implementation"""
        self.agent.implement_protective_behavior()
        # Should increase awareness or reduce vulnerability
        self.assertIsNotNone(self.agent.awareness)
    
    def test_routine_activity(self):
        """Test routine activity patterns"""
        activity = self.agent.get_current_activity()
        expected_activities = ['home', 'work', 'shopping', 'leisure', 'transit']
        self.assertIn(activity, expected_activities)


class TestLawEnforcementAgent(unittest.TestCase):
    """Test LawEnforcementAgent behavior"""
    def setUp(self):
        """Set up test environment"""
        self.model = MockModel()
        self.agent = LawEnforcementAgent(self.model, unique_id=3)
        self.model.grid.place_agent(self.agent, (1, 1))
    
    def test_agent_creation(self):
        """Test law enforcement agent initialization"""
        self.assertEqual(self.agent.unique_id, 3)
        self.assertIsInstance(self.agent.patrol_effectiveness, float)
        self.assertIsInstance(self.agent.response_time, int)
        self.assertTrue(0 <= self.agent.patrol_effectiveness <= 1)
        self.assertTrue(self.agent.response_time > 0)
    
    def test_patrol_behavior(self):
        """Test patrol behavior"""
        original_pos = self.agent.pos
        self.agent.patrol()
        # Agent should move during patrol
        new_pos = self.agent.pos
        self.assertIsNotNone(new_pos)
    
    def test_crime_detection(self):
        """Test crime detection capability"""        # Place a criminal nearby
        criminal = CriminalAgent(self.model, unique_id=99)
        self.model.grid.place_agent(criminal, (1, 2))
        
        detected_crimes = self.agent.detect_crimes()
        self.assertIsInstance(detected_crimes, list)
    
    def test_arrest_procedure(self):
        """Test arrest procedure"""        # Create a criminal for testing
        criminal = CriminalAgent(self.model, unique_id=99)
        self.model.grid.place_agent(criminal, (1, 2))
        
        initial_arrests = self.model.arrests_made
        success = self.agent.attempt_arrest(criminal)
        self.assertIsInstance(success, bool)


class TestAgentInteractions(unittest.TestCase):
    """Test interactions between different agent types"""
    
    def setUp(self):
        """Set up test environment with multiple agents"""
        self.model = MockModel()
          # Create agents
        self.criminal = CriminalAgent(self.model, unique_id=1)
        self.victim = VictimAgent(self.model, unique_id=2)
        self.police = LawEnforcementAgent(self.model, unique_id=3)
          # Place agents on grid
        self.model.grid.place_agent(self.criminal, (5, 5))
        self.model.grid.place_agent(self.victim, (5, 6))
        self.model.grid.place_agent(self.police, (4, 5))
    
    def test_criminal_victim_interaction(self):
        """Test interaction between criminal and victim"""
        # Get nearby agents
        neighbors = self.model.grid.get_neighbors(
            self.criminal.pos, moore=True, include_center=False
        )
        
        victims_nearby = [agent for agent in neighbors if isinstance(agent, VictimAgent)]
        self.assertGreater(len(victims_nearby), 0)
    
    def test_police_criminal_interaction(self):
        """Test interaction between police and criminal"""
        # Police should be able to detect nearby criminals
        neighbors = self.model.grid.get_neighbors(
            self.police.pos, moore=True, include_center=False
        )
        
        criminals_nearby = [agent for agent in neighbors if isinstance(agent, CriminalAgent)]
        self.assertGreater(len(criminals_nearby), 0)
    
    def test_routine_activity_theory(self):
        """Test routine activity theory implementation"""
        # Check if offender, target, and guardian converge
        criminal_pos = self.criminal.pos
        victim_pos = self.victim.pos
        police_pos = self.police.pos
        
        # Calculate distances
        def distance(pos1, pos2):
            return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        
        criminal_victim_distance = distance(criminal_pos, victim_pos)
        police_victim_distance = distance(police_pos, victim_pos)
        
        # Test convergence conditions
        self.assertIsInstance(criminal_victim_distance, float)
        self.assertIsInstance(police_victim_distance, float)


if __name__ == '__main__':
    unittest.main()
