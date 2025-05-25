"""
Core simulation engine for Criminal Behavior Modeling

This module implements the main simulation framework that orchestrates
agent-based modeling, environmental factors, and routine activity theory.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from mesa import Model, Agent, DataCollector
from mesa.space import MultiGrid

from agents.criminal_agent import CriminalAgent
from agents.victim_agent import VictimAgent  
from agents.law_enforcement_agent import LawEnforcementAgent
from environment.urban_environment import UrbanEnvironment


class CriminalBehaviorSimulation(Model):
    """
    Main simulation model implementing agent-based criminal behavior modeling
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize the Criminal Behavior Simulation
        
        Args:
            config: Configuration dictionary with simulation parameters
            **kwargs: Individual parameters (override config values)
        """
        super().__init__()        # Use config or default values
        if config:
            grid_size = config.get('grid_size', {'width': 100, 'height': 100})
            # Handle both dict and tuple formats for grid_size
            if isinstance(grid_size, tuple):
                self.width, self.height = grid_size
            elif isinstance(grid_size, dict):
                self.width = grid_size.get('width', 100)
                self.height = grid_size.get('height', 100)
            else:
                self.width = self.height = 100
            
            agent_pops = config.get('agent_populations', {})
            self.num_criminals = config.get('num_criminals', agent_pops.get('criminals', 50))
            self.num_victims = config.get('num_victims', agent_pops.get('victims', 200))
            self.num_law_enforcement = config.get('num_law_enforcement', agent_pops.get('law_enforcement', 10))
        else:
            self.width = kwargs.get('width', 100)
            self.height = kwargs.get('height', 100)
            self.num_criminals = kwargs.get('num_criminals', 50)
            self.num_victims = kwargs.get('num_victims', 200)
            self.num_law_enforcement = kwargs.get('num_law_enforcement', 10)
        
        # Override with kwargs if provided
        self.width = kwargs.get('width', self.width)
        self.height = kwargs.get('height', self.height)
        self.num_criminals = kwargs.get('num_criminals', self.num_criminals)        
        # Default crime types
        crime_types = kwargs.get('crime_types') or config.get('crime_types') if config else None
        self.crime_types = crime_types or [
            "theft", "burglary", "assault", "drug_related", "vandalism"
        ]
        
        # Environmental factors
        environmental_factors = kwargs.get('environmental_factors') or config.get('environmental_factors') if config else None
        self.environmental_factors = environmental_factors or {
            "lighting_quality": 0.7,
            "surveillance_coverage": 0.5,
            "police_presence": 0.3,
            "socioeconomic_status": 0.6,
            "population_density": 0.8
        }
        
        # Initialize components
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.urban_environment = UrbanEnvironment(self)
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Crimes": self.count_crimes,
                "Crime_Rate": self.calculate_crime_rate,
                "Police_Effectiveness": self.calculate_police_effectiveness,
                "Hotspot_Concentration": self.calculate_hotspot_concentration
            },
            agent_reporters={
                "Agent_Type": "agent_type",
                "X": "x",
                "Y": "y",
                "Risk_Level": lambda a: getattr(a, 'risk_level', 0),
                "Crime_Count": lambda a: getattr(a, 'crimes_committed', 0)
            }
        )
          # Initialize agents
        self._create_agents()
        
        # Simulation state
        self.running = True
        self.step_count = 0
        self.crime_incidents = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_agents(self):
        """Create and place agents in the simulation"""
        unique_id = 0
          # Create criminal agents
        for _ in range(self.num_criminals):
            agent = CriminalAgent(self, unique_id=unique_id)
            
            # Place agent randomly
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.grid.place_agent(agent, (x, y))
            unique_id += 1
            
        # Create victim agents
        for _ in range(self.num_victims):
            agent = VictimAgent(self, unique_id=unique_id)
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.grid.place_agent(agent, (x, y))
            unique_id += 1
              # Create law enforcement agents
        for _ in range(self.num_law_enforcement):
            agent = LawEnforcementAgent(self, unique_id=unique_id)
            
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.grid.place_agent(agent, (x, y))
            unique_id += 1
    
    def step(self):
        """Execute one step of the simulation"""
        self.datacollector.collect(self)
        
        # Step all agents (Mesa 3.x approach)
        for agent in self.agents:
            agent.step()
            
        self.urban_environment.update()
        self.step_count += 1
          # Log progress every 100 steps
        if self.step_count % 100 == 0:
            self.logger.info(f"Simulation step {self.step_count}")
    
    def count_crimes(self):
        """Count total crimes in current step"""
        return len([agent for agent in self.agents 
                   if hasattr(agent, 'crimes_committed')])
    
    def calculate_crime_rate(self):
        """Calculate crime rate per 1000 agents"""
        total_crimes = sum([getattr(agent, 'crimes_committed', 0) 
                           for agent in self.agents])
        return (total_crimes / len(self.agents)) * 1000
    
    def calculate_police_effectiveness(self):
        """Calculate police effectiveness metric"""
        police_agents = [agent for agent in self.agents 
                        if hasattr(agent, 'arrests_made')]
        if not police_agents:
            return 0
        
        total_arrests = sum([agent.arrests_made for agent in police_agents])
        total_crimes = self.count_crimes()
        
        return total_arrests / max(total_crimes, 1)
    
    def calculate_hotspot_concentration(self):
        """Calculate concentration of crimes in hotspots"""
        # Simplified hotspot calculation
        crime_locations = [(agent.pos[0], agent.pos[1]) 
                          for agent in self.agents 
                          if hasattr(agent, 'crimes_committed') and agent.crimes_committed > 0]
        
        if not crime_locations:
            return 0
            
        # Calculate concentration using variance
        if len(crime_locations) > 1:
            x_coords = [loc[0] for loc in crime_locations]
            y_coords = [loc[1] for loc in crime_locations]
            concentration = 1 / (np.var(x_coords) + np.var(y_coords) + 1)
            return min(concentration, 1.0)
        
        return 0
    
    def get_simulation_data(self):
        """Get comprehensive simulation data"""
        return {
            'model_data': self.datacollector.get_model_vars_dataframe(),
            'agent_data': self.datacollector.get_agent_vars_dataframe(),
            'crime_incidents': self.crime_incidents,
            'environmental_factors': self.environmental_factors,
            'step_count': self.step_count
        }
    
    def run_simulation(self, steps: int = 1000):
        """Run simulation for specified number of steps"""
        self.logger.info(f"Starting simulation for {steps} steps")
        
        for _ in range(steps):
            self.step()
            
        self.logger.info("Simulation completed")
        return self.get_simulation_data()
    
    def apply_intervention(self, intervention_type: str, intensity: float = 0.5):
        """
        Apply an intervention to the simulation
        
        Args:
            intervention_type: Type of intervention (e.g., 'increase_police_presence')
            intensity: Strength of the intervention (0-1)
        """
        if intervention_type == 'increase_police_presence':
            # Increase police presence in the environment
            for x in range(self.width):
                for y in range(self.height):
                    current_presence = self.urban_environment.police_presence_grid[y, x]
                    self.urban_environment.police_presence_grid[y, x] = min(1.0, current_presence + (intensity * 0.3))
        
        elif intervention_type == 'improve_lighting':
            # Improve lighting in low-light areas
            for x in range(self.width):
                for y in range(self.height):
                    current_lighting = self.urban_environment.lighting_grid[y, x]
                    if current_lighting < 0.5:  # Focus on low-light areas
                        self.urban_environment.lighting_grid[y, x] = min(1.0, current_lighting + (intensity * 0.4))
        
        elif intervention_type == 'increase_surveillance':
            # Increase surveillance coverage
            for x in range(self.width):
                for y in range(self.height):
                    current_surveillance = self.urban_environment.surveillance_grid[y, x]
                    self.urban_environment.surveillance_grid[y, x] = min(1.0, current_surveillance + (intensity * 0.3))
        
        # Log the intervention
        self.intervention_log.append({
            'step': self.step_count,
            'type': intervention_type,
            'intensity': intensity
        })
    
    def reset(self):
        """Reset the simulation to initial state but keep configuration"""
        # Store original config
        original_config = {
            'width': self.width,
            'height': self.height,
            'num_criminals': self.num_criminals,
            'num_victims': self.num_victims,
            'num_law_enforcement': self.num_law_enforcement,
            'crime_types': self.crime_types,
            'environmental_factors': self.environmental_factors
        }
        
        # Reinitialize with same config
        self.__init__(config=None, **original_config)
    
    # Additional methods for test compatibility
    def get_results(self):
        """Get simulation results (test compatibility)"""
        return self.get_simulation_data()
    
    @property 
    def environment(self):
        """Access to urban environment (test compatibility)"""
        return self.urban_environment
    
    @property
    def current_time(self):
        """Current simulation time (test compatibility)"""
        return self.step_count
    
    @property
    def schedule(self):
        """Mesa 2.x compatibility - provide scheduler-like interface"""
        class SchedulerCompat:
            def __init__(self, model):
                self.model = model
                
            @property
            def steps(self):
                return self.model.step_count
            
            @property
            def agents(self):
                return self.model.agents
        
        return SchedulerCompat(self)
    
    @property
    def scheduler(self):
        """Alternative name for schedule"""
        return self.schedule
    
    @property
    def total_crimes(self):
        """Total crimes committed"""
        return sum([getattr(agent, 'crimes_committed', 0) for agent in self.agents])
    
    @property
    def arrests_made(self):
        """Total arrests made"""
        return sum([getattr(agent, 'arrests_made', 0) for agent in self.agents 
                   if hasattr(agent, 'arrests_made')])
    
    def analyze_crime_patterns(self):
        """Analyze crime patterns in the simulation"""
        crime_data = []
        
        for agent in self.agents:
            if hasattr(agent, 'crimes_committed') and agent.crimes_committed > 0:
                crime_data.append({
                    'agent_id': agent.unique_id,
                    'x': agent.pos[0],
                    'y': agent.pos[1],
                    'crimes_committed': agent.crimes_committed,
                    'agent_type': getattr(agent, 'agent_type', 'unknown')
                })
        
        if not crime_data:
            return {'patterns': [], 'hotspots': [], 'statistics': {}}
        
        df = pd.DataFrame(crime_data)
        
        # Basic pattern analysis
        patterns = {
            'spatial_distribution': df.groupby(['x', 'y'])['crimes_committed'].sum().to_dict(),
            'agent_distribution': df.groupby('agent_type')['crimes_committed'].sum().to_dict(),
            'total_crimes': df['crimes_committed'].sum(),
            'unique_locations': len(df[['x', 'y']].drop_duplicates())
        }
        
        # Identify hotspots (locations with > average crimes)
        location_crimes = df.groupby(['x', 'y'])['crimes_committed'].sum()
        avg_crimes = location_crimes.mean()
        hotspots = location_crimes[location_crimes > avg_crimes].to_dict()
        
        return {
            'patterns': patterns,
            'hotspots': hotspots,
            'statistics': {
                'total_crimes': patterns['total_crimes'],
                'hotspot_count': len(hotspots),
                'crime_concentration': len(hotspots) / max(len(location_crimes), 1)
            }
        }
