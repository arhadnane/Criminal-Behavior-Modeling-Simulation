"""
Criminal Agent Implementation

This module defines criminal agents with decision-making capabilities
based on opportunity, risk assessment, and personal motives.
"""

import numpy as np
from mesa import Agent
from typing import Dict, List, Tuple, Optional
from enum import Enum


class CrimeType(Enum):
    """Enumeration of different crime types"""
    THEFT = "theft"
    BURGLARY = "burglary" 
    ASSAULT = "assault"
    DRUG_RELATED = "drug_related"
    VANDALISM = "vandalism"


class CriminalAgent(Agent):
    """
    Agent representing a potential criminal with decision-making capabilities
    """
    def __init__(self, model, unique_id: int = None):
        super().__init__(model)
        
        self.agent_type = "criminal"
        
        # Personal characteristics
        self.age = np.random.normal(25, 8)  # Age distribution
        self.risk_tolerance = np.random.uniform(0.1, 1.0)  # Risk taking propensity
        self.criminal_experience = np.random.exponential(2)  # Experience level
        self.motivation_level = np.random.uniform(0.3, 1.0)  # Current motivation
        
        # Criminal history and behavior
        self.crimes_committed = 0
        self.previous_arrests = np.random.poisson(1)  # Past arrests
        self.preferred_crime_type = np.random.choice(list(CrimeType))
        self.criminal_network_size = np.random.poisson(3)  # Social connections
        
        # Current state
        self.current_target = None
        self.planning_crime = False
        self.last_crime_step = -100  # Steps since last crime
        self.heat_level = 0.0  # Police attention level
        self.territory_familiarity = {}  # Familiarity with different areas
        
        # Routine Activity Theory factors
        self.daily_routine = self._generate_routine()
        self.current_routine_stage = 0
        
        # Decision-making parameters
        self.opportunity_threshold = np.random.uniform(0.4, 0.8)
        self.risk_assessment_skill = np.random.uniform(0.2, 0.9)
        
    def _generate_routine(self) -> List[Tuple[int, int]]:
        """Generate daily routine pattern"""
        routine = []
        for _ in range(24):  # 24 hour cycle
            x = np.random.randint(0, self.model.width)
            y = np.random.randint(0, self.model.height)
            routine.append((x, y))
        return routine
        
    def step(self):
        """Execute one step of criminal agent behavior"""
        # Update routine
        self._follow_routine()
        
        # Assess environment for opportunities
        opportunity_score = self._assess_opportunity()
        
        # Make criminal decision
        if self._should_commit_crime(opportunity_score):
            self._attempt_crime()
        
        # Update agent state
        self._update_state()
        
    def _follow_routine(self):
        """Move according to daily routine with some randomness"""
        routine_pos = self.daily_routine[self.current_routine_stage]
        
        # Add some randomness to routine
        if np.random.random() < 0.3:  # 30% chance of deviation
            target_x = routine_pos[0] + np.random.randint(-2, 3)
            target_y = routine_pos[1] + np.random.randint(-2, 3)
        else:
            target_x, target_y = routine_pos
            
        # Ensure within bounds
        target_x = max(0, min(self.model.width - 1, target_x))
        target_y = max(0, min(self.model.height - 1, target_y))
        
        # Move towards target
        current_x, current_y = self.pos
        
        if current_x < target_x:
            current_x += 1
        elif current_x > target_x:
            current_x -= 1
            
        if current_y < target_y:
            current_y += 1
        elif current_y > target_y:
            current_y -= 1
            
        self.model.grid.move_agent(self, (current_x, current_y))
        
        # Update routine stage
        self.current_routine_stage = (self.current_routine_stage + 1) % len(self.daily_routine)
        
    def _assess_opportunity(self) -> float:
        """Assess criminal opportunity at current location"""
        x, y = self.pos
        
        # Get neighbors in vicinity
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )
        
        # Check for suitable targets (victims)
        suitable_targets = [agent for agent in neighbors 
                          if hasattr(agent, 'agent_type') and agent.agent_type == 'victim']
        
        # Check for guardians (law enforcement)
        guardians = [agent for agent in neighbors 
                    if hasattr(agent, 'agent_type') and agent.agent_type == 'law_enforcement']
        
        # Environmental factors
        env_factors = self.model.environmental_factors
        lighting = env_factors.get('lighting_quality', 0.5)
        surveillance = env_factors.get('surveillance_coverage', 0.5)
        police_presence = env_factors.get('police_presence', 0.3)
        
        # Calculate opportunity score
        target_availability = len(suitable_targets) / max(len(neighbors), 1)
        guardian_absence = 1.0 - (len(guardians) / max(len(neighbors), 1))
        environmental_suitability = (1.0 - lighting) * (1.0 - surveillance) * (1.0 - police_presence)
        
        # Update territory familiarity
        territory_key = f"{x//10}_{y//10}"  # Grid sectors
        self.territory_familiarity[territory_key] = \
            self.territory_familiarity.get(territory_key, 0) + 0.1
        familiarity_bonus = min(self.territory_familiarity.get(territory_key, 0), 1.0)
        
        opportunity_score = (
            target_availability * 0.3 +
            guardian_absence * 0.3 +
            environmental_suitability * 0.3 +
            familiarity_bonus * 0.1
        )
        
        return opportunity_score
        
    def _should_commit_crime(self, opportunity_score: float) -> bool:
        """Decide whether to commit a crime based on various factors"""
        
        # Time since last crime affects motivation
        time_factor = min((self.model.step_count - self.last_crime_step) / 100, 1.0)
        
        # Risk assessment
        perceived_risk = self._calculate_perceived_risk()
        risk_adjusted_motivation = self.motivation_level * (1.0 - perceived_risk * self.risk_assessment_skill)
        
        # Experience factor
        experience_confidence = min(self.criminal_experience / 10, 1.0)
        
        # Heat level (police attention)
        heat_deterrent = 1.0 - min(self.heat_level, 1.0)
        
        # Final decision probability
        crime_probability = (
            opportunity_score * 0.4 +
            risk_adjusted_motivation * 0.3 +
            experience_confidence * 0.2 +
            time_factor * 0.1
        ) * heat_deterrent
        
        return (crime_probability > self.opportunity_threshold and 
                np.random.random() < crime_probability)
    
    def _calculate_perceived_risk(self) -> float:
        """Calculate perceived risk of committing crime"""
        x, y = self.pos
        
        # Check for law enforcement in area
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=3
        )
        
        law_enforcement_nearby = len([agent for agent in neighbors 
                                    if hasattr(agent, 'agent_type') and 
                                    agent.agent_type == 'law_enforcement'])
        
        # Environmental risk factors
        env_factors = self.model.environmental_factors
        surveillance_risk = env_factors.get('surveillance_coverage', 0.5)
        lighting_risk = env_factors.get('lighting_quality', 0.5)
        
        # Personal risk factors
        arrest_history_risk = min(self.previous_arrests / 10, 1.0)
        
        perceived_risk = (
            (law_enforcement_nearby / 5) * 0.4 +
            surveillance_risk * 0.3 +
            lighting_risk * 0.2 +
            arrest_history_risk * 0.1
        )
        
        return min(perceived_risk, 1.0)
    
    def _attempt_crime(self):
        """Attempt to commit a crime"""
        x, y = self.pos
        
        # Find potential victims
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=1
        )
        
        potential_victims = [agent for agent in neighbors 
                           if hasattr(agent, 'agent_type') and agent.agent_type == 'victim'
                           and not getattr(agent, 'victimized_recently', False)]
        
        if potential_victims:
            victim = np.random.choice(potential_victims)
            
            # Success probability based on experience and victim characteristics
            success_prob = (
                0.5 + 
                (self.criminal_experience / 20) +
                (self.risk_tolerance * 0.2) -
                getattr(victim, 'awareness_level', 0.5) * 0.3
            )
            
            if np.random.random() < success_prob:
                # Crime successful
                self._commit_successful_crime(victim)
            else:
                # Crime failed
                self._handle_failed_crime()
    
    def _commit_successful_crime(self, victim):
        """Handle successful crime commission"""
        self.crimes_committed += 1
        self.last_crime_step = self.model.step_count
        self.criminal_experience += 0.5
        self.heat_level += 0.3  # Increased police attention
        
        # Update victim
        victim.victimized_recently = True
        victim.trauma_level = getattr(victim, 'trauma_level', 0) + 0.4
        
        # Record crime incident
        crime_incident = {
            'step': self.model.step_count,
            'criminal_id': self.unique_id,
            'victim_id': victim.unique_id,
            'location': self.pos,
            'crime_type': self.preferred_crime_type.value,
            'success': True
        }
        
        self.model.crime_incidents.append(crime_incident)
        
        # Temporary motivation reduction after successful crime
        self.motivation_level = max(0.1, self.motivation_level - 0.2)
        
    def _handle_failed_crime(self):
        """Handle failed crime attempt"""
        self.heat_level += 0.1
        self.motivation_level += 0.1  # Increased frustration
        
        # Record failed attempt
        crime_incident = {
            'step': self.model.step_count,
            'criminal_id': self.unique_id,
            'victim_id': None,
            'location': self.pos,
            'crime_type': self.preferred_crime_type.value,
            'success': False
        }
        
        self.model.crime_incidents.append(crime_incident)
        
    def _update_state(self):
        """Update agent's internal state"""
        # Gradually reduce heat level over time
        self.heat_level = max(0, self.heat_level - 0.01)
        
        # Motivation fluctuates
        motivation_change = np.random.normal(0, 0.05)
        self.motivation_level = np.clip(
            self.motivation_level + motivation_change, 0.1, 1.0
        )
        
        # Update familiarity with current area
        x, y = self.pos
        territory_key = f"{x//10}_{y//10}"
        self.territory_familiarity[territory_key] = \
            self.territory_familiarity.get(territory_key, 0) + 0.01
