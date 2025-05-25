"""
Victim Agent Implementation

This module defines victim agents representing potential targets
with varying levels of vulnerability and protective behaviors.
"""

import numpy as np
from mesa import Agent
from typing import Dict, List, Tuple, Optional


class VictimAgent(Agent):
    """
    Agent representing potential crime victims with varying vulnerability levels
    """
    def __init__(self, model, unique_id: int = None):
        super().__init__(model)
        
        self.agent_type = "victim"
        
        # Personal characteristics
        self.age = np.random.normal(35, 15)  # Age distribution
        self.wealth_level = np.random.lognormal(0, 1)  # Wealth indicator
        self.physical_strength = np.random.uniform(0.2, 1.0)  # Physical capability
        self.awareness_level = np.random.uniform(0.3, 0.9)  # Situational awareness
        
        # Vulnerability factors
        self.visibility = np.random.uniform(0.4, 1.0)  # How visible/noticeable
        self.defensibility = np.random.uniform(0.2, 0.8)  # Ability to defend
        self.routine_predictability = np.random.uniform(0.3, 0.9)  # Routine predictability
        
        # Current state
        self.victimized_recently = False
        self.trauma_level = 0.0
        self.fear_level = np.random.uniform(0.1, 0.5)
        self.protective_behavior = np.random.uniform(0.3, 0.7)
        
        # Location and movement patterns
        self.home_location = (
            np.random.randint(0, model.width),
            np.random.randint(0, model.height)
        )
        self.work_location = (
            np.random.randint(0, model.width),
            np.random.randint(0, model.height)
        )
        
        # Daily routine
        self.daily_routine = self._generate_routine()
        self.current_routine_stage = 0
        
        # Risk perception and avoidance
        self.risk_perception = np.random.uniform(0.4, 0.8)
        self.avoidance_behavior = np.random.uniform(0.2, 0.7)
        
    def _generate_routine(self) -> List[Tuple[int, int]]:
        """Generate predictable daily routine"""
        routine = []
        
        # Morning routine (home area)
        for hour in range(6):
            x = self.home_location[0] + np.random.randint(-2, 3)
            y = self.home_location[1] + np.random.randint(-2, 3)
            routine.append((
                max(0, min(self.model.width - 1, x)),
                max(0, min(self.model.height - 1, y))
            ))
        
        # Work hours (work area)
        for hour in range(8):
            x = self.work_location[0] + np.random.randint(-3, 4)
            y = self.work_location[1] + np.random.randint(-3, 4)
            routine.append((
                max(0, min(self.model.width - 1, x)),
                max(0, min(self.model.height - 1, y))
            ))
        
        # Evening routine (mixed areas)
        for hour in range(6):
            if np.random.random() < 0.7:  # Usually home
                base_x, base_y = self.home_location
            else:  # Sometimes other areas
                base_x = np.random.randint(0, self.model.width)
                base_y = np.random.randint(0, self.model.height)
            
            x = base_x + np.random.randint(-3, 4)
            y = base_y + np.random.randint(-3, 4)
            routine.append((
                max(0, min(self.model.width - 1, x)),
                max(0, min(self.model.height - 1, y))
            ))
        
        # Night routine (home)
        for hour in range(4):
            x = self.home_location[0] + np.random.randint(-1, 2)
            y = self.home_location[1] + np.random.randint(-1, 2)
            routine.append((
                max(0, min(self.model.width - 1, x)),
                max(0, min(self.model.height - 1, y))
            ))
        
        return routine
    
    def step(self):
        """Execute one step of victim agent behavior"""
        # Follow routine with risk-based modifications
        self._follow_routine_with_risk_assessment()
        
        # Update protective behaviors based on environment
        self._update_protective_behavior()
        
        # Recover from recent victimization
        self._process_trauma_recovery()
        
        # Update awareness and fear levels
        self._update_psychological_state()
    
    def _follow_routine_with_risk_assessment(self):
        """Move according to routine while considering risk factors"""
        target_pos = self.daily_routine[self.current_routine_stage]
        
        # Assess risk at target location
        risk_level = self._assess_location_risk(target_pos)
        
        # Modify routine based on risk and fear
        if risk_level > self.risk_perception and np.random.random() < self.avoidance_behavior:
            # Avoid risky area - find alternative
            target_pos = self._find_safer_alternative(target_pos)
        
        # Add routine predictability variation
        if np.random.random() < (1.0 - self.routine_predictability):
            deviation_x = np.random.randint(-2, 3)
            deviation_y = np.random.randint(-2, 3)
            target_pos = (
                max(0, min(self.model.width - 1, target_pos[0] + deviation_x)),
                max(0, min(self.model.height - 1, target_pos[1] + deviation_y))
            )
        
        # Move towards target position
        self._move_towards(target_pos)
        
        # Update routine stage
        self.current_routine_stage = (self.current_routine_stage + 1) % len(self.daily_routine)
    
    def _assess_location_risk(self, location: Tuple[int, int]) -> float:
        """Assess risk level at a specific location"""
        x, y = location
        
        # Get agents in the area
        nearby_agents = []
        for agent in self.model.agents:
            if agent.pos:
                distance = np.sqrt((agent.pos[0] - x)**2 + (agent.pos[1] - y)**2)
                if distance <= 3:  # Within 3 cell radius
                    nearby_agents.append(agent)
        
        # Count potential threats (criminals)
        criminals_nearby = len([agent for agent in nearby_agents 
                              if hasattr(agent, 'agent_type') and agent.agent_type == 'criminal'])
        
        # Count protection (law enforcement)
        law_enforcement_nearby = len([agent for agent in nearby_agents 
                                    if hasattr(agent, 'agent_type') and agent.agent_type == 'law_enforcement'])
        
        # Environmental risk factors
        env_factors = self.model.environmental_factors
        lighting_safety = env_factors.get('lighting_quality', 0.5)
        surveillance_safety = env_factors.get('surveillance_coverage', 0.5)
        police_presence = env_factors.get('police_presence', 0.3)
        
        # Calculate risk score
        threat_level = min(criminals_nearby / 3, 1.0)  # Normalize
        protection_level = min(law_enforcement_nearby / 2, 1.0)
        environmental_risk = 1.0 - (lighting_safety + surveillance_safety + police_presence) / 3
        
        total_risk = (
            threat_level * 0.4 +
            (1.0 - protection_level) * 0.3 +
            environmental_risk * 0.3
        )
        
        return min(total_risk, 1.0)
    
    def _find_safer_alternative(self, original_target: Tuple[int, int]) -> Tuple[int, int]:
        """Find a safer alternative location near the original target"""
        best_pos = original_target
        lowest_risk = float('inf')
        
        # Check nearby locations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                    
                alt_x = max(0, min(self.model.width - 1, original_target[0] + dx))
                alt_y = max(0, min(self.model.height - 1, original_target[1] + dy))
                alt_pos = (alt_x, alt_y)
                
                risk = self._assess_location_risk(alt_pos)
                if risk < lowest_risk:
                    lowest_risk = risk
                    best_pos = alt_pos
        
        return best_pos
    
    def _move_towards(self, target_pos: Tuple[int, int]):
        """Move one step towards target position"""
        current_x, current_y = self.pos
        target_x, target_y = target_pos
        
        # Calculate direction
        dx = 0 if current_x == target_x else (1 if target_x > current_x else -1)
        dy = 0 if current_y == target_y else (1 if target_y > current_y else -1)
        
        # Move one step
        new_x = max(0, min(self.model.width - 1, current_x + dx))
        new_y = max(0, min(self.model.height - 1, current_y + dy))
        
        self.model.grid.move_agent(self, (new_x, new_y))
    
    def _update_protective_behavior(self):
        """Update protective behaviors based on current environment and state"""
        current_risk = self._assess_location_risk(self.pos)
        
        # Increase protective behavior in high-risk areas
        if current_risk > 0.7:
            self.protective_behavior = min(1.0, self.protective_behavior + 0.1)
            self.awareness_level = min(1.0, self.awareness_level + 0.05)
        
        # Gradually return to baseline if in safe areas
        if current_risk < 0.3:
            self.protective_behavior = max(0.2, self.protective_behavior - 0.02)
    
    def _process_trauma_recovery(self):
        """Process recovery from recent victimization"""
        if self.victimized_recently:
            # Gradual recovery over time
            self.trauma_level = max(0, self.trauma_level - 0.05)
            
            if self.trauma_level < 0.1:
                self.victimized_recently = False
                self.trauma_level = 0
    
    def _update_psychological_state(self):
        """Update fear levels and psychological state"""
        current_risk = self._assess_location_risk(self.pos)
        
        # Fear responds to current risk and trauma
        target_fear = current_risk * 0.5 + self.trauma_level * 0.3
        
        # Gradual adjustment towards target fear level
        if self.fear_level < target_fear:
            self.fear_level = min(1.0, self.fear_level + 0.1)
        else:
            self.fear_level = max(0, self.fear_level - 0.05)
        
        # Fear affects other behaviors
        if self.fear_level > 0.7:
            self.avoidance_behavior = min(1.0, self.avoidance_behavior + 0.05)
            self.routine_predictability = max(0.2, self.routine_predictability - 0.02)
    
    def get_vulnerability_score(self) -> float:
        """Calculate overall vulnerability score"""
        # Combine various vulnerability factors
        personal_vulnerability = (
            (1.0 - self.physical_strength) * 0.3 +
            (1.0 - self.awareness_level) * 0.2 +
            self.visibility * 0.2 +
            (1.0 - self.defensibility) * 0.3
        )
        
        # Adjust for current state
        state_modifier = (
            self.trauma_level * 0.3 +
            self.fear_level * 0.2 +
            (1.0 - self.protective_behavior) * 0.5
        )
        
        total_vulnerability = (personal_vulnerability + state_modifier) / 2
        return min(total_vulnerability, 1.0)
