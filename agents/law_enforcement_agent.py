"""
Law Enforcement Agent Implementation

This module defines law enforcement agents that patrol, investigate,
and respond to criminal activity within the simulation.
"""

import numpy as np
from mesa import Agent
from typing import Dict, List, Tuple, Optional, Set


class LawEnforcementAgent(Agent):
    """
    Agent representing law enforcement with patrol, investigation, and arrest capabilities
    """
    def __init__(self, model, unique_id: int = None):
        super().__init__(model)
        
        self.agent_type = "law_enforcement"
        
        # Professional characteristics
        self.experience_level = np.random.exponential(5)  # Years of experience
        self.patrol_efficiency = np.random.uniform(0.5, 0.9)  # Patrol effectiveness
        self.investigation_skill = np.random.uniform(0.4, 0.8)  # Investigation ability
        self.response_time = np.random.uniform(0.6, 0.95)  # Speed of response
        
        # Performance metrics
        self.arrests_made = 0
        self.crimes_prevented = 0
        self.investigations_completed = 0
        self.patrol_coverage = {}  # Areas covered and frequency
        
        # Current operational state
        self.patrol_route = self._generate_patrol_route()
        self.current_patrol_index = 0
        self.investigating_crime = False
        self.current_investigation = None
        self.pursuing_suspect = False
        self.target_criminal = None
        
        # Awareness and detection
        self.detection_radius = 3  # How far can detect criminal activity
        self.awareness_level = np.random.uniform(0.6, 0.9)
        self.known_criminals = set()  # Set of known criminal IDs
        self.crime_hotspots = {}  # Known high-crime areas
        
        # Resource allocation
        self.patrol_priority_areas = self._identify_priority_areas()
        self.backup_available = True
        self.shift_hours = 8  # Hours per shift
        self.current_shift_time = 0
        
    def _generate_patrol_route(self) -> List[Tuple[int, int]]:
        """Generate patrol route covering key areas"""
        route = []
        
        # Create route that covers different areas of the grid
        grid_sections = 4  # Divide grid into sections
        section_width = self.model.width // grid_sections
        section_height = self.model.height // grid_sections
        
        for section_x in range(grid_sections):
            for section_y in range(grid_sections):
                # Add multiple points per section
                for _ in range(3):
                    x = np.random.randint(
                        section_x * section_width,
                        min((section_x + 1) * section_width, self.model.width)
                    )
                    y = np.random.randint(
                        section_y * section_height,
                        min((section_y + 1) * section_height, self.model.height)
                    )
                    route.append((x, y))
        
        # Shuffle for randomness
        np.random.shuffle(route)
        return route
    
    def _identify_priority_areas(self) -> List[Tuple[int, int]]:
        """Identify high-priority patrol areas based on environmental factors"""
        priority_areas = []
        
        # Areas with poor lighting or low surveillance are priorities
        env_factors = self.model.environmental_factors
        
        for x in range(0, self.model.width, 5):
            for y in range(0, self.model.height, 5):
                # Simulate varying environmental conditions across grid
                local_lighting = env_factors.get('lighting_quality', 0.5) + np.random.uniform(-0.2, 0.2)
                local_surveillance = env_factors.get('surveillance_coverage', 0.5) + np.random.uniform(-0.2, 0.2)
                
                risk_score = (1.0 - local_lighting) + (1.0 - local_surveillance)
                
                if risk_score > 1.0:  # High-risk area
                    priority_areas.append((x, y))
        
        return priority_areas
    
    def step(self):
        """Execute one step of law enforcement behavior"""
        # Update shift time
        self.current_shift_time = (self.current_shift_time + 1) % (self.shift_hours * 10)
        
        if self.pursuing_suspect:
            self._pursue_criminal()
        elif self.investigating_crime:
            self._continue_investigation()
        else:
            # Regular patrol activities
            self._patrol()
            self._scan_for_criminal_activity()
            self._update_crime_knowledge()
        
        # Update patrol coverage statistics
        self._update_patrol_coverage()
    
    def _patrol(self):
        """Execute patrol movement"""
        if not self.patrol_route:
            self.patrol_route = self._generate_patrol_route()
            
        target_pos = self.patrol_route[self.current_patrol_index]
          # Adjust patrol based on priority areas and recent crimes
        if self._should_redirect_patrol():
            target_pos = self._find_priority_patrol_location()
        
        # Move towards patrol target
        self._move_towards(target_pos)
        
        # Check if reached patrol point
        if self.pos == target_pos:
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_route)
    
    def _should_redirect_patrol(self) -> bool:
        """Determine if patrol should be redirected to priority area"""
        # Check for recent crimes in known hotspots
        recent_crimes = [crime for crime in self.model.crime_incidents 
                        if self.model.step_count - crime['step'] < 50]  # Last 50 steps
        
        if recent_crimes and np.random.random() < 0.3:
            return True
            
        # Randomly patrol priority areas
        return np.random.random() < 0.2
    
    def _find_priority_patrol_location(self) -> Tuple[int, int]:
        """Find high-priority location for patrol"""
        if self.patrol_priority_areas:
            # Select random element from list of tuples
            idx = np.random.randint(0, len(self.patrol_priority_areas))
            return self.patrol_priority_areas[idx]
          # Fallback to hotspot areas
        if self.crime_hotspots:
            hotspot_locations = list(self.crime_hotspots.keys())
            selected_hotspot = np.random.choice(hotspot_locations)
            # Parse the string key back to tuple coordinates
            x_str, y_str = selected_hotspot.split('_')
            return (int(x_str), int(y_str))
        
        # Random location
        return (np.random.randint(0, self.model.width),
                np.random.randint(0, self.model.height))
    
    def _scan_for_criminal_activity(self):
        """Scan area for ongoing criminal activity"""
        # Get nearby agents
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=self.detection_radius
        )
        
        # Check for criminals
        criminals_in_area = [agent for agent in neighbors 
                           if hasattr(agent, 'agent_type') and agent.agent_type == 'criminal']
        
        for criminal in criminals_in_area:
            detection_probability = self._calculate_detection_probability(criminal)
            
            if np.random.random() < detection_probability:
                self._detect_criminal_activity(criminal)
    
    def _calculate_detection_probability(self, criminal) -> float:
        """Calculate probability of detecting criminal activity"""
        base_detection = self.awareness_level
        
        # Distance factor
        distance = np.sqrt((self.pos[0] - criminal.pos[0])**2 + 
                          (self.pos[1] - criminal.pos[1])**2)
        distance_factor = max(0, 1.0 - distance / self.detection_radius)
        
        # Experience factor
        experience_factor = min(self.experience_level / 10, 1.0)
        
        # Criminal visibility factors
        criminal_heat = getattr(criminal, 'heat_level', 0)
        criminal_activity = 1.0 if getattr(criminal, 'planning_crime', False) else 0.5
        
        # Environmental factors
        env_factors = self.model.environmental_factors
        lighting_help = env_factors.get('lighting_quality', 0.5)
        surveillance_help = env_factors.get('surveillance_coverage', 0.5)
        
        detection_prob = (
            base_detection * 0.3 +
            distance_factor * 0.2 +
            experience_factor * 0.2 +
            criminal_heat * 0.15 +
            criminal_activity * 0.1 +
            (lighting_help + surveillance_help) / 2 * 0.05
        )
        
        return min(detection_prob, 0.95)  # Cap at 95%
    
    def _detect_criminal_activity(self, criminal):
        """Respond to detected criminal activity"""
        # Add to known criminals
        self.known_criminals.add(criminal.unique_id)
        
        # Assess if immediate action is needed
        criminal_threat_level = self._assess_threat_level(criminal)
        
        if criminal_threat_level > 0.6:
            self._initiate_pursuit(criminal)
        elif criminal_threat_level > 0.3:
            self._initiate_investigation(criminal)
        else:
            # Increased surveillance
            self._increase_area_surveillance()
    
    def _assess_threat_level(self, criminal) -> float:
        """Assess threat level of detected criminal"""
        # Criminal characteristics
        crimes_committed = getattr(criminal, 'crimes_committed', 0)
        heat_level = getattr(criminal, 'heat_level', 0)
        planning_crime = getattr(criminal, 'planning_crime', False)
        
        # Current activity assessment
        activity_threat = 0.8 if planning_crime else 0.3
        history_threat = min(crimes_committed / 10, 1.0)
        attention_level = heat_level
        
        threat_level = (
            activity_threat * 0.4 +
            history_threat * 0.4 +
            attention_level * 0.2
        )
        
        return min(threat_level, 1.0)
    
    def _initiate_pursuit(self, criminal):
        """Start pursuing a criminal"""
        self.pursuing_suspect = True
        self.target_criminal = criminal
        self.investigating_crime = False
        
    def _initiate_investigation(self, criminal):
        """Start investigating criminal activity"""
        self.investigating_crime = True
        self.current_investigation = {
            'target': criminal,
            'start_step': self.model.step_count,
            'evidence_level': 0.1,
            'location': criminal.pos
        }
    
    def _pursue_criminal(self):
        """Continue pursuing target criminal"""
        if not self.target_criminal or self.target_criminal not in self.model.agents:
            self.pursuing_suspect = False
            self.target_criminal = None
            return
        
        # Move towards criminal
        self._move_towards(self.target_criminal.pos)
        
        # Check if close enough for arrest
        distance = np.sqrt((self.pos[0] - self.target_criminal.pos[0])**2 + 
                          (self.pos[1] - self.target_criminal.pos[1])**2)
        
        if distance <= 1.5:  # Adjacent or very close
            self._attempt_arrest()
    
    def _attempt_arrest(self):
        """Attempt to arrest the target criminal"""
        if not self.target_criminal:
            return
        
        # Arrest probability based on various factors
        arrest_probability = (
            self.patrol_efficiency * 0.4 +
            min(self.experience_level / 10, 1.0) * 0.3 +
            self.response_time * 0.2 +
            (1.0 - self.target_criminal.risk_tolerance) * 0.1
        )
        
        if np.random.random() < arrest_probability:
            self._successful_arrest()
        else:
            self._failed_arrest()
    
    def _successful_arrest(self):
        """Handle successful arrest"""
        if self.target_criminal:
            self.arrests_made += 1
            
            # Reduce criminal's motivation and increase heat
            self.target_criminal.heat_level = min(1.0, self.target_criminal.heat_level + 0.8)
            self.target_criminal.motivation_level = max(0.1, self.target_criminal.motivation_level - 0.5)
            self.target_criminal.previous_arrests += 1
            
            # Record arrest
            arrest_record = {
                'step': self.model.step_count,
                'officer_id': self.unique_id,
                'criminal_id': self.target_criminal.unique_id,
                'location': self.pos,
                'success': True
            }
            
            # Add to model data if available
            if hasattr(self.model, 'arrest_records'):
                self.model.arrest_records.append(arrest_record)
        
        # Reset pursuit state
        self.pursuing_suspect = False
        self.target_criminal = None
    
    def _failed_arrest(self):
        """Handle failed arrest attempt"""
        if self.target_criminal:
            # Criminal becomes more cautious
            self.target_criminal.heat_level = min(1.0, self.target_criminal.heat_level + 0.3)
            self.target_criminal.risk_tolerance = max(0.1, self.target_criminal.risk_tolerance - 0.1)
        
        # Continue pursuit for a few more steps
        pursuit_continuation = np.random.random() < 0.6
        if not pursuit_continuation:
            self.pursuing_suspect = False
            self.target_criminal = None
    
    def _continue_investigation(self):
        """Continue ongoing investigation"""
        if not self.current_investigation:
            self.investigating_crime = False
            return
        
        investigation = self.current_investigation
        
        # Gather evidence over time
        evidence_gain = self.investigation_skill * 0.1 * np.random.uniform(0.5, 1.5)
        investigation['evidence_level'] += evidence_gain
        
        # Move to investigation location
        self._move_towards(investigation['location'])
        
        # Check if investigation complete
        investigation_duration = self.model.step_count - investigation['start_step']
        
        if investigation['evidence_level'] > 0.8 or investigation_duration > 100:
            self._complete_investigation()
    
    def _complete_investigation(self):
        """Complete current investigation"""
        if self.current_investigation:
            self.investigations_completed += 1
            
            # If enough evidence, increase area surveillance
            if self.current_investigation['evidence_level'] > 0.6:
                location = self.current_investigation['location']
                self._add_crime_hotspot(location)
        
        self.investigating_crime = False
        self.current_investigation = None
    
    def _add_crime_hotspot(self, location: Tuple[int, int]):
        """Add location to known crime hotspots"""
        hotspot_key = f"{location[0]}_{location[1]}"
        self.crime_hotspots[hotspot_key] = self.crime_hotspots.get(hotspot_key, 0) + 1
    
    def _increase_area_surveillance(self):
        """Increase surveillance in current area"""
        # This could trigger increased patrol frequency in the area
        area_key = f"{self.pos[0]//10}_{self.pos[1]//10}"
        self.patrol_coverage[area_key] = self.patrol_coverage.get(area_key, 0) + 0.5
    
    def _update_crime_knowledge(self):
        """Update knowledge of crime patterns and hotspots"""
        # Analyze recent crime incidents
        recent_crimes = [crime for crime in self.model.crime_incidents 
                        if self.model.step_count - crime['step'] < 200]  # Last 200 steps
        
        # Update hotspot knowledge
        for crime in recent_crimes:
            location = crime['location']
            self._add_crime_hotspot(location)
    
    def _move_towards(self, target_pos: Tuple[int, int]):
        """Move one step towards target position"""
        current_x, current_y = self.pos
        target_x, target_y = target_pos
        
        # Calculate direction with patrol efficiency affecting speed
        moves_per_step = 1 if self.patrol_efficiency < 0.7 else 2
        
        for _ in range(moves_per_step):
            if current_x == target_x and current_y == target_y:
                break
                
            dx = 0 if current_x == target_x else (1 if target_x > current_x else -1)
            dy = 0 if current_y == target_y else (1 if target_y > current_y else -1)
            
            new_x = max(0, min(self.model.width - 1, current_x + dx))
            new_y = max(0, min(self.model.height - 1, current_y + dy))
            
            current_x, current_y = new_x, new_y
        
        self.model.grid.move_agent(self, (current_x, current_y))
    
    def _update_patrol_coverage(self):
        """Update patrol coverage statistics"""
        area_key = f"{self.pos[0]//5}_{self.pos[1]//5}"  # 5x5 grid sections
        self.patrol_coverage[area_key] = self.patrol_coverage.get(area_key, 0) + 1
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get officer performance metrics"""
        total_patrol_time = sum(self.patrol_coverage.values())
        coverage_efficiency = len(self.patrol_coverage) / max(total_patrol_time, 1)
        
        return {
            'arrests_made': self.arrests_made,
            'crimes_prevented': self.crimes_prevented,
            'investigations_completed': self.investigations_completed,
            'patrol_coverage_efficiency': coverage_efficiency,
            'known_criminals': len(self.known_criminals),
            'known_hotspots': len(self.crime_hotspots)
        }
