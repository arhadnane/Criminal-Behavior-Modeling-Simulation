"""
Urban Environment Implementation

This module defines the urban environment that influences criminal behavior
through environmental factors, social conditions, and physical infrastructure.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json


class UrbanEnvironment:
    """
    Manages environmental factors that influence criminal behavior
    """
    
    def __init__(self, model):
        self.model = model
        self.width = model.width
        self.height = model.height
        
        # Initialize infrastructure elements first (needed by grids)
        self.building_types = self._initialize_building_types()
        self.transport_hubs = self._initialize_transport_hubs()
        self.commercial_areas = self._initialize_commercial_areas()
        self.residential_areas = self._initialize_residential_areas()
        
        # Environmental grids - each cell has different characteristics
        self.lighting_grid = self._initialize_lighting_grid()
        self.surveillance_grid = self._initialize_surveillance_grid()
        self.socioeconomic_grid = self._initialize_socioeconomic_grid()
        self.population_density_grid = self._initialize_population_density_grid()
        self.police_presence_grid = self._initialize_police_presence_grid()
        
        # Time-based factors
        self.time_of_day = 0  # 0-23 hours
        self.day_of_week = 0  # 0-6 days
        self.seasonal_factor = 0.5  # 0-1 seasonal variation
        
        # Dynamic factors
        self.special_events = []
        self.weather_conditions = {"visibility": 0.8, "temperature": 20}
        self.economic_conditions = {"unemployment_rate": 0.05, "poverty_rate": 0.15}
        
    def _initialize_lighting_grid(self) -> np.ndarray:
        """Initialize lighting quality grid"""
        # Create base lighting with some spatial correlation
        lighting = np.random.uniform(0.3, 0.9, (self.height, self.width))
        
        # Commercial areas have better lighting
        for y in range(self.height):
            for x in range(self.width):
                if self._is_commercial_area(x, y):
                    lighting[y, x] = min(0.9, lighting[y, x] + 0.3)
                elif self._is_residential_area(x, y):
                    lighting[y, x] = min(0.8, lighting[y, x] + 0.1)
        
        return lighting
    
    def _initialize_surveillance_grid(self) -> np.ndarray:
        """Initialize surveillance coverage grid"""
        surveillance = np.random.uniform(0.2, 0.7, (self.height, self.width))
        
        # Commercial and high-traffic areas have more surveillance
        for y in range(self.height):
            for x in range(self.width):
                if self._is_commercial_area(x, y):
                    surveillance[y, x] = min(0.9, surveillance[y, x] + 0.4)
                elif self._is_transport_hub(x, y):
                    surveillance[y, x] = min(0.95, surveillance[y, x] + 0.5)
        
        return surveillance
    
    def _initialize_socioeconomic_grid(self) -> np.ndarray:
        """Initialize socioeconomic status grid"""
        # Create clustered socioeconomic patterns
        socioeconomic = np.random.uniform(0.2, 0.8, (self.height, self.width))
        
        # Add clustering - wealthy areas and poor areas
        wealthy_centers = [(np.random.randint(0, self.width), np.random.randint(0, self.height)) 
                          for _ in range(max(1, self.width // 20))]
        poor_centers = [(np.random.randint(0, self.width), np.random.randint(0, self.height)) 
                       for _ in range(max(1, self.width // 15))]
        
        for y in range(self.height):
            for x in range(self.width):
                # Distance to wealthy centers
                min_wealthy_dist = min([np.sqrt((x-wx)**2 + (y-wy)**2) 
                                      for wx, wy in wealthy_centers])
                
                # Distance to poor centers  
                min_poor_dist = min([np.sqrt((x-px)**2 + (y-py)**2) 
                                   for px, py in poor_centers])
                
                # Adjust socioeconomic status based on proximity
                if min_wealthy_dist < min(10, self.width // 2):
                    socioeconomic[y, x] = min(0.9, socioeconomic[y, x] + 0.4)
                elif min_poor_dist < min(8, self.width // 3):
                    socioeconomic[y, x] = max(0.1, socioeconomic[y, x] - 0.3)
        
        return socioeconomic
    
    def _initialize_population_density_grid(self) -> np.ndarray:
        """Initialize population density grid"""
        density = np.random.uniform(0.3, 0.8, (self.height, self.width))
        
        # Higher density in commercial and residential centers
        for y in range(self.height):
            for x in range(self.width):
                if self._is_commercial_area(x, y):
                    density[y, x] = min(0.95, density[y, x] + 0.3)
                elif self._is_residential_area(x, y):
                    density[y, x] = min(0.85, density[y, x] + 0.2)
        
        return density
    
    def _initialize_police_presence_grid(self) -> np.ndarray:
        """Initialize police presence grid"""
        presence = np.random.uniform(0.1, 0.4, (self.height, self.width))
        
        # Higher presence in commercial areas and known problem areas
        for y in range(self.height):
            for x in range(self.width):
                if self._is_commercial_area(x, y):
                    presence[y, x] = min(0.7, presence[y, x] + 0.3)
                elif self.socioeconomic_grid[y, x] < 0.3:  # Poor areas
                    presence[y, x] = min(0.6, presence[y, x] + 0.2)
        
        return presence
    
    def _initialize_building_types(self) -> Dict[str, List[Tuple[int, int]]]:
        """Initialize different building types across the grid"""
        buildings = {
            'residential': [],
            'commercial': [],
            'industrial': [],
            'educational': [],
            'recreational': [],
            'government': []
        }
        
        # Residential clusters - adjust based on grid size
        num_clusters = max(1, min(20, (self.width * self.height) // 25))
        border_margin = max(1, min(3, self.width // 4, self.height // 4))
        
        for _ in range(num_clusters):
            if self.width > border_margin * 2:
                center_x = np.random.randint(border_margin, self.width - border_margin)
            else:
                center_x = self.width // 2
                
            if self.height > border_margin * 2:
                center_y = np.random.randint(border_margin, self.height - border_margin)
            else:
                center_y = self.height // 2
                
            cluster_size = max(1, min(3, self.width // 5, self.height // 5))
            for dx in range(-cluster_size, cluster_size + 1):
                for dy in range(-cluster_size, cluster_size + 1):
                    x = max(0, min(self.width-1, center_x + dx))
                    y = max(0, min(self.height-1, center_y + dy))
                    buildings['residential'].append((x, y))
        
        # Commercial areas
        commercial_clusters = max(1, min(5, (self.width * self.height) // 50))
        commercial_margin = max(1, min(3, self.width // 5, self.height // 5))
        
        for _ in range(commercial_clusters):
            if self.width > commercial_margin * 2:
                center_x = np.random.randint(commercial_margin, self.width - commercial_margin)
            else:
                center_x = self.width // 2
                
            if self.height > commercial_margin * 2:
                center_y = np.random.randint(commercial_margin, self.height - commercial_margin)
            else:
                center_y = self.height // 2
                
            commercial_size = max(1, min(2, self.width // 6, self.height // 6))
            for dx in range(-commercial_size, commercial_size + 1):
                for dy in range(-commercial_size, commercial_size + 1):
                    x = max(0, min(self.width-1, center_x + dx))
                    y = max(0, min(self.height-1, center_y + dy))
                    buildings['commercial'].append((x, y))
        
        # Other building types - scale count with grid size
        building_counts = {
            'industrial': max(1, min(3, self.width * self.height // 100)),
            'educational': max(1, min(4, self.width * self.height // 80)),
            'recreational': max(1, min(6, self.width * self.height // 60)),
            'government': max(1, min(2, self.width * self.height // 150))
        }
        
        for building_type, count in building_counts.items():
            for _ in range(count):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                buildings[building_type].append((x, y))
        
        return buildings
    
    def _initialize_transport_hubs(self) -> List[Tuple[int, int]]:
        """Initialize transport hub locations"""
        hubs = []
        hub_count = max(2, min(8, self.width * self.height // 50))
        for _ in range(hub_count):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            hubs.append((x, y))
        return hubs
    
    def _initialize_commercial_areas(self) -> List[Tuple[int, int]]:
        """Get commercial area locations"""
        return self.building_types.get('commercial', [])
    
    def _initialize_residential_areas(self) -> List[Tuple[int, int]]:
        """Get residential area locations"""
        return self.building_types.get('residential', [])
    
    def _is_commercial_area(self, x: int, y: int) -> bool:
        """Check if location is in commercial area"""
        return (x, y) in self.commercial_areas
    
    def _is_residential_area(self, x: int, y: int) -> bool:
        """Check if location is in residential area"""
        return (x, y) in self.residential_areas
    
    def _is_transport_hub(self, x: int, y: int) -> bool:
        """Check if location is a transport hub"""
        for hub_x, hub_y in self.transport_hubs:
            if abs(x - hub_x) <= 1 and abs(y - hub_y) <= 1:
                return True
        return False
    
    def get_environmental_factors(self, x: int, y: int) -> Dict[str, float]:
        """Get environmental factors for a specific location"""
        # Ensure coordinates are within bounds
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        # Time-based adjustments
        time_factor = self._get_time_factor()
        
        return {
            'lighting_quality': self.lighting_grid[y, x] * time_factor['lighting'],
            'surveillance_coverage': self.surveillance_grid[y, x],
            'socioeconomic_status': self.socioeconomic_grid[y, x],
            'population_density': self.population_density_grid[y, x] * time_factor['activity'],
            'police_presence': self.police_presence_grid[y, x] * time_factor['police'],
            'building_type': self._get_building_type(x, y),
            'weather_visibility': self.weather_conditions['visibility'],
            'time_of_day': self.time_of_day,
            'day_of_week': self.day_of_week
        }
    
    def _get_time_factor(self) -> Dict[str, float]:
        """Get time-based adjustment factors"""
        # Night time reduces lighting effectiveness and police presence
        if 22 <= self.time_of_day or self.time_of_day <= 6:  # Night
            return {
                'lighting': 0.7,  # Reduced lighting effectiveness
                'activity': 0.3,  # Lower activity levels
                'police': 0.6     # Reduced police presence
            }
        elif 6 < self.time_of_day <= 9 or 17 <= self.time_of_day <= 20:  # Rush hours
            return {
                'lighting': 1.0,
                'activity': 1.2,  # Higher activity
                'police': 1.1     # Increased police presence
            }
        else:  # Day time
            return {
                'lighting': 1.0,
                'activity': 0.8,
                'police': 0.9
            }
    
    def _get_building_type(self, x: int, y: int) -> str:
        """Get building type at location"""
        for building_type, locations in self.building_types.items():
            if (x, y) in locations:
                return building_type
        return 'vacant'
    
    def update(self):
        """Update environmental conditions each simulation step"""
        # Update time
        self.time_of_day = (self.time_of_day + 1) % 24
        if self.time_of_day == 0:
            self.day_of_week = (self.day_of_week + 1) % 7
        
        # Update weather (simple model)
        self.weather_conditions['visibility'] += np.random.normal(0, 0.05)
        self.weather_conditions['visibility'] = np.clip(self.weather_conditions['visibility'], 0.3, 1.0)
        
        self.weather_conditions['temperature'] += np.random.normal(0, 2)
        self.weather_conditions['temperature'] = np.clip(self.weather_conditions['temperature'], -10, 35)
        
        # Seasonal updates (slow changes)
        if self.model.step_count % 1000 == 0:  # Every 1000 steps
            self.seasonal_factor += np.random.normal(0, 0.1)
            self.seasonal_factor = np.clip(self.seasonal_factor, 0, 1)
        
        # Dynamic police presence updates based on recent crime
        self._update_dynamic_police_presence()
    
    def _update_dynamic_police_presence(self):
        """Update police presence based on recent criminal activity"""
        recent_crimes = [crime for crime in self.model.crime_incidents 
                        if self.model.step_count - crime['step'] < 100]
        
        # Increase police presence in high-crime areas
        crime_locations = {}
        for crime in recent_crimes:
            loc = crime['location']
            area_key = f"{loc[0]//5}_{loc[1]//5}"
            crime_locations[area_key] = crime_locations.get(area_key, 0) + 1
        
        # Apply adjustments
        for area_key, crime_count in crime_locations.items():
            if crime_count > 2:  # High crime area
                try:
                    area_x, area_y = map(int, area_key.split('_'))
                    for dx in range(5):
                        for dy in range(5):
                            x = area_x * 5 + dx
                            y = area_y * 5 + dy
                            if 0 <= x < self.width and 0 <= y < self.height:
                                self.police_presence_grid[y, x] = min(0.9, 
                                    self.police_presence_grid[y, x] + 0.1)
                except (ValueError, IndexError):
                    continue
    
    def apply_intervention(self, intervention_type: str, location: Tuple[int, int], 
                          radius: int = 3, intensity: float = 0.3):
        """Apply urban intervention (improved lighting, surveillance, etc.)"""
        x_center, y_center = location
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = x_center + dx
                y = y_center + dy
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    if intervention_type == 'lighting':
                        self.lighting_grid[y, x] = min(1.0, 
                            self.lighting_grid[y, x] + intensity)
                    elif intervention_type == 'surveillance':
                        self.surveillance_grid[y, x] = min(1.0, 
                            self.surveillance_grid[y, x] + intensity)
                    elif intervention_type == 'police_patrol':
                        self.police_presence_grid[y, x] = min(1.0, 
                            self.police_presence_grid[y, x] + intensity)
                    elif intervention_type == 'socioeconomic_improvement':
                        self.socioeconomic_grid[y, x] = min(1.0, 
                            self.socioeconomic_grid[y, x] + intensity)
    
    def get_crime_risk_map(self) -> np.ndarray:
        """Generate crime risk map based on environmental factors"""
        risk_map = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                factors = self.get_environmental_factors(x, y)
                
                # Higher risk in areas with poor conditions
                risk_score = (
                    (1.0 - factors['lighting_quality']) * 0.3 +
                    (1.0 - factors['surveillance_coverage']) * 0.25 +
                    (1.0 - factors['police_presence']) * 0.25 +
                    (1.0 - factors['socioeconomic_status']) * 0.2
                )
                
                # Adjust for building type
                if factors['building_type'] == 'commercial':
                    risk_score *= 1.2  # Higher target value
                elif factors['building_type'] == 'vacant':
                    risk_score *= 1.5  # Abandoned areas
                elif factors['building_type'] == 'government':
                    risk_score *= 0.5  # Better security
                
                risk_map[y, x] = min(risk_score, 1.0)
        
        return risk_map
    
    def export_environment_data(self) -> Dict[str, Any]:
        """Export environment data for analysis"""
        return {
            'lighting_grid': self.lighting_grid.tolist(),
            'surveillance_grid': self.surveillance_grid.tolist(),
            'socioeconomic_grid': self.socioeconomic_grid.tolist(),
            'population_density_grid': self.population_density_grid.tolist(),
            'police_presence_grid': self.police_presence_grid.tolist(),
            'building_types': self.building_types,
            'transport_hubs': self.transport_hubs,
            'weather_conditions': self.weather_conditions,
            'economic_conditions': self.economic_conditions,
            'time_of_day': self.time_of_day,
            'day_of_week': self.day_of_week
        }
