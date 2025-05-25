#!/usr/bin/env python3
"""Quick test of the Criminal Behavior Simulation"""

from core.simulation import CriminalBehaviorSimulation
from config.settings import SIMULATION_CONFIG

# Run shorter simulation
config = SIMULATION_CONFIG.copy()
config['simulation_steps'] = 50  # Shorter run

print('Running short Criminal Behavior Simulation...')
simulation = CriminalBehaviorSimulation(config=config)

print(f'Grid: {simulation.width}x{simulation.height}')
print(f'Agents: {simulation.num_criminals} criminals, {simulation.num_victims} victims, {simulation.num_law_enforcement} law enforcement')

# Run simulation
for step in range(50):
    simulation.step()

# Get results
results = simulation.get_results()
print('\nAvailable result keys:', list(results.keys()))
print('\nFinal Results:')
for key, value in results.items():
    print(f'  {key}: {value}')

print('\nSimulation completed successfully!')
