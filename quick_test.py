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
    if (step + 1) % 10 == 0:
        print(f'Step {step + 1}/50 completed')

# Get results
results = simulation.get_results()
print('\nFinal Results:')
print(f'  Total crimes: {results["total_crimes"]}')
print(f'  Total arrests: {results["total_arrests"]}') 
print(f'  Crime rate: {results["crime_rate"]:.4f}')
print(f'  Police effectiveness: {results["police_effectiveness"]:.4f}')

print('\nSimulation completed successfully!')
