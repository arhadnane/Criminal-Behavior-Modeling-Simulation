"""
Basic Criminal Behavior Modeling Simulation Example

This example demonstrates how to set up and run a basic simulation
with default parameters and generate visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import CriminalBehaviorSimulation
from analytics.predictive_model import PredictiveModel
from visualization.crime_visualization import CrimeVisualizationTools
from config.settings import SIMULATION_CONFIG, ENVIRONMENTAL_CONFIG
import matplotlib.pyplot as plt


def run_basic_simulation():
    """Run a basic criminal behavior simulation"""
    
    print("Initializing Criminal Behavior Modeling Simulation...")
    
    # Create simulation with configuration
    simulation = CriminalBehaviorSimulation(config=SIMULATION_CONFIG)
    
    print(f"Simulation initialized with:")
    print(f"  - Grid size: {simulation.width}x{simulation.height}")
    print(f"  - {simulation.num_criminals} criminal agents")
    print(f"  - {simulation.num_victims} victim agents") 
    print(f"  - {simulation.num_law_enforcement} law enforcement agents")
    
    # Run simulation
    print("\nRunning simulation...")
    steps = SIMULATION_CONFIG['simulation_steps']
    
    for step in range(steps):
        simulation.step()
        
        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{steps} completed")
    
    print(f"\nSimulation completed after {steps} steps")
    
    # Get simulation results
    results = simulation.get_simulation_data()
    
    print(f"\nSimulation Results:")
    print(f"  - Total crime incidents: {len(simulation.crime_incidents)}")
    
    if simulation.crime_incidents:
        successful_crimes = len([c for c in simulation.crime_incidents if c['success']])
        print(f"  - Successful crimes: {successful_crimes}")
        print(f"  - Crime success rate: {successful_crimes/len(simulation.crime_incidents)*100:.1f}%")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = CrimeVisualizationTools(simulation)
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main plots
    print("  - Creating crime heatmap...")
    visualizer.plot_crime_heatmap(save_path=f"{output_dir}/crime_heatmap.png")
    
    print("  - Creating environmental factors plot...")
    visualizer.plot_environmental_factors(save_path=f"{output_dir}/environmental_factors.png")
    
    print("  - Creating time series analysis...")
    visualizer.plot_crime_time_series(save_path=f"{output_dir}/crime_timeline.png")
    
    print("  - Creating agent distribution plot...")
    visualizer.plot_agent_distribution(save_path=f"{output_dir}/agent_distribution.png")
    
    print("  - Creating agent behavior analysis...")
    visualizer.plot_agent_behavior_analysis(save_path=f"{output_dir}/agent_behavior.png")
    
    # Generate interactive dashboard
    print("  - Creating interactive dashboard...")
    interactive_fig = visualizer.create_interactive_crime_map(
        save_path=f"{output_dir}/interactive_dashboard.html"
    )
    
    # Run predictive analytics if enough data
    print("\nRunning predictive analytics...")
    predictor = PredictiveModel(simulation)
    
    if len(simulation.crime_incidents) >= 20:
        print("  - Training crime prediction models...")
        
        classifier_trained = predictor.train_crime_classifier()
        hotspot_trained = predictor.train_hotspot_predictor()
        
        if classifier_trained:
            print("    ✓ Crime classifier trained successfully")
        if hotspot_trained:
            print("    ✓ Hotspot predictor trained successfully")
        
        # Generate predictive report
        if classifier_trained or hotspot_trained:
            print("  - Generating predictive analytics report...")
            report = predictor.generate_predictive_report()
            
            print(f"\n  Predictive Analytics Summary:")
            print(f"    - Crime trend: {report['crime_trend']:.3f}")
            print(f"    - Top predicted hotspots: {len(report['top_predicted_hotspots'])}")
            
            if 'feature_importance' in report and report['feature_importance']:
                print("    - Top risk factors:")
                for model_name, importance in report['feature_importance'].items():
                    if importance:
                        top_features = sorted(importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        print(f"      {model_name}:")
                        for feature, imp in top_features:
                            print(f"        {feature}: {imp:.3f}")
    else:
        print("  - Insufficient data for predictive modeling (need at least 20 crime incidents)")
    
    print(f"\nVisualization files saved to '{output_dir}/' directory")
    print("\nSimulation completed successfully!")
    
    return simulation, results


if __name__ == "__main__":
    # Run the basic simulation
    simulation, results = run_basic_simulation()
    
    # Keep plots open for viewing
    print("\nPress Enter to close all plots and exit...")
    input()
    plt.close('all')
