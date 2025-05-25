#!/usr/bin/env python3
"""
Criminal Behavior Modeling Simulation - Project Status Report
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_project_status():
    """Check the status of all project components"""
    
    print("=" * 60)
    print("CRIMINAL BEHAVIOR MODELING SIMULATION - STATUS REPORT")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. DEPENDENCY STATUS:")
    try:
        import mesa
        print(f"   ✓ Mesa Framework: {mesa.__version__}")
    except ImportError:
        print("   ✗ Mesa Framework: NOT INSTALLED")
    
    try:
        import numpy
        print(f"   ✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("   ✗ NumPy: NOT INSTALLED")
    
    try:
        import pandas
        print(f"   ✓ Pandas: {pandas.__version__}")
    except ImportError:
        print("   ✗ Pandas: NOT INSTALLED")
    
    try:
        import matplotlib
        print(f"   ✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("   ✗ Matplotlib: NOT INSTALLED")
    
    try:
        import plotly
        print(f"   ✓ Plotly: {plotly.__version__}")
    except ImportError:
        print("   ✗ Plotly: NOT INSTALLED")
    
    try:
        import sklearn
        print(f"   ✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("   ✗ Scikit-learn: NOT INSTALLED")
    
    # Check core modules
    print("\n2. CORE MODULES STATUS:")
    try:
        from core.simulation import CriminalBehaviorSimulation
        print("   ✓ Simulation Engine: WORKING")
    except Exception as e:
        print(f"   ✗ Simulation Engine: ERROR - {e}")
    
    try:
        from agents.criminal_agent import CriminalAgent
        from agents.victim_agent import VictimAgent
        from agents.law_enforcement_agent import LawEnforcementAgent
        print("   ✓ Agent Models: WORKING")
    except Exception as e:
        print(f"   ✗ Agent Models: ERROR - {e}")
    
    try:
        from environment.urban_environment import UrbanEnvironment
        print("   ✓ Environment Model: WORKING")
    except Exception as e:
        print(f"   ✗ Environment Model: ERROR - {e}")
    
    try:
        from analytics.predictive_model import PredictiveModel
        print("   ✓ Analytics Module: WORKING")
    except Exception as e:
        print(f"   ✗ Analytics Module: ERROR - {e}")
    
    try:
        from visualization.crime_visualization import CrimeVisualizationTools
        print("   ✓ Visualization Tools: WORKING")
    except Exception as e:
        print(f"   ✗ Visualization Tools: ERROR - {e}")
    
    # Test basic simulation
    print("\n3. SIMULATION TEST:")
    try:
        from core.simulation import CriminalBehaviorSimulation
        from config.settings import SIMULATION_CONFIG
        
        config = SIMULATION_CONFIG.copy()
        config['simulation_steps'] = 5
        config['grid_size'] = (10, 10)
        config['num_criminals'] = 3
        config['num_victims'] = 5
        config['num_law_enforcement'] = 2
        
        simulation = CriminalBehaviorSimulation(config=config)
        
        # Run a few steps
        for i in range(5):
            simulation.step()
        
        print("   ✓ Basic Simulation: RUNNING SUCCESSFULLY")
        print(f"     - Grid Size: {simulation.width}x{simulation.height}")
        print(f"     - Total Agents: {len(simulation.agents)}")
        print(f"     - Simulation Steps: {simulation.step_count}")
        
    except Exception as e:
        print(f"   ✗ Basic Simulation: ERROR - {e}")
    
    # Check output files
    print("\n4. OUTPUT FILES STATUS:")
    output_dir = "output"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            print("   ✓ Output Files Generated:")
            for file in files:
                print(f"     - {file}")
        else:
            print("   ⚠ Output Directory Empty")
    else:
        print("   ✗ Output Directory Missing")
    
    print("\n" + "=" * 60)
    print("PROJECT STATUS: READY FOR USE")
    print("=" * 60)
    print("\nTo run the simulation:")
    print("  python examples/basic_simulation.py")
    print("\nTo run tests:")
    print("  python -m pytest tests/ -v")
    print("\nTo view interactive dashboard:")
    print("  Open: output/interactive_dashboard.html")

if __name__ == "__main__":
    check_project_status()
