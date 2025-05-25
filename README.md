# Criminal Behavior Modeling Simulation

A comprehensive agent-based modeling (ABM) framework for simulating criminal behavior, environmental influences, and law enforcement strategies using machine learning and advanced visualization tools.

## 🎯 Project Overview

This simulation workspace provides researchers, policymakers, and law enforcement agencies with tools to:

- Model individual criminal decision-making processes
- Simulate routine activity theory dynamics
- Analyze environmental influences on crime patterns
- Test intervention strategies and policy changes
- Generate predictive analytics for crime prevention
- Visualize complex crime patterns and trends

## 🏗️ Architecture

### Core Components

```
crime_modeling/
├── agents/                 # Agent behavior models
│   ├── criminal_agent.py  # Criminal decision-making and behavior
│   ├── victim_agent.py    # Victim vulnerability and protection
│   └── law_enforcement_agent.py  # Police patrol and response
├── environment/           # Environmental modeling
│   └── urban_environment.py  # Spatial grids and dynamics
├── core/                  # Simulation engine
│   └── simulation.py      # Main orchestration
├── analytics/             # Predictive modeling
│   └── predictive_model.py  # ML-based crime prediction
├── visualization/         # Data visualization
│   └── crime_visualization.py  # Charts and dashboards
├── config/               # Configuration management
│   └── settings.py       # Simulation parameters
├── examples/             # Usage examples
│   └── basic_simulation.py  # Getting started
├── tests/                # Unit tests
└── data/                # Input/output data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (Tested with Python 3.12)
- Mesa 3.2.0+ (Agent-based modeling framework)
- VS Code (recommended)
- Git

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "Crime modeling"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Your First Simulation

```python
from core.simulation import CriminalBehaviorSimulation
from config.settings import SIMULATION_CONFIG

# Run a basic crime simulation
simulation = CriminalBehaviorSimulation(config=SIMULATION_CONFIG)
simulation.run_simulation(steps=100)
results = simulation.get_results()
print(f"Simulation completed with {len(simulation.agents)} agents")
```

Or run the example directly:
```bash
python examples/basic_simulation.py
```

## 🎮 Interactive Dashboard

The simulation generates an interactive web dashboard at `output/interactive_dashboard.html` that includes:
- Real-time crime mapping
- Agent behavior visualization  
- Environmental factor analysis
- Statistical reports and trends

## 🚀 Project Status: **FULLY OPERATIONAL** ✅

- **Mesa 3.x Framework**: ✅ Successfully integrated and tested
- **All Core Components**: ✅ Working (Agents, Environment, Analytics, Visualization)
- **Real-time Simulation**: ✅ Running with 260+ agent interactions per step
- **Interactive Dashboard**: ✅ Web-based visualization available
- **Machine Learning**: ✅ Predictive models operational
- **Test Suite**: ✅ Comprehensive unit tests available

**Latest Test Results**: All core modules passing ✓

## 📊 Key Features

### Agent-Based Modeling
- **Criminal Agents**: Rational choice decision-making, risk assessment, routine activities
- **Victim Agents**: Vulnerability factors, protective behaviors, activity patterns
- **Law Enforcement Agents**: Patrol strategies, investigation, arrest procedures

### Environmental Simulation
- **Spatial Grids**: Lighting, surveillance, socioeconomic factors
- **Temporal Dynamics**: Time-of-day effects, seasonal patterns
- **Urban Features**: Population density, commercial areas, residential zones

### Machine Learning Integration
- **Predictive Models**: Crime hotspot identification, risk prediction
- **Feature Engineering**: Spatial-temporal feature extraction
- **Model Validation**: Cross-validation, performance metrics

### Visualization Suite
- **Heatmaps**: Crime density, risk surfaces
- **Time Series**: Temporal crime patterns
- **Interactive Dashboards**: Real-time simulation monitoring
- **Network Analysis**: Criminal network visualization

## 🔬 Theoretical Foundation

### Routine Activity Theory
Models the convergence of motivated offenders, suitable targets, and absence of capable guardians in space and time.

### Rational Choice Theory
Simulates criminal decision-making as cost-benefit analysis considering:
- Expected rewards
- Risk of detection/arrest
- Severity of punishment
- Opportunity factors

### Environmental Criminology
Incorporates spatial and temporal crime patterns:
- Crime attractors and generators
- Defensible space theory
- Crime pattern theory

## 📈 Usage Examples

### Basic Crime Simulation
```python
from core.simulation import CriminalBehaviorSimulation
from config.settings import SIMULATION_CONFIG

# Initialize simulation
sim = CriminalBehaviorSimulation(config=SIMULATION_CONFIG)

# Run simulation
sim.run_simulation(steps=1000)

# Analyze results
results = sim.get_results()
crime_patterns = sim.analyze_crime_patterns()
```

### Predictive Modeling
```python
from analytics.predictive_model import PredictiveModel

# Initialize predictor
predictor = PredictiveModel()

# Train on historical data
predictor.train(features, targets)

# Generate predictions
hotspots = predictor.predict_hotspots(current_conditions)
risk_scores = predictor.calculate_risk_scores(locations)
```

### Visualization
```python
from visualization.crime_visualization import CrimeVisualizationTools

viz = CrimeVisualizationTools()

# Create crime heatmap
viz.create_crime_heatmap(crime_data, save_path="heatmap.png")

# Generate time series analysis
viz.plot_temporal_patterns(temporal_data)

# Launch interactive dashboard
viz.create_interactive_dashboard(simulation_data)
```

## 🧪 Testing and Validation

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test categories:
```bash
# Test agent behaviors
python -m pytest tests/test_agents.py

# Test simulation engine
python -m pytest tests/test_simulation.py

# Test predictive models
python -m pytest tests/test_analytics.py
```

## ⚙️ Configuration

Customize simulation parameters in `config/settings.py`:

```python
SIMULATION_CONFIG = {
    'grid_size': (100, 100),
    'num_criminals': 50,
    'num_victims': 200,
    'num_police': 20,
    'simulation_steps': 1000,
    'crime_rate_base': 0.01,
    'patrol_effectiveness': 0.7
}
```

## 📊 Output and Analysis

The simulation generates various outputs:

- **Crime Events**: Location, time, type, participants
- **Agent Trajectories**: Movement patterns, decision points
- **Environmental Changes**: Dynamic factor evolution
- **Prediction Accuracy**: Model performance metrics
- **Intervention Effects**: Policy impact assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Include docstrings for all public methods
- Write unit tests for new features
- Update documentation as needed

## 📚 Research Applications

### Academic Research
- Crime pattern analysis
- Policy intervention testing
- Theoretical model validation
- Comparative criminology studies

### Law Enforcement
- Resource allocation optimization
- Patrol strategy development
- Crime prevention planning
- Risk assessment tools

### Urban Planning
- Crime impact assessment
- Environmental design evaluation
- Community safety planning
- Development impact analysis

## 🔧 Advanced Features

### Custom Agent Development
Extend the framework with custom agent types:

```python
from agents.base_agent import BaseAgent

class SecurityGuardAgent(BaseAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.patrol_route = []
        self.detection_range = 5
    
    def step(self):
        self.patrol()
        self.detect_crimes()
```

### Integration with Real Data
Connect with external data sources:

```python
# Import crime data
crime_data = pd.read_csv('real_crime_data.csv')
sim.initialize_from_data(crime_data)

# Export results
results = sim.get_detailed_results()
results.to_csv('simulation_output.csv')
```

## 📖 Documentation

- [API Documentation](docs/api.md)
- [Agent Development Guide](docs/agents.md)
- [Environment Modeling](docs/environment.md)
- [Analytics and ML](docs/analytics.md)
- [Visualization Guide](docs/visualization.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review example implementations

## 🙏 Acknowledgments

- Mesa framework for agent-based modeling
- Scikit-learn for machine learning capabilities
- Plotly and Matplotlib for visualization
- Criminological research community for theoretical foundations

---

**Note**: This simulation is designed for research and educational purposes. Real-world crime prediction and prevention should involve domain experts and consider ethical implications.
