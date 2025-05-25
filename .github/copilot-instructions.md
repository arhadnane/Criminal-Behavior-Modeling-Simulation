# GitHub Copilot Instructions for Criminal Behavior Modeling Simulation

## Project Overview
This is a comprehensive criminal behavior modeling simulation workspace using agent-based modeling (ABM) with Mesa framework, machine learning, and visualization tools for criminological research and policy analysis.

## Architecture Guidelines

### Core Components
- **agents/**: Individual agent types (Criminal, Victim, LawEnforcement) with behavioral models
- **environment/**: Urban environment simulation with spatial grids and temporal dynamics
- **core/**: Main simulation engine and orchestration
- **analytics/**: Predictive modeling and statistical analysis
- **visualization/**: Data visualization and interactive dashboards
- **config/**: Centralized configuration management

### Coding Standards

#### Agent Development
- Inherit from `mesa.Agent` for all agent types
- Implement `step()` method for agent behavior per simulation tick
- Use decision-making models based on criminological theories
- Include risk assessment and routine activity patterns
- Maintain agent state and memory for learning behaviors

#### Environment Modeling
- Use spatial grids for geographic representation
- Implement temporal dynamics (time of day, seasonality)
- Model environmental factors (lighting, surveillance, demographics)
- Support dynamic changes during simulation

#### Machine Learning Integration
- Use scikit-learn for predictive models
- Implement feature engineering for crime prediction
- Support model validation and cross-validation
- Include explainable AI for policy insights

#### Visualization Best Practices
- Create both static (matplotlib/seaborn) and interactive (plotly) visualizations
- Implement heatmaps for spatial crime analysis
- Use time series plots for temporal patterns
- Support real-time simulation monitoring

### Theoretical Foundations
- **Routine Activity Theory**: Offender-target-guardian convergence
- **Rational Choice Theory**: Cost-benefit decision making
- **Social Disorganization Theory**: Neighborhood effects
- **Environmental Criminology**: Spatial crime patterns

### Data Handling
- Use pandas for data manipulation
- Implement proper data validation
- Support various data formats (CSV, JSON, spatial data)
- Include data preprocessing pipelines

### Testing Guidelines
- Write unit tests for all major components
- Include integration tests for simulation workflows
- Test agent behaviors and interactions
- Validate statistical outputs

### Configuration Management
- Use centralized configuration in `config/settings.py`
- Support environment-specific settings
- Include parameter validation
- Enable easy experimentation with different scenarios

## Common Patterns

### Agent Implementation
```python
class CustomAgent(Agent):
    def __init__(self, unique_id, model, **kwargs):
        super().__init__(unique_id, model)
        # Initialize agent properties
        
    def step(self):
        # Implement agent behavior
        pass
        
    def make_decision(self):
        # Implement decision-making logic
        pass
```

### Environment Updates
```python
def update_environment(self):
    # Update spatial grids
    # Apply temporal changes
    # Handle dynamic factors
```

### Model Training
```python
def train_predictive_model(self, features, targets):
    # Feature engineering
    # Model training with validation
    # Performance evaluation
```

## Development Priorities
1. **Accuracy**: Ensure theoretical grounding in criminological research
2. **Performance**: Optimize for large-scale simulations
3. **Modularity**: Maintain component independence
4. **Extensibility**: Support easy addition of new agent types and behaviors
5. **Validation**: Include comprehensive testing and validation

## File Naming Conventions
- Snake_case for Python files
- Descriptive names reflecting functionality
- Separate classes into individual files when complex
- Use __init__.py files for package organization

## Dependencies Management
- Keep requirements.txt updated
- Use specific version pinning for reproducibility
- Include development dependencies separately
- Document any special installation requirements

## Documentation Standards
- Include docstrings for all classes and methods
- Provide usage examples in docstrings
- Maintain README.md with setup and usage instructions
- Document theoretical foundations and assumptions
