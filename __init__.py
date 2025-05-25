"""
Criminal Behavior Modeling Simulation

A comprehensive agent-based modeling system for studying criminal behavior patterns,
environmental influences, and predictive analytics for law enforcement and urban planning.

This package includes:
- Agent-based modeling for individual criminal decision-making
- Routine Activity Theory implementation
- Environmental influence modeling
- Predictive analytics using machine learning
- Visualization and analysis tools
"""

__version__ = "1.0.0"
__author__ = "Criminal Behavior Modeling Team"

from .core.simulation import CriminalBehaviorSimulation
from .agents.criminal_agent import CriminalAgent
from .agents.victim_agent import VictimAgent
from .agents.law_enforcement_agent import LawEnforcementAgent
from .environment.urban_environment import UrbanEnvironment
from .analytics.predictive_model import PredictiveModel

__all__ = [
    "CriminalBehaviorSimulation",
    "CriminalAgent", 
    "VictimAgent",
    "LawEnforcementAgent",
    "UrbanEnvironment",
    "PredictiveModel"
]
