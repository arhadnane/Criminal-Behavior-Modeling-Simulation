"""
Visualization Tools for Criminal Behavior Modeling

This module provides comprehensive visualization capabilities for analyzing
simulation results, crime patterns, and environmental factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class CrimeVisualizationTools:
    """
    Comprehensive visualization tools for crime simulation analysis
    """
    
    def __init__(self, model):
        self.model = model
        self.color_schemes = {
            'crime_heat': 'Reds',
            'police_presence': 'Blues', 
            'socioeconomic': 'RdYlGn',
            'risk_assessment': 'YlOrRd'
        }
        
    def plot_crime_heatmap(self, time_window: Optional[Tuple[int, int]] = None, 
                          save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap of crime incidents"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Filter crimes by time window if specified
        crimes = self.model.crime_incidents
        if time_window:
            start_time, end_time = time_window
            crimes = [crime for crime in crimes 
                     if start_time <= crime['step'] <= end_time]
        
        if not crimes:
            ax.text(0.5, 0.5, 'No crimes in specified time window', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create 2D histogram of crime locations
        x_coords = [crime['location'][0] for crime in crimes]
        y_coords = [crime['location'][1] for crime in crimes]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, 
            bins=[self.model.width//2, self.model.height//2],
            range=[[0, self.model.width], [0, self.model.height]]
        )
        
        # Plot heatmap
        im = ax.imshow(heatmap.T, origin='lower', cmap=self.color_schemes['crime_heat'],
                      extent=[0, self.model.width, 0, self.model.height])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Crime Incidents', rotation=270, labelpad=20)
        
        # Customize plot
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Crime Incident Heatmap - Steps {time_window if time_window else "All"}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_environmental_factors(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot environmental factor grids"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        env = self.model.urban_environment
        
        # Environmental grids to plot
        grids = [
            (env.lighting_grid, 'Lighting Quality', self.color_schemes['socioeconomic']),
            (env.surveillance_grid, 'Surveillance Coverage', self.color_schemes['police_presence']),
            (env.police_presence_grid, 'Police Presence', self.color_schemes['police_presence']),
            (env.socioeconomic_grid, 'Socioeconomic Status', self.color_schemes['socioeconomic']),
            (env.population_density_grid, 'Population Density', 'viridis'),
            (env.get_crime_risk_map(), 'Crime Risk Assessment', self.color_schemes['risk_assessment'])
        ]
        
        for i, (grid, title, cmap) in enumerate(grids):
            im = axes[i].imshow(grid, cmap=cmap, vmin=0, vmax=1)
            axes[i].set_title(title, fontsize=12)
            axes[i].set_xlabel('X Coordinate')
            axes[i].set_ylabel('Y Coordinate')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Level', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_agent_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot current distribution of agents"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Get agent positions by type
        agent_positions = {'criminal': [], 'victim': [], 'law_enforcement': []}
        
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'agent_type') and agent.pos:
                agent_positions[agent.agent_type].append(agent.pos)
        
        # Plot each agent type
        colors = {'criminal': 'red', 'victim': 'blue', 'law_enforcement': 'green'}
        markers = {'criminal': 'x', 'victim': 'o', 'law_enforcement': '^'}
        
        for agent_type, positions in agent_positions.items():
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                ax.scatter(x_coords, y_coords, c=colors[agent_type], 
                          marker=markers[agent_type], s=50, alpha=0.7,
                          label=f'{agent_type.replace("_", " ").title()} ({len(positions)})')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Agent Distribution - Step {self.model.step_count}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.model.width)
        ax.set_ylim(0, self.model.height)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_crime_time_series(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot crime incidents over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        if not self.model.crime_incidents:
            ax1.text(0.5, 0.5, 'No crime data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        # Prepare time series data
        crime_df = pd.DataFrame(self.model.crime_incidents)
        crime_df['success'] = crime_df['success'].astype(int)
        
        # Crime incidents over time
        crime_counts = crime_df.groupby('step').size()
        
        ax1.plot(crime_counts.index, crime_counts.values, linewidth=2, color='red')
        ax1.fill_between(crime_counts.index, crime_counts.values, alpha=0.3, color='red')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Crime Incidents')
        ax1.set_title('Crime Incidents Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Success rate over time (rolling average)
        if len(crime_df) > 10:
            window_size = min(50, len(crime_df) // 5)
            success_rate = crime_df.set_index('step')['success'].rolling(window=window_size).mean()
            
            ax2.plot(success_rate.index, success_rate.values, linewidth=2, color='orange')
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Crime Success Rate')
            ax2.set_title(f'Crime Success Rate (Rolling Average, Window={window_size})')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_crime_map(self, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive crime map using Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Crime Incidents', 'Environmental Risk', 
                          'Police Presence', 'Agent Distribution'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Crime incidents scatter plot
        if self.model.crime_incidents:
            crime_df = pd.DataFrame(self.model.crime_incidents)
            fig.add_trace(
                go.Scatter(
                    x=[loc[0] for loc in crime_df['location']],
                    y=[loc[1] for loc in crime_df['location']],
                    mode='markers',
                    marker=dict(color='red', size=6, opacity=0.6),
                    name='Crime Incidents',
                    text=[f"Step: {step}<br>Type: {crime_type}<br>Success: {success}" 
                          for step, crime_type, success in 
                          zip(crime_df['step'], crime_df['crime_type'], crime_df['success'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Environmental risk heatmap
        risk_map = self.model.urban_environment.get_crime_risk_map()
        fig.add_trace(
            go.Heatmap(
                z=risk_map,
                colorscale='Reds',
                showscale=False,
                name='Risk Level'
            ),
            row=1, col=2
        )
        
        # Police presence heatmap
        fig.add_trace(
            go.Heatmap(
                z=self.model.urban_environment.police_presence_grid,
                colorscale='Blues',
                showscale=False,
                name='Police Presence'
            ),
            row=2, col=1
        )
        
        # Agent distribution
        agent_data = {'criminal': [], 'victim': [], 'law_enforcement': []}
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'agent_type') and agent.pos:
                agent_data[agent.agent_type].append(agent.pos)
        
        colors = {'criminal': 'red', 'victim': 'blue', 'law_enforcement': 'green'}
        for agent_type, positions in agent_data.items():
            if positions:
                fig.add_trace(
                    go.Scatter(
                        x=[pos[0] for pos in positions],
                        y=[pos[1] for pos in positions],
                        mode='markers',
                        marker=dict(color=colors[agent_type], size=4),
                        name=agent_type.replace('_', ' ').title()
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text=f"Criminal Behavior Simulation Dashboard - Step {self.model.step_count}",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_agent_behavior_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Analyze and plot agent behavior patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Criminal agent analysis
        criminal_agents = [agent for agent in self.model.schedule.agents 
                          if hasattr(agent, 'agent_type') and agent.agent_type == 'criminal']
        
        if criminal_agents:
            # Crime distribution by criminal
            crimes_per_criminal = [agent.crimes_committed for agent in criminal_agents]
            axes[0, 0].hist(crimes_per_criminal, bins=10, alpha=0.7, color='red')
            axes[0, 0].set_xlabel('Crimes Committed')
            axes[0, 0].set_ylabel('Number of Criminals')
            axes[0, 0].set_title('Distribution of Crimes per Criminal')
            
            # Motivation vs Heat Level
            motivations = [agent.motivation_level for agent in criminal_agents]
            heat_levels = [agent.heat_level for agent in criminal_agents]
            scatter = axes[0, 1].scatter(motivations, heat_levels, alpha=0.6, c=crimes_per_criminal, 
                                       cmap='Reds', s=50)
            axes[0, 1].set_xlabel('Motivation Level')
            axes[0, 1].set_ylabel('Heat Level (Police Attention)')
            axes[0, 1].set_title('Criminal Motivation vs Police Attention')
            plt.colorbar(scatter, ax=axes[0, 1], label='Crimes Committed')
        
        # Law enforcement analysis
        le_agents = [agent for agent in self.model.schedule.agents 
                    if hasattr(agent, 'agent_type') and agent.agent_type == 'law_enforcement']
        
        if le_agents:
            # Arrests per officer
            arrests_per_officer = [agent.arrests_made for agent in le_agents]
            axes[1, 0].bar(range(len(arrests_per_officer)), arrests_per_officer, alpha=0.7, color='blue')
            axes[1, 0].set_xlabel('Officer ID')
            axes[1, 0].set_ylabel('Arrests Made')
            axes[1, 0].set_title('Arrests per Law Enforcement Officer')
            
            # Efficiency vs Experience
            efficiencies = [agent.patrol_efficiency for agent in le_agents]
            experiences = [agent.experience_level for agent in le_agents]
            axes[1, 1].scatter(experiences, efficiencies, alpha=0.6, c=arrests_per_officer, 
                             cmap='Blues', s=50)
            axes[1, 1].set_xlabel('Experience Level')
            axes[1, 1].set_ylabel('Patrol Efficiency')
            axes[1, 1].set_title('Officer Experience vs Efficiency')
            
            if arrests_per_officer:
                scatter2 = axes[1, 1].scatter(experiences, efficiencies, alpha=0.6, 
                                            c=arrests_per_officer, cmap='Blues', s=50)
                plt.colorbar(scatter2, ax=axes[1, 1], label='Arrests Made')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_simulation_dashboard(self, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Create comprehensive simulation dashboard"""
        dashboard = {}
        
        # Main crime analysis
        dashboard['crime_heatmap'] = self.plot_crime_heatmap()
        dashboard['environmental_factors'] = self.plot_environmental_factors()
        dashboard['time_series'] = self.plot_crime_time_series()
        dashboard['agent_distribution'] = self.plot_agent_distribution()
        dashboard['agent_behavior'] = self.plot_agent_behavior_analysis()
        
        # Summary statistics plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Crime statistics
        total_crimes = len(self.model.crime_incidents)
        successful_crimes = len([c for c in self.model.crime_incidents if c['success']])
        
        crime_stats = ['Total Crimes', 'Successful Crimes', 'Failed Crimes']
        crime_counts = [total_crimes, successful_crimes, total_crimes - successful_crimes]
        
        axes[0, 0].bar(crime_stats, crime_counts, color=['red', 'darkred', 'orange'])
        axes[0, 0].set_title('Crime Statistics Summary')
        axes[0, 0].set_ylabel('Count')
        
        # Crime types distribution
        if self.model.crime_incidents:
            crime_types = pd.Series([c['crime_type'] for c in self.model.crime_incidents])
            crime_type_counts = crime_types.value_counts()
            
            axes[0, 1].pie(crime_type_counts.values, labels=crime_type_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Crime Types Distribution')
        
        # Agent counts
        agent_counts = {}
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'agent_type'):
                agent_type = agent.agent_type
                agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        
        if agent_counts:
            axes[1, 0].bar(agent_counts.keys(), agent_counts.values(), 
                          color=['red', 'blue', 'green'])
            axes[1, 0].set_title('Agent Population')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance metrics
        if hasattr(self.model, 'datacollector'):
            model_data = self.model.datacollector.get_model_vars_dataframe()
            if not model_data.empty and 'Crime_Rate' in model_data.columns:
                axes[1, 1].plot(model_data.index, model_data['Crime_Rate'], 
                               linewidth=2, color='purple')
                axes[1, 1].set_xlabel('Simulation Step')
                axes[1, 1].set_ylabel('Crime Rate')
                axes[1, 1].set_title('Crime Rate Over Time')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dashboard['summary_statistics'] = fig
        
        if save_path:
            # Save all plots
            for name, figure in dashboard.items():
                figure.savefig(f"{save_path}_{name}.png", dpi=300, bbox_inches='tight')
        
        return dashboard
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """Export data for external visualization tools"""
        # Crime incidents data
        crime_data = pd.DataFrame(self.model.crime_incidents) if self.model.crime_incidents else pd.DataFrame()
        
        # Agent data
        agent_data = []
        for agent in self.model.schedule.agents:
            if hasattr(agent, 'agent_type') and agent.pos:
                agent_info = {
                    'id': agent.unique_id,
                    'type': agent.agent_type,
                    'x': agent.pos[0],
                    'y': agent.pos[1]
                }
                
                # Add type-specific data
                if agent.agent_type == 'criminal':
                    agent_info.update({
                        'crimes_committed': agent.crimes_committed,
                        'motivation_level': agent.motivation_level,
                        'heat_level': agent.heat_level
                    })
                elif agent.agent_type == 'law_enforcement':
                    agent_info.update({
                        'arrests_made': agent.arrests_made,
                        'patrol_efficiency': agent.patrol_efficiency,
                        'experience_level': agent.experience_level
                    })
                
                agent_data.append(agent_info)
        
        agent_df = pd.DataFrame(agent_data)
        
        # Environmental data
        env_data = self.model.urban_environment.export_environment_data()
        
        return {
            'crime_incidents': crime_data.to_dict('records') if not crime_data.empty else [],
            'agents': agent_df.to_dict('records') if not agent_df.empty else [],
            'environmental_grids': env_data,
            'simulation_metadata': {
                'step_count': self.model.step_count,
                'grid_size': (self.model.width, self.model.height),
                'agent_counts': {
                    'criminals': self.model.num_criminals,
                    'victims': self.model.num_victims,
                    'law_enforcement': self.model.num_law_enforcement
                }
            }
        }
