"""
Visualization module for Illinois county-level wage gap analysis.
Generates choropleth maps, scatter plots, and other visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WageGapVisualizer:
    """
    This class handles all the visualization for our wage gap analysis.
    We'll create maps, charts, and plots to communicate our findings effectively.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the visualizer with processed data.
        
        Args:
            data: Processed county-level data
        """
        self.data = data
        self.geo_data = None
        self.figures = {}
        
    def load_data(self, filepath: str = 'data/illinois_county_processed.csv') -> pd.DataFrame:
        """
        Load processed data from CSV file.
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"Loaded processed data from {filepath}")
            return self.data
        except FileNotFoundError:
            print(f"File {filepath} not found. Please run data processing first.")
            return pd.DataFrame()
    
    def load_geographic_data(self, shapefile_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load Illinois county shapefile for mapping.
        """
        try:
            if shapefile_path:
                self.geo_data = gpd.read_file(shapefile_path)
            else:
                # Try to load from a default location
                try:
                    self.geo_data = gpd.read_file('data/illinois_counties.shp')
                except:
                    print("No shapefile found. Creating simplified geographic data...")
                    self.geo_data = self._create_simplified_geo_data()
            
            # Merge with our data
            if self.data is not None and not self.data.empty:
                # Ensure FIPS codes match
                self.geo_data['FIPS'] = self.geo_data['FIPS'].astype(str)
                self.data['fips'] = self.data['fips'].astype(str)
                
                # Merge
                self.geo_data = self.geo_data.merge(
                    self.data, 
                    left_on='FIPS', 
                    right_on='fips', 
                    how='left'
                )
            
            print(f"Loaded geographic data for {len(self.geo_data)} counties")
            return self.geo_data
            
        except Exception as e:
            print(f"Error loading geographic data: {e}")
            print("Creating simplified geographic data for visualization...")
            self.geo_data = self._create_simplified_geo_data()
            return self.geo_data
    
    def _create_simplified_geo_data(self) -> gpd.GeoDataFrame:
        """
        Create simplified geographic data for demonstration.
        """
        if self.data is None or self.data.empty:
            print("No data available for geographic visualization.")
            return gpd.GeoDataFrame()
        
        # Create simple polygons for each county
        from shapely.geometry import Point, Polygon
        
        # Create a simple grid layout
        n_counties = len(self.data)
        cols = int(np.ceil(np.sqrt(n_counties)))
        rows = int(np.ceil(n_counties / cols))
        
        geometries = []
        for i in range(n_counties):
            row = i // cols
            col = i % cols
            
            # Create a simple square polygon
            x, y = col * 2, row * 2
            polygon = Polygon([
                (x, y), (x+1.5, y), (x+1.5, y+1.5), (x, y+1.5), (x, y)
            ])
            geometries.append(polygon)
        
        # Create GeoDataFrame
        geo_df = gpd.GeoDataFrame(
            self.data.copy(),
            geometry=geometries,
            crs="EPSG:4326"
        )
        
        return geo_df
    
    def create_choropleth_map(self, variable: str = 'gender_gap_pct', 
                            title: str = None, save_path: str = None) -> plt.Figure:
        """
        Create a choropleth map of the specified variable across Illinois counties.
        """
        if self.geo_data is None or self.geo_data.empty:
            print("No geographic data available. Loading...")
            self.load_geographic_data()
        
        if self.geo_data is None or self.geo_data.empty:
            print("Could not load geographic data.")
            return None
        
        if variable not in self.geo_data.columns:
            print(f"Variable '{variable}' not found in data.")
            return None
        
        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot the choropleth
        self.geo_data.plot(
            column=variable,
            ax=ax,
            legend=True,
            legend_kwds={'label': f'{variable.replace("_", " ").title()}',
                         'orientation': 'vertical'},
            missing_kwds={'color': 'lightgrey'},
            edgecolor='black',
            linewidth=0.5
        )
        
        # Customize the plot
        if title is None:
            title = f'{variable.replace("_", " ").title()} by County'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add county labels for major counties
        major_counties = ['Cook', 'DuPage', 'Lake', 'Will', 'Kane']
        for idx, row in self.geo_data.iterrows():
            if row['county_name'] in major_counties:
                centroid = row.geometry.centroid
                ax.annotate(
                    row['county_name'],
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to {save_path}")
        
        self.figures[f'choropleth_{variable}'] = fig
        return fig
    
    def create_scatter_plots(self, save_path: str = None) -> plt.Figure:
        """
        Create scatter plots showing relationships between key variables.
        """
        if self.data is None or self.data.empty:
            print("No data available for scatter plots.")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wage Gap Correlations with County Characteristics', 
                    fontsize=16, fontweight='bold')
        
        # Define variable pairs for scatter plots
        scatter_pairs = [
            ('pct_bach', 'gender_gap_pct', 'Education vs Gender Gap'),
            ('pct_black', 'gender_gap_pct', 'Black Population vs Gender Gap'),
            ('pct_asian', 'gender_gap_pct', 'Asian Population vs Gender Gap'),
            ('log_pop', 'gender_gap_pct', 'Population vs Gender Gap'),
            ('pct_manuf', 'gender_gap_pct', 'Manufacturing vs Gender Gap'),
            ('median_income', 'gender_gap_pct', 'Income vs Gender Gap')
        ]
        
        for i, (x_var, y_var, title) in enumerate(scatter_pairs):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Check if variables exist
            if x_var in self.data.columns and y_var in self.data.columns:
                # Create scatter plot
                ax.scatter(self.data[x_var], self.data[y_var], alpha=0.7, s=50)
                
                # Add trend line
                z = np.polyfit(self.data[x_var], self.data[y_var], 1)
                p = np.poly1d(z)
                ax.plot(self.data[x_var], p(self.data[x_var]), "r--", alpha=0.8)
                
                # Add correlation coefficient
                corr = self.data[x_var].corr(self.data[y_var])
                ax.text(0.05, 0.95, f'ρ = {corr:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                # Labels
                ax.set_xlabel(x_var.replace('_', ' ').title())
                ax.set_ylabel(y_var.replace('_', ' ').title())
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Data not available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plots saved to {save_path}")
        
        self.figures['scatter_plots'] = fig
        return fig
    
    def create_gender_gap_distribution(self, save_path: str = None) -> plt.Figure:
        """
        Create histogram and box plot of gender wage gap distribution.
        """
        if self.data is None or self.data.empty:
            print("No data available for distribution plots.")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(self.data['gender_gap_pct'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.data['gender_gap_pct'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.data["gender_gap_pct"].mean():.1f}%')
        ax1.axvline(self.data['gender_gap_pct'].median(), color='green', linestyle='--', 
                   label=f'Median: {self.data["gender_gap_pct"].median():.1f}%')
        ax1.set_xlabel('Gender Wage Gap (%)')
        ax1.set_ylabel('Number of Counties')
        ax1.set_title('Distribution of Gender Wage Gap Across Counties')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(self.data['gender_gap_pct'], patch_artist=True, 
                   boxprops=dict(facecolor='lightblue'))
        ax2.set_ylabel('Gender Wage Gap (%)')
        ax2.set_title('Box Plot of Gender Wage Gap')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to {save_path}")
        
        self.figures['gender_gap_distribution'] = fig
        return fig
    
    def create_top_bottom_counties(self, n: int = 10, save_path: str = None) -> plt.Figure:
        """
        Create bar chart showing top and bottom counties by gender wage gap.
        """
        if self.data is None or self.data.empty:
            print("No data available for county comparison.")
            return None
        
        # Get top and bottom counties
        top_counties = self.data.nlargest(n, 'gender_gap_pct')[['county_name', 'gender_gap_pct']]
        bottom_counties = self.data.nsmallest(n, 'gender_gap_pct')[['county_name', 'gender_gap_pct']]
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top counties
        bars1 = ax1.barh(range(len(top_counties)), top_counties['gender_gap_pct'], 
                        color='red', alpha=0.7)
        ax1.set_yticks(range(len(top_counties)))
        ax1.set_yticklabels(top_counties['county_name'])
        ax1.set_xlabel('Gender Wage Gap (%)')
        ax1.set_title(f'Top {n} Counties by Gender Wage Gap')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, top_counties['gender_gap_pct'])):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}%', va='center', fontsize=9)
        
        # Bottom counties
        bars2 = ax2.barh(range(len(bottom_counties)), bottom_counties['gender_gap_pct'], 
                        color='green', alpha=0.7)
        ax2.set_yticks(range(len(bottom_counties)))
        ax2.set_yticklabels(bottom_counties['county_name'])
        ax2.set_xlabel('Gender Wage Gap (%)')
        ax2.set_title(f'Bottom {n} Counties by Gender Wage Gap')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, bottom_counties['gender_gap_pct'])):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"County comparison saved to {save_path}")
        
        self.figures['top_bottom_counties'] = fig
        return fig
    
    def create_racial_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Create side-by-side bar charts comparing median earnings by race.
        """
        if self.data is None or self.data.empty:
            print("No data available for racial comparison.")
            return None
        
        # Calculate mean earnings by race (if available)
        # For demonstration, we'll use the population percentages as proxies
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create a simple comparison using available data
        race_vars = ['pct_white', 'pct_black', 'pct_asian', 'pct_multi']
        available_vars = [var for var in race_vars if var in self.data.columns]
        
        if available_vars:
            # Calculate mean values for each racial group
            race_means = {}
            for var in available_vars:
                race_means[var.replace('pct_', '').title()] = self.data[var].mean()
            
            # Create bar chart
            races = list(race_means.keys())
            values = list(race_means.values())
            
            bars = ax.bar(races, values, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
            ax.set_ylabel('Mean Percentage of Population')
            ax.set_title('Racial Composition Across Illinois Counties')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Racial composition data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Racial Composition Comparison')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Racial comparison saved to {save_path}")
        
        self.figures['racial_comparison'] = fig
        return fig
    
    def create_regression_coefficients(self, results: Dict, save_path: str = None) -> plt.Figure:
        """
        Create coefficient plot for regression results.
        """
        if not results or 'baseline' not in results:
            print("No regression results available.")
            return None
        
        baseline_results = results['baseline']
        coefficients = baseline_results['coefficients']
        std_errors = baseline_results['std_errors']
        p_values = baseline_results['p_values']
        
        # Filter out intercept
        coef_data = {k: v for k, v in coefficients.items() if k != 'Intercept'}
        
        if not coef_data:
            print("No coefficients to plot.")
            return None
        
        # Create coefficient plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        variables = list(coef_data.keys())
        coef_values = list(coef_data.values())
        se_values = [std_errors[var] for var in variables]
        
        # Calculate confidence intervals
        ci_lower = [coef - 1.96 * se for coef, se in zip(coef_values, se_values)]
        ci_upper = [coef + 1.96 * se for coef, se in zip(coef_values, se_values)]
        
        # Color based on significance
        colors = []
        for var in variables:
            pval = p_values[var]
            if pval < 0.01:
                colors.append('red')
            elif pval < 0.05:
                colors.append('orange')
            elif pval < 0.1:
                colors.append('yellow')
            else:
                colors.append('lightgrey')
        
        # Create the plot
        y_pos = np.arange(len(variables))
        bars = ax.barh(y_pos, coef_values, xerr=se_values, capsize=5, 
                      color=colors, alpha=0.7)
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([var.replace('_', ' ').title() for var in variables])
        ax.set_xlabel('Coefficient Estimate')
        ax.set_title('Regression Coefficients: Gender Wage Gap Determinants')
        ax.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (var, pval) in enumerate(p_values.items()):
            if var != 'Intercept':
                if pval < 0.01:
                    ax.text(coef_values[i] + se_values[i] + 0.5, i, '***', 
                           va='center', fontweight='bold')
                elif pval < 0.05:
                    ax.text(coef_values[i] + se_values[i] + 0.5, i, '**', 
                           va='center', fontweight='bold')
                elif pval < 0.1:
                    ax.text(coef_values[i] + se_values[i] + 0.5, i, '*', 
                           va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coefficient plot saved to {save_path}")
        
        self.figures['regression_coefficients'] = fig
        return fig
    
    def create_all_visualizations(self, results: Dict = None) -> Dict:
        """
        Create all visualizations for the wage gap analysis.
        """
        print("=== CREATING ALL VISUALIZATIONS ===")
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Create all plots
        visualizations = {}
        
        # 1. Choropleth map of gender gap
        print("1. Creating gender gap choropleth map...")
        fig1 = self.create_choropleth_map(
            'gender_gap_pct', 
            'Gender Wage Gap by County (%)',
            'results/gender_gap_map.png'
        )
        visualizations['gender_gap_map'] = fig1
        
        # 2. Scatter plots
        print("2. Creating scatter plots...")
        fig2 = self.create_scatter_plots('results/scatter_plots.png')
        visualizations['scatter_plots'] = fig2
        
        # 3. Gender gap distribution
        print("3. Creating gender gap distribution...")
        fig3 = self.create_gender_gap_distribution('results/gender_gap_distribution.png')
        visualizations['gender_gap_distribution'] = fig3
        
        # 4. Top/bottom counties
        print("4. Creating county comparison...")
        fig4 = self.create_top_bottom_counties(10, 'results/county_comparison.png')
        visualizations['county_comparison'] = fig4
        
        # 5. Racial comparison
        print("5. Creating racial comparison...")
        fig5 = self.create_racial_comparison('results/racial_comparison.png')
        visualizations['racial_comparison'] = fig5
        
        # 6. Regression coefficients (if results provided)
        if results:
            print("6. Creating regression coefficient plot...")
            fig6 = self.create_regression_coefficients(results, 'results/regression_coefficients.png')
            visualizations['regression_coefficients'] = fig6
        
        print("\n=== VISUALIZATION COMPLETE ===")
        print("All figures saved to results/ directory")
        
        return visualizations

def main():
    """
    Main function to create visualizations for the wage gap analysis.
    """
    print("=== ILLINOIS COUNTY WAGE GAP VISUALIZATION ===")
    
    # Initialize visualizer
    visualizer = WageGapVisualizer()
    
    # Load data
    data = visualizer.load_data()
    
    if data.empty:
        print("No data available. Please run data processing first.")
        return visualizer, {}
    
    # Load geographic data
    visualizer.load_geographic_data()
    
    # Create all visualizations
    visualizations = visualizer.create_all_visualizations()
    
    return visualizer, visualizations

if __name__ == "__main__":
    visualizer, visualizations = main() 