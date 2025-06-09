#!/usr/bin/env python3
"""
Illinois County Wage Gap Analysis - Python Script Version
This script performs the complete analysis of gender and racial wage gaps across Illinois counties.

Run this script to:
1. Collect Census data
2. Process and clean data
3. Run statistical analysis
4. Create visualizations
5. Generate summary report
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path - fix the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

# Import our custom modules
from data_collection import CensusDataCollector
from data_processing import DataProcessor
from analysis import WageGapAnalyzer
from visualization import WageGapVisualizer

def setup_environment():
    """Setup the analysis environment."""
    print("=" * 80)
    print("ILLINOIS COUNTY WAGE GAP ANALYSIS")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("✅ Environment setup complete")
    print("✅ Directories created: ../data, ../results")

def collect_data():
    """Step 1: Collect Census data."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)
    
    # Initialize data collector
    collector = CensusDataCollector(year=2022, survey='acs1')
    
    # Download county data
    raw_data = collector.download_county_data()
    
    print(f"\n✅ Collected data for {len(raw_data)} counties")
    print(f"📊 Columns: {list(raw_data.columns)}")
    
    # Display basic info
    print(f"\n📈 Data Summary:")
    print(f"   • Counties: {len(raw_data)}")
    print(f"   • Variables: {len(raw_data.columns)}")
    print(f"   • Memory usage: {raw_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    return raw_data

def process_data(raw_data):
    """Step 2: Process and clean data."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PROCESSING")
    print("=" * 60)
    
    # Initialize data processor
    processor = DataProcessor(raw_data)
    
    # Process the data
    processed_data = processor.process_data()
    
    print(f"\n✅ Processed data for {len(processed_data)} counties")
    
    # Display key variables
    key_vars = ['county_name', 'gender_gap_pct', 'male_med_earn', 'female_med_earn', 
               'pct_bach', 'pct_black', 'pct_asian', 'log_pop', 'pct_manuf']
    available_vars = [var for var in key_vars if var in processed_data.columns]
    
    print(f"\n📊 Key Variables Available:")
    for var in available_vars:
        print(f"   • {var}")
    
    return processed_data

def descriptive_analysis(processed_data):
    """Step 3: Descriptive analysis and initial visualizations."""
    print("\n" + "=" * 60)
    print("STEP 3: DESCRIPTIVE ANALYSIS")
    print("=" * 60)
    
    # Summary statistics
    print("\n📈 Summary Statistics:")
    summary = processed_data.describe()
    print(summary)
    
    # Gender gap statistics
    print(f"\n🎯 Gender Wage Gap Statistics:")
    print(f"   • Mean gap: {processed_data['gender_gap_pct'].mean():.1f}%")
    print(f"   • Median gap: {processed_data['gender_gap_pct'].median():.1f}%")
    print(f"   • Standard deviation: {processed_data['gender_gap_pct'].std():.1f}%")
    print(f"   • Range: {processed_data['gender_gap_pct'].min():.1f}% to {processed_data['gender_gap_pct'].max():.1f}%")
    
    # Top and bottom counties
    top_county = processed_data.loc[processed_data['gender_gap_pct'].idxmax(), 'county_name']
    bottom_county = processed_data.loc[processed_data['gender_gap_pct'].idxmin(), 'county_name']
    print(f"\n🏆 County Extremes:")
    print(f"   • Highest gap: {top_county} ({processed_data['gender_gap_pct'].max():.1f}%)")
    print(f"   • Lowest gap: {bottom_county} ({processed_data['gender_gap_pct'].min():.1f}%)")
    
    # Create basic visualizations
    print(f"\n📊 Creating basic visualizations...")
    
    # Distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(processed_data['gender_gap_pct'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(processed_data['gender_gap_pct'].mean(), color='red', linestyle='--', 
               label=f'Mean: {processed_data["gender_gap_pct"].mean():.1f}%')
    ax1.axvline(processed_data['gender_gap_pct'].median(), color='green', linestyle='--', 
               label=f'Median: {processed_data["gender_gap_pct"].median():.1f}%')
    ax1.set_xlabel('Gender Wage Gap (%)')
    ax1.set_ylabel('Number of Counties')
    ax1.set_title('Distribution of Gender Wage Gap Across Counties')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(processed_data['gender_gap_pct'], patch_artist=True, 
               boxprops=dict(facecolor='lightblue'))
    ax2.set_ylabel('Gender Wage Gap (%)')
    ax2.set_title('Box Plot of Gender Wage Gap')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/gender_gap_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Basic visualizations created and saved")

def statistical_analysis(processed_data):
    """Step 4: Statistical analysis."""
    print("\n" + "=" * 60)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = WageGapAnalyzer(processed_data)
    
    # Run baseline regression
    print("\n📊 Running baseline regression...")
    baseline_model = analyzer.baseline_regression()
    
    # Run extended regression
    print("\n📊 Running extended regression...")
    extended_model = analyzer.extended_regression()
    
    # Run quantile regression
    print("\n📊 Running quantile regression...")
    quantile_results = analyzer.quantile_regression()
    
    # County clustering
    print("\n📊 Running county clustering...")
    cluster_results = analyzer.county_clustering(n_clusters=4)
    
    # Robustness checks
    print("\n📊 Running robustness checks...")
    robustness_results = analyzer.robustness_checks()
    
    # Spatial analysis
    print("\n📊 Running spatial analysis...")
    spatial_results = analyzer.spatial_autocorrelation_test()
    
    print("\n✅ Statistical analysis complete")
    
    return analyzer

def create_visualizations(processed_data, analyzer):
    """Step 5: Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("STEP 5: VISUALIZATION")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = WageGapVisualizer(processed_data)
    
    # Load geographic data
    print("\n🗺️ Loading geographic data...")
    visualizer.load_geographic_data()
    
    # Create all visualizations
    print("\n📊 Creating all visualizations...")
    all_visualizations = visualizer.create_all_visualizations(analyzer.results)
    
    print("\n✅ All visualizations created and saved to results/ directory")
    
    return visualizer

def generate_summary_report(processed_data, analyzer):
    """Step 6: Generate summary report."""
    print("\n" + "=" * 60)
    print("STEP 6: SUMMARY REPORT")
    print("=" * 60)
    
    print("\n📊 KEY FINDINGS:")
    print("-" * 40)
    
    # Gender gap statistics
    print(f"1. Gender Wage Gap Statistics:")
    print(f"   • Mean gap: {processed_data['gender_gap_pct'].mean():.1f}%")
    print(f"   • Median gap: {processed_data['gender_gap_pct'].median():.1f}%")
    print(f"   • Standard deviation: {processed_data['gender_gap_pct'].std():.1f}%")
    print(f"   • Range: {processed_data['gender_gap_pct'].min():.1f}% to {processed_data['gender_gap_pct'].max():.1f}%")
    
    # Top and bottom counties
    top_county = processed_data.loc[processed_data['gender_gap_pct'].idxmax(), 'county_name']
    bottom_county = processed_data.loc[processed_data['gender_gap_pct'].idxmin(), 'county_name']
    print(f"\n2. County Extremes:")
    print(f"   • Highest gap: {top_county} ({processed_data['gender_gap_pct'].max():.1f}%)")
    print(f"   • Lowest gap: {bottom_county} ({processed_data['gender_gap_pct'].min():.1f}%)")
    
    # Regression results
    if 'baseline' in analyzer.results:
        baseline = analyzer.results['baseline']
        print(f"\n3. Baseline Regression Results:")
        print(f"   • R²: {baseline['r_squared']:.3f}")
        print(f"   • F-statistic: {baseline['f_statistic']:.2f}")
        print(f"   • Observations: {baseline['n_observations']}")
        
        # Most significant variables
        p_values = baseline['p_values']
        significant_vars = {k: v for k, v in p_values.items() if k != 'Intercept' and v < 0.05}
        if significant_vars:
            print(f"   • Significant variables (p < 0.05): {list(significant_vars.keys())}")
    
    # Spatial analysis
    if 'spatial' in analyzer.results:
        spatial = analyzer.results['spatial']
        print(f"\n4. Spatial Analysis:")
        print(f"   • Moran's I: {spatial['moran_i']:.3f}")
        print(f"   • P-value: {spatial['moran_p_value']:.3f}")
        print(f"   • Significant spatial autocorrelation: {spatial['is_significant']}")
    
    # Clustering results
    if 'clustering' in analyzer.results:
        clusters = analyzer.results['clustering']
        print(f"\n5. County Clustering:")
        print(f"   • Number of clusters: {len(clusters)}")
        for cluster_name, stats in clusters.items():
            print(f"   • {cluster_name}: {stats['count']} counties, mean gap: {stats['mean_gender_gap']:.1f}%")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   • ../data/illinois_county_data.csv")
    print(f"   • ../data/illinois_county_processed.csv")
    print(f"   • ../results/regression_table.csv")
    print(f"   • ../results/*.png (visualizations)")
    
    print(f"\n🎯 POLICY IMPLICATIONS:")
    print(f"   • Geographic targeting needed for high-gap counties")
    print(f"   • Education and training programs recommended")
    print(f"   • Industry-specific interventions required")
    print(f"   • Regular monitoring and data collection essential")

def main():
    """Main function to run the complete analysis."""
    try:
        # Setup
        setup_environment()
        
        # Step 1: Data Collection
        raw_data = collect_data()
        
        # Step 2: Data Processing
        processed_data = process_data(raw_data)
        
        # Step 3: Descriptive Analysis
        descriptive_analysis(processed_data)
        
        # Step 4: Statistical Analysis
        analyzer = statistical_analysis(processed_data)
        
        # Step 5: Visualization
        visualizer = create_visualizations(processed_data, analyzer)
        
        # Step 6: Summary Report
        generate_summary_report(processed_data, analyzer)
        
        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        print("All results have been saved to the data/ and results/ directories.")
        print("Check the generated files for detailed analysis and visualizations.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check the error and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 