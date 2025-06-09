#!/usr/bin/env python3
"""
Main script for Illinois County Wage Gap Analysis
Orchestrates the complete pipeline: data collection, processing, analysis, and visualization.
"""

import os
import sys
from src.data_collection import CensusDataCollector
from src.data_processing import DataProcessor
from src.analysis import WageGapAnalyzer
from src.visualization import WageGapVisualizer

def main():
    """
    Run the complete Illinois wage gap analysis pipeline.
    """
    print("=" * 60)
    print("ILLINOIS COUNTY WAGE GAP ANALYSIS")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        # Step 1: Data Collection
        print("\n1. COLLECTING CENSUS DATA...")
        print("-" * 40)
        collector = CensusDataCollector(year=2022, survey='acs1')
        raw_data = collector.download_county_data()
        
        if raw_data.empty:
            print("❌ Failed to collect data. Exiting.")
            return False
        
        print(f"✅ Collected data for {len(raw_data)} counties")
        
        # Step 2: Data Processing
        print("\n2. PROCESSING DATA...")
        print("-" * 40)
        processor = DataProcessor(raw_data)
        processed_data = processor.process_data()
        
        if processed_data.empty:
            print("❌ Failed to process data. Exiting.")
            return False
        
        print(f"✅ Processed data for {len(processed_data)} counties")
        
        # Step 3: Statistical Analysis
        print("\n3. RUNNING STATISTICAL ANALYSIS...")
        print("-" * 40)
        analyzer = WageGapAnalyzer(processed_data)
        results = analyzer.run_full_analysis()
        
        if not results:
            print("❌ Failed to complete analysis. Exiting.")
            return False
        
        print("✅ Completed statistical analysis")
        
        # Step 4: Visualization
        print("\n4. CREATING VISUALIZATIONS...")
        print("-" * 40)
        visualizer = WageGapVisualizer(processed_data)
        visualizer.load_geographic_data()
        visualizations = visualizer.create_all_visualizations(results)
        
        if not visualizations:
            print("❌ Failed to create visualizations.")
        else:
            print("✅ Created all visualizations")
        
        # Step 5: Summary
        print("\n5. ANALYSIS COMPLETE!")
        print("-" * 40)
        print("📊 Summary of Results:")
        
        # Key statistics
        if 'baseline' in results:
            baseline = results['baseline']
            print(f"   • Gender gap mean: {processed_data['gender_gap_pct'].mean():.1f}%")
            print(f"   • Regression R²: {baseline['r_squared']:.3f}")
            print(f"   • Counties analyzed: {baseline['n_observations']}")
        
        # Spatial analysis
        if 'spatial' in results:
            spatial = results['spatial']
            print(f"   • Spatial autocorrelation: {'Yes' if spatial['is_significant'] else 'No'}")
        
        # Clustering
        if 'clustering' in results:
            clusters = results['clustering']
            print(f"   • County clusters identified: {len(clusters)}")
        
        print(f"\n📁 Files created:")
        print(f"   • data/illinois_county_data.csv")
        print(f"   • data/illinois_county_processed.csv")
        print(f"   • results/regression_table.csv")
        print(f"   • results/*.png (visualizations)")
        
        print(f"\n🎯 Next steps:")
        print(f"   • Open notebooks/illinois_wage_gap_analysis.ipynb for interactive analysis")
        print(f"   • Check results/ directory for all outputs")
        print(f"   • Review regression_table.csv for detailed results")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check the error and try again.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n💥 Analysis failed. Please check the errors above.")
        sys.exit(1) 