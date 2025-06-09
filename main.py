#!/usr/bin/env python3
"""
Mapping Inequality: County-Level Gender and Racial Earnings Disparities in Illinois
Main script for running the complete economic analysis pipeline.
"""

import os
import sys
from src.data_collection import CensusDataCollector
from src.data_processing import DataProcessor
from src.analysis import WageGapAnalyzer
from src.economic_analysis import EconomicWageGapAnalyzer
from src.visualization import WageGapVisualizer

def main():
    print("=" * 80)
    print("MAPPING INEQUALITY: ILLINOIS WAGE DISPARITIES - ECONOMIC ANALYSIS")
    print("=" * 80)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        print("\n1. COLLECTING CENSUS DATA...")
        print("-" * 50)
        collector = CensusDataCollector(year=2022, survey='acs1')
        raw_data = collector.download_county_data()
        
        if raw_data.empty:
            print("❌ Failed to collect data. Exiting.")
            return False
        
        print(f"✅ Collected data for {len(raw_data)} counties")
        
        print("\n2. PROCESSING DATA...")
        print("-" * 50)
        processor = DataProcessor(raw_data)
        processed_data = processor.process_data()
        
        if processed_data.empty:
            print("❌ Failed to process data. Exiting.")
            return False
        
        print(f"✅ Processed data for {len(processed_data)} counties")
        
        print("\n3. RUNNING DESCRIPTIVE STATISTICAL ANALYSIS...")
        print("-" * 50)
        analyzer = WageGapAnalyzer(processed_data)
        results = analyzer.run_full_analysis()
        
        if not results:
            print("❌ Failed to complete descriptive analysis. Exiting.")
            return False
        
        print("✅ Completed descriptive statistical analysis")
        
        print("\n4. RUNNING ECONOMIC ANALYSIS...")
        print("-" * 50)
        economic_analyzer = EconomicWageGapAnalyzer(processed_data)
        economic_results = economic_analyzer.run_complete_economic_analysis()
        
        if not economic_results:
            print("❌ Failed to complete economic analysis.")
        else:
            print("✅ Completed economic analysis")
        
        print("\n5. CREATING VISUALIZATIONS...")
        print("-" * 50)
        visualizer = WageGapVisualizer(processed_data)
        visualizer.load_geographic_data()
        visualizations = visualizer.create_all_visualizations(results)
        
        if not visualizations:
            print("❌ Failed to create visualizations.")
        else:
            print("✅ Created all visualizations")
        
        print("\n6. ECONOMIC ANALYSIS COMPLETE!")
        print("-" * 50)
        print("📊 Summary of Economic Results:")
        
        # Mincer equation results
        if 'mincer_equation' in economic_results:
            mincer = economic_results['mincer_equation']
            print(f"   • Mincer equation R²: {mincer['r_squared']:.3f}")
            print(f"   • Education elasticity: {mincer['elasticities']['education']:.3f}")
            print(f"   • Experience elasticity: {mincer['elasticities']['experience']:.3f}")
            print(f"   • Gender discrimination elasticity: {mincer['elasticities']['gender_discrimination']:.3f}")
        
        # IV analysis results
        if 'iv_analysis' in economic_results:
            iv = economic_results['iv_analysis']
            print(f"   • IV first stage F-statistic: {iv['first_stage']['f_statistic']:.2f}")
            print(f"   • IV education coefficient: {iv['second_stage']['coefficients']['education_hat']:.3f}")
        
        # Natural experiment results
        if 'natural_experiments' in economic_results:
            ne = economic_results['natural_experiments']
            print(f"   • Minimum wage treatment effect: {ne['minimum_wage']['treatment_effect']:.3f}")
            print(f"   • Manufacturing treatment effect: {ne['manufacturing']['treatment_effect']:.3f}")
        
        # Structural interpretation
        if 'structural_interpretation' in economic_results:
            structural = economic_results['structural_interpretation']
            if 'welfare' in structural:
                print(f"   • Estimated deadweight loss: ${structural['welfare']['deadweight_loss']:,.0f}")
        
        # Policy simulations
        if 'policy_simulation' in economic_results:
            policy = economic_results['policy_simulation']
            print(f"   • {policy['policy_type'].title()} effect: {policy['effects']['interpretation']}")
        
        print(f"\n📁 Files created:")
        print(f"   • data/illinois_county_data.csv")
        print(f"   • data/illinois_county_processed.csv")
        print(f"   • results/regression_table.csv")
        print(f"   • results/economic_analysis_results.json")
        print(f"   • results/*.png (visualizations)")
        
        print(f"\n📚 Research Framework:")
        print(f"   • ECONOMIC_FRAMEWORK.md - Theoretical foundation")
        print(f"   • LITERATURE_REVIEW.md - Literature review")
        print(f"   • src/economic_analysis.py - Economic analysis methods")
        
        print(f"\n🎯 Next steps:")
        print(f"   • Check results/ directory for all outputs")
        print(f"   • Review economic_analysis_results.json for detailed results")
        print(f"   • Examine ECONOMIC_FRAMEWORK.md for theoretical foundation")
        print(f"   • Read LITERATURE_REVIEW.md for academic context")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check the error and try again.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Economic analysis completed successfully!")
        print("This research now includes:")
        print("  ✓ Theoretical framework (Mincer equation with discrimination)")
        print("  ✓ Literature review and research positioning")
        print("  ✓ Identification strategy (IV, natural experiments)")
        print("  ✓ Structural interpretation (elasticities, welfare)")
        print("  ✓ Policy analysis (simulations, counterfactuals)")
    else:
        print("\n💥 Analysis failed. Please check the errors above.")
        sys.exit(1) 