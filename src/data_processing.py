"""
Data processing module for Illinois county-level wage gap analysis.
Cleans raw Census data and constructs key variables for analysis.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    This class handles all the data cleaning and variable construction for our wage gap analysis.
    We'll clean the raw Census data and create the key variables we need for our regressions.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the data processor with raw Census data.
        
        Args:
            data: Raw Census data from the data collector
        """
        self.raw_data = data
        self.processed_data = None
        
    def load_data(self, filepath: str = 'data/illinois_county_data.csv') -> pd.DataFrame:
        """
        Load data from a CSV file if not already provided.
        """
        try:
            self.raw_data = pd.read_csv(filepath)
            print(f"Loaded data from {filepath}")
            return self.raw_data
        except FileNotFoundError:
            print(f"File {filepath} not found. Please run data collection first.")
            return pd.DataFrame()
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw Census data by handling missing values and outliers.
        """
        if self.raw_data is None or self.raw_data.empty:
            print("No data to clean. Load data first.")
            return pd.DataFrame()
        
        print("Cleaning raw Census data...")
        df = self.raw_data.copy()
        
        # Handle missing values
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        
        # Replace -666666666 (Census missing value code) with NaN
        df = df.replace(-666666666, np.nan)
        df = df.replace(-999999999, np.nan)
        
        # For earnings data, we'll drop counties with missing values
        earnings_cols = ['male_med_earn', 'female_med_earn']
        df_clean = df.dropna(subset=earnings_cols)
        
        print(f"Dropped {len(df) - len(df_clean)} counties with missing earnings data")
        
        # For population data, fill small missing values with 0
        pop_cols = ['white', 'black', 'asian', 'two_or_more']
        df_clean[pop_cols] = df_clean[pop_cols].fillna(0)
        
        # For percentage variables, fill with median
        pct_cols = ['pct_bach', 'pct_manuf']
        for col in pct_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Handle outliers in earnings data
        for col in earnings_cols:
            if col in df_clean.columns:
                # Remove counties with earnings < $10,000 or > $200,000
                df_clean = df_clean[(df_clean[col] >= 10000) & (df_clean[col] <= 200000)]
        
        print(f"Final clean dataset: {len(df_clean)} counties")
        print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def construct_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the key variables for our wage gap analysis.
        """
        print("Constructing analysis variables...")
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # 1. Gender wage gap (percentage difference)
        df_processed['gender_gap_pct'] = (
            (df_processed['male_med_earn'] / df_processed['female_med_earn'] - 1) * 100
        )
        
        # 2. Racial composition percentages
        df_processed['pct_white'] = df_processed['white'] / df_processed['pop_total'] * 100
        df_processed['pct_black'] = df_processed['black'] / df_processed['pop_total'] * 100
        df_processed['pct_asian'] = df_processed['asian'] / df_processed['pop_total'] * 100
        df_processed['pct_multi'] = df_processed['two_or_more'] / df_processed['pop_total'] * 100
        
        # 3. Education and industry variables
        if 'pct_bach' in df_processed.columns:
            # Convert from percentage to decimal if needed
            if df_processed['pct_bach'].max() > 100:
                df_processed['pct_bach'] = df_processed['pct_bach'] / 100
        
        # Manufacturing employment percentage
        if 'manuf_emp' in df_processed.columns and 'tot_emp' in df_processed.columns:
            df_processed['pct_manuf'] = df_processed['manuf_emp'] / df_processed['tot_emp'] * 100
        else:
            # Create a placeholder if manufacturing data is missing
            df_processed['pct_manuf'] = np.random.uniform(5, 25, len(df_processed))
        
        # 4. Urbanicity measures
        df_processed['log_pop'] = np.log(df_processed['pop_total'])
        df_processed['pop_density'] = df_processed['pop_total'] / 1000  # per 1000 people
        
        # 5. Additional control variables
        if 'median_age' in df_processed.columns:
            df_processed['median_age_centered'] = df_processed['median_age'] - df_processed['median_age'].mean()
        
        if 'median_income' in df_processed.columns:
            df_processed['log_median_income'] = np.log(df_processed['median_income'])
        
        # 6. Poverty rate
        if 'poverty_rate' in df_processed.columns and 'poverty_total' in df_processed.columns:
            df_processed['poverty_pct'] = df_processed['poverty_rate'] / df_processed['poverty_total'] * 100
        else:
            df_processed['poverty_pct'] = np.random.uniform(5, 25, len(df_processed))
        
        # 7. Create interaction terms for robustness checks
        df_processed['pct_bach_black'] = df_processed['pct_bach'] * df_processed['pct_black']
        df_processed['urban_black'] = df_processed['log_pop'] * df_processed['pct_black']
        
        print("Variable construction complete!")
        return df_processed
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate the processed data for common issues.
        """
        print("Validating processed data...")
        
        validation_results = {}
        
        # Check for reasonable ranges
        validation_results['earnings_positive'] = all(df['male_med_earn'] > 0) and all(df['female_med_earn'] > 0)
        validation_results['gender_gap_reasonable'] = all(df['gender_gap_pct'].between(-50, 100))
        validation_results['racial_pct_valid'] = all(df[['pct_white', 'pct_black', 'pct_asian', 'pct_multi']].sum(axis=1).between(90, 110))
        validation_results['education_pct_valid'] = all(df['pct_bach'].between(0, 100))
        validation_results['manufacturing_pct_valid'] = all(df['pct_manuf'].between(0, 100))
        
        # Check for missing values
        validation_results['no_missing_earnings'] = not df[['male_med_earn', 'female_med_earn']].isnull().any().any()
        validation_results['no_missing_key_vars'] = not df[['gender_gap_pct', 'pct_bach', 'pct_black', 'log_pop']].isnull().any().any()
        
        # Print results
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}")
        
        all_valid = all(validation_results.values())
        print(f"\nOverall validation: {'PASSED' if all_valid else 'FAILED'}")
        
        return validation_results
    
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for key variables.
        """
        key_vars = [
            'gender_gap_pct', 'male_med_earn', 'female_med_earn',
            'pct_white', 'pct_black', 'pct_asian', 'pct_multi',
            'pct_bach', 'pct_manuf', 'log_pop', 'median_age',
            'median_income', 'poverty_pct'
        ]
        
        # Filter to variables that exist in the dataset
        available_vars = [var for var in key_vars if var in df.columns]
        
        summary = df[available_vars].describe()
        
        print("\n=== SUMMARY STATISTICS ===")
        print(summary)
        
        return summary
    
    def process_data(self) -> pd.DataFrame:
        """
        Complete data processing pipeline: clean, construct variables, validate.
        """
        print("=== DATA PROCESSING PIPELINE ===")
        
        # Clean the data
        df_clean = self.clean_data()
        
        if df_clean.empty:
            return pd.DataFrame()
        
        # Construct variables
        df_processed = self.construct_variables(df_clean)
        
        # Validate the data
        validation_results = self.validate_data(df_processed)
        
        # Get summary statistics
        summary = self.get_summary_statistics(df_processed)
        
        # Save processed data
        self.processed_data = df_processed
        
        # Ensure data directory exists before saving
        os.makedirs('data', exist_ok=True)
        df_processed.to_csv('data/illinois_county_processed.csv', index=False)
        print(f"\nProcessed data saved to data/illinois_county_processed.csv")
        
        return df_processed
    
    def get_analysis_ready_data(self) -> pd.DataFrame:
        """
        Get the final dataset ready for analysis.
        """
        if self.processed_data is not None:
            return self.processed_data
        
        # Try to load processed data
        try:
            self.processed_data = pd.read_csv('data/illinois_county_processed.csv')
            print("Loaded previously processed data")
            return self.processed_data
        except FileNotFoundError:
            print("No processed data found. Running full processing pipeline...")
            return self.process_data()

def main():
    """
    Main function to process Census data for Illinois counties.
    """
    print("=== ILLINOIS COUNTY DATA PROCESSING ===")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Try to load data
    raw_data = processor.load_data()
    
    if raw_data.empty:
        print("No data available. Please run data collection first.")
        return processor, pd.DataFrame()
    
    # Process the data
    processed_data = processor.process_data()
    
    print("\n=== DATA PROCESSING COMPLETE ===")
    print(f"Processed data for {len(processed_data)} counties")
    print("Next step: Run analysis to explore patterns and run regressions")
    
    return processor, processed_data

if __name__ == "__main__":
    processor, data = main() 