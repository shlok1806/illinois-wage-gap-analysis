import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class WageGapAnalyzer:
    """
    This class handles all the statistical analysis for our wage gap study.
    We'll run regressions, check for spatial patterns, and explore heterogeneity.
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the analyzer with processed data.
        
        Args:
            data: Processed county-level data
        """
        self.data = data
        self.models = {}
        self.results = {}
        
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
    
    def baseline_regression(self) -> sm.regression.linear_model.RegressionResults:
        """
        Run the baseline OLS regression for gender wage gap determinants.
        
        Model: gender_gap_pct = α + β₁pct_bach + β₂pct_black + β₃pct_asian + 
               β₄pct_multi + β₅log_pop + β₆pct_manuf + ε
        """
        if self.data is None or self.data.empty:
            print("No data available for analysis.")
            return None
        
        print("Running baseline OLS regression...")
        
        # Define the regression formula
        formula = (
            "gender_gap_pct ~ pct_bach + pct_black + pct_asian + pct_multi "
            "+ log_pop + pct_manuf"
        )
        
        # Run the regression
        model = smf.ols(formula, data=self.data).fit()
        
        # Store the model
        self.models['baseline'] = model
        
        # Print results
        print("\n=== BASELINE REGRESSION RESULTS ===")
        print(model.summary())
        
        # Extract key statistics
        self.results['baseline'] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'n_observations': model.nobs,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'std_errors': model.bse.to_dict()
        }
        
        return model
    
    def extended_regression(self) -> sm.regression.linear_model.RegressionResults:
        """
        Run extended regression with additional controls and interaction terms.
        """
        if self.data is None or self.data.empty:
            print("No data available for analysis.")
            return None
        
        print("Running extended regression with controls...")
        
        # Extended formula with additional controls
        formula = (
            "gender_gap_pct ~ pct_bach + pct_black + pct_asian + pct_multi "
            "+ log_pop + pct_manuf + median_age_centered + log_median_income "
            "+ poverty_pct + pct_bach_black + urban_black"
        )
        
        # Run the regression
        model = smf.ols(formula, data=self.data).fit()
        
        # Store the model
        self.models['extended'] = model
        
        # Print results
        print("\n=== EXTENDED REGRESSION RESULTS ===")
        print(model.summary())
        
        # Extract key statistics
        self.results['extended'] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'n_observations': model.nobs,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'std_errors': model.bse.to_dict()
        }
        
        return model
    
    def quantile_regression(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> Dict:
        """
        Run quantile regressions to explore heterogeneity across the wage gap distribution.
        """
        if self.data is None or self.data.empty:
            print("No data available for analysis.")
            return {}
        
        print(f"Running quantile regressions for quantiles: {quantiles}")
        
        # Define variables for quantile regression
        y = self.data['gender_gap_pct']
        X_vars = ['pct_bach', 'pct_black', 'pct_asian', 'pct_multi', 'log_pop', 'pct_manuf']
        X = self.data[X_vars]
        
        # Add constant
        X = sm.add_constant(X)
        
        quantile_results = {}
        
        for q in quantiles:
            print(f"Running quantile regression for q = {q}")
            
            # Run quantile regression
            mod = QuantReg(y, X)
            res = mod.fit(q=q)
            
            quantile_results[q] = {
                'coefficients': res.params.to_dict(),
                'p_values': res.pvalues.to_dict(),
                'std_errors': res.bse.to_dict(),
                'pseudo_r_squared': res.prsquared
            }
            
            print(f"Quantile {q}: Pseudo R² = {res.prsquared:.3f}")
        
        self.results['quantile'] = quantile_results
        return quantile_results
    
    def spatial_autocorrelation_test(self) -> Dict:
        """
        Test for spatial autocorrelation in regression residuals using Moran's I.
        """
        try:
            import esda
            from libpysal.weights import W
            
            if self.data is None or self.data.empty:
                print("No data available for spatial analysis.")
                return {}
            
            print("Testing for spatial autocorrelation...")
            
            # Get residuals from baseline model
            if 'baseline' not in self.models:
                print("Run baseline regression first.")
                return {}
            
            residuals = self.models['baseline'].resid
            
            # Create spatial weights matrix (Queen contiguity)
            # For now, we'll use a simple approach - in practice you'd need county coordinates
            print("Note: Spatial analysis requires county coordinates/shapefile")
            print("Creating simplified spatial weights for demonstration...")
            
            # Create a simple spatial weights matrix (nearest neighbors)
            n = len(self.data)
            weights_dict = {}
            
            # Simple approach: connect each county to its 3 nearest neighbors
            for i in range(n):
                # Calculate distances (simplified)
                distances = np.abs(np.arange(n) - i)
                # Get 3 nearest neighbors (excluding self)
                nearest = np.argsort(distances)[1:4]
                weights_dict[i] = {j: 1.0 for j in nearest}
            
            # Convert to libpysal weights
            weights = W(weights_dict)
            
            # Calculate Moran's I
            moran = esda.moran.Moran(residuals, weights)
            
            spatial_results = {
                'moran_i': moran.I,
                'moran_p_value': moran.p_norm,
                'moran_z_score': moran.z_norm,
                'is_significant': moran.p_norm < 0.05
            }
            
            print(f"Moran's I: {moran.I:.3f}")
            print(f"P-value: {moran.p_norm:.3f}")
            print(f"Significant spatial autocorrelation: {spatial_results['is_significant']}")
            
            self.results['spatial'] = spatial_results
            return spatial_results
            
        except ImportError:
            print("Spatial analysis packages not available. Install pysal and esda.")
            return {}
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            print("Skipping spatial analysis...")
            return {}
    
    def county_clustering(self, n_clusters: int = 4) -> Dict:
        """
        Cluster counties based on key characteristics to identify patterns.
        """
        if self.data is None or self.data.empty:
            print("No data available for clustering.")
            return {}
        
        print(f"Clustering counties into {n_clusters} groups...")
        
        # Select variables for clustering
        cluster_vars = ['pct_bach', 'pct_black', 'pct_asian', 'gender_gap_pct', 'log_pop']
        available_vars = [var for var in cluster_vars if var in self.data.columns]
        
        if len(available_vars) < 2:
            print("Not enough variables available for clustering.")
            return {}
        
        # Prepare data for clustering
        X = self.data[available_vars].copy()
        
        # Handle missing values
        X = X.dropna()
        
        # Standardize variables
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.data.loc[X.index, 'cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            cluster_analysis[f'cluster_{i}'] = {
                'count': len(cluster_data),
                'counties': cluster_data['county_name'].tolist(),
                'mean_gender_gap': cluster_data['gender_gap_pct'].mean(),
                'mean_pct_bach': cluster_data['pct_bach'].mean(),
                'mean_pct_black': cluster_data['pct_black'].mean(),
                'mean_log_pop': cluster_data['log_pop'].mean()
            }
        
        print("\n=== CLUSTER ANALYSIS ===")
        for cluster_name, stats in cluster_analysis.items():
            print(f"\n{cluster_name.upper()}:")
            print(f"  Counties: {stats['count']}")
            print(f"  Mean gender gap: {stats['mean_gender_gap']:.1f}%")
            print(f"  Mean % bachelor's: {stats['mean_pct_bach']:.1f}%")
            print(f"  Mean % black: {stats['mean_pct_black']:.1f}%")
            print(f"  Sample counties: {', '.join(stats['counties'][:3])}")
        
        self.results['clustering'] = cluster_analysis
        return cluster_analysis
    
    def robustness_checks(self) -> Dict:
        """
        Perform various robustness checks on the baseline model.
        """
        if self.data is None or self.data.empty:
            print("No data available for robustness checks.")
            return {}
        
        print("Running robustness checks...")
        
        robustness_results = {}
        
        # 1. Outlier analysis
        print("1. Checking for influential observations...")
        if 'baseline' in self.models:
            model = self.models['baseline']
            
            # Cook's distance
            cooks_d = model.get_influence().cooks_distance[0]
            outlier_threshold = 4 / len(self.data)
            outliers = cooks_d > outlier_threshold
            
            robustness_results['outliers'] = {
                'n_outliers': outliers.sum(),
                'outlier_counties': self.data[outliers]['county_name'].tolist() if outliers.any() else []
            }
            
            print(f"   Found {outliers.sum()} influential observations")
        
        # 2. Heteroskedasticity test
        print("2. Testing for heteroskedasticity...")
        if 'baseline' in self.models:
            from statsmodels.stats.diagnostic import het_breuschpagan
            
            model = self.models['baseline']
            bp_stat, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
            
            robustness_results['heteroskedasticity'] = {
                'breusch_pagan_stat': bp_stat,
                'breusch_pagan_pvalue': bp_pval,
                'is_heteroskedastic': bp_pval < 0.05
            }
            
            print(f"   Breusch-Pagan test p-value: {bp_pval:.3f}")
        
        # 3. Multicollinearity check
        print("3. Checking for multicollinearity...")
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        if 'baseline' in self.models:
            model = self.models['baseline']
            X = model.model.exog
            
            vif_data = []
            for i in range(1, X.shape[1]):  # Skip constant
                vif = variance_inflation_factor(X, i)
                vif_data.append({
                    'variable': model.model.exog_names[i],
                    'vif': vif
                })
            
            robustness_results['multicollinearity'] = vif_data
            
            high_vif = [item for item in vif_data if item['vif'] > 10]
            print(f"   Variables with VIF > 10: {len(high_vif)}")
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def generate_regression_table(self) -> pd.DataFrame:
        """
        Generate a publication-ready regression table.
        """
        if not self.models:
            print("No models available. Run regressions first.")
            return pd.DataFrame()
        
        print("Generating regression table...")
        
        # Collect results from all models
        table_data = []
        
        for model_name, model in self.models.items():
            results = {
                'Model': model_name.title(),
                'Observations': int(model.nobs),
                'R-squared': f"{model.rsquared:.3f}",
                'Adj. R-squared': f"{model.rsquared_adj:.3f}"
            }
            
            # Add coefficients and standard errors
            for var in model.params.index:
                if var != 'Intercept':
                    coef = model.params[var]
                    se = model.bse[var]
                    pval = model.pvalues[var]
                    
                    # Format coefficient with significance stars
                    stars = ''
                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.1:
                        stars = '*'
                    
                    results[f'{var}_coef'] = f"{coef:.3f}{stars}"
                    results[f'{var}_se'] = f"({se:.3f})"
            
            table_data.append(results)
        
        # Create DataFrame
        table_df = pd.DataFrame(table_data)
        
        # Save table
        table_df.to_csv('results/regression_table.csv', index=False)
        print("Regression table saved to results/regression_table.csv")
        
        return table_df
    
    def run_full_analysis(self) -> Dict:
        """
        Run the complete analysis pipeline.
        """
        print("=== COMPLETE WAGE GAP ANALYSIS ===")
        
        if self.data is None or self.data.empty:
            print("No data available. Loading processed data...")
            self.load_data()
        
        if self.data is None or self.data.empty:
            print("Still no data available. Please run data processing first.")
            return {}
        
        # Run all analyses
        baseline_model = self.baseline_regression()
        extended_model = self.extended_regression()
        quantile_results = self.quantile_regression()
        spatial_results = self.spatial_autocorrelation_test()
        cluster_results = self.county_clustering()
        robustness_results = self.robustness_checks()
        
        # Generate tables
        regression_table = self.generate_regression_table()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("All results stored in self.results")
        print("Next step: Generate visualizations and reports")
        
        return self.results

def main():
    """
    Main function to run the complete wage gap analysis.
    """
    print("=== ILLINOIS COUNTY WAGE GAP ANALYSIS ===")
    
    # Initialize analyzer
    analyzer = WageGapAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    
    if data.empty:
        print("No data available. Please run data processing first.")
        return analyzer, {}
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 