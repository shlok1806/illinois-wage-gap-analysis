#!/usr/bin/env python3
"""
Illinois County Wage Gap Analysis - Streamlit App
Interactive web application for analyzing gender and racial wage gaps across Illinois counties.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Import our custom modules
from data_collection import CensusDataCollector
from data_processing import DataProcessor
from analysis import WageGapAnalyzer
from visualization import WageGapVisualizer

# Page configuration
st.set_page_config(
    page_title="Illinois Wage Gap Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    .stDataFrame {
        border-radius: 10px;
    }
    
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data."""
    try:
        # Try to load processed data first
        data_path = 'data/illinois_county_processed.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            st.success(f"✅ Loaded processed data: {len(data)} counties")
            return data
        
        # If processed data doesn't exist, try raw data
        raw_data_path = 'data/illinois_county_data.csv'
        if os.path.exists(raw_data_path):
            raw_data = pd.read_csv(raw_data_path)
            processor = DataProcessor(raw_data)
            data = processor.process_data()
            st.success(f"✅ Processed raw data: {len(data)} counties")
            return data
        
        # If no data exists, create sample data
        st.warning("No data found. Creating sample data for demonstration...")
        collector = CensusDataCollector()
        data = collector.create_sample_data()
        processor = DataProcessor(data)
        processed_data = processor.process_data()
        st.success(f"✅ Created sample data: {len(processed_data)} counties")
        return processed_data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def run_analysis(data):
    """Run statistical analysis and cache results."""
    try:
        analyzer = WageGapAnalyzer(data)
        
        # Run analyses with error handling and progress tracking
        with st.spinner('Running baseline regression...'):
            try:
                analyzer.baseline_regression()
                st.success("✅ Baseline regression complete")
            except Exception as e:
                st.warning(f"⚠️ Baseline regression failed: {e}")
        
        with st.spinner('Running extended regression...'):
            try:
                analyzer.extended_regression()
                st.success("✅ Extended regression complete")
            except Exception as e:
                st.warning(f"⚠️ Extended regression failed: {e}")
        
        with st.spinner('Running quantile regression...'):
            try:
                analyzer.quantile_regression()
                st.success("✅ Quantile regression complete")
            except Exception as e:
                st.warning(f"⚠️ Quantile regression failed: {e}")
        
        with st.spinner('Running county clustering...'):
            try:
                analyzer.county_clustering()
                st.success("✅ County clustering complete")
            except Exception as e:
                st.warning(f"⚠️ County clustering failed: {e}")
        
        with st.spinner('Running robustness checks...'):
            try:
                analyzer.robustness_checks()
                st.success("✅ Robustness checks complete")
            except Exception as e:
                st.warning(f"⚠️ Robustness checks failed: {e}")
        
        with st.spinner('Running spatial analysis...'):
            try:
                analyzer.spatial_autocorrelation_test()
                st.success("✅ Spatial analysis complete")
            except Exception as e:
                st.warning(f"⚠️ Spatial analysis failed: {e}")
        
        return analyzer
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        return None

def create_enhanced_visualizations(data, analyzer):
    """Create enhanced visualizations with better styling."""
    try:
        # Gender gap distribution with enhanced styling
        fig_dist = px.histogram(
            data, 
            x='gender_gap_pct', 
            nbins=15,
            title="Distribution of Gender Wage Gap Across Counties",
            labels={'gender_gap_pct': 'Gender Wage Gap (%)', 'count': 'Number of Counties'},
            color_discrete_sequence=['#3498db']
        )
        fig_dist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            title_font_size=16
        )
        fig_dist.add_vline(x=data['gender_gap_pct'].mean(), line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {data['gender_gap_pct'].mean():.1f}%")
        
        # Scatter plot matrix
        fig_scatter = px.scatter_matrix(
            data,
            dimensions=['gender_gap_pct', 'pct_bach', 'pct_black', 'log_pop'],
            color='pct_manuf',
            hover_data=['county_name'],
            title="Variable Relationships Matrix",
            color_continuous_scale='Viridis'
        )
        fig_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=10),
            title_font_size=16
        )
        
        return fig_dist, fig_scatter
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None, None

def main():
    """Main Streamlit app."""
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">📊 Illinois County Wage Gap Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar with enhanced navigation
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📈 Data Explorer", "📊 Statistical Analysis", "🗺️ Geographic Analysis", "📋 Results Summary", "🎯 Policy Insights"]
    )
    
    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check the data collection process.")
        return
    
    # Show quick stats in sidebar
    if data is not None:
        st.sidebar.metric("Counties", len(data))
        st.sidebar.metric("Mean Gap", f"{data['gender_gap_pct'].mean():.1f}%")
        st.sidebar.metric("Max Gap", f"{data['gender_gap_pct'].max():.1f}%")
        st.sidebar.metric("Min Gap", f"{data['gender_gap_pct'].min():.1f}%")
    
    # Initialize analyzer for pages that need it
    analyzer = None
    if page in ["📊 Statistical Analysis", "🗺️ Geographic Analysis", "📋 Results Summary", "🎯 Policy Insights"]:
        with st.spinner('Running statistical analysis...'):
            analyzer = run_analysis(data)
        
        if analyzer is None:
            st.error("Failed to run analysis. Some features may not be available.")
            analyzer = WageGapAnalyzer(data)  # Create empty analyzer as fallback
    
    # Overview Page
    if page == "🏠 Overview":
        st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
        
        # Project description with enhanced layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Research Questions
            1. **What is the percentage difference between male and female median earnings in each Illinois county?**
            2. **How do median earnings for different racial groups compare across counties?**
            3. **Which county characteristics explain cross-county differences in wage gaps?**
            
            ### 📊 Data Sources
            - **ACS 2022 1-Year county estimates** via Census API
            - **Median earnings by sex**, population by race, education, industry mix
            - **Geographic and demographic variables** for comprehensive analysis
            
            ### 🔬 Methodology
            - **Descriptive analysis** and visualization
            - **OLS regression analysis** with multiple specifications
            - **Quantile regression** for heterogeneity exploration
            - **Spatial autocorrelation** testing
            - **County clustering** analysis
            - **Robustness checks** and validation
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Key Variables
            - **Gender wage gap (%)** - Primary outcome variable
            - **Education (% bachelor's)** - Human capital proxy
            - **Racial composition (%)** - Demographic controls
            - **Population (log)** - Urban/rural indicator
            - **Manufacturing employment (%)** - Industry mix
            - **Median income (log)** - Economic conditions
            - **Poverty rate (%)** - Socioeconomic status
            """)
        
        # Key metrics with enhanced styling
        st.markdown('<h3>📊 Key Statistics</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Counties Analyzed",
                len(data),
                help="Total number of Illinois counties in the analysis"
            )
        
        with col2:
            mean_gap = data['gender_gap_pct'].mean()
            st.metric(
                "Mean Gender Gap",
                f"{mean_gap:.1f}%",
                help="Average gender wage gap across all counties"
            )
        
        with col3:
            max_gap = data['gender_gap_pct'].max()
            st.metric(
                "Highest Gap",
                f"{max_gap:.1f}%",
                help="Maximum gender wage gap observed"
            )
        
        with col4:
            min_gap = data['gender_gap_pct'].min()
            st.metric(
                "Lowest Gap",
                f"{min_gap:.1f}%",
                help="Minimum gender wage gap observed"
            )
        
        # Quick insights with visualizations
        st.markdown('<h3>💡 Quick Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender gap distribution
            fig_gap = px.histogram(
                data, 
                x='gender_gap_pct', 
                nbins=15,
                title="Gender Wage Gap Distribution",
                labels={'gender_gap_pct': 'Gender Wage Gap (%)', 'count': 'Number of Counties'},
                color_discrete_sequence=['#3498db']
            )
            fig_gap.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=14
            )
            fig_gap.add_vline(x=mean_gap, line_dash="dash", line_color="red", 
                            annotation_text=f"Mean: {mean_gap:.1f}%")
            st.plotly_chart(fig_gap, use_container_width=True)
        
        with col2:
            # Education vs gender gap
            fig_edu = px.scatter(
                data, 
                x='pct_bach', 
                y='gender_gap_pct',
                hover_data=['county_name'],
                title="Education vs Gender Wage Gap",
                labels={'pct_bach': 'Education (% Bachelor\'s)', 'gender_gap_pct': 'Gender Wage Gap (%)'},
                color='pct_black',
                color_continuous_scale='Viridis'
            )
            fig_edu.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=14
            )
            st.plotly_chart(fig_edu, use_container_width=True)
        
        # Data download section
        st.markdown('<h3>📥 Data Download</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download processed data
            csv = data.to_csv(index=False)
            st.download_button(
                label="📊 Download Processed Data (CSV)",
                data=csv,
                file_name="illinois_wage_gap_data.csv",
                mime="text/csv",
                help="Download the processed dataset used in the analysis"
            )
        
        with col2:
            # Download summary statistics
            summary_stats = data.describe().round(3)
            csv_summary = summary_stats.to_csv()
            st.download_button(
                label="📋 Download Summary Stats (CSV)",
                data=csv_summary,
                file_name="summary_statistics.csv",
                mime="text/csv",
                help="Download summary statistics for all variables"
            )
        
        with col3:
            # Download county rankings
            rankings = data[['county_name', 'gender_gap_pct', 'pct_bach', 'pct_black', 'log_pop']].sort_values('gender_gap_pct', ascending=False)
            csv_rankings = rankings.to_csv(index=False)
            st.download_button(
                label="🏆 Download County Rankings (CSV)",
                data=csv_rankings,
                file_name="county_rankings.csv",
                mime="text/csv",
                help="Download county rankings by gender wage gap"
            )
        
        # Project information
        st.markdown('<h3>ℹ️ Project Information</h3>', unsafe_allow_html=True)
        
        with st.expander("📚 Technical Details"):
            st.markdown("""
            **Analysis Framework:**
            - **Statistical Software**: Python with pandas, statsmodels, scikit-learn
            - **Visualization**: Plotly for interactive charts
            - **Spatial Analysis**: PySAL for spatial autocorrelation
            - **Web Interface**: Streamlit for interactive dashboard
            
            **Data Processing:**
            - **Cleaning**: Removal of missing values, outlier detection
            - **Transformation**: Log transformations, percentage calculations
            - **Validation**: Data quality checks and consistency validation
            
            **Statistical Methods:**
            - **OLS Regression**: Baseline and extended specifications
            - **Quantile Regression**: Heterogeneity across wage gap distribution
            - **Spatial Analysis**: Moran's I for spatial autocorrelation
            - **Clustering**: K-means for county grouping
            - **Robustness**: Influential observations, heteroskedasticity, multicollinearity
            """)
        
        with st.expander("🔗 References and Resources"):
            st.markdown("""
            **Data Sources:**
            - U.S. Census Bureau American Community Survey (ACS)
            - Census API for automated data retrieval
            
            **Methodology References:**
            - Wooldridge, J.M. (2019). Introductory Econometrics: A Modern Approach
            - Anselin, L. (1988). Spatial Econometrics: Methods and Models
            
            **Technical Documentation:**
            - Full documentation available in `DOCUMENTATION.md`
            - Code repository with complete analysis pipeline
            - Reproducible research framework
            """)
    
    # Data Explorer Page
    elif page == "📈 Data Explorer":
        st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
        
        # Data overview with enhanced metrics
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Counties", len(data))
        with col2:
            st.metric("Variables", len(data.columns))
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Variable selection with enhanced interface
        st.subheader("📈 Variable Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_var = st.selectbox(
                "Select variable to analyze:",
                ['gender_gap_pct', 'male_med_earn', 'female_med_earn', 'pct_bach', 'pct_black', 'pct_asian', 'log_pop', 'pct_manuf'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            plot_type = st.selectbox(
                "Plot type:",
                ["Histogram", "Box Plot", "Scatter Plot", "Distribution"]
            )
        
        with col3:
            color_by = st.selectbox(
                "Color by:",
                ["None", "pct_bach", "pct_black", "pct_manuf", "log_pop"]
            )
        
        # Create enhanced plots
        if plot_type == "Histogram":
            # For histogram, we can only use discrete color, not continuous
            if color_by != "None":
                fig = px.histogram(
                    data, 
                    x=selected_var, 
                    nbins=20, 
                    title=f"Distribution of {selected_var.replace('_', ' ').title()}",
                    color=color_by,
                    barmode='overlay'
                )
            else:
                fig = px.histogram(
                    data, 
                    x=selected_var, 
                    nbins=20, 
                    title=f"Distribution of {selected_var.replace('_', ' ').title()}"
                )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            fig = px.box(
                data, 
                y=selected_var, 
                title=f"Box Plot of {selected_var.replace('_', ' ').title()}",
                color=color_by if color_by != "None" else None
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Scatter Plot":
            x_var = st.selectbox(
                "Select X variable:", 
                data.columns, 
                index=data.columns.get_loc('pct_bach'),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            fig = px.scatter(
                data, 
                x=x_var, 
                y=selected_var, 
                hover_data=['county_name'],
                color=color_by if color_by != "None" else None,
                color_continuous_scale='Viridis' if color_by != "None" else None,
                title=f"{selected_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}"
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Distribution":
            # Create a comprehensive distribution plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Summary Stats'),
                specs=[[{"type": "histogram"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=data[selected_var], name="Histogram"),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=data[selected_var], name="Box Plot"),
                row=1, col=2
            )
            
            # Q-Q plot (simplified)
            sorted_data = np.sort(data[selected_var])
            theoretical_quantiles = np.quantile(sorted_data, np.linspace(0, 1, len(sorted_data)))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name="Q-Q Plot"),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"Comprehensive Distribution Analysis: {selected_var.replace('_', ' ').title()}",
                showlegend=False,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced summary statistics
        st.subheader("📋 Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic stats
            stats_df = data[selected_var].describe()
            st.write("**Basic Statistics:**")
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Additional stats
            additional_stats = {
                'Skewness': data[selected_var].skew(),
                'Kurtosis': data[selected_var].kurtosis(),
                'Missing Values': data[selected_var].isnull().sum(),
                'Unique Values': data[selected_var].nunique()
            }
            st.write("**Additional Statistics:**")
            for stat, value in additional_stats.items():
                st.metric(stat, f"{value:.3f}" if isinstance(value, float) else value)
        
        # County rankings with enhanced display
        st.subheader("🏆 County Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Top 10 Counties by {selected_var.replace('_', ' ').title()}:**")
            top_counties = data.nlargest(10, selected_var)[['county_name', selected_var]]
            top_counties.columns = ['County', selected_var.replace('_', ' ').title()]
            # Ensure consistent data types to avoid PyArrow issues
            top_counties[selected_var.replace('_', ' ').title()] = top_counties[selected_var.replace('_', ' ').title()].astype(str)
            st.dataframe(top_counties, use_container_width=True)
        
        with col2:
            st.write(f"**Bottom 10 Counties by {selected_var.replace('_', ' ').title()}:**")
            bottom_counties = data.nsmallest(10, selected_var)[['county_name', selected_var]]
            bottom_counties.columns = ['County', selected_var.replace('_', ' ').title()]
            # Ensure consistent data types to avoid PyArrow issues
            bottom_counties[selected_var.replace('_', ' ').title()] = bottom_counties[selected_var.replace('_', ' ').title()].astype(str)
            st.dataframe(bottom_counties, use_container_width=True)
        
        # Interactive county selector
        st.subheader("🔍 County Details")
        selected_county = st.selectbox(
            "Select a county for detailed view:",
            data['county_name'].sort_values()
        )
        
        if selected_county:
            county_data = data[data['county_name'] == selected_county].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gender Gap", f"{county_data['gender_gap_pct']:.1f}%")
            with col2:
                st.metric("Education (% Bachelor's)", f"{county_data['pct_bach']:.1f}%")
            with col3:
                st.metric("Population (log)", f"{county_data['log_pop']:.2f}")
            with col4:
                st.metric("Manufacturing (%)", f"{county_data['pct_manuf']:.1f}%")
    
    # Statistical Analysis Page
    elif page == "📊 Statistical Analysis":
        st.markdown('<h2 class="section-header">Statistical Analysis</h2>', unsafe_allow_html=True)
        
        # Regression results
        st.subheader("Regression Analysis")
        
        if analyzer and hasattr(analyzer, 'results') and 'baseline' in analyzer.results:
            baseline = analyzer.results['baseline']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R²", f"{baseline['r_squared']:.3f}")
            
            with col2:
                st.metric("Adjusted R²", f"{baseline['adj_r_squared']:.3f}")
            
            with col3:
                st.metric("Observations", baseline['n_observations'])
            
            # Coefficient plot
            coef_data = []
            for var, coef in baseline['coefficients'].items():
                if var != 'Intercept':
                    coef_data.append({
                        'Variable': var.replace('_', ' ').title(),
                        'Coefficient': coef,
                        'Std Error': baseline['std_errors'][var],
                        'P-value': baseline['p_values'][var]
                    })
            
            coef_df = pd.DataFrame(coef_data)
            # Ensure consistent data types to avoid PyArrow issues
            coef_df['Coefficient'] = coef_df['Coefficient'].astype(str)
            coef_df['Std Error'] = coef_df['Std Error'].astype(str)
            coef_df['P-value'] = coef_df['P-value'].astype(str)
            st.dataframe(coef_df, use_container_width=True)
            
            # Coefficient visualization
            fig = px.bar(coef_df, x='Variable', y='Coefficient', 
                        error_y='Std Error', title="Regression Coefficients")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Baseline regression results not available.")
        
        # Quantile regression
        if analyzer and hasattr(analyzer, 'results') and 'quantile' in analyzer.results:
            st.subheader("Quantile Regression Results")
            
            quantile_data = []
            for q, results in analyzer.results['quantile'].items():
                for var, coef in results['coefficients'].items():
                    if var != 'const':
                        quantile_data.append({
                            'Quantile': q,
                            'Variable': var.replace('_', ' ').title(),
                            'Coefficient': coef,
                            'P-value': results['p_values'][var]
                        })
            
            quantile_df = pd.DataFrame(quantile_data)
            # Ensure consistent data types to avoid PyArrow issues
            quantile_df['Coefficient'] = quantile_df['Coefficient'].astype(str)
            quantile_df['P-value'] = quantile_df['P-value'].astype(str)
            
            # Heatmap of coefficients
            pivot_df = quantile_df.pivot(index='Variable', columns='Quantile', values='Coefficient')
            fig = px.imshow(pivot_df, title="Quantile Regression Coefficients Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Quantile regression results not available.")
    
    # Geographic Analysis Page
    elif page == "🗺️ Geographic Analysis":
        st.markdown('<h2 class="section-header">Geographic Analysis</h2>', unsafe_allow_html=True)
        
        # Choropleth map (simplified)
        st.subheader("Gender Wage Gap by County")
        
        # Create a simple geographic visualization
        fig = px.scatter(data, x='log_pop', y='gender_gap_pct', 
                        size='pct_bach', color='pct_black',
                        hover_data=['county_name'],
                        title="Gender Wage Gap vs Population (bubble size = education, color = % black)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Spatial analysis results
        if analyzer and hasattr(analyzer, 'results') and 'spatial' in analyzer.results:
            st.subheader("Spatial Autocorrelation")
            
            spatial = analyzer.results['spatial']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Moran's I", f"{spatial['moran_i']:.3f}")
            
            with col2:
                st.metric("P-value", f"{spatial['moran_p_value']:.3f}")
            
            with col3:
                st.metric("Significant", "Yes" if spatial['is_significant'] else "No")
        else:
            st.info("Spatial analysis results not available.")
        
        # Clustering results
        if analyzer and hasattr(analyzer, 'results') and 'clustering' in analyzer.results:
            st.subheader("County Clustering")
            
            clusters = analyzer.results['clustering']
            
            cluster_data = []
            for cluster_name, stats in clusters.items():
                cluster_data.append({
                    'Cluster': cluster_name,
                    'Counties': stats['count'],
                    'Mean Gap': f"{stats['mean_gender_gap']:.1f}%",
                    'Mean Education': f"{stats['mean_pct_bach']:.1f}%",
                    'Sample Counties': ', '.join(stats['counties'][:3])
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            # Ensure consistent data types to avoid PyArrow issues
            cluster_df['Counties'] = cluster_df['Counties'].astype(str)
            st.dataframe(cluster_df, use_container_width=True)
        else:
            st.info("Clustering results not available.")
    
    # Results Summary Page
    elif page == "📋 Results Summary":
        st.markdown('<h2 class="section-header">Results Summary</h2>', unsafe_allow_html=True)
        
        # Key findings
        st.subheader("Key Findings")
        
        findings = [
            "**Geographic Variation**: Significant variation in gender wage gaps across Illinois counties",
            "**Education Effect**: Higher education levels generally associated with different gap patterns",
            "**Racial Composition**: Racial demographics show complex relationships with wage gaps",
            "**Urban-Rural Divide**: Population density affects wage gap patterns",
            "**Industry Mix**: Manufacturing employment shows varying effects across counties"
        ]
        
        for finding in findings:
            st.markdown(f"• {finding}")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        if analyzer and hasattr(analyzer, 'results') and 'baseline' in analyzer.results:
            baseline = analyzer.results['baseline']
            
            # Fix the data type issue by ensuring all values are strings
            summary_data = {
                'Metric': ['Mean Gender Gap', 'Regression R²', 'Observations', 'Significant Variables'],
                'Value': [
                    f"{data['gender_gap_pct'].mean():.1f}%",
                    f"{baseline['r_squared']:.3f}",
                    str(baseline['n_observations']),
                    str(len([k for k, v in baseline['p_values'].items() if k != 'Intercept' and v < 0.05]))
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            # Display with better formatting
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Gender Gap", f"{data['gender_gap_pct'].mean():.1f}%")
                st.metric("Regression R²", f"{baseline['r_squared']:.3f}")
            
            with col2:
                st.metric("Observations", baseline['n_observations'])
                st.metric("Significant Variables", 
                         len([k for k, v in baseline['p_values'].items() if k != 'Intercept' and v < 0.05]))
            
            # Show detailed results in an expander
            with st.expander("📊 Detailed Regression Results"):
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("Baseline regression results not available.")
        
        # Enhanced robustness checks display
        if analyzer and hasattr(analyzer, 'results') and 'robustness' in analyzer.results:
            st.subheader("Robustness Checks")
            
            robustness = analyzer.results['robustness']
            
            # Create a more visual display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'influential_observations' in robustness:
                    st.metric("Influential Observations", robustness['influential_observations'])
                else:
                    st.metric("Influential Observations", "N/A")
            
            with col2:
                if 'breusch_pagan_pvalue' in robustness:
                    bp_pval = robustness['breusch_pagan_pvalue']
                    st.metric("Heteroskedasticity Test", f"{bp_pval:.3f}")
                    if bp_pval < 0.05:
                        st.error("Heteroskedasticity detected")
                    else:
                        st.success("No heteroskedasticity")
                else:
                    st.metric("Heteroskedasticity Test", "N/A")
            
            with col3:
                if 'high_vif_vars' in robustness:
                    high_vif_count = len(robustness['high_vif_vars'])
                    st.metric("High VIF Variables", high_vif_count)
                    if high_vif_count > 0:
                        st.warning("Multicollinearity detected")
                    else:
                        st.success("No multicollinearity")
                else:
                    st.metric("High VIF Variables", "N/A")
            
            # Detailed results in expander
            with st.expander("🔍 Detailed Robustness Results"):
                for check, results in robustness.items():
                    st.write(f"**{check.replace('_', ' ').title()}:**")
                    if isinstance(results, dict):
                        for key, value in results.items():
                            st.write(f"  - {key.replace('_', ' ').title()}: {value}")
                    else:
                        st.write(f"  - {results}")
        else:
            st.info("Robustness check results not available.")
    
    # Policy Insights Page
    elif page == "🎯 Policy Insights":
        st.markdown('<h2 class="section-header">Policy Insights</h2>', unsafe_allow_html=True)
        
        # Executive summary
        st.markdown("""
        ### 📋 Executive Summary
        This analysis reveals significant variation in gender wage gaps across Illinois counties, 
        with gaps ranging from -56% to 231%. The findings suggest that targeted policy interventions 
        are needed to address wage disparities, with different strategies required for different 
        geographic and demographic contexts.
        """)
        
        st.subheader("🎯 Key Policy Recommendations")
        
        # Geographic targeting
        st.markdown("""
        ### 1. 🌍 Geographic Targeting
        - **Focus on high-gap counties**: Target interventions in counties with the largest wage gaps
        - **Urban-rural considerations**: Different strategies needed for urban vs. rural areas
        - **Regional coordination**: Address spatial clustering of wage disparities
        - **County-specific programs**: Develop tailored interventions based on local characteristics
        """)
        
        # Education and training
        st.markdown("""
        ### 2. 🎓 Education and Training
        - **STEM education**: Increase access to STEM education in high-gap counties
        - **Vocational training**: Provide targeted training for women in male-dominated industries
        - **Higher education access**: Improve access to higher education in underserved areas
        - **Skills development**: Focus on in-demand skills and career pathways
        """)
        
        # Industry-specific interventions
        st.markdown("""
        ### 3. 🏭 Industry-Specific Interventions
        - **Pay transparency**: Implement pay transparency policies in key industries
        - **Leadership development**: Support women's advancement in leadership roles
        - **Workplace culture**: Address workplace culture and bias in male-dominated sectors
        - **Equal opportunity**: Ensure equal access to high-paying positions
        """)
        
        # Data and monitoring
        st.markdown("""
        ### 4. 📊 Data and Monitoring
        - **Regular monitoring**: Establish regular monitoring of county-level wage gaps
        - **Progress tracking**: Track the effectiveness of intervention programs
        - **Evidence-based policy**: Use data to inform future policy decisions
        - **Transparency**: Public reporting of wage gap metrics by county
        """)
        
        # Enhanced policy impact simulator
        st.subheader("🔮 Policy Impact Simulator")
        st.markdown("""
        Use this interactive tool to simulate the potential impact of different policy interventions 
        on the gender wage gap. Adjust the sliders below to see how changes in key variables 
        might affect wage disparities.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Policy Interventions**")
            education_increase = st.slider(
                "Education increase (% points)", 
                0, 10, 2,
                help="Increase in percentage of population with bachelor's degree"
            )
            manufacturing_change = st.slider(
                "Manufacturing employment change (% points)", 
                -5, 5, 0,
                help="Change in manufacturing employment as percentage of total employment"
            )
            transparency_effect = st.slider(
                "Pay transparency effect (% points)", 
                0, 15, 5,
                help="Estimated reduction in wage gap due to pay transparency policies"
            )
            training_effect = st.slider(
                "Training program effect (% points)", 
                0, 20, 8,
                help="Estimated reduction in wage gap due to targeted training programs"
            )
        
        with col2:
            st.markdown("**📊 Predicted Impact**")
            
            if analyzer and hasattr(analyzer, 'results') and 'baseline' in analyzer.results:
                baseline = analyzer.results['baseline']
                
                # Get coefficients
                pct_bach_coef = baseline['coefficients'].get('pct_bach', 0)
                pct_manuf_coef = baseline['coefficients'].get('pct_manuf', 0)
                
                # Calculate predicted changes
                education_effect = education_increase * pct_bach_coef
                manufacturing_effect = manufacturing_change * pct_manuf_coef
                total_policy_effect = education_effect + manufacturing_effect - transparency_effect - training_effect
                
                # Current vs predicted gap
                current_gap = data['gender_gap_pct'].mean()
                predicted_gap = current_gap + total_policy_effect
                
                st.metric("Current Mean Gap", f"{current_gap:.1f}%")
                st.metric("Predicted Mean Gap", f"{predicted_gap:.1f}%")
                st.metric("Net Change", f"{total_policy_effect:.2f} percentage points")
                
                # Policy effectiveness indicator
                if total_policy_effect < -5:
                    st.success("🎉 Excellent! This combination would significantly reduce the gender wage gap!")
                elif total_policy_effect < 0:
                    st.success("✅ Good! This combination would reduce the gender wage gap.")
                elif total_policy_effect < 5:
                    st.warning("⚠️ Caution! This combination might slightly increase the gap.")
                else:
                    st.error("❌ Warning! This combination could increase the gender wage gap.")
                
                # Detailed breakdown
                with st.expander("📋 Detailed Impact Breakdown"):
                    st.markdown("**Individual Policy Effects:**")
                    st.write(f"- Education increase: {education_effect:.2f} percentage points")
                    st.write(f"- Manufacturing change: {manufacturing_effect:.2f} percentage points")
                    st.write(f"- Pay transparency: -{transparency_effect:.2f} percentage points")
                    st.write(f"- Training programs: -{training_effect:.2f} percentage points")
                    st.write(f"- **Net effect: {total_policy_effect:.2f} percentage points**")
            else:
                st.info("Baseline regression results not available for simulation.")
        
        # County-specific recommendations
        st.subheader("🗺️ County-Specific Recommendations")
        
        # Show top counties needing intervention
        high_gap_counties = data.nlargest(5, 'gender_gap_pct')[['county_name', 'gender_gap_pct', 'pct_bach', 'pct_black', 'pct_manuf']]
        
        st.markdown("**🏆 Top 5 Counties Needing Intervention:**")
        
        for idx, row in high_gap_counties.iterrows():
            with st.expander(f"{row['county_name']} - {row['gender_gap_pct']:.1f}% gap"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Gender Gap", f"{row['gender_gap_pct']:.1f}%")
                    st.metric("Education (% Bachelor's)", f"{row['pct_bach']:.1f}%")
                
                with col2:
                    st.metric("Black Population (%)", f"{row['pct_black']:.1f}%")
                    st.metric("Manufacturing (%)", f"{row['pct_manuf']:.1f}%")
                
                # County-specific recommendations
                st.markdown("**🎯 Recommended Interventions:**")
                if row['pct_bach'] < 30:
                    st.write("• **Education programs**: Increase access to higher education")
                if row['pct_manuf'] > 15:
                    st.write("• **Industry diversification**: Reduce reliance on manufacturing")
                if row['pct_black'] > 10:
                    st.write("• **Racial equity programs**: Address intersectional wage disparities")
                st.write("• **Pay transparency**: Implement mandatory pay reporting")
                st.write("• **Leadership development**: Support women's advancement")
        
        # Implementation timeline
        st.subheader("📅 Implementation Timeline")
        
        timeline_data = {
            'Phase': ['Phase 1 (Months 1-6)', 'Phase 2 (Months 7-12)', 'Phase 3 (Months 13-24)', 'Phase 4 (Months 25-36)'],
            'Activities': [
                'Data collection and baseline assessment, stakeholder engagement',
                'Pilot programs in high-gap counties, policy development',
                'Full implementation, monitoring systems, training programs',
                'Evaluation and refinement, expansion to additional counties'
            ],
            'Expected Outcomes': [
                'Baseline established, partnerships formed',
                'Initial program results, policy framework in place',
                'Measurable gap reduction, comprehensive monitoring',
                'Sustained improvements, scalable model developed'
            ]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        # Success metrics
        st.subheader("📊 Success Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Short-term (1 year):**")
            st.write("• 5% reduction in mean gender wage gap")
            st.write("• 50% of high-gap counties show improvement")
            st.write("• Implementation of pay transparency policies")
        
        with col2:
            st.markdown("**Medium-term (2-3 years):**")
            st.write("• 15% reduction in mean gender wage gap")
            st.write("• 80% of counties show improvement")
            st.write("• Established monitoring and reporting systems")
        
        with col3:
            st.markdown("**Long-term (5 years):**")
            st.write("• 25% reduction in mean gender wage gap")
            st.write("• All counties below 50% gap threshold")
            st.write("• Sustainable policy framework in place")

if __name__ == "__main__":
    main() 