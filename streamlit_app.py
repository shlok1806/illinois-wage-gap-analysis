"""
Economic Research: County-Level Gender Wage Gaps
Streamlit Application for Interactive Economic Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from src.models import WageGapAnalyzer

# Page configuration
st.set_page_config(
    page_title="Economic Research: Gender Wage Gaps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">Economic Research: County-Level Gender Wage Gaps</h1>', unsafe_allow_html=True)
    st.markdown("**Theoretical Foundation, Causal Identification, Economic Interpretation**")
    
    # Sidebar
    st.sidebar.title("Analysis Parameters")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_counties = st.sidebar.slider("Number of Counties", 50, 200, 100, 10)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    run_mincer = st.sidebar.checkbox("Mincer Equation", value=True)
    run_gender_gap = st.sidebar.checkbox("Gender Gap Analysis", value=True)
    run_did = st.sidebar.checkbox("Difference-in-Differences", value=True)
    run_iv = st.sidebar.checkbox("Instrumental Variables", value=True)
    run_diagnostics = st.sidebar.checkbox("Econometric Diagnostics", value=True)
    run_interpretation = st.sidebar.checkbox("Economic Interpretation", value=True)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WageGapAnalyzer()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Data Exploration", "Economic Analysis", "Results", "Policy Implications"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">Research Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Research Question:**
            How do county-level educational attainment and industry composition 
            causally affect gender wage gaps through human capital accumulation 
            and labor market structure?
            
            **Theoretical Framework:**
            This study extends the Mincer-Oaxaca framework by incorporating 
            county-level variation in human capital returns and labor market structure.
            """)
        
        with col2:
            st.markdown("""
            **Identification Strategy:**
            - Difference-in-Differences with education policy changes
            - Instrumental Variables using distance to colleges
            - Robust standard errors and comprehensive diagnostics
            
            **Economic Interpretation:**
            - Elasticities and economic magnitudes
            - Welfare analysis and deadweight loss
            - Policy implications and counterfactuals
            """)
        
        # Run analysis button
        if st.button("Run Complete Economic Analysis", type="primary"):
            with st.spinner("Generating synthetic data and running analysis..."):
                # Generate data
                data = st.session_state.analyzer.generate_synthetic_data(n_counties=n_counties)
                st.session_state.data = data
                
                # Run analysis
                results = {}
                if run_mincer:
                    results['mincer'] = st.session_state.analyzer.estimate_mincer_equation()
                if run_gender_gap:
                    results['gender_gap'] = st.session_state.analyzer.analyze_gender_gap()
                if run_did:
                    results['did'] = st.session_state.analyzer.estimate_difference_in_differences()
                if run_iv:
                    results['iv'] = st.session_state.analyzer.estimate_instrumental_variables()
                if run_diagnostics:
                    results['diagnostics'] = st.session_state.analyzer.perform_diagnostics()
                if run_interpretation:
                    results['interpretation'] = st.session_state.analyzer.calculate_economic_interpretation()
                
                st.session_state.results = st.session_state.analyzer.results
                st.success("Economic analysis completed successfully!")
    
    with tab2:
        st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True)
        
        if 'data' in st.session_state:
            data = st.session_state.data
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Summary Statistics")
                st.dataframe(data.describe())
            
            with col2:
                st.subheader("Data Overview")
                st.write(f"**Number of counties:** {len(data)}")
                st.write(f"**Variables:** {len(data.columns)}")
                st.write(f"**Missing values:** {data.isnull().sum().sum()}")
            
            # Key variables distribution
            st.subheader("Distribution of Key Variables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(data, x='education_level', 
                                  title="Distribution of Education Levels",
                                  labels={'education_level': 'College Attainment Rate'})
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.histogram(data, x='gender_gap', 
                                  title="Distribution of Gender Wage Gaps",
                                  labels={'gender_gap': 'Gender Wage Gap'})
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.histogram(data, x='manufacturing_share', 
                                  title="Distribution of Manufacturing Share",
                                  labels={'manufacturing_share': 'Manufacturing Employment Share'})
                st.plotly_chart(fig3, use_container_width=True)
                
                fig4 = px.histogram(data, x='median_wage', 
                                  title="Distribution of Median Wages",
                                  labels={'median_wage': 'Median Wage ($)'})
                st.plotly_chart(fig4, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               title="Correlation Matrix of Key Variables",
                               color_continuous_scale='RdBu',
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        else:
            st.info("Please run the analysis first to explore the data.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Economic Analysis</h2>', unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Mincer Equation Results
            if 'mincer_equation' in results:
                st.subheader("Mincer Equation Estimation")
                
                mincer = results['mincer_equation']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R²", f"{mincer['r_squared']:.3f}")
                with col2:
                    st.metric("Adjusted R²", f"{mincer['adj_r_squared']:.3f}")
                with col3:
                    st.metric("F-statistic", f"{mincer['f_statistic']:.2f}")
                
                # Coefficients table
                coef_df = pd.DataFrame({
                    'Variable': list(mincer['coefficients'].keys()),
                    'Coefficient': list(mincer['coefficients'].values()),
                    'Std Error': list(mincer['std_errors'].values()),
                    'P-value': list(mincer['p_values'].values())
                })
                
                st.dataframe(coef_df, use_container_width=True)
            
            # Gender Gap Analysis
            if 'gender_gap_analysis' in results:
                st.subheader("Gender Gap Analysis")
                
                gap = results['gender_gap_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("R²", f"{gap['r_squared']:.3f}")
                with col2:
                    st.metric("Observations", f"{gap['n_observations']}")
                
                # Key coefficients
                education_effect = gap['coefficients'].get('education_level', 0)
                manufacturing_effect = gap['coefficients'].get('manufacturing_share', 0)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Education Effect", f"{education_effect:.4f}")
                with col2:
                    st.metric("Manufacturing Effect", f"{manufacturing_effect:.4f}")
            
            # DiD Results
            if 'did_analysis' in results:
                st.subheader("Difference-in-Differences Analysis")
                
                did = results['did_analysis']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Treatment Effect", f"{did['treatment_effect']:.4f}")
                with col2:
                    st.metric("Standard Error", f"{did['treatment_se']:.4f}")
                with col3:
                    st.metric("P-value", f"{did['treatment_pvalue']:.4f}")
            
            # IV Results
            if 'iv_analysis' in results:
                st.subheader("Instrumental Variables Analysis")
                
                iv = results['iv_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("First Stage F-stat", f"{iv['first_stage']['f_statistic']:.2f}")
                    st.metric("First Stage R²", f"{iv['first_stage']['r_squared']:.3f}")
                
                with col2:
                    st.metric("Education Coefficient", f"{iv['second_stage']['education_coefficient']:.4f}")
                    st.metric("Standard Error", f"{iv['second_stage']['education_se']:.4f}")
        
        else:
            st.info("Please run the analysis first to view results.")
    
    with tab4:
        st.markdown('<h2 class="section-header">Detailed Results</h2>', unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Diagnostics
            if 'diagnostics' in results:
                st.subheader("Econometric Diagnostics")
                
                diag = results['diagnostics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Heteroskedasticity Test (Breusch-Pagan)**")
                    st.write(f"Statistic: {diag['breusch_pagan']['statistic']:.4f}")
                    st.write(f"P-value: {diag['breusch_pagan']['p_value']:.4f}")
                    st.write(f"Heteroskedastic: {'Yes' if diag['breusch_pagan']['heteroskedastic'] else 'No'}")
                
                with col2:
                    st.markdown("**Normality Test (Jarque-Bera)**")
                    st.write(f"Statistic: {diag['jarque_bera']['statistic']:.4f}")
                    st.write(f"P-value: {diag['jarque_bera']['p_value']:.4f}")
                    st.write(f"Normal residuals: {'Yes' if diag['jarque_bera']['normal'] else 'No'}")
                
                # VIF table
                if 'vif' in diag:
                    st.subheader("Multicollinearity (VIF)")
                    vif_df = pd.DataFrame(diag['vif'])
                    st.dataframe(vif_df, use_container_width=True)
            
            # Economic Interpretation
            if 'economic_interpretation' in results:
                st.subheader("Economic Interpretation")
                
                econ = results['economic_interpretation']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Elasticities**")
                    st.write(f"Education elasticity: {econ['elasticities']['education']:.3f}")
                    st.write(f"Gender gap elasticity: {econ['elasticities']['gender_gap']:.3f}")
                
                with col2:
                    st.markdown("**Economic Magnitudes**")
                    st.write(f"10pp education effect: {econ['economic_magnitudes']['education_10pp_effect']:.4f}")
                    st.write(f"10pp manufacturing effect: {econ['economic_magnitudes']['manufacturing_10pp_effect']:.4f}")
                
                st.markdown("**Welfare Analysis**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Deadweight Loss", f"${econ['welfare']['deadweight_loss']:,.0f}")
                with col2:
                    st.metric("Per Capita DWL", f"${econ['welfare']['deadweight_loss_per_capita']:,.0f}")
                with col3:
                    st.metric("Competitive Wage", f"${econ['welfare']['competitive_wage']:,.0f}")
        
        else:
            st.info("Please run the analysis first to view detailed results.")
    
    with tab5:
        st.markdown('<h2 class="section-header">Policy Implications</h2>', unsafe_allow_html=True)
        
        if 'results' in st.session_state and 'economic_interpretation' in st.session_state.results:
            econ = st.session_state.results['economic_interpretation']
            
            st.markdown("**Key Policy Findings**")
            
            # Policy implications
            for key, implication in econ['policy_implications'].items():
                st.markdown(f"""
                <div class="result-box">
                    <strong>{key.replace('_', ' ').title()}:</strong><br>
                    {implication}
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive policy simulation
            st.subheader("Policy Simulation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                education_increase = st.slider("Education Increase (%)", 0, 20, 10, 1)
                st.write(f"Simulated effect: {education_increase * econ['economic_magnitudes']['education_10pp_effect'] * 10:.3f}")
            
            with col2:
                manufacturing_change = st.slider("Manufacturing Change (%)", -20, 20, 0, 1)
                st.write(f"Simulated effect: {manufacturing_change * econ['economic_magnitudes']['manufacturing_10pp_effect'] * 10:.3f}")
            
            # Summary
            st.markdown("**Research Contributions**")
            st.markdown("""
            - **Theoretical Foundation**: Extends Mincer-Oaxaca framework with county-level variation
            - **Causal Identification**: Multiple strategies including DiD and IV approaches
            - **Economic Interpretation**: Elasticities, welfare analysis, and policy implications
            - **Rigorous Methods**: Comprehensive diagnostics and robust standard errors
            - **Policy Relevance**: Quantified effects for education and industry policies
            """)
        
        else:
            st.info("Please run the analysis first to view policy implications.")

if __name__ == "__main__":
    main() 