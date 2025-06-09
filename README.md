# Economic Research: County-Level Gender Wage Gaps

A comprehensive **economic analysis** of gender wage gaps using rigorous econometric methods, theoretical foundation, and causal identification strategies.

## Research Overview

This project examines county-level gender wage gaps through the lens of **economic theory**, providing:
- **Theoretical Framework**: Mincer earnings equation with discrimination terms
- **Causal Identification**: Difference-in-differences and instrumental variables
- **Economic Interpretation**: Elasticities, welfare analysis, and policy implications
- **Rigorous Diagnostics**: Comprehensive econometric testing and validation

## Economic Framework

### Theoretical Foundation
- **Mincer Earnings Equation**: `ln(w_i) = α + β₁S_i + β₂X_i + β₃X_i² + γ₁D_i + γ₂(D_i × S_i) + δ₁Z_c + δ₂(Z_c × D_i) + ε_i`
- **Discrimination Theory**: Becker (1957) model with taste-based discrimination
- **Human Capital Theory**: Education returns and experience accumulation
- **Spatial Variation**: County-level heterogeneity in labor market structure

### Identification Strategy
1. **Difference-in-Differences**: Education policy changes as natural experiments
2. **Instrumental Variables**: Distance to colleges for education variation
3. **Robust Standard Errors**: HC1 and clustered standard errors
4. **Comprehensive Diagnostics**: VIF, heteroskedasticity, normality tests

### Literature Engagement
Positioned within foundational economic literature:
- **Mincer (1974)**: Human capital theory of wage determination
- **Oaxaca (1973)**: Wage gap decomposition methodology
- **Blau & Kahn (2017)**: Gender wage gap trends and explanations
- **Card & Krueger (1992)**: Natural experiments in education
- **Neumark & Wascher (2008)**: Policy evaluation methodology

## Key Features

- **Theoretical Model**: Mincer equation with discrimination and interaction terms
- **Causal Inference**: Multiple identification strategies (DiD, IV)
- **Economic Interpretation**: Elasticities, welfare analysis, deadweight loss
- **Policy Simulation**: Interactive counterfactual scenarios
- **Rigorous Diagnostics**: Comprehensive econometric testing
- **Interactive Interface**: Professional Streamlit web application

## Installation

```bash
git clone <repository-url>
cd EconProject_Labor_Wage
pip install -r requirements.txt
```

## Usage

### Command Line Analysis
```bash
python analysis_main.py
```
Runs the complete economic analysis pipeline including:
- Mincer equation estimation with discrimination terms
- Gender gap analysis using Oaxaca-Blinder framework
- Difference-in-differences analysis for policy evaluation
- Instrumental variables analysis with distance instruments
- Comprehensive econometric diagnostics
- Economic interpretation with elasticities and welfare analysis

### Interactive Web Application
```bash
streamlit run streamlit_app.py
```
Launches an interactive web application with:
- **Overview**: Research framework and methodology
- **Data Exploration**: Summary statistics and visualizations
- **Economic Analysis**: Regression results and coefficients
- **Results**: Detailed diagnostics and economic interpretation
- **Policy Implications**: Interactive policy simulation tools

## Project Structure

```
EconProject_Labor_Wage/
├── src/
│   └── models.py                    # Economic analysis models
├── data/
│   └── synthetic_data.csv          # Synthetic county-level data
├── results/
│   └── analysis_results.json       # Analysis results and diagnostics
├── analysis_main.py                # Main analysis script
├── streamlit_app.py                # Interactive web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_FRAMEWORK.md            # Detailed research framework
├── .git/                          # Version control
└── .venv/                         # Virtual environment
```

## Economic Analysis Components

### 1. Mincer Equation Estimation
- Log-linear wage equation with human capital variables
- Gender discrimination terms and interaction effects
- County-level characteristics and spatial variation
- Robust standard errors (HC1)

### 2. Gender Gap Analysis
- Oaxaca-Blinder decomposition framework
- County-level gender gap determinants
- Education and industry composition effects
- Urban-rural heterogeneity

### 3. Causal Identification
- **Difference-in-Differences**: Education policy changes
- **Instrumental Variables**: Distance to colleges
- **Fixed Effects**: County and time controls
- **Clustered Standard Errors**: State-level clustering

### 4. Econometric Diagnostics
- **Multicollinearity**: Variance Inflation Factor (VIF)
- **Heteroskedasticity**: Breusch-Pagan test
- **Normality**: Jarque-Bera test
- **Model Specification**: F-tests and R² analysis

### 5. Economic Interpretation
- **Elasticities**: Education and gender gap elasticities
- **Economic Magnitudes**: Policy-relevant effect sizes
- **Welfare Analysis**: Deadweight loss calculations
- **Policy Implications**: Counterfactual scenarios

## Key Findings

- **Education Effects**: County-level educational attainment significantly affects gender wage gaps
- **Industry Composition**: Manufacturing-dominated counties show larger gender gaps
- **Policy Relevance**: Education policies have quantifiable effects on wage inequality
- **Spatial Heterogeneity**: Significant variation across counties in gap determinants
- **Economic Magnitudes**: Quantified elasticities for policy evaluation

## Academic Standards

This research demonstrates:
- ✅ **Theoretical Foundation**: Clear economic model with testable hypotheses
- ✅ **Literature Engagement**: Positioned within foundational economic literature
- ✅ **Causal Identification**: Multiple strategies with proper assumptions
- ✅ **Economic Interpretation**: Elasticities, welfare analysis, policy implications
- ✅ **Rigorous Methods**: Comprehensive diagnostics and robust standard errors
- ✅ **Reproducibility**: Modular code, clear documentation, version control

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **statsmodels**: Econometric analysis and regression
- **scipy**: Statistical testing and diagnostics
- **streamlit**: Interactive web application
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plotting

## License

MIT License - see LICENSE file for details

## Contributing

This project follows academic research standards. For contributions:
1. Ensure theoretical foundation is maintained
2. Add appropriate econometric diagnostics
3. Include economic interpretation of results
4. Maintain reproducibility and documentation

---

**This represents professional economics research methodology with rigorous theoretical foundation, causal identification, and economic interpretation.**