# Mapping Inequality: County-Level Gender and Racial Earnings Disparities in Illinois

A comprehensive **economic analysis** of wage disparities across Illinois counties using American Community Survey (ACS) 2022 data, grounded in economic theory with proper identification strategies.

## Overview

This project examines gender and racial wage gaps at the county level in Illinois through the lens of **economic theory**, providing:
- **Theoretical Framework**: Mincer earnings equation with discrimination terms
- **Identification Strategy**: Instrumental variables and natural experiments
- **Structural Interpretation**: Economic magnitudes and welfare implications
- **Policy Analysis**: Counterfactual simulations and cost-benefit analysis

## Economic Framework

### Theoretical Foundation
- **Mincer Earnings Equation**: `ln(w_i) = α + β₁S_i + β₂X_i + β₃Z_i + γ₁D_i + γ₂(D_i × S_i) + ε_i`
- **Discrimination Theory**: Taste-based vs. statistical discrimination (Becker 1957, Arrow 1973)
- **Spatial Equilibrium**: Roback (1982) framework for local labor markets

### Identification Strategy
1. **Instrumental Variables**: Distance to land-grant colleges for education
2. **Natural Experiments**: Minimum wage variation, manufacturing decline
3. **Fixed Effects**: County-year and county-demographic specifications

### Literature Review
Positioned within 5 key papers:
- Blau & Kahn (2017) - Gender wage gap trends
- Autor et al. (2003) - Skill-biased technical change
- Card & Krueger (1992) - School quality and earnings
- Moretti (2004) - Human capital externalities
- Bertrand & Mullainathan (2004) - Racial discrimination

## Features

- **Economic Analysis**: Mincer equation, elasticities, welfare calculations
- **Causal Inference**: IV analysis, natural experiments, robustness checks
- **Policy Simulation**: Counterfactual scenarios, cost-benefit analysis
- **Structural Interpretation**: Economic magnitudes, distributional effects
- **Spatial Analysis**: County-level heterogeneity, local policy effects

## Installation

```bash
git clone <repository-url>
cd EconProject_Labor_Wage
pip install -r requirements.txt
```

## Usage

### Complete Economic Analysis
```bash
python main.py
```
Runs the full economic analysis pipeline including theoretical framework, identification strategies, and policy simulations.

### Individual Components
```bash
# Economic analysis only
python -c "from src.economic_analysis import EconomicWageGapAnalyzer; import pandas as pd; data = pd.read_csv('data/illinois_county_processed.csv'); analyzer = EconomicWageGapAnalyzer(data); analyzer.run_complete_economic_analysis()"

# Interactive analysis
streamlit run streamlit_app.py
```

## Project Structure

```
├── ECONOMIC_FRAMEWORK.md     # Theoretical foundation and methodology
├── LITERATURE_REVIEW.md      # Academic literature review
├── data/                     # Raw and processed data files
├── results/                  # Analysis outputs and visualizations
├── src/                      # Core analysis modules
│   ├── data_collection.py    # Census API data collection
│   ├── data_processing.py    # Data cleaning and processing
│   ├── analysis.py           # Descriptive statistical analysis
│   ├── economic_analysis.py  # Economic framework implementation
│   └── visualization.py      # Plotting and mapping
├── streamlit_app.py          # Interactive web application
├── main.py                   # Complete analysis pipeline
└── requirements.txt          # Python dependencies
```

## Key Economic Findings

- **Education Elasticity**: Estimated returns to education across counties
- **Discrimination Magnitude**: Economic cost of wage discrimination
- **Policy Effects**: Simulated impacts of minimum wage and education policies
- **Spatial Heterogeneity**: County-level variation in wage determination
- **Welfare Implications**: Deadweight loss and distributional effects

## Research Contributions

1. **Spatial Analysis**: First comprehensive county-level wage gap analysis within a state
2. **Policy Relevance**: Local policy simulation framework with economic magnitudes
3. **Structural Interpretation**: Move beyond descriptive statistics to causal inference
4. **Identification Strategy**: Multiple approaches including IV and natural experiments
5. **Theoretical Foundation**: Grounded in established economic theory

## Data Sources

- **American Community Survey (ACS) 2022**: Individual and household data
- **U.S. Census Bureau API**: Public access, no registration required
- **Illinois County Boundaries**: Geographic and demographic data
- **Policy Variables**: Minimum wage, education funding, anti-discrimination laws

## Academic Context

This research contributes to the economics literature on:
- **Labor Economics**: Wage determination and discrimination
- **Urban Economics**: Spatial wage variation and local labor markets
- **Public Economics**: Policy evaluation and welfare analysis
- **Applied Econometrics**: Identification strategies and causal inference

## License

MIT License - see LICENSE file for details 