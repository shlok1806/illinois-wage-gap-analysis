# Cross-County Analysis of Gender and Racial Median Earnings Gaps in Illinois

## Project Overview

This project analyzes wage disparities by gender and race across Illinois counties using publicly available ACS 2022 1-Year estimates via the Census API. The analysis uncovers local labor-market structures and provides policymakers with geographic insights for targeted interventions.

## Why This Matters

- **Wage disparities persist** even after accounting for education and industry
- **County-level analysis** reveals urban Chicago versus rural downstate differences
- **Geographic breakdown** helps policymakers target training and anti-bias programs effectively

## Key Research Questions

1. **Gender Gap**: What is the percentage difference between male and female median earnings in each Illinois county?
2. **Racial Gaps**: How do median earnings for Black, Asian, and multiracial populations compare to White median earnings across counties?
3. **Drivers of Variation**: Which county characteristics (education attainment, racial composition, industry mix, urbanicity) explain cross-county differences in these gaps?

## Data Sources

We use the **Census Data API** (no registration required) via the `censusdata` Python package to pull ACS 2022 1-Year county estimates.

### Key Variables

- **Median Earnings by Sex**: Table B20017
- **Population by Race**: Table B20004  
- **Education**: Percent bachelor's degree or higher (S1501)
- **Industry Mix**: Percent employed in manufacturing (C24050)
- **Urbanicity**: County population and land area

## Project Structure

```
├── data/                   # Data files and shapefiles
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Python source code
│   ├── data_collection.py # Census API data collection
│   ├── data_processing.py # Data cleaning and variable construction
│   ├── analysis.py        # Statistical analysis functions
│   ├── visualization.py   # Mapping and plotting functions
│   └── spatial_analysis.py # Spatial econometrics
├── results/               # Output files (figures, tables, reports)
├── requirements.txt       # Python dependencies
└── main.py               # Main execution script
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Run the full analysis**:
   ```bash
   python main.py
   ```

2. **Interactive analysis**:
   ```bash
   jupyter notebook notebooks/illinois_wage_gap_analysis.ipynb
   ```

3. **View results**:
   - Check `results/` directory for generated figures and reports
   - Open `results/analysis_report.pdf` for the complete analysis

## Key Features

- **Automated data collection** from Census API
- **Interactive choropleth maps** of wage gaps by county
- **Regression analysis** with spatial diagnostics
- **Quantile regression** for heterogeneous effects
- **County clustering** to identify patterns
- **Policy recommendations** based on findings

## Deliverables

1. **Jupyter Notebook** (.ipynb): Complete analysis with inline figures
2. **Static Report** (PDF): 6-8 page academic report
3. **Slide Deck**: 8-10 slides for presentations
4. **Interactive Maps**: Web-based visualizations

## Timeline

- **Weeks 1-2**: Data collection and cleaning
- **Weeks 3-4**: Descriptive analysis and visualization
- **Weeks 5-6**: Regression analysis and interpretation
- **Weeks 7-8**: Extensions and final deliverables

## Why This Is Pure Economics

- **Economic Questions**: Focused on earnings gaps and determinants
- **Econometric Tools**: OLS, spatial diagnostics, quantile regressions
- **Policy Focus**: Direct recommendations for labor-market interventions

## Contact

For questions about this analysis or to contribute, please open an issue or submit a pull request.

---

*This project uses publicly available Census data and is designed for educational and policy research purposes.* 