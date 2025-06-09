# Illinois County Wage Gap Analysis - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation & Setup](#installation--setup)
3. [Project Structure](#project-structure)
4. [Data Sources](#data-sources)
5. [Methodology](#methodology)
6. [Usage Instructions](#usage-instructions)
7. [Streamlit App](#streamlit-app)
8. [API Documentation](#api-documentation)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## Project Overview

This project analyzes wage disparities by gender and race across Illinois counties using publicly available ACS 2022 1-Year estimates via the Census API. The analysis provides policymakers with geographic insights for targeted interventions.

### Key Features
- **Automated Data Collection**: Pulls data directly from Census API
- **Comprehensive Analysis**: OLS regression, quantile regression, spatial analysis, clustering
- **Interactive Visualizations**: Streamlit web app with Plotly charts
- **Policy Insights**: Actionable recommendations based on findings
- **Reproducible Research**: Complete pipeline from data to insights

### Research Questions
1. What is the percentage difference between male and female median earnings in each Illinois county?
2. How do median earnings for different racial groups compare across counties?
3. Which county characteristics explain cross-county differences in wage gaps?

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EconProject_Labor_Wage
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import streamlit, pandas, numpy, plotly; print('Installation successful!')"
   ```

### Environment Variables
No API keys required - uses publicly available Census data.

---

## Project Structure

```
EconProject_Labor_Wage/
├── data/                           # Data files
│   ├── illinois_county_data.csv    # Raw Census data
│   └── illinois_county_processed.csv # Processed data
├── notebooks/                      # Analysis scripts
│   └── illinois_wage_gap_analysis.py # Python script version
├── results/                        # Output files
│   ├── regression_table.csv        # Statistical results
│   ├── gender_gap_map.png         # Visualizations
│   └── *.png                      # Other charts
├── src/                           # Source code modules
│   ├── data_collection.py         # Census API data collection
│   ├── data_processing.py         # Data cleaning and processing
│   ├── analysis.py                # Statistical analysis
│   └── visualization.py           # Plotting and mapping
├── main.py                        # Main pipeline script
├── streamlit_app.py               # Interactive web app
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── DOCUMENTATION.md               # This file
```

---

## Data Sources

### Census Data API
- **Source**: U.S. Census Bureau American Community Survey (ACS)
- **Year**: 2022 1-Year estimates
- **Geography**: Illinois counties (FIPS code 17)
- **Access**: Public API, no registration required

### Key Variables

#### Median Earnings by Sex (Table B20017)
- `B20017_002E`: Male median earnings
- `B20017_003E`: Female median earnings

#### Population by Race (Table B20004)
- `B20004_001E`: Total population
- `B20004_003E`: White alone
- `B20004_004E`: Black or African American alone
- `B20004_006E`: Asian alone
- `B20004_009E`: Two or more races

#### County Characteristics
- `S1501_C01_015E`: Percent bachelor's degree or higher
- `C24050_003E`: Manufacturing employment
- `C24050_001E`: Total employment
- `B01003_001E`: Total population (alternative)

### Data Quality
- **Coverage**: All 102 Illinois counties
- **Missing Data**: Handled with imputation and exclusion
- **Outliers**: Identified and addressed in processing

---

## Methodology

### 1. Data Collection
```python
from src.data_collection import CensusDataCollector

collector = CensusDataCollector(year=2022, survey='acs1')
raw_data = collector.download_county_data()
```

**Process**:
- Define Census variables and geography
- Download via Census API
- Handle API errors with fallback to sample data
- Save raw data to CSV

### 2. Data Processing
```python
from src.data_processing import DataProcessor

processor = DataProcessor(raw_data)
processed_data = processor.process_data()
```

**Steps**:
- Clean missing values and outliers
- Construct key variables (gaps, percentages, log transforms)
- Validate data quality
- Save processed data

### 3. Statistical Analysis
```python
from src.analysis import WageGapAnalyzer

analyzer = WageGapAnalyzer(processed_data)
results = analyzer.run_full_analysis()
```

**Methods**:
- **OLS Regression**: Baseline and extended models
- **Quantile Regression**: Heterogeneity across distribution
- **Spatial Analysis**: Moran's I for autocorrelation
- **Clustering**: K-means for county groups
- **Robustness Checks**: Outliers, heteroskedasticity, multicollinearity

### 4. Visualization
```python
from src.visualization import WageGapVisualizer

visualizer = WageGapVisualizer(processed_data)
visualizations = visualizer.create_all_visualizations(results)
```

**Charts**:
- Choropleth maps of wage gaps
- Scatter plots of correlations
- Distribution plots
- Regression coefficient plots
- County comparison charts

---

## Usage Instructions

### Option 1: Complete Pipeline
```bash
python main.py
```
Runs the entire analysis from data collection to visualization.

### Option 2: Individual Modules
```bash
# Data collection
python src/data_collection.py

# Data processing
python src/data_processing.py

# Statistical analysis
python src/analysis.py

# Visualization
python src/visualization.py
```

### Option 3: Python Script
```bash
cd notebooks
python illinois_wage_gap_analysis.py
```

### Option 4: Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## Streamlit App

### Features
- **Interactive Dashboard**: 6-page web application
- **Real-time Analysis**: Cached data and analysis
- **Dynamic Visualizations**: Plotly charts with hover data
- **Policy Simulator**: Interactive tool for policy impact

### Pages

#### 1. Overview
- Project description and methodology
- Key statistics and metrics
- Research questions and data sources

#### 2. Data Explorer
- Interactive variable selection
- Histograms, box plots, scatter plots
- Summary statistics and county rankings

#### 3. Statistical Analysis
- Regression results with significance indicators
- Coefficient plots and tables
- Quantile regression heatmaps

#### 4. Geographic Analysis
- Geographic visualizations
- Spatial autocorrelation results
- County clustering analysis

#### 5. Results Summary
- Key findings and statistical summary
- Robustness check results
- Comprehensive results overview

#### 6. Policy Insights
- Policy recommendations by category
- Interactive policy impact simulator
- Evidence-based intervention suggestions

### Running the App
```bash
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

---

## API Documentation

### CensusDataCollector

#### `__init__(year=2022, survey='acs1')`
Initialize the data collector.

**Parameters**:
- `year` (int): Census year (default: 2022)
- `survey` (str): Survey type ('acs1' or 'acs5')

#### `download_county_data()`
Download Census data for Illinois counties.

**Returns**: pandas.DataFrame with raw Census data

### DataProcessor

#### `__init__(data=None)`
Initialize the data processor.

**Parameters**:
- `data` (DataFrame): Raw Census data

#### `process_data()`
Complete data processing pipeline.

**Returns**: pandas.DataFrame with processed data

### WageGapAnalyzer

#### `__init__(data=None)`
Initialize the analyzer.

**Parameters**:
- `data` (DataFrame): Processed county data

#### `baseline_regression()`
Run baseline OLS regression.

**Returns**: statsmodels regression results

#### `run_full_analysis()`
Run complete analysis pipeline.

**Returns**: dict with all analysis results

### WageGapVisualizer

#### `__init__(data=None)`
Initialize the visualizer.

**Parameters**:
- `data` (DataFrame): Processed county data

#### `create_all_visualizations(results=None)`
Create all visualizations.

**Parameters**:
- `results` (dict): Analysis results

**Returns**: dict with all visualization objects

---

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Census API Errors
- **Issue**: API timeout or connection errors
- **Solution**: Uses fallback sample data automatically
- **Check**: Internet connection and Census API status

#### 3. Spatial Analysis Errors
- **Issue**: Missing pysal or esda packages
- **Solution**: Install with `pip install pysal esda`
- **Alternative**: Spatial analysis is optional

#### 4. Streamlit App Issues
- **Issue**: Port already in use
- **Solution**: `streamlit run streamlit_app.py --server.port 8502`

#### 5. Memory Issues
- **Issue**: Large datasets causing memory problems
- **Solution**: Use smaller sample or optimize data types

### Error Messages

#### "No module named 'data_collection'"
```bash
# Fix: Run from project root directory
cd /path/to/EconProject_Labor_Wage
python main.py
```

#### "Cannot save file into non-existent directory"
```bash
# Fix: Create directories manually
mkdir -p data results
```

#### "Census API error"
```bash
# Fix: Check internet connection
# The app will use sample data as fallback
```

---

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature-name`
6. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features

### Testing
```bash
# Run basic tests
python -m pytest tests/

# Run specific module
python -m pytest tests/test_data_collection.py
```

### Documentation
- Update this documentation for new features
- Add inline comments for complex logic
- Include usage examples

---

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the troubleshooting section above

---

## Acknowledgments

- U.S. Census Bureau for providing the data
- Streamlit team for the web framework
- Plotly for interactive visualizations
- The open-source community for supporting libraries

---

*Last updated: [Current Date]*
*Version: 1.0.0* 