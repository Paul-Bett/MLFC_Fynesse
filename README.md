# Time-Series Analysis of Insect Count and Weather Data

This project implements an end-to-end pipeline for analyzing insect monitoring data alongside environmental conditions such as temperature, humidity, and rainfall. The notebook builds a structured workflow from data loading and merging to exploratory analysis, temporal aggregation, and visualization.  

---

## Project Overview

The pipeline enables:
- Integration of insect count data with weather station data.  
- Quick exploratory data analysis (EDA) for initial insights.  
- Temporal resampling to hourly aggregates for smoother trends.  
- Visualization of insect activity against weather variables.  
- Insight into how environmental features (temperature, humidity, rainfall) may influence insect counts.  

---

## Pipeline Stages and Functions

### 1. Data Loading
Functions:
- `load_and_merge_datasets()` *(from fyness)* – Reads insect and weather datasets, merges them on timestamp.  

### 2. Exploratory Data Analysis (EDA)
Functions:
- `quick_eda()` *(from fyness)* – Generates summary statistics, distributions, and quick plots.  
- Fallback: `.head()` and `.describe()` if `quick_eda` is missing.  

### 3. Data Cleaning & Handling Missing Values
Functions:
- `handle_missing_data()` *(from fyness)* – Deals with NaNs, interpolation, or dropping strategies.  

### 4. Feature Engineering
Functions:
- `create_time_features()` *(from fyness)* – Extracts hour, day, month, etc., from timestamp for analysis.  

### 5. Aggregation
- **Hourly Resampling**:  
  - Insect counts → summed per hour.  
  - Weather features (temperature, humidity, rainfall) → averaged per hour.  

### 6. Visualization
Functions:
- `plot_time_series_cols()` *(from fyness)* – Plots insect counts vs weather features over time.  
- `plot_count_histogram()` *(from fyness)* – Shows distribution of insect counts.  
- Additional time-series plots for:  
  - **Temperature**  
  - **Humidity**  
  - **Rainfall**  

---

## Example Outputs

### Sample EDA Output
![EDA Summary](images/eda_summary.png)

### Hourly Count and Weather Trends
![Hourly Time Series](images/hourly_timeseries.png)

### Distribution of Insect Counts
![Count Distribution](images/count_distribution.png)

### Humidity and Rainfall Trends
![Humidity Rainfall Trends](images/humidity_rainfall.png)


---

## Insights

- Insect counts exhibit strong hourly variations, peaking during specific night-time intervals.  
- Environmental features (temperature, humidity, rainfall) align with activity patterns, suggesting possible causal or correlative relationships.  
- Hourly resampling smooths noise and provides interpretable trends.  

---

