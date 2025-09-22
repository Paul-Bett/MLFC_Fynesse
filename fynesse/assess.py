from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Standard libraries and essential imports
import os
import re
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns

# Sklearn models & metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, classification_report, precision_score, recall_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.inspection import permutation_importance

# Statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Scipy
from scipy import stats

# Optional libraries
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import missingno as msno
except Exception:
    pass


# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def align_counts_and_weather(counts_df, weather_df, freq='10S', weather_agg='nearest'):
    """
    Align insect counts to weather records.
    - counts_df: DataFrame with 'timestamp' column (python datetime) and 'count'
    - weather_df: DataFrame indexed by datetime
    - freq: desired output frequency (e.g., '10S' because snapshot every 10s)
    Strategy:
    - Convert counts_df timestamp to index
    - Reindex to freq (so missing images become NaN counts)
    - For each count timestamp, find nearest weather row (or forward/backward fill)
    Returns merged DataFrame with columns: ['count', <weather columns>]
    """
    # prepare counts series
    cdf = counts_df.copy()
    cdf['timestamp'] = pd.to_datetime(cdf['timestamp'])
    cdf = cdf.set_index('timestamp').sort_index()
    # keep as is (we don't want to upsample artificially but we'll merge)
    # For each count timestamp find nearest weather sample
    # Use pandas.merge_asof (must have numeric or datetime index as columns)
    w = weather_df.reset_index().rename(columns={weather_df.index.name or 'index':'timestamp'})
    w['timestamp'] = pd.to_datetime(w['timestamp'])
    cdf_reset = cdf.reset_index().rename(columns={'index':'timestamp'})
    merged = pd.merge_asof(cdf_reset.sort_values('timestamp'),
                           w.sort_values('timestamp'),
                           on='timestamp',
                           direction='nearest',
                           tolerance=pd.Timedelta('30s'))  # tune tolerance
    # If merged weather columns missing, we can also do join by rounding timestamps
    merged = merged.set_index('timestamp').sort_index()
    return merged

def summarize_missing(merged_df):
    miss = merged_df.isna().mean()
    return miss[miss>0].sort_values(ascending=False)

def quick_eda(merged_df, sample_n=10):
    print("Shape:", merged_df.shape)
    display(merged_df.head(sample_n))
    print("\nMissing fractions:")
    display(summarize_missing(merged_df))
    print("\nNumeric summary:")
    display(merged_df.describe().T)

def quick_summary(df, n=5):
    print("Shape:", df.shape)
    print("\nHead:")
    display(df.head(n))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing fraction per column:")
    display(df.isna().mean().sort_values(ascending=False).head(20))

# ---------- VISUALIZATIONS ----------
def plot_time_series_cols(df, cols, figsize=(14,4), title=None):
    plt.figure(figsize=figsize)
    df[cols].dropna(how='all').plot(ax=plt.gca(), linewidth=1)
    if title:
        plt.title(title)
    plt.tight_layout()

def plot_count_histogram(df, bins=30):
    plt.figure(figsize=(8,3))
    plt.hist(df['count'].dropna(), bins=bins)
    plt.title("Distribution of insect counts")
    plt.xlabel("count")
    plt.ylabel("frequency")
    plt.tight_layout()

def plot_pairwise_sample(df, cols=None, sample=1000):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    small = df[cols].dropna().sample(min(len(df), sample), random_state=0)
    sns.pairplot(small)
    plt.suptitle("Pairwise relationships (sample)", y=1.02)


def plot_time_series(df, dt_col=None, y_cols=None, figsize=(14,4)):
    """
    Plot one or more time-series columns.
    """
    if dt_col:
        df = df.set_index(dt_col)
    if y_cols is None:
        y_cols = ['count']
    df[y_cols].plot(subplots=False, figsize=figsize, linewidth=1)
    plt.title("Time series: " + ", ".join(y_cols))
    plt.tight_layout()

def plot_pairwise(df, cols=None, sample=1000):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_df = df[cols].dropna().sample(min(len(df), sample), random_state=0)
    sns.pairplot(sample_df)
    plt.suptitle("Pairwise relationships (sample)", y=1.02)

def plot_missingness_matrix(df, figsize=(12,4)):
    """Show missingness matrix across time (boolean)."""
    import missingno as msno  # small dependency; install if missing: pip install missingno
    fig = plt.figure(figsize=figsize)
    msno.matrix(df, figsize=figsize)
    plt.title("Missingness matrix")
    plt.tight_layout()
    return fig

def plot_missingness_bar(df, figsize=(8,4)):
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    miss.plot(kind='bar', ax=ax)
    ax.set_ylabel('Fraction missing')
    ax.set_ylim(0,1)
    plt.title('Missing fraction by column')
    plt.tight_layout()
    return fig

def plot_time_missing(df, time_col_indexed=True, window='1H', figsize=(14,3)):
    """
    Show number of missing count values over time aggregated by window.
    """
    temp = df.copy()
    if not isinstance(temp.index, pd.DatetimeIndex):
        temp.index = pd.to_datetime(temp['timestamp'])
    miss_counts = temp['count'].isna().astype(int).resample(window).sum()
    fig, ax = plt.subplots(figsize=figsize)
    miss_counts.plot(ax=ax)
    ax.set_ylabel('Missing count occurrences per ' + window)
    plt.title('Missing count occurrences over time')
    plt.tight_layout()
    return fig

def plot_distributions(df, cols=None, figsize=(12,6)):
    """
    Plot histograms + KDEs and boxplots for list of numeric columns.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    rows = int(np.ceil(n/2))
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    for i,c in enumerate(cols):
        sns.histplot(df[c].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram + KDE: {c}")
    plt.tight_layout()
    return fig

def plot_violin_and_box(df, col, by='hour', figsize=(12,4)):
    """
    Compare distribution of 'col' grouped by time (e.g., hour or weekday).
    """
    if by == 'hour' and 'hour' not in df.columns:
        df = add_time_features(df)
    fig, axes = plt.subplots(1,2, figsize=figsize)
    sns.boxplot(x=by, y=col, data=df, ax=axes[0])
    axes[0].set_title(f"Boxplot of {col} by {by}")
    sns.violinplot(x=by, y=col, data=df, ax=axes[1], inner='quartile')
    axes[1].set_title(f"Violin of {col} by {by}")
    plt.tight_layout()
    return fig

def detect_outliers_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    out = (series < lower) | (series > upper)
    return out, lower, upper

def detect_outliers_zscore(series, thresh=3.0):
    z = np.abs(stats.zscore(series.dropna()))
    z = pd.Series(z, index=series.dropna().index)
    out = z > thresh
    return out
def plot_multi_scale_time_series(df, cols=['count'], figsize=(16,6), resample_rules=['10S', '1min', '5min', '1H']):
    """
    Plot same columns at multiple resolutions to visualize micro / macro patterns.
    """
    n = len(resample_rules)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], 3*n))
    for i, rule in enumerate(resample_rules):
        rs = df[cols].resample(rule).mean()
        axes[i].plot(rs.index, rs[cols[0]], lw=0.8)
        axes[i].set_title(f"Resampled ({rule}) mean for {cols}")
    plt.tight_layout()
    return fig

def seasonal_decompose_and_plot(series, freq=None, model='additive'):
    """
    Use statsmodels seasonal_decompose. freq = int number of samples per cycle (e.g., per day).
    If index frequency is irregular, resample first.
    """
    s = series.dropna()
    # attempt to infer period if not given
    if freq is None:
        # e.g., if samples every 10s and we want daily period: 24*3600/10 = 8640
        # instead we let seasonal_decompose infer via period argument if using newer statsmodels
        period = None
    else:
        period = freq
    # if series index is not regularly spaced: resample to a reasonable freq
    # try to infer freq
    if s.index.inferred_freq is None:
        # attempt resample to nearest second
        s_rs = s.resample('10S').mean().interpolate()
    else:
        s_rs = s
    # try decomposition
    try:
        res = seasonal_decompose(s_rs, model=model, period=period or int(len(s_rs)/7))
        fig = res.plot()
        fig.set_size_inches(12,8)
        plt.tight_layout()
        return fig, res
    except Exception as e:
        print("seasonal_decompose failed:", e)
        return None, None

def plot_acf_pacf_series(series, lags=48, figsize=(12,4)):
    fig, axes = plt.subplots(1,2, figsize=figsize)
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm')
    plt.tight_layout()
    return fig

def cross_correlation(x, y, maxlags=50):
    """
    Returns cross-correlation values for lags in [-maxlags, maxlags].
    Positive lag means x leads y by lag (i.e., x_t correlates with y_{t+lag}).
    """
    x = (x - np.nanmean(x)) / np.nanstd(x)
    y = (y - np.nanmean(y)) / np.nanstd(y)
    n = len(x)
    corr = np.correlate(np.nan_to_num(x), np.nan_to_num(y), mode='full') / n
    lags = np.arange(-n+1, n)
    center = len(corr)//2
    slice_idx = slice(center-maxlags, center+maxlags+1)
    return lags[slice_idx], corr[slice_idx]

def plot_crosscorr(df, col_x, col_y, maxlags=200, figsize=(10,4)):
    x = df[col_x].dropna()
    y = df[col_y].dropna()
    # align indexes by inner join
    both = df[[col_x, col_y]].dropna()
    lags, corr = cross_correlation(both[col_x].values, both[col_y].values, maxlags=maxlags)
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(lags, corr, use_line_collection=True)
    ax.set_xlabel('lag')
    ax.set_ylabel('cross-correlation')
    ax.set_title(f'Cross-correlation: {col_x} vs {col_y}')
    plt.tight_layout()
    return fig

def plot_hourly_trends(df, col='count', figsize=(10,4)):
    tmp = df.copy()
    if 'hour' not in tmp.columns:
        tmp = add_time_features(tmp)
    agg = tmp.groupby('hour')[col].agg(['mean','median','std','count'])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(agg.index, agg['mean'], label='mean')
    ax.fill_between(agg.index, agg['mean']-agg['std'], agg['mean']+agg['std'], alpha=0.2, label='±1 std')
    ax.set_xlabel('hour of day')
    ax.set_ylabel(col)
    ax.set_title(f'Hourly trend for {col}')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_weekday_hour_heatmap(df, col='count', figsize=(10,6)):
    tmp = df.copy()
    tmp.index = pd.to_datetime(tmp.index)
    tmp['weekday'] = tmp.index.weekday  # 0=Mon
    tmp['hour'] = tmp.index.hour
    pivot = tmp.pivot_table(values=col, index='weekday', columns='hour', aggfunc='mean')
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, cmap='viridis', ax=ax)
    ax.set_ylabel('weekday (0=Mon)')
    ax.set_xlabel('hour')
    ax.set_title(f'Weekday-hour mean {col} heatmap')
    plt.tight_layout()
    return fig


def plot_corr_matrix(df, numeric_only=True, figsize=(10,8), cluster=False):
    if numeric_only:
        mat = df.select_dtypes(include=[np.number]).corr()
    else:
        mat = df.corr()
    if cluster:
        sns.clustermap(mat, figsize=figsize, cmap='vlag', center=0)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(mat, annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax)
        plt.title("Correlation matrix")
        plt.tight_layout()
        return fig

def visualize_pca(df, features=None, n_comp=3, figsize=(8,6), random_state=0):
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in features:
            features.remove('count')
    X = df[features].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_comp, Xs.shape[1]), random_state=random_state)
    pcs = pca.fit_transform(Xs)
    pc_df = pd.DataFrame(pcs, index=X.index, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(pc_df['PC1'], pc_df['PC2'], c=df.loc[pc_df.index,'count'], cmap='viridis', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='count')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA scatter (colored by count)')
    plt.tight_layout()
    return fig, pca, pc_df

def visualize_umap(df, features=None, n_neighbors=15, min_dist=0.1, n_components=2, random_state=0):
    if not UMAP_AVAILABLE:
        print("UMAP not available. `pip install umap-learn` to enable.")
        return None
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'count' in features:
            features.remove('count')
    X = df[features].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    emb = reducer.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(emb[:,0], emb[:,1], c=df.loc[X.index,'count'], cmap='viridis', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='count')
    ax.set_title('UMAP embedding colored by count')
    plt.tight_layout()
    return fig
def add_time_features(df):
    """Add cyclical time features and other derived features."""
    df = df.copy()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['timestamp'])
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    df['hour'] = idx.hour
    df['minute'] = idx.minute
    df['second'] = idx.second
    # cyclical encoding
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    return df

