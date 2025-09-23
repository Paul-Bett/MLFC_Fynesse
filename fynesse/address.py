"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import logging
import numpy as np
import xgboost as xgb
XGBOOST_AVAILABLE = True

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}



# Example: feature importance for RF
def rf_feature_importance(model, X_train, top_n=20):
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    display(fi.head(top_n))
    fi.head(top_n).plot(kind='bar', figsize=(10,4))
    plt.title('Feature importances')
    plt.tight_layout()

def plot_predictions_vs_truth(y_true, preds_dict, nrows=2, figsize=(16,6)):
    """
    Plot actual vs predicted for multiple models on a shared plot and separate subplots.
    """
    plt.figure(figsize=figsize)
    plt.plot(y_true.values, label='truth', alpha=0.8, lw=1)
    for k, p in preds_dict.items():
        plt.plot(p, label=k, alpha=0.8, lw=1)
    plt.legend()
    plt.title("Predictions vs Truth (indexed by sample order)")
    plt.tight_layout()

def plot_residuals(y_true, preds, figsize=(12,4)):
    plt.figure(figsize=figsize)
    for k, ypred in preds.items():
        resid = y_true.values - ypred
        sns.kdeplot(resid, label=k)
    plt.axvline(0, color='k', linestyle='--')
    plt.title("Residual distributions")
    plt.legend()
    plt.tight_layout()

def plot_metrics_table(metrics_df, title="Model metrics"):
    display(metrics_df.style.background_gradient(cmap='viridis'))


def plot_feature_importance_tree(model, X_train, top_n=20, figsize=(8,4)):
    """
    Plot feature importances for tree model (e.g., RandomForest).
    """
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    fi.head(top_n).plot(kind='bar', ax=ax)
    ax.set_title('Feature importances (tree)')
    plt.tight_layout()
    return fig, fi

def plot_permutation_importance(model, X_valid, y_valid, n_repeats=10, top_n=20, random_state=0):
    res = permutation_importance(model, X_valid, y_valid, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    pi = pd.Series(res.importances_mean, index=X_valid.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    pi.head(top_n).plot(kind='bar', ax=ax)
    ax.set_title('Permutation importance (validation)')
    plt.tight_layout()
    return fig, pi

def detect_change_points(series, model='l2', pen=10):
    """
    Use ruptures library to detect change points in a numeric series.
    """
    if not RUPTURES_AVAILABLE:
        print("ruptures not installed. pip install ruptures to enable.")
        return None
    s = series.dropna().values
    algo = rpt.Pelt(model=model).fit(s)
    bkpts = algo.predict(pen=pen)
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(series.index, series.values)
    for bk in bkpts:
        if bk < len(series):
            ax.axvline(series.index[bk], color='r', linestyle='--')
    ax.set_title('Change points detected')
    plt.tight_layout()
    return fig, bkpts
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

def prepare_features_for_model(df, feature_cols=None, target_col='count', dropna=True):
    dfm = df.copy()
    if feature_cols is None:
        # pick numeric columns except target
        feature_cols = dfm.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
    X = dfm[feature_cols]
    y = dfm[target_col]
    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    return X, y

def train_regressors(X_train, y_train, models=None):
    """
    Train a dictionary of sklearn-like regressors. Return fitted models.
    models: dict name -> estimator instance
    """
    if models is None:
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=0),
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=0)
        }
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, random_state=0, use_label_encoder=False, eval_metric='rmse')
    fitted = {}
    for name, m in models.items():
        print("Training:", name)
        m.fit(X_train, y_train)
        fitted[name] = m
    return fitted

def evaluate_regression_models(models, X_test, y_test):
    rows = []
    preds = {}
    for name, m in models.items():
        ypred = m.predict(X_test)

        # Compute metrics manually
        mse = mean_squared_error(y_test, ypred)   # no squared argument
        rmse = mse ** 0.5
        r2 = r2_score(y_test, ypred)
        mae = mean_absolute_error(y_test, ypred)

        rows.append({
            'model': name,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae
        })
        preds[name] = ypred

    metrics_df = pd.DataFrame(rows).sort_values('RMSE')
    return metrics_df, preds

def create_bins_from_counts(y, bins=[0,1,3,10,100]):
    """
    Convert continuous counts to categorical bins (example boundaries).
    Returns labels array and mapping.
    """
    labels = pd.cut(y, bins=bins, labels=False, right=True, include_lowest=True)
    return labels

def train_classifiers(X_train, y_train_cat, models=None):
    if models is None:
        models = {
            'RF_clf': RandomForestClassifier(n_estimators=200, random_state=0)
        }
    fitted = {}
    for name, m in models.items():
        print("Training classifier:", name)
        m.fit(X_train, y_train_cat)
        fitted[name] = m
    return fitted

def evaluate_classifiers(models, X_test, y_test_cat):
    rows = []
    preds = {}
    for name, m in models.items():
        ypred = m.predict(X_test)
        acc = accuracy_score(y_test_cat, ypred)
        f1 = f1_score(y_test_cat, ypred, average='weighted')
        rows.append({'model': name, 'Accuracy': acc, 'F1_weighted': f1})
        preds[name] = ypred
    return pd.DataFrame(rows).sort_values('F1_weighted', ascending=False), preds


def evaluate_regression_models_extended(models, X_test, y_test):
    rows = []
    preds = {}
    for name, m in models.items():
        ypred = m.predict(X_test)
        rmse = mean_squared_error(y_test, ypred, squared=False)
        r2 = r2_score(y_test, ypred)
        mae = mean_absolute_error(y_test, ypred)
        mape = mean_absolute_percentage_error(y_test, ypred)
        r = pearson_r(y_test, ypred)
        rows.append({'model': name, 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'MAPE':mape, 'Pearson_r': r})
        preds[name] = ypred
    metrics_df = pd.DataFrame(rows).sort_values('RMSE')
    return metrics_df, preds

def evaluate_classification_models(models, X_test, y_test):
    rows = []
    preds = {}
    for name, m in models.items():
        ypred = m.predict(X_test)
        acc = accuracy_score(y_test, ypred)
        f1 = f1_score(y_test, ypred, average='weighted')
        prec = precision_score(y_test, ypred, average='weighted', zero_division=0)
        rec = recall_score(y_test, ypred, average='weighted', zero_division=0)
        rows.append({'model': name, 'Accuracy': acc, 'F1_weighted': f1, 'Precision': prec, 'Recall': rec})
        preds[name] = ypred
    return pd.DataFrame(rows).sort_values('F1_weighted', ascending=False), preds

# plotting helpers for extra diagnostics
def plot_pred_vs_true(y_true, preds_dict, sample_limit=500, figsize=(14,4)):
    # align lengths by index order, plot first sample_limit points
    n = min(len(y_true), sample_limit)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(n), y_true.values[:n], label='truth', lw=1.5)
    for k,p in preds_dict.items():
        plt.plot(np.arange(n), np.array(p)[:n], label=k, alpha=0.9)
    plt.legend()
    plt.title('Predictions vs truth (first {} samples)'.format(n))
    plt.tight_layout()

def plot_residuals_vs_pred(y_true, preds_dict, figsize=(12,4)):
    plt.figure(figsize=figsize)
    for name, p in preds_dict.items():
        resid = y_true.values - p
        plt.scatter(p, resid, s=10, alpha=0.6, label=name)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residual (true - pred)')
    plt.legend()
    plt.title('Residuals vs Predictions')
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6,5)):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues', normalize='true')
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.title('Confusion matrix (normalized)')
    plt.tight_layout()
    return fig

def plot_calibration_curve(model, X_val, y_val, n_bins=10, figsize=(6,4)):
    # only for probabilistic classifiers
    if not hasattr(model, "predict_proba"):
        print("Model has no predict_proba; cannot plot calibration.")
        return None
    prob_pos = model.predict_proba(X_val)
    # For multiclass — compute calibration per class (plot for top classes)
    n_classes = prob_pos.shape[1]
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(min(n_classes,3)):  # plot up to 3 classes
        prob_k = prob_pos[:, k]
        # create binary ground truth for class k
        y_bin = (y_val == k).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, prob_k, n_bins=n_bins, strategy='uniform')
        ax.plot(mean_pred, frac_pos, marker='o', label=f'class {k}')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend()
    ax.set_title('Calibration (per-class)')
    plt.tight_layout()
    return fig





