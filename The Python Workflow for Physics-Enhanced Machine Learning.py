# The Python Workflow for Physics-Enhanced Machine Learning

# Currently untitled, but in future versions a name will be given

# Code developed by Ryan Cheung as part of a 3rd Year Individual Project at Lancaster University
# Title: Developing a Python workflow to aid in Physics-Enhanced Machine Learning applications in engineering
# Date: Friday, March 28th 2025
# Python Version used: 3.12.6
# Dependencies: See requirements.txt

'''
Description:

This script implements a highly customisable workflow for regression tasks using machine learning.
It incorporates a wide range of features suited for engineering problems and physics-enhanced machine learning.
The workflow includes:

- Box and scatter plots of features and target variables
- Distance correlation matrix for feature selection
- Model training and evaluation of multiple regression algorithms using scikit-learn
- Bootstrapping and conformal prediction uncertainty quantification
- PDP, ICE and SHAP plots for model interpretability
- Hyperparameter optimisation methods such as Random Search, Hyperopt, and Scikit-Optimize
- Model comparison and result exporting using Pickle
- Postprocessing and Cross-Validation
- Cook's Distance plots and influential point identification
- Histogram of residuals, automatic transformation of residuals, and Q-Q plots
'''
# Usage:
# Make sure to install of the dependencies as described in requirements.txt
# Run the script from the terminal or an IDE of your choice.
# Ensure you read the README file beforehand for more detailed instructions.

# License: MIT License 

#%%
# -------------------------------------------------------------------------------------------------
# 1. Initialisation and System Information
# -------------------------------------------------------------------------------------------------

# Import all of the necessary libraries:
# System information
import os
import sys
import platform
import multiprocessing
import cpuinfo
import psutil
import torch
import win32com.client

# General libraries needed for functions, mathematical operations and plotting
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import dcor
import seaborn as sns

# Scikit-learn libraries needed to perform ML evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.utils import resample

# The models
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# More plotting
from tabulate import tabulate
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import shap
from pprint import pprint 

# Parallelisation, progress bars, etc
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools

# Hyperparameter optimisation
from scipy.stats.qmc import Sobol, Halton, LatinHypercube
from scipy.optimize import dual_annealing
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Cross validation
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, LeaveOneOut, LeavePOut, ShuffleSplit, cross_val_score

# Stuff needed for postprocessing
import scipy.stats as stats
from scipy.stats import boxcox, anderson, skew, kurtosis, probplot, norm
from sklearn.preprocessing import PowerTransformer
import statsmodels.api as sm
import pickle

# To stop all the warnings and info
import warnings
warnings.filterwarnings("ignore")

class SuppressLibraryLogs:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# System information
def gather_system_info():
    system_info = {
        "Feature": [
            "Operating System", 
            "Processor", 
            "CPU",
            "Number of Physical Cores",  
            "Threads", 
            "Python Version", 
            "RAM"
        ],
        "Details": [
            f"{platform.system()} {platform.release()}",
            platform.processor(),
            cpuinfo.get_cpu_info().get("brand_raw", "Unknown"),
            psutil.cpu_count(logical=False),  # Number of physical cores
            psutil.cpu_count(logical=True),   # Number of threads
            sys.version.split()[0],
            f"{round(psutil.virtual_memory().total / 1e9, 2)} GB"
        ]
    }

    # GPU Detection using WMI
    try:
        wmi = win32com.client.GetObject("winmgmts:")
        gpu_list = []
        for gpu in wmi.InstancesOf("Win32_VideoController"):
            gpu_list.append(gpu.Name.strip())

        if gpu_list:
            gpu_details = ", ".join(gpu_list)
        else:
            gpu_details = "No dedicated GPU detected"
    except Exception as e:
        gpu_details = f"GPU detection failed: {e}"

    system_info["Feature"].append("GPU")
    system_info["Details"].append(gpu_details)

    return system_info


# Function to display system information as a DataFrame
def display_system_info():
    info = gather_system_info()
    df = pd.DataFrame(info)
    print(df.to_string(index=False))


#%%
# -------------------------------------------------------------------------------------------------
# 1. Display System Information
# -------------------------------------------------------------------------------------------------

# Used for recording the total time taken for the whole program to run
start_time_program = time.time()

# Display the system information
print("System Information:\n")
display_system_info()
print(
    "Note: GPU acceleration in this workflow is optimised for NVIDIA GPUs using CUDA.\n"
    "This is because popular Machine Learning frameworks like PyTorch rely on CUDA, a proprietary technology developed by NVIDIA for GPU acceleration.\n"
    "While there are alternative frameworks and libraries, such as ROCm for AMD GPUs or oneAPI for Intel GPUs, these are not yet universally supported or integrated in many ML workflows.\n"
    "As a result, this workflow defaults to CUDA for GPU acceleration.\n\n"
    "For systems with AMD GPUs, users may explore ROCm for compatibility with specific frameworks. Similarly, Intel GPU users can consider Intel oneAPI. Note that additional setup may be required to enable GPU support with these alternatives.\n"
    "If no compatible GPU is detected, the workflow will default to using the CPU, which may significantly increase computation time.\n\n"
    "For more details on GPU support, you can explore the following resources:\n"
    "CUDA (NVIDIA): https://docs.nvidia.com/cuda/ \n"
    "ROCm (AMD): https://rocm.docs.amd.com/en/latest/ \n"
    "oneAPI (Intel): https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html \n"
)

#%%
# -------------------------------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# -------------------------------------------------------------------------------------------------

def load_and_preprocess_data(filepath, test_size, split_method="random", target_columns=None):
    df = pd.read_csv(filepath)

    # Automatically make the last column the target variable if none are specified
    if target_columns is None:
        target_columns = df.columns[-1:]

    # Split into features (X) and target variables (y)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    feature_names = X.columns.tolist()

    # Split data based on the chosen method
    if split_method.lower() == "random":
        # Random split using train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    elif split_method.lower() == "first":
        # Use the first 'test_size' proportion of rows as the test set
        test_count = int(np.ceil(test_size * len(X)))
        X_test = X.iloc[:test_count]
        y_test = y.iloc[:test_count]
        X_train = X.iloc[test_count:]
        y_train = y.iloc[test_count:]
    elif split_method.lower() == "last":
        # Use the last 'test_size' proportion of rows as the test set
        test_count = int(np.ceil(test_size * len(X)))
        X_test = X.iloc[-test_count:]
        y_test = y.iloc[-test_count:]
        X_train = X.iloc[:-test_count]
        y_train = y.iloc[:-test_count]
    else:
        raise ValueError("split_method must be 'random', 'first', or 'last'.")

    # Standardise features (mean = 0, variance = 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, target_columns, feature_names

def plot_target_vs_target(y_train, y_test, target_columns):
    if len(target_columns) < 2:
        print("Not enough target variables specified to plot graph of target variables.")
        return

    # Scatter plot of target variables
    target1, target2 = target_columns[:2]  # Use first two for now
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train[target1], y_train[target2], color='black', alpha=0.5, label=f'Training Data (n={len(y_train)})')
    plt.scatter(y_test[target1], y_test[target2], color='red', alpha=0.5, label=f'Testing Data (n={len(y_test)})')
    plt.xlabel(target1)
    plt.ylabel(target2)
    plt.title(f'Train/Test Split of {target1} vs {target2}')
    plt.legend()
    plt.show()

def plot_features_vs_targets(X_train, y_train, target_columns):
    for target_var in target_columns:
        num_features = X_train.shape[1]
        num_cols = 3  # Fixed number of columns
        num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows dynamically

        # Dynamically adjust figure size based on the number of rows
        fig_width = 15  # Fixed width
        row_height = 4  # Height per row
        fig_height = min(row_height * num_rows, 50)  # Prevent excessive figure size

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f"Scatter Plots of features against {target_var}", fontsize=16)
        axes = axes.flatten()

        for i, column in enumerate(X_train.columns):
            ax = axes[i]
            ax.scatter(X_train[column], y_train[target_var], alpha=0.5)

            # Fit and plot a regression line
            slope, intercept = np.polyfit(X_train[column], y_train[target_var], 1)
            ax.plot(X_train[column], slope * X_train[column] + intercept, color='red')

            ax.set_xlabel(column)
            ax.set_ylabel(target_var)
            ax.set_title(f'{column} vs {target_var}')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def plot_boxplots(df, target_columns):
    # Separate features and targets
    features = df.drop(columns=target_columns)
    targets = df[target_columns]

    # Combine features and targets for plotting
    combined_df = pd.concat([features, targets], axis=1)

    # Get column names for the combined data
    all_columns = combined_df.columns.tolist()
    num_columns = len(all_columns)

    # Dynamic layout
    num_cols = 3  # Number of columns for subplots
    num_rows = (num_columns + num_cols - 1) // num_cols  # Dynamically calculate rows
    max_height_per_row = 5  # Maximum height for each row of subplots

    # Adjust figure size dynamically
    fig_height = min(max_height_per_row * num_rows, 50)  # Limit overall height
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, fig_height), sharey=True)
    fig.suptitle("Box Plots of Features and Target Variables", fontsize=16)
    axes = axes.flatten()

    for i, column in enumerate(all_columns):
        ax = axes[i]
        data = combined_df[column]

        # Plot the box plot
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        flierprops=dict(marker='o', color='red', alpha=0.5))

        # Add statistical information
        median = data.median()
        mean = data.mean()
        q1, q3 = data.quantile([0.25, 0.75])
        p5, p95 = data.quantile([0.05, 0.95])
        min_val, max_val = data.min(), data.max()

        # Draw vertical lines for the mean, median, and percentiles
        ax.axvline(mean, color='red', linestyle='-', linewidth=1.5)
        ax.axvline(median, color='orange', linestyle='-', linewidth=1.5)
        ax.axvline(p5, color='green', linestyle='--', linewidth=1)
        ax.axvline(p95, color='green', linestyle='--', linewidth=1)

        ax.set_title(column, fontsize=12)
        ax.set_xlabel("Value")
        ax.set_yticks([])

        # Custom legend handles (using matplotlib.lines and patches)
        legend_handles = [
            mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label=f"Mean: {mean:.2f}"),
            mlines.Line2D([], [], color='orange', linestyle='-', linewidth=1.5, label=f"Median: {median:.2f}"),
            mpatches.Patch(facecolor='lightblue', edgecolor='black', label=f"IQR: {q1:.2f} to {q3:.2f}"),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label=f"5th/95th: {p5:.2f}, {p95:.2f}"),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label=f"Min/Max: {min_val:.2f}, {max_val:.2f}")
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, title="Statistics", title_fontsize='9')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#%%
# -------------------------------------------------------------------------------------------------
# 2. Display Scatter Plots and Run Analysis
# -------------------------------------------------------------------------------------------------

# Option to change train/test split
test_size = 0.2

# Insert link to dataset here
file_path = # Insert link to dataset here, in .csv format

print("\nAvailable columns in the dataset:")
df_preview = pd.read_csv(file_path)
print(df_preview.columns.tolist())

# User selects target columns
target_columns = input("Enter target columns as a comma-separated list (e.g. A,B,C) or press Enter to use the last two columns: ")
if target_columns.strip():
    target_columns = [col.strip() for col in target_columns.split(",")]
else:
    target_columns = df_preview.columns[-2:].tolist()

# User selects split method: random, first, or last
split_method = input("Enter split method: 'random' for a random split, 'first' for the first entries as testing, or 'last' for the last entries testing: ").strip().lower()
if split_method not in ["random", "first", "last"]:
    print("Invalid input for split method. Defaulting to random split.")
    split_method = "random"

# Load and preprocess data using the selected split method
df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, target_columns, feature_names = load_and_preprocess_data(
    file_path, test_size=test_size, split_method=split_method, target_columns=target_columns)

# Print dataset info
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"Using the following target columns: {target_columns}")

# Plot first two target variables
plot_target_vs_target(y_train, y_test, target_columns)

# Ensure X_train is in DataFrame format for column names
X_train_df = pd.DataFrame(X_train, columns=df.columns.drop(target_columns))

# Plot features vs targets
print("\nGenerating Scatter Plots...")
plot_features_vs_targets(X_train_df, y_train, target_columns)

# Display box plots for features and target variables
print("\nGenerating Box Plots...")
plot_boxplots(df, target_columns)


#%%
# -------------------------------------------------------------------------------------------------
# 3. Distance Correlation Matrix
# -------------------------------------------------------------------------------------------------

def plot_distance_correlation_matrix(df, title="Distance Correlation Matrix", cmap='RdYlGn', dummy=False, annotate=True):
    # Option to add a dummy feature first
    if dummy:
        df = df.copy()
        df["Dummy"] = np.random.normal(size=len(df))
    
    # All columns must be numeric
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns in the dataset must be numeric for distance correlation calculation.")
    
    # Extract features and initialise distance correlation matrix
    features = df.columns
    n = len(features)
    dist_corr_matrix = np.zeros((n, n))

    # Calculate distance correlation
    for i in range(n):
        for j in range(n):
            dist_corr_matrix[i, j] = dcor.distance_correlation(df[features[i]], df[features[j]])

    # Convert to DataFrame
    dist_corr_df = pd.DataFrame(dist_corr_matrix, index=features, columns=features)

    # Dynamically adjust figure size
    fig_width = max(12, n * 0.5)  # Base width with a minimum of 12
    fig_height = max(10, n * 0.5)  # Base height with a minimum of 10

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(dist_corr_df, annot=annotate, cmap=cmap, square=True, linewidths=0.5, fmt=".4f", 
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    
    # Rotate axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    return dist_corr_df

#%%
# -------------------------------------------------------------------------------------------------
# 3. Plot Distance Correlation Matrix
# -------------------------------------------------------------------------------------------------

dist_corr_df = plot_distance_correlation_matrix(df, title="Distance Correlation Matrix", dummy=True)

#%%
# -------------------------------------------------------------------------------------------------
# 4. Model Training and Evaluation
# -------------------------------------------------------------------------------------------------

# Suppress logs
os.environ['LIGHTGBM_VERBOSE'] = '0'
os.environ['XGBOOST_VERBOSITY'] = '0'

# Define metrics to be analysed
metrics_dict = {
    "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    "R^2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    "ADJUSTED R^2": lambda y_true, y_pred, n, p: 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1),
    "Q^2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
} # More can be added later, but make sure they get changed throughout the code!

# Define default hyperparameters 
# This is here to add at the end, there probably is a less bodgey way of coding this though

default_hyperparameters = {
    "SVR (RBF)": {
        "C": 1.0, 
        "epsilon": 0.01, 
        "gamma": 0.01
    },
    "Random Forest Regressor": {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "Gaussian Process Regressor": {"alpha": 1e-6},
    "XGBoost Regressor": {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    "HistGradientBoosting Regressor": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_depth": None,
        "min_samples_leaf": 20,
    },
    "LGBM Regressor": {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
    },
    "Bagging Regressor": {
        "n_estimators": 10, 
        "max_samples": 1.0
    },
    "MLP Regressor": {
        "hidden_layer_sizes": (100,),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "max_iter": 1000,
    },
    "KNeighbors Regressor": {
        "n_neighbors": 5, 
        "p": 2
    },
    "Extra Trees Regressor": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
    },
}

# Note from RC: There is no 'correct' range of parameter spaces, just add the range that you feel is appropriate.
# Obviously the bigger the range the longer the runtime, sometimes by a lot
# Adding non-numerical hyperparameters (such as type of kernel) breaks the code.

param_spaces = {
    "SVR (RBF)": {
        "C": {"type": "loguniform", "bounds": (1e-3, 1e3)},
        "epsilon": {"type": "uniform", "bounds": (0.01, 0.1)},
        "gamma": {"type": "loguniform", "bounds": (1e-4, 1e1)},
    },
    "Random Forest Regressor": {
        "n_estimators": {"type": "int", "bounds": (100, 1000)},
        "max_depth": {"type": "int", "bounds": (10, 100)},
        "max_features": {"type": "uniform", "bounds": (0.1, 1.0)},
        "min_samples_split": {"type": "int", "bounds": (2, 20)},
        "min_samples_leaf": {"type": "int", "bounds": (1, 20)},
    },
    "Gaussian Process Regressor": {
        "alpha": {"type": "loguniform", "bounds": (1e-10, 1e-1)},
    },
    "XGBoost Regressor": {
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "n_estimators": {"type": "int", "bounds": (50, 200)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "subsample": {"type": "uniform", "bounds": (0.5, 1.0)},
        "colsample_bytree": {"type": "uniform", "bounds": (0.5, 1.0)},
    },
    "HistGradientBoosting Regressor": {
        "max_iter": {"type": "int", "bounds": (50, 300)},
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "min_samples_leaf": {"type": "int", "bounds": (1, 20)},
    },
    "LGBM Regressor": {
        "learning_rate": {"type": "uniform", "bounds": (0.01, 0.2)},
        "n_estimators": {"type": "int", "bounds": (50, 300)},
        "max_depth": {"type": "int", "bounds": (3, 30)},
        "num_leaves": {"type": "int", "bounds": (10, 50)},
    },
    "Bagging Regressor": {
        "n_estimators": {"type": "int", "bounds": (10, 200)},
        "max_samples": {"type": "uniform", "bounds": (0.5, 1.0)},
    },
    "MLP Regressor": {
        "hidden_layer_sizes": {"type": "int", "bounds": (50, 300)},
        "alpha": {"type": "loguniform", "bounds": (1e-6, 1e-2)},
        "learning_rate_init": {"type": "loguniform", "bounds": (1e-4, 1e-1)},
        "max_iter": {"type": "int", "bounds": (500, 2000)},
    },
    "KNeighbors Regressor": {
        "n_neighbors": {"type": "int", "bounds": (1, 50)},
        "p": {"type": "int", "bounds": (1, 5)},
    },
    "Extra Trees Regressor": {
        "n_estimators": {"type": "int", "bounds": (50, 500)},
        "max_depth": {"type": "int", "bounds": (10, 100)},
        "min_samples_split": {"type": "int", "bounds": (2, 20)},
    },
}

# Function to reset models to their default parameters
def reset_model_to_defaults(model_name):
    model_map = {
        "SVR (RBF)": SVR(kernel='rbf'),
        "Random Forest Regressor": RandomForestRegressor(random_state=0),
        "Gaussian Process Regressor": GaussianProcessRegressor(),
        "XGBoost Regressor": XGBRegressor(objective="reg:squarederror"),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(),
        "LGBM Regressor": LGBMRegressor(),
        "Bagging Regressor": BaggingRegressor(),
        "MLP Regressor": MLPRegressor(),
        "KNeighbors Regressor": KNeighborsRegressor(),
        "Extra Trees Regressor": ExtraTreesRegressor(),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    return model_map[model_name].set_params(**default_hyperparameters[model_name])


# Train and evaluate the model (Mostly taken from LazyPredict)
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics):
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train[target_var])
    y_pred = model.predict(X_test)

    # Get dataset dimensions
    n, p = X_test.shape
    if n <= 1 or p < 1:
        raise ValueError(f"Invalid dimensions for Adjusted R^2: n={n}, p={p}.")

    # Compute metrics
    metrics_results = {
        metric_name: (
            metric_func(y_test[target_var], y_pred, n, p)
            if metric_name == "ADJUSTED R^2" 
            else metric_func(y_test[target_var], y_pred)
        )
        for metric_name, metric_func in selected_metrics.items()
    }

    metrics_results["Time Elapsed (s)"] = time.time() - start_time
    return metrics_results

# Run models for evaluation
def run_models(models_dict, X_train, X_test, y_train, y_test, target_columns, selected_metrics):
    results = []
    default_metrics = {}
    default_params = {}
    for model_name, model in models_dict.items():
        default_metrics[model_name] = {}
        default_params[model_name] = {}

        for target_var in target_columns:
            print(f"Training {model_name} for target variable '{target_var}'...")
            model = reset_model_to_defaults(model_name)
            metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, target_var, selected_metrics)

            default_metrics[model_name][target_var] = {metric: metrics[metric] for metric in selected_metrics.keys()}
            default_params[model_name][target_var] = default_hyperparameters[model_name]

            results.append({
                "Model": model_name,
                "Target Variable": target_var,
                **metrics
            })

    # Save the summary as a separate DataFrame
    model_performance_summary = pd.DataFrame(results)

    return pd.DataFrame(results), default_metrics, default_params, model_performance_summary

#%%
# -------------------------------------------------------------------------------------------------
# 4. Display ML Model Results
# -------------------------------------------------------------------------------------------------

# Define models dictionary
models_dict = {
    "SVR (RBF)": SVR(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gaussian Process Regressor": GaussianProcessRegressor(),
    "XGBoost Regressor": XGBRegressor(objective="reg:squarederror", verbosity=0),
    "HistGradientBoosting Regressor": HistGradientBoostingRegressor(),
    "LGBM Regressor": LGBMRegressor(verbosity=-1, verbose=-1),
    "Bagging Regressor": BaggingRegressor(),
    "MLP Regressor": MLPRegressor(max_iter=1000),
    "KNeighbors Regressor": KNeighborsRegressor(),
    "Extra Trees Regressor": ExtraTreesRegressor(),
}

# Dynamically filter based on selected_model_names
selected_model_names = [
    "SVR (RBF)",
    "Random Forest Regressor",
    "Gaussian Process Regressor",
    "XGBoost Regressor",
    "HistGradientBoosting Regressor",
    "LGBM Regressor",
    "Bagging Regressor",
    "MLP Regressor",
    "KNeighbors Regressor",
    "Extra Trees Regressor"
]  # Update this list to dynamically include/exclude models

models_dict = {name: model for name, model in models_dict.items() if name in selected_model_names}

# Specify selected metrics
selected_metrics = {key: metrics_dict[key] for key in ["MSE", "R^2", "ADJUSTED R^2", "Q^2"]}

# Run filtered models
if not models_dict:
    print("No valid models selected. Check the selected_model_names list.")
else:
    results_df, default_metrics, default_params, model_performance_summary = run_models(
        models_dict, X_train_scaled, X_test_scaled, y_train, y_test, target_columns, selected_metrics
    )
    print("\nModel Performance Summary:")
    print(tabulate(model_performance_summary, headers="keys", tablefmt="grid", showindex="never"))

# %%
# -------------------------------------------------------------------------------------------------
# 5. Bootstrapping and Conformal Predictions
# -------------------------------------------------------------------------------------------------

def bootstrap_uncertainty(model, X_train, y_train, X_test, target_var, n_bootstrap, confidence_interval):
    predictions = []
    
    for i in range(n_bootstrap):
        # Resample training data, with replacement, fit and predict
        X_resampled, y_resampled = resample(X_train, y_train[target_var], random_state=i)
        model.fit(X_resampled, y_resampled)
        predictions.append(model.predict(X_test))
    
    # Convert into array
    predictions = np.array(predictions)
    predictions_mean = predictions.mean(axis=0)

    # Calculate confidence interval bounds
    lower_percentile = (100 - confidence_interval) / 2
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    
    return predictions_mean, lower_bound, upper_bound

def conformal_predictions(model, X_train, y_train, X_test, target_var, calibration_frac):

    # Split into calibration and residual sets
    X_calib, X_residual, y_calib, y_residual = train_test_split(
        X_train, y_train[target_var], test_size=calibration_frac)
    model.fit(X_residual, y_residual)
    residuals = np.abs(y_calib - model.predict(X_calib))
    quantile = np.quantile(residuals, 1 - calibration_frac)
    
    test_preds = model.predict(X_test)

    # Calculate prediction intervals
    lower_bound = test_preds - quantile
    upper_bound = test_preds + quantile
    
    return lower_bound, upper_bound, test_preds, calibration_frac

def perform_uncertainty_quantification_for_model(
    uq_method, model, X_train, y_train, X_test, target_var, n_bootstrap, calibration_frac, subsample_test_size, confidence_interval, model_name
):
    # Subsample test data for visualisation
    if len(X_test) > subsample_test_size:
        test_indices = np.random.choice(len(X_test), subsample_test_size, replace=False)
        X_test_subsample = X_test[test_indices]
    else:
        X_test_subsample = X_test

    results = {}

    if uq_method in ["Bootstrapping", "Both"]:
        # Perform Bootstrapping
        predictions_mean, lower_bound_bs, upper_bound_bs = bootstrap_uncertainty(
            model, X_train, y_train, X_test_subsample, target_var, n_bootstrap, confidence_interval
        )
        results["Bootstrapping"] = (predictions_mean, lower_bound_bs, upper_bound_bs)

    if uq_method in ["Conformal", "Both"]:
        # Perform Conformal Predictions
        lower_bound_cp, upper_bound_cp, test_preds, calibration_frac = conformal_predictions(
            model, X_train, y_train, X_test_subsample, target_var, calibration_frac
        )
        results["Conformal"] = (test_preds, lower_bound_cp, upper_bound_cp)

    return results, X_test_subsample


def plot_uncertainty_results_for_model(results_by_target, model_name, uq_method, target_columns, confidence_interval, calibration_frac):
    num_targets = len(results_by_target)
    methods = []
    if uq_method == "Both":
        methods = ["Bootstrapping", "Conformal"]
    elif uq_method == "Bootstrapping":
        methods = ["Bootstrapping"]
    elif uq_method == "Conformal":
        methods = ["Conformal"]
    num_methods = len(methods)

    # Create subplots dynamically
    fig, axes = plt.subplots(
        num_targets,
        num_methods,
        figsize=(8 * num_methods, 5 * num_targets),
        squeeze=False
    )
    fig.suptitle(f"Uncertainty Quantification for {model_name}", fontsize=16)

    for row_idx, target_var in enumerate(results_by_target):
        result_data = results_by_target[target_var]

        for col_idx, method in enumerate(methods):
            if method in result_data:
                ax = axes[row_idx, col_idx]
                if method == "Bootstrapping":
                    predictions_mean, lower_bound_bs, upper_bound_bs = result_data[method]
                    ax.plot(predictions_mean, label="Mean Prediction", color="blue")
                    ax.fill_between(
                        range(len(predictions_mean)),
                        lower_bound_bs,
                        upper_bound_bs,
                        color="red",
                        alpha=0.2,
                        label=f"{confidence_interval}% Confidence Interval",
                    )
                    ax.set_title(f"Bootstrap Aggregation for {target_var}")
                elif method == "Conformal":
                    test_preds, lower_bound_cp, upper_bound_cp = result_data[method]
                    prediction_interval = (1 - calibration_frac) * 100
                    ax.plot(test_preds, label="Conformal Prediction", color="blue")
                    ax.fill_between(
                        range(len(test_preds)),
                        lower_bound_cp,
                        upper_bound_cp,
                        color="red",
                        alpha=0.2,
                        label=f"{prediction_interval:.0f}% Prediction Interval",
                    )
                    ax.set_title(f"Conformal Prediction for {target_var}")
                ax.set_xlabel("Sample #")
                ax.set_ylabel(f"Predicted {target_var}")
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
#%%
# -------------------------------------------------------------------------------------------------
# 5. Plot Bootstrapping and Conformal Prediction graphs
# -------------------------------------------------------------------------------------------------

# Define UQ options
uq_settings = {
    "uq_method": "Both",  # Options: "Bootstrapping", "Conformal", or "Both"
    "n_bootstrap": 50,   # Number of bootstrap iterations
    "confidence_interval": 95,  # Confidence interval percentage for bootstrapping
    "calibration_frac": 0.05,  # Fraction of calibration set for conformal predictions
    "subsample_test_size": 50,  # Subsample test size for visualisation
}

# Apply UQ to all models and targets
for model_name, model in models_dict.items():
    print(f"\nPerforming Uncertainty Quantification for {model_name}...")
    results_by_target = {}  # Store results for each target variable
    for target_var in target_columns:
        results, X_test_subsample = perform_uncertainty_quantification_for_model(
            uq_method=uq_settings["uq_method"],
            model=model,
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            target_var=target_var,
            n_bootstrap=uq_settings["n_bootstrap"],
            calibration_frac=uq_settings["calibration_frac"],
            subsample_test_size=uq_settings["subsample_test_size"],
            confidence_interval=uq_settings["confidence_interval"],
            model_name=model_name
        )
        results_by_target[target_var] = results

    # Plot combined results for all targets of the model
    plot_uncertainty_results_for_model(
        results_by_target,
        model_name,
        uq_settings["uq_method"],
        target_columns,
        uq_settings["confidence_interval"],
        uq_settings["calibration_frac"]
    )

# %%
# -------------------------------------------------------------------------------------------------
# 6. ICE, PDP and SHAP plots
# -------------------------------------------------------------------------------------------------

# Function to measure training time for models
def get_fastest_model(models_dict, X_train, y_train, target_var, sample_size=1000):
    fastest_model = None
    min_time = float("inf")
    X_sample = X_train[:sample_size]
    y_sample = y_train.iloc[:sample_size]
    
    for model_name, model in models_dict.items():
        try:
            start_time = time.time()
            model.fit(X_sample, y_sample[target_var])
            elapsed_time = time.time() - start_time
            if elapsed_time < min_time:
                fastest_model = model_name
                min_time = elapsed_time
        except Exception as e:
            print(f"Model '{model_name}' failed during training: {str(e)}")
            continue
    
    if not fastest_model:
        raise ValueError("No valid models available for ICE, PDP, and SHAP plots.")
    
    print(f"Fastest model selected: {fastest_model} with training time: {min_time:.4f} seconds")
    return fastest_model

# Function for ICE and PDP Plots
def plot_ice_and_pdp(model, X_train, feature_names, target_var, model_name):
    # Dynamically calculating rows and columns
    num_features = len(feature_names)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Plot each feature
    for i, feature_idx in enumerate(range(num_features)):
        ax = axes[i]
        PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            [feature_idx],
            feature_names=feature_names,
            kind="both",  # 'both' being ICE and PDP plots
            subsample=subsample,
            grid_resolution=grid_resolution,
            percentiles=(0.1, 0.9),  # Adjust to avoid being too close
            ax=ax,
        )
        ax.set_title(f"Feature: {feature_names[feature_idx]}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"ICE and PDP Plots for {target_var} ({model_name})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Function to select the appropriate SHAP explainer
# I can't really explain this too well, but essentially it splits your models into 3 categories, TreeExplainer, GradientExplainer, KernelExplainer
# Each ML regressor can be split into those three (These might have to be added if you add new models)
# If it doesn't fit into any category, it just won't create a SHAP explainer
def select_shap_explainer(model, X_train, background_sample_size):
    try:
        if isinstance(model, (RandomForestRegressor, XGBRegressor)):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_") or hasattr(model, "gradient"):
            explainer = shap.GradientExplainer(model, X_train)
        else:
            background = X_train[
                np.random.choice(X_train.shape[0], background_sample_size, replace=False)
            ]
            explainer = shap.KernelExplainer(model.predict, background)
        return explainer
    except Exception as e:
        raise ValueError(f"Could not create SHAP explainer for model {model}: {str(e)}")

# Function to plot SHAP summary
def plot_shap_summary(model, X_train, feature_names, background_sample_size):
    explainer = select_shap_explainer(model, X_train, background_sample_size)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names)


# Function for SHAP dependence plot
def plot_shap_dependence(model, X_train, feature_names, background_sample_size, target_var, model_name):
    # Select SHAP explainer
    explainer = select_shap_explainer(model, X_train, background_sample_size)
    shap_values = explainer.shap_values(X_train)

    num_features = len(feature_names)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Normalisation
    norm = plt.Normalize(vmin=X_train.min(), vmax=X_train.max())
    cmap = plt.cm.coolwarm

    # Plot dependence plots
    for i, feature_idx in enumerate(range(num_features)):
        ax = axes[i]
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_train,
            feature_names=feature_names,
            ax=ax,
            show=False,  # Suppress individual plots
            color=plt.cm.coolwarm(norm(X_train[:, feature_idx])),  # Shared color scale
        )
        ax.set_title(f"SHAP Dependence: {feature_names[feature_idx]}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Consistent colourbars
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.suptitle(f"SHAP Dependence Plots for {target_var} ({model_name})", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# %%
# -------------------------------------------------------------------------------------------------
# 6. Plot plots
# -------------------------------------------------------------------------------------------------

# Specify a preferred model
preferred_model_name = "XGBoost Regressor"

# Dynamically select the preferred model or fallback to the fastest
target_var = target_columns[0]  # Select the first target variable for model timing

if preferred_model_name in models_dict:
    model_name = preferred_model_name
    print(f"Preferred model selected: {model_name}")
else:
    model_name = get_fastest_model(models_dict, X_train_scaled, y_train, target_var)
    print(f"Fallback to fastest model: {model_name}")

# Check if the selected model is valid
if model_name not in models_dict:
    raise ValueError(f"Model '{model_name}' is not in the models dictionary.")

model = models_dict[model_name]

# Change parameters here
test_sample_size = 1000  # Sample size for plotting
background_sample_size = 20  # Training samples used as background for SHAP
subsample = 250  # Number of samples taken from sample size
grid_resolution = 20  # Number of intervals divided for plotting

X_train_sampled = X_train_scaled[:test_sample_size]
y_train_sampled = y_train.iloc[:test_sample_size]

print(f"\nPerforming Feature Analysis with ICE, PDP, and SHAP Plots for {model_name}...")

# Iterate over all target variables
for target_var in target_columns:
    print(f"Analysing Target Variable: {target_var}")

    model.fit(X_train_sampled, y_train_sampled[target_var])
    feature_names = X_train_df.columns.tolist()

    # Plot everything
    print("\nGenerating ICE and PDP plots...")
    plot_ice_and_pdp(model, X_train_sampled, feature_names, target_var, model_name)

    print("\nGenerating SHAP Summary Plot...")
    plot_shap_summary(model, X_train_sampled, feature_names, background_sample_size)

    print("\nGenerating SHAP Dependence plots...")
    plot_shap_dependence(model, X_train_sampled, feature_names, background_sample_size, target_var, model_name)

#%%
# -------------------------------------------------------------------------------------------------
# 7. Hyperparamter Optimisation using Random Search
# -------------------------------------------------------------------------------------------------

def run_random_search(
    model, param_space, X_train, X_test, y_train, y_test, sample_size, n_iter, n_jobs, target_var, metric, sampling_method
):
    def generate_param_samples(param_space, sampler):
        param_samples = {}
        for i, (param, bounds) in enumerate(param_space.items()):
            if bounds["type"] in ["loguniform", "uniform"]:
                param_samples[param] = bounds["bounds"][0] + (bounds["bounds"][1] - bounds["bounds"][0]) * sampler[:, i]
            elif bounds["type"] == "choice":
                param_samples[param] = [
                    bounds["options"][int(idx)] for idx in sampler[:, i] * len(bounds["options"])
                ]
            elif bounds["type"] == "int":
                param_samples[param] = np.round(
                    bounds["bounds"][0] + (bounds["bounds"][1] - bounds["bounds"][0]) * sampler[:, i]
                ).astype(int)
            elif param == "hidden_layer_sizes":
                # Generate tuple values for MLPRegressor
                layer_size = np.round(
                    bounds["bounds"][0] + (bounds["bounds"][1] - bounds["bounds"][0]) * sampler[:, i]
                ).astype(int)
                param_samples[param] = [(layer_size,)]  # Convert to single-element tuple
        return param_samples

    # Generate parameter samples
    if sampling_method == "Random":
        sampler = np.random.random((n_iter, len(param_space)))
    elif sampling_method == "Sobol":
        sampler = Sobol(d=len(param_space), scramble=True, seed=0).random_base2(m=int(np.log2(n_iter)))
    elif sampling_method == "Halton":
        sampler = Halton(d=len(param_space), scramble=True, seed=0).random(n=n_iter)
    elif sampling_method == "Latin Hypercube":
        sampler = LatinHypercube(d=len(param_space), seed=0).random(n=n_iter)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    param_samples = generate_param_samples(param_space, sampler)
    param_combinations = [dict(zip(param_space.keys(), values)) for values in zip(*param_samples.values())]

    # Subsample training data if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train


    # Function to evaluate a single configuration
    def evaluate(params):
        model.set_params(**params)
        model.fit(X_train_sample, y_train_sample[target_var])
        y_pred = model.predict(X_test)

        # Evaluate the chosen metric
        n, p = X_test.shape
        if metric == "MSE":
            metric_value = mean_squared_error(y_test[target_var], y_pred)
        elif metric == "R^2":
            metric_value = r2_score(y_test[target_var], y_pred)
        elif metric == "ADJUSTED R^2":
            metric_value = 1 - (1 - r2_score(y_test[target_var], y_pred)) * (n - 1) / (n - p - 1)
        elif metric == "Q^2":
            metric_value = 1 - np.sum((y_test[target_var] - y_pred) ** 2) / np.sum(
                (y_test[target_var] - np.mean(y_test[target_var])) ** 2
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return params, metric_value

    # Parallelise evaluations
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(params) for params in param_combinations
    )

    # Compile results into a DataFrame
    results_df = pd.DataFrame(
        [{**params, metric.upper(): metric_value} for params, metric_value in results]
    )

    # Determine whether to maximise or minimise the metric
    if metric in ["MSE"]:
        best_row = results_df.loc[results_df[metric.upper()].idxmin()]
    elif metric in ["R^2", "ADJUSTED R^2", "Q^2"]:
        best_row = results_df.loc[results_df[metric.upper()].idxmax()]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Extract the best parameters and metric value
    best_params = best_row.drop([metric.upper()]).to_dict()
    best_metric_value = best_row[metric.upper()]

    return results_df, best_params, best_metric_value


def plot_random_search_results(results_by_target, param_space, model_name, metric, sampling_method):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots for each target variable
    fig, axes = plt.subplots(num_targets, num_params, figsize=(5 * num_params, 4 * num_targets), sharey=True)
    fig.suptitle(f"Random Search ({sampling_method}) Results for {model_name} - {metric_column} ", fontsize=16)

    if num_targets == 1:
        axes = [axes]  # Make sure axes is iterable if there's only one target
    if num_params == 1:
        axes = [[ax] for ax in axes]  # Adjust for single parameter

    # Plot results for each target variable
    for row_idx, (target_var, results_df) in enumerate(results_by_target.items()):
        for col_idx, (param, config) in enumerate(param_space.items()):
            ax = axes[row_idx][col_idx]
            if config["type"] in ["int", "loguniform", "uniform"]:
                data = results_df[param]
            else:
                data = results_df[param].apply(lambda x: x[0] if isinstance(x, tuple) else x)

            ax.scatter(data, results_df[metric_column], alpha=0.7, c="blue")
            ax.set_xlabel(param)
            ax.set_ylabel(metric_column if col_idx == 0 else "")
            ax.set_title(f"{target_var}: {param} vs {metric_column}")
            ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%%
# -------------------------------------------------------------------------------------------------
# 7. Run and plot Random Search Hyperparamter Optimisation
# -------------------------------------------------------------------------------------------------

# Parameters for Random Search
sample_size = 10000 # Sample size of original dataset
n_iter = 1000 # Number of data points
n_jobs = -1 # Number of logical processors assigned to the task (aka workers). -1 = all
metric = "Q^2" # Choose between "MSE", "R^2", "ADJUSTED R^2", or "Q^2"
sampling_method = "Sobol" # Choose between "Random", "Sobol", "Halton", "Latin Hypercube"

# Dictionaries to store metrics and parameters
random_metrics = {}
random_params = {}

for model_name, model in models_dict.items():
    results_by_target = {}  # Store results for all targets for this model
    for target_var in target_columns:
        print(f"\nRunning Random Search for {model_name} (Target: {target_var}) using {sampling_method} sampling...")
        results_df, best_params, best_metric_value = run_random_search(
            model=model,
            param_space=param_spaces[model_name],
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            sample_size=sample_size,
            n_iter=n_iter,
            n_jobs=n_jobs,
            target_var=target_var,
            metric=metric,
            sampling_method=sampling_method
        )

        # Store results for this target variable
        results_by_target[target_var] = results_df

        # Save metrics and parameters
        if model_name not in random_metrics:
            random_metrics[model_name] = {}
            random_params[model_name] = {}
        random_metrics[model_name][target_var] = {metric.upper(): best_metric_value}
        random_params[model_name][target_var] = best_params

        # Print results
        print(f"Best Hyperparameters for {model_name} (Target: {target_var}): {best_params}")
        print(f"Best {metric.upper()}: {best_metric_value:.4f}")

    # Plot all target variables in separate subplots within the same figure for the current model
    plot_random_search_results(results_by_target, param_spaces[model_name], model_name, metric, sampling_method)


#%%
# -------------------------------------------------------------------------------------------------
# 8. Hyperparameter Optimisation using Hyperopt's TPE
# -------------------------------------------------------------------------------------------------

# Tracking lists to store the best values
def tracking():
    return [], []

# Objective Function for Hyperopt
def hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    # Convert float parameters to integers or tuples where necessary
    for key, value in params.items():
        if param_spaces[model_name][key]["type"] == "int":
            params[key] = int(value)
        elif key == "hidden_layer_sizes":  # Special handling for tuple parameters
            params[key] = (int(value),)

    # Ensure constraints (e.g., min_samples_split > min_samples_leaf)
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in params and "min_samples_leaf" in params:
            if params["min_samples_split"] <= params["min_samples_leaf"]:
                params["min_samples_split"] = params["min_samples_leaf"] + 1

    # Perform dataset subsampling
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Set model parameters and fit
    model.set_params(**params)
    model.fit(X_train_sample, y_train_sample[target_var])
    y_pred = model.predict(X_test)

    # Evaluate the chosen metric
    n, p = X_test.shape
    metric_value = (
        mean_squared_error(y_test[target_var], y_pred) if metric == "MSE" else
        r2_score(y_test[target_var], y_pred) if metric == "R^2" else
        1 - (1 - r2_score(y_test[target_var], y_pred)) * (n - 1) / (n - p - 1) if metric == "ADJUSTED R^2" else
        1 - np.sum((y_test[target_var] - y_pred) ** 2) / np.sum((y_test[target_var] - np.mean(y_test[target_var])) ** 2) if metric == "Q^2" else
        None
    )
    if metric_value is None:
        raise ValueError(f"Unsupported metric: {metric}")

    # For maximising metrics, return the negative value; for minimising, return as-is
    return {"loss": metric_value if metric == "MSE" else -metric_value, "status": STATUS_OK, **params}

# Update best values for tracking
def update_best_values(tracking_lists, params, metric_value, metric_type):
    best_params, best_metric = tracking_lists

    # Revert the negative metric to its original value for maximisation metrics
    metric_value = -metric_value if metric_type in ["R^2", "ADJUSTED R^2", "Q^2"] else metric_value

    if not best_metric:
        best_params.append(params)
        best_metric.append(metric_value)
    else:
        # Update the best value based on the metric type
        if metric_type in ["R^2", "ADJUSTED R^2", "Q^2"]:
            if metric_value > max(best_metric):
                best_params.append(params)
                best_metric.append(metric_value)
            else:
                best_params.append(best_params[-1])
                best_metric.append(max(best_metric))
        else: # This is for MSE
            if metric_value < min(best_metric):
                best_params.append(params)
                best_metric.append(metric_value)
            else:
                best_params.append(best_params[-1])
                best_metric.append(min(best_metric))

# Define Hyperopt-compatible parameter spaces
def convert_to_hyperopt_space(param_space):
    hyperopt_space = {}
    for param, config in param_space.items():
        if config["type"] == "loguniform":
            hyperopt_space[param] = hp.loguniform(param, np.log(config["bounds"][0]), np.log(config["bounds"][1]))
        elif config["type"] == "uniform":
            hyperopt_space[param] = hp.uniform(param, config["bounds"][0], config["bounds"][1])
        elif config["type"] == "int":
            hyperopt_space[param] = hp.quniform(param, config["bounds"][0], config["bounds"][1], 1)
        elif config["type"] == "choice":
            hyperopt_space[param] = hp.choice(param, config["options"])
        elif param == "hidden_layer_sizes":
            hyperopt_space[param] = hp.quniform(param, config["bounds"][0], config["bounds"][1], 1)
    return hyperopt_space


# Run Hyperopt Optimisation
def run_hyperopt_optimisation(model_name, model, param_space, evals, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    trials = Trials()
    tracking_lists = tracking()  # Tracking best parameters and metric

    # Convert param_space to Hyperopt-compatible format
    hyperopt_space = convert_to_hyperopt_space(param_space)

    # Run optimisation
    best_params = fmin(
        fn=lambda params: hyperopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size),
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)

    # Convert integer parameters back from floats if needed
    for key, value in best_params.items():
        if param_space[key]["type"] == "int":
            best_params[key] = int(value)

    # Track the best values during optimisation
    for trial in trials.trials:
        trial_result = {key: val for key, val in trial["result"].items() if key not in ["loss", "status"]}
        update_best_values(tracking_lists, trial_result, trial["result"]["loss"], metric)

    return best_params, tracking_lists

# Plotting for Hyperopt
def plot_hyperopt_results(results_by_target, param_space, model_name, metric):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots for each target variable
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always returns a 2D array
        sharey=False
    )
    fig.suptitle(f"Hyperopt Results for {model_name} - {metric_column}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        param_names = list(best_params[0].keys())  # Hyperparameter names

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(metric_column)
        ax.set_title(f"{target_var}: {metric_column}")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%%
# -------------------------------------------------------------------------------------------------
# 8. Run Hyperopt
# -------------------------------------------------------------------------------------------------

evals = 100 # Number of iterations
sample_size = 10000 # Sample size taken from dataset

# Blank dictionaries to store metrics and parameters from Hyperopt
hyperopt_metrics = {}
hyperopt_params = {}

# Perform Hyperopt for each model and target variable
for model_name, model in models_dict.items():
    results_by_target = {}  # Store results for all targets for this model
    for target_var in target_columns:
        print(f"\nRunning Hyperopt for {model_name} (Target: {target_var})...")

        # Run Hyperopt
        best_params, tracking_lists = run_hyperopt_optimisation(
            model_name=model_name,
            model=model,
            param_space=param_spaces[model_name],
            evals=evals,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            target_var=target_var,
            metric=metric,
            sample_size=sample_size
        )

        # Save results for this target variable
        results_by_target[target_var] = tracking_lists

        # Save metrics and parameters
        if model_name not in hyperopt_metrics:
            hyperopt_metrics[model_name] = {}
            hyperopt_params[model_name] = {}
        best_metric_value = max(tracking_lists[1]) if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else min(tracking_lists[1])
        hyperopt_metrics[model_name][target_var] = {metric.upper(): best_metric_value}
        hyperopt_params[model_name][target_var] = best_params

        # Print results for the current model and target variable
        print(f"Best Parameters for {model_name} (Target: {target_var}): {best_params}")
        print(f"Best {metric.upper()} for {model_name} (Target: {target_var}): {best_metric_value:.4f}")

    # Plot all target variables in separate rows within the same figure for the current model
    plot_hyperopt_results(results_by_target, param_spaces[model_name], model_name, metric)

#%%
# -------------------------------------------------------------------------------------------------
# 9. Hyperparameter Optimisation using Scikit-Optimize (via Gaussian Process Minimisation)
# -------------------------------------------------------------------------------------------------

# Define scikit-optimize's objective function
def skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size):
    # Convert params to dictionary
    param_dict = {param_name: param_value for param_name, param_value in zip(param_spaces[model_name].keys(), params)}

    # Handle parameter types
    for key, value in param_dict.items():
        if param_spaces[model_name][key]["type"] == "int":
            param_dict[key] = int(value)
        elif key == "hidden_layer_sizes":
            param_dict[key] = (int(value),)

    # Enforce constraints
    if model_name == "Random Forest Regressor":
        if "min_samples_split" in param_dict and "min_samples_leaf" in param_dict:
            if param_dict["min_samples_split"] <= param_dict["min_samples_leaf"]:
                param_dict["min_samples_split"] = param_dict["min_samples_leaf"] + 1

    # Subsample the dataset if needed
    if len(X_train) > sample_size:
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=sample_size, random_state=0)
    else:
        X_train_sample, y_train_sample = X_train, y_train

    # Fit the model with the sampled parameters
    model.set_params(**param_dict)
    model.fit(X_train_sample, y_train_sample[target_var])
    y_pred = model.predict(X_test)

    # Calculate metrics
    n, p = X_test.shape
    metric_value = (
        mean_squared_error(y_test[target_var], y_pred) if metric == "MSE" else
        r2_score(y_test[target_var], y_pred) if metric == "R^2" else
        1 - (1 - r2_score(y_test[target_var], y_pred)) * (n - 1) / (n - p - 1) if metric == "ADJUSTED R^2" else
        1 - np.sum((y_test[target_var] - y_pred) ** 2) / np.sum((y_test[target_var] - np.mean(y_test[target_var])) ** 2) if metric == "Q^2" else
        None
    )

    if metric_value is None:
        raise ValueError(f"Unsupported metric: {metric}")

    return metric_value

def run_skopt_optimisation(model_name, model, param_space, calls, X_train, X_test, y_train, y_test, target_var, metric, sample_size, n_jobs):
    tracking_lists = tracking()

    # Define Scikit-Optimize's search space
    scikit_opt_space = [
        Real(bounds["bounds"][0], bounds["bounds"][1], "log-uniform" if bounds["type"] == "loguniform" else "uniform", name=param)
        if bounds["type"] in ["loguniform", "uniform"] else
        Integer(bounds["bounds"][0], bounds["bounds"][1], name=param)
        if bounds["type"] == "int" else
        Categorical(bounds["options"], name=param)
        for param, bounds in param_space.items()
    ]

    with tqdm(total=calls, desc=f"Scikit-Optimize Progress for {model_name} (Target: {target_var})", unit="call") as pbar:
        def callback(res):
            pbar.update()
            metric_value = -res.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else res.fun
            param_dict = {param_name: param_value for param_name, param_value in zip(param_space.keys(), res.x)}
            tracking_lists[0].append(param_dict)
            tracking_lists[1].append(metric_value)

        results = gp_minimize(
            lambda params: -skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size)
            if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else
            skopt_objective(params, model_name, model, X_train, X_test, y_train, y_test, target_var, metric, sample_size),
            scikit_opt_space,
            n_calls=calls,
            n_jobs=n_jobs,
            callback=[callback]
        )

    best_params = {param_name: param_value for param_name, param_value in zip(param_space.keys(), results.x)}
    best_metric_value = -results.fun if metric in ["R^2", "ADJUSTED R^2", "Q^2"] else results.fun

    return best_params, tracking_lists, best_metric_value


def plot_skopt_results(results_by_target, param_space, model_name, metric):
    metric_column = metric.upper()
    num_targets = len(results_by_target)
    num_params = len(param_space)

    # Create subplots: one row per target variable, columns for hyperparameters and the metric
    fig, axes = plt.subplots(
        num_targets,
        num_params + 1,
        figsize=(5 * (num_params + 1), 4 * num_targets),
        squeeze=False,  # Always create a 2D array
        sharey=False
    )
    fig.suptitle(f"Scikit-Optimize Results for {model_name} - {metric_column}", fontsize=16)

    # Plot results for each target variable
    for row_idx, (target_var, tracking_lists) in enumerate(results_by_target.items()):
        best_params, best_metric = tracking_lists
        param_names = list(best_params[0].keys())

        # Plot hyperparameter evolution
        for col_idx, param_name in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            param_values = [params[param_name] for params in best_params]
            ax.plot(range(1, len(param_values) + 1), param_values, marker=".")
            ax.set_xlabel("Iteration #")
            ax.set_ylabel(f"{param_name}")
            ax.set_title(f"{target_var}: {param_name}")
            ax.grid(True)

        # Plot metric evolution
        ax = axes[row_idx, -1]
        ax.plot(range(1, len(best_metric) + 1), best_metric, color="green", marker=".")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel(metric_column)
        ax.set_title(f"{target_var}: {metric_column}")
        ax.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#%%
# -------------------------------------------------------------------------------------------------
# 9. Plot Scikit-Optimize
# -------------------------------------------------------------------------------------------------

# Scikit-Optimize Parameters
calls = 100 # Number of iterations
sample_size = 10000 # Sample size from dataset
n_jobs = -1 # Number of "workers"

# Dictionaries to store metrics and parameters for Scikit-Optimize
skopt_metrics = {}
skopt_params = {}

# Perform Scikit-Optimize for each model and target variable
for model_name, model in models_dict.items():
    skopt_metrics[model_name] = {}
    skopt_params[model_name] = {}
    results_by_target = {}  # Store tracking lists for each target variable

    for target_var in target_columns:
        print(f"\nRunning Scikit-Optimize for {model_name} (Target: {target_var})...")

        # Run Scikit-Optimize for the current target variable
        best_params, tracking_lists, best_metric_value = run_skopt_optimisation(
            model_name=model_name,
            model=model,
            param_space=param_spaces[model_name],
            calls=calls,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            target_var=target_var,
            metric=metric,
            sample_size=sample_size,
            n_jobs=n_jobs
        )

        # Store results for the current target variable
        skopt_metrics[model_name][target_var] = {metric.upper(): best_metric_value}
        skopt_params[model_name][target_var] = best_params
        results_by_target[target_var] = tracking_lists

        # Print the best results
        print(f"Best Parameters for {model_name} (Target: {target_var}): {best_params}")
        print(f"Best {metric.upper()}: {best_metric_value:.4f}")

    # Plot all target variables in separate rows within the same figure for the current model
    plot_skopt_results(results_by_target, param_spaces[model_name], model_name, metric)
#%%
# -------------------------------------------------------------------------------------------------
# 10. Comparing and Plotting Optimisation Methods
# -------------------------------------------------------------------------------------------------

# Function to compare and plot optimisation methods
def compare_and_plot_optimisation_methods(
    model_name, target_var,
    default_metrics, default_params,
    random_metrics, random_params, random_sampling_method,
    hyperopt_metrics, hyperopt_params,
    skopt_metrics, skopt_params,
    param_names,
    metric_name
):
    # Define methods
    methods = [
        "Default",
        f"Random Search ({random_sampling_method})",
        "Hyperopt (TPE)",
        "Scikit-Optimize (GP Minimisation)"
    ]
    metrics = [default_metrics, random_metrics, hyperopt_metrics, skopt_metrics]
    params = [default_params, random_params, hyperopt_params, skopt_params]

    # Extract metric values
    metric_values = [
        m.get(metric_name, float('inf')) if m else float('inf') 
        for m in metrics
    ]

    # Extract parameter values
    param_data = {
        param: [
            p.get(param, None) if isinstance(p, dict) else None 
            for p in params
        ]
        for param in param_names
    }

    # Plotting
    x = np.arange(len(methods))  # Positions for the methods
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bars for each parameter
    for i, param in enumerate(param_names):
        values = [
            np.log10(value) if isinstance(value, (int, float)) and value > 0 else 0
            for value in param_data[param]
        ]
        bar_positions = x + (i - len(param_names) / 2) * bar_width
        bars = ax.bar(
            bar_positions,
            values,
            bar_width,
            label=f'Log({param})' if any(v > 0 for v in values) else param
        )

        # Add labels to bars
        for bar, orig_value in zip(bars, param_data[param]):
            if orig_value is not None:
                value_str = (
                    f"{orig_value:.2e}" if isinstance(orig_value, (float, int)) else str(orig_value)
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    value_str,
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

    # Plot bars for the metric
    metric_positions = x + (len(param_names) / 2) * bar_width
    metric_bars = ax.bar(
        metric_positions,
        metric_values,
        bar_width,
        label=metric_name.upper(),
        color='lightgreen'
    )

    # Add labels to metric bars
    for bar, value in zip(metric_bars, metric_values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Labels, legend, and title
    ax.set_xlabel('Optimisation Method')
    ax.set_ylabel('Values (log scale for parameters)')
    ax.set_title(f'Comparison of Hyperparameter Optimisation Methods\n{model_name} (Target: {target_var}, Metric: {metric_name.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.tight_layout()
    plt.show()

    # Determine optimal parameters
    optimal_idx = np.argmin(metric_values) if metric_name.lower() in ["mse", "mae"] else np.argmax(metric_values)
    return params[optimal_idx], metrics[optimal_idx]

#%%
# -------------------------------------------------------------------------------------------------
# 10. Run Comparison and Plot for Each Model
# -------------------------------------------------------------------------------------------------

all_results = []

for model_name in models_dict.keys():
    for target_var in target_columns:
        # Safely retrieve metrics and parameters, defaulting to empty dictionaries if missing
        optimal_params, optimal_metrics = compare_and_plot_optimisation_methods(
            model_name=model_name,
            target_var=target_var,
            default_metrics=default_metrics.get(model_name, {}).get(target_var, {}),
            default_params=default_params.get(model_name, {}).get(target_var, {}),
            random_metrics=random_metrics.get(model_name, {}).get(target_var, {}),
            random_params=random_params.get(model_name, {}).get(target_var, {}),
            random_sampling_method=sampling_method,
            hyperopt_metrics=hyperopt_metrics.get(model_name, {}).get(target_var, {}),
            hyperopt_params=hyperopt_params.get(model_name, {}).get(target_var, {}),
            skopt_metrics=skopt_metrics.get(model_name, {}).get(target_var, {}),
            skopt_params=skopt_params.get(model_name, {}).get(target_var, {}),
            param_names=list(param_spaces.get(model_name, {}).keys()),
            metric_name=metric.upper()
        )

        all_results.append({
            "Model": model_name,
            "Target Variable": target_var,
            "Optimal Parameters": optimal_params,
            "Optimal Metric": optimal_metrics
        })

# %%
# -------------------------------------------------------------------------------------------------
# 11. A summary
# -------------------------------------------------------------------------------------------------
def collect_results_as_dataframe(
    models_dict,
    target_columns,
    default_metrics, default_params,
    random_metrics, random_params, random_sampling_method,
    hyperopt_metrics, hyperopt_params,
    skopt_metrics, skopt_params,
    metric_name
):
    # Define the optimisation methods
    methods = [
        "Default",
        f"Random Search ({random_sampling_method})",
        "Hyperopt (TPE)",
        "Scikit-Optimize (GP_minimize)"
    ]

    metrics_list = [default_metrics, random_metrics, hyperopt_metrics, skopt_metrics]
    params_list = [default_params, random_params, hyperopt_params, skopt_params]

    results_list = []

    # Iterate through models and target variables
    for model_name in models_dict.keys():
        for target_var in target_columns:
            for method, metrics, params in zip(methods, metrics_list, params_list):
                # Fetch metric value
                if method == "Default":
                    metric_value = metrics.get(model_name, {}).get(target_var, {}).get(metric_name, None)
                else:
                    metric_value = metrics.get(model_name, {}).get(target_var, {}).get(metric_name, None)

                # Fetch parameter values
                param_values = params.get(model_name, {}).get(target_var, None)

                # Append results to the list
                results_list.append({
                    "Model": model_name,
                    "Target Variable": target_var,
                    "Hyperparameter Optimisation Method": method,
                    "Value of Hyperparameters": str(param_values),
                    metric_name: metric_value,
                })

    return pd.DataFrame(results_list)

# Collect results into a DataFrame
collected_results_df = collect_results_as_dataframe(
    models_dict=models_dict,
    target_columns=target_columns,
    default_metrics=default_metrics,
    default_params=default_params,
    random_metrics=random_metrics,
    random_params=random_params,
    random_sampling_method=sampling_method,
    hyperopt_metrics=hyperopt_metrics,
    hyperopt_params=hyperopt_params,
    skopt_metrics=skopt_metrics,
    skopt_params=skopt_params,
    metric_name=metric,
)

# Print the DataFrame to verify results
print("\nHyperparameter Optimisation Results:\n")
print(collected_results_df)

#%%
# -------------------------------------------------------------------------------------------------
# 11. Exporting DataFrame as a .csv file
# -------------------------------------------------------------------------------------------------
# Define the directory path
directory_path = # Paste the path to your hyperparameter optimisation results here

# Create the directory if it doesn't exist
os.makedirs(directory_path, exist_ok=True)

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d %H%M%S")

# Define the file path with the timestamp in the filename
file_path = os.path.join(directory_path, f"Hyperparameter Optimisation Results {current_time}.csv")

# Export the DataFrame to CSV
collected_results_df.to_csv(file_path, index=False, na_rep="N/A", float_format="%.10f")

# Print the success message with the file path
print(f"DataFrame exported to {file_path}")

# %%
# -------------------------------------------------------------------------------------------------
# 12. Best Model and Hyperparameters and Cross Validation
# -------------------------------------------------------------------------------------------------

# Metrics dictionary
metrics_dict = {
    "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
    "Explained Variance": lambda y_true, y_pred: explained_variance_score(y_true, y_pred),
    "R^2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    "ADJUSTED R^2": lambda y_true, y_pred, n, p: 1 - (1 - r2_score(y_true, y_pred)) * (n - 1) / (n - p - 1),
    "Q^2": lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
}

# CV methods
cv_methods = {
    "K-Fold": KFold,
    "Repeated K-Fold": RepeatedKFold,
    "Group K-Fold": GroupKFold,
    "LOO": LeaveOneOut,
    "LpO": LeavePOut,
    "Shuffle Split": ShuffleSplit,
}

def process_hyperparameters(hyperparams, model_name):
    processed_hyperparams = hyperparams.copy()
    param_space = param_spaces.get(model_name, {})

    for key, value in hyperparams.items():
        if value is None:
            processed_hyperparams[key] = None
        elif isinstance(value, (float, int)) and key in [
            "n_estimators", "max_depth", "num_iterations", "num_leaves",
            "n_neighbors", "min_samples_split", "min_samples_leaf",
        ]:
            processed_hyperparams[key] = int(value)  # Convert floats to integers
        elif key in param_space:  # Handle other defined parameter types
            param_type = param_space[key]["type"]
            if param_type == "int":
                processed_hyperparams[key] = int(value)
            elif param_type in ["uniform", "loguniform"]:
                processed_hyperparams[key] = float(value)
        elif isinstance(value, str):  # Allow strings like 'scale' to pass through
            processed_hyperparams[key] = value

    return processed_hyperparams

def find_best_model_and_hyperparams(collected_results_df, metric):
    if metric not in metrics_dict:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {list(metrics_dict.keys())}")

    is_minimized = metric in ["MSE", "MAE"]
    best_models = {}

    for target_var in collected_results_df['Target Variable'].unique():
        target_results = collected_results_df[collected_results_df['Target Variable'] == target_var]
        best_row = (
            target_results.loc[target_results[metric].idxmin()] if is_minimized
            else target_results.loc[target_results[metric].idxmax()]
        )

        print(f"\nBest Results for Target Variable: {target_var}")
        print(f"Model: {best_row['Model']}")
        print(f"Hyperparameter Optimisation Method: {best_row['Hyperparameter Optimisation Method']}")
        print(f"Hyperparameters: {best_row['Value of Hyperparameters']}")
        print(f"{metric}: {best_row[metric]:.4f}")

        best_models[target_var] = best_row

    return best_models


def perform_cross_validation_with_summary(
    model, X_train, y_train, target_var, cv_method, cv_args, scoring_metric):
    # Initialise the chosen CV method
    cv = cv_methods[cv_method](**cv_args) if cv_args else cv_methods[cv_method]()

    # Map scoring metric for compatibility with cross_val_score
    scoring_mapping = {
        "MSE": "neg_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
        "Explained Variance": "explained_variance",
        "R^2": "r2",
    }

    if scoring_metric in scoring_mapping:
        scoring = scoring_mapping[scoring_metric]
    elif scoring_metric in ["Q^2", "ADJUSTED R^2"]:
        def custom_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            n, p = X.shape
            if scoring_metric == "Q^2":
                return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            elif scoring_metric == "ADJUSTED R^2":
                r2 = r2_score(y, y_pred)
                return 1 - (1 - r2) * (n - 1) / (n - p - 1)

        scoring = custom_scorer
    else:
        raise ValueError(
            f"Invalid scoring metric '{scoring_metric}'. Choose from: {list(scoring_mapping.keys()) + ['Q^2', 'Adjusted R^2']}"
        )

    scores = cross_val_score(
        model,
        X_train,
        y_train[target_var],
        cv=cv,
        scoring=scoring,
    )

    # Adjust output for negative scores
    if scoring_metric in ["MSE", "MAE"]:
        scores = -scores  # Negate to get positive values

    return {
        "method": cv_method,
        "params": cv_args,
        "mean_score": scores.mean(),
        "std_dev": scores.std(),
    }


def process_all_targets_with_summary(
    best_models, X_train, y_train, cv_method, cv_args, scoring_metric
):
    cv_summary = {}
    for target_var, best_row in best_models.items():
        try:
            model_name = best_row['Model']
            raw_hyperparams = eval(best_row['Value of Hyperparameters'])

            # Process hyperparameters
            hyperparams = process_hyperparameters(raw_hyperparams, model_name)

            # Reset and configure the model
            best_model = reset_model_to_defaults(model_name)
            best_model.set_params(**hyperparams)

            print(f"\nProcessing Target Variable: {target_var}")
            print(f"Model: {model_name}")
            print(f"Hyperparameters: {hyperparams}")

            # Perform cross-validation
            cv_results = perform_cross_validation_with_summary(
                best_model, X_train, y_train, target_var, cv_method, cv_args, scoring_metric
            )
            print(f"Cross-Validation Method: {cv_results['method']}")
            print(f"CV Parameters: {cv_results['params']}")
            print(f"\nMean of Cross-Validation Scores ({scoring_metric}): {cv_results['mean_score']:.4f}")
            print(f"Standard Deviation of Cross-Validation Scores ({scoring_metric}): {cv_results['std_dev']:.4f}")
            cv_summary[target_var] = cv_results
        except Exception as e:
            print(f"An error occurred during cross-validation for target variable '{target_var}': {e}")

    return cv_summary


def create_cv_summary_df(cv_summary):
    summary_data = []
    for target_var, results in cv_summary.items():
        summary_data.append({
            "Target Variable": target_var,
            "CV Method": results["method"],
            "CV Parameters": results["params"],
            "Mean Score": results["mean_score"],
            "Std Deviation": results["std_dev"],
        })
    return pd.DataFrame(summary_data)
#%%
# -------------------------------------------------------------------------------------------------
# 12. Run best results and CV
# -------------------------------------------------------------------------------------------------

# Find the best models and hyperparameters for each target variable
best_models_per_target = find_best_model_and_hyperparams(collected_results_df, metric=metric)

# User-selected CV method and arguments
cv_configurations = {
    "K-Fold": {"n_splits": 5, "shuffle": True, "random_state": 0},
    "Repeated K-Fold": {"n_splits": 5, "n_repeats": 100, "random_state": 0},
    "LOO": {},  # Leave-One-Out does not require additional arguments
    "LpO": {"p": 2},  # Leave-P-Out with p=2
    "Shuffle Split": {"n_splits": 100, "test_size": 0.2, "random_state": 0},
}

# Fill out method and args here
cv_method = "Repeated K-Fold"
cv_args = cv_configurations[cv_method]

scoring_metric = "R^2"  # User-defined scoring metric

print("\nPerforming Cross Validation...")
cv_summary = process_all_targets_with_summary(
    best_models_per_target,
    X_train_scaled,
    y_train,
    cv_method,
    cv_args,
    scoring_metric,
)

# Create Cross-Validation Summary DataFrame
cv_summary_df = create_cv_summary_df(cv_summary)
# %%
# -------------------------------------------------------------------------------------------------
# 13. Cook's Distance, Residuals and Q-Q Plot
# -------------------------------------------------------------------------------------------------

def calculate_cooks_distance(X, residuals):
    X_const = sm.add_constant(X)  # Add intercept for OLS regression
    model = sm.OLS(residuals, X_const).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    return cooks_d

def plot_cooks_distance_all_targets(best_models, X_train, X_test, y_train, y_test):
    num_targets = len(best_models)
    fig, axes = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))

    if num_targets == 1:
        axes = [axes]

    for idx, (target_var, best_row) in enumerate(best_models.items()):
        model_name = best_row['Model']
        hyperparams = eval(best_row['Value of Hyperparameters'])

        # Reset and configure the model
        best_model = reset_model_to_defaults(model_name)
        best_model.set_params(**process_hyperparameters(hyperparams, model_name))

        # Fit the model
        best_model.fit(X_train, y_train[target_var])

        # Predict and calculate residuals
        y_pred = best_model.predict(X_test)
        residuals = np.array(y_test[target_var] - y_pred)

        # Calculate Cook's Distance
        cooks_d = calculate_cooks_distance(X_test, residuals)
        threshold = 4 / len(cooks_d)

        # Plot Cook's Distance
        axes[idx].stem(range(len(cooks_d)), cooks_d, basefmt=" ", markerfmt=".", linefmt="b")
        axes[idx].axhline(y=threshold, color="red", linestyle="--", label=f"Threshold (4/n = {threshold:.4f})")
        axes[idx].set_yscale("log")
        axes[idx].set_title(f"Cook's Distance for {model_name} ({target_var})")
        axes[idx].set_xlabel("Testing Sample Index #")
        axes[idx].set_ylabel("Cook's Distance (log scale)")
        axes[idx].grid(True, which="both", linestyle="--", linewidth=0.5)
        axes[idx].legend()

        influential_points = np.where(cooks_d > threshold)[0]
        proportion_influential = len(influential_points) / len(cooks_d)
        print(f"Target Variable: {target_var}")
        print(f"Number of Influential Points: {len(influential_points)}")
        print(f"Proportion of Influential Points: {proportion_influential:.2%}")

    plt.tight_layout()
    plt.show()


def plot_residuals_with_influential_points_all_targets(best_models, X_train, X_test, y_train, y_test):
    num_targets = len(best_models)
    fig, axes = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))

    if num_targets == 1:
        axes = [axes]

    for idx, (target_var, best_row) in enumerate(best_models.items()):
        model_name = best_row['Model']
        raw_hyperparams = eval(best_row['Value of Hyperparameters'])

        # Process hyperparameters to ensure correct types
        hyperparams = process_hyperparameters(raw_hyperparams, model_name)

        # Reset and configure the model
        best_model = reset_model_to_defaults(model_name)
        best_model.set_params(**hyperparams)

        try:
            best_model.fit(X_train, y_train[target_var])

            # Predict and calculate residuals
            y_pred = np.array(best_model.predict(X_test))
            residuals = np.array(y_test[target_var] - y_pred)

            # Calculate Cook's Distance
            cooks_d = calculate_cooks_distance(X_test, residuals)
            threshold = 4 / X_test.shape[0]

            # Identify influential points (positional indices)
            influential_points = np.where(cooks_d > threshold)[0]

            # Scatterplot
            axes[idx].scatter(y_pred, residuals, alpha=0.6, edgecolor="k", label="Non-Influential Points")
            if len(influential_points) > 0:
                axes[idx].scatter(
                    y_pred[influential_points],
                    residuals[influential_points],
                    color="red",
                    label="Influential Points",
                    edgecolor="k"
                )
            axes[idx].axhline(0, color="green", linestyle="--", linewidth=1, label="Zero Residual Line")
            axes[idx].set_title(f"Residual Plot for {model_name} ({target_var})")
            axes[idx].set_xlabel("Predicted Values")
            axes[idx].set_ylabel("Residuals")
            axes[idx].legend()
            axes[idx].grid(True)

        except Exception as e:
            print(f"Error processing target variable '{target_var}': {e}")
            axes[idx].set_title(f"Failed to process {target_var}")
            axes[idx].text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", transform=axes[idx].transAxes)

    plt.tight_layout()
    plt.show()

def apply_transformation(residuals, transformation):
    if transformation == "Log":
        return np.log(np.abs(residuals) + 1e-6) * np.sign(residuals) # Epsilon is 1e-6 here
    elif transformation == "Sqrt":
        return np.sqrt(np.abs(residuals)) * np.sign(residuals)
    elif transformation == "Box-Cox":
        positive_residuals = residuals - residuals.min() + 1e-6
        return boxcox(positive_residuals)[0]
    elif transformation == "Yeo-Johnson":
        transformer = PowerTransformer(method="yeo-johnson")
        residuals_reshaped = residuals.values.reshape(-1, 1)
        return transformer.fit_transform(residuals_reshaped).flatten()
    else:
        return residuals


def evaluate_transformations(best_models, X_train, X_test, y_train, y_test):
    transformations = ["None", "Log", "Sqrt", "Box-Cox", "Yeo-Johnson"]
    results = []

    for target_var, best_row in best_models.items():
        model_name = best_row['Model']
        raw_hyperparams = eval(best_row['Value of Hyperparameters'])

        # Process hyperparameters
        hyperparams = process_hyperparameters(raw_hyperparams, model_name)

        # Configure the model
        best_model = reset_model_to_defaults(model_name)
        best_model.set_params(**hyperparams)

        try:
            best_model.fit(X_train, y_train[target_var])
            y_pred = best_model.predict(X_test)

            residuals = y_test[target_var] - y_pred

            for transformation in transformations:
                transformed_residuals = apply_transformation(residuals, transformation)

                skewness_value = skew(transformed_residuals)
                kurtosis_value = kurtosis(transformed_residuals, fisher=True)
                ad_stat, _, _ = anderson(transformed_residuals)

                results.append({
                    "Target Variable": target_var,
                    "Model": model_name,
                    "Transformation": transformation,
                    "Skewness": skewness_value,
                    "Excess Kurtosis": kurtosis_value,
                    "AD Statistic": ad_stat
                })

        except Exception as e:
            print(f"Error processing target variable '{target_var}': {e}")
            results.append({
                "Target Variable": target_var,
                "Model": model_name,
                "Transformation": "Error",
                "Skewness": None,
                "Excess Kurtosis": None,
                "AD Statistic": None,
                "Error": str(e)
            })

    return pd.DataFrame(results)

def plot_all_transformations(results_df, best_models, X_train, X_test, y_train, y_test):
    # Filter valid rows
    results_df = results_df.dropna(subset=["AD Statistic"])
    if results_df.empty:
        print("No valid transformations available for plotting.")
        return

    # Use dictionary-style access for DataFrame rows
    best_rows = results_df.loc[results_df.groupby("Target Variable")["AD Statistic"].idxmin()]

    num_targets = len(best_rows)

    # Create subplots for each type
    fig_residuals, ax_residuals = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    fig_histograms, ax_histograms = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))
    fig_qq, ax_qq = plt.subplots(num_targets, 1, figsize=(10, 5 * num_targets))

    # Ensure axes are lists, even for a single subplot
    if num_targets == 1:
        ax_residuals = [ax_residuals]
        ax_histograms = [ax_histograms]
        ax_qq = [ax_qq]

    for idx, row in enumerate(best_rows.iterrows()):
        row = row[1]  # Access the Series for the current row
        target_var = row['Target Variable']
        transformation = row['Transformation']
        model_name = row['Model']

        # Ensure integer hyperparameters are cast to int
        raw_hyperparams = eval(best_models[target_var]['Value of Hyperparameters'])
        hyperparams = process_hyperparameters(raw_hyperparams, model_name)
        for key, value in hyperparams.items():
            if isinstance(value, float) and key in ["n_estimators", "max_depth", "num_iterations", "num_leaves"]:
                hyperparams[key] = int(value)

        best_model = reset_model_to_defaults(model_name)
        best_model.set_params(**hyperparams)
        best_model.fit(X_train, y_train[target_var])
        y_pred = best_model.predict(X_test)

        residuals = y_test[target_var] - y_pred
        transformed_residuals = apply_transformation(residuals, transformation)

        # Residuals plot
        ax_residuals[idx].scatter(y_pred, transformed_residuals, alpha=0.6, edgecolor="k", label="Data Points")
        ax_residuals[idx].axhline(y=0, color="green", linestyle="--", linewidth=1, label="Zero Residual Line")
        ax_residuals[idx].set_title(f"Residuals: {model_name} ({target_var})\n({transformation} Transformation)")
        ax_residuals[idx].set_xlabel("Predicted Values")
        ax_residuals[idx].set_ylabel("Residuals")
        ax_residuals[idx].legend()
        ax_residuals[idx].grid(True)

        # Histogram
        count, bins, ignored = ax_histograms[idx].hist(transformed_residuals, bins=30, density=True, color="blue", edgecolor="k", alpha=0.7)
        ax_histograms[idx].axvline(x=0, color="green", linestyle="--", linewidth=2)  # Thicker green line
        mu, sigma = np.mean(transformed_residuals), np.std(transformed_residuals)
        normal_dist = norm.pdf(bins, mu, sigma)
        ax_histograms[idx].plot(bins, normal_dist, color="red", lw=2, label="Normal Distribution")
        ax_histograms[idx].set_title(f"Histogram of Residuals: {model_name} ({target_var})\n({transformation} Transformation)")
        ax_histograms[idx].set_xlabel("Residuals")
        ax_histograms[idx].set_ylabel("Density")
        ax_histograms[idx].legend()
        ax_histograms[idx].grid(True)

        # Q-Q Plot
        res = probplot(transformed_residuals, dist="norm")
        x, y = res[0][0], res[0][1]
        ax_qq[idx].scatter(x, y, alpha=0.6, edgecolor="k", label="Data Points")
        fitted_line = res[1][0] * x + res[1][1]
        ax_qq[idx].plot(x, fitted_line, color="red", label="Fitted Line")
        ax_qq[idx].set_title(f"Q-Q Plot: {model_name} ({target_var})\n({transformation} Transformation)")
        ax_qq[idx].set_xlabel("Theoretical Quantiles")
        ax_qq[idx].set_ylabel("Actual Quantiles")
        ax_qq[idx].legend()
        ax_qq[idx].grid(True)

    # Adjust and display
    fig_residuals.tight_layout()
    fig_histograms.tight_layout()
    fig_qq.tight_layout()

    plt.show()

    # Print best transformation summary
    print("Best Transformations:\n")
    for _, row in best_rows.iterrows():  # Iterate with dictionary-style access
        skewness_type = "Symmetric" if -0.5 <= row['Skewness'] <= 0.5 else ("Right-Skewed" if row['Skewness'] > 0.5 else "Left-Skewed")
        kurtosis_type = "Mesokurtic (Normal-Like)" if -1 <= row['Excess Kurtosis'] <= 1 else ("Leptokurtic (Heavy-Tailed)" if row['Excess Kurtosis'] > 1 else "Platykurtic (Light-Tailed)\n")

        print(f"Target Variable: {row['Target Variable']}")
        print(f"Transformation: {row['Transformation']}")
        print(f"Skewness: {row['Skewness']:.4f} ({skewness_type})")
        print(f"Excess Kurtosis: {row['Excess Kurtosis']:.4f} ({kurtosis_type})")
        print(f"Anderson-Darling Statistic: {row['AD Statistic']:.4f}")

# %%
# -------------------------------------------------------------------------------------------------
# 13. Plot Cook's Distance, Residuals and Q-Q Plot
# -------------------------------------------------------------------------------------------------

print("\nProcessing Cook's Distance and Residuals Analysis...")
plot_cooks_distance_all_targets(
    best_models_per_target,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

plot_residuals_with_influential_points_all_targets(
    best_models_per_target,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

# Continue to Evaluate Transformations and Plot
transformation_results_df = evaluate_transformations(
    best_models_per_target,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

print("Deciding best transformation of residuals...")
print(transformation_results_df)

plot_all_transformations(
    transformation_results_df,
    best_models_per_target,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

#%%
# -------------------------------------------------------------------------------------------------
# 14. Exporting workflow summary using pickle
# -------------------------------------------------------------------------------------------------

def save_workflow_results(workflow_results, output_dir="workflow_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pickle_file = os.path.join(output_dir, f"workflow_results_{timestamp}.pkl")

    with open(pickle_file, "wb") as file:
        pickle.dump(workflow_results, file)

    print(f"Workflow results saved to {pickle_file}")

    # Save each model
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for target_var, best_row in workflow_results["best_results"].items():
        model_name = best_row["Model"]
        hyperparams = eval(best_row["Value of Hyperparameters"])
        best_model = reset_model_to_defaults(model_name)
        best_model.set_params(**process_hyperparameters(hyperparams, model_name))
        best_model.fit(X_train_scaled, y_train[target_var])

        model_file = os.path.join(models_dir, f"{target_var}_{model_name}_{timestamp}.pkl")
        with open(model_file, "wb") as file:
            pickle.dump(best_model, file)
        print(f"Model for target '{target_var}' saved to {model_file}")

    return pickle_file

def load_workflow_results(pickle_file):
    if not os.path.exists(pickle_file):
        print(f"File not found: {pickle_file}")
        return None

    with open(pickle_file, "rb") as file:
        workflow_results = pickle.load(file)

    print(f"Workflow results loaded from {pickle_file}")
    return workflow_results

def display_workflow_results(workflow_results):
    print("\n===== Model Performance Summary =====\n")
    if "model_performance_summary" in workflow_results:
        print(tabulate(workflow_results["model_performance_summary"], headers="keys", tablefmt="grid"))
    else:
        print("No model performance summary available.")

    print("\n===== Hyperparameter Optimisation Results =====\n")
    if "hyperparameter_optimisation_results" in workflow_results:
        print(tabulate(workflow_results["hyperparameter_optimisation_results"], headers="keys", tablefmt="grid"))
    else:
        print("No hyperparameter optimisation results available.")

    print("\n===== Best Results =====\n")
    if "best_results" in workflow_results:
        best_results_df = pd.DataFrame.from_dict(workflow_results["best_results"], orient="index")
        print(tabulate(best_results_df, headers="keys", tablefmt="grid"))
    else:
        print("No best results available.")

    print("\n===== Cross-Validation Summary =====\n")
    if "cv_summary_df" in workflow_results:
        print(tabulate(workflow_results["cv_summary_df"], headers="keys", tablefmt="grid"))
    else:
        print("No cross-validation summary available.")

    print("\n===== Model Transformation Results =====\n")
    if "model_transformation_results" in workflow_results:
        print(tabulate(workflow_results["model_transformation_results"], headers="keys", tablefmt="grid"))
    else:
        print("No model transformation results available.")

# Collect results into a single dictionary
workflow_results = {
    "model_performance_summary": model_performance_summary,  # Replace with the DataFrame holding the model performance summary
    "hyperparameter_optimisation_results": collected_results_df,  # Final summary DataFrame
    "best_results": best_models_per_target,
    "cv_summary_df": cv_summary_df,  # Adjust based on your CV results
    "model_transformation_results": transformation_results_df,  # Replace with transformation-specific DataFrame
}

# Save workflow results
saved_pickle_file = save_workflow_results(workflow_results)
#%%
# -------------------------------------------------------------------------------------------------
# 14. Load the summary (Optional)
# -------------------------------------------------------------------------------------------------

# Load workflow results
loaded_results = load_workflow_results(saved_pickle_file)

# Display the results
if loaded_results:
    display_workflow_results(loaded_results)
# %%
# -------------------------------------------------------------------------------------------------
# 15. Finally, display the time elapsed for the whole program
# -------------------------------------------------------------------------------------------------

# Measure and display program runtime
end_time_program = time.time()
elapsed_time_program = end_time_program - start_time_program
hours, rem = divmod(elapsed_time_program, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total time elapsed for this program: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}")


# %%
# Run the code here


#%%
# Code for opening the .pkl file

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# Right click on the file and press "Copy as path". Then, paste as a raw string
pkl_file_path = # Paste the file path here
print(load_pickle(pkl_file_path))
