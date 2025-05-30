import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *

# Basic Feature Importance Chart
def plot_basic_feature_importance(model, feature_names):
    """Simple horizontal bar chart of feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[indices])
    plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
# Basic Model Training Performance
def model_performance_simple(y_val, y_pred):
    MSE = mean_squared_error(y_val, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_val, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {MSE:.4f}")
    print(f"Root Mean Squared Error (RMSE): {RMSE:.4f}")
    print(f"R-squared (R2): {R2:.4f}")

# Print Visualization of predicted vs actuals for regression based ML. Overlays fitted x line.
def regressor_pred_actual_viz(y_val, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # y=x line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.grid(True)
    plt.show()