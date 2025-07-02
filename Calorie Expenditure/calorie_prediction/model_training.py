import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from sklearn.model_selection import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import *
from sklearn.metrics import *
from eda_functions import *
from ml_diagnostics import *
import statsmodels.api as sm
from gradient_boost_tuner import *
from random_forest_tuner import *
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
warnings.filterwarnings('ignore')
DATA_PATH = '/Users/tawate/Documents/Competition Code/Data/Calorie_Predictor/'
DSIN_TRAIN = 'train.csv'
DSIN_TEST = 'test.csv'

# Import Data
train_df = pd.read_csv(DATA_PATH + DSIN_TRAIN)

""" Data Prep """
# OneHot Encoding Categorical Vars
train_df_prep = pd.get_dummies(train_df, columns=['Sex']).drop(columns=['Sex_female'])
train_df_prep['Sex_male'] = train_df_prep['Sex_male'].astype(int)

# Split Training Data in Train/Validation
X = train_df_prep.drop(columns=['Calories','id'])
y = train_df_prep['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42
)

# Create list of feature names
feature_names = X.columns.tolist()

""" Gradient Boosting Tuner """
# Initialize tuner
gbtuner = GradientBoostingTuner(
    task_type='regression',
    scoring='r2',
    cv_folds=5,
    random_state=42
)
# Print search space
print("\nSearch Space:")
gbtuner.print_current_search_space()
# Perform Bayesian optimization
print(f"\nStarting hyperparameter optimization...")
best_params = gbtuner.bayesian_search(X_train, y_train, n_calls=10, verbose=True)

# Fit best model and evaluate
best_gb_model = gbtuner.fit_best_model(X_train, y_train)

# Make predictions
y_pred = best_gb_model.predict(X_test)
model_performance_simple(y_val=y_test, y_pred=y_pred)


""" Random Forest Tuner """
# Create search space
focused_space = [
    Integer(100, 300, name='n_estimators'),
    Integer(5, 15, name='max_depth'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf'),
    Categorical(['sqrt', 'log2'], name='max_features'),
    Categorical([True], name='bootstrap')
]

# Initialize tuner
rf_tuner = RandomForestTuner(
    task_type='regression',
    scoring='r2',
    bayesian_space=focused_space,
    random_state=42
)

# Perform Bayesian optimization
print(f"\nStarting hyperparameter optimization...")
best_params = rf_tuner.bayesian_search(X_train, y_train, n_calls=10, verbose=True)

# Fit best model
best_rf_model = rf_tuner.fit_best_model(X_train, y_train)

# Feature importance
plot_basic_feature_importance(model = best_rf_model, feature_names=feature_names)

# Make predictions
y_pred = best_rf_model.predict(X_test)
model_performance_simple(y_val=y_test, y_pred=y_pred)

