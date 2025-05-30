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
from eda_functions import *
from ml_diagnostics import *
from hyperparameter_tuning import *
warnings.filterwarnings('ignore')
DATA_PATH = '/Users/tawate/Documents/Competition Code/Data/Calorie_Predictor/'
DSIN_TRAIN = 'train.csv'
DSIN_TEST = 'test.csv'

# Import Data
train_df = pd.read_csv(DATA_PATH + DSIN_TRAIN)

# Describe Data
train_df.info()
train_df.describe()
train_df.head()

sns.histplot(data=train_df, x='Calories', kde=True)
sns.scatterplot(data=train_df, x='Age', y='Calories')
sns.lineplot(data=train_df, x='Age', y='Duration')

# EDA
univariate_analysis(df = train_df, var='Age')


# Prep Data for Modeling
# Check on log of target

# OneHot Encoding Categorical Vars
train_df_prep = pd.get_dummies(train_df, columns=['Sex']).drop(columns=['Sex_female'])

# Simple Random Model
# Split Training Data in Train/Validation
X = train_df_prep.drop(columns=['Calories','id'])
y = train_df_prep['Calories']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42
)

# Create Basic Random Forest Regressor
rf_model_basic = RandomForestRegressor(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth of trees
    min_samples_split=2,     # Minimum samples required to split
    min_samples_leaf=1,      # Minimum samples required at leaf node
    max_features='sqrt',     # Number of features to consider for best split
    random_state=42,
    n_jobs=-1,               # Use all available cores
    oob_score=True
)

# Hyper parameter tuning
# Define the search space
search_space = bayseian_search_space(n_est_min = 50, n_est_max = 300, 
                                feature_split_min = .5, feature_split_max = 1, 
                                depth_min = 5, depth_max = 20, 
                                sample_split_min = 2, sample_split_max = 10,
                                sample_leaf_min = 1, sample_leaf_max = 5)

rf_model = RandomForestRegressor(random_state=42)
print("\nStarting Bayesian hyperparameter search...")
bayes_opt = BayesSearchCV(
    estimator=rf_model
    ,search_spaces=search_space
    ,n_iter=50
    ,cv=5
    ,scoring='r2'
    ,random_state=42
    ,n_jobs=-1
    ,verbose=1
)
# Fit BayesSearchCV on the training data to find the best hyperparameters
bayes_opt.fit(X_train, y_train)

print("\nBayesian hyperparameter search complete.")
print(f"Best R-squared score found during search: {bayes_opt.best_score_:.4f}")
print(f"Best hyperparameters found: {bayes_opt.best_params_}")

# Model Training
rf_train_basic = rf_model_basic.fit(X_train, y_train)
rf_bayes_best = bayes_opt.best_estimator_ # Get the best estimator (model with optimal hyperparameters)

# Model training evaluation
# Variable Importance
feature_names = X.columns.tolist()
plot_basic_feature_importance(model = rf_model_basic, feature_names=feature_names)
plot_basic_feature_importance(model = rf_bayes_best, feature_names=feature_names)

# Model Performance
rf_pred_basic = rf_model.predict(X_test)
rf_pred_bayes = rf_bayes_best.predict(X_test)
model_performance_simple(y_val=y_test, y_pred=rf_pred_basic)
model_performance_simple(y_val=y_test, y_pred=rf_pred_bayes)

