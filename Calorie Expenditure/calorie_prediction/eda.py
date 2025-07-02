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
# Correlation Matrix (Numerical Features)
plt.figure(figsize=(8, 6))
correlation_matrix = train_df.drop(columns=['Sex']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Check for linearity and homoscedasticity visually
scatter_plot_indv(df = train_df, varx = 'Heart_Rate', vary = 'Calories', plot_title = 'Calories vs Heart Rate')
scatter_plot_indv(df = train_df, varx = 'Body_Temp', vary = 'Duration', plot_title = 'Body_Temp vs Duration')
scatter_plot_indv(df = train_df, varx = 'Heart_Rate', vary = 'Calories', plot_title = 'Calories vs Heart Rate')
scatter_plot_indv(df = train_df, varx = 'Heart_Rate', vary = 'Calories', plot_title = 'Calories vs Heart Rate')
scatter_plot_indv(df = train_df, varx = 'Heart_Rate', vary = 'Calories', plot_title = 'Calories vs Heart Rate')
scatter_plot_indv(df = train_df, varx = 'Heart_Rate', vary = 'Calories', plot_title = 'Calories vs Heart Rate')


# Prep Data for Modeling
# Check on log of target

# OneHot Encoding Categorical Vars
train_df_prep = pd.get_dummies(train_df, columns=['Sex']).drop(columns=['Sex_female'])
train_df_prep['Sex_male'] = train_df_prep['Sex_male'].astype(int)


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

# Create Gradient Boosting Regressor
gbr_model_basic = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=.1,
    max_depth=3,
    random_state=42
)

# Hyper parameter tuning
# Define the random forest search space
rf_search_space = rf_bayseian_search_space(
    n_est_min = 50, 
    n_est_max = 300, 
    feature_split_min = .5, 
    feature_split_max = 1, 
    depth_min = 1, 
    depth_max = 20, 
    sample_split_min = 2, 
    sample_split_max = 10,
    sample_leaf_min = 1, 
    sample_leaf_max = 5)

rf_model = RandomForestRegressor(random_state=42)

ransearch_opt = RandomizedSearchCV(
    estimator=rf_model
    ,param_distributions=rf_search_space
    ,n_iter=5
    ,cv=2
    ,scoring='r2'
    ,random_state=42
    ,n_jobs=-1
    ,verbose=2
)
# Fit RandomSearch on the training data to find the best hyperparameters
ransearch_opt.fit(X_train, y_train)

print("\nStarting Bayesian hyperparameter search...")
bayes_opt = BayesSearchCV(
    estimator=rf_model
    ,search_spaces=search_space
    ,n_iter=5
    ,cv=3
    ,scoring='r2'
    ,random_state=42
    ,n_jobs=-1
    ,verbose=2
)
# Fit BayesSearchCV on the training data to find the best hyperparameters
bayes_opt.fit(X_train, y_train)

print("\nBayesian hyperparameter search complete.")
print(f"Best R-squared score found during search: {bayes_opt.best_score_:.4f}")
print(f"Best hyperparameters found: {bayes_opt.best_params_}")

# Linear Regression
linreg_model = LinearRegression()
# Add a constant (intercept) to the independent variable(s) for statsmodels
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
# Create and fit the OLS (Ordinary Least Squares) model
model_sm = sm.OLS(y_train, X_train_const)
results_sm = model_sm.fit()

# Print the comprehensive summary
print("\nStatsmodels Regression Summary:")
print(results_sm.summary())

# Model Training
rf_train_basic = rf_model_basic.fit(X_train, y_train)   # Basic Random Forest
rf_bayes_best = bayes_opt.best_estimator_               # Get the best estimator (model with optimal hyperparameters)
linreg_train = linreg_model.fit(X_train, y_train)       # Basic Linear Regression


# Model training evaluation
# RF Variable Importance
feature_names = X.columns.tolist()
plot_basic_feature_importance(model = rf_model_basic, feature_names=feature_names)
plot_basic_feature_importance(model = rf_bayes_best, feature_names=feature_names)


# Model Performance
rf_pred_basic = rf_train_basic.predict(X_test)
rf_pred_bayes = rf_bayes_best.predict(X_test)
linreg_pred = linreg_train.predict(X_test)
model_performance_simple(y_val=y_test, y_pred=rf_pred_basic)
model_performance_simple(y_val=y_test, y_pred=rf_pred_bayes)
model_performance_simple(y_val=y_test, y_pred=linreg_pred)

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
best_model = gbtuner.fit_best_model(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"\nTest Results:")
print(f"R² Score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")

plot_basic_feature_importance(model = best_model, feature_names=feature_names)
