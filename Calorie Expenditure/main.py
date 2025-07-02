from calorie_prediction.components.data_ingestion import CSVDataLoader
from calorie_prediction.components.data_transformation import one_hot_encode_dataframe
from calorie_prediction.components.feature_selection import FeatureSelectorRF
from calorie_prediction.components.bayesian_hyperparameter_tuning import BayesianRFTuner
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_PATH = '/Users/tawate/Documents/Competition Code/Data/Calorie_Predictor/'
TRAINING_DATA = 'train.csv'
TESTING_DATA = 'test.csv'

# Read in Training Data
dataloader = CSVDataLoader()
train_df = dataloader.read_csv_to_dataframe(DATA_PATH, TRAINING_DATA)

# Split Training Data into Training and Validation
# OneHot Encoding Categorical Vars
train_df_prep = one_hot_encode_dataframe(train_df, columns_to_encode='Sex', drop_first=False)

# Split Training Data in Train/Validation
X = train_df_prep.drop(columns=['Calories','id'])
y = train_df_prep['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Random Forest Var Importance
rf_reg_selector = FeatureSelectorRF(model_type='regressor',n_estimators=100, random_state=42)
rf_reg_selector.fit(X_train, y_train)
imp_vars = rf_reg_selector.get_important_variables(threshold=.01)

# Reduce Inputs to Importance Variables
X_train_imp = X_train[imp_vars]
X_test_imp = X_test[imp_vars]

# Hyper Parameter Tuning
bayes_tuner_rf = BayesianRFTuner(
    X=X_train_imp,
    y=y_train,
    model_type='regressor',
    cv_folds=3,
    scoring='accuracy',
    max_evals=100,
    random_state=42
    )

bayes_tuner_rf.tune()
best_params = bayes_tuner_rf.get_best_params()
best_score = bayes_tuner_rf.get_best_score()
optimized_model = bayes_tuner_rf.get_optimized_model()
print(f"\nBest Classification Params: {best_params}")
print(f"Best Classification Score: {best_score:.4f}")
print(f"Optimized Classifier: {optimized_model}")