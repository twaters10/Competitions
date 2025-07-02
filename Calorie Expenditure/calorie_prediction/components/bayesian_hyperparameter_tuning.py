import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from typing import Dict, Any, Union, Optional, List

class BayesianRFTuner:
    """
    Performs Bayesian Hyperparameter Tuning for RandomForestClassifier or RandomForestRegressor
    using the Hyperopt library.
    """

    def __init__(self,
                 X: pd.DataFrame,
                 y: Union[pd.Series, np.ndarray],
                 model_type: str = 'classifier',
                 cv_folds: int = 5,
                 scoring: Optional[Union[str, callable]] = None,
                 max_evals: int = 50,
                 random_state: Optional[int] = None):
        """
        Initializes the BayesianRFTuner.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (Union[pd.Series, np.ndarray]): The target variable.
            model_type (str): Type of Random Forest model ('classifier' or 'regressor').
                              Defaults to 'classifier'.
            cv_folds (int): Number of cross-validation folds. Defaults to 5.
            scoring (Optional[Union[str, callable]]): Scoring metric for cross-validation.
                                                        If None: 'accuracy' for classifier, 'neg_mean_squared_error' for regressor.
                                                        Can be a scikit-learn scoring string or a callable.
            max_evals (int): Maximum number of iterations for Bayesian optimization. Defaults to 50.
            random_state (Optional[int]): Random seed for reproducibility. Defaults to None.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array.")
        if model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'.")

        self.X = X
        self.y = y
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.max_evals = max_evals
        self.random_state = random_state

        # Set default scoring if not provided
        if scoring is None:
            self.scoring = 'accuracy' if self.model_type == 'classifier' else 'neg_mean_squared_error'
        else:
            self.scoring = scoring

        self.trials = Trials()
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _define_search_space(self) -> Dict[str, Any]:
        """
        Defines the hyperparameter search space for Random Forest using hyperopt.
        """
        if self.model_type == 'classifier':
            # Example search space for RandomForestClassifier
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 300, 50), # 50, 100, 150, ..., 300
                'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.5, 0.7, 1.0]),
                'max_depth': hp.quniform('max_depth', 5, 20, 1), # 5, 6, ..., 20
                'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1), # float between 0.01 and 0.1
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1), # 1, 2, ..., 10
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'random_state': self.random_state # Pass fixed random state for internal RF model
            }
        else: # regressor
            # Example search space for RandomForestRegressor
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 300, 50),
                'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.5, 0.7, 1.0]),
                'max_depth': hp.quniform('max_depth', 5, 20, 1),
                'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.1),
                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'criterion': hp.choice('criterion', ['squared_error', 'absolute_error']), # 'mse' in older sklearn, 'friedman_mse' is also option
                'random_state': self.random_state # Pass fixed random state for internal RF model
            }
        return space

    def _objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for Hyperopt.
        Trains a Random Forest model with given params and returns cross-validation score.
        Hyperopt minimizes the objective, so we return negative score for metrics to maximize.
        """
        # Convert n_estimators, max_depth, min_samples_split, min_samples_leaf to int if they are float
        # This happens with hp.quniform
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth']) if params['max_depth'] is not None else None
        params['min_samples_split'] = int(params['min_samples_split']) if params['min_samples_split'] < 1 else params['min_samples_split']
        params['min_samples_leaf'] = int(params['min_samples_leaf']) if params['min_samples_leaf'] < 1 else params['min_samples_leaf']


        if self.model_type == 'classifier':
            model = RandomForestClassifier(**params)
            cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else: # regressor
            model = RandomForestRegressor(**params)
            cv_strategy = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        try:
            # Perform cross-validation
            scores = cross_val_score(model, self.X, self.y, cv=cv_strategy, scoring=self.scoring, n_jobs=-1)
            mean_score = np.mean(scores)

            # Hyperopt minimizes, so negate score if it's a metric to maximize
            loss = -mean_score if self.scoring in ['accuracy', 'f1', 'roc_auc', 'r2', 'precision', 'recall'] else mean_score

            return {'loss': loss, 'status': STATUS_OK}
        except Exception as e:
            # Handle potential errors during training/scoring (e.g., invalid param combinations)
            print(f"Error during evaluation: {e}. Params: {params}")
            return {'loss': np.inf, 'status': STATUS_OK} # Assign high loss for errors


    def tune(self):
        """
        Runs the Bayesian optimization process to find the best hyperparameters.
        """
        print(f"Starting Bayesian Hyperparameter Tuning for RandomForest ({self.model_type})...")
        print(f"Optimization will run for {self.max_evals} evaluations.")
        print(f"Using scoring metric: '{self.scoring}'")

        space = self._define_search_space()

        self.trials = Trials()
        best = fmin(
            fn=self._objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(self.random_state), # For hyperopt's internal randomness
            verbose=False
        )

        # Retrieve best parameters and corresponding score
        self.best_params = space_eval(space, best)

        # Convert n_estimators, max_depth, min_samples_split, min_samples_leaf to int if they are float
        self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
        if 'max_depth' in self.best_params:
            self.best_params['max_depth'] = int(self.best_params['max_depth']) if self.best_params['max_depth'] is not None else None
        if 'min_samples_split' in self.best_params:
            self.best_params['min_samples_split'] = int(self.best_params['min_samples_split']) if self.best_params['min_samples_split'] < 1 else self.best_params['min_samples_split']
        if 'min_samples_leaf' in self.best_params:
            self.best_params['min_samples_leaf'] = int(self.best_params['min_samples_leaf']) if self.best_params['min_samples_leaf'] < 1 else self.best_params['min_samples_leaf']

        # The best_score is the negative of the best loss found by hyperopt
        best_loss = self.trials.best_trial['result']['loss']
        self.best_score = -best_loss if self.scoring in ['accuracy', 'f1', 'roc_auc', 'r2', 'precision', 'recall'] else best_loss

        print("\nBayesian Tuning Complete.")
        print(f"Best Score ({self.scoring}): {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Returns the best hyperparameters found."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Returns the best score (e.g., accuracy, R2) found."""
        return self.best_score

    def get_trials(self) -> Trials:
        """Returns the hyperopt.Trials object for detailed inspection."""
        return self.trials

    def get_optimized_model(self) -> Union[RandomForestClassifier, RandomForestRegressor, None]:
        """
        Returns an instance of the Random Forest model configured with the best found hyperparameters.
        Returns None if tuning has not been performed.
        """
        if self.best_params:
            if self.model_type == 'classifier':
                return RandomForestClassifier(**self.best_params)
            else:
                return RandomForestRegressor(**self.best_params)
        return None