import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.datasets import make_classification, make_regression
from scipy.stats import uniform, randint
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization unavailable.")

class GradientBoostingTuner:
    """
    Hyperparameter tuning class for Gradient Boosting models using Bayesian optimization.
    """
    
    def __init__(self, task_type='classification', scoring=None, cv_folds=5, random_state=42, 
                 bayesian_space=None):
        """
        Initialize the tuner.
        
        Parameters:
        -----------
        task_type : str, default='classification'
            Type of task - 'classification' or 'regression'
        scoring : str or callable, default=None
            Scoring metric. If None, uses accuracy for classification, r2 for regression
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        bayesian_space : list, default=None
            Custom parameter space for Bayesian optimization. If None, uses defaults
        """
        self.task_type = task_type.lower()
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Set default scoring
        if scoring is None:
            self.scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        else:
            self.scoring = scoring
            
        # Initialize parameter spaces
        self.custom_bayesian_space = bayesian_space
        self._define_parameter_spaces()
        
        # Results storage
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_results_ = []
        
    def _define_parameter_spaces(self):
        """Define parameter search space for Bayesian optimization."""
        
        # Use custom Bayesian space if provided, otherwise use defaults
        if BAYESIAN_AVAILABLE:
            if self.custom_bayesian_space is not None:
                self.bayesian_space = self.custom_bayesian_space.copy()
            else:
                # Default Bayesian optimization space
                self.bayesian_space = [
                    Integer(50, 500, name='n_estimators'),
                    Real(0.01, 0.3, name='learning_rate'),
                    Integer(3, 15, name='max_depth'),
                    Integer(2, 20, name='min_samples_split'),
                    Integer(1, 10, name='min_samples_leaf'),
                    Real(0.6, 1.0, name='subsample')
                ]
    
    def update_search_space(self, bayesian_space=None):
        """
        Update the parameter search space after initialization.
        
        Parameters:
        -----------
        bayesian_space : list, optional
            New parameter space for Bayesian optimization
        """
        if bayesian_space is not None and BAYESIAN_AVAILABLE:
            self.bayesian_space = bayesian_space.copy()
            print("Updated Bayesian optimization parameter space")
        elif bayesian_space is not None and not BAYESIAN_AVAILABLE:
            print("Warning: Bayesian space provided but scikit-optimize not available")
    
    def get_default_bayesian_space(self):
        """
        Get the default parameter space for Bayesian optimization.
        
        Returns:
        --------
        list : Default Bayesian parameter space (None if scikit-optimize unavailable)
        """
        if not BAYESIAN_AVAILABLE:
            return None
            
        return [
            Integer(50, 500, name='n_estimators'),
            Real(0.01, 0.3, name='learning_rate'),
            Integer(3, 15, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf'),
            Real(0.6, 1.0, name='subsample')
        ]
    
    def print_current_search_space(self):
        """Print the current parameter search space."""
        if BAYESIAN_AVAILABLE and hasattr(self, 'bayesian_space'):
            print("Current Bayesian Optimization Parameter Space:")
            print("-" * 50)
            for dim in self.bayesian_space:
                if hasattr(dim, 'low') and hasattr(dim, 'high'):
                    print(f"{dim.name}: {type(dim).__name__}({dim.low}, {dim.high})")
                else:
                    print(f"{dim.name}: {dim}")
        else:
            print("Bayesian optimization space not available")
    
    def _get_model(self, **params):
        """Get gradient boosting model with specified parameters."""
        if self.task_type == 'classification':
            return GradientBoostingClassifier(random_state=self.random_state, **params)
        else:
            return GradientBoostingRegressor(random_state=self.random_state, **params)
    
    def _evaluate_model(self, params, X, y):
        """Evaluate model with given parameters using cross-validation."""
        try:
            model = self._get_model(**params)
            
            # Set up cross-validation
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                   random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, 
                          random_state=self.random_state)
            
            # Perform cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1)
            return np.mean(scores)
            
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf if self.scoring in ['accuracy', 'f1', 'r2'] else np.inf
    
    def bayesian_search(self, X, y, n_calls=50, verbose=True):
        """
        Perform Bayesian optimization for hyperparameter tuning.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training targets
        n_calls : int, default=50
            Number of function evaluations
        verbose : bool, default=True
            Whether to print progress
            
        Returns:
        --------
        dict : Best parameters found
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. "
                            "Install with: pip install scikit-optimize")
        
        if verbose:
            print(f"Starting Bayesian Optimization with {n_calls} calls...")
        
        # Define objective function
        @use_named_args(self.bayesian_space)
        def objective(**params):
            # Handle categorical parameters that might not be in Bayesian space
            param_names = [dim.name for dim in self.bayesian_space]
            if 'max_features' not in param_names and 'max_features' not in params:
                params['max_features'] = 'sqrt'  # default
                
            score = self._evaluate_model(params, X, y)
            
            # Bayesian optimization minimizes, so negate if we want to maximize
            return -score if self.scoring in ['accuracy', 'f1', 'r2'] else score
        
        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.bayesian_space,
            n_calls=n_calls,
            random_state=self.random_state,
            verbose=verbose
        )
        
        # Extract best parameters
        best_params = {}
        param_names = [dim.name for dim in self.bayesian_space]
        for i, param_name in enumerate(param_names):
            best_params[param_name] = result.x[i]
        
        # Add any default parameters not in Bayesian space
        if 'max_features' not in param_names:
            best_params['max_features'] = 'sqrt'
        
        self.best_params_ = best_params
        self.best_score_ = -result.fun if self.scoring in ['accuracy', 'f1', 'r2'] else result.fun
        
        if verbose:
            print(f"\nBayesian Optimization Complete!")
            print(f"Best Score: {self.best_score_:.4f}")
            print(f"Best Parameters: {best_params}")
        
        return best_params
    
    def fit_best_model(self, X, y):
        """
        Fit a model using the best parameters found during optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training targets
            
        Returns:
        --------
        model : Trained gradient boosting model
        """
        if self.best_params_ is None:
            raise ValueError("No optimization has been performed yet. "
                           "Run bayesian_search() first.")
        
        model = self._get_model(**self.best_params_)
        model.fit(X, y)
        return model




""" Example Implemenatations for Classification and Regression """
# Example Implementation
def example_classification():
    """Example usage for classification task."""
    print("="*60)
    print("CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Generate sample classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize tuner
    tuner = GradientBoostingTuner(
        task_type='classification',
        scoring='accuracy',
        cv_folds=5,
        random_state=42
    )
    
    # Print search space
    print("\nSearch Space:")
    tuner.print_current_search_space()
    
    # Perform Bayesian optimization
    print(f"\nStarting hyperparameter optimization...")
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=30, verbose=True)
    
    # Fit best model and evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    return tuner, best_model


def example_regression():
    """Example usage for regression task."""
    print("\n" + "="*60)
    print("REGRESSION EXAMPLE")
    print("="*60)
    
    # Generate sample regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize tuner for regression
    tuner = GradientBoostingTuner(
        task_type='regression',
        scoring='r2',
        cv_folds=5,
        random_state=42
    )
    
    # Print search space
    print("\nSearch Space:")
    tuner.print_current_search_space()
    
    # Perform Bayesian optimization
    print(f"\nStarting hyperparameter optimization...")
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=30, verbose=True)
    
    # Fit best model and evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"R² Score: {test_r2:.4f}")
    print(f"MSE: {test_mse:.4f}")
    
    return tuner, best_model


def example_custom_search_space():
    """Example with custom parameter search space."""
    print("\n" + "="*60)
    print("CUSTOM SEARCH SPACE EXAMPLE")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define custom search space - more focused ranges
    if BAYESIAN_AVAILABLE:
        from skopt.space import Real, Integer
        custom_space = [
            Integer(100, 300, name='n_estimators'),      # Narrower range
            Real(0.05, 0.2, name='learning_rate'),       # Smaller learning rates
            Integer(4, 8, name='max_depth'),             # Shallower trees
            Integer(5, 15, name='min_samples_split'),    # Higher minimum splits
            Integer(2, 8, name='min_samples_leaf'),      # Higher minimum leaves
            Real(0.8, 1.0, name='subsample')            # Higher subsample ratios
        ]
    else:
        custom_space = None
        print("scikit-optimize not available, using default space")
    
    # Initialize tuner with custom space
    tuner = GradientBoostingTuner(
        task_type='classification',
        scoring='f1_weighted',
        bayesian_space=custom_space,
        random_state=42
    )
    
    print("Custom Search Space:")
    tuner.print_current_search_space()
    
    # Perform optimization
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=25, verbose=True)
    
    # Evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest F1-Score: {test_f1:.4f}")
    
    return tuner, best_model