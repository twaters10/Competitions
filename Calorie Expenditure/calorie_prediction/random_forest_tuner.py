import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization unavailable.")

class RandomForestTuner:
    """
    Hyperparameter tuning class for Random Forest models using Bayesian optimization.
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
                # Default Bayesian optimization space for Random Forest
                self.bayesian_space = [
                    Integer(50, 500, name='n_estimators'),
                    Integer(3, 30, name='max_depth'),
                    Integer(2, 20, name='min_samples_split'),
                    Integer(1, 10, name='min_samples_leaf'),
                    Categorical(['sqrt', 'log2', None], name='max_features'),
                    Real(0.1, 1.0, name='max_samples'),
                    Categorical([True, False], name='bootstrap')
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
            Integer(3, 30, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf'),
            Categorical(['sqrt', 'log2', None], name='max_features'),
            Real(0.1, 1.0, name='max_samples'),
            Categorical([True, False], name='bootstrap')
        ]
    
    def print_current_search_space(self):
        """Print the current parameter search space."""
        if BAYESIAN_AVAILABLE and hasattr(self, 'bayesian_space'):
            print("Current Bayesian Optimization Parameter Space:")
            print("-" * 50)
            for dim in self.bayesian_space:
                if hasattr(dim, 'low') and hasattr(dim, 'high'):
                    print(f"{dim.name}: {type(dim).__name__}({dim.low}, {dim.high})")
                elif hasattr(dim, 'categories'):
                    print(f"{dim.name}: Categorical{dim.categories}")
                else:
                    print(f"{dim.name}: {dim}")
        else:
            print("Bayesian optimization space not available")
    
    def _get_model(self, **params):
        """Get Random Forest model with specified parameters."""
        if self.task_type == 'classification':
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **params)
        else:
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **params)
    
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
        model : Trained Random Forest model
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
    print("RANDOM FOREST CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Generate sample classification data
    X, y = make_classification(
        n_samples=2000, 
        n_features=25, 
        n_informative=20, 
        n_redundant=5, 
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize tuner
    tuner = RandomForestTuner(
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
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=40, verbose=True)
    
    # Fit best model and evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    # Feature importance (top 10)
    feature_importance = best_model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print(f"\nTop 10 Feature Importances:")
    for i, idx in enumerate(top_features):
        print(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    return tuner, best_model


def example_regression():
    """Example usage for regression task."""
    print("\n" + "="*60)
    print("RANDOM FOREST REGRESSION EXAMPLE")
    print("="*60)
    
    # Generate sample regression data
    X, y = make_regression(
        n_samples=1500,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Initialize tuner for regression
    tuner = RandomForestTuner(
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
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=40, verbose=True)
    
    # Fit best model and evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    
    print(f"\nTest Results:")
    print(f"R² Score: {test_r2:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Feature importance (top 10)
    feature_importance = best_model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print(f"\nTop 10 Feature Importances:")
    for i, idx in enumerate(top_features):
        print(f"Feature {idx}: {feature_importance[idx]:.4f}")
    
    return tuner, best_model


def example_custom_search_space():
    """Example with custom parameter search space for Random Forest."""
    print("\n" + "="*60)
    print("CUSTOM SEARCH SPACE EXAMPLE")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_features=15, 
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define custom search space - focused on ensemble size and tree depth
    if BAYESIAN_AVAILABLE:
        from skopt.space import Real, Integer, Categorical
        custom_space = [
            Integer(200, 800, name='n_estimators'),      # More trees
            Integer(5, 20, name='max_depth'),            # Moderate depth
            Integer(10, 50, name='min_samples_split'),   # Higher splits
            Integer(5, 20, name='min_samples_leaf'),     # Higher leaves
            Categorical(['sqrt', 'log2'], name='max_features'),  # Only good options
            Real(0.7, 1.0, name='max_samples'),          # High sampling
            Categorical([True], name='bootstrap')         # Always use bootstrap
        ]
    else:
        custom_space = None
        print("scikit-optimize not available, using default space")
    
    # Initialize tuner with custom space
    tuner = RandomForestTuner(
        task_type='classification',
        scoring='f1_weighted',
        bayesian_space=custom_space,
        random_state=42
    )
    
    print("Custom Search Space:")
    tuner.print_current_search_space()
    
    # Perform optimization
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=30, verbose=True)
    
    # Evaluate
    best_model = tuner.fit_best_model(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    # Model info
    print(f"\nBest Model Info:")
    print(f"Number of trees: {best_model.n_estimators}")
    print(f"Max depth: {best_model.max_depth}")
    print(f"Out-of-bag score: {best_model.oob_score_:.4f}" if best_model.bootstrap else "N/A (no bootstrap)")
    
    return tuner, best_model


def example_feature_selection():
    """Example showing feature importance analysis with Random Forest."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS EXAMPLE")
    print("="*60)
    
    # Generate data with some irrelevant features
    X, y = make_classification(
        n_samples=1500,
        n_features=30,
        n_informative=10,  # Only 10 out of 30 features are informative
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Total features: {X.shape[1]}")
    print(f"Informative features: 10")
    print(f"Redundant features: 10")
    print(f"Random features: 10")
    
    # Use a focused search space for feature importance analysis
    if BAYESIAN_AVAILABLE:
        from skopt.space import Integer, Categorical
        focused_space = [
            Integer(100, 300, name='n_estimators'),
            Integer(5, 15, name='max_depth'),
            Integer(2, 10, name='min_samples_split'),
            Integer(1, 5, name='min_samples_leaf'),
            Categorical(['sqrt', 'log2'], name='max_features'),
            Categorical([True], name='bootstrap')
        ]
    else:
        focused_space = None
    
    # Initialize tuner
    tuner = RandomForestTuner(
        task_type='classification',
        scoring='accuracy',
        bayesian_space=focused_space,
        random_state=42
    )
    
    # Optimize
    best_params = tuner.bayesian_search(X_train, y_train, n_calls=25, verbose=True)
    best_model = tuner.fit_best_model(X_train, y_train)
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Analyze feature importance
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    print(f"\nFeature Importance Analysis:")
    print("-" * 40)
    print(f"{'Rank':<4} {'Feature':<10} {'Importance':<12} {'Type':<12}")
    print("-" * 40)
    
    for rank, idx in enumerate(sorted_idx[:20], 1):  # Top 20 features
        if idx < 10:
            feature_type = "Informative"
        elif idx < 20:
            feature_type = "Redundant"
        else:
            feature_type = "Random"
        
        print(f"{rank:<4} Feature_{idx:<4} {feature_importance[idx]:<12.4f} {feature_type:<12}")
    
    # Summary statistics
    informative_importance = np.mean(feature_importance[:10])
    redundant_importance = np.mean(feature_importance[10:20])
    random_importance = np.mean(feature_importance[20:])
    
    print(f"\nImportance Summary:")
    print(f"Informative features (0-9): {informative_importance:.4f}")
    print(f"Redundant features (10-19): {redundant_importance:.4f}")
    print(f"Random features (20-29): {random_importance:.4f}")
    
    return tuner, best_model