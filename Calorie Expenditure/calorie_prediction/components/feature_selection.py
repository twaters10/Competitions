import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import List, Union, Dict, Any, Optional

class FeatureSelectorRF:
    """
    A class to select important features using a Random Forest model
    based on a specified relative importance threshold.
    """

    def __init__(self,
                 model_type: str = 'classifier',
                 n_estimators: int = 100,
                 random_state: Optional[int] = None,
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the FeatureSelectorRF.

        Args:
            model_type (str): Type of Random Forest model to use ('classifier' or 'regressor').
                              Defaults to 'classifier'.
            n_estimators (int): The number of trees in the forest. Defaults to 100.
            random_state (Optional[int]): Controls the randomness of the bootstrapping
                                          of the samples and the splitting of features.
                                          Pass an int for reproducible results. Defaults to None.
            model_params (Optional[Dict[str, Any]]): A dictionary of additional parameters
                                                    to pass to the RandomForestClassifier/Regressor constructor.
        """
        if model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'.")

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_params = model_params if model_params is not None else {}
        self.feature_importances_ = None
        self.feature_names_ = None
        self.rf_model = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """
        Fits the Random Forest model to compute feature importances.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (Union[pd.Series, np.ndarray]): The target variable.
        """
        if self.model_type == 'classifier':
            self.rf_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.model_params
            )
        else: # regressor
            self.rf_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.model_params
            )

        print(f"Training Random Forest {self.model_type}...")
        self.rf_model.fit(X, y)
        self.feature_importances_ = self.rf_model.feature_importances_
        self.feature_names_ = X.columns.tolist()
        print("Random Forest training complete.")

    def get_important_variables(self, threshold: float = 0.01) -> List[str]:
        """
        Retrieves a list of important variable names based on a relative importance threshold.

        Args:
            threshold (float): The minimum relative importance (as a fraction of total importance)
                               a feature must have to be considered 'important'.
                               Defaults to 0.01 (1%).

        Returns:
            List[str]: A list of variable names that meet or exceed the importance threshold.

        Raises:
            RuntimeError: If the Random Forest model has not been fitted yet.
        """
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise RuntimeError("Model has not been fitted yet. Call the 'fit' method first.")

        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1 (inclusive).")

        # Create a DataFrame for easier sorting and filtering
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.feature_importances_
        })

        # Sort by importance in descending order
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Filter features based on the threshold
        important_vars = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

        print(f"Identified {len(important_vars)} important variables with threshold >= {threshold:.4f}:")
        for i, (feature, importance) in enumerate(importance_df[importance_df['Importance'] >= threshold].values):
            print(f"  {i+1}. {feature}: {importance:.4f}")

        if not important_vars:
            print("No variables met the specified importance threshold.")

        return important_vars

    def plot_feature_importances(self, top_n: Optional[int] = None):
        """
        Plots the feature importances.

        Args:
            top_n (Optional[int]): Number of top features to display. If None, all features are shown.
        """
        if self.feature_importances_ is None or self.feature_names_ is None:
            raise RuntimeError("Model has not been fitted yet. Call the 'fit' method first.")

        importances = pd.Series(self.feature_importances_, index=self.feature_names_)
        importances = importances.sort_values(ascending=False)

        if top_n:
            importances = importances.head(top_n)

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, max(6, len(importances) * 0.4)))
        sns.barplot(x=importances.values, y=importances.index, palette='viridis')
        plt.title('Random Forest Feature Importances')
        plt.xlabel('Relative Importance (Gini/MSE)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()