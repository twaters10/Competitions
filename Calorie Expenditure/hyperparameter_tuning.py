# Import for Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define the search space for Random Forest Regressor hyperparameters
# Real for continuous values, Integer for integer values
def bayseian_search_space(  n_est_min, n_est_max, 
                            feature_split_min, feature_split_max, 
                            depth_min, depth_max, 
                            sample_split_min, sample_split_max,
                            sample_leaf_min, sample_leaf_max):
    search_space = {
        # Number of trees from 50 to 200
        'n_estimators': Integer(n_est_min, n_est_max),  
        # Percentage of features to consider at each split
        'max_features': Real(feature_split_min, feature_split_max, prior='uniform'), 
        # Maximum depth of the tree
        'max_depth': Integer(depth_min, depth_max),    
        # Minimum samples required to split a node  
        'min_samples_split': Integer(sample_split_min, sample_split_max), 
        # Minimum samples required at a leaf node
        'min_samples_leaf': Integer(sample_leaf_min, sample_leaf_max)    
    }

    print("\nHyperparameter search space defined:")
    for param, space in search_space.items():
        print(f"  {param}: {space}")
        
    return search_space