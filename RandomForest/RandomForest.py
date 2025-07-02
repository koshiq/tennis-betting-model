import numpy as np
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_features, n_estimators=100, tree_params=dict(max_depth=20, min_samples_split=10, bagging=True), bagging=True):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.tree_params = tree_params
        self.estimators = []
        self.feature_importances = None
    
    def build_forest(self, data):
        # Initialize feature importances
        self.feature_importances = np.zeros(data.shape[1] - 1)
        
        for _ in range(self.n_estimators):
            # Use feature importances for sampling if available
            if self.feature_importances is not None and np.sum(self.feature_importances) > 0:
                # Normalize feature importances
                normalized_importances = self.feature_importances / np.sum(self.feature_importances)
                # Sample features based on importance
                selected_features = np.random.choice(
                    data.shape[1] - 1,
                    size=self.n_features,
                    p=normalized_importances,
                    replace=False
                )
            else:
                selected_features = np.random.choice(
                    data.shape[1] - 1,
                    size=self.n_features,
                    replace=False
                )
            
            new_tree = DecisionTree.DecisionTree(**self.tree_params)
            new_tree.build_tree(data, self.n_features)
            
            # Update feature importances
            if hasattr(new_tree, 'feature_importances'):
                self.feature_importances += new_tree.feature_importances
            
            self.estimators.append(new_tree)
        
        # Normalize final feature importances
        if self.feature_importances is not None:
            self.feature_importances /= self.n_estimators

    def predict(self, data, print_predictions=False):
        predictions = []
        weights = []
        
        for tree in self.estimators:
            prediction = tree.predict(data)
            # Calculate tree weight based on its accuracy
            weight = tree.calculate_accuracy(data)
            predictions.append(prediction)
            weights.append(weight)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted voting
        weighted_predictions = np.zeros((data.shape[0], 1))
        for i in range(len(self.estimators)):
            weighted_predictions += weights[i] * predictions[i]
        
        # Convert to binary predictions
        output_array = (weighted_predictions > 0.5).astype(int)
        
        if print_predictions:
            print(predictions)
        
        return output_array

    def calculate_accuracy(self, data):
        labels = data[:, data.shape[1]-1:]
        samples = data[:, :data.shape[1]-1]
        
        results = self.predict(samples)
        
        return np.mean(results == labels)

# Example of improved hyperparameters
params = dict(
    max_depth=15,  # Reduced from 20 to prevent overfitting
    min_samples_split=20,  # Increased from 10 for better generalization
    min_gini_change=0.01,  # Added minimum gini change threshold
    bagging=True
)
forest = RandomForest(
    n_features=7,  # Increased from 5 to use more features
    n_estimators=200,  # Increased from 100 for better ensemble
    tree_params=params
)
