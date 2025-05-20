"""
Energy Model Module

This module provides a gradient boosting model for energy demand forecasting
with interpretability constraints.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import joblib

class GBMEnergyForecaster:
    """
    Gradient Boosting Model for energy demand forecasting with interpretability constraints.
    """
    
    def __init__(self, n_features: int, interpretability_level: float = 0.5):
        """
        Initialize the energy forecasting model with interpretability constraints.
        
        Args:
            n_features: Number of input features
            interpretability_level: Interpretability constraint (0-1, higher means more interpretable)
        """
        self.n_features = n_features
        self.interpretability_level = interpretability_level
        self.model = self._build_model()
        self.feature_names = None
        
    def _build_model(self) -> GradientBoostingRegressor:
        """
        Build the GBM model with interpretability constraints.
        
        Returns:
            Gradient Boosting Regressor
        """
        # Scale hyperparameters based on interpretability
        # Higher interpretability means simpler model with fewer parameters
        max_depth = int(3 + (1 - self.interpretability_level) * 7)  # Between 3 and 10
        n_estimators = int(50 + (1 - self.interpretability_level) * 150)  # Between 50 and 200
        
        # Create model with appropriate complexity
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             feature_names: List[str] = None,
             sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the model with optional validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            feature_names: Optional list of feature names
            sample_weight: Optional sample weights for training
            
        Returns:
            Dictionary with training metrics
        """
        # Store feature names if provided
        self.feature_names = feature_names
        
        # Train the model
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        # Get training metrics
        train_score = self.model.score(X_train, y_train)
        train_predictions = self.model.predict(X_train)
        train_mse = np.mean((train_predictions - y_train) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(train_predictions - y_train))
        
        # Calculate validation metrics if validation data is provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            val_predictions = self.model.predict(X_val)
            val_mse = np.mean((val_predictions - y_val) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(val_predictions - y_val))
            
            val_metrics = {
                'val_score': val_score,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae
            }
        
        # Get feature importances
        feature_importances = self.get_feature_importances()
        
        # Calculate model complexity metrics
        model_complexity = self._calculate_model_complexity()
        
        # Combine metrics
        results = {
            'training_metrics': {
                'train_score': train_score,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                **val_metrics
            },
            'feature_importances': feature_importances,
            'model_complexity': model_complexity
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        
        # Calculate R-squared
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Calculate MAPE if no zeros in y_test
        if not np.any(y_test == 0):
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        else:
            mape = None
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'mape': mape
        }
        
        return metrics
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model has not been trained yet")
        
        importances = self.model.feature_importances_
        
        # If feature names are provided, return a dictionary mapping names to importances
        if self.feature_names is not None:
            if len(self.feature_names) != len(importances):
                raise ValueError(f"Number of feature names ({len(self.feature_names)}) "
                               f"does not match number of features ({len(importances)})")
            
            return {name: float(importance) for name, importance in zip(self.feature_names, importances)}
        else:
            return {f"feature_{i}": float(importance) for i, importance in enumerate(importances)}
    
    def plot_feature_importances(self, output_path: str = None) -> None:
        """
        Plot feature importances.
        
        Args:
            output_path: Optional path to save the plot
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model has not been trained yet")
        
        importances = self.get_feature_importances()
        
        # Sort importances
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        feature_names = [item[0] for item in sorted_importances]
        importance_values = [item[1] for item in sorted_importances]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_values)), importance_values, align='center')
        plt.yticks(range(len(importance_values)), feature_names)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def _calculate_model_complexity(self) -> Dict[str, Any]:
        """
        Calculate model complexity metrics.
        
        Returns:
            Dictionary with model complexity metrics
        """
        # Get model parameters
        n_estimators = self.model.n_estimators
        max_depth = self.model.max_depth
        
        # Calculate total number of nodes (approximate)
        # For a balanced tree with max_depth, the number of nodes is 2^(max_depth+1) - 1
        # But GBM trees are often not fully balanced, so this is an upper bound
        max_nodes_per_tree = 2**(max_depth + 1) - 1
        max_total_nodes = n_estimators * max_nodes_per_tree
        
        # Calculate memory footprint (rough estimate)
        # Each node typically stores a split value, feature index, and pointers to children
        bytes_per_node = 16  # Approximate
        memory_footprint_bytes = max_total_nodes * bytes_per_node
        memory_footprint_mb = memory_footprint_bytes / (1024 * 1024)
        
        return {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_nodes_per_tree': max_nodes_per_tree,
            'max_total_nodes': max_total_nodes,
            'memory_footprint_mb': memory_footprint_mb,
            'interpretability_level': self.interpretability_level
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for sustainability evaluation.
        
        Returns:
            Dictionary with model information
        """
        # Get model complexity
        model_complexity = self._calculate_model_complexity()
        
        # Get feature importances if model is trained
        feature_importances = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.get_feature_importances()
        
        # Combine information
        model_info = {
            'model_type': 'gbm_energy_forecaster',
            'n_features': self.n_features,
            'interpretability_constraint': self.interpretability_level,
            'complexity': model_complexity,
            'feature_importances': feature_importances,
            'feature_names': self.feature_names,
            'num_parameters': model_complexity['max_total_nodes'],
            'model_parameters': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'subsample': self.model.subsample
            }
        }
        
        return model_info
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'n_features': self.n_features,
            'interpretability_level': self.interpretability_level,
            'feature_names': self.feature_names
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GBMEnergyForecaster':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded GBMEnergyForecaster instance
        """
        # Load model and metadata
        data = joblib.load(filepath)
        
        # Create instance
        instance = cls(data['n_features'], data['interpretability_level'])
        
        # Restore model and feature names
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        
        return instance
