"""
Fairness-Aware Classifier Module

This module provides a fairness-aware classifier for healthcare applications,
with a focus on bias mitigation and interpretability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import shap

class FairnessAwareClassifier:
    """
    Fairness-aware classifier for healthcare applications.
    """
    
    def __init__(self, n_features: int, n_classes: int = 2, 
                fairness_constraint: float = 0.7,
                interpretability_level: float = 0.5):
        """
        Initialize the fairness-aware classifier.
        
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            fairness_constraint: Fairness constraint (0-1, higher means more fair)
            interpretability_level: Interpretability constraint (0-1, higher means more interpretable)
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.fairness_constraint = fairness_constraint
        self.interpretability_level = interpretability_level
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize metrics
        self.metrics = {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'auc': None,
            'fairness_metrics': {},
            'feature_importance': None
        }
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
    
    def _create_model(self) -> Any:
        """
        Create the underlying model based on interpretability level.
        
        Returns:
            Model instance
        """
        # Choose model based on interpretability level
        if self.interpretability_level >= 0.7:
            # High interpretability: Logistic Regression
            model = LogisticRegression(
                C=1.0,
                penalty='l1' if self.fairness_constraint >= 0.5 else 'l2',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        else:
            # Lower interpretability: Random Forest
            # Adjust hyperparameters based on fairness constraint
            if self.fairness_constraint >= 0.7:
                # High fairness: More trees, deeper
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:
                # Lower fairness: Fewer trees, shallower
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           sensitive_attributes: Optional[Dict[str, np.ndarray]] = None,
           sample_weight: Optional[np.ndarray] = None) -> 'FairnessAwareClassifier':
        """
        Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            sensitive_attributes: Optional dictionary mapping attribute names to attribute values
            sample_weight: Optional sample weights
            
        Returns:
            Self
        """
        # Apply fairness constraints if sensitive attributes are provided
        if sensitive_attributes is not None and self.fairness_constraint > 0:
            # Adjust sample weights based on sensitive attributes
            if sample_weight is None:
                sample_weight = np.ones(len(y))
            
            # Apply fairness-aware weighting
            sample_weight = self._apply_fairness_weights(X, y, sensitive_attributes, sample_weight)
        
        # Fit the model
        self.model.fit(X, y, sample_weight=sample_weight)
        
        # Compute feature importance
        self._compute_feature_importance(X, y)
        
        # Initialize SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.Explainer(self.model.predict_proba, X)
        else:
            self.explainer = shap.Explainer(self.model, X)
        
        return self
    
    def _apply_fairness_weights(self, X: np.ndarray, y: np.ndarray,
                              sensitive_attributes: Dict[str, np.ndarray],
                              sample_weight: np.ndarray) -> np.ndarray:
        """
        Apply fairness-aware weighting.
        
        Args:
            X: Input features
            y: Target values
            sensitive_attributes: Dictionary mapping attribute names to attribute values
            sample_weight: Initial sample weights
            
        Returns:
            Adjusted sample weights
        """
        adjusted_weights = sample_weight.copy()
        
        for attr_name, attr_values in sensitive_attributes.items():
            # Calculate positive prediction rate for each group
            unique_values = np.unique(attr_values)
            
            if len(unique_values) < 2:
                continue
            
            # Calculate positive rate for each group
            positive_rates = {}
            for value in unique_values:
                group_mask = (attr_values == value)
                if np.sum(group_mask) > 0:
                    positive_rates[value] = np.mean(y[group_mask] == 1)
                else:
                    positive_rates[value] = 0.0
            
            # Calculate average positive rate
            avg_positive_rate = np.mean(list(positive_rates.values()))
            
            # Adjust weights to balance positive rates
            for value in unique_values:
                group_mask = (attr_values == value)
                group_positive_rate = positive_rates[value]
                
                if group_positive_rate > 0:
                    # Calculate adjustment factor
                    adjustment_factor = (avg_positive_rate / group_positive_rate) ** self.fairness_constraint
                    
                    # Apply adjustment to positive examples in this group
                    positive_mask = group_mask & (y == 1)
                    adjusted_weights[positive_mask] *= adjustment_factor
        
        # Normalize weights
        if np.sum(adjusted_weights) > 0:
            adjusted_weights = adjusted_weights * (len(y) / np.sum(adjusted_weights))
        
        return adjusted_weights
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute feature importance.
        
        Args:
            X: Input features
            y: Target values
        """
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            self.metrics['feature_importance'] = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            self.metrics['feature_importance'] = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            # Use permutation importance
            perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
            self.metrics['feature_importance'] = perm_importance.importances_mean
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Convert binary predictions to probabilities
            preds = self.predict(X)
            probs = np.zeros((len(X), 2))
            probs[:, 1] = preds
            probs[:, 0] = 1 - preds
            return probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                sensitive_attributes: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            X: Input features
            y: Target values
            sensitive_attributes: Optional dictionary mapping attribute names to attribute values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate performance metrics
        self.metrics['accuracy'] = accuracy_score(y, y_pred)
        
        if self.n_classes == 2:
            self.metrics['precision'] = precision_score(y, y_pred)
            self.metrics['recall'] = recall_score(y, y_pred)
            self.metrics['f1_score'] = f1_score(y, y_pred)
            
            # Calculate AUC if predict_proba is available
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.predict_proba(X)[:, 1]
                self.metrics['auc'] = roc_auc_score(y, y_prob)
        else:
            self.metrics['precision'] = precision_score(y, y_pred, average='weighted')
            self.metrics['recall'] = recall_score(y, y_pred, average='weighted')
            self.metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
            
            # Calculate AUC if predict_proba is available
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.predict_proba(X)
                self.metrics['auc'] = roc_auc_score(y, y_prob, multi_class='ovr')
        
        # Calculate fairness metrics if sensitive attributes are provided
        if sensitive_attributes is not None:
            fairness_metrics = {}
            
            for attr_name, attr_values in sensitive_attributes.items():
                attr_metrics = {}
                
                # Calculate demographic parity difference
                attr_metrics['demographic_parity_difference'] = self._calculate_demographic_parity(
                    y_pred, attr_values
                )
                
                # Calculate equal opportunity difference
                attr_metrics['equal_opportunity_difference'] = self._calculate_equal_opportunity(
                    y, y_pred, attr_values
                )
                
                fairness_metrics[attr_name] = attr_metrics
            
            self.metrics['fairness_metrics'] = fairness_metrics
        
        return self.metrics
    
    def _calculate_demographic_parity(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate demographic parity difference.
        
        Args:
            y_pred: Predicted target values
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Demographic parity difference
        """
        # Get unique attribute values
        attr_values = np.unique(sensitive_attr)
        
        if len(attr_values) < 2:
            return 0.0
        
        # Calculate positive prediction rate for each group
        positive_rates = {}
        for value in attr_values:
            group_mask = (sensitive_attr == value)
            if np.sum(group_mask) > 0:
                positive_rates[value] = np.mean(y_pred[group_mask] == 1)
            else:
                positive_rates[value] = 0.0
        
        # Calculate maximum difference
        max_diff = max(positive_rates.values()) - min(positive_rates.values())
        
        return max_diff
    
    def _calculate_equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   sensitive_attr: np.ndarray) -> Optional[float]:
        """
        Calculate equal opportunity difference.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Equal opportunity difference or None if not applicable
        """
        # Get unique attribute values
        attr_values = np.unique(sensitive_attr)
        
        if len(attr_values) < 2:
            return 0.0
        
        # Check if binary classification
        if len(np.unique(y_true)) != 2:
            return None
        
        # Calculate true positive rate for each group
        tpr = {}
        for value in attr_values:
            group_mask = (sensitive_attr == value) & (y_true == 1)
            if np.sum(group_mask) > 0:
                tpr[value] = np.mean(y_pred[group_mask] == 1)
            else:
                tpr[value] = 0.0
        
        # Calculate maximum difference
        max_diff = max(tpr.values()) - min(tpr.values())
        
        return max_diff
    
    def explain(self, X: np.ndarray, instance_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate explanations for model predictions.
        
        Args:
            X: Input features
            instance_index: Optional index of instance to explain (if None, generates global explanations)
            
        Returns:
            Dictionary with explanations
        """
        if self.explainer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        explanations = {}
        
        # Generate SHAP values if not already computed
        if self.shap_values is None:
            self.shap_values = self.explainer(X)
        
        # Generate global explanations
        explanations['global'] = {
            'feature_importance': self.metrics['feature_importance'].tolist() if self.metrics['feature_importance'] is not None else None,
            'shap_importance': np.abs(self.shap_values.values).mean(0).tolist()
        }
        
        # Generate local explanations for specific instance if requested
        if instance_index is not None:
            if 0 <= instance_index < len(X):
                explanations['local'] = {
                    'instance_index': instance_index,
                    'prediction': self.predict(X[instance_index:instance_index+1])[0],
                    'shap_values': self.shap_values[instance_index].values.tolist(),
                    'base_value': self.shap_values[instance_index].base_values
                }
        
        return explanations
    
    def plot_feature_importance(self, feature_names: Optional[List[str]] = None,
                              output_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: Optional list of feature names
            output_path: Optional path to save the plot
        """
        if self.metrics['feature_importance'] is None:
            raise ValueError("Feature importance not computed. Call fit() first.")
        
        # Use default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features)]
        
        # Sort features by importance
        indices = np.argsort(self.metrics['feature_importance'])[::-1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), self.metrics['feature_importance'][indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_fairness_metrics(self, output_path: Optional[str] = None) -> None:
        """
        Plot fairness metrics.
        
        Args:
            output_path: Optional path to save the plot
        """
        if not self.metrics['fairness_metrics']:
            raise ValueError("Fairness metrics not computed. Call evaluate() with sensitive_attributes.")
        
        # Create figure
        n_attributes = len(self.metrics['fairness_metrics'])
        fig, axes = plt.subplots(n_attributes, 1, figsize=(10, 4 * n_attributes))
        
        # Handle single attribute case
        if n_attributes == 1:
            axes = [axes]
        
        # Plot fairness metrics for each attribute
        for i, (attribute, metrics) in enumerate(self.metrics['fairness_metrics'].items()):
            ax = axes[i]
            
            # Create bar chart
            metric_names = []
            metric_values = []
            
            if 'demographic_parity_difference' in metrics:
                metric_names.append('Demographic Parity Diff')
                metric_values.append(metrics['demographic_parity_difference'])
            
            if 'equal_opportunity_difference' in metrics and metrics['equal_opportunity_difference'] is not None:
                metric_names.append('Equal Opportunity Diff')
                metric_values.append(metrics['equal_opportunity_difference'])
            
            ax.bar(metric_names, metric_values, color='blue')
            
            # Add reference line at 0.1 (common threshold)
            ax.axhline(y=0.1, color='red', linestyle='--')
            
            # Add labels
            ax.set_ylabel('Difference')
            ax.set_title(f'Fairness Metrics for {attribute}')
            
            # Add text annotations
            for j, value in enumerate(metric_values):
                ax.text(j, value + 0.01, f'{value:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_shap_summary(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                        output_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary.
        
        Args:
            X: Input features
            feature_names: Optional list of feature names
            output_path: Optional path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate SHAP values if not already computed
        if self.shap_values is None:
            self.shap_values = self.explainer(X)
        
        # Use default feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features)]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=feature_names, show=False)
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            'model_type': type(self.model).__name__,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'fairness_constraint': self.fairness_constraint,
            'interpretability_level': self.interpretability_level,
            'metrics': self.metrics
        }
        
        # Add model parameters
        if hasattr(self.model, 'get_params'):
            model_info['model_parameters'] = self.model.get_params()
        
        return model_info


