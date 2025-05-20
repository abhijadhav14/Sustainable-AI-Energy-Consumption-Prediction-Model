"""
Data Processor Module

This module provides tools for processing and preparing data for sustainable AI models,
with a focus on bias detection and mitigation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class DataProcessor:
    """
    Processes and prepares data for sustainable AI models.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
        self.processed_data = None
        self.sensitive_attributes = []
        self.target_column = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.preprocessing_steps = []
        self.bias_metrics = {}
    
    def load_data(self, data_path: str, file_format: str = 'auto') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to the data file
            file_format: Format of the data file ('csv', 'excel', 'json', 'auto')
            
        Returns:
            Loaded data as DataFrame
        """
        # Determine file format if auto
        if file_format == 'auto':
            if data_path.endswith('.csv'):
                file_format = 'csv'
            elif data_path.endswith(('.xls', '.xlsx')):
                file_format = 'excel'
            elif data_path.endswith('.json'):
                file_format = 'json'
            else:
                raise ValueError(f"Could not determine file format for {data_path}")
        
        # Load data based on format
        if file_format == 'csv':
            self.data = pd.read_csv(data_path)
        elif file_format == 'excel':
            self.data = pd.read_excel(data_path)
        elif file_format == 'json':
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'load_data',
            'data_path': data_path,
            'file_format': file_format,
            'rows': len(self.data),
            'columns': len(self.data.columns)
        })
        
        return self.data
    
    def set_column_types(self, target_column: str, 
                        feature_columns: Optional[List[str]] = None,
                        categorical_columns: Optional[List[str]] = None,
                        numerical_columns: Optional[List[str]] = None,
                        sensitive_attributes: Optional[List[str]] = None) -> None:
        """
        Set column types for the dataset.
        
        Args:
            target_column: Name of the target column
            feature_columns: Optional list of feature columns (if None, all columns except target)
            categorical_columns: Optional list of categorical columns
            numerical_columns: Optional list of numerical columns
            sensitive_attributes: Optional list of sensitive attributes for bias detection
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.target_column = target_column
        
        # Set feature columns
        if feature_columns is None:
            self.feature_columns = [col for col in self.data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
        
        # Set categorical and numerical columns
        if categorical_columns is None and numerical_columns is None:
            # Auto-detect column types
            self.categorical_columns = [col for col in self.feature_columns 
                                      if self.data[col].dtype == 'object' or 
                                      (self.data[col].dtype == 'int64' and self.data[col].nunique() < 10)]
            self.numerical_columns = [col for col in self.feature_columns if col not in self.categorical_columns]
        else:
            if categorical_columns is not None:
                self.categorical_columns = categorical_columns
            if numerical_columns is not None:
                self.numerical_columns = numerical_columns
        
        # Set sensitive attributes
        if sensitive_attributes is not None:
            self.sensitive_attributes = sensitive_attributes
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'set_column_types',
            'target_column': target_column,
            'feature_columns': len(self.feature_columns),
            'categorical_columns': len(self.categorical_columns),
            'numerical_columns': len(self.numerical_columns),
            'sensitive_attributes': self.sensitive_attributes
        })
    
    def preprocess_data(self, handle_missing: bool = True,
                       normalize_numerical: bool = True,
                       encode_categorical: bool = True,
                       normalization_method: str = 'standard') -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            handle_missing: Whether to handle missing values
            normalize_numerical: Whether to normalize numerical features
            encode_categorical: Whether to encode categorical features
            normalization_method: Method for normalizing numerical features ('standard', 'minmax')
            
        Returns:
            Preprocessed data as DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.feature_columns or not self.target_column:
            raise ValueError("Column types not set. Call set_column_types() first.")
        
        # Create a copy of the data
        self.processed_data = self.data.copy()
        
        # Handle missing values
        if handle_missing:
            self._handle_missing_values()
        
        # Normalize numerical features
        if normalize_numerical and self.numerical_columns:
            self._normalize_numerical_features(method=normalization_method)
        
        # Encode categorical features
        if encode_categorical and self.categorical_columns:
            self._encode_categorical_features()
        
        return self.processed_data
    
    def _handle_missing_values(self) -> None:
        """Handle missing values in the data."""
        # Record initial missing values
        initial_missing = self.processed_data.isnull().sum().sum()
        
        # Handle missing values in numerical columns
        for col in self.numerical_columns:
            if self.processed_data[col].isnull().any():
                # Fill with median
                median_value = self.processed_data[col].median()
                self.processed_data[col].fillna(median_value, inplace=True)
        
        # Handle missing values in categorical columns
        for col in self.categorical_columns:
            if self.processed_data[col].isnull().any():
                # Fill with mode
                mode_value = self.processed_data[col].mode()[0]
                self.processed_data[col].fillna(mode_value, inplace=True)
        
        # Record preprocessing step
        final_missing = self.processed_data.isnull().sum().sum()
        self.preprocessing_steps.append({
            'step': 'handle_missing_values',
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'values_imputed': initial_missing - final_missing
        })
    
    def _normalize_numerical_features(self, method: str = 'standard') -> None:
        """
        Normalize numerical features.
        
        Args:
            method: Normalization method ('standard', 'minmax')
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Apply scaling
        if self.numerical_columns:
            self.processed_data[self.numerical_columns] = scaler.fit_transform(
                self.processed_data[self.numerical_columns]
            )
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'normalize_numerical_features',
            'method': method,
            'columns_normalized': len(self.numerical_columns)
        })
    
    def _encode_categorical_features(self) -> None:
        """Encode categorical features using one-hot encoding."""
        # Apply one-hot encoding
        encoded_data = pd.get_dummies(
            self.processed_data[self.categorical_columns], 
            drop_first=True,
            prefix=self.categorical_columns
        )
        
        # Remove original categorical columns and add encoded columns
        self.processed_data = pd.concat([
            self.processed_data.drop(columns=self.categorical_columns),
            encoded_data
        ], axis=1)
        
        # Update feature columns
        new_categorical_columns = encoded_data.columns.tolist()
        self.feature_columns = [col for col in self.feature_columns if col not in self.categorical_columns] + new_categorical_columns
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'encode_categorical_features',
            'original_categorical_columns': len(self.categorical_columns),
            'encoded_columns': len(new_categorical_columns)
        })
        
        # Update categorical columns
        self.categorical_columns = new_categorical_columns
    
    def split_data(self, test_size: float = 0.2, validation_size: Optional[float] = None,
                 random_state: int = 42, stratify: bool = True) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Split data into train, test, and optionally validation sets.
        
        Args:
            test_size: Proportion of data to use for testing
            validation_size: Optional proportion of training data to use for validation
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits based on target variable
            
        Returns:
            Dictionary with train, test, and optionally validation data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        # Prepare features and target
        X = self.processed_data[self.feature_columns]
        y = self.processed_data[self.target_column]
        
        # Determine stratify parameter
        stratify_param = y if stratify else None
        
        # Split data into train and test sets
        if validation_size is None:
            # Simple train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
            
            # Record preprocessing step
            self.preprocessing_steps.append({
                'step': 'split_data',
                'test_size': test_size,
                'validation_size': None,
                'stratify': stratify,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:
            # Calculate effective validation size
            effective_validation_size = validation_size / (1 - test_size)
            
            # First split into train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
            
            # Then split train+val into train and val
            stratify_param_val = y_train_val if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=effective_validation_size, 
                random_state=random_state, stratify=stratify_param_val
            )
            
            # Record preprocessing step
            self.preprocessing_steps.append({
                'step': 'split_data',
                'test_size': test_size,
                'validation_size': validation_size,
                'stratify': stratify,
                'train_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test)
            })
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
    
    def detect_bias(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Detect bias in the data with respect to sensitive attributes.
        
        Args:
            data: Optional data to analyze (if None, uses processed_data)
            
        Returns:
            Dictionary with bias metrics for each sensitive attribute
        """
        if data is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call preprocess_data() first.")
            data = self.processed_data
        
        if not self.sensitive_attributes:
            raise ValueError("No sensitive attributes specified. Call set_column_types() with sensitive_attributes.")
        
        bias_metrics = {}
        
        for attribute in self.sensitive_attributes:
            if attribute not in data.columns:
                continue
            
            attribute_metrics = {}
            
            # Calculate class distribution by attribute
            class_dist = data.groupby([attribute, self.target_column]).size().unstack().fillna(0)
            
            # Normalize to get proportions
            class_dist_norm = class_dist.div(class_dist.sum(axis=1), axis=0)
            
            # Calculate statistical parity difference
            if len(class_dist_norm.columns) == 2:  # Binary classification
                positive_class = class_dist_norm.columns[1]
                groups = class_dist_norm.index.tolist()
                
                if len(groups) >= 2:
                    max_group_rate = class_dist_norm[positive_class].max()
                    min_group_rate = class_dist_norm[positive_class].min()
                    
                    # Statistical parity difference
                    attribute_metrics['demographic_parity_difference'] = max_group_rate - min_group_rate
                    
                    # Group with highest positive rate
                    attribute_metrics['highest_rate_group'] = class_dist_norm[positive_class].idxmax()
                    
                    # Group with lowest positive rate
                    attribute_metrics['lowest_rate_group'] = class_dist_norm[positive_class].idxmin()
            
            bias_metrics[attribute] = attribute_metrics
        
        # Store bias metrics
        self.bias_metrics = bias_metrics
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'detect_bias',
            'sensitive_attributes': self.sensitive_attributes,
            'bias_detected': {attr: metrics.get('demographic_parity_difference', None) 
                             for attr, metrics in bias_metrics.items()}
        })
        
        return bias_metrics
    
    def mitigate_bias(self, method: str = 'reweighing', 
                     sensitive_attribute: Optional[str] = None) -> pd.DataFrame:
        """
        Mitigate bias in the data.
        
        Args:
            method: Bias mitigation method ('reweighing', 'sampling')
            sensitive_attribute: Sensitive attribute to mitigate bias for (if None, uses first attribute)
            
        Returns:
            Bias-mitigated data as DataFrame
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        if not self.sensitive_attributes:
            raise ValueError("No sensitive attributes specified. Call set_column_types() with sensitive_attributes.")
        
        # Use first sensitive attribute if none specified
        if sensitive_attribute is None:
            sensitive_attribute = self.sensitive_attributes[0]
        
        if sensitive_attribute not in self.processed_data.columns:
            raise ValueError(f"Sensitive attribute {sensitive_attribute} not found in data.")
        
        # Create a copy of the data
        mitigated_data = self.processed_data.copy()
        
        if method == 'reweighing':
            # Implement reweighing method
            weights = self._calculate_reweighing_weights(sensitive_attribute)
            
            # Apply weights (for now, just store them as a column)
            mitigated_data['sample_weight'] = mitigated_data.apply(
                lambda row: weights.get((row[sensitive_attribute], row[self.target_column]), 1.0),
                axis=1
            )
            
        elif method == 'sampling':
            # Implement sampling method
            mitigated_data = self._apply_fair_sampling(mitigated_data, sensitive_attribute)
            
        else:
            raise ValueError(f"Unsupported bias mitigation method: {method}")
        
        # Record preprocessing step
        self.preprocessing_steps.append({
            'step': 'mitigate_bias',
            'method': method,
            'sensitive_attribute': sensitive_attribute
        })
        
        return mitigated_data
    
    def _calculate_reweighing_weights(self, sensitive_attribute: str) -> Dict[Tuple[Any, Any], float]:
        """
        Calculate weights for reweighing method.
        
        Args:
            sensitive_attribute: Sensitive attribute to calculate weights for
            
        Returns:
            Dictionary mapping (attribute_value, target_value) to weight
        """
        # Get counts
        counts = self.processed_data.groupby([sensitive_attribute, self.target_column]).size()
        total = len(self.processed_data)
        
        # Calculate expected and observed probabilities
        attr_counts = self.processed_data[sensitive_attribute].value_counts()
        attr_probs = attr_counts / total
        
        target_counts = self.processed_data[self.target_column].value_counts()
        target_probs = target_counts / total
        
        # Calculate weights
        weights = {}
        for attr_val in attr_counts.index:
            for target_val in target_counts.index:
                expected_prob = attr_probs[attr_val] * target_probs[target_val]
                
                if (attr_val, target_val) in counts:
                    observed_prob = counts[(attr_val, target_val)] / total
                    weights[(attr_val, target_val)] = expected_prob / observed_prob
                else:
                    weights[(attr_val, target_val)] = 1.0
        
        return weights
    
    def _apply_fair_sampling(self, data: pd.DataFrame, sensitive_attribute: str) -> pd.DataFrame:
        """
        Apply fair sampling to mitigate bias.
        
        Args:
            data: Data to apply fair sampling to
            sensitive_attribute: Sensitive attribute to mitigate bias for
            
        Returns:
            Fairly sampled data as DataFrame
        """
        # Get unique combinations of sensitive attribute and target
        combinations = data.groupby([sensitive_attribute, self.target_column]).size()
        
        # Find minimum count for each target value
        min_counts = {}
        for target_val in data[self.target_column].unique():
            target_combinations = [(attr_val, t_val) for attr_val, t_val in combinations.index if t_val == target_val]
            if target_combinations:
                min_counts[target_val] = min(combinations[comb] for comb in target_combinations)
            else:
                min_counts[target_val] = 0
        
        # Sample from each group
        sampled_dfs = []
        for attr_val in data[sensitive_attribute].unique():
            for target_val in data[self.target_column].unique():
                group_data = data[(data[sensitive_attribute] == attr_val) & 
                                 (data[self.target_column] == target_val)]
                
                if len(group_data) > 0:
                    # Sample min_count samples from this group
                    sampled_group = group_data.sample(
                        n=min(len(group_data), min_counts[target_val]),
                        random_state=42
                    )
                    sampled_dfs.append(sampled_group)
        
        # Combine sampled data
        if sampled_dfs:
            return pd.concat(sampled_dfs, axis=0).reset_index(drop=True)
        else:
            return data
    
    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        sensitive_attributes: Optional[List[str]] = None,
                        sensitive_values: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate fairness of model predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            sensitive_attributes: Optional list of sensitive attributes to evaluate
            sensitive_values: Optional dictionary mapping attribute names to attribute values
            
        Returns:
            Dictionary with fairness metrics for each sensitive attribute
        """
        if sensitive_attributes is None:
            sensitive_attributes = self.sensitive_attributes
        
        if not sensitive_attributes:
            raise ValueError("No sensitive attributes specified.")
        
        if sensitive_values is None and self.processed_data is None:
            raise ValueError("No processed data available and no sensitive values provided.")
        
        fairness_metrics = {}
        
        for attribute in sensitive_attributes:
            # Get attribute values
            if sensitive_values is not None and attribute in sensitive_values:
                attr_values = sensitive_values[attribute]
            elif self.processed_data is not None and attribute in self.processed_data.columns:
                attr_values = self.processed_data[attribute].values
            else:
                continue
            
            if len(attr_values) != len(y_true):
                continue
            
            attribute_metrics = {}
            
            # Calculate demographic parity difference
            attribute_metrics['demographic_parity_difference'] = self._calculate_demographic_parity(
                y_pred, attr_values
            )
            
            # Calculate equal opportunity difference
            attribute_metrics['equal_opportunity_difference'] = self._calculate_equal_opportunity(
                y_true, y_pred, attr_values
            )
            
            fairness_metrics[attribute] = attribute_metrics
        
        return fairness_metrics
    
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
    
    def plot_bias_metrics(self, output_path: Optional[str] = None) -> None:
        """
        Plot bias metrics.
        
        Args:
            output_path: Optional path to save the plot
        """
        if not self.bias_metrics:
            raise ValueError("No bias metrics available. Call detect_bias() first.")
        
        # Create figure
        n_attributes = len(self.bias_metrics)
        fig, axes = plt.subplots(n_attributes, 1, figsize=(10, 4 * n_attributes))
        
        # Handle single attribute case
        if n_attributes == 1:
            axes = [axes]
        
        # Plot bias metrics for each attribute
        for i, (attribute, metrics) in enumerate(self.bias_metrics.items()):
            ax = axes[i]
            
            if 'demographic_parity_difference' in metrics:
                # Create bar chart
                dpd = metrics['demographic_parity_difference']
                ax.bar(['Demographic Parity Difference'], [dpd], color='blue')
                
                # Add reference line at 0.1 (common threshold)
                ax.axhline(y=0.1, color='red', linestyle='--')
                
                # Add labels
                ax.set_ylabel('Difference')
                ax.set_title(f'Bias Metrics for {attribute}')
                
                # Add text annotation
                ax.text(0, dpd + 0.01, f'{dpd:.3f}', ha='center')
                
                # Add interpretation
                if dpd <= 0.05:
                    interpretation = "Low bias"
                elif dpd <= 0.1:
                    interpretation = "Moderate bias"
                else:
                    interpretation = "High bias"
                
                ax.text(0.5, 0.9, f"Interpretation: {interpretation}", 
                       transform=ax.transAxes, ha='center',
                       bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_distributions(self, features: Optional[List[str]] = None,
                                 by_attribute: Optional[str] = None,
                                 output_path: Optional[str] = None) -> None:
        """
        Plot feature distributions.
        
        Args:
            features: Optional list of features to plot (if None, uses numerical_columns)
            by_attribute: Optional attribute to group by
            output_path: Optional path to save the plot
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        # Use numerical columns if no features specified
        if features is None:
            features = self.numerical_columns[:5]  # Limit to first 5 to avoid too many plots
        
        # Create figure
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features))
        
        # Handle single feature case
        if n_features == 1:
            axes = [axes]
        
        # Plot distribution for each feature
        for i, feature in enumerate(features):
            if feature not in self.processed_data.columns:
                continue
                
            ax = axes[i]
            
            if by_attribute is not None and by_attribute in self.processed_data.columns:
                # Plot distribution by attribute
                for value in self.processed_data[by_attribute].unique():
                    sns.kdeplot(
                        self.processed_data[self.processed_data[by_attribute] == value][feature],
                        ax=ax,
                        label=f'{by_attribute}={value}'
                    )
                ax.legend()
            else:
                # Plot overall distribution
                sns.histplot(self.processed_data[feature], ax=ax, kde=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature}')
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'steps': self.preprocessing_steps,
            'initial_shape': None if self.data is None else self.data.shape,
            'processed_shape': None if self.processed_data is None else self.processed_data.shape,
            'feature_columns': len(self.feature_columns),
            'categorical_columns': len(self.categorical_columns),
            'numerical_columns': len(self.numerical_columns),
            'sensitive_attributes': self.sensitive_attributes,
            'bias_metrics': self.bias_metrics
        }


