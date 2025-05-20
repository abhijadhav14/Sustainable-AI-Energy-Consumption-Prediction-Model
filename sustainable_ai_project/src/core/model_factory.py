"""
Model Factory Module

This module provides a factory for creating AI models with sustainability constraints.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Type

# Import model classes
from src.models.climate.lstm_climate_model import LSTMClimateModel
from src.models.energy.gbm_energy_forecaster import GBMEnergyForecaster
from src.models.healthcare.fairness_aware_classifier import FairnessAwareClassifier

class ModelFactory:
    """
    Factory for creating AI models with sustainability constraints.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        # Register available model types
        self.model_registry = {
            'lstm_climate_model': LSTMClimateModel,
            'gbm_energy_forecaster': GBMEnergyForecaster,
            'fairness_aware_classifier': FairnessAwareClassifier
        }
    
    def create_model(self, model_type: str, config: Dict[str, Any]) -> Any:
        """
        Create a model of the specified type with the given configuration.
        
        Args:
            model_type: Type of model to create
            config: Configuration parameters for the model
            
        Returns:
            Model instance
        """
        # Check if model type is registered
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self.model_registry.keys())}")
        
        # Get model class
        model_class = self.model_registry[model_type]
        
        # Create model instance
        try:
            model = self._create_model_instance(model_class, config)
        except Exception as e:
            raise ValueError(f"Failed to create model of type {model_type}: {str(e)}")
        
        return model
    
    def _create_model_instance(self, model_class: Type, config: Dict[str, Any]) -> Any:
        """
        Create a model instance of the specified class with the given configuration.
        
        Args:
            model_class: Model class
            config: Configuration parameters for the model
            
        Returns:
            Model instance
        """
        # Create model instance based on model class
        if model_class == LSTMClimateModel:
            return self._create_lstm_climate_model(config)
        elif model_class == GBMEnergyForecaster:
            return self._create_gbm_energy_forecaster(config)
        elif model_class == FairnessAwareClassifier:
            return self._create_fairness_aware_classifier(config)
        else:
            raise ValueError(f"Unsupported model class: {model_class.__name__}")
    
    def _create_lstm_climate_model(self, config: Dict[str, Any]) -> LSTMClimateModel:
        """
        Create an LSTM climate model.
        
        Args:
            config: Configuration parameters for the model
            
        Returns:
            LSTMClimateModel instance
        """
        # Extract required parameters
        input_dim = config.get('input_dim')
        if input_dim is None:
            raise ValueError("input_dim is required for LSTM climate model")
        
        # Extract optional parameters with defaults
        sequence_length = config.get('sequence_length', 24)
        hidden_units = config.get('hidden_units', [64, 32])
        dropout_rate = config.get('dropout_rate', 0.2)
        energy_efficiency_constraint = config.get('energy_efficiency_constraint', 0.5)
        
        # Create model
        model = LSTMClimateModel(
            input_dim=input_dim,
            sequence_length=sequence_length,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            energy_efficiency_constraint=energy_efficiency_constraint
        )
        
        return model
    
    def _create_gbm_energy_forecaster(self, config: Dict[str, Any]) -> GBMEnergyForecaster:
        """
        Create a GBM energy forecaster.
        
        Args:
            config: Configuration parameters for the model
            
        Returns:
            GBMEnergyForecaster instance
        """
        # Extract required parameters
        n_features = config.get('n_features')
        if n_features is None:
            raise ValueError("n_features is required for GBM energy forecaster")
        
        # Extract optional parameters with defaults
        interpretability_level = config.get('interpretability_level', 0.5)
        
        # Create model
        model = GBMEnergyForecaster(
            n_features=n_features,
            interpretability_level=interpretability_level
        )
        
        return model
    
    def _create_fairness_aware_classifier(self, config: Dict[str, Any]) -> FairnessAwareClassifier:
        """
        Create a fairness-aware classifier.
        
        Args:
            config: Configuration parameters for the model
            
        Returns:
            FairnessAwareClassifier instance
        """
        # Extract required parameters
        n_features = config.get('n_features')
        if n_features is None:
            raise ValueError("n_features is required for fairness-aware classifier")
        
        # Extract optional parameters with defaults
        n_classes = config.get('n_classes', 2)
        fairness_constraint = config.get('fairness_constraint', 0.7)
        interpretability_level = config.get('interpretability_level', 0.5)
        
        # Create model
        model = FairnessAwareClassifier(
            n_features=n_features,
            n_classes=n_classes,
            fairness_constraint=fairness_constraint,
            interpretability_level=interpretability_level
        )
        
        return model
    
    def get_available_model_types(self) -> List[str]:
        """
        Get a list of available model types.
        
        Returns:
            List of available model types
        """
        return list(self.model_registry.keys())
    
    def get_model_config_schema(self, model_type: str) -> Dict[str, Any]:
        """
        Get the configuration schema for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Configuration schema
        """
        # Check if model type is registered
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self.model_registry.keys())}")
        
        # Get configuration schema based on model type
        if model_type == 'lstm_climate_model':
            return {
                'input_dim': {
                    'type': 'integer',
                    'required': True,
                    'description': 'Number of input features'
                },
                'sequence_length': {
                    'type': 'integer',
                    'required': False,
                    'default': 24,
                    'description': 'Length of input sequences'
                },
                'hidden_units': {
                    'type': 'array',
                    'required': False,
                    'default': [64, 32],
                    'description': 'Number of hidden units in each LSTM layer'
                },
                'dropout_rate': {
                    'type': 'number',
                    'required': False,
                    'default': 0.2,
                    'description': 'Dropout rate for regularization'
                },
                'energy_efficiency_constraint': {
                    'type': 'number',
                    'required': False,
                    'default': 0.5,
                    'description': 'Energy efficiency constraint (0-1, higher means more efficient)'
                }
            }
        elif model_type == 'gbm_energy_forecaster':
            return {
                'n_features': {
                    'type': 'integer',
                    'required': True,
                    'description': 'Number of input features'
                },
                'interpretability_level': {
                    'type': 'number',
                    'required': False,
                    'default': 0.5,
                    'description': 'Interpretability constraint (0-1, higher means more interpretable)'
                }
            }
        elif model_type == 'fairness_aware_classifier':
            return {
                'n_features': {
                    'type': 'integer',
                    'required': True,
                    'description': 'Number of input features'
                },
                'n_classes': {
                    'type': 'integer',
                    'required': False,
                    'default': 2,
                    'description': 'Number of output classes'
                },
                'fairness_constraint': {
                    'type': 'number',
                    'required': False,
                    'default': 0.7,
                    'description': 'Fairness constraint (0-1, higher means more fair)'
                },
                'interpretability_level': {
                    'type': 'number',
                    'required': False,
                    'default': 0.5,
                    'description': 'Interpretability constraint (0-1, higher means more interpretable)'
                }
            }
        else:
            return {}
    
    def register_model_type(self, model_type: str, model_class: Type) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Type name for the model
            model_class: Model class
        """
        self.model_registry[model_type] = model_class


