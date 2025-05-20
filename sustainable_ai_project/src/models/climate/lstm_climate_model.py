"""
Climate Model Module

This module provides an energy-efficient LSTM model for climate prediction
with sustainability constraints.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from typing import Dict, List, Union, Optional, Tuple, Any

class LSTMClimateModel:
    """
    Energy-efficient LSTM model for climate prediction with sustainability constraints.
    """
    
    def __init__(self, sequence_length: int, n_features: int, 
                 n_outputs: int = 1, energy_efficiency: float = 0.7):
        """
        Initialize the climate model with sustainability constraints.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            n_outputs: Number of output features to predict
            energy_efficiency: Energy efficiency constraint (0-1, higher means more efficient)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.energy_efficiency = energy_efficiency
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model with energy efficiency constraints.
        
        Returns:
            TensorFlow LSTM model
        """
        # Scale model complexity based on energy efficiency
        # Higher energy efficiency means simpler model with fewer parameters
        base_units = 64
        lstm_units = int(base_units * (1 - 0.5 * self.energy_efficiency))
        dense_units = int(base_units * (1 - 0.5 * self.energy_efficiency))
        
        # Create model with appropriate complexity
        model = models.Sequential([
            layers.LSTM(lstm_units, 
                      input_shape=(self.sequence_length, self.n_features),
                      return_sequences=True, 
                      activation='tanh',
                      recurrent_dropout=0.0),  # Avoid recurrent dropout for better hardware acceleration
            layers.Dropout(0.2),
            layers.LSTM(lstm_units // 2, 
                      activation='tanh',
                      recurrent_dropout=0.0),
            layers.Dropout(0.2),
            layers.Dense(dense_units, activation='relu'),
            layers.Dense(self.n_outputs)
        ])
        
        # Use efficient optimizer with mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             batch_size: int = 32, epochs: int = 100,
             patience: int = 10, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model with energy-efficient settings.
        
        Args:
            X_train: Training features (shape: [samples, sequence_length, features])
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Dictionary with training history and metrics
        """
        # Set up callbacks for efficient training
        callbacks_list = [
            # Early stopping to prevent unnecessary computation
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            # Reduce learning rate when plateau is reached
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        # Get training metrics
        train_metrics = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_used': len(history.history['loss']),
            'early_stopping_occurred': len(history.history['loss']) < epochs
        }
        
        # Calculate model complexity metrics
        model_complexity = self._calculate_model_complexity()
        
        # Combine metrics
        results = {
            'training_history': history.history,
            'training_metrics': train_metrics,
            'model_complexity': model_complexity
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features (shape: [samples, sequence_length, features])
            
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
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared
        }
        
        return metrics
    
    def _calculate_model_complexity(self) -> Dict[str, Any]:
        """
        Calculate model complexity metrics.
        
        Returns:
            Dictionary with model complexity metrics
        """
        # Get total number of parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Count layers by type
        layer_counts = {}
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            if layer_type in layer_counts:
                layer_counts[layer_type] += 1
            else:
                layer_counts[layer_type] = 1
        
        # Calculate memory footprint (rough estimate)
        # Assuming 4 bytes per parameter (float32)
        memory_footprint_bytes = total_params * 4
        memory_footprint_mb = memory_footprint_bytes / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'layer_counts': layer_counts,
            'memory_footprint_mb': memory_footprint_mb
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information for sustainability evaluation.
        
        Returns:
            Dictionary with model information
        """
        # Get model complexity
        model_complexity = self._calculate_model_complexity()
        
        # Get model architecture summary
        model_summary = []
        self.model.summary(print_fn=lambda x: model_summary.append(x))
        
        # Combine information
        model_info = {
            'model_type': 'lstm_climate_model',
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_outputs': self.n_outputs,
            'energy_efficiency_constraint': self.energy_efficiency,
            'complexity': model_complexity,
            'architecture_summary': '\n'.join(model_summary),
            'num_parameters': model_complexity['total_parameters'],
            'num_layers': len(self.model.layers)
        }
        
        return model_info
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str, sequence_length: int, n_features: int, 
                 n_outputs: int = 1, energy_efficiency: float = 0.7) -> 'LSTMClimateModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            sequence_length: Length of input sequences
            n_features: Number of input features
            n_outputs: Number of output features
            energy_efficiency: Energy efficiency constraint
            
        Returns:
            Loaded LSTMClimateModel instance
        """
        # Create instance
        instance = cls(sequence_length, n_features, n_outputs, energy_efficiency)
        
        # Load model weights
        instance.model = tf.keras.models.load_model(filepath)
        
        return instance
