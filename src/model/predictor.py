"""
Climate Predictor

Handles inference logic with proper input scaling and output transformations.
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.model.architecture import ClimateRNN


class ClimatePredictor:
    """
    Inference wrapper for the climate forecasting model.

    Handles:
    - Input validation and scaling
    - Model inference
    - Output inverse transformation to Celsius
    """

    def __init__(self, model: ClimateRNN, scaler: MinMaxScaler, config: Dict):
        """
        Initialize the predictor.

        Args:
            model: Trained ClimateRNN model
            scaler: Fitted MinMaxScaler (must match training preprocessing)
            config: Model configuration dict with hyperparameters and feature info
        """
        self.model = model
        self.scaler = scaler
        self.config = config
        self.device = next(model.parameters()).device

        self.features = config['features']
        self.target_idx = self.features['target_idx']
        self.seq_length = config['hyperparameters']['seq_length']
        self.input_size = config['hyperparameters']['input_size']

        print(f'Predictor initialized with target_idx={self.target_idx}, seq_length={self.seq_length}')

    def predict(self, sequence: List[Dict[str, float]]) -> float:
        """
        Predict next-day temperature from a 30-day climate sequence.

        Args:
            sequence: List of 30 dicts, each with keys: meantemp, humidity, wind_speed, meanpressure
                     Example: [{"meantemp": 25.0, "humidity": 60.0, ...}, ...]

        Returns:
            Predicted temperature in Celsius

        Raises:
            ValueError: If sequence length is incorrect or features are missing
        """
        # Validate sequence length
        if len(sequence) != self.seq_length:
            raise ValueError(f'Expected sequence of length {self.seq_length}, got {len(sequence)}')

        feature_names = self.features['input_features']
        try:
            sequence_array = np.array([[obs[feat] for feat in feature_names] for obs in sequence], dtype=np.float32)
        except KeyError as e:
            raise ValueError(f'Missing required feature: {e}')

        # Validate shape
        if sequence_array.shape != (self.seq_length, self.input_size):
            raise ValueError(f'Expected shape ({self.seq_length}, {self.input_size}), got {sequence_array.shape}')

        sequence_scaled = self.scaler.transform(sequence_array)

        x = torch.tensor(sequence_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)  # Shape: (1, 30, 4)

        with torch.inference_mode():
            prediction_scaled = self.model(x).cpu().numpy()[0]  # Get scalar value

        # Inverse transform ONLY the target feature (temperature)
        # Create dummy array with all features, but only the temperature column matters
        temperature_celsius = self._inverse_transform_target(prediction_scaled)

        return float(temperature_celsius)

    def _inverse_transform_target(self, scaled_value: float) -> float:
        """
        Inverse transform a single scaled target value back to Celsius.

        The scaler was fit on all 4 features, but we only want to inverse
        transform the target column (meantemp at index 0).

        Args:
            scaled_value: Scaled prediction value

        Returns:
            Temperature in Celsius
        """
        # Create a dummy array with the scaled value in the target position
        dummy = np.zeros((1, self.input_size), dtype=np.float32)
        dummy[0, self.target_idx] = scaled_value

        inverse = self.scaler.inverse_transform(dummy)
        return inverse[0, self.target_idx]

    def predict_batch(self, sequences: List[List[Dict[str, float]]]) -> List[float]:
        """
        Batch prediction for multiple sequences.

        Args:
            sequences: List of sequences, each with 30 observations

        Returns:
            List of predicted temperatures in Celsius
        """
        return [self.predict(seq) for seq in sequences]

    def get_expected_features(self) -> List[str]:
        """Return the list of required features in order."""
        return self.features['input_features']
