"""
ClimateRNN Model Architecture

LSTM-based time series forecasting model for climate prediction.
Extracted from the Ray-tuned notebook implementation.
"""

import torch
import torch.nn as nn


class ClimateRNN(nn.Module):
    """
    LSTM-based RNN for climate time series forecasting.

    The model takes a sequence of climate observations (temperature, humidity,
    wind speed, pressure) and predicts the next day's temperature.

    Args:
        input_size: Number of input features (4 for climate data)
        hidden_size: Number of LSTM hidden units (best: 96)
        num_layers: Number of LSTM layers (best: 2)
        dropout: Dropout probability between LSTM layers (best: ~0.1)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Predicted values of shape (batch_size,)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        prediction = self.fc(out[:, -1, :])

        return prediction.squeeze(-1)

    def __repr__(self) -> str:
        return (
            f'ClimateRNN('
            f'input_size={self.input_size}, '
            f'hidden_size={self.hidden_size}, '
            f'num_layers={self.num_layers}, '
            f'parameters={sum(p.numel() for p in self.parameters()):,})'
        )
