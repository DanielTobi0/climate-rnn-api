---
language: en
license: mit
tags:
- climate
- time-series
- lstm
- temperature-forecasting
- pytorch
datasets:
- delhi-climate
metrics:
- mae
- rmse
---

# Climate Forecasting RNN

LSTM-based time series forecasting model for next-day temperature prediction.

## Model Description

This model predicts the next day's temperature based on 30 days of historical climate data (temperature, humidity, wind speed, and atmospheric pressure). It was trained on the Daily Delhi Climate dataset using hyperparameter optimization with Ray Tune.

**Architecture**: LSTM (2 layers, 96 hidden units)
**Framework**: PyTorch 2.12.0+cu130
**Input**: 30-day sequence of 4 climate features
**Output**: Next-day temperature in Celsius

## Performance

Evaluated on held-out test data (114 samples):

- **MAE**: 1.930°C
- **RMSE**: 2.422°C
- **Validation MSE**: 0.007749

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
import pickle
import json
import numpy as np

# Download model files
model_path = hf_hub_download("DanielTobi0/climate-rnn-model", "pytorch_model.bin")
scaler_path = hf_hub_download("DanielTobi0/climate-rnn-model", "scaler.pkl")
config_path = hf_hub_download("DanielTobi0/climate-rnn-model", "config.json")

# Load model architecture (you need the ClimateRNN class)
with open(config_path, 'r') as f:
    config = json.load(f)

from src.model.architecture import ClimateRNN

model = ClimateRNN(
    input_size=config['hyperparameters']['input_size'],
    hidden_size=config['hyperparameters']['hidden_size'],
    num_layers=config['hyperparameters']['num_layers'],
    dropout=config['hyperparameters']['dropout']
)

# Load weights
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Prepare input (30-day sequence)
sequence = [
    [25.0, 60.0, 5.0, 1010.0],  # Day 1: [temp, humidity, wind_speed, pressure]
    [26.0, 58.0, 6.0, 1012.0],  # Day 2
    # ... 28 more days
]
sequence_scaled = scaler.transform(sequence)
x = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

# Predict
with torch.inference_mode():
    prediction_scaled = model(x).item()

# Inverse transform (temperature is at index 0)
dummy = np.zeros((1, 4))
dummy[0, 0] = prediction_scaled
temperature = scaler.inverse_transform(dummy)[0, 0]
print(f"Predicted temperature: {temperature:.2f}°C")
```

## Training Data

**Dataset**: Daily Delhi Climate (2013-2017)
**Training samples**: 1,170
**Test samples**: 114
**Features**: meantemp, humidity, wind_speed, meanpressure

## Hyperparameters

Optimized using Ray Tune with ASHA scheduler:

- Hidden size: 96
- Num layers: 2
- Dropout: 0.1003
- Sequence length: 30 days
- Learning rate: 0.000374
- Batch size: 32

## Limitations

- Trained only on Delhi climate data (may not generalize to other regions)
- Requires exactly 30 consecutive days of input
- Predicts only one day ahead
- Does not account for extreme weather events or climate change trends

## License

MIT License
