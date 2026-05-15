"""
Prepare Model for HuggingFace Hub

Packages model weights, scaler, and config into the artifacts directory
ready for upload to HuggingFace Hub.
"""

import json
import shutil
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    artifacts_dir = project_root / 'artifacts'
    model_file = project_root / 'best_climate_rnn_ray.pt'

    print('Preparing model for HuggingFace Hub upload...')
    print(f'Project root: {project_root}')
    print(f'Artifacts directory: {artifacts_dir}')

    # Check required files exist
    required_files = {
        'Model weights': model_file,
        'Scaler': artifacts_dir / 'scaler.pkl',
        'Config': artifacts_dir / 'config.json',
    }

    missing = []
    for name, path in required_files.items():
        if path.exists():
            print(f'✓ {name}: {path}')
        else:
            print(f'✗ {name}: {path} (MISSING)')
            missing.append(name)

    if missing:
        print(f'\nMissing required files: {", ".join(missing)}')
        print('\nRun these scripts first:')
        print('  python scripts/save_scaler.py')
        return 1

    # Copy model weights with HuggingFace naming convention
    hf_model_file = artifacts_dir / 'pytorch_model.bin'
    print(f'\nCopying {model_file.name} → {hf_model_file.name}...')
    shutil.copy2(model_file, hf_model_file)
    print(f'✓ Model copied ({hf_model_file.stat().st_size:,} bytes)')

    # Create README.md (model card)
    readme_path = artifacts_dir / 'README.md'
    print(f'\nGenerating {readme_path.name}...')

    with open(artifacts_dir / 'config.json', 'r') as f:
        config = json.load(f)

    readme_content = f"""---
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
**Framework**: PyTorch {config['metadata']['pytorch_version']}
**Input**: 30-day sequence of 4 climate features
**Output**: Next-day temperature in Celsius

## Performance

Evaluated on held-out test data (114 samples):

- **MAE**: {config['performance']['test_mae']:.3f}°C
- **RMSE**: {config['performance']['test_rmse']:.3f}°C
- **Validation MSE**: {config['performance']['validation_mse']:.6f}

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
print(f"Predicted temperature: {{temperature:.2f}}°C")
```

## Training Data

**Dataset**: Daily Delhi Climate (2013-2017)
**Training samples**: {config['training']['training_samples']:,}
**Test samples**: {config['training']['test_samples']}
**Features**: {', '.join(config['features']['input_features'])}

## Hyperparameters

Optimized using Ray Tune with ASHA scheduler:

- Hidden size: {config['hyperparameters']['hidden_size']}
- Num layers: {config['hyperparameters']['num_layers']}
- Dropout: {config['hyperparameters']['dropout']:.4f}
- Sequence length: {config['hyperparameters']['seq_length']} days
- Learning rate: {config['hyperparameters']['learning_rate']:.6f}
- Batch size: {config['hyperparameters']['batch_size']}

## Limitations

- Trained only on Delhi climate data (may not generalize to other regions)
- Requires exactly 30 consecutive days of input
- Predicts only one day ahead
- Does not account for extreme weather events or climate change trends

## License

MIT License
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f'✓ README.md generated ({readme_path.stat().st_size:,} bytes)')

    print('\n' + '=' * 60)
    print('Model preparation complete!')
    print('=' * 60)
    print('\nFiles ready for upload:')
    for file in sorted(artifacts_dir.glob('*')):
        if file.is_file():
            print(f'  - {file.name} ({file.stat().st_size:,} bytes)')

    print('\nNext step: Run upload_to_hf.py to push to HuggingFace Hub')
    return 0


if __name__ == '__main__':
    exit(main())
