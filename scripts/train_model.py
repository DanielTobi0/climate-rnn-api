"""
Train Climate RNN Model Locally

Trains the LSTM model on Delhi climate data and saves all artifacts locally.
Optionally accepts custom hyperparameters or uses the best config from Ray Tune.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Ensure `src` imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.architecture import ClimateRNN


class ClimateDataset(Dataset):
    """PyTorch dataset for climate time series."""

    def __init__(self, data: np.ndarray, seq_len: int, target_idx: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, self.target_idx]
        return x, y


def train_epoch(model, loader, criterion, optimizer, device, grad_clip_max_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item()
    return total_loss / len(loader)


def inverse_transform_target(values, scaler, target_idx, n_features):
    """Inverse transform only the target feature."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values
    return scaler.inverse_transform(dummy)[:, target_idx]


def main():
    print('=' * 60)
    print('Climate RNN Training Pipeline')
    print('=' * 60)

    # Configuration
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'datasets'
    output_dir = project_root / 'models'
    output_dir.mkdir(exist_ok=True)

    # Hyperparameters (best from Ray Tune)
    config = {
        'input_size': 4,
        'hidden_size': 96,
        'num_layers': 2,
        'dropout': 0.10027879545211107,
        'seq_length': 30,
        'learning_rate': 0.00037446563861783026,
        'batch_size': 32,
        'grad_clip_max_norm': 1.0,
        'epochs': 50,
        'val_ratio': 0.2,
    }

    features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    target = 'meantemp'
    target_idx = features.index(target)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nDevice: {device}')
    print(f'Config: {json.dumps({k: v for k, v in config.items()}, indent=2)}')

    print(f'\nLoading data from {data_dir}...')
    train_df = pd.read_csv(data_dir / 'DailyDelhiClimateTrain.csv', parse_dates=['date'])
    test_df = pd.read_csv(data_dir / 'DailyDelhiClimateTest.csv', parse_dates=['date'])

    train_df = train_df.sort_values('date').reset_index(drop=True)
    test_df = test_df.sort_values('date').reset_index(drop=True)

    print(f'Train samples: {len(train_df):,}')
    print(f'Test samples: {len(test_df):,}')

    print("\nSplitting train into train/val")
    val_size = int(len(train_df) * config['val_ratio'])
    train_split_df = train_df.iloc[:-val_size].reset_index(drop=True)
    val_df = train_df.iloc[-val_size:].reset_index(drop=True)

    print(f'Split: train={len(train_split_df):,}, val={len(val_df):,}')

    print('\nFitting MinMaxScaler...')
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_split_df[features])
    val_scaled = scaler.transform(val_df[features])

    # For test set, prepend context from end of training data
    context = train_df.iloc[-(config['seq_length'] + 1) : -1][features].values
    test_raw = test_df[features].values
    test_scaled = scaler.transform(np.vstack([context, test_raw]))

    print(f'Scaler fitted: {len(features)} features')
    for i, feat in enumerate(features):
        print(f'  {feat}: [{scaler.data_min_[i]:.2f}, {scaler.data_max_[i]:.2f}]')

    print('\nCreating data loaders...')
    train_dataset = ClimateDataset(train_scaled, config['seq_length'], target_idx)
    val_dataset = ClimateDataset(val_scaled, config['seq_length'], target_idx)
    test_dataset = ClimateDataset(test_scaled, config['seq_length'], target_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')

    print('\nInitializing model...')

    model = ClimateRNN(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    ).to(device)

    print(f'Model: {model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print('\n' + '=' * 60)
    print('Training')
    print('=' * 60)

    best_val_loss = float('inf')
    best_model_path = output_dir / 'best_model.pt'
    history = {'train': [], 'val': []}

    for epoch in tqdm(range(config['epochs']), desc='Training'):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config['grad_clip_max_norm'])
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

    print(f'\n✓ Best validation MSE: {best_val_loss:.6f}')

    # Load best model and evaluate on test set
    print('\nEvaluating on test set...')
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    preds, actuals = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).cpu().numpy())
            actuals.extend(y.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Inverse transform to Celsius
    preds_celsius = inverse_transform_target(preds, scaler, target_idx, len(features))
    actuals_celsius = inverse_transform_target(actuals, scaler, target_idx, len(features))

    mae = np.mean(np.abs(preds_celsius - actuals_celsius))
    rmse = np.sqrt(np.mean((preds_celsius - actuals_celsius) ** 2))
    test_mse = np.mean((preds - actuals) ** 2)

    print(f'Test MSE (scaled): {test_mse:.6f}')
    print(f'Test MAE: {mae:.3f}°C')
    print(f'Test RMSE: {rmse:.3f}°C')

    print('\n' + '=' * 60)
    print('Saving Artifacts')
    print('=' * 60)

    # Save scaler
    import pickle

    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'✓ Scaler: {scaler_path}')

    final_model_path = output_dir / 'climate_rnn.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'✓ Model: {final_model_path}')
    print(f'  (Best model also at: {best_model_path})')

    # Save config with results
    config_data = {
        'model_type': 'ClimateRNN',
        'architecture': 'LSTM',
        'framework': 'PyTorch',
        'version': '1.0.0',
        'hyperparameters': {k: v for k, v in config.items() if k not in ['epochs', 'val_ratio']},
        'features': {
            'input_features': features,
            'target_feature': target,
            'target_idx': target_idx,
            'feature_order': f'Features must be provided in exact order: {", ".join(features)}',
        },
        'preprocessing': {'scaler': 'MinMaxScaler', 'fit_on': 'training_data', 'scaler_file': 'scaler.pkl'},
        'performance': {'test_mae': round(mae, 3), 'test_rmse': round(rmse, 3), 'validation_mse': round(best_val_loss, 6), 'unit': 'celsius'},
        'training': {
            'dataset': 'Daily Delhi Climate',
            'training_samples': len(train_split_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'training_date_range': f'{train_df["date"].min()} to {train_df["date"].max()}',
            'test_date_range': f'{test_df["date"].min()} to {test_df["date"].max()}',
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'epochs': config['epochs'],
        },
        'inference': {'input_format': '30-day sequence of 4 climate features', 'output_format': 'Next-day temperature prediction in Celsius', 'device': 'cpu', 'expected_latency_ms': 20},
        'metadata': {'created_at': '2026-05-15', 'pytorch_version': torch.__version__, 'python_version': '3.11+', 'license': 'MIT'},
    }

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)
    print(f'✓ Config: {config_path}')

    # Summary
    print('\n' + '=' * 60)
    print('Training Complete!')
    print('=' * 60)
    print(f'\nArtifacts saved in: {output_dir}/')
    print(f'  - climate_rnn.pt ({final_model_path.stat().st_size:,} bytes)')
    print(f'  - best_model.pt ({best_model_path.stat().st_size:,} bytes)')
    print(f'  - scaler.pkl ({scaler_path.stat().st_size:,} bytes)')
    print(f'  - config.json ({config_path.stat().st_size:,} bytes)')
    print(f'\nTo deploy this model:')
    print(f'1. Copy files to artifacts/ directory:')
    print(f'   cp {output_dir}/climate_rnn.pt {project_root}/best_climate_rnn_ray.pt')
    print(f'   cp {output_dir}/scaler.pkl {project_root}/artifacts/')
    print(f'   cp {output_dir}/config.json {project_root}/artifacts/')
    print(f'2. Run: python scripts/prepare_hf_model.py')
    print(f'3. Run: python scripts/upload_to_hf.py')


if __name__ == '__main__':
    main()
