"""
Save fitted MinMaxScaler from training data.

This script refits the scaler on the training data to ensure
it matches the preprocessing used during model training.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / 'datasets' / 'DailyDelhiClimateTrain.csv'
OUTPUT_PATH = PROJECT_ROOT / 'artifacts' / 'scaler.pkl'
FEATURES = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']


def main():
    print(f'Loading training data from {TRAIN_DATA_PATH}...')
    train_df = pd.read_csv(TRAIN_DATA_PATH, parse_dates=['date'])
    train_df = train_df.sort_values('date').reset_index(drop=True)

    print(f'Training data shape: {train_df.shape}')
    print(f'Features: {FEATURES}')

    print('\nFitting MinMaxScaler on training data...')
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES])

    print(f'Scaler fitted with data range:')
    for i, feature in enumerate(FEATURES):
        print(f'  {feature}: [{scaler.data_min_[i]:.2f}, {scaler.data_max_[i]:.2f}]')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'\n✓ Scaler saved to {OUTPUT_PATH}')
    print(f'  File size: {OUTPUT_PATH.stat().st_size:,} bytes')

    with open(OUTPUT_PATH, 'rb') as f:
        loaded_scaler = pickle.load(f)
    assert np.allclose(loaded_scaler.data_min_, scaler.data_min_)
    print('✓ Verification: Scaler loads correctly')


if __name__ == '__main__':
    main()
