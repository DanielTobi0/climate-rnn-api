"""
Model Loader for HuggingFace Hub Integration

Handles downloading, caching, and initializing the climate model
and its preprocessing artifacts from HuggingFace Hub.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import MinMaxScaler

from src.model.architecture import ClimateRNN


class ModelLoader:
    """
    Manages model loading from HuggingFace Hub.

    Downloads and caches:
    - pytorch_model.bin: Model weights
    - scaler.pkl: Fitted MinMaxScaler
    - config.json: Model configuration and metadata
    """

    def __init__(self, model_id: str, token: str | None = None, cache_dir: str | Path = './model_cache'):
        """
        Initialize the model loader.

        Args:
            model_id: HuggingFace repository ID (e.g., "username/model-name")
            token: HuggingFace API token (required for private repos)
            cache_dir: Local directory for caching downloaded files
        """
        self.model_id = model_id
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model: ClimateRNN | None = None
        self.scaler: MinMaxScaler | None = None
        self.config: Dict[str, Any] | None = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_all(self) -> Tuple[ClimateRNN, MinMaxScaler, Dict[str, Any]]:
        """
        Load model, scaler, and config from HuggingFace Hub.

        Returns:
            Tuple of (model, scaler, config)

        Raises:
            Exception: If any required file fails to download or load
        """
        print(f'Loading model from HuggingFace: {self.model_id}')
        print(f'Using device: {self.device}')

        model_path = self._download_file('pytorch_model.bin')
        scaler_path = self._download_file('scaler.pkl')
        config_path = self._download_file('config.json')

        print('Loading configuration...')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        print('Loading scaler...')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print('Initializing model architecture...')
        hyperparams = self.config['hyperparameters']
        self.model = ClimateRNN(
            input_size=hyperparams['input_size'],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
        )

        print('Loading model weights...')
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f'✓ Model loaded successfully: {self.model}')
        print(f'✓ Performance: MAE={self.config["performance"]["test_mae"]:.3f}°C, RMSE={self.config["performance"]["test_rmse"]:.3f}°C')

        return self.model, self.scaler, self.config

    def _download_file(self, filename: str) -> Path:
        """
        Download a file from HuggingFace Hub with caching.

        Args:
            filename: Name of the file to download

        Returns:
            Path to the downloaded/cached file

        Raises:
            Exception: If download fails
        """
        try:
            file_path = hf_hub_download(
                repo_id=self.model_id, filename=filename, token=self.token, cache_dir=str(self.cache_dir), force_download=False  # Use cache if available
            )
            print(f'  ✓ {filename} (cached: {Path(file_path).exists()})')
            return Path(file_path)
        except Exception as e:
            raise Exception(f'Failed to download {filename} from {self.model_id}: {e}')

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata for the /model/info endpoint.

        Returns:
            Dictionary with model type, version, hyperparameters, performance, features
        """
        if self.config is None:
            raise RuntimeError('Model not loaded. Call load_all() first.')

        return {
            'model_type': self.config['model_type'],
            'version': self.config['version'],
            'hyperparameters': self.config['hyperparameters'],
            'performance': self.config['performance'],
            'features': self.config['features'],
        }
