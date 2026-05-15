# Climate Forecasting RNN API

Production-ready FastAPI service for next-day temperature prediction using LSTM-RNN.

## Overview

This API provides weather forecasting capabilities using a deep learning model trained on Delhi climate data. Given 30 consecutive days of weather observations (temperature, humidity, wind speed, and atmospheric pressure), it predicts the next day's temperature.

## Features

- **LSTM-based prediction**: 96-unit, 2-layer LSTM optimized with Ray Tune
- **FastAPI backend**: High-performance async API with automatic OpenAPI docs
- **HuggingFace integration**: Model and preprocessing artifacts hosted on HF Hub
- **Production-ready**: Docker support, health checks, CORS, comprehensive error handling
- **Low latency**: ~20ms per prediction on CPU
- **Performance**: MAE 1.852°C, RMSE 2.298°C on test data

## Architecture

```
User → FastAPI (Render) → Downloads from HF Hub → LSTM Model → Temperature Prediction
         [Stateless API]      [Model + Scaler]
```

## Quick Start

### Prerequisites

- Python 3.11+
- HuggingFace account and token ([get one here](https://huggingface.co/settings/tokens))

### 1. Clone and Setup

```bash
git clone <repository-url>
cd RNN

# Copy environment template
cp .env.example .env

# Edit .env and add your HF_TOKEN
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

### 3. Train Model (Optional - or use pre-trained)

**Option A: Use the existing trained model** (skip to step 4)

**Option B: Train from scratch:**

```bash
# Install training dependencies
uv pip install -e ".[training]"

# Train model locally (saves to models/ directory)
python scripts/train_model.py

# Copy trained model to project root for deployment
cp models/climate_rnn.pt best_climate_rnn_ray.pt
```

Training takes ~5-10 minutes on CPU and produces:
- `models/climate_rnn.pt` - Trained model weights
- `models/scaler.pkl` - Fitted scaler
- `models/config.json` - Model configuration

### 4. Prepare Model for HuggingFace

```bash
# Generate scaler from training data (if not already done)
python scripts/save_scaler.py

# Package model for HuggingFace Hub
python scripts/prepare_hf_model.py

# Upload to HuggingFace (requires HF_TOKEN in environment)
export HF_TOKEN=your_token_here
python scripts/upload_to_hf.py
```

### 5. Run API Locally

```bash
uvicorn src.api.app:app --reload --port 8000
```

API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### `POST /predict`

Predict next-day temperature from 30-day sequence.

**Request:**
```json
{
  "sequence": [
    {
      "meantemp": 25.0,
      "humidity": 60.0,
      "wind_speed": 5.0,
      "meanpressure": 1010.0
    },
    // ... 29 more observations (30 total)
  ]
}
```

**Response:**
```json
{
  "predicted_temperature": 27.5,
  "unit": "celsius",
  "metadata": {
    "sequence_length": 30,
    "model_version": "1.0.0",
    "latency_ms": 15.2,
    "features": ["meantemp", "humidity", "wind_speed", "meanpressure"]
  }
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### `GET /health`

Health check for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /model/info`

Get model metadata and performance metrics.

**Response:**
```json
{
  "model_type": "ClimateRNN",
  "version": "1.0.0",
  "hyperparameters": {
    "hidden_size": 96,
    "num_layers": 2,
    "seq_length": 30
  },
  "performance": {
    "test_mae": 1.852,
    "test_rmse": 2.298
  },
  "features": {
    "input_features": ["meantemp", "humidity", "wind_speed", "meanpressure"],
    "target_feature": "meantemp"
  }
}
```

### `POST /predict/batch`

Batch prediction for multiple sequences (processes sequentially).

## Project Structure

```
climate-rnn-api/
├── src/
│   ├── model/
│   │   ├── architecture.py    # ClimateRNN LSTM model
│   │   ├── loader.py          # HuggingFace Hub integration
│   │   └── predictor.py       # Inference logic with scaling
│   ├── data/
│   │   └── schemas.py         # Pydantic models for validation
│   ├── api/
│   │   ├── app.py             # FastAPI application
│   │   └── routes.py          # API endpoints
│   ├── config/
│   │   └── settings.py        # Environment configuration
│   └── utils/
│       └── logging.py         # Structured logging
├── scripts/
│   ├── save_scaler.py         # Generate scaler from training data
│   ├── prepare_hf_model.py    # Package model for upload
│   └── upload_to_hf.py        # Upload to HuggingFace Hub
├── deployment/
│   ├── Dockerfile             # Container image
│   └── render.yaml            # Render deployment config
├── artifacts/                 # Model artifacts for HF upload
├── datasets/                  # Training data (Delhi climate)
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## Deployment to Render

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Climate RNN API"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Connect to Render

1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml`

### 3. Set Environment Variables

In Render dashboard, add:

- `HF_TOKEN`: Your HuggingFace token (get from [here](https://huggingface.co/settings/tokens))

Other variables are pre-configured in `render.yaml`.

### 4. Deploy

Render will automatically build and deploy your Docker container.

**Expected startup time**: ~30-60 seconds (includes model download from HF Hub)

## Training Your Own Model

The project includes a complete training pipeline if you want to retrain the model with different data or hyperparameters.

### Training Script

```bash
# Install training dependencies (includes pandas, matplotlib, tqdm)
uv pip install -e ".[training]"

# Run training
python scripts/train_model.py
```

The training script:
1. Loads data from `datasets/DailyDelhiClimateTrain.csv` and `datasets/DailyDelhiClimateTest.csv`
2. Splits training data into train/validation (80/20)
3. Fits MinMaxScaler on training data
4. Trains LSTM model with best hyperparameters from Ray Tune
5. Evaluates on test set
6. Saves all artifacts to `models/` directory

### Training Output

```
models/
├── climate_rnn.pt      # Trained model weights
├── best_model.pt       # Best model from validation
├── scaler.pkl          # Fitted MinMaxScaler
└── config.json         # Model configuration + metrics
```

### Custom Hyperparameters

Edit `scripts/train_model.py` to customize:
- `hidden_size` - LSTM hidden units (default: 96)
- `num_layers` - Number of LSTM layers (default: 2)
- `seq_length` - Input sequence length in days (default: 30)
- `learning_rate` - Adam learning rate (default: 0.000374)
- `epochs` - Training epochs (default: 50)
- `batch_size` - Batch size (default: 32)

### Using Your Trained Model

After training:

```bash
# Copy to project root for deployment
cp models/climate_rnn.pt best_climate_rnn_ray.pt

# Update artifacts for HuggingFace
cp models/scaler.pkl artifacts/
cp models/config.json artifacts/

# Prepare and upload to HF
python scripts/prepare_hf_model.py
python scripts/upload_to_hf.py
```

---

## Testing

Run tests:

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Model Details

**Architecture**: 2-layer LSTM with 96 hidden units

**Hyperparameters** (optimized with Ray Tune):
- Hidden size: 96
- Dropout: 0.10
- Sequence length: 30 days
- Learning rate: 0.000374

**Training Data**: Daily Delhi Climate (2013-2017)
- Training: 1,462 samples
- Test: 114 samples

**Performance**:
- Test MAE: 1.852°C
- Test RMSE: 2.298°C

## Input Requirements

The model expects exactly 30 consecutive days of observations, each with 4 features:

1. **meantemp** (°C): Mean temperature, range [-50, 60]
2. **humidity** (%): Humidity percentage, range [0, 100]
3. **wind_speed** (km/h): Wind speed, range [0, 200]
4. **meanpressure** (hPa): Atmospheric pressure, range [800, 1100]

Features must be provided in this exact order.

## Limitations

- Trained only on Delhi climate data (may not generalize to other regions)
- Requires exactly 30 consecutive days of input
- Predicts only one day ahead
- Does not account for extreme weather events or long-term climate trends

## Development

### Run with auto-reload:

```bash
uvicorn src.api.app:app --reload --port 8000
```

### Run in Docker:

```bash
# Build image
docker build -f deployment/Dockerfile -t climate-rnn-api .

# Run container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  climate-rnn-api
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_MODEL_ID` | No | `DanielTobi0/climate-rnn-model` | HuggingFace model repository |
| `HF_TOKEN` | **Yes** | - | HuggingFace API token (for private repos) |
| `MODEL_CACHE_DIR` | No | `./model_cache` | Local cache for downloaded models |
| `DEBUG` | No | `false` | Enable debug logging |
| `ALLOWED_ORIGINS` | No | `*` | CORS allowed origins |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

For issues or questions, open an issue on GitHub.

## Citation

```bibtex
@software{climate-rnn-api,
  title={Climate Forecasting RNN API},
  author={DanielTobi0},
  year={2026},
  url={https://github.com/DanielTobi0/climate-rnn-api}
}
```
