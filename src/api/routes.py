"""
API Routes

Defines all API endpoints for the climate forecasting service.
"""

import time
from typing import List

from fastapi import APIRouter, HTTPException, Request

from src.data.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse

router = APIRouter(tags=['Climate Forecasting'])


@router.get('/health', response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns service status and whether the model is loaded.
    Used by Render and other monitoring services.
    """
    model_loaded = getattr(request.app.state, 'model_loaded', False)

    return HealthResponse(status='healthy' if model_loaded else 'degraded', model_loaded=model_loaded)


@router.get('/model/info', response_model=ModelInfoResponse)
async def model_info(request: Request):
    """
    Get model metadata and performance information.

    Returns:
        Model type, version, hyperparameters, performance metrics, and features
    """
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail='Model not loaded')

    loader = request.app.state.loader
    info = loader.get_model_info()

    return ModelInfoResponse(**info)


@router.post('/predict', response_model=PredictionResponse)
async def predict_temperature(request_data: PredictionRequest, request: Request):
    """
    Predict next-day temperature from a 30-day climate sequence.

    Args:
        request_data: 30-day sequence of climate observations

    Returns:
        Predicted temperature in Celsius with metadata

    Raises:
        503: If model is not loaded
        422: If input validation fails
        500: If prediction fails
    """
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail='Model not loaded. Service may be starting up.')

    predictor = request.app.state.predictor

    try:
        # Convert Pydantic models to dict for predictor
        sequence = [obs.model_dump() for obs in request_data.sequence]

        # Time the prediction
        start_time = time.time()
        predicted_temp = predictor.predict(sequence)
        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            predicted_temperature=round(predicted_temp, 2),
            unit='celsius',
            metadata={
                'sequence_length': len(sequence),
                'model_version': '1.0.0',
                'latency_ms': round(latency_ms, 2),
                'features': predictor.get_expected_features(),
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f'Invalid input: {str(e)}')
    except Exception as e:
        print(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail='Prediction failed. Check server logs.')


@router.post('/predict/batch', response_model=List[PredictionResponse])
async def predict_temperature_batch(requests: List[PredictionRequest], request: Request):
    """
    Batch prediction endpoint for multiple sequences.

    Args:
        requests: List of prediction requests

    Returns:
        List of prediction responses

    Note: This endpoint processes requests sequentially.
    For truly parallel processing, use multiple /predict calls.
    """
    if not request.app.state.model_loaded:
        raise HTTPException(status_code=503, detail='Model not loaded')

    results = []
    for req_data in requests:
        # Reuse the single prediction endpoint logic
        response = await predict_temperature(req_data, request)
        results.append(response)

    return results
