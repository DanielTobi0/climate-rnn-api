"""
Pydantic Models for API Request/Response Validation

Defines the data structures for climate prediction API endpoints.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class ClimateObservation(BaseModel):
    """
    A single climate observation with 4 features.

    All values must be within physically reasonable ranges to prevent
    invalid model inputs.
    """

    meantemp: float = Field(
        ...,
        ge=-50,
        le=60,
        description='Mean temperature in Celsius',
        json_schema_extra={'example': 25.0},
    )
    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description='Humidity percentage',
        json_schema_extra={'example': 60.0},
    )
    wind_speed: float = Field(
        ...,
        ge=0,
        le=200,
        description='Wind speed in km/h',
        json_schema_extra={'example': 5.0},
    )
    meanpressure: float = Field(
        ...,
        ge=800,
        le=1100,
        description='Mean atmospheric pressure in hPa',
        json_schema_extra={'example': 1010.0},
    )

    model_config = {'json_schema_extra': {'example': {'meantemp': 25.0, 'humidity': 60.0, 'wind_speed': 5.0, 'meanpressure': 1010.0}}}


class PredictionRequest(BaseModel):
    """
    Request body for temperature prediction.

    Requires exactly 30 consecutive days of climate observations.
    The model will predict the temperature for day 31.
    """

    sequence: List[ClimateObservation] = Field(
        ...,
        min_length=30,
        max_length=30,
        description='30-day sequence of climate observations (most recent last)',
    )

    @field_validator('sequence')
    @classmethod
    def validate_sequence_length(cls, v: List[ClimateObservation]) -> List[ClimateObservation]:
        if len(v) != 30:
            raise ValueError(f'Sequence must contain exactly 30 observations, got {len(v)}')
        return v

    model_config = {
        'json_schema_extra': {
            'example': {
                'sequence': [
                    {'meantemp': 10.0, 'humidity': 70.0, 'wind_speed': 10.0, 'meanpressure': 1015.0},
                    {'meantemp': 12.0, 'humidity': 65.0, 'wind_speed': 8.0, 'meanpressure': 1012.0},
                    # ... 28 more observations
                ]
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Response body for temperature prediction.

    Returns the predicted temperature and metadata about the prediction.
    """

    predicted_temperature: float = Field(..., description='Predicted next-day temperature in Celsius')

    unit: str = Field(default='celsius', description='Temperature unit')

    metadata: dict = Field(
        default_factory=dict,
        description='Additional metadata about the prediction',
        json_schema_extra={'example': {'sequence_length': 30, 'model_version': '1.0.0', 'latency_ms': 15}},
    )

    model_config = {
        'json_schema_extra': {'example': {'predicted_temperature': 27.5, 'unit': 'celsius', 'metadata': {'sequence_length': 30, 'model_version': '1.0.0'}}}
    }


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(default='healthy', description='Service status')
    model_loaded: bool = Field(..., description='Whether the model is loaded in memory')

    model_config = {'json_schema_extra': {'example': {'status': 'healthy', 'model_loaded': True}}}


class ModelInfoResponse(BaseModel):
    """Response for model metadata endpoint."""

    model_type: str
    version: str
    hyperparameters: dict
    performance: dict
    features: dict

    model_config = {
        'json_schema_extra': {
            'example': {
                'model_type': 'ClimateRNN',
                'version': '1.0.0',
                'hyperparameters': {'hidden_size': 96, 'num_layers': 2, 'seq_length': 30},
                'performance': {'test_mae': 1.852, 'test_rmse': 2.298},
                'features': {'input_features': ['meantemp', 'humidity', 'wind_speed', 'meanpressure'], 'target_feature': 'meantemp'},
            }
        }
    }
