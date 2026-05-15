"""
FastAPI Application

Main application with lifespan management for model loading.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config.settings import settings
from src.model.loader import ModelLoader
from src.model.predictor import ClimatePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for model loading on startup.

    Loads the model, scaler, and config from HuggingFace Hub once
    and stores them in app.state for use by all requests.
    """
    print('=' * 60)
    print(f'{settings.APP_NAME} v{settings.APP_VERSION}')
    print('=' * 60)

    start_time = time.time()

    try:
        # Load model from HuggingFace Hub
        loader = ModelLoader(
            model_id=settings.HF_MODEL_ID,
            token=settings.HF_TOKEN,
            cache_dir=settings.MODEL_CACHE_DIR,
        )

        model, scaler, config = loader.load_all()

        # Create predictor
        predictor = ClimatePredictor(model=model, scaler=scaler, config=config)

        # Store in app state
        app.state.predictor = predictor
        app.state.loader = loader
        app.state.model_loaded = True

        load_time = time.time() - start_time
        print(f'\n✓ Startup complete in {load_time:.2f}s')
        print('=' * 60)

    except Exception as e:
        print(f'\n✗ Failed to load model: {e}')
        app.state.model_loaded = False
        app.state.predictor = None
        raise

    yield  # Application runs

    # Cleanup (if needed)
    print('\nShutting down...')


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description='LSTM-based climate forecasting API for next-day temperature prediction',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    print(f'Unhandled error: {exc}')
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'detail': str(exc) if settings.DEBUG else 'An unexpected error occurred',
        },
    )


app.include_router(router)


@app.get('/')
async def root():
    """Root endpoint with API information."""
    return {
        'message': f'Welcome to {settings.APP_NAME}',
        'version': settings.APP_VERSION,
        'docs': '/docs',
        'health': '/health',
        'model': settings.HF_MODEL_ID,
    }
