# src/model_configs.py

import time
from functools import partial
import torch
from pytorch_lightning.callbacks import EarlyStopping, Callback
from optuna.integration import PyTorchLightningPruningCallback

from darts.models import TFTModel, Prophet
from darts.utils.likelihood_models import QuantileRegression
from utils.utils import eval_model

# --- Reusable Helper Functions for this file ---

class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    pass

# NR epochs
NR_EPOCHS = 1
def _tft_fitter(params, train_series, future_covariates):
    """Specific fitting logic for TFT model."""

    tft_kwargs = {
        "batch_size": 32,
        'n_epochs': 1, # Use a higher number of epochs for final training
        'random_state': 42,
        'force_reset': True,
    }
    model = TFTModel(**params, **tft_kwargs, likelihood=QuantileRegression())
    
    start_time = time.time()
    model.fit(
        series=train_series,
        future_covariates=future_covariates,
        verbose=True
    )
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def _prophet_fitter(params, train_series, future_covariates):
    """Specific fitting logic for Prophet model."""
    model = Prophet(**params)
    
    start_time = time.time()
    model.fit(
        series=train_series,
        future_covariates=future_covariates
    )
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def _default_predictor(model, n, covariates_scaled):
    """Default prediction logic for most Darts models."""
    return model.predict(n=n, future_covariates=covariates_scaled)

# --- Objective Functions (from training phase, unchanged) ---

def _tft_objective(trial, data_pipeline):
    # This function remains the same as before...
    in_len = trial.suggest_int("input_chunk_length", 12, 36)
    out_len = trial.suggest_int("output_chunk_length", 6, 24)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4])
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    callbacks = [PatchedPruningCallback(trial, monitor="val_loss"), EarlyStopping("val_loss", patience=3, verbose=False)]
    model = TFTModel(
        input_chunk_length=in_len, output_chunk_length=out_len, hidden_size=hidden_size,
        lstm_layers=1, num_attention_heads=heads, dropout=dropout, batch_size=32,
        n_epochs=NR_EPOCHS, likelihood=QuantileRegression(), optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": True},
        random_state=42, force_reset=True, save_checkpoints=True, model_name="optuna_tft"
    )
    val_extended = data_pipeline.series_scaled[-(data_pipeline.val_len + in_len):]
    cov_val_ext = data_pipeline.covariates_scaled.slice_intersect(val_extended)
    model.fit(
        series=data_pipeline.train_scaled, val_series=val_extended,
        future_covariates=data_pipeline.cov_train_scaled, val_future_covariates=cov_val_ext,
        verbose=True
    )
    best_model = TFTModel.load_from_checkpoint("optuna_tft", best=True)

# this might now be needed we evaualte on later steps 
    return eval_model(
        model=best_model, n=data_pipeline.val_len, actual_series=data_pipeline.series_scaled,
        val_series=data_pipeline.val_scaled, 
        # num_samples=100, dont use we want to make the prediction deterministic, not from some quantile for now, maybe we want to change it?
        future_covariates=data_pipeline.covariates_scaled
    )

def _prophet_objective(trial, data_pipeline):
    # This function also remains the same...
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.01, 0.5, log=True)
    seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.1, 10.0, log=True)
    monthly_seasonality_in_hours = 30.5 * 24
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        add_seasonalities={'name': 'monthly', 'seasonal_periods': monthly_seasonality_in_hours, 'fourier_order': 5},
    )
    model.fit(series=data_pipeline.train_scaled, future_covariates=data_pipeline.cov_train_scaled)

    # this is maybe no needed we evaulte them in separate step
    return eval_model(
        model=model, n=data_pipeline.val_len, actual_series=data_pipeline.series_scaled,
        val_series=data_pipeline.val_scaled, future_covariates=data_pipeline.covariates_scaled
    )

# --- The Main Configuration Dictionary ---

MODEL_CONFIGS = {
    "tft": {
        "model_class": TFTModel,
        "objective": _tft_objective,
        "fit_model": _tft_fitter,          # <-- ADDED
        "predict_model": _default_predictor # <-- ADDED
    },
    "prophet": {
        "model_class": Prophet,
        "objective": _prophet_objective,
        "fit_model": _prophet_fitter,      # <-- ADDED
        "predict_model": _default_predictor # <-- ADDED
    }
}