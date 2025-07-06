# src/model_configs.py
from utils import fix_pythonpath_if_working_locally
fix_pythonpath_if_working_locally()

import time
from functools import partial
import torch
from pytorch_lightning.callbacks import EarlyStopping, Callback
from optuna.integration import PyTorchLightningPruningCallback

from darts.models import TFTModel, Prophet
from darts.models import TFTSSMModel

from darts.utils.likelihood_models import QuantileRegression
from utils.utils import eval_model

# --- Reusable Helper Functions for this file ---

class PatchedPruningCallback(PyTorchLightningPruningCallback, Callback):
    pass

#  ---- Fitters - used once we have trined the model
# NR epochs
NR_EPOCHS = 1


def _tftssm_fitter(params, data_pipeline, train_series, past_covariates, future_covariates):
    """Specific fitting logic for TFT model."""

    tftssm_kwargs = {
        "batch_size": 32,
        'n_epochs': NR_EPOCHS, 
        'random_state': 42,
        'force_reset': True,
        'output_chunk_length': data_pipeline.forecast_horizon,  # do not search for best horizon just define it!
    }
    model = TFTSSMModel(**params, **tftssm_kwargs, likelihood=QuantileRegression())
    
    start_time = time.time()
    model.fit(
        series=train_series,
        future_covariates=future_covariates,
        past_covariates=past_covariates,
        verbose=True
    )
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time


def _tft_fitter(params, data_pipeline, train_series, past_covariates, future_covariates):
    """Specific fitting logic for TFT model."""

    tft_kwargs = {
        "batch_size": 32,
        'n_epochs': NR_EPOCHS, 
        'random_state': 42,
        'force_reset': True,
        'output_chunk_length': data_pipeline.forecast_horizon,  # do not search for best horizon just define it!
    }
    model = TFTModel(**params, **tft_kwargs, likelihood=QuantileRegression())
    
    start_time = time.time()
    model.fit(
        series=train_series,
        future_covariates=future_covariates,
        past_covariates=past_covariates,
        verbose=True
    )
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def _prophet_fitter(params, data_pipeline, train_series, past_covariates, future_covariates):
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

def _default_predictor(model, n, future_covariates, ):
    """Default prediction logic for most Darts models."""
    return model.predict(n=n, 
                         future_covariates=future_covariates)

# --- Objective Functions - for training ---

def _tftssm_objective(trial, data_pipeline):
    # This function remains the same as before...
    in_len = trial.suggest_int("input_chunk_length", 12, 36)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4])
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    callbacks = [PatchedPruningCallback(trial, monitor="val_loss"), EarlyStopping("val_loss", patience=3, verbose=False)]
    model = TFTSSMModel(
        input_chunk_length=in_len, hidden_size=hidden_size,
        lstm_layers=1, num_attention_heads=heads, dropout=dropout, batch_size=32,
        n_epochs=NR_EPOCHS, likelihood=QuantileRegression(),
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": True},
        random_state=42, force_reset=True, save_checkpoints=True, model_name="optuna_tft",
        output_chunk_length=data_pipeline.forecast_horizon,
    )


    model.fit(
        series=data_pipeline.train_scaled, 
        past_covariates=data_pipeline.cov_train_scaled,
        future_covariates=data_pipeline.future_covariates_scaled,

        val_series=data_pipeline.val_scaled,
        val_past_covariates=data_pipeline.cov_val_scaled,
        val_future_covariates=data_pipeline.future_covariates_scaled,
        verbose=True
    )

    best_model = TFTSSMModel.load_from_checkpoint("optuna_tft", best=True)

    return eval_model(
        model=best_model, 
        n=data_pipeline.val_len, 
        actual_series=data_pipeline.val_scaled,
        val_series=data_pipeline.val_scaled, 
        future_covariates=data_pipeline.future_covariates_scaled,
        past_covariates=data_pipeline.cov_val_scaled
    )

    
def _tft_objective(trial, data_pipeline):
    # This function remains the same as before...
    in_len = trial.suggest_int("input_chunk_length", 12, 36)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    heads = trial.suggest_categorical("num_attention_heads", [1, 2, 4])
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    callbacks = [PatchedPruningCallback(trial, monitor="val_loss"), EarlyStopping("val_loss", patience=3, verbose=False)]
    model = TFTModel(
        input_chunk_length=in_len,  hidden_size=hidden_size,
        lstm_layers=1, num_attention_heads=heads, dropout=dropout, batch_size=32,
        n_epochs=NR_EPOCHS, likelihood=QuantileRegression(),
        pl_trainer_kwargs={"callbacks": callbacks, "enable_progress_bar": True},
        random_state=42, force_reset=True, save_checkpoints=True, model_name="optuna_tft",
        output_chunk_length=data_pipeline.forecast_horizon,  #do not search for best horizon just define it!
    )


    model.fit(
        series=data_pipeline.train_scaled, 
        past_covariates=data_pipeline.cov_train_scaled,
        future_covariates=data_pipeline.future_covariates_scaled,

        val_series=data_pipeline.val_scaled,
        val_past_covariates=data_pipeline.cov_val_scaled,
        val_future_covariates=data_pipeline.future_covariates_scaled,
        verbose=True
    )

    best_model = TFTModel.load_from_checkpoint("optuna_tft", best=True)

    return eval_model(
        model=best_model, n=data_pipeline.val_len, 
        actual_series=data_pipeline.val_scaled,
        val_series=data_pipeline.val_scaled, 
        future_covariates=data_pipeline.future_covariates_scaled,
        past_covariates=data_pipeline.cov_val_scaled
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
    model.fit(series=data_pipeline.train_scaled, 
            #   past_covariates=data_pipeline.cov_train_scaled,
              future_covariates=data_pipeline.future_covariates_scaled
        )

    # this is maybe no needed we evaulte them in separate step
    return eval_model(
        model=model, n=data_pipeline.val_len, 
        actual_series=data_pipeline.val_scaled,
        val_series=data_pipeline.val_scaled, 
        past_covariates=[], # no past covariates, cannot accept them
        future_covariates=data_pipeline.future_covariates_scaled
    )

# --- The Main Configuration Dictionary ---

MODEL_CONFIGS = {
    "tft": {
        "model_class": TFTModel,
        "objective": _tft_objective,
        "fit_model": _tft_fitter,         
        "predict_model": _default_predictor
    },
    "prophet": {
        "model_class": Prophet,
        "objective": _prophet_objective,
        "fit_model": _prophet_fitter,     
        "predict_model": _default_predictor
    },
    'ssm-tft': {
        "model_class": TFTSSMModel,
        "objective": _tftssm_objective,
        "fit_model": _tftssm_fitter,
        "predict_model": _default_predictor
    }
}