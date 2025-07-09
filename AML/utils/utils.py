import os
import csv
import sys
import matplotlib.pyplot as plt
from darts.metrics import smape
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
import numpy as np
from pathlib import Path

# Unility functions 

def fix_pythonpath_if_working_locally():
    """Add the project root to the python path if running locally."""
    if os.path.basename(os.getcwd()) == "AML":
        os.chdir("..")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    print(f"Working directory: {os.getcwd()}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# we allow actual and val series so that they can be different, if we start prediction later
def eval_model(
        model:MixedCovariatesTorchModel, 
        n, 
        actual_series, 
        val_series, 
        past_covariates, 
        future_covariates, 
        epochs_trained,
        normalized_execution_time,
        training_series=False, 
        ):


    if model.supports_optimized_historical_forecasts:
        validation_backtest = model.historical_forecasts(
            series=actual_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=n,
            stride=1,
            retrain=False,
            overlap_end=False,
            last_points_only=False
        )
        all_smapes = model.backtest(actual_series, historical_forecasts=validation_backtest, metric=smape)
        num_params = count_parameters(model.model)
    else:
        print("Model doesn't support optimized historical forecast.")
        print("Performing monthly retraining on existing model")
        assert training_series, "Evaluating a model that needs a refit requires the parameter `training_series`"
        
        eval_timeframe = 30 * 24 * 4 # 30 days * 24 hours * 15 minutes -> monthly
        n_evals = int(np.floor(len(val_series) / eval_timeframe))

        train_ts = training_series
        val_ts = val_series

        all_smapes = []

        for iteration in range(n_evals):
            print(f"retraining iteration {iteration+1}/{n_evals}")
            retrain_ts = train_ts.append(val_ts[:iteration*eval_timeframe])
            model.fit(
                series=retrain_ts,
                future_covariates=future_covariates
            )
            prediction = model.predict(eval_timeframe)

            all_smapes.append(smape(val_ts[:(1+iteration)*eval_timeframe], prediction))
        
        num_params = "NA"

    smape_score = np.mean(all_smapes) 

    accuracy_rate = 1 - (smape_score / 100)
    ies = normalized_execution_time / accuracy_rate if accuracy_rate > 0 else float('inf')
    rcs = accuracy_rate / normalized_execution_time if normalized_execution_time > 0 else 0

    if not epochs_trained:
        epochs_trained = "NA"

    with open(f"./{type(model).__name__}-eval-log.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([type(model).__name__, epochs_trained, normalized_execution_time, smape_score, ies, rcs, num_params])

    return ies