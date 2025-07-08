import os
import sys
import matplotlib.pyplot as plt
from darts.metrics import smape
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
import numpy as np

# Unility functions 

def fix_pythonpath_if_working_locally():
    """Add the project root to the python path if running locally."""
    if os.path.basename(os.getcwd()) == "AML":
        os.chdir("..")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    print(f"Working directory: {os.getcwd()}")

# we allow actual and val series so that they can be different, if we start prediction later
def eval_model(model:MixedCovariatesTorchModel, n, actual_series, val_series, past_covariates, future_covariates, plot=False):

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
    else:
        print("THIS MODEL DOES NOT SUPPORT")
        all_smapes = model.backtest(actual_series, metric=smape)

    smape_score = np.mean(all_smapes) 

    return smape_score