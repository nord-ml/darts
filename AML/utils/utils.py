import os
import sys
import matplotlib.pyplot as plt
from darts.metrics import smape
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
def eval_model(model, n, actual_series, val_series, past_covariates, future_covariates, plot=False):

    pred_series = model.predict(
        n=n, 
        num_samples=n,
        series=actual_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates
    )

    smape_score = smape(val_series, pred_series)

    if plot:
        plot_prediction(actual_series, pred_series)

        # plot actual series
        plt.figure(figsize=figsize)
        actual_series[: pred_series.end_time()].plot(label="actual")

        # plot prediction with quantile ranges
        pred_series.plot(
            low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        )
        pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)


        # print(len(actual_series), len(pred_series))
        plt.title(f"SMAPE: {smape_score:.2f}%")
        plt.legend()

    if len(pred_series) == 0 or len(val_series) == 0:
        print("⚠️ Empty prediction or validation series – skipping trial")
        return float("inf")  # so Optuna penalizes the trial, not crash it

    try:
        score = smape(val_series, pred_series)
        if np.isnan(score) or np.isinf(score):
            print("⚠️ SMAPE is NaN or Inf – skipping trial")
            return float("inf")
        return score
    except Exception as e:
        print(f"⚠️ Error in SMAPE: {e}")
        return float("inf")