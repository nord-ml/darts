import os
import sys
import matplotlib.pyplot as plt
from darts.metrics import smape

# Unility functions 

def fix_pythonpath_if_working_locally():
    """Add the project root to the python path if running locally."""
    if os.path.basename(os.getcwd()) == "src":
        os.chdir("..")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    print(f"Working directory: {os.getcwd()}")

def eval_model(model, n, actual_series, val_series, plot=False, **predict_kwargs):
    """
    Predicts, optionally plots, and returns the SMAPE score.
    """
    # run the prediction
    pred = model.predict(n=n, **predict_kwargs)

    # plot
    if plot:
        plt.figure(figsize=(10, 6))
        actual_series[:pred.end_time()].plot(label="actual")
        # Check if prediction is probabilistic
        if pred.is_probabilistic:
            pred.plot(low_quantile=0.05, high_quantile=0.95, label="5%-95% quantiles")
        else:
            pred.plot(label="forecast")
        plt.title(f"SMAPE: {smape(val_series, pred):.2f}%")
        plt.legend()
        plt.show()

    # return the SMAPE so we can use it as our objective value
    return smape(val_series, pred)