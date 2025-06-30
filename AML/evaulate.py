# src/evaluate.py

import argparse
import os
import json
import pandas as pd

from data_pipeline import ElectricityDataPipeline
from models_config import MODEL_CONFIGS
from utils.utils import fix_pythonpath_if_working_locally
from darts.metrics import smape

fix_pythonpath_if_working_locally()
TARGET_COMPONENT = "Value_NE5"  # Default target component for evaluation

def run_evaluation(model_name: str, results_dir: str):
    """
    Trains and evaluates a model using generic functions from model_configs.py.
    """
    run_id = f"{model_name}_{TARGET_COMPONENT}"
    run_dir = os.path.join(results_dir, run_id)
    summary_path = os.path.join(run_dir, "summary.json")

    print(f"--- Starting Final Evaluation for: {run_id} ---")

    # 1. Load Best Hyperparameters and get model config
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.json not found for run '{run_id}'. Please run train.py first.")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    best_params = summary["best_params"]
    config = MODEL_CONFIGS[model_name]
    print(f"Loaded best hyperparameters: {best_params}")

    # 2. Prepare Data
    data_pipeline = ElectricityDataPipeline(
        target_component=TARGET_COMPONENT,
        subset_percentage=0.05,
    )
    data_pipeline.prepare_data()

    # Combine train and validation sets for final training
    final_train_series = data_pipeline.train_scaled.append(data_pipeline.val_scaled)
    final_covariates = data_pipeline.cov_train_scaled.append(data_pipeline.cov_val_scaled)
    print(f"Combined train and val sets for final training. Length: {len(final_train_series)}")

    # 3. Instantiate and Train the Final Model using the generic fitter
    print("Training final model...")
    fit_function = config["fit_model"]
    final_model, training_time = fit_function(
        params=best_params,
        train_series=final_train_series,
        future_covariates=final_covariates
    )
    print(f"Final model training took: {training_time:.2f} seconds.")

    # 4. Evaluate on the Test Set using the generic predictor
    print("Evaluating model on the test set...")
    predict_function = config["predict_model"]
    pred = predict_function(
        model=final_model,
        n=len(data_pipeline.test_scaled),
        covariates_scaled=data_pipeline.covariates_scaled
    )

    # 5. Calculate and Report Metrics
    test_smape = smape(data_pipeline.test_scaled, pred)
    accuracy_rate = 1 - (test_smape / 100)
    ies = training_time / accuracy_rate if accuracy_rate > 0 else float('inf')
    rcs = accuracy_rate / training_time if training_time > 0 else 0

    print(f"Test Set SMAPE: {test_smape:.4f}%")
    print(f"Accuracy Rate (1 - SMAPE): {accuracy_rate:.4f}")
    print(f"Inverse Efficiency Score (IES): {ies:.4f} (lower is better)")
    print(f"Rate-Correct Score (RCS): {rcs:.4f} (higher is better)")

    # 6. Save Evaluation Report
    evaluation_report = {
        "model_name": model_name,
        "target_component": TARGET_COMPONENT,
        "test_smape": test_smape,
        "accuracy_rate": accuracy_rate,
        "training_time_seconds": training_time,
        "inverse_efficiency_score": ies,
        "rate_correct_score": rcs,
        "best_hyperparameters": best_params
    }
    report_path = os.path.join(run_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=4)
    print(f"--- Evaluation complete. Report saved to {report_path} ---")

if __name__ == "__main__":
    # The main execution block remains the same
    parser = argparse.ArgumentParser(description="Run Final Model Evaluation")
    parser.add_argument("model_name", type=str, choices=MODEL_CONFIGS.keys(), help="The name of the model to evaluate.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory where training results are stored.")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model_name,
        results_dir=args.results_dir,
    )