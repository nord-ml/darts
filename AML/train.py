# src/train.py
from utils import fix_pythonpath_if_working_locally
fix_pythonpath_if_working_locally()

import argparse
import os
import json
from functools import partial
import joblib
import optuna

from data_pipeline import ElectricityDataPipeline
from models_config import MODEL_CONFIGS



# Set up project root
TARGET_COMPONENT= "Value_NE5"

def run_training(model_name: str, target_component: str, output_dir: str, n_trials: int):
    """
    Main function to run the training pipeline for a specified model.
    """
    run_id = f"{model_name}_{target_component}"
    print(f"--- Starting Hyperparameter Tuning for: {run_id} ---")

    # 1. Get Model Configuration
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not defined in model_configs.py")
    config = MODEL_CONFIGS[model_name]

    # 2. Prepare Data
    data_pipeline = ElectricityDataPipeline(
        target_component=target_component,
    )
    data_pipeline.prepare_data()

    # 3. Set up Optuna Study
    study_dir = os.path.join(output_dir, run_id)
    os.makedirs(study_dir, exist_ok=True)
    
    study_db_path = f"sqlite:///{os.path.join(study_dir, 'optuna-study.db')}"
    study = optuna.create_study(
        direction="minimize",
        storage=study_db_path,
        study_name=run_id,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    # callback for each epoch?
    

    # 4. Run Optimization
    objective_with_data = partial(config["objective"], data_pipeline=data_pipeline)
    study.optimize(objective_with_data, n_trials=n_trials)

    print(f"--- Tuning complete for {run_id} ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Validation SMAPE: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    # 5. Save Results (Hyperparameters and Study Object)
    joblib.dump(study, os.path.join(study_dir, "study.pkl"))

    summary = {
        "model_name": model_name,
        "target_component": target_component,
        "best_validation_value": study.best_value,
        "best_params": study.best_params,
        "n_trials_completed": len(study.trials),
    }
    summary_path = os.path.join(study_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Tuning summary saved to {summary_path}")
    print("--- Tuning phase complete. To get final performance, run evaluate.py ---")


if __name__ == "__main__":
    fix_pythonpath_if_working_locally()
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning")
    parser.add_argument("model_name", type=str, choices=MODEL_CONFIGS.keys(), help="The name of the model to tune.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    args = parser.parse_args()

    run_training(
        model_name=args.model_name,
        target_component=TARGET_COMPONENT,
        output_dir=args.output_dir,
        n_trials=args.n_trials
    )