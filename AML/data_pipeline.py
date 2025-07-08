from utils import fix_pythonpath_if_working_locally
fix_pythonpath_if_working_locally()


import numpy as np
import pandas as pd
import holidays
from darts import TimeSeries
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.logging import get_logger

logger = get_logger(__name__)

class ElectricityDataPipeline:
    def __init__(
        self,
        target_component: str = "Value_NE5",
        subset_percentage: float = 0.01, #use it for now to make it as fast as possible in DEV
        train_percent: float = 0.65,
        val_percent: float = 0.15,
    ):
    # number samples to predict
        nr_samples = 12
        self.forecast_horizon = nr_samples
        self.val_len = nr_samples 

        # The two valid targets for this specific problem
        self.potential_targets = ["Value_NE5", "Value_NE7"]
        if target_component not in self.potential_targets:
            raise ValueError(
                f"Target must be one of {self.potential_targets}, but got {target_component}"
            )
        self.target_component = target_component
        
        if subset_percentage < 1.0:
            logger.error(
                f"❗❗Using subset_percentage < 1.0 ({subset_percentage}) for development purposes. "
                "This will speed up the data loading and processing, but results may not be representative.❗❗"
            )
        
        self.subset_percentage = subset_percentage
        self.train_percent = train_percent
        self.val_percent = val_percent
        
        # --- Hardcode the data-based covariates ---
        self.covariate_cols = [
            "Hr [%Hr]", "RainDur [min]", "T [°C]", "WD [°]",
            "WVv [m/s]", "p [hPa]", "WVs [m/s]", "StrGlo [W/m2]"
        ]

        self.future_covariate_cols = [
            "holidays_custom", "month", "weekday"
        ]

        # Initialize all other attributes
        self.series = None
        self.covariates = None
        self.future_covariates_scaled = None
        self.train, self.val, self.test = None, None, None
        self.train_scaled, self.val_scaled, self.test_scaled = None, None, None
        
        self.cov_train, self.cov_val, self.cov_test = None, None, None
        self.cov_train_scaled, self.cov_val_scaled, self.cov_test_scaled = None, None, None
        
        self.scaler_target = Scaler()
        # self.scaler_cov = Scaler()


    def prepare_data(self):
        """Loads data, separates target from covariates, splits, and preprocesses."""
        print("--- Starting Data Preparation ---")
        
        full_multivariate_series = ElectricityConsumptionZurichDataset().load()
        print(f"Loaded full dataset with components: {list(full_multivariate_series.components)}")


        # manually add covariates 
        full_multivariate_series = (
            full_multivariate_series
            .add_holidays(country_code="CH", state="ZH")
             .with_columns_renamed("holidays", "holidays_custom")  # Rename to avoid Prophet conflict
            .stack(datetime_attribute_timeseries(full_multivariate_series, attribute="month"))
            .stack(datetime_attribute_timeseries(full_multivariate_series, attribute="weekday"))
            # exclude linear increase -> no trend in data (at least not visual)
        )

        self.target_series = full_multivariate_series[self.target_component]
        print(f"Target series selected: '{self.target_component}'")

        # 1. Select the UNIVARIATE target series
        self.series = full_multivariate_series[self.target_component]
        print(f"Target series selected: '{self.target_component}'")

        # 2. Select the hardcoded data-based covariates
        self.covariates = full_multivariate_series[self.covariate_cols]
        print(f"Using {self.covariates.width} data-based covariates: {self.covariate_cols}")
        
        #  3 define future covaiates
        self.future_covariates = full_multivariate_series[self.future_covariate_cols]
        print(f"Using {self.future_covariates.width} future covariates: {self.future_covariate_cols}")


        # 6. Subset data for faster development if needed
        if self.subset_percentage < 1.0:
            subset_len = int(len(self.series) * self.subset_percentage)
            self.series = self.series[:subset_len]
            self.covariates = self.covariates[:subset_len]
            self.future_covariates = self.future_covariates[:subset_len]
            print(f"Subsetted data to {len(self.series)} points ({self.subset_percentage*100:.0f}%)")

        # 6. Split data into train/val/test sets
        self._split_data()

        # 7. Scale data


        self.target_scaler, self.train_scaled, self.val_scaled, self.test_scaled = self._scale_series(Scaler(), self.train, [self.val, self.test])
        self.covariate_scaler, self.cov_train_scaled, self.cov_val_scaled, self.cov_test_scaled = self._scale_series(Scaler(), self.cov_train, [self.cov_val, self.cov_test])
        self.future_covariates_scaler, self.future_covariates_scaled = self._scale_series(Scaler(), self.future_covariates)

        print("--- Data Preparation Complete ---")

    def _split_series(self, series, train_end:str, val_end:str):
        train, remainder = series.split_after(train_end)
        val, test = remainder.split_after(val_end)

        return train, val, test

    def _scale_series(self, scaler, reference_series, additional_series=None):
        if not additional_series:
            additional_series = []
        transformed_reference = scaler.fit_transform(reference_series).astype(np.float32)
        transformed_additionals = [scaler.transform(additional).astype(np.float32) for additional in additional_series]

        return scaler, transformed_reference, *transformed_additionals

    
    def _split_data(self):
        """Splits both the target series and the covariates into train, val, and test."""
        
        total_len = len(self.series)
        train_size = int(total_len * self.train_percent)
        val_size = int(total_len * self.val_percent)

        train_end_time = self.series.time_index[train_size - 1]
        val_end_time = self.series.time_index[train_size + val_size - 1]

        self.train, self.val, self.test = self._split_series(self.series, train_end=train_end_time, val_end=val_end_time)
        self.cov_train, self.cov_val, self.cov_test = self._split_series(self.covariates, train_end=train_end_time, val_end=val_end_time)

        print(f"Splitting data at {train_end_time} and {val_end_time}")
        print(f"Train size: {len(self.train)}, Val size: {len(self.val)}, Test size: {len(self.test)}")

# test the data laoder via main
if __name__ == "__main__":
    pipeline = ElectricityDataPipeline(target_component="Value_NE5", subset_percentage=0.05)
    pipeline.prepare_data()
    
    print(f"Train series length: {len(pipeline.train)}")
    print(f"Val series length: {len(pipeline.val)}")
    print(f"Test series length: {len(pipeline.test)}")
    
    print(f"Train covariates shape: {pipeline.cov_train.shape}")
    print(f"Val covariates shape: {pipeline.cov_val.shape}")
    print(f"Test covariates shape: {pipeline.cov_test.shape}")
    
    print("Data preparation completed successfully.")
    
