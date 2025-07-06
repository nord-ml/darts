import numpy as np
import pandas as pd
import holidays
from darts import TimeSeries
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

class ElectricityDataPipeline:
    def __init__(
        self,
        target_component: str,
        subset_percentage: float = 0.05, #use it for now to make it as fast as possible in DEV
        train_percent: float = 0.6,
        val_percent: float = 0.2,
    ):
        # The two valid targets for this specific problem
        self.potential_targets = ["Value_NE5", "Value_NE7"]
        if target_component not in self.potential_targets:
            raise ValueError(
                f"Target must be one of {self.potential_targets}, but got {target_component}"
            )
        self.target_component = target_component
        
        self.subset_percentage = subset_percentage
        self.train_percent = train_percent
        self.val_percent = val_percent
        
        # --- Hardcode the data-based covariates ---
        self.data_covariate_cols = [
            "Hr [%Hr]", "RainDur [min]", "T [°C]", "WD [°]",
            "WVv [m/s]", "p [hPa]", "WVs [m/s]", "StrGlo [W/m2]"
        ]

        # Initialize all other attributes
        self.series = None
        self.covariates = None
        self.train, self.val, self.test = None, None, None
        self.cov_train, self.cov_val, self.cov_test = None, None, None
        self.train_scaled, self.val_scaled, self.test_scaled = None, None, None
        self.cov_train_scaled, self.cov_val_scaled, self.cov_test_scaled = None, None, None
        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
        self.val_len = 0


    def prepare_data(self):
        """Loads data, separates target from covariates, splits, and preprocesses."""
        print("--- Starting Data Preparation ---")
        
        full_multivariate_series = ElectricityConsumptionZurichDataset().load()
        print(f"Loaded full dataset with components: {list(full_multivariate_series.components)}")

        # 1. Select the UNIVA_RATE target series
        self.series = full_multivariate_series[self.target_component]
        print(f"Target series selected: '{self.target_component}'")

        # 2. Select the hardcoded data-based covariates
        data_covariates = full_multivariate_series[self.data_covariate_cols]
        print(f"Using {len(self.data_covariate_cols)} data-based covariates: {self.data_covariate_cols}")

        # 3. Create time-based covariates (these are always useful)
        time_covariates = datetime_attribute_timeseries(self.series, attribute="month", one_hot=False)
        time_covariates = time_covariates.stack(datetime_attribute_timeseries(self.series, attribute="day", one_hot=False))
        time_covariates = time_covariates.stack(
            TimeSeries.from_times_and_values(
                times=self.series.time_index,
                values=np.arange(len(self.series)),
                columns=["linear_increase"],
            )
        )
        zh_holidays = holidays.Switzerland(prov="ZH")
        is_holiday = np.array([1 if d in zh_holidays else 0 for d in self.series.time_index])
        holiday_series = TimeSeries.from_times_and_values(
            self.series.time_index, is_holiday.reshape(-1, 1), columns=["is_holiday"]
        )
        time_covariates = time_covariates.stack(holiday_series)

        # 4. Stack all covariates together
        self.covariates = data_covariates.stack(time_covariates)
        print(f"Total covariates created with {self.covariates.width} features.")

        # 5. Subset data for faster development if needed
        if self.subset_percentage < 1.0:
            subset_len = int(len(self.series) * self.subset_percentage)
            self.series = self.series[:subset_len]
            self.covariates = self.covariates[:subset_len]
            print(f"Subsetted data to {len(self.series)} points ({self.subset_percentage*100:.0f}%)")

        # 6. Split data into train/val/test sets
        self._split_data()

        # 7. Scale data
        self.train_scaled = self.scaler_target.fit_transform(self.train)
        self.val_scaled = self.scaler_target.transform(self.val)
        self.test_scaled = self.scaler_target.transform(self.test)
        self.series_scaled = self.scaler_target.transform(self.series)

        self.cov_train_scaled = self.scaler_cov.fit_transform(self.cov_train)
        self.cov_val_scaled = self.scaler_cov.transform(self.cov_val)
        self.cov_test_scaled = self.scaler_cov.transform(self.cov_test)
        self.covariates_scaled = self.scaler_cov.transform(self.covariates)

        print("--- Data Preparation Complete ---")

    def _split_data(self):
        """Splits both the target series and the covariates into train, val, and test."""
        total_len = len(self.series)
        train_size = int(total_len * self.train_percent)
        val_size = int(total_len * self.val_percent)
        train_end_time = self.series.time_index[train_size - 1]
        val_end_time = self.series.time_index[train_size + val_size - 1]
        self.train, remainder = self.series.split_after(train_end_time)
        self.val, self.test = remainder.split_after(val_end_time)
        self.val_len = len(self.val)
        self.cov_train = self.covariates.slice_intersect(self.train)
        self.cov_val = self.covariates.slice_intersect(self.val)
        self.cov_test = self.covariates.slice_intersect(self.test)
        print(f"Splitting data at {train_end_time} and {val_end_time}")
        print(f"Train size: {len(self.train)}, Val size: {len(self.val)}, Test size: {len(self.test)}")