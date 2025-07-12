# Replacing Attention in TFT with State Space Models for Efficient Time Series Prediction with Covariates

## Motivation

We aim to develop a more computationally efficient time series prediction model that can effectively utilize future covariates, similar to the Temporal Fusion Transformer (TFT). While TFT achieves state-of-the-art performance, its attention mechanism can be computationally expensive. State Space Models (SSMs), such as Mamba, offer potentially cheaper alternatives but typically do not directly incorporate future covariates. We want to explore the feasibility of replacing the attention mechanism in TFT with an SSM while including future information.

## Formal Problem Setting

Given a time series sequence $X = \{x_1, x_2, \dots, x_t\}$, where each $x_i$ represents a multivariate vector of observed features at time step $i$, and a set of known future covariates $C = \{c_{t+1}, c_{t+2}, \dots, c_{t+T}\}$, our objective is to predict the future values $Y = \{y_{t+1}, y_{t+2}, \dots, y_{t+T}\}$.

The Transformer backbone in TFT is replaced with a State Space Model (SSM). The SSM will be responsible for capturing temporal dependencies while incorporating the future covariates to generate predictions. Other layers from the TFT will remain the same.

The primary goal is to achieve similar or better predictive accuracy compared to TFT, while reducing computational complexity. We will measure both efficiency (speed) and effectiveness (accuracy) using IES and RCS.

### Metrics

We will use following two metrics for evaluating the efficiency versus accuracy trade-off:

* **Inverse Efficiency Score (IES):** This metric is calculated as follows:

$$
IES = \frac{\text{Mean Training Time}}{\text{Accuracy Rate}}
$$

This score helps measure how much time is required per correct prediction. Lower IES values indicate more efficient models.

* **Rate-Correct Score (RCS):** This metric is calculated as follows:

$$
RCS = \frac{\text{Number of Correct Classifications}}{\text{Total Training Time}}
$$

This score represents the number of correct classifications per time unit, with higher RCS values indicating more efficient models.

These metrics will help quantify the trade-off between speed and accuracy in our proposed SSM-based model versus the original TFT.

## Hyperparameter Tuning
As our research is focused on implementing a new model rather than fine-tuning an existing one, we will concentrate primarily on the lookback period. Since SSM models are theoretically capable of capturing indefinite relationships, we aim to identify a lookback period where IES optimization is maximized.

## Dataset and split
We will evaluate our proposed model using the publicly available Zurich energy consumption dataset (https://data.stadt-zuerich.ch/dataset/ewz_stromabgabe_netzebenen_stadt_zuerich) from the darts library. This dataset provides 15-minute electricity consumption readings in Zurich between January 1, 2015, and August 31, 2022. Additionally, it includes relevant exogenous covariates such as humidity, temperature, rainy days, wind speed, and wind direction etc. Furthermore, we will augment this dataset by encoding holiday information as additional future covariates.

We will split the dataset into training, validation and test sets using a ratio of 65:15:20. As we have 8 periods(peaks) we want at least 1/8th(12.5%) of the duration in the validation set. This is assured by assigning 15% to the validation set. Given the total of 268,705 data points, this results in the following splits:
| Split      | Index Range    | Date Range                               |
|------------|----------------|------------------------------------------|
| Training   | 0, 174657      | 2015-01-01 00:00:00, 2019-12-25 08:15:00 |
| Validation | 174658, 214963 | 2019-12-25 08:30:00, 2021-02-17 04:45:00 |
| Test       | 214964, 268704 | 2021-02-17 05:00:00, 2022-08-31 00:00:00 |

![65:15:20 split visualization](tran-val-test-split.svg)

## Comparison to Other Models

We will compare our proposed model to the Temporal Fusion Transformer (TFT), as this is the baseline we are aiming to improve upon. In addition, for statistical comparison, we will include models without covariates, such as ARIMA or Exponential Smoothing, to assess the impact of covariate inclusion on predictive performance.
