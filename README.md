# How to use this repo
1. Clone the repository
2. open it in devcontainers (you will need to wait about 30 minutes for the forst time (pip needs to install all the packages))

3. open AML folder and there are our experiments


# How to run the expetiments and everything?
1) open the terminal
2) to run the evaulation (hyperparam finetuning, it works currently for all models)
- Number of trials is how many experiments can be run (easier to run 1 for testing)
```bash
python AML/train.py {model_name} --n-trials {n_trials}
```

eg 
```bash
python AML/train.py tft --n-trials 1
```

3) Evaulate
- evaulation is done on best models for each architecture
```bash
python AML/evaulate.py {model_name} 
eg
python AML/evaulate.py tft
```

# current available models
`tft` , `ssm-tft` and `prophet`




# What do we exactly do?
- We take an TFT with covariates and train it simple right?

Which covaraites do we define? -> month days and holidays in switzerland

# What do we compare
- The Default TFT with covariates -> with best params search? -> I feel like we have to?
  Or we take params from the original paper?
- our SSM where we did params search 
- Prophet or something like this

# current covariates
- month, day, the numerical index, and boolean if it is a holiday in Switzerland


# Adam Comments
- Smape is infinity for my small samples, but might be issue of testing, needs definitely some tests!!
- TFT - works for training/evaulation


@ we have Train -> val -> test split ( i hope it is not test->val)


# What we need to do
- Add reasonable grid search for hyperparams on SSM! (approve that they are being reflected)
- Mamba specific hyperparams
  1. 1D convolution kernel: parameters that captures temporal dependencies, experimented on values [1, 2, 3]. Typically set on 2.
  2. Hidden state dimension: controls overfitting and underfitting, tested within range of [2-256]. Dynamically determined by number of variates considered within data.
  3. Expansion factor(controversal parameter): Differ from [1-32]. Only considered by some articles, refering to be 1, so that it may increase parameter size and GPU cost. However, the origianl mamba library and some articles set 16 as default.

# Some installs
`pip install einops`
- you might encouter problems with smm models becaus ethe params are not passed
you have to copy it into editable
`cp /workspaces/darts/darts/models/forecasting/tft_ssm_model.py /app/darts/models/forecasting/tft_ssm_model.py`
