
<!-- README.md is generated from README.Rmd. Please edit that file -->

# kantime

<!-- badges: start -->

<!-- badges: end -->

{kantime} is a minimal wrapper for Time Series Kolmogorov Arnold
Networks in R. By binding nixtla’s {neuralforecast} engine, KAN, from
Python to use in R. Additional steps were made to bridged to
{modeltime}, hence the name {kantime}!

## Installation

You can install the development version of kantime like so:

``` r
devtools::install_github("frankiethull/kantime")
```

## Example

This is a basic example showing the barebones {reticulate} Python
wrappers. These are the core bindings before registering can configuring
to {parsnip} and then bridging to {modeltime}.

#### setup {kantime}

kantime requires Python and neuralforecast. Python dependencies are
managed automatically via `reticulate::py_require()` when the package
loads.

``` r
# load the R library -------------------
# neuralforecast is installed automatically on first use
library(kantime)
```

##### example data for testing the library:

``` r
## data comes from sister repo

air_passengers_df <- arrow::read_csv_arrow("https://raw.githubusercontent.com/frankiethull/nixtla-r-tutorial/refs/heads/main/airpassengersDF.csv")

air_passengers_df <- air_passengers_df |>
    dplyr::mutate(
      ds = as.Date(ds)
    )

Sys.sleep(1)

train_df <- air_passengers_df |> dplyr::slice(1:132)
test_df  <- air_passengers_df |> dplyr::anti_join(train_df)
#> Joining with `by = join_by(unique_id, ds, y)`
```

#### barebone internals, no xregs, with conformal prediction

note that nixtla’s design requires a *unique_id* for each unique time
series, *ds* for the time column, and *y* for the outcome variable. The
internals work as minimal python wrappers to **fit** and **predict** on
a time series data like so:

``` r

# kan specs --
kan_model_specs <- kantime:::kan_spec(h = 12L)

# fit
kan_model_fit   <- kantime:::conformal_fit(model_spec = kan_model_specs, df = train_df)

# predict
conf_kan_preds  <- kantime:::conformal_predict(
                                     model_spec = kan_model_specs, 
                                     model_fit = kan_model_fit,
                                     level = 90L)
```

These internals are built thanks to {reticulate} and of using Nixtla’s
{neuralforecast} via R. While the idea works ok, the results and
workflow can be clunky for R, especially when building for many unique
IDs (many models for different time series loses pandas row IDs).
Additionally, handling parallel processes in R via {reticulate} for cv,
tuning and training in Python requires additional robustness.

{kantime} is super experimental in it’s design. Which is also why this
is not a full {neuralforecast} binding. This binding in particular is
going to leverage the root KAN model, a few helper utils, then bind the
base KAN model to {parsnip} and bridge to {modeltime}. This loses pieces
of the underlying Nixtla functionality but replacing with {tidymodels} &
{modeltime} functionality.

#### {neuralforecast}’s kan with a {modeltime} bridge

given our underlying internals, we can bind these functions to
{modeltime} which is similar to registering a parsnip model but requires
an additional bridge. The bridge implementation is shown below, but we
have to add predict methods and there’s quite a bit of underlying work
that makes it fully functioning.

``` r
kan_bridge <- kantime:::kan_bridge_fit_impl(
  x = train_df |> dplyr::select(-y),
  y = train_df |> dplyr::pull(y),
  h = 12L,
  input_size = 24L,
  max_steps = 10L,
  freq = "ME"
)

kan_bridge |> predict()
```

#### {kantime} workflow within {modeltime}

Remembering that once the model has been bridged to {modeltime}, we lose
some of the underlying nixtla utilities, but get access to tidymodels +
modeltime utilities in R. What’s this mean? We can now use this KAN
binding with tidymodels tools like `initial_time_split` and we can use
leverage `modeltime`’s toolkit to further calibrate & validate KAN
models. In fact, we went through these additional bridging steps to have
a full suite of backtesting, ensembling, calibrating, and scoring tools
that will be very familiar if already using {modeltime} and/or
{tidymodels}.

``` r
library(parsnip)
library(modeltime)

# time split
splits   <- rsample::initial_time_split(air_passengers_df, prop = .92)
training <- rsample::training(splits)
testing  <- rsample::testing(splits)

# kantime fit
kantime_fit <- kan(h = 12L,
                   input_size = 24L,
                   max_steps = 10L,
                   freq = "ME") |>
                set_engine("kan") |>
               fit(y ~ ds + unique_id, data = training)
#> Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 1858.35it/s]                                                                             Training: |          | 0/? [00:00<?, ?it/s]Training: |          | 0/? [00:00<?, ?it/s]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  2.79it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  2.79it/s, v_num=30, train_loss_step=1.860]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  2.79it/s, v_num=30, train_loss_step=1.860, train_loss_epoch=1.860]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.860, train_loss_epoch=1.860]        Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.860, train_loss_epoch=1.860]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  2.50it/s, v_num=30, train_loss_step=1.860, train_loss_epoch=1.860]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  2.50it/s, v_num=30, train_loss_step=1.810, train_loss_epoch=1.860]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  2.50it/s, v_num=30, train_loss_step=1.810, train_loss_epoch=1.810]Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.810, train_loss_epoch=1.810]        Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.810, train_loss_epoch=1.810]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.36it/s, v_num=30, train_loss_step=1.810, train_loss_epoch=1.810]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.36it/s, v_num=30, train_loss_step=1.740, train_loss_epoch=1.810]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.36it/s, v_num=30, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.740, train_loss_epoch=1.740]        Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.23it/s, v_num=30, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.23it/s, v_num=30, train_loss_step=1.680, train_loss_epoch=1.740]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.22it/s, v_num=30, train_loss_step=1.680, train_loss_epoch=1.680]Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.680, train_loss_epoch=1.680]        Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.680, train_loss_epoch=1.680]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.82it/s, v_num=30, train_loss_step=1.680, train_loss_epoch=1.680]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.81it/s, v_num=30, train_loss_step=1.620, train_loss_epoch=1.680]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.81it/s, v_num=30, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.620, train_loss_epoch=1.620]        Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  4.05it/s, v_num=30, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  4.05it/s, v_num=30, train_loss_step=1.570, train_loss_epoch=1.620]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  4.04it/s, v_num=30, train_loss_step=1.570, train_loss_epoch=1.570]Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.570, train_loss_epoch=1.570]        Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.570, train_loss_epoch=1.570]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  2.98it/s, v_num=30, train_loss_step=1.570, train_loss_epoch=1.570]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  2.97it/s, v_num=30, train_loss_step=1.500, train_loss_epoch=1.570]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  2.97it/s, v_num=30, train_loss_step=1.500, train_loss_epoch=1.500]Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.500, train_loss_epoch=1.500]        Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.500, train_loss_epoch=1.500]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.06it/s, v_num=30, train_loss_step=1.500, train_loss_epoch=1.500]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.06it/s, v_num=30, train_loss_step=1.430, train_loss_epoch=1.500]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.05it/s, v_num=30, train_loss_step=1.430, train_loss_epoch=1.430]Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.430, train_loss_epoch=1.430]        Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.430, train_loss_epoch=1.430]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.21it/s, v_num=30, train_loss_step=1.430, train_loss_epoch=1.430]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.20it/s, v_num=30, train_loss_step=1.390, train_loss_epoch=1.430]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.20it/s, v_num=30, train_loss_step=1.390, train_loss_epoch=1.390]Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.390, train_loss_epoch=1.390]        Epoch 9:   0%|          | 0/1 [00:00<?, ?it/s, v_num=30, train_loss_step=1.390, train_loss_epoch=1.390]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.95it/s, v_num=30, train_loss_step=1.390, train_loss_epoch=1.390]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.95it/s, v_num=30, train_loss_step=1.320, train_loss_epoch=1.390]
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s][A
#> Validation DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 1460.92it/s][A
#>                                                                         [AEpoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.94it/s, v_num=30, train_loss_step=1.320, train_loss_epoch=1.390]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.94it/s, v_num=30, train_loss_step=1.320, train_loss_epoch=1.320]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.89it/s, v_num=30, train_loss_step=1.320, train_loss_epoch=1.320]
#> Predicting: |          | 0/? [00:00<?, ?it/s]Predicting: |          | 0/? [00:00<?, ?it/s]Predicting DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 144.60it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 137.94it/s]
#> Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 2272.10it/s]                                                                             Training: |          | 0/? [00:00<?, ?it/s]Training: |          | 0/? [00:00<?, ?it/s]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  3.20it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  3.20it/s, v_num=32, train_loss_step=1.210]Epoch 0: 100%|██████████| 1/1 [00:00<00:00,  3.19it/s, v_num=32, train_loss_step=1.210, train_loss_epoch=1.210]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.210, train_loss_epoch=1.210]        Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.210, train_loss_epoch=1.210]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  1.92it/s, v_num=32, train_loss_step=1.210, train_loss_epoch=1.210]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  1.92it/s, v_num=32, train_loss_step=1.160, train_loss_epoch=1.210]Epoch 1: 100%|██████████| 1/1 [00:00<00:00,  1.92it/s, v_num=32, train_loss_step=1.160, train_loss_epoch=1.160]Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.160, train_loss_epoch=1.160]        Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.160, train_loss_epoch=1.160]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.47it/s, v_num=32, train_loss_step=1.160, train_loss_epoch=1.160]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.47it/s, v_num=32, train_loss_step=1.090, train_loss_epoch=1.160]Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.46it/s, v_num=32, train_loss_step=1.090, train_loss_epoch=1.090]Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.090, train_loss_epoch=1.090]        Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.090, train_loss_epoch=1.090]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.34it/s, v_num=32, train_loss_step=1.090, train_loss_epoch=1.090]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.34it/s, v_num=32, train_loss_step=1.030, train_loss_epoch=1.090]Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.33it/s, v_num=32, train_loss_step=1.030, train_loss_epoch=1.030]Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.030, train_loss_epoch=1.030]        Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=1.030, train_loss_epoch=1.030]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.85it/s, v_num=32, train_loss_step=1.030, train_loss_epoch=1.030]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.85it/s, v_num=32, train_loss_step=0.975, train_loss_epoch=1.030]Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.84it/s, v_num=32, train_loss_step=0.975, train_loss_epoch=0.975]Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.975, train_loss_epoch=0.975]        Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.975, train_loss_epoch=0.975]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  2.07it/s, v_num=32, train_loss_step=0.975, train_loss_epoch=0.975]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  2.07it/s, v_num=32, train_loss_step=0.909, train_loss_epoch=0.975]Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  2.07it/s, v_num=32, train_loss_step=0.909, train_loss_epoch=0.909]Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.909, train_loss_epoch=0.909]        Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.909, train_loss_epoch=0.909]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  1.77it/s, v_num=32, train_loss_step=0.909, train_loss_epoch=0.909]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  1.77it/s, v_num=32, train_loss_step=0.857, train_loss_epoch=0.909]Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  1.77it/s, v_num=32, train_loss_step=0.857, train_loss_epoch=0.857]Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.857, train_loss_epoch=0.857]        Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.857, train_loss_epoch=0.857]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.09it/s, v_num=32, train_loss_step=0.857, train_loss_epoch=0.857]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.09it/s, v_num=32, train_loss_step=0.809, train_loss_epoch=0.857]Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  3.09it/s, v_num=32, train_loss_step=0.809, train_loss_epoch=0.809]Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.809, train_loss_epoch=0.809]        Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.809, train_loss_epoch=0.809]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.65it/s, v_num=32, train_loss_step=0.809, train_loss_epoch=0.809]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.65it/s, v_num=32, train_loss_step=0.763, train_loss_epoch=0.809]Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.64it/s, v_num=32, train_loss_step=0.763, train_loss_epoch=0.763]Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.763, train_loss_epoch=0.763]        Epoch 9:   0%|          | 0/1 [00:00<?, ?it/s, v_num=32, train_loss_step=0.763, train_loss_epoch=0.763]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  3.59it/s, v_num=32, train_loss_step=0.763, train_loss_epoch=0.763]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  3.59it/s, v_num=32, train_loss_step=0.722, train_loss_epoch=0.763]
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s][A
#> Validation DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 2283.24it/s][A
#>                                                                         [AEpoch 9: 100%|██████████| 1/1 [00:00<00:00,  3.56it/s, v_num=32, train_loss_step=0.722, train_loss_epoch=0.763]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  3.55it/s, v_num=32, train_loss_step=0.722, train_loss_epoch=0.722]Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  3.41it/s, v_num=32, train_loss_step=0.722, train_loss_epoch=0.722]
#> Predicting: |          | 0/? [00:00<?, ?it/s]Predicting: |          | 0/? [00:00<?, ?it/s]Predicting DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00,  5.68it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00,  5.66it/s]

# kantime predict
kan_point_predictions <- 
kantime_fit |>
 modeltime_table() |>
  modeltime_forecast(actual_data = training, new_data = testing) 
#> Predicting: |          | 0/? [00:00<?, ?it/s]Predicting: |          | 0/? [00:00<?, ?it/s]Predicting DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 400.45it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 354.52it/s]

kan_point_predictions |>
  plot_modeltime_forecast(.conf_interval_show = FALSE, .interactive = FALSE)
```

<img src="man/figures/README-unnamed-chunk-4-1.png" alt="" width="100%" />

### RMoK example

``` r
# RMoK (Robust Mixture of Kernels) via neuralforecast
rmok_fit <- rmok(h = 12L,
                 input_size = 24L,
                 max_steps = 10L,
                 freq = "ME") |>
              set_engine("rmok") |>
             fit(y ~ ds + unique_id, data = training)
#> Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 3151.24it/s]                                                                             Training: |          | 0/? [00:00<?, ?it/s]Training: |          | 0/? [00:00<?, ?it/s]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 81.52it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 79.62it/s, v_num=33, train_loss_step=1.890]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 77.92it/s, v_num=33, train_loss_step=1.890, train_loss_epoch=1.890]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.890, train_loss_epoch=1.890]        Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.890, train_loss_epoch=1.890]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 20.04it/s, v_num=33, train_loss_step=1.890, train_loss_epoch=1.890]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 19.86it/s, v_num=33, train_loss_step=1.800, train_loss_epoch=1.890]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 19.68it/s, v_num=33, train_loss_step=1.800, train_loss_epoch=1.800]Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.800, train_loss_epoch=1.800]        Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.800, train_loss_epoch=1.800]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 122.15it/s, v_num=33, train_loss_step=1.800, train_loss_epoch=1.800]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 117.92it/s, v_num=33, train_loss_step=1.730, train_loss_epoch=1.800]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 113.71it/s, v_num=33, train_loss_step=1.730, train_loss_epoch=1.730]Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.730, train_loss_epoch=1.730]         Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.730, train_loss_epoch=1.730]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 91.15it/s, v_num=33, train_loss_step=1.730, train_loss_epoch=1.730]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 88.18it/s, v_num=33, train_loss_step=1.700, train_loss_epoch=1.730]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 84.90it/s, v_num=33, train_loss_step=1.700, train_loss_epoch=1.700]Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.700, train_loss_epoch=1.700]        Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.700, train_loss_epoch=1.700]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 96.21it/s, v_num=33, train_loss_step=1.700, train_loss_epoch=1.700]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 93.51it/s, v_num=33, train_loss_step=1.740, train_loss_epoch=1.700]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 89.97it/s, v_num=33, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.740, train_loss_epoch=1.740]        Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 81.74it/s, v_num=33, train_loss_step=1.740, train_loss_epoch=1.740]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 79.37it/s, v_num=33, train_loss_step=1.620, train_loss_epoch=1.740]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 76.55it/s, v_num=33, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.620, train_loss_epoch=1.620]        Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 17.36it/s, v_num=33, train_loss_step=1.620, train_loss_epoch=1.620]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 17.23it/s, v_num=33, train_loss_step=1.600, train_loss_epoch=1.620]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 17.09it/s, v_num=33, train_loss_step=1.600, train_loss_epoch=1.600]Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.600, train_loss_epoch=1.600]        Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.600, train_loss_epoch=1.600]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 87.76it/s, v_num=33, train_loss_step=1.600, train_loss_epoch=1.600]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 85.41it/s, v_num=33, train_loss_step=1.550, train_loss_epoch=1.600]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 82.43it/s, v_num=33, train_loss_step=1.550, train_loss_epoch=1.550]Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.550, train_loss_epoch=1.550]        Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.550, train_loss_epoch=1.550]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 16.36it/s, v_num=33, train_loss_step=1.550, train_loss_epoch=1.550]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 16.24it/s, v_num=33, train_loss_step=1.510, train_loss_epoch=1.550]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 16.12it/s, v_num=33, train_loss_step=1.510, train_loss_epoch=1.510]Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.510, train_loss_epoch=1.510]        Epoch 9:   0%|          | 0/1 [00:00<?, ?it/s, v_num=33, train_loss_step=1.510, train_loss_epoch=1.510]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 23.78it/s, v_num=33, train_loss_step=1.510, train_loss_epoch=1.510]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 23.54it/s, v_num=33, train_loss_step=1.490, train_loss_epoch=1.510]
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s][A
#> Validation DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 1417.95it/s][A
#>                                                                         [AEpoch 9: 100%|██████████| 1/1 [00:00<00:00, 21.68it/s, v_num=33, train_loss_step=1.490, train_loss_epoch=1.510]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 21.46it/s, v_num=33, train_loss_step=1.490, train_loss_epoch=1.490]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 18.01it/s, v_num=33, train_loss_step=1.490, train_loss_epoch=1.490]
#> Predicting: |          | 0/? [00:00<?, ?it/s]Predicting: |          | 0/? [00:00<?, ?it/s]Predicting DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 560.06it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 474.84it/s]
#> Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking: |          | 0/? [00:00<?, ?it/s]Sanity Checking DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Sanity Checking DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 3495.25it/s]                                                                             Training: |          | 0/? [00:00<?, ?it/s]Training: |          | 0/? [00:00<?, ?it/s]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 90.34it/s]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 87.86it/s, v_num=35, train_loss_step=1.400]Epoch 0: 100%|██████████| 1/1 [00:00<00:00, 85.49it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]        Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 82.10it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 80.10it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 1: 100%|██████████| 1/1 [00:00<00:00, 77.60it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 1:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]        Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 90.16it/s, v_num=35, train_loss_step=1.400, train_loss_epoch=1.400]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 87.48it/s, v_num=35, train_loss_step=1.280, train_loss_epoch=1.400]Epoch 2: 100%|██████████| 1/1 [00:00<00:00, 84.41it/s, v_num=35, train_loss_step=1.280, train_loss_epoch=1.280]Epoch 2:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.280, train_loss_epoch=1.280]        Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.280, train_loss_epoch=1.280]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 76.49it/s, v_num=35, train_loss_step=1.280, train_loss_epoch=1.280]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 74.51it/s, v_num=35, train_loss_step=1.290, train_loss_epoch=1.280]Epoch 3: 100%|██████████| 1/1 [00:00<00:00, 72.22it/s, v_num=35, train_loss_step=1.290, train_loss_epoch=1.290]Epoch 3:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.290, train_loss_epoch=1.290]        Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.290, train_loss_epoch=1.290]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 18.01it/s, v_num=35, train_loss_step=1.290, train_loss_epoch=1.290]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 17.87it/s, v_num=35, train_loss_step=1.300, train_loss_epoch=1.290]Epoch 4: 100%|██████████| 1/1 [00:00<00:00, 17.73it/s, v_num=35, train_loss_step=1.300, train_loss_epoch=1.300]Epoch 4:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.300, train_loss_epoch=1.300]        Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.300, train_loss_epoch=1.300]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 81.99it/s, v_num=35, train_loss_step=1.300, train_loss_epoch=1.300]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 79.84it/s, v_num=35, train_loss_step=1.270, train_loss_epoch=1.300]Epoch 5: 100%|██████████| 1/1 [00:00<00:00, 77.25it/s, v_num=35, train_loss_step=1.270, train_loss_epoch=1.270]Epoch 5:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.270, train_loss_epoch=1.270]        Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.270, train_loss_epoch=1.270]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 86.91it/s, v_num=35, train_loss_step=1.270, train_loss_epoch=1.270]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 84.56it/s, v_num=35, train_loss_step=1.220, train_loss_epoch=1.270]Epoch 6: 100%|██████████| 1/1 [00:00<00:00, 81.58it/s, v_num=35, train_loss_step=1.220, train_loss_epoch=1.220]Epoch 6:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.220, train_loss_epoch=1.220]        Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.220, train_loss_epoch=1.220]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 98.22it/s, v_num=35, train_loss_step=1.220, train_loss_epoch=1.220]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 95.39it/s, v_num=35, train_loss_step=1.060, train_loss_epoch=1.220]Epoch 7: 100%|██████████| 1/1 [00:00<00:00, 92.48it/s, v_num=35, train_loss_step=1.060, train_loss_epoch=1.060]Epoch 7:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.060, train_loss_epoch=1.060]        Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.060, train_loss_epoch=1.060]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 75.67it/s, v_num=35, train_loss_step=1.060, train_loss_epoch=1.060]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 73.84it/s, v_num=35, train_loss_step=1.150, train_loss_epoch=1.060]Epoch 8: 100%|██████████| 1/1 [00:00<00:00, 71.59it/s, v_num=35, train_loss_step=1.150, train_loss_epoch=1.150]Epoch 8:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.150, train_loss_epoch=1.150]        Epoch 9:   0%|          | 0/1 [00:00<?, ?it/s, v_num=35, train_loss_step=1.150, train_loss_epoch=1.150]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 17.39it/s, v_num=35, train_loss_step=1.150, train_loss_epoch=1.150]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 17.26it/s, v_num=35, train_loss_step=1.090, train_loss_epoch=1.150]
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation: |          | 0/? [00:00<?, ?it/s][A
#> Validation DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s][A
#> Validation DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 1391.15it/s][A
#>                                                                         [AEpoch 9: 100%|██████████| 1/1 [00:00<00:00, 16.32it/s, v_num=35, train_loss_step=1.090, train_loss_epoch=1.150]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 16.19it/s, v_num=35, train_loss_step=1.090, train_loss_epoch=1.090]Epoch 9: 100%|██████████| 1/1 [00:00<00:00, 14.13it/s, v_num=35, train_loss_step=1.090, train_loss_epoch=1.090]

# rmok predict
rmok_point_predictions <- 
rmok_fit |>
 modeltime_table() |>
  modeltime_forecast(actual_data = training, new_data = testing) 
#> Predicting: |          | 0/? [00:00<?, ?it/s]Predicting: |          | 0/? [00:00<?, ?it/s]Predicting DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 552.17it/s]Predicting DataLoader 0: 100%|██████████| 1/1 [00:00<00:00, 468.79it/s]

rmok_point_predictions |>
  plot_modeltime_forecast(.conf_interval_show = FALSE, .interactive = FALSE)
```

<img src="man/figures/README-unnamed-chunk-5-1.png" alt="" width="100%" />
