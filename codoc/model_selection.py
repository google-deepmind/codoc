# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Includes functions for parameter sweep and model selection.

This module provides functions for hyperparameter sweep and model selection.
The former admits a set of hyperparameter ranges, and based on the provided
data, estimates CoDoC models for all hyperparameter combinations. The latter
allows selecting model for a desired statistic, i.e. sensitivity and
specificity. Please consult the main paper for the definition of
hyperparameters and model selection options.
"""

import copy
import functools
from typing import Any, Mapping, Sequence, Union

import jax
import joblib
import numpy as np
import pandas as pd

from codoc import deferral_models
from codoc import density_estimation
from codoc import evaluation
from codoc import utils


# Setting N_JOBS > 1 parallelizes the experiments using joblib.Parallel.
_N_JOBS = 1

_KEYS = [
    "params",
    "a_z",
    "sens_tune",
    "spec_tune",
    "sens_val",
    "spec_val",
    "comp_sens_tune",
    "comp_spec_tune",
    "comp_sens_val",
    "comp_spec_val",
    "deferral_ratio_tune",
    "deferral_ratio_val",
]


def _filtering_idx_wrt_baseline(
    baseline_model: str,
    results: Mapping[str, Any],
    non_inf_coef_spec: float,
    non_inf_coef_sens: float,
) -> np.ndarray:
  """Obtains a bool filtering index to drop models that score below baseline.

  Args:
    baseline_model: Baseline model for model selection.
    results: Results dictionary as produced by the parameter_sweep function.
    non_inf_coef_spec: The CoDoC models that have specificity below
      non_inf_coef_spec * baseline_spec will be ignored.
    non_inf_coef_sens: The CoDoC models that have sensitivity below
      non_inf_coef_sens * baseline_sens will be ignored.

  Returns:
    A boolean vector which includes the value False for models that score
        sufficiently worse than the baseline model, and True otherwise.
  """
  if baseline_model == "reader":
    baseline_column = "reader"
  elif baseline_model == "avg_sens_spec_v2":
    baseline_column = "avg_model"
  else:
    raise NotImplementedError(f"Strategy {baseline_model} not implemented")

  idx_tune = (
      results["comp_spec_tune"]
      >= ((results[f"{baseline_column}_spec_tune"]) * non_inf_coef_spec)
  ) & (
      results["comp_sens_tune"]
      >= (results[f"{baseline_column}_sens_tune"] * non_inf_coef_sens)
  )
  idx_val = (
      results["comp_spec_val"]
      >= ((results[f"{baseline_column}_spec_val"]) * non_inf_coef_spec)
  ) & (
      results["comp_sens_val"]
      >= (results[f"{baseline_column}_sens_val"] * non_inf_coef_sens)
  )

  return idx_val & idx_tune


def parameter_sweep(
    df_tune: pd.DataFrame,
    df_val: pd.DataFrame,
    sweep_params: Mapping[str, Sequence[Union[int, float, None]]],
    deferral_ratio: float = 0.5,
) -> Mapping[str, Any]:
  """Conducts parameter sweep over the provided hyperparameter ranges for CoDoC.

  This function conducts a parameter sweep for a given dataset, and provides
  performance estimates and other auxiliary statistics for all computed models.
  Before returning results it drops models that have substantially inferior
  performance to baselines or have a deferral ratio above the value provided
  to the function in order to save memory.

  Args:
    df_tune: DataFrame object that contains the data for the tune set. Includes
      the columns for ground truth ("y_true"), ML model predictions ("y_model"),
      and clinical workflow opinions ("reader_score").
    df_val: DataFrame object that contains the data for the validation set.
      Includes the columns for ground truth ("y_true"), ML model predictions
      ("y_model"), and clinical workflow opinions ("reader_score").
    sweep_params: Includes the hyperparameter ranges for which CoDoC models will
      be estimated.
    deferral_ratio: The maximum ratio of cases in [0, 1] which can be deferred
      to the clinical workflow.

  Returns:
    A dictionary that includes hyperparameters, performance estimates, and other
    auxiliary statistics for each hyperparameter combination that has
    competitive performance with the baselines and defers to the clinical
    workflow for an acceptable proportion of cases.
  """
  num_bins_range, pseudocounts_range, smoothing_bandwidth_range, lam_range = (
      sweep_params["num_bins_range"],
      sweep_params["pseudocounts_range"],
      sweep_params["smoothing_bandwidth_range"],
      sweep_params["lam_range"],
  )
  # Results are stored as a dictionary of lists. Each index is occupied by the
  # statistics of a single model.
  results = {key: [] for key in _KEYS}

  # Obtaining sens and spec values for reader and baseline model.
  results["reader_sens_tune"], results["reader_spec_tune"] = (
      evaluation.evaluate_baseline_reader(df_tune)
  )
  results["reader_sens_val"], results["reader_spec_val"] = (
      evaluation.evaluate_baseline_reader(df_val)
  )
  results["avg_model_sens_tune"], results["avg_model_spec_tune"] = (
      evaluation.evaluate_baseline_model(df_tune)
  )
  results["avg_model_sens_val"], results["avg_model_spec_val"] = (
      evaluation.evaluate_baseline_model(df_val)
  )
  print("Started hyperparameter sweep.")
  for num_bins in num_bins_range:
    if num_bins % 10 == 0:
      print(f"Conducting experiments for T = {num_bins}.")
    partialed_compute_p_z_h_given_y = functools.partial(
        density_estimation.compute_p_z_h_given_y,
        num_bins=num_bins,
        pseudocounts=0,
        smoothing_bandwidth=None,
    )
    p_z_h_given_y_tune = partialed_compute_p_z_h_given_y(df_tune)
    p_z_h_given_y_val = partialed_compute_p_z_h_given_y(df_val)

    count_z_tune, _ = np.histogram(
        df_tune["y_model"].values, bins=num_bins, range=(0, 1)
    )
    count_z_val, _ = np.histogram(
        df_val["y_model"].values, bins=num_bins, range=(0, 1)
    )

    num_mult = (
        len(pseudocounts_range)
        * len(smoothing_bandwidth_range)
        * len(lam_range)
    )
    results["sens_tune"].extend(
        density_estimation.sens(p_z_h_given_y_tune).tolist() * num_mult
    )
    results["spec_tune"].extend(
        density_estimation.spec(p_z_h_given_y_tune).tolist() * num_mult
    )
    results["sens_val"].extend(
        density_estimation.sens(p_z_h_given_y_val).tolist() * num_mult
    )
    results["spec_val"].extend(
        density_estimation.spec(p_z_h_given_y_val).tolist() * num_mult
    )

    for pseudocounts in pseudocounts_range:
      for smoothing_bandwidth in smoothing_bandwidth_range:
        p_z_h_given_y_tune_smoothed = density_estimation.compute_p_z_h_given_y(
            df_tune,
            num_bins,
            pseudocounts,
            smoothing_bandwidth,
        )

        partialed_lam_outputs = jax.tree_util.Partial(
            deferral_models.lam_outputs,
            p_z_h_given_y_tune_smoothed=p_z_h_given_y_tune_smoothed,
            num_bins=num_bins,
            count_z_tune=count_z_tune,
            count_z_val=count_z_val,
            p_z_h_given_y_tune=p_z_h_given_y_tune,
            p_z_h_given_y_val=p_z_h_given_y_val,
        )

        computed_lam_outputs = joblib.Parallel(n_jobs=_N_JOBS)(
            joblib.delayed(partialed_lam_outputs)(**{"lam": lam})
            for lam in lam_range
        )

        for lam_i, lam in enumerate(lam_range):
          # Under this innermost loop, all operations are done for all
          # taus in a parallelized fashion (or with list comprehension).
          for tau_i in range(num_bins):
            results["params"].append(
                dict(
                    lam=lam,
                    num_bins=num_bins,
                    tau=tau_i,
                    pseudocounts=pseudocounts,
                    smoothing_bandwidth=smoothing_bandwidth,
                )
            )

          for key in [
              "a_z",
              "deferral_ratio_tune",
              "deferral_ratio_val",
              "comp_sens_tune",
              "comp_spec_tune",
              "comp_sens_val",
              "comp_spec_val",
          ]:
            results[key].extend(computed_lam_outputs[lam_i][key])

  for key in results.keys():
    if key not in ["params", "a_z"]:
      results[key] = np.array(results[key])

  results["num_a_z_transitions"] = np.array(
      [(np.diff(a_z_i) != 0).sum() for a_z_i in results["a_z"]]
  )

  idx_all_models = (
      (results["deferral_ratio_tune"] < deferral_ratio)
      & (results["deferral_ratio_val"] < deferral_ratio)
      & (results["comp_sens_tune"] > 0.85 * results["reader_sens_tune"])
      & (results["comp_spec_tune"] > 0.85 * results["reader_spec_tune"])
      & (results["num_a_z_transitions"] > 0)
  )
  for key in results.keys():
    if key in ["params", "a_z"]:
      results[key] = [r for r, f in zip(results[key], idx_all_models) if f]
    elif "pareto" not in key and "reader" not in key and "avg" not in key:
      results[key] = results[key][idx_all_models]

  results["sweep_params"] = sweep_params
  print("Completed hyperparameter sweep successfully.")
  return results


def select_model(
    results: Mapping[str, Any],
    ordering_variable: str = "comp_spec_val",
    drop_percent: float = 0.01,
    a_z_start: int = 2,
    non_inf_coef_sens: float = 0.99,
    non_inf_coef_spec: float = 0.99,
    experiment_name: str = "us_mammo_2",
    num_viable_models_threshold: int = 10,
    absolute_max_num_a_z_transitions: int = 8,
):
  """Selects model among provided CoDoC models with the provided hyperparams.

  See the main paper for detailed explanations of model selection options.

  Args:
    results: Results dictionary as produced by the parameter_sweep function.
    ordering_variable: The statistic according to which the models will be
      ordered to select from among.
    drop_percent: The top percent of models to be ignored to avoid overfitting
      on a small validation set.
    a_z_start: The minimum number of transitions in Defer(z) to be included in
      models to be considered.
    non_inf_coef_sens: The CoDoC models that have sensitivity below
      non_inf_coef_sens * baseline_sens will be ignored.
    non_inf_coef_spec: The CoDoC models that have specificity below
      non_inf_coef_spec * baseline_spec will be ignored.
    experiment_name: The experiment name as defined in the main notebook file.
    num_viable_models_threshold: If the number of available models fall below
      this value, the number of allowed transitions in the Defer(z) will be
      increased to include CoDoC models with more deferral regions.
    absolute_max_num_a_z_transitions: Absolute maximum of allowed transitions in
      the deferral function Defer(z), beyond which the model selection will not
      progress.

  Returns:
    The updated results dictionary with the details of the selected model
    included.
  """
  # Make copy of the results.
  results = copy.deepcopy(results)

  baseline_model = (
      "reader"
      if experiment_name in ["uk_mammo_arbitration", "us_mammo_2"]
      else "avg_sens_spec_v2"
  )
  idx_tune_val = _filtering_idx_wrt_baseline(
      baseline_model,
      results,
      non_inf_coef_spec,
      non_inf_coef_sens,
  )
  # Limit the models to those that defer for less than .5 of tune and val
  # samples.
  idx_dr = (results["deferral_ratio_tune"] < 0.5) & (
      results["deferral_ratio_val"] < 0.5
  )
  # Getting the number of a_z transitions for each hyperparameter combination.
  num_a_z_transitions = results["num_a_z_transitions"]
  # We initially will allow max. two Defer(z) transitions, will increase
  # this if we cannot find enough models.
  max_allowed_num_a_z_transitions = a_z_start
  idx_a_z = num_a_z_transitions <= max_allowed_num_a_z_transitions
  # Indices of the models that are viable for selection.
  idx = np.arange(len(results["comp_spec_val"]), dtype=int)[
      idx_tune_val & idx_a_z & idx_dr
  ]  # idx_y
  print("Started model selection.")
  # If no viable model exists, increase the max number of a_z transitions
  # until a model is found or a hard max. is reached.
  while len(idx) < num_viable_models_threshold:
    max_allowed_num_a_z_transitions += 2
    if max_allowed_num_a_z_transitions > absolute_max_num_a_z_transitions:
      break
    print(
        "Warning: Max allowed Defer(z) transitions are",
        max_allowed_num_a_z_transitions,
    )
    idx_a_z = num_a_z_transitions <= max_allowed_num_a_z_transitions
    idx = np.arange(len(results["comp_spec_val"]), dtype=int)[
        idx_tune_val & idx_a_z & idx_dr
    ]
  # If we still have not found any viable model, conclude unsuccessfully.
  if not np.any(idx):
    print("No models found!")
    results["val_idx"] = np.nan
    results["operating_point"] = np.nan
    results["thresholds"] = np.nan
  else:
    num_selected_model = int(len(idx) * drop_percent)
    # Among the viable models, select the one with max. comp-spec in val set.
    i = 0
    val_idx = -1
    for j in np.flip(np.argsort(results[ordering_variable])):
      val_idx = j
      if j in idx:
        i += 1
        if i >= num_selected_model:
          break
    print(f"Completed model selection: Model idx {val_idx} selected.")
    results["val_idx"] = results["model_idx"] = val_idx
    results["operating_point"] = (
        results["params"][val_idx]["tau"]
        / results["params"][val_idx]["num_bins"]
    )
    results["thresholds"] = utils.thresholds(results["a_z"][val_idx])
  return results
