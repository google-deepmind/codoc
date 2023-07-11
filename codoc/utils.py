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

"""This module includes miscallenaous utility functions for CoDoC."""

import json
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
from sklearn import metrics as skmetrics


_DATA_PATHS = {
    "uk_mammo_arbitration": "data/uk_mammo/arbitration",
    "uk_mammo_single": "data/uk_mammo/single_reader",
    "us_mammo_2": "data/us_mammo_2",
}


def _build_abs_path(path: str) -> str:
  """Builds an absolute path from project relative path."""
  project_path = os.path.dirname(os.path.dirname(__file__))
  return os.path.join(project_path, path)


# Mapping the existing datasets to high and low dataset regimes.
data_regime = lambda x: "high" if "uk_mammo" in x else "low"


plt.rc("font", size=10)
plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("axes", titlesize=15.5)


def thresholds(a_z: jnp.ndarray) -> Sequence[Sequence[float]]:
  """Extracts the thresholds for the deferral regions given a Defer(z) function.

  Args:
    a_z: A binary vector corresponding to the discretized Defer(z) function.

  Returns:
    A list of lists including the lower and upper bounds for the possibly
    multiple deferral regions.
  """
  changepoints = list(np.arange(len(a_z) - 1)[np.diff(a_z) != 0] + 1)
  if a_z[0] == 1:
    changepoints = [0] + changepoints
  if a_z[-1] == 1:
    changepoints = changepoints + [len(a_z)]
  changepoints = np.array(changepoints) / len(a_z)
  return [
      [changepoints[i * 2], changepoints[i * 2 + 1]]
      for i in range(len(changepoints) // 2)
  ]


def plot_advantage_z(
    phi: jnp.ndarray,
    tau: int,
    a_z: Optional[jnp.ndarray] = None,
    title: str = "",
):
  r"""Plots Advantage(z) and the associated deferral regions based on Defer(z).

  Args:
    phi: The discretized Advantage(z) function.
    tau: The bin whose lower edge serves as the operating point such that the
      main paper's $\theta = \tau/T$.
    a_z: Discretized Defer(z) function.
    title: Desired title for the plot.
  """
  num_bins = len(phi)
  fig, ax = plt.subplots(figsize=(10, 2.4))
  ax.set_xticks(jnp.linspace(0, num_bins, 6))
  ax.set_xticklabels(jnp.linspace(0, 1, 6))
  ax.plot(jnp.arange(num_bins) + 0.5, phi)
  ax.plot(
      jnp.arange(num_bins) + 0.5,
      jnp.zeros(num_bins),
      linewidth=0.5,
      color="gray",
  )
  if title:
    ax.set_title(title)
  ax.axvline(tau, color="orange")
  if a_z is not None:
    for i in range(num_bins):
      if a_z[i] == 1:
        color = "white"
      else:
        color = "green" if i < tau else "red"
      ax.axvspan(i, (i + 1), alpha=0.25, color=color, linewidth=0)
  ax.set_xlim(0, num_bins)
  ax.set_ylabel(r"$Advantage(z)$")
  ax.set_xlabel("Predictive AI confidence score ($z$)")
  ax.set_yticklabels([])
  fig.tight_layout()


def load_hyperparameters() -> Dict[str, Any]:
  """Loads the hyperparameters stored in the related file."""
  with open(
      _build_abs_path("data/hyperparameters.json"),
      "r",
  ) as f:
    return json.load(f)


def ppv(ground_truth: pd.Series, scores: pd.Series) -> float:
  """Computes the PPV statistic for a given ground truth and responses.

  Args:
    ground_truth: Series that includes ground truth for all cases.
    scores: Series that includes model or clinican responses for all cases.

  Returns:
    PPV statistic.
  """
  _, fp, _, tp = skmetrics.confusion_matrix(ground_truth, scores).ravel()
  return tp / (tp + fp)


def sens(ground_truth: pd.Series, scores: pd.Series) -> float:
  """Computes the sensitivity for a given ground truth and responses.

  Args:
    ground_truth: Series that includes ground truth for all cases.
    scores: Series that includes model or clinican responses for all cases.

  Returns:
    Sensitivity statistic.
  """
  _, _, fn, tp = skmetrics.confusion_matrix(ground_truth, scores).ravel()
  return tp / (tp + fn)


def spec(ground_truth: pd.Series, scores: pd.Series) -> float:
  """Computes the specificity for a given ground truth and responses.

  Args:
    ground_truth: Series that includes ground truth for all cases.
    scores: Series that includes model or clinican responses for all cases.

  Returns:
    Specifictiy statistic.
  """
  tn, fp, _, _ = skmetrics.confusion_matrix(ground_truth, scores).ravel()
  return tn / (tn + fp)


def model_op_getter(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.Series, pd.Series], float],
    target_metric: Optional[float],
) -> float:
  """Returns model operating point matching the reader at the given metric.

  Args:
    df: The dataset split on which the operating points will be computed.
      Includes the columns for ground truth ("y_true"), ML model predictions
      ("y_model"), and clinical workflow opinions ("reader_score").
    metric_fn: The function that computes the desired metric given ground truth
      and scores.
    target_metric: If provided, this value is used instead of the output of
      metric_fn.

  Returns:
    The computed model operating point.
  """
  if not target_metric:
    # Set target to reader metric
    target_metric = metric_fn(df.y_true, df["reader_score"])

  def opt(x):
    scores = (df.y_model > x).astype(int)
    return metric_fn(df.y_true, scores) - target_metric

  return scipy.optimize.bisect(opt, 0, 1)


def load_data(
    experiment_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Loads tune, validation, and test splits given an experiment name.

  Args:
    experiment_name: The string that corresponds to the specific experiment
      described in the main paper. If not recognized, it is used as the folder
      name for a custom dataset, and data splits are expected under
      data/{experiment_name}.

  Returns:
    Data sets that include tune, validation, and test splits.
  """
  if experiment_name in _DATA_PATHS.keys():
    data_folder = _DATA_PATHS[experiment_name]
  else:
    data_folder = experiment_name
  print(_build_abs_path(f"{data_folder}/tune.csv"))
  df_tune = pd.read_csv(_build_abs_path(f"{data_folder}/tune.csv"))
  df_val = pd.read_csv(_build_abs_path(f"{data_folder}/val.csv"))
  df_test = pd.read_csv(_build_abs_path(f"{data_folder}/test.csv"))
  return df_tune, df_val, df_test
