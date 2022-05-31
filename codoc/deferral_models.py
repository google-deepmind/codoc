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

"""Includes functions to estimate a CoDoC model given data and hyperparameters.

This module contains the functionality that allows user to estimate a single
CoDoC model, as well as its evaluation on the tune and validation splits. The
user is expected to provide the hyperparameters for the model estimation.
"""


from typing import Any, Mapping, Sequence, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from codoc import density_estimation
from codoc import utils


def lam_outputs(
    lam: jnp.ndarray,
    p_z_h_given_y_tune_smoothed: jnp.ndarray,
    num_bins: int,
    count_z_tune: jnp.ndarray,
    count_z_val: jnp.ndarray,
    p_z_h_given_y_tune: jnp.ndarray,
    p_z_h_given_y_val: jnp.ndarray,
) -> Mapping[str, Union[Sequence[float], Sequence[Sequence[float]]]]:
  r"""Returns CoDoC model estimates for a single $\lambda$ value.

  This function obtains the results of CoDoC model estimation for a single
  $\lambda$:
  these include the discretized Defer(z) function, its performance evaluation on
  the tune and validation splits, and deferral ratios for tune and validation
  set, for all possible $\tau$s. If desired, the discretized Advantage(z)
  functions are also returned.

  Args:
      lam: The trade-off hyperparameter for sensitivity-specificity. $\lambda$
        from the main paper.
      p_z_h_given_y_tune_smoothed: Smoothed estimate of the discretized joint
        probability distribution conditioned on y_true.
      num_bins: Number of bins to be used when discretizing the model outputs.
        $T$ from the main paper.
      count_z_tune: The number of observations that fall in each bin after
        discretization in the tune set.
      count_z_val: The number of observations that fall in each bin after
        discretization in the validation set.
      p_z_h_given_y_tune: Estimate of the discretized joint probability
        distribution conditioned on y_true for the tune set.
      p_z_h_given_y_val: Estimate of the discretized joint probability
        distribution conditioned on y_true for the validation set.

  Returns:
      A dictionary of lists or lists of lists that include the results for all
      $\tau$s given
      the value of $\lambda$.
  """
  phi = density_estimation.phi(
      p_z_h_given_y_tune_smoothed, jnp.arange(num_bins), lam
  )
  a_z = phi >= 0
  results = {
      "a_z": a_z,
      "deferral_ratio_tune": (a_z * count_z_tune[np.newaxis, :]).sum(
          1
      ) / count_z_tune.sum(),
      "deferral_ratio_val": (a_z * count_z_val[np.newaxis, :]).sum(
          1
      ) / count_z_val.sum(),
      "comp_sens_tune": density_estimation.comp_sens(
          p_z_h_given_y_tune, jnp.arange(num_bins), a_z
      ),
      "comp_spec_tune": density_estimation.comp_spec(
          p_z_h_given_y_tune, jnp.arange(num_bins), a_z
      ),
      "comp_sens_val": density_estimation.comp_sens(
          p_z_h_given_y_val, jnp.arange(num_bins), a_z
      ),
      "comp_spec_val": density_estimation.comp_spec(
          p_z_h_given_y_val, jnp.arange(num_bins), a_z
      ),
      "phis": phi,
  }
  results = {key: value.tolist() for key, value in results.items()}
  return results


def estimate_model(
    df_tune: pd.DataFrame,
    df_val: pd.DataFrame,
    tau: int,
    num_bins: int,
    pseudocounts: Union[float, int],
    smoothing_bandwidth: Union[float, int, None],
    lam: float,
) -> Mapping[str, Any]:
  r"""Estimates and evaluates a CoDoC model, given data and hyperparameters.

  This function estimates a CoDoC model, obtains the corresponding deferral
  thresholds, evaluates it on training and validation sets, and obtains other
  statistics such as deferral ratio.

  Args:
    df_tune: DataFrame object that contains the data for the tune set. Includes
      the columns for ground truth ("y_true"), ML model predictions ("y_model"),
      and clinical workflow opinions ("reader_score").
    df_val: DataFrame object that contains the data for the validation set.
      Includes the columns for ground truth ("y_true"), ML model predictions
      ("y_model"), and clinical workflow opinions ("reader_score").
    tau: The bin whose lower edge serves as the operating point such that the
      main paper's $\theta = \tau/T$.
    num_bins: Number of bins to be used when discretizing the model outputs. $T$
      from the main paper.
    pseudocounts: Pseudocount value to be added for the histogram bins. $\kappa$
      from the main paper.
    smoothing_bandwidth: Smoothing bandwidth value to be added for the histogram
      bins. $\sigma$ from the main paper.
    lam: The trade-off hyperparameter for sensitivity-specificity. $\lambda$
      from the main paper.

  Returns:
      A dictionary of the estimated model parameters and statistics.
  """
  pseudocounts = (
      pseudocounts[0] if isinstance(pseudocounts, list) else pseudocounts
  )
  smoothing_bandwidth = (
      smoothing_bandwidth[0]
      if isinstance(smoothing_bandwidth, list)
      else smoothing_bandwidth
  )
  smoothing_bandwidth = (
      None if smoothing_bandwidth == 0 else smoothing_bandwidth
  )
  p_z_h_given_y_tune = density_estimation.compute_p_z_h_given_y(
      df_tune,
      num_bins,
      pseudocounts=0,
      smoothing_bandwidth=None,
  )
  p_z_h_given_y_val = density_estimation.compute_p_z_h_given_y(
      df_val,
      num_bins,
      pseudocounts=0,
      smoothing_bandwidth=None,
  )

  count_z_tune, _ = np.histogram(
      df_tune["y_model"].values, bins=num_bins, range=(0, 1)
  )
  count_z_val, _ = np.histogram(
      df_val["y_model"].values, bins=num_bins, range=(0, 1)
  )

  p_z_h_given_y_tune_smoothed = density_estimation.compute_p_z_h_given_y(
      df_tune,
      num_bins,
      pseudocounts,
      smoothing_bandwidth,
  )

  results = lam_outputs(
      lam=lam,
      p_z_h_given_y_tune_smoothed=p_z_h_given_y_tune_smoothed,
      num_bins=num_bins,
      count_z_tune=count_z_tune,
      count_z_val=count_z_val,
      p_z_h_given_y_tune=p_z_h_given_y_tune,
      p_z_h_given_y_val=p_z_h_given_y_val,
  )

  results["params"] = []
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

  # Obtaining results for all taus.
  results["sens_tune"] = density_estimation.sens(p_z_h_given_y_tune).tolist()
  results["spec_tune"] = density_estimation.spec(p_z_h_given_y_tune).tolist()
  results["sens_val"] = density_estimation.sens(p_z_h_given_y_val).tolist()
  results["spec_val"] = density_estimation.spec(p_z_h_given_y_val).tolist()

  # Obtaining results for only the chosen tau.
  results = {key: value[tau] for key, value in results.items()}
  results["operating_point"] = tau / results["params"]["num_bins"]
  results["thresholds"] = utils.thresholds(results["a_z"])
  return results
