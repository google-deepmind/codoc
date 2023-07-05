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

"""Includes functions for density and deferral function estimation for CoDoC.

This module contains the functionality that allows estimation of the conditional
density central to CoDoC, $P(z, h | y)$, as well as the Advantage(z) and
Defer(z) functions thereby implied.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy


def compute_p_z_h_given_y(
    df: pd.DataFrame,
    num_bins: int,
    pseudocounts: float,
    smoothing_bandwidth: Optional[float] = None,
) -> jnp.ndarray:
  """Estimates the probabilities for P(z, h | y) for a given dataset.

  Args:
    df: The dataset split on which the density estimations will be based.
      Includes the columns for ground truth ("y_true"), ML model predictions
      ("y_model"), and clinical workflow opinions ("reader_score").
    num_bins: Number of bins to be used when discretizing the model outputs. $T$
      from the main paper.
    pseudocounts: Number of pseudo-observations to add to each bin. $kappa$ from
      the main paper.
    smoothing_bandwidth: The bandwidth of the Gaussian convolution to be applied
      to original probabilities. $sigma$ from the main paper.

  Returns:
    Discretized estimate of the joint probability distribution conditioned on
      y_true, given the density estimation hyperparameters.
  """
  smoothing_bandwidth = (
      None if smoothing_bandwidth == 0 else smoothing_bandwidth
  )
  counts_given_z_h_y = np.zeros((num_bins, 2, 2))
  for h in range(2):
    for y in range(2):
      z_given_h_y = df.query(f"reader_score == {h} and y_true == {y}")[
          "y_model"
      ].values
      counts_given_z_h_y[:, h, y], _ = np.histogram(
          z_given_h_y, bins=num_bins, range=(0, 1)
      )
  counts_given_z_h_y += pseudocounts

  if smoothing_bandwidth is not None:
    counts_given_z_h_y = scipy.ndimage.gaussian_filter1d(
        counts_given_z_h_y, smoothing_bandwidth**2, axis=0
    )
  return jnp.array(
      counts_given_z_h_y
      / np.sum(counts_given_z_h_y, axis=(0, 1), keepdims=True)
  )


@jax.jit
def sens(p_z_h_given_y: jnp.ndarray) -> jnp.ndarray:
  """Computes sensitivity estimates of the ML model for a given P(z, h | y).

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.

  Returns:
    Sensitivity values for all potential operating points with the
      discretization implied by p_z_h_given_y.
  """
  p_z_given_t1 = p_z_h_given_y.sum(1)[:, 1]
  return jnp.array(
      [jnp.sum(p_z_given_t1[tau:]) for tau in range(p_z_h_given_y.shape[0])]
  )


@jax.jit
def spec(p_z_h_given_y: jnp.ndarray) -> jnp.ndarray:
  """Computes specificity estimates of the ML model for a given P(z, h | y).

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.

  Returns:
    Specificity values for all potential operating points with the
      discretization implied by p_z_h_given_y.
  """
  p_z_given_t0 = p_z_h_given_y.sum(1)[:, 0]
  return jnp.array(
      [jnp.sum(p_z_given_t0[:tau]) for tau in range(p_z_h_given_y.shape[0])]
  )


@jax.jit
def _phi_0(p_z_h_given_y: jnp.ndarray, tau: int) -> jnp.ndarray:
  r"""Computes the terms $p(z,h=0|y=0)$ and $-p(z,h=1|y=0)$ for an op. point.

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose lower edge serves as the operating point such that the
      main paper's $\theta = \tau/T$.

  Returns:
    An array with $p(z,h=0|y=0)1(z>=\theta) - p(z,h=1|y=0)1(z<\theta)$ for all
      operating point $\theta$'s that correspond to bin edges, with $1(\cdot)$
      evaluating to 1 if the statement inside is correct and 0 otherwise.
  """
  return jnp.array(
      [
          jnp.array(z >= tau, int) * p_z_h_given_y[z, 0, 0]
          - jnp.array(z < tau, int) * p_z_h_given_y[z, 1, 0]
          for z in range(p_z_h_given_y.shape[0])
      ]
  )


@jax.jit
def _phi_1(p_z_h_given_y: jnp.ndarray, tau: int) -> jnp.ndarray:
  r"""Computes the terms $p(z,h=1|y=1)$ and $-p(z,h=0|y=1)$ for an op. point.

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose lower edge serves as the operating point such that the
      main paper's $\theta = \tau/T$.

  Returns:
    An array with $p(z,h=1|y=1)1(z<\theta)$-p(z,h=0|y=1)1(z>=\theta)$ for all
      operating point $\theta$'s that correspond to bin edges, with $1(\cdot)$
      evaluating to 1 if the statement inside is correct and 0 otherwise.
  """
  return jnp.array(
      [
          jnp.array(z < tau, int) * p_z_h_given_y[z, 1, 1]
          - jnp.array(z >= tau, int) * p_z_h_given_y[z, 0, 1]
          for z in range(p_z_h_given_y.shape[0])
      ]
  )


def _phi_single(
    p_z_h_given_y: jnp.ndarray, tau: int, lam: float
) -> jnp.ndarray:
  r"""Computes the Advantage(z) for an operating point and trade-off param.

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose lower edge serves as the operating point such that the
      main paper's $\theta = \tau/T$.
    lam: The trade-off hyperparameter for sensitivity-specificity. $\lambda$
      from the main paper.

  Returns:
    The discretized advantage function, Advantage(z), from the main paper.
  """
  return (1 - lam) * _phi_0(p_z_h_given_y, tau) + lam * _phi_1(
      p_z_h_given_y, tau
  )


phi = jax.jit(jax.vmap(_phi_single), in_axes=[None, 0, None])


def _compute_a_z(
    p_z_h_given_y: jnp.ndarray, tau: int, lam: float
) -> jnp.ndarray:
  r"""Computes the Defer(z) for an operating point and trade-off parameter.

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose edge serves as the operating point such that $\theta =
      \tau/T$.
    lam: The trade-off hyperparameter for sensitivity-specificity. $\lambda$
      from the main paper.

  Returns:
    The discretized deferral function Defer(z).
  """
  a_z_bool = _phi_single(p_z_h_given_y, tau, lam) >= 0
  return a_z_bool.astype(int)


def _comp_sens_single(
    p_z_h_given_y: jnp.ndarray,
    tau: int,
    a_z: jnp.ndarray,
) -> jnp.ndarray:
  r"""Computes sensitivity estimates of a CoDoC model for a given P(z, h | y).

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose edge serves as the operating point such that $\theta =
      \tau/T$.
    a_z: The deferral function from the main paper for each bin, i.e. Defer(z).

  Returns:
    Sensitivity estimates of CoDoC for the given operating point.
  """
  return sens(p_z_h_given_y)[tau] + jnp.sum(a_z * _phi_1(p_z_h_given_y, tau))


comp_sens = jax.jit(jax.vmap(_comp_sens_single, in_axes=[None, 0, 0]))


def _comp_spec_single(
    p_z_h_given_y: jnp.ndarray,
    tau: int,
    a_z: jnp.ndarray,
) -> jnp.ndarray:
  r"""Computes specificity estimates of a CoDoC model for a given P(z, h | y).

  Args:
    p_z_h_given_y: Discretized joint probability distribution conditioned on
      y_true.
    tau: The bin whose edge serves as the operating point such that $\theta =
      \tau/T$.
    a_z: The deferral function from the main paper for each bin, i.e. Defer(z).

  Returns:
    Specificity estimates of CoDoC for the given operating point.
  """
  return spec(p_z_h_given_y)[tau] + jnp.sum(a_z * _phi_0(p_z_h_given_y, tau))


comp_spec = jax.jit(jax.vmap(_comp_spec_single, in_axes=[None, 0, 0]))
