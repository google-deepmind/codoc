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

"""Evaluation functions for CoDoC, baseline ML models, and clinical workflow.

This module includes functions that estimates sensitivity and specificity for
CoDoC models as well as the baselines of ML model (predictive AI) and clinician
opinion.
"""


from typing import Sequence, Tuple

import pandas as pd
from sklearn import metrics as skmetrics

from codoc import utils


def evaluate_codoc_model(
    data: pd.DataFrame,
    operating_point: float,
    thresholds: Sequence[Sequence[float]],
) -> Tuple[float, float]:
  r"""Evaluates CoDoC model with a dataset, ML predictions, and CoDoC thresholds.

  Args:
    data: The dataframe that holds the dataset on which evaluation will be
      based. Includes the columns for ground truth ("y_true"), ML model
      predictions ("y_model"), and clinical workflow opinions ("reader_score").
    operating_point: The operating point of the ML model, $\theta$ from the main
      paper.
    thresholds: A list of lists that includes all regions in $[0,1]$ for which
      the CoDoC model will defer to the expert.

  Returns:
    The sensitivity and specificity estimates of the CoDoC model.
  """
  y_pred = (data.y_model > operating_point).astype(float)
  for lower_t, upper_t in thresholds:
    y_pred[(data.y_model > lower_t) & (data.y_model < upper_t)] = data.loc[
        (data.y_model > lower_t) & (data.y_model < upper_t), "reader_score"
    ]
  sens = y_pred[data.y_true == 1].mean()
  spec = (1 - y_pred[data.y_true == 0]).mean()
  return sens, spec


def evaluate_baseline_reader(data: pd.DataFrame) -> Tuple[float, float]:
  """Evaluates clinical workflow.

  Args:
    data: The dataframe that holds the dataset on which evaluation will be
      based. Includes the columns for ground truth ("y_true"), ML model
      predictions ("y_model"), and clinical workflow opinions ("reader_score").

  Returns:
    The sensitivity and specificity estimates of the clinical workflow.
  """
  baseline_fpr, baseline_tpr, _ = skmetrics.roc_curve(
      data.y_true, data.reader_score
  )
  sens = baseline_tpr[1]
  spec = 1 - baseline_fpr[1]
  return sens, spec


def evaluate_baseline_model(data: pd.DataFrame) -> Tuple[float, float]:
  """Evaluates the baseline ML model.

  Args:
    data: The dataframe that holds the dataset on which evaluation will be
      based. Includes the columns for ground truth ("y_true"), ML model
      predictions ("y_model"), and clinical workflow opinions ("reader_score").

  Returns:
    The sensitivity and specificity estimates of the ML model.
  """
  baseline_op = model_op_at_reader_op(
      data,
      reader_match_strategy="average_sens_spec_v2",
  )
  cm = skmetrics.confusion_matrix(data["y_true"], data["y_model"] > baseline_op)
  sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
  spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
  return sens, spec


def model_op_at_reader_op(
    data: pd.DataFrame,
    reader_match_strategy: str = "ppv",
) -> float:
  """Obtains a baseline operating point for ML model that matches the reader.

  Args:
    data: The dataframe that holds the dataset on which computation will be
      based. Includes the columns for ground truth ("y_true"), ML model
      predictions ("y_model"), and clinical workflow opinions ("reader_score").
    reader_match_strategy: Strategy that determines how to match the reader in
      obtaining a baseline ML model.

  Returns:
    The operating point for the baseline ML model.
  """
  if reader_match_strategy == "ppv":
    return utils.model_op_getter(data, utils.ppv, None)
  elif reader_match_strategy == "sens":
    return utils.model_op_getter(data, utils.sens, None)
  elif reader_match_strategy == "spec":
    return utils.model_op_getter(data, utils.spec, None)
  elif reader_match_strategy == "average_sens_spec":
    model_op_at_reader_sens = utils.model_op_getter(data, utils.sens, None)
    model_op_at_reader_spec = utils.model_op_getter(data, utils.spec, None)
    return (model_op_at_reader_sens + model_op_at_reader_spec) / 2
  elif reader_match_strategy == "average_sens_spec_v2":
    # Modification from v1: Instead of averaging the operating points,
    # we average the specificities of those operating points.
    # Specificity is chosen, since we have more negatives.
    model_op_at_reader_sens = utils.model_op_getter(data, utils.sens, None)
    # Compute average spec of operating points
    spec_at_reader_sens = utils.spec(
        data.y_true, scores=(data.y_model > model_op_at_reader_sens).astype(int)
    )
    reader_spec = utils.spec(data.y_true, data["reader_score"])
    average_spec = (spec_at_reader_sens + reader_spec) / 2
    # Get model OP at average spec
    return utils.model_op_getter(
        data,
        utils.spec,
        target_metric=average_spec,
    )
  else:
    raise NotImplementedError(
        f"Strategy {reader_match_strategy} not implemented"
    )
