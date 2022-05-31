# Complementarity-driven Deferral to Clinicians

This repository contains an implementation of the Complementarity-driven
Deferral to Clinicians (CoDoC) algorithms for learning to decide when to rely
on a diagnostic AI model and when to defer to a human clinician.

## Installation

1. Clone the CoDoC repository:

   ```bash
   git clone https://github.com/deepmind/codoc.git
   ```

1. Recommended: set up a virtual Python environment:

   ```bash
   python3 -m venv codoc_env
   source codoc/bin/activate
   ```
   (To leave the virtual environment, type `deactivate`.)

1. Install the dependencies:

   ```bash
   pip3 install -r codoc/requirements.txt
   ```

## Usage

The following command trains and evaluates a deferral model on Optimam data
(first reader) using the DP algorithm:

_To be run from the parent directory that contains the codoc repository as a
sub-directory._

```bash
python3 -m codoc.examples.train --config=codoc/examples/config.py
```

This will train a deferral model on the configured "tune" dataset A, and
evaluate it on both "tune" and "validation" datasets. A successful run will log
the following metrics for each:

| Metric             | Description                                            |
| ------------------ | ------------------------------------------------------ |
| count              | number of examples                                     |
| num_case_pos       | number of positive examples according to ground truth  |
| num_case_neg       | number of negative examples according to ground truth  |
| withholding_ratio  | proportion of examples for which the deferral model selects to abstain (defer to human) |
| sens               | sensitivity of the prediction model                    |
| spec               | specificity of the prediction model                    |
| min_composite_sens | sensitivity of the model with selective human deferral |
| min_composite_spec | specificity of the model with selective human deferral |

## Citing this work

If you use this code in your work, we ask that you cite this paper:

TODO

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
