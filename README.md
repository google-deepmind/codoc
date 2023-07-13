# Repository for Complementarity-Driven Deferral to Clinicians (CoDoC)

This repository includes the source code for the paper "Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians (CoDoC)" by Dvijotham et al. (2023), published in the journal _Nature Medicine_. The contents of the repository can be used to replicate the experiments provided in the paper, as well as to utilize the CoDoC framework in independent human-AI complementarity research.

## Data

If you're interested in the data, please email codoc-team@google.com and you will be contacted once it is available.


## Installation

The following command sets up python virtual environment and installs all the
dependencies. This uses `virtualenv` python module
to create virtual environment. If it doesn not exist, please install it with
`pip`.

   ```bash
   bash install.sh
   ```

## Running

   ```bash
   bash run.sh
   ```

The above script should open a notebook server from which `codoc_experiments.ipynb`
can be run. The notebook has further instructions and documentation to guide
through running the experimentation pipeline.

## Quickstart

For both purposes mentioned above, we recommend starting from the Jupyter notebook file `Replicating_CoDoC_Experiment_Results.ipynb`. This file walks the user through various functionalities of the implementation provided, familiarizes them with the data format adopted, and if desired provides more specific instructions for the exact replication of existing results.

Please refer to the original paper for a detailed introduction to the CoDoC framework, its clinical and statistical properties, and experimental results on a variety of datasets.

## Datasets

The UK Mammography Dataset (AI scores, clinical predictions, ground truth) will be made available on this page soon, based on a click to accept licensing agreement.

The US Mammography Dataset 2 can be obtained for research purposes by contacting Prof. Krzysztof J Geras (k.j.geras@nyu.edu).

Other datasets featured in the paper are not available as they belong to third party institutions and were used under a licensing agreement specific to this study that does not permit further sharing.

The data format instructions provided in the aforementioned notebook also allows the user to utilize this implementation with their own custom dataset.

## Contact

For any questions regarding this repository or the paper, please contact Krishnamurthy (Dj) Dvijotham (dvij@cs.washington.edu) and Jim Winkens (jimwinkens@google.com).
