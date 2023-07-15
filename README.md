This repository contains all code for running analyses and producing results presented in the following paper:

Wahlstrøm, R. R. (2023). Explainable Artificial Intelligence (xAI) for Interpreting Machine Learning Methods and Their Individual Predictions. Working Paper, Norwegian University of Science and Technology. https://doi.org/10.2139/ssrn.4321303

## 1_create_data.py
The code in this file prepares the data.

## 2_tune_hyperparameters.py
The code in this file tunes the hyper-parameters and save the tuned values in the folder *"tuned_hyperparameters"*.

## 3_comparing_variable_sets.py
The code in this file compares the performance of the LR and XGBoost methods. Results are saved in the folder *"results"*, which contains figures shown in the paper as well as LR regression tables with *t*-statistics.

## 4_interpreting_with_SHAP.py
The code in this file interprets the models and individual predictions. Figures are saved in *"results_SHAP"*.

## functions
The folder *"functions"* contains functions used by the code in the files mentioned above.

## Data
I am not allowed to share the data due to restrictions from the provider.

<br/><br/>
**Permanent link to this repository:** https://xai2023.ranik.no/