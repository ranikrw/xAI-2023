import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

# Do not show FutureWarning
import warnings
warnings.filterwarnings("ignore")

import xgboost

from sklearn.model_selection import GridSearchCV

import pickle # For saving dict with tuned parameter values

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

# Load data
data = pd.read_csv('../data/data.csv',sep=';',low_memory=False)

# Make folder for saving tuned hyperparameters
folder_name = 'tuned_hyperparameters'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Defining variable sets
variables_all = get_variables_altman_1968() + get_variables_altman_and_sabato_2007() + get_variables_paraschiv_2021()
variables_all = list(np.unique(variables_all)) # Making sure all are unique
variable_sets = {
  'Altman (1968)': get_variables_altman_1968(),
  'Altman and Sabato (2007)': get_variables_altman_and_sabato_2007(),
  'Paraschiv et al. (2021)': get_variables_paraschiv_2021(),
  'All': variables_all,
}

# Years to evaluate, 2010-2020
years = list(np.arange(2010,2020+1))

# Grid of parameter values to evaluate
param_grid = {
    'learning_rate': (0.1,0.2,0.3), # step size shrinkage parameter
    'subsample':  (0.5,0.75,1),
    'gamma':(0.0,0.2,0.4),
    'reg_lambda':(1,2,3),
    'n_estimators': (25,50,100), # J
    'max_depth': (1,2,3), # Maximum depth of the decision trees
}

for var_set_name in variable_sets:
    variable_set = variable_sets[var_set_name]

    # Dict for tuned parameter values
    best_hyperparameters = {}

    for year in tqdm(years): # For each accounting year 2010-2020
        # Defining data
        data_train = data[data['regnaar']<year].reset_index(drop=True)
        
        X_train = data_train[variable_set]
        y_train = data_train['bankrupt_fs']

        model = xgboost.XGBClassifier(
                objective= 'binary:logistic',
                eval_metric='logloss',
                random_state=1,
                )
        grid_search = GridSearchCV(
            model,\
            param_grid = param_grid,\
            scoring='roc_auc',\
            refit=True,\
            cv=3)
        grid_search.fit(X_train,y_train)
        best_hyperparameters[year] = grid_search.best_params_

    # Saving tuned hyperparameters
    f = open(folder_name+'/'+var_set_name+'.pkl','wb')
    pickle.dump(best_hyperparameters,f)
    f.close()