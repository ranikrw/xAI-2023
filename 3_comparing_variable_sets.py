import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import statsmodels.api as sm

from sklearn import metrics

import xgboost

import pickle # For loading tuned hyperparameters

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # For formatting axis with percent

from tqdm import tqdm # for-loop progress bar

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *
from functions_3_analyses import *

# Load data
data = pd.read_csv('../data/data.csv',sep=';',low_memory=False)

# Make folder for saving results
folder_name = 'results'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Parameters
num_decimals = 2 # For regression results of LR models in Excel

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

# Data frames for saving AR results across all variable sets
df_results_AR_in_sample        = pd.DataFrame(index = years)
df_results_AR_out_of_sample    = pd.DataFrame(index = years)

for method in ['LR','XGBoost']:

    for var_set_name in variable_sets:
        variable_set = variable_sets[var_set_name]

        if method == 'XGBoost':
            # Loading tuned hyperparameters
            f = open('tuned_hyperparameters/'+var_set_name+'.pkl', 'rb')
            best_hyperparameters = pickle.load(f)
            f.close()

        # Data frames and lists for saving results 
        # for the variable set
        df_regression_results   = pd.DataFrame()
        series_AR_in_results       = []
        series_AR_out_results      = []

        for year in years: # For each accounting year 2010-2020
            # Defining data
            data_test = data[data['regnaar']==year].reset_index(drop=True)
            data_train = data[data['regnaar']<year].reset_index(drop=True)

            X_train = data_train[variable_set]
            X_test  = data_test[variable_set]

            y_train = data_train['bankrupt_fs']
            y_test  = data_test['bankrupt_fs']

            if method == 'LR':
                X_train = sm.add_constant(X_train,prepend=False) # Add constant
                X_test = sm.add_constant(X_test,prepend=False) # Add constant
                model = sm.Logit(y_train,X_train).fit()
                
                list_variables = variable_set + ['const']
                
                series_coef     = pd.Series(dtype=object)
                series_t_val    = pd.Series(dtype=object)
                for i in list_variables:
                    coef = np.round(model.params[i],num_decimals).astype(str)
                    coef = add_tailing_zeros_decimals(coef,num_decimals)
                    series_coef[i] = coef

                    tval = np.round(model.tvalues[i],num_decimals).astype(str)
                    tval = add_tailing_zeros_decimals(tval,num_decimals)
                    pval = model.pvalues[i]
                    if pval<0.01:
                        tval = tval+'***'
                    elif pval<0.05:
                        tval = tval+'**'
                    elif pval<0.10:
                        tval = tval+'*'
                    series_t_val[i] = tval
            
            elif method == 'XGBoost':
                model = xgboost.XGBClassifier(
                    learning_rate       = best_hyperparameters[year]['learning_rate'],\
                    subsample           = best_hyperparameters[year]['subsample'],\
                    gamma               = best_hyperparameters[year]['gamma'],\
                    reg_lambda          = best_hyperparameters[year]['reg_lambda'],\
                    n_estimators        = best_hyperparameters[year]['n_estimators'],\
                    max_depth           = best_hyperparameters[year]['max_depth'],\
                    objective= 'binary:logistic',
                    eval_metric='logloss',
                    random_state=1,
                    )
                model.fit(X_train,y_train)

            else:
                print('ERROR defining method')

            # Evaluation metrics
            if method == 'LR':
                AUC_in_sample       = metrics.roc_auc_score(y_train,model.predict(X_train))
                AUC_out_of_sample   = metrics.roc_auc_score(y_test,model.predict(X_test))
            elif method == 'XGBoost':
                AUC_in_sample       = metrics.roc_auc_score(y_train,model.predict_proba(X_train)[:,1])
                AUC_out_of_sample   = metrics.roc_auc_score(y_test,model.predict_proba(X_test)[:,1])

            AR_in_sample       = (AUC_in_sample-0.5)*2
            AR_out_of_sample   = (AUC_out_of_sample-0.5)*2

            series_AR_in_results   = series_AR_in_results + [AR_in_sample]
            series_AR_out_results  = series_AR_out_results + [AR_out_of_sample]

            if method == 'LR':
                series_coef['R2'] = add_tailing_zeros_decimals(np.round(model.prsquared,num_decimals).astype(str),num_decimals)

                series_coef['In-sample AR']             = add_tailing_zeros_decimals(np.round(AR_in_sample,num_decimals).astype(str),num_decimals)
                series_coef['In-sample AUC']            = add_tailing_zeros_decimals(np.round(AUC_in_sample,num_decimals).astype(str),num_decimals)
                series_coef['In-sample Brier score']    = add_tailing_zeros_decimals(np.round(metrics.brier_score_loss(y_train,model.predict(X_train)),num_decimals).astype(str),num_decimals)

                series_coef['Out-of-sample AR']             = add_tailing_zeros_decimals(np.round(AR_out_of_sample,num_decimals).astype(str),num_decimals)
                series_coef['Out-of-sample AUC']            = add_tailing_zeros_decimals(np.round(AUC_out_of_sample,num_decimals).astype(str),num_decimals)
                series_coef['Out-of-sample Brier score']    = add_tailing_zeros_decimals(np.round(metrics.brier_score_loss(y_test,model.predict(X_test)),num_decimals).astype(str),num_decimals)

                # Number of observations
                series_coef['No. of obs.'] = thousand_seperator(X_train.shape[0])

                # Merging coefficient and pvalue results
                series_results = pd.concat([series_coef,series_t_val],axis=1)
                series_results.columns = [year,'T-test']

                # Merging with data frame for all results of the variable set
                df_regression_results = pd.concat([df_regression_results,series_results],axis=1)

        df_results_AR_in_sample[method+' - '+var_set_name]       = pd.Series(series_AR_in_results,index=years)
        df_results_AR_out_of_sample[method+' - '+var_set_name]   = pd.Series(series_AR_out_results,index=years)

        if method == 'LR':
            # Saving regression results of LR model in an Excel file
            df_regression_results.to_excel(folder_name+'/'+var_set_name+'.xlsx')


########################################################
## Make plot of differences in AR per variable set
########################################################
differences_AR = pd.DataFrame(index=years)
for var_set_name in variable_sets:
    LR_AR = df_results_AR_out_of_sample['LR - ' + var_set_name]
    XGBoost_AR = df_results_AR_out_of_sample['XGBoost - ' + var_set_name]
    differences_AR[var_set_name] = 100*(XGBoost_AR-LR_AR)/LR_AR

# Parameters that determine how the plots will look like
fig_width  = 20 # Width of the figure
fig_length = 10 # Length of the figure
linewidth  = 3  # Width of the lines in the plots
fontsize   = 28

colors = [
    'royalblue',
    'forestgreen',
    'orangered',
    'darkviolet',
    ]

# Making plot of differences in AR
fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
fig.patch.set_facecolor('white')
for c,i in zip(colors,differences_AR.columns):
    ax.plot(differences_AR[i],c=c,lw=linewidth,linestyle='solid')
ax.set_xlabel('Accounting year',fontsize=fontsize)
ax.set_ylabel('Improvement in AR',fontsize=fontsize)
ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
legend_items = list(differences_AR.columns)
ax.legend(legend_items,fontsize=fontsize,loc='upper left',ncol=1,bbox_to_anchor=(1,1))
plt.grid()
plt.savefig(folder_name+'/AR_differences.png',dpi=150, bbox_inches='tight')



########################################################
## Make plot of evaolution of AR for all eight models
########################################################
colors = colors+colors
linestyles = [
    'solid',
    'solid',
    'solid',
    'solid',
    (5, (10, 3)),
    (5, (10, 3)),
    (5, (10, 3)),
    (5, (10, 3)),
    ]

# Out-of-sample AR
fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
fig.patch.set_facecolor('white')
for c,linestyle,i in zip(colors,linestyles,df_results_AR_out_of_sample.columns):
    ax.plot(df_results_AR_out_of_sample[i],c=c,lw=linewidth,linestyle=linestyle)
ax.set_xlabel('Accounting year',fontsize=fontsize)
ax.set_ylabel('Out-of-sample AR',fontsize=fontsize)
ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
plt.xticks(rotation=0)
legend_items = list(df_results_AR_out_of_sample.columns)
ax.legend(legend_items,fontsize=fontsize,loc='upper left',ncol=1,bbox_to_anchor=(1,1))
plt.grid()
plt.savefig(folder_name+'/AR_out_of_sample.png',dpi=150, bbox_inches='tight')

# In-of-sample AR
fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
fig.patch.set_facecolor('white')
for c,linestyle,i in zip(colors,linestyles,df_results_AR_in_sample.columns):
    ax.plot(df_results_AR_in_sample[i],c=c,lw=linewidth,linestyle=linestyle)
ax.set_xlabel('Accounting year',fontsize=fontsize)
ax.set_ylabel('In-sample AR',fontsize=fontsize)
ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
plt.xticks(rotation=0)
legend_items = list(df_results_AR_in_sample.columns)
ax.legend(legend_items,fontsize=fontsize,loc='upper left',ncol=1,bbox_to_anchor=(1,1))
plt.grid()
plt.savefig(folder_name+'/AR_in_sample.png',dpi=150, bbox_inches='tight')