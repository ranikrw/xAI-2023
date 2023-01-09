import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

import xgboost

import shap
shap.initjs() # Load Javascript library for plots

import pickle # For loading tuned hyperparameters

import matplotlib.pyplot as plt

# Importing file(s) with functions
import sys
sys.path.insert(1, 'functions')
from functions_variables import *

# Load data
data = pd.read_csv('../data/data.csv',sep=';',low_memory=False)

# Make folders for saving results
folder_name = 'results_SHAP'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
subfolder = 'SHAP_dependence_plots'
if not os.path.exists(folder_name+'/'+subfolder):
    os.makedirs(folder_name+'/'+subfolder)

# Defining variable set
variable_set = get_variables_altman_1968() + get_variables_altman_and_sabato_2007() + get_variables_paraschiv_2021()
variable_set = list(np.unique(variable_set)) # Making sure all are unique

# Loading tuned hyperparameters
var_set_name = 'All'
f = open('tuned_hyperparameters/'+var_set_name+'.pkl', 'rb')
best_hyperparameters = pickle.load(f)
f.close()

# Defining testing year
year = 2020

# Defining data
data_test = data[data['regnaar']==year].reset_index(drop=True)
data_train = data[data['regnaar']<year].reset_index(drop=True)

X_train = data_train[variable_set]
X_test  = data_test[variable_set]

y_train = data_train['bankrupt_fs']
y_test  = data_test['bankrupt_fs']

# Training model
model = xgboost.XGBClassifier(
    learning_rate       = best_hyperparameters[year]['learning_rate'],\
    subsample           = best_hyperparameters[year]['subsample'],\
    gamma               = best_hyperparameters[year]['gamma'],\
    reg_lambda          = best_hyperparameters[year]['reg_lambda'],\
    n_estimators        = best_hyperparameters[year]['n_estimators'],\
    max_depth           = best_hyperparameters[year]['max_depth'],\
    objective= 'binary:logistic',
    eval_metric='logloss',
    random_state=1)
model.fit(X_train,y_train)

# Construct explainer
explainer = shap.TreeExplainer(model)

# Extracting shap values
sv = explainer(X_train)
shap_values = sv.values

# Order variables based on importance cf. SHAP values
variables_ordered           = pd.DataFrame(index=X_train.columns)
variables_ordered['values'] = pd.Series(np.abs(shap_values).mean(0),index=X_train.columns)
variables_ordered           = variables_ordered.sort_values(by=['values'],ascending=False)
variables_ordered           = list(variables_ordered.index)

###################################
## SHAP Beeswarm plot
###################################
shap.summary_plot(
    shap_values, X_train,
    max_display=6,
    show=False
)
plt.xlabel('SHAP value')
plt.savefig(folder_name+'/SHAP_beeswarm_plot_'+str(year)+'.png',dpi=150, bbox_inches='tight')
plt.close() # So the figure does not show in kernel

###################################
## Explaining single predictions
###################################
obs_num = 0
observation_values = pd.DataFrame()
for obs in [7,8,81]:
    obs_num = obs_num+1
    obs_name = 'Firm '+str(obs_num)

    prob = 100*model.predict_proba(X_train)[:,1][obs]
    print(obs_name+' prob. bankruptcy: {}'.format(np.round(prob,2)))

    exp = shap.Explanation(sv, sv.base_values, X_train, feature_names=X_train.columns)
    shap.waterfall_plot(exp[obs],show=False,max_display=7)
    plt.savefig(folder_name+'/'+obs_name+'.png',dpi=150, bbox_inches='tight')
    plt.close() # So the figures do not show in kernel

    observation_values[obs_name] = pd.Series(exp[obs].data,index=X_train.columns)

###################################
## Explaining single features
###################################
# Making frames for plotting
shap_values_df = pd.DataFrame(shap_values,columns=X_train.columns)

# Moving average
MA = 1000
plot_dict = {}
for var_to_plot in variables_ordered:
    temp = pd.DataFrame()
    temp['SHAP value'] = shap_values_df[var_to_plot]
    temp['Feature value'] = X_train[var_to_plot]
    temp = temp.sort_values(by=['Feature value']).reset_index(drop=True)
    
    temp = temp.rolling(MA).mean()
    temp = temp[pd.isnull(temp['SHAP value'])==False]

    plot_dict[var_to_plot] = temp

# Parameters that determine how the plots will look like
fig_width  = 10 # Width of the figure
fig_length = 10 # Length of the figure
linewidth  = 3  # Width of the lines in the plots
fontsize   = 28

# Find max and min values
for var_to_plot in variables_ordered:
    if var_to_plot==variables_ordered[0]:
        ylim_max = np.max(plot_dict[var_to_plot]['SHAP value'])
        ylim_min = np.min(plot_dict[var_to_plot]['SHAP value'])
    else:
        ylim_max = np.max([ylim_max,np.max(plot_dict[var_to_plot]['SHAP value'])])
        ylim_min = np.min([ylim_min,np.min(plot_dict[var_to_plot]['SHAP value'])])

colors = [
    'forestgreen',
    'orangered',
    'darkviolet',
    ]

# Plotting
var_name_for_save = 0
for var_to_plot in variables_ordered:
    temp = plot_dict[var_to_plot]

    std = temp['Feature value'].std()
    min_threshold = np.min([-2*std,np.min(observation_values.loc[var_to_plot])])
    max_threshold = np.max([2*std,np.max(observation_values.loc[var_to_plot])])
    temp = temp[(temp['Feature value']>min_threshold)&(temp['Feature value']<max_threshold)]

    if var_to_plot=='(current liabilities - short-term liquidity) / total assets':
        fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length-0.6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
    fig.patch.set_facecolor('white')
    for col,c in zip(observation_values.columns,colors):
        ax.axvline(observation_values.at[var_to_plot,col],color=c,lw=linewidth,linestyle='dashed') # Event 2 for event 1
    ax.plot(temp['Feature value'],temp['SHAP value'],lw=linewidth,color='royalblue')
    if var_to_plot=='(current liabilities - short-term liquidity) / total assets':
        # This name is so long that it needs to be shown in two lines
        ax.set_xlabel('(current liabilities - short-term liquidity) \n/ total assets',fontsize=fontsize)
    else:
        ax.set_xlabel(var_to_plot,fontsize=fontsize)
    ax.set_ylabel('SHAP value',fontsize=fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.xticks(rotation=0)
    xlim_min = np.min([np.min(temp['Feature value']),np.min(observation_values.loc[var_to_plot])])
    xlim_max = np.max([np.max(temp['Feature value']),np.max(observation_values.loc[var_to_plot])])
    xlim_slack = 0.1
    if xlim_min<0:
        xlim_min=xlim_min*(1+xlim_slack)
    else:
        xlim_min=xlim_min*(1-xlim_slack)
    if xlim_max>0:
        xlim_max=xlim_max*(1+xlim_slack)
    else:
        xlim_max=xlim_max*(1-xlim_slack)
    ax.set_xlim((xlim_min, xlim_max))
    ax.set_ylim((ylim_min, ylim_max))
    ax.grid()
    ax.legend(list(observation_values.columns),fontsize=fontsize,framealpha=1)
    var_name_for_save = var_name_for_save+1
    plt.savefig(folder_name+'/SHAP_dependence_plots/'+str(var_name_for_save)+'.png',dpi=150, bbox_inches='tight')
    plt.close() # So the figures do not show in kernel
