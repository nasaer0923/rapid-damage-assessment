# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:19:31 2020

@author: Yudi Chen
"""

'''
In this script, four conventional machine learning algorithms are employed to 
train an integrated damage assessment model
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':16
        }
rcParams.update(params)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()

svr = SVR(gamma='scale')
lr = ElasticNet()

parameters_svr = {'kernel': ['linear', 'rbf'], 'C': [0.01, 1, 100]}
parameters_lr = {'l1_ratio': [0.01, 0.1, 0.9]}

#%%
def corr_visualize(df):
    '''Visualize the correlation coefficient matrix
    '''
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=(8, 6.5))
    ax = sns.heatmap(corr, cmap='jet', annot=True, fmt='.3f')
    ax.collections[0].colorbar.set_label('Pearson coefficient')
    plt.gca().set_yticklabels(df.columns, rotation=0)
    plt.tight_layout()
    plt.savefig('correlation.tif', dpi=300)

disasters = ['Harvey', 'Irma', 'Michael']
all_data = pd.DataFrame()
columns_to_keep = ['TNDI', 'GDP', 'POP', 'MHI', 'poverty pop', 'wind', 'precipitation', 'damage']
for disaster in disasters:
    file = disaster + '-feats.csv'
    disaster_data = pd.read_csv(file, index_col='County')
    disaster_data = disaster_data[columns_to_keep]
    disaster_data['disaster'] = disaster
    all_data = all_data.append(disaster_data)

all_data = all_data.rename({'wind': 'MWS', 'precipitation': 'TP', 'damage': 'Damage'}, axis=1)
all_data.fillna(value=0,  inplace=True)
all_data['TNDI'] = all_data['TNDI'] + 1
all_data = all_data.loc[all_data['Damage'] > 1E4]
#all_data = all_data.loc[all_data['TNDI'] > 1]

columns_to_log = ['TNDI', 'GDP', 'POP', 'Damage']
for column in columns_to_log:
    all_data[column] = all_data[column].apply(np.log10)
    

columns_to_keep = ['disaster', 'TNDI', 'GDP', 'POP', 'MWS', 'TP', 'Damage']
all_data = all_data[columns_to_keep].drop('disaster', axis=1)
corr_visualize(all_data)

feat_combs = [[0], [1, 2], [3, 4], [0, 1, 2], [0, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]]
index = [str(feat_comb) for feat_comb in feat_combs]
columns=['train_mse', 'train_mae', 'train_r2', 
         'test_mse', 'test_mae', 'test_r2']

metrics = ['RMSE', 'MAE', 'R2']
models = ['MLR', 'SVR']
raw_feats = ['TNDT', 'GDP', 'POP', 'MWS', 'TR']
text_feat_combs = list('ABCDEFG')
name_dict = {str(k): str(v) for (k, v) in zip(feat_combs, text_feat_combs)}
estimators = [lr, svr]
parameters = [parameters_lr, parameters_svr]
num = 100

MLR_error_df = pd.DataFrame(index=all_data.index)
SVR_error_df = pd.DataFrame(index=all_data.index)
MLR_hat_df = pd.DataFrame(index=all_data.index)
SVR_hat_df = pd.DataFrame(index=all_data.index)

for estimator, parameter, model in zip(estimators, parameters, models):
    results_lst = []
    for jj in range(num):
        training_feat, testing_feat, training_label, testing_label = train_test_split(
                all_data.drop('Damage', axis=1), all_data['Damage'], test_size=0.33)
        training_feat = scaler.fit_transform(training_feat)
        testing_feat = scaler.transform(testing_feat)
        tmp_results = pd.DataFrame(index=index, columns=columns)
        for ii in range(len(feat_combs)):
            feat_comb = feat_combs[ii]
            estimator_grid = GridSearchCV(estimator=estimator, 
                                          param_grid=parameter, 
                                          cv=5, 
                                          scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'], 
                                          return_train_score=True,
                                          refit='r2'
                                          ).fit(training_feat[:, feat_comb], training_label)
            print('\n' + '=' * 50)
            print(estimator_grid.best_params_)
            y_hat_train = estimator_grid.predict(training_feat[:, feat_comb])
            y_hat_test = estimator_grid.predict(testing_feat[:, feat_comb])
            feat_comb = str(feat_comb)
            tmp_results.loc[feat_comb, 'train_rmse'] = np.sqrt(mean_squared_error(training_label, y_hat_train))
            tmp_results.loc[feat_comb, 'train_mae'] = mean_absolute_error(training_label, y_hat_train)
            tmp_results.loc[feat_comb, 'train_r2'] = r2_score(training_label, y_hat_train)
            tmp_results.loc[feat_comb, 'test_rmse'] = np.sqrt(mean_squared_error(testing_label, y_hat_test))
            tmp_results.loc[feat_comb, 'test_mae'] = mean_absolute_error(testing_label, y_hat_test)
            tmp_results.loc[feat_comb, 'test_r2'] = r2_score(testing_label, y_hat_test)
            if (feat_comb == str(feat_combs[-1])) & (model == 'MLR'):
                error = testing_label - y_hat_test
                error = error.to_frame(jj)
                MLR_error_df = MLR_error_df.merge(error, how='outer', left_index=True, right_index=True)
                y_hat_test_df = pd.DataFrame(y_hat_test, index=testing_label.index, columns=[jj])
                MLR_hat_df = MLR_hat_df.merge(y_hat_test_df, how='outer', left_index=True, right_index=True)
            if (feat_comb == str(feat_combs[-1])) & (model == 'SVR'):
                error = testing_label - y_hat_test
                error = error.to_frame(jj)
                SVR_error_df = SVR_error_df.merge(error, how='outer', left_index=True, right_index=True)
                y_hat_test_df = pd.DataFrame(y_hat_test, index=testing_label.index, columns=[jj])
                SVR_hat_df = SVR_hat_df.merge(y_hat_test_df, how='outer', left_index=True, right_index=True)
                
        results_lst.append(tmp_results)

    all_boot_lst = []
    for metric in metrics:
        training_df = pd.DataFrame()
        testing_df = pd.DataFrame()
        for a_boot_result in results_lst:
            testing_df = testing_df.append(a_boot_result.loc[:, 'test_' + metric.lower()].to_frame())
        testing_df['Feature'] = testing_df.index
        testing_df['Feature'] = testing_df['Feature'].apply(lambda x: str(name_dict[x]))
        testing_df['Data'] = 'Testing'
        testing_df = testing_df.rename({'test_' + metric.lower(): metric}, axis=1)
        all_boot_lst.append(testing_df)
    if model == 'MLR':
        mlr_boot_lst = all_boot_lst
    else:
        svr_boot_lst = all_boot_lst
    
#%% Visualization
        
blue_square = {'marker':'s', 'markerfacecolor':'blue', 'markeredgecolor':'blue'}
red_square = {'marker':'s', 'markerfacecolor':'red', 'markeredgecolor':'red'}
green_square = {'marker':'s', 'markerfacecolor':'green', 'markeredgecolor':'green'}
blue = {'color': 'blue'}
red = {'color': 'red'}
green = {'color': 'green'}
blue_plus = {'marker':'+', 'markerfacecolor':'blue', 'markeredgecolor':'blue'}
red_plus = {'marker':'+', 'markerfacecolor':'red', 'markeredgecolor':'red'}
green_plus = {'marker':'+', 'markerfacecolor':'green', 'markeredgecolor':'green'}

boot_lsts = [mlr_boot_lst, svr_boot_lst]
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
ticklabels = ['$X_{\mathrm{s}}$',
              '$X_{\mathrm{c}}$',
              '$X_{\mathrm{w}}$',
              '$X_{\mathrm{s+c}}$',
              '$X_{\mathrm{s+w}}$',
              '$X_{\mathrm{c+w}}$',
              '$X_{\mathrm{s+c+w}}$',
              'Mean']
counter = 0
for boot_lst, model in zip(boot_lsts, models):
    for ii, metric in enumerate(metrics):
        data = boot_lst[ii].drop('Data', axis=1)
        
        a_data = []
        for pos, jj in enumerate(list('ABC')):
            a_data.append(data.loc[data['Feature'] == jj][metric])
        axes[ii, counter].boxplot(a_data, positions=range(3),
            medianprops=blue, boxprops=blue, 
            whiskerprops=blue, capprops=blue, flierprops=blue_plus, widths=[0.5, 0.5, 0.5])
        mean_metric_no_integ = np.mean(a_data)
        
        a_data = []
        for pos, jj in enumerate(list('DEF')):
            a_data.append(data.loc[data['Feature'] == jj][metric])
        axes[ii, counter].boxplot(a_data, positions=range(3, 6),
            medianprops=red, boxprops=red, whiskerprops=red,
            capprops=red, flierprops=red_plus, widths=[0.5, 0.5, 0.5])
        mean_metric_partial_integ = np.mean(a_data)
        
        a_data = data.loc[data['Feature'] == 'G'][metric]
        axes[ii, counter].boxplot(a_data, positions=[6],
            medianprops=green, boxprops=green, whiskerprops=green,
            capprops=green, flierprops=green_plus, widths=0.5)
        mean_metric_full_integ = np.mean(a_data)
        axes[ii, counter].plot([7 - 0.25, 7 + 0.25], [mean_metric_no_integ, mean_metric_no_integ], '--b', lw='2')
        axes[ii, counter].plot([7 - 0.25, 7 + 0.25], [mean_metric_partial_integ, mean_metric_partial_integ], '--r', lw='2')
        axes[ii, counter].plot([7 - 0.25, 7 + 0.25], [mean_metric_full_integ, mean_metric_full_integ], '--g', lw='2')
        
        axes[ii, counter].set_xticks(ticks=range(8))
        axes[ii, counter].set_xticklabels(ticklabels)
        if metric == 'R2':
            axes[ii, counter].plot([6.5, 6.5], [-0.4, 0.8], c='black', lw='1')
            axes[ii, counter].plot([-0.5, 6.5], [0, 0], c='black', lw='1', ls='--')
            axes[ii, counter].set_ylim(-0.4, 0.8)
            axes[ii, counter].set_yticks(ticks=[x / 10 for x in range(-4, 9, 2)])
            axes[ii, counter].set_ylabel(model + ': ' + '$R^2$')
            axes[ii, counter].set_xlabel('Feature set')
        elif metric == 'MAE':
            axes[ii, counter].plot([6.5, 6.5], [0.3, 0.9], c='black', lw='1')
            axes[ii, counter].set_ylim(0.3, 0.9)
            axes[ii, counter].set_yticks(ticks=[x / 10 for x in range(3, 10, 2)])
            axes[ii, counter].set_ylabel(model + ': ' + metric)
        else:
            axes[ii, counter].plot([6.5, 6.5], [0.4, 1.2], c='black', lw='1')
            axes[ii, counter].set_ylim(0.4, 1.2)
            axes[ii, counter].set_yticks(ticks=[x / 10 for x in range(4, 13, 2)])
            axes[ii, counter].set_ylabel(model + ': ' + metric)
        feat_metric = data.loc[data['Feature']=='A'][metric].mean()
        print('The {} with feature set A: {} = {:.3f}'.format(metric, model, feat_metric))
        feat_metric = data.loc[data['Feature']=='G'][metric].mean()
        print('The {} with feature set G: {} = {:.3f}'.format(metric, model, feat_metric))
    counter += 1
    plt.tight_layout()
#    plt.savefig('performance.tif', dpi=300)
    
    
#%% Error analysis
    
def linear_fit(x, y, xlabel, ylabel, error):
    plt.figure(figsize=(8, 6))
    plt.plot([4, max(max(x), max(y)) + 0.1],
             [4, max(max(x), max(y)) + 0.1], '-k', lw='1', label='X=Y')
    plt.scatter(x, y, c=error, cmap='jet')
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#    y_fit = slope * x.sort_values() + intercept
#    plt.plot(x.sort_values(), y_fit, '--k',
#             label=('Fit: R=' + str(round(r_value, 2)) + '$^{**}$' + '$^{*}$'))
#    print('r value: {:.2f} \nP value: {:.3f}'.format(r_value, p_value))
    plt.xlabel(xlabel)
    plt.xlim([min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1])
    plt.ylim([min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1])
    plt.xticks(ticks=range(4, 10))
    plt.yticks(ticks=range(4, 10))
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig('linear fit.tif', dpi=300)
    

def box_plot(counties, assessed_damage, actual_damage, base=0):
    plt.figure(figsize=(10, 5))
    county_data = []
    county_damage = []
    for county in counties:
        county_data.append(assessed_damage.loc[county, :].dropna())
        county_damage.append(actual_damage.loc[county])
    plt.boxplot(county_data, positions=range(len(counties)), medianprops=blue, 
                boxprops=blue, whiskerprops=blue, capprops=blue, 
                flierprops=blue_plus, widths=0.5*np.ones((len(counties), 1)))
    plt.plot(county_damage, 'bs')
    plt.xticks(ticks=range(len(counties)), labels=counties)
    plt.ylabel('Damage (Log scale)')
    plt.tight_layout()


for model in ['SVR',]:
    if model == 'MLR':
        error_df = MLR_error_df
        hat_df = MLR_hat_df
    else:
        error_df  = SVR_error_df
        hat_df = SVR_hat_df
    error_df_mean = error_df.mean(axis=1)
    error_df_mean.sort_values(ascending=True, inplace=True)
    num = 2
    counties_with_largest_errors = list(error_df_mean.index[:num])
    counties_with_largest_errors.extend(error_df_mean.index[-num:])
    hat_mean = hat_df.mean(axis=1)
    linear_fit(all_data['Damage'], hat_mean, 'Actual Damage (Log scale)', 
               'Assessed Damage (Log scale)', error_df_mean.sort_index().to_list())
    box_plot(counties_with_largest_errors, hat_df, all_data['Damage'])

    
    
    
    
    
    
    
    
    
    
    
    
    