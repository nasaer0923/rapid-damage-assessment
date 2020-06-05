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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

svr = SVR(gamma='scale')
lr = ElasticNet(max_iter=10000)
dt = DecisionTreeRegressor()
knn = KNeighborsRegressor()
nb = GaussianNB()

parameters_svr = {'kernel': ('linear', 'rbf'),  'C': [0.01, 1, 10, 100, 500]}
parameters_knn = {'n_neighbors': [1, 2, 3]}
parameters_lr = {'l1_ratio': [0, 0.01, 0.1, 0.9]}
parameters_dt = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 3, 5]}

#%%
def corr_visualize(df):
    '''Visualize the correlation coefficient matrix
    '''
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=(8, 6.5))
    ax = sns.heatmap(corr, cmap='jet', annot=True, fmt='.2f')
    ax.collections[0].colorbar.set_label('Pearson coefficient')
    plt.gca().set_yticklabels(df.columns, rotation=0)
    plt.tight_layout()

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
all_data = all_data.loc[all_data['TNDI'] > 1]

columns_to_log = ['TNDI', 'GDP', 'POP', 'Damage']
for column in columns_to_log:
    all_data[column] = all_data[column].apply(np.log10)
    

columns_to_keep = ['disaster', 'TNDI', 'GDP', 'POP', 'MWS', 'TP', 'Damage']
all_data = all_data[columns_to_keep]
training = all_data.loc[all_data['disaster'] != 'Michael'].drop('disaster', axis=1)
corr_visualize(training)
training_feat = training.drop('Damage', axis=1)
training_feat = scaler.fit_transform(training_feat)
training_label = training['Damage']

testing = all_data.loc[all_data['disaster'] == 'Michael'].drop('disaster', axis=1)
testing_feat = testing.drop('Damage', axis=1)
testing_feat = scaler.transform(testing_feat)
testing_label = testing['Damage']

estimator = lr
parameter = parameters_lr
feat_combs = [[0], [0, 1, 2], [0, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]]
index = [str(feat_comb) for feat_comb in feat_combs]
results = pd.DataFrame(index=index, columns=['train_mse', 'train_mae', 'train_r', 'train_p_value', 
                                             'train_baseline_r', 'train_baseline_p_value',
                                             'test_mse', 'test_mae', 'test_r', 'test_p_value',
                                             'test_baseline_r', 'test_baseline_p_value'])
for ii in range(len(feat_combs)):
    feat_comb = feat_combs[ii]
    estimator_grid = GridSearchCV(estimator=estimator, 
                                  param_grid=parameter, 
                                  cv=5, 
                                  scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'], 
                                  return_train_score=True,
                                  refit='neg_mean_squared_error'
                                  ).fit(training_feat[:, feat_comb], training_label)
    print('\n' + '=' * 50)
    print(estimator_grid.best_params_)
    y_hat_train = estimator_grid.predict(training_feat[:, feat_comb])
    y_hat_test = estimator_grid.predict(testing_feat[:, feat_comb])
    feat_comb = str(feat_comb)
    results.loc[feat_comb]['train_mse'] = mean_squared_error(training_label, y_hat_train)
    results.loc[feat_comb]['train_mae'] = mean_absolute_error(training_label, y_hat_train)
    results.loc[feat_comb]['train_r'] = pearsonr(training_label, y_hat_train)[0]
    results.loc[feat_comb]['train_p_value'] = pearsonr(training_label, y_hat_train)[1]
    results.loc[feat_comb]['test_mse'] = mean_squared_error(testing_label, y_hat_test)
    results.loc[feat_comb]['test_mae'] = mean_absolute_error(testing_label, y_hat_test)
    results.loc[feat_comb]['test_r'] = pearsonr(testing_label, y_hat_test)[0]
    results.loc[feat_comb]['test_p_value'] = pearsonr(testing_label, y_hat_test)[1]
    results.loc[feat_comb]['train_baseline_r'] = pearsonr(training_feat[:, 0], training_label)[0]
    results.loc[feat_comb]['train_baseline_p_value'] = pearsonr(training_feat[:, 0], training_label)[1]
    results.loc[feat_comb]['test_baseline_r'] = pearsonr(testing_feat[:, 0], testing_label)[0]
    results.loc[feat_comb]['test_baseline_p_value'] = pearsonr(testing_feat[:, 0], testing_label)[1]
    
    
#%%
#from scipy import stats
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#def linear_fit(x, y, xlabel, ylabel):
#    x = np.array(x, dtype=float)
#    y = np.array(y, dtype=float)
#    plt.figure(figsize=(5, 5))
#    min_xy  = min(min(x), min(y)) - 0.1
#    max_xy = max(max(x), max(y)) + 0.1
##    plt.plot([min_xy, max_xy], [min_xy, max_xy], '--', color='black', lw=1, label='Y=X')
#    plt.scatter(x, y, color='blue', label='Raw data')
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#    y_fit = slope * x + intercept
#    plt.plot(x, y_fit, '-r', lw=1, color='black',
#             label=('Fit: R=' + str(round(r_value, 2)) + '$^{**}$' + '$^{*}$'))
#    print('r value: {:.2f} \nP value: {:.3f}'.format(r_value, p_value))
#    plt.xlabel(xlabel)
#    plt.xlim([min_xy, max_xy])
#    plt.ylim([min_xy, max_xy])
#    plt.ylabel(ylabel)
#    plt.legend(loc='upper left')
#    plt.gca().set_aspect('equal')
#    plt.tight_layout()
#    plt.savefig('Michael.tiff', dpi=300)
#
## model with best parameters
#model = SVR(gamma='scale', kernel='linear', C=100)
##model = ElasticNet(l1_ratio=0.1)
#
#feature_combs = [[0], [0, 1, 2], [1, 2, 3, 4], [0, 3, 4]]
#for feature_comb in feature_combs:
#    model.fit(training_feat[:, feature_comb], training_label)
#    y_hat_train = model.predict(training_feat[:, feature_comb])
#    y_hat_test = model.predict(testing_feat[:, feature_comb])
#    
#    print('\n*********************************************************')
#    print('Training MSE: ', mean_squared_error(training_label, y_hat_train))
#    print('Testing MSE: ', mean_squared_error(testing_label, y_hat_test))
#    print('SVR - Pearson correlation on Test data:')
#    linear_fit(y_hat_train, training_label, 'Training Predicted Damage (X)', 'Training FEMA Damage (Y)')
#    linear_fit(y_hat_test, testing_label, 'Predicted Damage: Log', 'FEMA Damage: Log')


#import utils
## geographic visualization
#index = [x.split('-')[1] + '-' + x.split('-')[2] for x in testing_label.index]
#values = pd.Series(data=y_hat_test, index=index)
#coords = [[-87.89, 25.00],
#          [-78.88, 35.00]]
#coords = [[-86.50, 29.50],
#          [-82.50, 33.00]]
#title = 'Predicted Damage: Log'
#utils.geographic_visualization(coords, values, title)
#
#values = testing_label.copy()
#values.index = index
#title = 'FEMA Damage: Log'
#utils.geographic_visualization(coords, values, title)





