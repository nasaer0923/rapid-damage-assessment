# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:12:45 2019

@author: JID002
"""
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':12
        }
rcParams.update(params)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
markers = ['o', 's', '*', '^', 'p', '+', 'd', 'x', '|', '.']

import statsmodels.api as sm
class LinearRegreesor(object):
    def __init__(self, normalized=False):
        self.normalized = normalized
    
    def fit(self, X, y):
        if self.normalized:
            X_max = X.max(axis=0)
            X_min = X.min(axis=0)
            X = (X - X_max) / (X - X_min)
        X = sm.add_constant(X)
        self.results = sm.OLS(y, X).fit()
        
    def predict(self, X, repeat=1000):
        X = sm.add_constant(X)
        y_hat = np.zeros((X.shape[0], repeat))
        params = self.results.params
        std = self.results.bse
        new_params = np.zeros((len(params), repeat))
        for ii in range(len(params)):
            new_params[ii, :] = np.random.normal(params[ii], std[ii], size=repeat)
        for counter, x in enumerate(X):
            for repeat in range(repeat):
                y_hat[counter, repeat] = sum(new_params[:, repeat] * x)
        y_mean = np.mean(y_hat, axis=1)
        y_std = np.std(y_hat, axis=1)
        return y_mean, y_std


def linear_fit(x, y, xlabel, ylabel):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, color='blue', label='Raw data')
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x + intercept
    plt.plot(x, y_fit, '-r', lw=1, color='black',
             label=('Fit: R=' + str(round(r_value, 2)) + '$^{**}$' + '$^{*}$'))
    print('r value: {:.2f} \nP value: {:.3f}'.format(r_value, p_value))
    plt.xlabel(xlabel)
    plt.xlim([min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1])
    plt.ylim([min(min(x), min(y)) - 0.1, max(max(x), max(y)) + 0.1])
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal')
    plt.tight_layout()


# import features and damages
columns_to_keep = ["damage", "TNDI", "GDP", "disaster"]
#columns_to_keep = ["damage", "severity", "GDP", "disaster"]
disasters = ["Harvey", "Irma", "Michael"]
for disaster in disasters:
    feat_dmg = pd.read_csv(disaster + "-feats.csv", index_col="County")
    feat_dmg = feat_dmg.loc[feat_dmg["damage"] > 1E5]
    feat_dmg["TNDI"] = feat_dmg["TNDI"].fillna(value=0) + 1
    feat_dmg["TNDA"] = feat_dmg["TNDA"].fillna(value=0) + 1
    feat_dmg = feat_dmg.loc[feat_dmg["TNDI"] > 1]
    feat_dmg.loc[:, "damage":"population"] = feat_dmg.loc[:, "damage":"population"].apply(np.log)
    feat_dmg["disaster"] = disaster
    feat_dmg["severity"] = feat_dmg["TNDI"] / feat_dmg["population"]
    feat_dmg = feat_dmg[columns_to_keep]
    feat_dmg.dropna(how="any",  inplace=True)
    if disaster == "Harvey":
        Harvey = feat_dmg
    elif disaster == "Irma":
        Irma = feat_dmg
    else:
        Michael = feat_dmg
    plt.figure()
    plt.xticks(ticks=range(len(columns_to_keep)), labels=feat_dmg.columns)
    plt.yticks(ticks=range(len(columns_to_keep)), labels=feat_dmg.columns)
    plt.imshow(feat_dmg.corr())
    plt.tight_layout()
    plt.colorbar()
    plt.title(disaster)
        
# Exploratory data analysis (raw data visualization)
# raw data visualization
all_disasters = pd.concat([Harvey, Irma, Michael], axis=0, join='outer')
all_disasters = all_disasters[columns_to_keep]
sns.pairplot(all_disasters, hue="disaster", markers=["o", "s", "D"])
plt.tight_layout()

# modeling training and prediction
train = pd.concat([Harvey, Irma], axis=0)
X_train = train.drop(labels=["damage", "disaster"], axis=1)
y_train  = train["damage"]
X_test = Michael.drop(labels=["damage", "disaster"], axis=1)
y_test  = Michael["damage"]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
from sklearn.model_selection import KFold
rs_val = []
rs_subtrain = []
cv = KFold(n_splits=5, shuffle=True)
LR = LinearRegreesor()
for train_index, val_index in cv.split(X_train_scaled):
    X_subtrain, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_subtrain, y_val = y_train[train_index], y_train[val_index]
    LR.fit(X_subtrain, y_subtrain)
    y_hat_val, _ = LR.predict(X_val)
    y_hat_subtrain, _ = LR.predict(X_subtrain)
    rs_val.append(pearsonr(y_val, y_hat_val)[0])
    rs_subtrain.append(pearsonr(y_subtrain, y_hat_subtrain)[0])
print("Average R on train using 5Fold: {:.2f}".format(np.mean(rs_val)))
print("Average R on validation using 5Fold: {:.2f}".format(np.mean(rs_subtrain)))


LR.fit(X_train_scaled, y_train)
print(LR.results.summary())
for X, y in zip([X_train_scaled, X_test_scaled], [y_train, y_test]):
    y_mean, y_error = LR.predict(X)
    plt.figure(figsize=(6, 4))
    ticklabels = [x.split('-')[1] + '-' + x.split('-')[2] for x in y.index]
    plt.errorbar(range(len(y_mean)), y_mean, yerr=y_error, fmt='none',
                 color='blue', elinewidth=1, capthick=1, capsize=6, label='95% CI')
    plt.scatter(range(len(y_mean)), y, color='red', marker='s', label='FEMA Damage')
    plt.xticks(range(len(y_mean)), ticklabels, rotation=90)
    plt.ylabel("FEMA Damage (Log)")
    plt.xlabel("County")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
y_hat_train, _ = LR.predict(X_train_scaled)
y_hat_test, _ = LR.predict(X_test_scaled)
print(stats.normaltest(y_train - y_hat_train))
plt.figure(figsize=(5, 5))
plt.hist(y_train - y_hat_train, bins=10, density=True, color='b', label='Train',
         edgecolor='black', linewidth=1.2)
print('*********************************************************')
print('LR - Pearson correlation on Train data:')
linear_fit(y_train, y_hat_train, 'FEMA Damage', 'Predicted Damage')

print(stats.normaltest(y_test - y_hat_test))
plt.figure(figsize=(5, 5))
plt.hist(y_test - y_hat_test, bins=10, density=True, color='b', label='Test',
         edgecolor='black', linewidth=1.2)
print('*********************************************************')
print('LR - Pearson correlation on Test data:')
linear_fit(y_test, y_hat_test, 'FEMA Damage', 'Predicted Damage')

print('*********************************************************')
print('TNDI - Pearson correlation on Test data:')
linear_fit(y_test, X_test['TNDI'], 'FEMA Damage', 'TNDI')


# geographic visualization
index = [x.split('-')[1] + '-' + x.split('-')[2] for x in y_test.index]
values = pd.Series(data=y_hat_test, index=index)
coords = [[-87.89, 25.00],
          [-78.88, 35.00]]
title = 'Predicted Damage'
utils.geographic_visualization(coords, values, title)

values = y_test.copy()
values.index = index
title = 'FEMA Damage'
utils.geographic_visualization(coords, values, title)













