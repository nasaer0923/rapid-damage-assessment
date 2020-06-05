# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:03:47 2019

@author: Yudi Chen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from decimal import Decimal
from scipy import stats
import datetime
import matplotlib.cm

from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os

os.environ['PROJ_LIB'] = 'C:/Users/Yudi Chen/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share/'
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

from matplotlib import rcParams
params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal',
        'font.size':12
        }
rcParams.update(params)
markers = ['o', 's', 'p', '^', '*', '+', 'd', 'x', '|', '.']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
abbrs = pd.read_csv('state abbreviations.csv', index_col='Name')
          
class Disaster(object):
    
    def __init__(self, name, date_start, date_end, disaster_kwds, damage_kwds):
        self.name = str(name)
        self.date_start = datetime.datetime(date_start[0], date_start[1], date_start[2])
        self.date_end = datetime.datetime(date_end[0], date_end[1], date_end[2])
        self.disaster_kwds = disaster_kwds
        self.damage_kwds = damage_kwds
        
    def money_to_decimal(self, money):
        return float(Decimal(re.sub(r'[^\d.]', '', money)))
    
    def lemmatize(self, x):
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(lemmatizer.lemmatize(x, pos='v')) 
    
    def get_clean_tweet(self, tweets, useless_words=[], word_len=1):
        clean_tweets = tweets.copy()
        texts = tweets['text']
        tok = WordPunctTokenizer()
        stop_words = stopwords.words('english')
        negation_dic = {"isn't":"is not", 
                        "aren't":"are not", 
                        "wasn't":"was not", 
                        "weren't":"were not",
                        "haven't":"have not",
                        "hasn't":"has not",
                        "hadn't":"had not",
                        "won't":"will not",
                        "wouldn't":"would not", 
                        "don't":"do not", 
                        "doesn't":"does not",
                        "didn't":"did not",
                        "can't":"can not",
                        "couldn't":"could not",
                        "shouldn't":"should not",
                        "mightn't":"might not", 
                        "mustn't":"must not"}
        new_texts = []
        for text in texts:
            try:
                valid_text = text.decode("utf-8-sig").replace(u"\ufffd", "")
            except:
                valid_text = text
            user_pat = r'@[A-Za-z0-9]+'
            http_pat1 = r'http://[^ ]+'
            http_pat2 = r'https://[^ ]+'
            www_pat = r'www.[^ ]+'
            combined_pat = r'|'.join([user_pat, http_pat1, http_pat2, www_pat])
            pat_removed_text = re.sub(combined_pat, '', valid_text)
            pat_removed_text = pat_removed_text.lower()
            negation_pat = re.compile(r'\b(' + '|'.join(negation_dic.keys()) + r')\b')
            negation_handled_text = negation_pat.sub(
                    lambda x: negation_dic[x.group()], pat_removed_text)
            
            letters = re.sub("[^a-zA-Z0-9]", " ", negation_handled_text)
            words = [self.lemmatize(x) for x in tok.tokenize(letters)]
            if word_len > 1:
                words = [x for x in words if len(x) > word_len]
            if useless_words:
                stop_words = stop_words + useless_words
            words = [x for x in words if not x in stop_words]
            new_texts.append((' '.join(words)).strip())
        clean_tweets['text'] = new_texts
        return clean_tweets
    
    def keyword_filter(self, texts):
        disaster_idxs = []
        damage_idxs = []
        idx = 0
        for text in texts:
            if not isinstance(text, float):
                words = text.split()
                disaster_intxn = [ii for ii in words if ii in self.disaster_kwds]
                damage_intxn = [ii for ii in words if ii in self.damage_kwds]
                if disaster_intxn:
                    disaster_idxs.append(idx)
                if damage_intxn:
                    damage_idxs.append(idx)
            idx += 1
        return disaster_idxs, damage_idxs
    
    def get_tweet(self, states):
        file = 'data\\' + self.name + '_tweets.csv'
        tweets = pd.read_csv(file)
        state_tweets = pd.concat([tweets.loc[tweets['state'] == state] for state in states], 
                                 axis=0, join='outer', ignore_index=True)
        state_tweets['county'] = state_tweets['county'].apply(
                lambda x: ' '.join(x.split()[:-1]).strip())
        state_tweets['state'] = state_tweets['state'].apply(
                lambda x: abbrs.loc[x, 'Postal Code'])
        state_tweets['county'] = (self.name + '-' + state_tweets['state'] + '-'
                    + state_tweets['county'])
        cols_to_keep = ['created_at', 'county', 'text']
        return state_tweets[cols_to_keep]
    
    def get_tweet_feat(self, tweets):
        tweets['created_at'] = tweets['created_at'].apply(pd.to_datetime)
        disaster_idxs, damage_idxs = self.keyword_filter(tweets['text'])
        disaster_twts = tweets.iloc[disaster_idxs]
        damage_twts = tweets.iloc[damage_idxs]
        print('Toally, there are {} disaster-related tweets in {}'.format(
                disaster_twts.shape[0], self.name))
        print('Toally, there are {} damage-related tweeets in {}'.format(
                damage_twts.shape[0], self.name))
        counties = set(list(disaster_twts['county']) + list(damage_twts['county']))
        feats = pd.DataFrame(index=counties, columns=['TNDI', 'TNDA'])
        for county in counties:
            DI_county_twts = disaster_twts.loc[disaster_twts['county'] == county]
            DA_county_twts = damage_twts.loc[damage_twts['county'] == county]
            bool_idxs = ((DI_county_twts['created_at'] > self.date_start) & 
                         (DI_county_twts['created_at'] < self.date_end))
            feats.loc[county]['TNDI'] = DI_county_twts[bool_idxs].shape[0]
            bool_idxs = ((DA_county_twts['created_at'] > self.date_start) & 
                         (DA_county_twts['created_at'] < self.date_end))
            feats.loc[county]['TNDA'] = DA_county_twts[bool_idxs].shape[0]
        self.tweet_feat = feats
        return feats
    
    def get_damage(self, disaster_IDs):
        all_indiv_dmg = pd.read_excel('data/damages/FEMA_Individual.xlsx',
                                      sheet_name='RI and IHP', skiprows=4)
        all_public_dmg = pd.read_csv('data/damages/FEMA_public_assistance.csv')
        indiv_dmg = pd.concat([all_indiv_dmg.loc[all_indiv_dmg['Disaster']==ID] for ID in disaster_IDs], 
                              axis=0, join='outer', ignore_index=True)
        public_dmg = pd.concat([all_public_dmg.loc[all_public_dmg['Disaster Number']==ID] for ID in disaster_IDs], 
                               axis=0, join='outer', ignore_index=True)
        public_dmg = public_dmg.loc[public_dmg['County'] != 'Statewide']
        indiv_dmg['IHP Amount'] = indiv_dmg['IHP Amount'].astype(str)
        indiv_dmg['IHP Amount'] = indiv_dmg['IHP Amount'].apply(self.money_to_decimal)
        indiv_dmg['County'] = indiv_dmg['County'].apply(
                lambda x: ' '.join(x.split(' ')[:-1]).strip())
        indiv_dmg['County'] = self.name + '-' + indiv_dmg['State'] + '-' + indiv_dmg['County']
        indiv_dmg = indiv_dmg[['County', 'IHP Amount']].groupby(['County']).sum()
            
        public_dmg['State'] = public_dmg['State'].apply(
                lambda x: abbrs.loc[x]['Postal Code'])
        public_dmg['County'] = self.name + '-' + public_dmg['State'] + '-' + public_dmg['County']
        public_dmg['Federal Share Obligated'] = (public_dmg['Federal Share Obligated']
        .astype(str)
        .apply(self.money_to_decimal))
        public_dmg = public_dmg[['County', 'Federal Share Obligated']].groupby(['County']).sum()
        dmg = indiv_dmg.merge(public_dmg, how='outer', left_index=True, right_index=True)
        dmg = dmg.fillna(value=0).sum(axis=1)
        self.dmg = dmg
        return dmg
    
    def get_demographic(self, states):
        year = self.date_start.year
        all_GDPs = pd.read_excel('data/demographics/county_gdp_all.xlsx', 
                                 sheet_name='Current Dollar GDP',
                                 headers=None,
                                 names=['county', 'state', 'linecode', 
                                        'industry_name',
                                        'GDP'],
                                 skiprows=3)
        all_GDPs = all_GDPs.loc[all_GDPs['industry_name'] == 'All Industries']
        all_pops = pd.read_csv('data/demographics/census_population.csv',
                               encoding='latin-1')
        all_pops = all_pops.loc[all_pops['SUMLEV'] == 50].rename(
                columns = {'STNAME': 'state', 'CTYNAME': 'county'})
        GDPs = pd.concat([all_GDPs.loc[all_GDPs['state'] == abbrs.loc[state]['Postal Code']] for state in states], 
                         axis=0, join='outer', ignore_index=True)
        pops = pd.concat([all_pops.loc[all_pops['state'] == state] for state in states], 
                         axis=0, join='outer', ignore_index=True)
        GDPs['county'] = self.name + '-' + GDPs['state'] + '-' + GDPs['county']
        GDPs = GDPs.set_index('county')['GDP']
        pops['state'] = pops['state'].apply(lambda x: abbrs.loc[x]['Postal Code'])
        pops['county'] = pops['county'].apply(lambda x: ' '.join(x.split(' ')[:-1]).strip())
        pops['county'] = self.name + '-' + pops['state'] + '-' + pops['county']
        pops.set_index('county', inplace=True)
        pops = pops['POPESTIMATE' + str(year)]
        all_MHIs = pd.read_excel('data/demographics/county_mhi_all.xls', skiprows=1)
        all_MHIs = pd.concat([all_MHIs.loc[all_MHIs['Postal Code'] == abbrs.loc[state]['Postal Code']] for state in states],
                             axis=0, join='outer', ignore_index=True)
        all_MHIs['county'] = [' '.join(x.split(' ')[:-1]).strip() for x in all_MHIs['county']]
        all_MHIs['county'] = self.name + '-' + all_MHIs['Postal Code'] + '-' + all_MHIs['county']
        all_MHIs.rename(columns={'median_household_income': 'MHI', 'poverty_pop': 'poverty pop'}, inplace=True)
        MHIs = all_MHIs[['county', 'MHI', 'poverty pop']]
        MHIs.set_index('county', inplace=True)
        demographic = pd.concat([GDPs, pops, MHIs], axis=1, join='inner')
        self.demographic = demographic
        return demographic
    
    def get_weather(self, states):
        weather_lst = []
        for state in states:
            path = 'data/weather/' + self.name + '/' + state + '/'
            wind = pd.read_csv(path + 'max_sustained_wind_speed.csv', index_col='date', 
                               engine='python')
            wind.fillna(value=0, inplace=True)
            precip = pd.read_csv(path + 'precipitation.csv', index_col='date')
            precip.fillna(value=0, inplace=True)
            max_wind = wind.max(axis=0).to_frame()
            max_wind.reset_index(inplace=True)
            max_wind.columns = ['county', 'wind']
            if max_wind['county'].iloc[0].split()[-1].lower() == 'county':
                max_wind['county'] = [abbrs.loc[state]['Postal Code'] + '-' + \
                         ' '.join(x.split(' ')[:-1]).strip() for x in max_wind['county']]
            else:
                max_wind['county'] = abbrs.loc[state]['Postal Code'] + '-' + max_wind['county']
            total_precip = precip.sum(axis=0).to_frame()
            total_precip.reset_index(inplace=True)
            total_precip.columns = ['county', 'precipitation']
            if total_precip['county'].iloc[0].split()[-1].lower() == 'county':
                total_precip['county'] = [abbrs.loc[state]['Postal Code'] + '-' \
                             + ' '.join(x.split(' ')[:-1]).strip() for x in total_precip['county']]
            else:
                total_precip['county'] = abbrs.loc[state]['Postal Code'] + '-' + total_precip['county']
            weather_lst.append(pd.merge(max_wind, total_precip, 
                                        how='outer', on='county'))
        weather = pd.concat([x for x in weather_lst], 
                            axis=0, join='outer', ignore_index=True)
        weather['county'] = self.name + '-' + weather['county']
        weather.set_index('county', drop=True, inplace=True)
        self.weather = weather
        return weather


def correlation_analysis(df, flag, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    if flag == 'img':
        img = ax.imshow(df.corr(), cmap='jet')
        ax.set_xticks(range(df.shape[1]))
        ax.set_yticks(range(df.shape[1]))
        ax.set_xticklabels(list(df.columns), rotation=90)
        ax.set_yticklabels(list(df.columns))
        ax.set_title(title)
        plt.colorbar(img)
        fig.tight_layout()
    elif flag == 'scatter':
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_fit = slope * x + intercept
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
        ax.scatter(x, y, label='Data')
        ax.plot(x, y_fit, '-r', lw=1, label='Fit: R = {} & P = {}'.format(
                str(round(r_value, 2)),
                str(round(p_value, 3))))
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[0])
        ax.legend(loc='lower right')
        ax.set_title(title)
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    else:
        print('Wrong input of flag: img or scatter!')
    return fig, ax


def geographic_visualization(coords, values, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    m = Basemap(resolution='i',
                projection='merc',
                llcrnrlon=coords[0][0], 
                llcrnrlat=coords[0][1],
                urcrnrlon=coords[1][0], 
                urcrnrlat=coords[1][1]
                )
    m.drawcoastlines()
    m.drawcountries(color='black')
    m.readshapefile('data/geography/st99_d00', 'us_states', linewidth=2)
    m.readshapefile('data/geography/cb_2015_us_county_500k', 'us_counties', linewidth=0.5)
    us_counties_info = m.us_counties_info
    us_counties_bdy = m.us_counties
        
    counties = list(values.index)    
    counties_bdy = []
    counties_value = []
    for info, bdy in zip(us_counties_info, us_counties_bdy):
        idx = abbrs['FIPS'] == int(info['STATEFP'])
        county = abbrs.loc[idx]['Postal Code'] + '-' + info['NAME']
        county = county.iloc[0]
        if county in counties:
            counties_bdy.append(bdy)
            counties_value.append(values.loc[county])
            
    poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape), True) for shape in counties_bdy],
            'values': counties_value})
    cmap = plt.get_cmap('Reds')
    pc = PatchCollection(poly['shapes'], zorder=2, alpha=1.0)
    norm = Normalize()
    pc.set_facecolor(cmap(norm(poly['values'].fillna(0).values)))
    ax.add_collection(pc)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_clim(vmin=3, vmax=10)
    mapper.set_array(poly['values'])
    cbar = m.colorbar(mapper, location='right', pad=0.05)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(str(title.split(' ')[0])+ '-Geographic Visualization.tiff', dpi=300)
    return fig, ax

if __name__ == '__main__':
    damage_kwds = open('damage keywords.txt').read().split('\n')
    disaster_kwds = ['hurricane', 'michael', 'hurricanemichael', 'storm']
    disaster_IDs = [4399, 4400]
    states = ['Florida', 'Georgia']
    disaster_name = 'Michael'
    date_start = (2018, 10, 1)
    date_end = (2018, 10, 30)
    disaster = Disaster('Michael', date_start, date_end, disaster_kwds, damage_kwds)
    raw_tweets = disaster.get_tweet(states)
    cleaned_tweets = disaster.get_clean_tweet(raw_tweets, useless_words=[], word_len=1)
    tweet_feat = disaster.get_tweet_feat(cleaned_tweets)
    damage = disaster.get_damage(disaster_IDs).to_frame()
    damage.columns = ['damage']
    demographic = disaster.get_demographic(states)
    weather = disaster.get_weather(states)
    feats = damage.join([tweet_feat, demographic, weather], how='left')