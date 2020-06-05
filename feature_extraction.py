# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:53:14 2019

@author: JID002
"""

import pandas as pd
import utils

Harvey = {'disaster_name': 'Harvey',
          'date_start': (2017, 8, 25),
          'date_end': (2017, 8, 30),
          'states': ['Texas'],
          'disaster_IDs': [4332],
          'disaster_kwds': ['hurricane', 
                            'hurricaneharvey',
                            'harvey'],
          'coords': [[25.3, -106.6],
                     [37.0, -89.5]]
                       }
Irma = {'disaster_name': 'Irma',
        'date_start': (2017, 9, 10),
        'date_end': (2017, 9, 12),
        'states': ['Florida'],
        'disaster_IDs': [4337],
        'disaster_kwds': ['hurricane', 
                          'hurricaneirma',
                          'irma'],
        'coords': [[24.3, -88.0],
                   [35.1, -79.8]]
        }
Michael = {'disaster_name': 'Michael',
           'date_start': (2018, 10, 9),
           'date_end': (2018, 10, 15),
           'states': ['Florida', 'Georgia'],
           'disaster_IDs': [4399, 4400],
           'disaster_kwds': ['hurricane', 
                             'hurricanemichael',
                             'michael'],
            'coords': [[24.3, -88.0],
                       [35.1, -79.8]]
            }
disaster_dicts = [Harvey, Irma, Michael]
damage_kwds = open('damage keywords.txt').read().split('\n')

for disaster_dict in disaster_dicts:
    disaster_kwds = disaster_dict['disaster_kwds']
    disaster_IDs = disaster_dict['disaster_IDs']
    states = disaster_dict['states']
    disaster_name = disaster_dict['disaster_name']
    date_start = disaster_dict['date_start']
    date_end = disaster_dict['date_end']
    disaster = utils.Disaster(disaster_name, date_start, date_end, disaster_kwds, damage_kwds)
#    raw_tweets = disaster.get_tweet(states)
#    cleaned_tweets = disaster.get_clean_tweet(raw_tweets, useless_words=[], word_len=1)
#    cleaned_tweets.to_csv(disaster_name + '-cleaned tweets.csv')
    cleaned_tweets = pd.read_csv(disaster_name + '-cleaned tweets.csv', index_col=0)
    tweet_feat = disaster.get_tweet_feat(cleaned_tweets)
    damage = disaster.get_damage(disaster_IDs).to_frame()
    damage.columns = ['damage']
    demographic = disaster.get_demographic(states)
    weather = disaster.get_weather(states)
    feats = damage.join([tweet_feat, demographic, weather], how='left')
    feats.to_csv(disaster_name + '-feats.csv')
    
    