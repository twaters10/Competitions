# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:48:42 2021

@author: water
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os

path = 'C:\\Users\\water\\Desktop\\Kaggle\\Predict Future Sales\\Data\\'
train = pd.read_csv(path + 'sales_train.csv')
test = pd.read_csv(path + 'test.csv')
items = pd.read_csv(path + 'items.csv')
item_cat = pd.read_csv(path + 'item_categories.csv')
shops = pd.read_csv(path + 'shops.csv')
submit = pd.read_csv(path + 'sample_submission.csv')
full_train = pd.read_csv(path + 'train_full.csv')

# Initial Data Exploration
train.head()
items.head()
item_cat.head()
shops.head()
full_train.head()
full_train.describe()
full_train.columns
full_train['item_price'].describe()
full_train['item_cnt_day'].describe()
full_train['date'].head()
full_train['item_name'].unique()
full_train['item_category_name'].unique()
sns.distplot(full_train['item_price'])
full_train['date'] = pd.to_datetime(full_train['date']).dt.date

# Shop Data Exploration
shops = full_train['shop_name'].unique()
for shop in shops:
    print(shop)
    print(full_train[full_train['shop_name'] == shop]['item_cnt_day'].describe())

# sort by date
full_train = full_train.sort_values(by = 'date')


# test on item СЕКСУАЛЬНЫЕ ХРОНИКИ ФРАНЦУЗСКОЙ СЕМЬИ
train_1 = full_train[full_train['item_name'] == 'СЕКСУАЛЬНЫЕ ХРОНИКИ ФРАНЦУЗСКОЙ СЕМЬИ']
sns.lineplot(data = full_train, x = 'date', y = 'item_price', hue = 'shop_id')






