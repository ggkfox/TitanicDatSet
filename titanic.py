# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

###########################################

# Import test and train datasets
df_train = pd.read_csv('datasets/titanic/train.csv')
df_test = pd.read_csv('datasets/titanic/test.csv')

# View first lines of training data
print (df_train.info())
df_train.head(n=4)

###########################################

df_train.hist(column='Survived')

###########################################

df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('datasets/titanic/no_survivors.csv', index=False)
df_test[['PassengerId', 'Survived']].head()

###########################################

df_test['Survived'] = df_test.Sex == 'female'
print (df_test.head())

###########################################

df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()

###########################################

df_test[['PassengerId', 'Survived']].to_csv('datasets/titanic/men_died.csv', index=False)
print (df_test[['PassengerId', 'Survived']].head())
