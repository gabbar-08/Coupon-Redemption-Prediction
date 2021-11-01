import pandas as pd
import numpy as np
import datetime
from pandas.core.indexes import category

from pandas.core.algorithms import mode
from sklearn import model_selection
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', None)
filename = 'model/finalized_model.sav'


