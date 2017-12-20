import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

from scipy import stats
from scipy.stats import boxcox

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from scrapy.selector import Selector
from scrapy.http import HtmlResponse

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

from sqlalchemy import create_engine

import re

import missingno as msno

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, log_loss, f1_score, fbeta_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample