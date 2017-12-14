import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

from sklearn.utils import resample

from scrapy.selector import Selector
from scrapy.http import HtmlResponse

import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

from sqlalchemy import create_engine

import re

import missingno as msno

from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler