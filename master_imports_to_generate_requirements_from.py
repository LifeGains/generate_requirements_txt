# User to mod this:
# %cd /content/drive/My\ Drive/Github/Datasets/data-society-uber-pickups-in-nyc
# ---------------------------------
# from google.colab import drive
# drive.mount('/content/drive')

# Generate requirements.txt file
# ! pip install pipreqs
# ! pipreqs .

# ! pip install shap
# ! pip install stargazer
# ! pip install scikit-plot

# Math and data processing
import pandas as pd
import numpy as np
from pprint import pprint
import itertools
import random
import math
import calendar
import datetime
import json
import re
import importlib
import subprocess
from holidays import UnitedStates

# Config
from pathlib import Path
import inspect
import glob
import os
import sys
from urllib.request import urlretrieve

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline
import plotly.graph_objs as go
# %matplotlib inline

# Statistics
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from stargazer.stargazer import Stargazer

# ML, Alphabetical so you can add more without losing previous info.
# General for anything not imported here already. Eg. get_scorer_names()
import sklearn
from sklearn import set_config, ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree, discriminant_analysis
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, IsolationForest, StackingRegressor, BaggingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Perceptron, LinearRegression, LogisticRegression, ElasticNet, BayesianRidge, Lasso, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, SGDClassifier
from sklearn.metrics import make_scorer, mean_squared_error, median_absolute_error, PredictionErrorDisplay, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score, auc, classification_report, confusion_matrix, log_loss, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit, RepeatedStratifiedKFold, RepeatedKFold, KFold, train_test_split, cross_validate, cross_val_score, learning_curve, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer, QuantileTransformer
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# Upsample and downsample if y is imbalanced.
from sklearn.utils import class_weight, resample
from scikitplot.estimators import plot_feature_importances
from mlxtend.regressor import StackingRegressor
from xgboost import XGBClassifier, XGBRegressor, cv as xgbcv 
from lightgbm import LGBMRegressor, cv as lgbmcv
import shap
# Prettyprint of Regression Residuals and Classification Errors
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, PrecisionRecallCurve, ClassPredictionError, DiscriminationThreshold
from yellowbrick.model_selection import LearningCurve, ValidationCurve, FeatureImportances
from yellowbrick.regressor import PredictionError, ResidualsPlot

# Time Series
from prophet import Prophet
# Neural Networks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup

# Set Numpy, Pandas, Matplotlib settings, turn off scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
plt.style.use('seaborn')
np.set_printoptions( suppress = True, formatter = {'float_kind' :'{:0.2f}'. format})
# Visualize diagrams for sklearn objects
set_config(display = 'diagram')

# Set a globally used random seed for every method
rand_seed = 0

# Change to 0 to disable below graph functions
enable_graphs = 1