# importation libraries
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
import folium 
from folium import plugins
import seaborn as sns
import pandas as pd 
import random
import math
from scipy.stats import expon
import time
from sklearn.svm import SVR
import datetime
import operator 
from pandas import DataFrame
from pandas import read_csv
import scipy.optimize as optim
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn')
%matplotlib inline 
from geopy.geocoders import Nominatim
geolocator = Nominatim()
from plotly.subplots import make_subplots


# load and preparation data
file_name = 'Covid19SN_datas.xlsx' 
df = pd.read_excel(file_name, index_col=0)
df.shape
df = df.reset_index()

date_c = df.groupby('date')['cas_positif','importes','contacts','communautaires'].sum().reset_index()

fig = make_subplots(rows=1, cols=4, subplot_titles=("cas_positif", "importes", "contacts",'communautaires'))

trace1 = go.Scatter(
                x=date_c['date'],
                y=date_c['cas_positif'],
                name="cas_positif",
                line_color='orange',
                mode='lines+markers',
                opacity=0.8)
trace2 = go.Scatter(
                x=date_c['date'],
                y=date_c['importes'],
                name="importes",
                line_color='red',
                mode='lines+markers',
                opacity=0.8)

trace3 = go.Scatter(
                x=date_c['date'],
                y=date_c['contacts'],
                name="contacts",
                mode='lines+markers',
                line_color='green',
                opacity=0.8)

trace4 = go.Scatter(
                x=date_c['date'],
                y=date_c['communautaires'],
                name="communautaires",
                line_color='blue',
                mode='lines+markers',
                opacity=0.8)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 1, 4)
fig.update_layout(template="plotly_dark",title_text = '<b>Global West Africa  Spread of the Coronavirus Over Time </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
fig.show()
