# importation libraries
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly import graph_objs as go
import plotly.express as px
import plotly.io as pio
import plotly
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
import folium 
from folium import plugins
import seaborn as sns
import pandas as pd 
import random
from io import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg

from io import BytesIO
import math
import json
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
# %matplotlib inline 
from geopy.geocoders import Nominatim
geolocator = Nominatim()
from plotly.subplots import make_subplots
import io
import base64

class Covid19:

    def __init__(self):
        pass


    # load and preparation 

    # file_name = 'E:\Covid19\covid-west-africa-dit\Covid19SN_datas.xlsx' 
    def covid(self,file_name):
        df = pd.read_excel(file_name, index_col=0)
        df.shape
        df = df.reset_index()

        date_c = df.groupby('date')['cas_positif','importes','contacts','communautaires'].sum().reset_index()

        plt = make_subplots(rows=1, cols=4, subplot_titles=("cas_positif", "importes", "contacts",'communautaires'))

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

        plt.append_trace(trace1, 1, 1)
        plt.append_trace(trace2, 1, 2)
        plt.append_trace(trace3, 1, 3)
        plt.append_trace(trace4, 1, 4)
        plt.update_layout(template="plotly_dark",title_text = '<b>Global West Africa  Spread of the Coronavirus Over Time </b>',
                        font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
       
        graphJSON = json.dumps(plt, cls=plotly.utils.PlotlyJSONEncoder)
        print(graphJSON)

        return graphJSON
        # return plt
     


covid = Covid19()
file_name = 'E:\Covid19\covid-west-africa-dit\Covid19SN_datas.xlsx'
covid.covid(file_name)
