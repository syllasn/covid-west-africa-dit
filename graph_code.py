# importation Libraries
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

import plotly
from matplotlib.backends.backend_agg import FigureCanvasAgg

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots
import folium 
from folium import plugins
from tqdm.notebook import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')


import seaborn as sns
import pandas as pd 
import random
import math
from scipy.stats import expon
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
import json

from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
import scipy.optimize as optim

#from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn')
# %matplotlib inline 
import geopy
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="myapp2")
from plotly.subplots import make_subplots
class Graph:
    def __init__(self):
         pass


    # load data 
    file_name = 'Covid19SN_datas.xlsx' 
    df = pd.read_excel(file_name, index_col=0)
    df.shape
    df = df.reset_index()
    df.head()

    # fonction graphe
    # cette fonction recois une date de debut, une date de fin et des options 
    def graphe_sn(self,ct, df, date_deb ='2020-03-02' , date_fin='2020-03-20', option_1='Confirme', option_2='Contact', option_3='Importe', option_4='Communautaire',option_5='Recovered', option_6='Dead'):
        mask = (df['Date'] >= date_deb) & (df['Date'] <= date_fin)
        df_test = df.loc[mask]
        date_c = df_test[['Date','Confirmed_cases','Imported_cases','contacts','Communities_cases','Recovered','Dead', 'evacuat_out' ]]
        #print(date_c.head())
        fig = make_subplots(rows=1, cols=1)
        
        if option_1=='Confirme':
            trace1 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['Confirmed_cases'],
                    name="cas_positif",
                    line_color='red',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace1, 1, 1)

        
        if  option_2 == 'Contact':
            trace2 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['contacts'],
                    name="contacts",
                    line_color='blue',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace2, 1, 1)
            
        
        if  option_3 == 'Importe':
            trace3 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['Imported_cases'],
                    name="Imported_cases",
                    line_color='orange',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace3, 1, 1)

        
        if  option_4 == 'Communautaire':
            trace4 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['Communities_cases'],
                    name="Communities_cases",
                    line_color='darkred',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace4, 1, 1)

        
        if  option_5 == 'Recovered':
            trace5 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['Recovered'],
                    name="cas_gueris",
                    line_color='green',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace5, 1, 1)

        
        if  option_6 == 'Dead':
            trace6 = go.Scatter(
                    x=date_c['Date'],
                    y=date_c['Dead'],
                    name="cas_decedes",
                    line_color='orange',
                    mode='lines+markers',
                    opacity=0.8)
            fig.append_trace(trace4, 1, 1)

        
        
        fig.append_trace(trace1, 1, 1)
        # fig.append_trace(trace2, 1, 1)
        # fig.append_trace(trace3, 1, 1)
        # fig.append_trace(trace4, 1, 1)
        # fig.append_trace(trace5, 1, 1)
        # fig.append_trace(trace6, 1, 1)




        fig.update_layout(template="plotly_dark",title_text = '<b>   Coronavirus au Senegal en temps reel  </b>',
                    font=dict(family="Arial, Balto, Courier New, Droid Sans",color='white'))
        #fig.show()
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        #print(graphJSON)

        return graphJSON
        #return None


    #  Exemple d appel de la fonction
  
    #graphe_sn('Senegal', df, date_deb ='2020-03-02' , date_fin='2020-04-26', option_1='Confirme', option_2='Contact', option_3='Importe', option_4='Communautaire',option_5='Recovered', option_6='Dead' )


file_name = 'Covid19SN_datas.xlsx' 
df = pd.read_excel(file_name, index_col=0)
df.shape
df = df.reset_index()
df.head()
graph = Graph()
graph.graphe_sn('Senegal', df, date_deb ='2020-03-02' , date_fin='2020-04-26' ,option_1='Confirme',option_2='Cnfirme',option_3='Importe',option_4='Communautaire',option_5='Confirm',option_6='Confirm')
