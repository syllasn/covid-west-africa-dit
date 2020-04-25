# importation des libraries
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
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
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


class Modelisation:
    def __init__(self):
        pass

    # Lecture et preparation des donnees
    file_name = 'Covid19SN_datas.xlsx'
    df_senegal = pd.read_excel(file_name, index_col=0)
    df_senegal.shape
    df_senegal = df_senegal.reset_index()
    df_senegal['Date'] = pd.to_datetime(df_senegal['Date'])
    df_senegal.tail()

    # Modelisation 
    # fonction get_daily_var 
    def get_daily_var(self,ct,df):
        
        """
        This function takes the countries name and database in parameters.
        It returns a datframe with daily cases value and date as index
        """
        #Prepare country data
        ct_df = df[df['Country'] == ct]
        first_date = ct_df[ct_df['Confirmed_cases'] != 0]['Date'].values[0]
        confirmed = ct_df[ct_df['Date'] >= first_date]
        confirmed_cases = confirmed[['Date', 'Daily_case']].set_index('Date')
        
        
        return confirmed_cases


    # fonction preparation donnees
    def get_pred_data (conf):
            
        """
        This function takes the daily dataframe and return it as arrays value
        for modelling purpose
        """
        #rearange dataframe values in array format
        cases_daily = np.array(conf).reshape(-1,1)
        cases_cum = np.array(conf.cumsum()).reshape(-1,1)
        days_since = np.arange(len(conf)).reshape(-1,1)
        
        return  cases_daily, cases_cum, days_since

    # fonction du temps a predire
    def get_pred_date (n, conf):
        
        """
        This function generates n days for prediction
        """
        future_forcast = np.arange(len(conf) + n).reshape(-1,1)
        start_date = conf.index[0]
        future_forcast_dates = [(start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y') for i in range(len(future_forcast))] 
        
        return future_forcast,future_forcast_dates

    # Fonction : bayesian model
    def bayes_model(ct, df, n, mode = 'normal', plot = False):
        
        """
        This function defines init and run the bayesian model
        """
        
        cases_df = get_daily_var(ct,df)   
        cases_daily, cases_cum, days_since = get_pred_data(cases_df)
        future_forcast,future_forcast_dates = get_pred_date(n, cases_df)
        
        # bayesian ridge (Init and fit)
        tol = [1e-4, 1e-3, 1e-2]
        alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
        alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
        lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
        lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

        bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

        bayesian = BayesianRidge(compute_score=True)
        bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)

        if mode == 'normal':
            cases = cases_daily
            X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(days_since, cases, test_size=0.25, shuffle=False)  
            bayesian_search.fit(X_train_daily, y_train_daily)
            bayesian_daily_params = bayesian_search.best_estimator_
            bayesian_pred = bayesian_daily_params.predict(X_test_daily)
            print('MAE:', mean_absolute_error(bayesian_pred, y_test_daily))
            print('MSE:',mean_squared_error(bayesian_pred, y_test_daily))
            bayesian_pred_future = bayesian_daily_params.predict(future_forcast)       
        
            
        if mode == 'cum':
            cases = cases_cum
            X_train_cum, X_test_cum, y_train_cum, y_test_cum = train_test_split(days_since, cases, test_size=0.25, shuffle=False)  
            bayesian_search.fit(X_train_cum, y_train_cum)
            bayesian_cum_params = bayesian_search.best_estimator_
            bayesian_pred = bayesian_cum_params.predict(X_test_cum)
            print('MAE:', mean_absolute_error(bayesian_pred, y_test_cum))
            print('MSE:',mean_squared_error(bayesian_pred, y_test_cum))
            bayesian_pred_future = bayesian_cum_params.predict(future_forcast)       
            
            
        if plot == True:
            plt.figure(figsize=(20, 12))
            plt.plot(future_forcast_dates[:-n], cases)
            plt.plot(future_forcast_dates, bayesian_pred_future, linestyle='dashed', color='green')
            plt.title(f'Nombre de cas confirmes au {ct} en fonction du temps', size=30)
            plt.xlabel('Temps', size=30)
            plt.ylabel('Nombre de cas', size=30)
            plt.legend(['Cas Confirmes', 'Bayesian Ridge Regression Predictions'], prop={'size': 20})
            plt.xticks(size=20, rotation= 90)
            plt.yticks(size=20)
            plt.show()
    # Future predictions using Linear Regression \n",
        print('bayesian model future predictions:')
        zipped =zip(future_forcast_dates, np.round(bayesian_pred_future) )
        # Converting to list
        zipped = list(zipped)
        #Using sorted and lambda
        res = sorted(zipped, key = lambda x: x[1]) 
        # printing result \n",
        df_result = pd.DataFrame(res)
        df_result.columns=('Date','Predictions')
        print(df_result.tail(n))
            
        return None      

    # appel de la fonction bayesian
    #bayes_model('Senegal', df, n=10, mode = 'cum', plot = True)
    #bayes_model('Senegal', df_senegal, n=10, mode = 'cum', plot = True)

    # Polynomial Regression 
    def polynom_regression_model(self,ct, df, n=5, p=2, plot = False):
        """
        This function defines init and run the polynomial regression model
        """
        #Data prep
        cases_df = self.get_daily_var(ct,df)   
        _, cases_cum, days_since = get_pred_data(cases_df)
        future_forcast,future_forcast_dates = get_pred_date(n, cases_df)
        poly = PolynomialFeatures(degree = p)
        
        #Split Data
        X_train_cum, X_test_cum, y_train_cum, y_test_cum = train_test_split(days_since, cases_cum, test_size=0.25, shuffle=False)
        
        #fit and init model                                                                   
        X_poly = poly.fit_transform(X_train_cum)
        poly.fit(X_poly, y_train_cum)
        lin2 = LinearRegression()
        lin2.fit(X_poly, y_train_cum)
        cum_poly_pred_future = lin2.predict(poly.fit_transform(future_forcast))
                                                                            
        #ploting_results
        if plot == True:
                                                                            
            plt.figure(figsize=(20, 12))
            plt.plot(future_forcast_dates[:-n], cases_cum, color='blue')
            plt.plot(future_forcast_dates, cum_poly_pred_future, linestyle='dashed', color='green')
            plt.title(f'Nombre de cas confirmes au {ct} en fonction du temps', size=30)
            plt.xlabel('Temps', size=20)
            plt.ylabel('Nombre de cas', size=30)
            plt.legend(['Cas Confirmes', 'polynomial Regression Predictions'], prop={'size': 30})
            plt.xticks(size=20, rotation = 90)
            plt.yticks(size=20)
            plt.show() 
        # Future predictions using Linear Regression 
        print('Polynomial future predictions:')
        zipped =zip(future_forcast_dates, np.round(cum_poly_pred_future) )
        # Converting to list 
        zipped = list(zipped) 
        # Using sorted and lambda 
        res = sorted(zipped, key = lambda x: x[1])
        # printing result \n",
        df_result = pd.DataFrame(res)
        df_result.columns=('Date','Predictions')
        print(df_result.tail(n))      
                                                                            
        return None

    # appel de la fonction polynomial 
   # polynom_regression_model('Senegal', df_senegal, n=10, p=2, plot = True)
    #polynom_regression_model('Senegal', df_senegal, n=10, p=3, plot = True)


    # Fonction Model logistique
    def logistic_growth_model(ct, df, n=5, size = 3):
        
        cases_df = get_daily_var(ct,df)
        future_forcast,future_forcast_dates = get_pred_date(n, cases_df)
        
        # Define funcion with the coefficients to estimate
        def my_logistic(t, a, b, c):
            return c / (1 + a * np.exp(-b*t))
        
        #Randomly initialize the coefficients
        p0 = np.random.exponential(size=3)
        
        # Set min bound 0 on all coefficients, and set different max bounds for each coefficient
        bounds = (0, [100, 3., 1000])
        
        # Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients
        x = np.arange(len(cases_df)) + 1
        y = np.array(cases_df.cumsum()['Daily_case'])
        
        (a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)
        
        # Redefine the function with the new a, b and c
        def my_logistic(t):
            return c / (1 + a * np.exp(-b*t))
        plt.figure(figsize=(20,12))
        plt.scatter(future_forcast_dates[0:len(y)], y)
        plt.plot(x, my_logistic(x),color = 'red')
        plt.title('Logistic Model vs Real Observations of Senegal Coronavirus')
        plt.legend([ 'Logistic Model', 'Real data'])
        plt.xlabel('Time')
        plt.ylabel('Infections')
        plt.xticks(rotation = 90)
        plt.show()
        
        plt.figure(figsize=(20,12))
        plt.scatter(x, y)
        plt.plot(future_forcast_dates, my_logistic(future_forcast),color = 'red')
        plt.title(f'Logistic Model vs Real Observations of {ct} Coronavirus')
        plt.legend([' Predict Logistic Model ', 'Real data'])
        plt.xlabel('Time')
        plt.ylabel('Infections')
        plt.xticks(rotation = 90)
        plt.show()
        
        # The time step at which the growth is fastest
        t_fastest = np.log(a) / b
        t_fastest
        print(np.round(t_fastest))
        future_forcast_dates[27]
        # First way to find the y of the fastest growth moment
        y_fastest = c / 2
        y_fastest
        # Second way to find the y of the fastest growth moment
        print(my_logistic(t_fastest))
        # Future predictions using Linear Regression 
        print('logistic future predictions:')
        zipped =zip(future_forcast_dates, np.round(my_logistic(future_forcast)) )
        # Converting to list 
        zipped = list(zipped) 
        # Using sorted and lambda 
        res = sorted(zipped, key = lambda x: x[1]) 
        # printing result \n",
        df_result = pd.DataFrame(res)
        df_result.columns=('Date','Predictions')
        print(df_result.tail(n)) 

        return None
    # appel de la fonction logistique
    #logistic_growth_model('Senegal', df_senegal, n=10, size = 3)

    # Fonction model Prophet
    def prophet_model(ct, df,n, mode, method, plot = False):
            #Data prep
            
            
            first_forecasted_date = sorted(list(set(df['Date'].values)))[-n]


            #forecast_dfs = []
            #absolute_errors = [] # collate absolute errors so that we can find MAE later on
        # data preparation for forecast with Prophet at state level
            if mode == 'normal':
                state_df = df[['Date', 'Daily_case']]
                state_df.columns = ['ds','y']
                state_df['ds'] = pd.to_datetime(state_df['ds'])
                #state_df_val = state_df[(state_df['ds'] >= pd.to_datetime(first_forecasted_date))] # validation set
                #state_df = state_df[(state_df['ds'] < pd.to_datetime(first_forecasted_date))] # train set
            
            if mode == 'cum':
                                state_df = df[['Date', 'Confirmed_cases']]
                                state_df.columns = ['ds','y']
                                state_df['ds'] = pd.to_datetime(state_df['ds'])
                                #state_df_val = state_df[(state_df['ds'] >= pd.to_datetime(first_forecasted_date))] # validation set
                                #state_df = state_df[(state_df['ds'] < pd.to_datetime(first_forecasted_date))] # train set
            
            
            if method == 'default':
                                
                                m = Prophet()
                                m.fit(state_df)
                                m_cv = cross_validation(m, initial='1 days', period='14 days', horizon = '21 days')
                                df_p = performance_metrics(m_cv)
                                model_cv = Prophet()
                                model_cv.fit(m_cv)
                                        
                            
                                future = model_cv.make_future_dataframe(periods=n)
                                forecast = model_cv.predict(future)
                                # evaluate forecasts with validation set and save absolute errors to absolute_errors
                                forecast_df = forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']]
                                result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]
                                result_val_df = pd.merge(result_df,state_df, on=['ds'])
                                result_val_df['abs_diff'] = (result_val_df['y'] - result_val_df['yhat']).abs()
                                absolute_errors = result_val_df['abs_diff'].values
                                #print(result_val_df.tail())
                                #print(future.tail())


                                
            if method == 'custom':
                                m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                                m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
                                m.add_seasonality(name='weekly', period=7, fourier_order=21)
                                m.add_seasonality(name='daily', period=1, fourier_order=3)
                                m.fit(state_df)
                                future = m.make_future_dataframe(periods=n)
                                forecast = m.predict(future)
                                #print(forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail(n))
                                # evaluate forecasts with validation set and save absolute errors to absolute_errors
                                forecast_df = forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']]
                                result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]
                                result_val_df = pd.merge(result_df,state_df, on=['ds'])
                                result_val_df['abs_diff'] = (result_val_df['y'] - result_val_df['yhat']).abs()
                                absolute_errors = result_val_df['abs_diff'].values
                                #print(result_val_df.tail())
                                #print(future.tail())
                                #N = len(absolute_errors)
                                #mean_absolute_error = sum(absolute_errors)/N
                                #print('The mean absolute error for ' + str(n) + ' days of forecasts with the default Prophet model is: ' + str(round(mean_absolute_error, 2))) # round to 2 decimal places




                                
            if plot == True:
                
                plt.figure(figsize=(20, 12))
                recovered_forecast_plot = m.plot(forecast)
                #plt.plot(future_forcast_dates[:-n], cases)
                #plt.plot(future_forcast_dates, bayesian_pred_future, linestyle='dashed', color='green')
                plt.title(f'Nombre de cas confirmes au {ct} en fonction du temps', size=30)
                plt.xlabel('Temps', size=30)
                plt.ylabel('Nombre de cas', size=30)
                plt.legend(['Cas Confirmes', 'Prophet Predictions'], prop={'size': 20})
                plt.xticks(size=20, rotation= 90)
                plt.yticks(size=20)
                plt.show()
                forecast_components = m.plot_components(forecast)

            # Future predictions using Linear Regression \n",
            print('Prophet model future predictions:')
            print(forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail(n))
            
            
            





            return None




model = Modelisation()
file_name = 'Covid19SN_datas.xlsx'

df_senegal = pd.read_excel(file_name, index_col=0)
df_senegal.shape
df_senegal = df_senegal.reset_index()
df_senegal['Date'] = pd.to_datetime(df_senegal['Date'])
df_senegal.tail()
senegal = 'Senegal'
print(model.get_daily_var(senegal,df_senegal))
    # appel fonction Prophet
    # prophet_model('Senegal', df_senegal, n=14, mode ='cum', method ='default', plot =True)
    # prophet_model('Senegal', df_senegal, n=14, mode ='cum', method ='custom', plot =True)
    # prophet_model('Senegal', df_senegal, n=14, mode ='normal', method ='default', plot =True)
    # prophet_model('Senegal', df_senegal, n=14, mode ='normal', method ='custom', plot =True)


