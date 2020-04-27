
from flask import Flask ,jsonify,request, send_from_directory,render_template, Response, send_file, make_response
from app import Covid19
import sys
import pandas as pd 

from flask_cors import CORS
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import datetime
import os
# import StringIO
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from graph_code import Graph


import json
import base64

app = Flask(__name__)
CORS(app)
dataframe = Covid19()
graph = Graph()


@app.route('/graph', methods=['POST'])
def get_dashboard():
    # bar=dataframe.covid('Covid19SN_datas.xlsx')
    file_name = 'Covid19SN_datas.xlsx' 
    df = pd.read_excel(file_name, index_col=0)
    df.shape
    df = df.reset_index()
    values = request.json
    print(values)
    if not values :
        response = {
            'message':'no data found' 

        }
        return jsonify(response)  ,400
    require_values = ['date_deb','date_fin','option']
    if not all( field in values for field in require_values):
        response = {
            'message': 'Required data is missing'
        }
        return jsonify(response)  ,400
    #url = values['date_deb','date_fin','option']
    date_deb = values['date_deb']
    print(date_deb)
    date_fin = values['date_fin']
    print(date_fin)
    option = values['option']
     #option_1='Confirme', option_2='Contact', option_3='Importe', option_4='Communautaire',option_5='Recovered', option_6='Dead'
    if(option=='Confirme'):
        option_1 = option
        option_2='a'
        option_3='b'
        option_4 ='cc '
        option_5='dd'
        option_6='dd'
    elif(otpion=='Contact'):
        option_2=option
        option_1='a'
        option_3='b'
        option_4 ='cc '
        option_5='dd'
        option_6='dd'
    elif(option=='Importe'):
        option_3=option
        option_2='a'
        option_1='b'
        option_4 ='cc '
        option_5='dd'
        option_6='dd'
    elif(option=='Communautaire'):
        option_4=option
        option_2='a'
        option_3='b'
        option_1 ='cc '
        option_5='dd'
        option_6='dd'
    elif(option=='Recovered'):
        option_5=option
        option_2='a'
        option_3='b'
        option_4 ='cc '
        option_1='dd'
        option_6='dd'
    elif(option=='Dead'):
        option_6=option
        option_2='a'
        option_3='b'
        option_4 ='cc '
        option_5='dd'
        option_1='dd'
    print(option)
    data = graph.graphe_sn('Senegal',df,date_deb =date_deb , date_fin=date_fin ,option_1=option_1,option_2=option_2,option_3=option_3,option_4=option_4,option_5=option_5,option_6=option_6)
    print(data)
    response = {
        'data': data
    }
    return jsonify(response)  , 200
    # canvas = FigureCanvasAgg(fig)  
    # response= Response( mimetype='image/png') 
    # canvas.print_png(response)
    

    # respons = {

    # } 
    # response = HttpResponse(content_type='image/png')
    # canvas.print_png(response)
    # matplotlib.pyplot.close(f)   
    # return response
    
    #return render_template("interface.html",  figure = fig)

     



# ...
# port = int(os.environ.get('PORT', 5000))
# ...
# app.run(host='0.0.0.0', port=port, debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(host = '0.0.0.0' ,port = 5000 )