
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
    print(values['option']['confirmes'])
    if not values :
        response = {
            'message':'no data found' 
        }
        print(response['message'])
        return jsonify(response)  ,400
    require_values = ['date_deb','date_fin','option']
    if not all( field in values for field in require_values):
        response = {
            'message': 'Required data is missing'
        }
        print(response['message'])
        return jsonify(response)  ,400
    #url = values['date_deb','date_fin','option']
    date_deb = values['date_deb']
    print(date_deb)
    date_fin = values['date_fin']
    print(date_fin)
    option = values['option']
    print(option['confirmes'])
     #option_1='Confirme', option_2='Contact', option_3='Importe', option_4='Communautaire',option_5='Recovered', option_6='Dead'
    #option_1='confirmesvide'
    if(option['confirmes']==True):
        option_1 = 'Confirme'
        print(option_1)
    elif(option['confirmes']!=True):
        option_1='confirmesvide'
        # option_2='a'
        # option_3='b'
        # option_4 ='cc '
        # option_5='dd'
        # option_6='dd'
    if(option['contact']==True):
        option_2='Contact'
    elif(option['contact']==''):

        option_2='contactvide'

        # option_1='a'
        # option_3='b'
        # option_4 ='cc '
        # option_5='dd'
        # option_6='dd'
    if(option['importes']==True):
        option_3='Importe'
    elif(option['importes']==''):
        option_3='importesvide'
        # option_2='a'
        # option_1='b'
        # option_4 ='cc '
        # option_5='dd'
        # option_6='dd'
    if(option['communautaire']==True):
        option_4='Communautaire'
    elif(option['communautaire']==''):
        option_4='Communautairevide'
        # option_2='a'
        # option_3='b'
        # option_1 ='cc '
        # option_5='dd'
        # option_6='dd'
    if(option['recovered']==True):
        option_5='Recovered'
    elif(option['recovered']==''):
        option_5='Recoveredvide'
        # option_2='a'
        # option_3='b'
        # option_4 ='cc '
        # option_1='dd'
        # option_6='dd'
    if(option['dead']==True):
        option_6='Dead'
    elif(option['dead']==''):
        option_6='deadvide'
        # option_2='a'
        # option_3='b'
        # option_4 ='cc '
        # option_5='dd'
        # option_1='dd'
    print(option)
    data = graph.graphe_sn('Senegal',df,date_deb =date_deb , date_fin=date_fin ,option_1=option_1,option_2=option_2,option_3=option_3,option_4=option_4,option_5=option_5,option_6=option_6)
    print(data)
    response = {
        'data': data
    }
    return jsonify(response) , 200

    
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