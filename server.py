
from flask import Flask ,jsonify,request, send_from_directory,render_template, Response, send_file, make_response
from app import Covid19
import sys
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


import json
import base64

app = Flask(__name__)
CORS(app)
dataframe = Covid19()


@app.route('/graph', methods=['GET'])
def get_dashboard():
    bar=dataframe.covid('Covid19SN_datas.xlsx')
    response = {
        'data': bar
    }
    return jsonify(response)  ,200
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