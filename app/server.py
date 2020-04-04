
from flask import Flask ,jsonify,request, send_from_directory,render_template, Response, send_file, make_response
from app import Covid19
import sys
from flask_cors import CORS
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import datetime
import StringIO
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter


import json
import base64

app = Flask(__name__)
CORS(app)
dataframe = Covid19()


@app.route('/api/dashboard/', methods=['GET'])
def get_dashboard():
    fig=dataframe.covid('E:\Covid19\covid-west-africa-dit\Covid19SN_datas.xlsx')
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response
    
    #return render_template("interface.html",  figure = fig)

     




if __name__ == '__main__':
    app.run(host = '0.0.0.0' ,port = 5000 )