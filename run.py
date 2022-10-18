import pandas as pd
import statsmodels.api as sm
import gradio as gr
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import statsmodels.api as sm

df = pd.read_csv('https://raw.githubusercontent.com/billster45/taxi_trip_nyc/main/hsb2.csv')
model = sm.OLS(df['math'], sm.add_constant(df['science'])).fit()

def build_model(intercept_val,slope_val):
    
    intercept = intercept_val
    slope=slope_val

    df['yhat'] = intercept+slope*df['science']
    df['sqrd_dist'] = np.square(df['math']-df['yhat'])

    # combine data point co-ords with slope-cords to be able to draw vertical distance line
    df['data_x_y'] = list(zip(df.science, df.math)) # data point co-ords
    df['slope_x_y'] = list(zip(df.science, df.yhat)) # slope co-ords for each value of x 
    data_x_y_list = df['data_x_y'].tolist() # data point co-ords
    slope_x_y_list = df['slope_x_y'].tolist() # slope co-ords for each value of x 
    vertical_line_coords = zip(data_x_y_list, slope_x_y_list)
    vertical_line_coords_list = list(vertical_line_coords)

    fig,ax=plt.subplots(figsize=(7,4))
    plt.scatter(x=df['science'],y=df['math'],color="#338844", edgecolor="white", s=50, lw=1,alpha=0.5)
    ax.axline(xy1=(0,intercept), slope=slope, color='C0') # https://matplotlib.org/3.5.1/gallery/pyplots/axline.html#sphx-glr-gallery-pyplots-axline-py
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 90) 
    ax.set_ylabel('Math scores')
    ax.set_xlabel('Science scores')
    ax.set_title('Does Science Score Predict Math Score?')

    # plotting distance lines
    if slope==0.6 and intercept==21.7:
        color='green'
        width=3
    else:
        color='grey'
        width=1

    lc = mc.LineCollection(vertical_line_coords_list, colors=color, linewidths=width, zorder=1)
    ax.add_collection(lc)

    text_kwargs = dict(ha='right', va='center', fontsize=14, color='C1')
    plt.text(75, 20, 'Your line: y = '+str(round(intercept,1))+' + '+str(round(slope,1))+' x Science Score', **text_kwargs)
    plt.text(75, 10, 'Best OLS fit line: y = '+str(round(model.params[0],1))+' + '+str(round(model.params[1],1))+' x Science Score', **text_kwargs)

    fig1, ax  = plt.subplots(figsize=(7,2))
    ax.barh([1], df['sqrd_dist'].sum(),tick_label=['SSR'], align='center',alpha=0.5)
    ax.set_xlim(0, 100000)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,}'))
    plt.text(80000,1,'Sum of the Squared Residuals', **text_kwargs)
    plt.axvline(x=model.ssr,color=color, ls='solid', lw=6) # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html

    # Model summary
    mod_out = model.summary()

    return [fig,fig1,mod_out]

input_intercept = gr.Slider(0, 80, label='Select y Intercept', value=30,step=0.1)
input_slope = gr.Slider(0, 3, label='Select Slope', value=1,step=0.1)

outputs = [gr.Plot(label='Fit your own line'),gr.Plot(show_label=False),gr.Text(label="Model")]
title = "Simple Linear regression"
description = "Select the slope and intercept that best fits the data"

# https://github.com/gradio-app/gradio/issues/2093#issuecomment-1248657665
io = gr.Interface(fn = build_model, 
             inputs = [input_intercept,input_slope], 
             outputs = outputs, 
             examples=[[21.7,0.6]],
             cache_examples=True,
             allow_flagging='never',
             title = title, 
             description = description, 
             live=True)

io.dependencies[0]["show_progress"] = False  # the hack
io.dependencies[1]["show_progress"] = False  # the hack

io.launch(share=True)