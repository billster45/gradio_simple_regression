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

def build_model(slope_val,intercept_val):
    
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
    ax.axline(xy1=(0,intercept), slope=slope, color='C0', label='your slope') # https://matplotlib.org/3.5.1/gallery/pyplots/axline.html#sphx-glr-gallery-pyplots-axline-py
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 90) 
    ax.set_ylabel('Math scores')
    ax.set_xlabel('Science scores')
    ax.set_title('Does Science Score Predict Math Score?')
    ax.legend()

    # plotting distance lines
    if slope>=0.59 and slope<=0.61 and intercept>=21.0 and intercept<=22.0:
        color='green'
        width=2
    else:
        color='grey'
        width=1

    lc = mc.LineCollection(vertical_line_coords_list, colors=color, linewidths=width, zorder=1)
    ax.add_collection(lc)

    # rsquared calculated
    df['ybar'] = np.mean(df['math'])
    rss = np.sum(np.square(df['yhat']-df['ybar']))
    tss = np.sum(np.square(df['math']-df['ybar']))
    rsquared = round((rss/tss)*100,1)
    text_kwargs = dict(ha='center', va='center', fontsize=14, color='C1')
    plt.text(20, 80, 'Your R-squared is '+str(rsquared)+'%', **text_kwargs)
    plt.text(40, 10, 'y ='+str(round(model.params[0],1))+' + '+str(round(model.params[1],1))+' x Science Score', **text_kwargs)

    fig1, ax  = plt.subplots(figsize=(7,1))
    ax.barh([1], df['sqrd_dist'].sum(),
        tick_label=['SSR'], align='center')
    ax.set_xlim(0, 100000)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,}'))
    ax.set_title('Sum of the Squared Residuals')
    plt.axvline(x=model.ssr,color=color, ls='solid', lw=6) # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html

    # Model summary
    mod_out = model.summary()

    return [fig,fig1,mod_out]

input_slope = gr.Slider(0, 3, label='Select Slope', value=1,step=0.1)
input_intercept = gr.Slider(0, 80, label='Select y Intercept', value=30,step=1)

outputs = [gr.Plot(label='Fit your own line'),gr.Plot(show_label=False),gr.Text(label="Model")]
title = "Simple Linear regression"
description = "Select the slope that best fits the data"

gr.Interface(fn = build_model, 
             inputs = [input_slope,input_intercept], 
             outputs = outputs, 
             examples=[[0.6,21]],
             cache_examples=True,
             allow_flagging='never',
             title = title, 
             description = description, 
             live=False).launch(debug=False)