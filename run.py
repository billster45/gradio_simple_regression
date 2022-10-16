import pandas as pd
import statsmodels.api as sm
import gradio as gr
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def get_forecast(add_mul, time):

        # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
    def distance(point,coef):
        return abs((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

    df = pd.read_csv('https://raw.githubusercontent.com/billster45/taxi_trip_nyc/main/hsb2.csv')

    intercept = 22
    slope =0.6

    df['distance'] = df.apply(lambda row: distance((row['science'], row['math']),(intercept,slope)), axis=1)
    df['sqrd_dist'] = np.square(df['distance'])
    df['yhat'] = intercept+slope*df['science']

    # combine data point co-ords with slope-cords to be able to draw vertical distance line
    df['data_x_y'] = list(zip(df.science, df.math)) # data point co-ords
    df['slope_x_y'] = list(zip(df.science, df.yhat)) # slope co-ords for each value of x 
    data_x_y_list = df['data_x_y'].tolist() # data point co-ords
    slope_x_y_list = df['slope_x_y'].tolist() # slope co-ords for each value of x 
    vertical_line_coords = zip(data_x_y_list, slope_x_y_list)
    vertical_line_coords_list = list(vertical_line_coords)

    # https://seaborn.pydata.org/tutorial/axis_grids.html#conditional-small-multiples
    fig = plt.figure()
    ax = fig.gca() #which is used to extract the axes
    ax.scatter(x=df['science'],y=df['math'],color="#338844", edgecolor="white", s=50, lw=1,alpha=0.5)
    ax.axline((0, intercept), slope=slope, color='C0', label='your slope')
    ax.set_xlim(20, 80)
    ax.set_ylim(25, 90) 
    ax.set_ylabel('Math scores')
    ax.set_xlabel('Science scores')
    ax.set_title('Does Science Score Predict Math Score?')
    ax.legend()

    # splotting grey lines
    lc = mc.LineCollection(vertical_line_coords_list, colors='grey', linewidths=1, zorder=1)
    ax.add_collection(lc);
    
    return plt

with gr.Blocks() as demo:
    gr.Markdown(
    """
    Prophet forecast
    """)
    with gr.Row():
        add_mul = gr.Dropdown(["additive", "multiplicative"], label="Seasonality", value="additive")
        time = gr.Slider(2, 48, step=4,value=4, label="Forecast length")

    plt = gr.Plot()

    add_mul.change(get_forecast, [add_mul, time], plt, queue=False)
    time.change(get_forecast, [add_mul, time], plt, queue=False)    
    demo.load(get_forecast, [add_mul, time], plt, queue=False)    

demo.launch(debug=True)