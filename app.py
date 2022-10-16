import gradio as gr
import pypistats
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
from prophet import Prophet
pd.options.plotting.backend = "plotly"

def get_forecast(add_mul, time):

    data = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv')

    m = Prophet(seasonality_mode=add_mul)
    m.fit(data)
    future = m.make_future_dataframe(periods=time)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    return fig1 

with gr.Blocks() as demo:
    gr.Markdown(
    """
    Prophet forecast
    """)
    with gr.Row():
        add_mul = gr.Dropdown(["additive", "multiplicative"], label="Seasonality", value="additive")
        time = gr.Dropdown([3, 6, 9, 12], label="Forecast length...", value=12)

    plt = gr.Plot()

    add_mul.change(get_forecast, [add_mul, time], plt, queue=False)
    time.change(get_forecast, [add_mul, time], plt, queue=False)    
    demo.load(get_forecast, [add_mul, time], plt, queue=False)    

demo.launch()