from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go

app = Flask(__name__)

# Load the saved models for each attribute
models = {
    'Open': load_model("stock_prediction_model_Open.h5"),
    'Close': load_model("stock_prediction_model_Close.h5"),
    'High': load_model("stock_prediction_model_High.h5"),
    'Low': load_model("stock_prediction_model_Low.h5")
}

# Initialize global variables to store data and graphs
company_data = {}
graphs = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        company = request.form['company']
        return redirect(url_for('home', company=company))
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    company = request.args.get('company')
    if request.method == 'POST':
        return redirect(url_for('home1', company=company))
    return render_template('home.html', company=company)

@app.route('/home1', methods=['GET', 'POST'])
def home1():
    global graphs

    if request.method == 'POST':
        company = request.form['company']
        graph_type = request.form['graph_type']  # Get the selected graph type
        y_axis_attribute = request.form['y_axis']  # Get the selected y-axis attribute
    else:
        company = None
        graph_type = None
        y_axis_attribute = None

    predicted_value = None

    if company:
        # Check if data for the selected company is already loaded
        if company not in company_data:
            # Load data for the given company
            start = dt.datetime(2020, 1, 1)
            end = dt.datetime.today()

            try:
                data = yf.download(company, start=start, end=end)
            except Exception as e:
                error_message = f"Failed to download data for {company}: {e}"
                return render_template('home1.html', error=error_message)

            company_data[company] = data

        data = company_data[company]

        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[y_axis_attribute].values.reshape(-1, 1))

        prediction_days = 60

        # Prepare model inputs
        total_dataset = pd.concat((data[y_axis_attribute], data[y_axis_attribute]), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        x_test = []
        
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Make predictions using the corresponding model
        model = models[y_axis_attribute]
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Get the predicted value for the last day
        predicted_value = predicted_prices[-1][0]

        # Render the graph
        if company not in graphs:
            graphs[company] = {}

        if graph_type not in graphs[company]:
            # Create the graph if it doesn't exist
            if graph_type == 'candlestick':
                graph = create_candlestick_graph(data, company, y_axis_attribute)
            elif graph_type == 'line':
                graph = create_line_graph(data, company, y_axis_attribute)
            elif graph_type == 'bar':
                graph = create_bar_graph(data, company, y_axis_attribute)
            elif graph_type == 'line_markers':
                graph = create_line_markers_graph(data, company, y_axis_attribute)
            elif graph_type == 'step_line':
                graph = create_step_line_graph(data, company, y_axis_attribute)
            elif graph_type == 'hollow_candle':
                graph = create_hollow_candle_graph(data, company, y_axis_attribute)
            elif graph_type == 'volume_candle':
                graph = create_volume_candle_graph(data, company, y_axis_attribute)
            elif graph_type == 'area':
                graph = create_area_graph(data, company, y_axis_attribute)
            elif graph_type == 'hlc_area':
                graph = create_hlc_area_graph(data, company, y_axis_attribute)
            elif graph_type == 'baseline':
                graph = create_baseline_graph(data, company, y_axis_attribute)

            graphs[company][graph_type] = graph
        else:
            graph = graphs[company][graph_type]

        return render_template('home1.html', company=company, graph=graph.to_html(full_html=False), graph_type=graph_type, predicted_value=predicted_value, y_axis_attribute=y_axis_attribute)

    # If the request method is GET or if no company is provided, render the form template
    return render_template('home1.html') 

def create_candlestick_graph(data, company, y_axis_attribute):
    # Create candlestick trace
    trace_candlestick = go.Candlestick(x=data.index,
                                       open=data['Open'],
                                       high=data['High'],
                                       low=data['Low'],
                                       close=data[y_axis_attribute],
                                       increasing_line_color='green',
                                       decreasing_line_color='red',
                                       name=f'Actual {company} Prices')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='white',  # Set background color to white
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_candlestick], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    return graph

def create_line_graph(data, company, y_axis_attribute):
    # Create line trace
    trace_line = go.Scatter(x=data.index,
                            y=data[y_axis_attribute],
                            mode='lines',
                            name=f'Actual {company} Prices')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='white',  # Set background color to white
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_line], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    return graph

def create_line_markers_graph(data, company, y_axis_attribute):
    # Create line with markers trace
    trace_line_markers = go.Scatter(x=data.index,
                                    y=data[y_axis_attribute],
                                    mode='lines+markers',
                                    name=f'{company} Prices')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_line_markers], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_bar_graph(data, company, y_axis_attribute):
    # Create bar trace
    trace_bar = go.Bar(x=data.index,
                       y=data[y_axis_attribute],
                       name=f'{company} Prices')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_bar], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_step_line_graph(data, company, y_axis_attribute):
    # Create step line trace
    trace_step_line = go.Scatter(x=data.index,
                                 y=data[y_axis_attribute],
                                 mode='lines+markers',
                                 name=f'{company} Prices',
                                 line_shape='hv')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_step_line], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_hollow_candle_graph(data, company, y_axis_attribute):
    # Create hollow candle trace
    trace_hollow_candle = go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data[y_axis_attribute],
                                         increasing_fillcolor='white',
                                         decreasing_fillcolor='black',
                                         increasing_line_color='green',
                                         decreasing_line_color='red',
                                         name=f'{company} Prices')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_hollow_candle], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_volume_candle_graph(data, company, y_axis_attribute):
    # Create volume candle trace
    trace_volume_candle = go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         increasing_fillcolor='green',
                                         decreasing_fillcolor='red',
                                         increasing_line_color='green',
                                         decreasing_line_color='red',
                                         name=f'{company} Prices',
                                         yaxis='y2')

    # Create volume trace
    trace_volume = go.Bar(x=data.index,
                          y=data['Volume'],
                          marker_color='grey',
                          name='Volume',
                          yaxis='y')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       yaxis2=dict(title='Volume', overlaying='y', side='right'),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_volume_candle, trace_volume], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_area_graph(data, company, y_axis_attribute):
    # Create area trace
    trace_area = go.Scatter(x=data.index,
                            y=data[y_axis_attribute],
                            fill='tozeroy',
                            mode='none',
                            name=f'{company} Prices',
                            line=dict(color='blue'))

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_area], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_hlc_area_graph(data, company, y_axis_attribute):
    # Create HLC area trace
    trace_hlc_area = go.Scatter(x=data.index,
                                y=(data['High'] + data['Low'] + data['Close']) / 3,
                                fill='tozeroy',
                                mode='none',
                                name=f'{company} Prices',
                                line=dict(color='blue'))

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_hlc_area], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

def create_baseline_graph(data, company, y_axis_attribute):
    # Create baseline trace
    trace_baseline = go.Scatter(x=data.index,
                                y=np.zeros(len(data)),
                                mode='lines',
                                name='Baseline')

    # Create layout
    layout = go.Layout(title=f'{company} Share Price',
                       xaxis=dict(title='Time'),
                       yaxis=dict(title=y_axis_attribute),
                       plot_bgcolor='black',  # Set background color to black
                       showlegend=True)  # Show legend for both traces

    # Combine plots
    graph = go.Figure(data=[trace_baseline], layout=layout)

    # Increase the size of the graph
    graph.update_layout(height=800, width=1200)

    # Implement zoom functionality
    graph.update_xaxes(rangeslider_visible=True)

    return graph

if __name__ == "__main__":
    app.run(debug=True)
