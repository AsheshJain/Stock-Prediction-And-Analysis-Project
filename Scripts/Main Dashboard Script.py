#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv('https://github.com/AsheshJain/Stock-Prediction-Project-Files/blob/cdf73e729bad36c1ed5059649a6ece7c1111d2d1/Data%20Sets/ASIANPAINT.csv?raw=true')

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("saved_lstm_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:]
valid['Predictions']=closing_price



df= pd.read_csv(r'C:\Users\pc\Downloads\archive\NIFTY50_all.csv')
df['Symbol']=df['Symbol'].replace(['MUNDRAPORT','UTIBANK','BAJAUTOFIN','BHARTI','HEROHONDA','HINDALC0','HINDLEVER','INFOSYSTCH','JSWSTL','KOTAKMAH','TELCO','TISCO','SESAGOA','SSLT','ZEETELE'],['ADANIPORTS','AXISBANK','BAJFINANCE','BHARTIARTL','HEROMOTOCO','HINDALCO','HINDUNILVR','INFY','JSWSTEEL','KOTAKBANK','TATAMOTORS','TATASTEEL','VEDL','VEDL','ZEEL'])


app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Asian Paints CLosing Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='NSE Stocks Data', style={'color': 'Red'}, children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Asian Paint','value': 'ASIANPAINT'},
                                       {'label': 'Axis Bank', 'value': 'AXISBANK'},
                                      {'label': 'Adani Ports', 'value': 'ADANIPORTS'},
                                      {'label': 'Bajaj Auto', 'value': 'BAJAJ-AUTO'}, 
                                      {'label': 'Bajaj Finserv','value': 'BAJAJFINSV'},
                                      {'label': 'Bajaj Finance','value': 'BAJFINANCE'},
                                       {'label': 'Bharti Airtel', 'value': 'BHARTIARTL'},
                                      {'label': 'BPCL', 'value': 'BPCL'},
                                      {'label': 'Britannia Inds', 'value': 'BRITANNIA'},
                                      {'label': 'Cipla', 'value': 'CIPLA'},
                                      {'label': 'Coal India', 'value': 'COALINDIA'},
                                      {'label': 'Dr. Reddys Lab', 'value': 'DRREDDY'},
                                      {'label': 'Eicher Motors', 'value': 'EICHERMOT'},
                                      {'label': 'GAIL India', 'value': 'GAIL'},
                                      {'label': 'Grasim Industries', 'value': 'GRASIM'},
                                      {'label': 'HCL Tech.', 'value': 'HCLTECH'},
                                      {'label': 'ICICI Bank', 'value': 'ICICIBANK'},
                                      {'label': 'Indusind Bank', 'value': 'INDUSINDBK'},
                                      {'label': 'Indus Tower', 'value': 'INFRATEL'},
                                      {'label': 'Infosys', 'value': 'INFY'},
                                      {'label': 'Indian Oil Corp.', 'value': 'IOC'},
                                      {'label': 'ITC', 'value': 'ITC'},
                                      {'label': 'JSW Steel', 'value': 'JSWSTEEL'},
                                      {'label': 'Kotak Mahindra Bank', 'value': 'KOTAKBANK'},
                                      {'label': 'Larsen & Toubro Limited', 'value': 'LT'},
                                      {'label': 'Mahindra & Mahindra', 'value': 'M&M'},
                                      {'label': 'Maruti Suzuki India', 'value': 'MARUTI'},
                                      {'label': 'Nestle', 'value': 'NESTLEIND'},
                                      {'label': 'NTPC', 'value': 'NTPC'},
                                      {'label': 'ONGC', 'value': 'ONGC'},
                                      {'label': 'Power Grid Corp.', 'value': 'POWERGRID'},
                                      {'label': 'Reliance Industries', 'value': 'RELIANCE'},
                                      {'label': 'SBI Bank', 'value': 'SBIN'},
                                      {'label': 'Shree Cement', 'value': 'SHREECEM'},
                                      {'label': 'Sun Pharma Inds.', 'value': 'SUNPHARMA'},
                                      {'label': 'Tata Motors', 'value': 'TATAMOTORS'},
                                      {'label': 'Tata Steel', 'value': 'TATASTEEL'},
                                      {'label': 'TCS', 'value': 'TCS'},
                                      {'label': 'Tech Mahindra', 'value': 'TECHM'},
                                      {'label': 'Titan', 'value': 'TITAN'},
                                      {'label': 'Ultratech Cement', 'value': 'ULTRACEMCO'},
                                      {'label': 'Uniphos Enterprises', 'value': 'UNIPHOS'},
                                      {'label': 'UPL', 'value': 'UPL'},
                                      {'label': 'Vedanta', 'value': 'VEDL'},
                                      {'label': 'Wipro', 'value': 'WIPRO'},
                                      {'label': 'Zee Entertainment', 'value': 'ZEEL'},
                                     ], 
                             multi=True,value=['ASIANPAINT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Asian Paint','value': 'ASIANPAINT'},
                                       {'label': 'Axis Bank', 'value': 'AXISBANK'},
                                      {'label': 'Adani Ports', 'value': 'ADANIPORTS'},
                                      {'label': 'Bajaj Auto', 'value': 'BAJAJ-AUTO'}, 
                                      {'label': 'Bajaj Finserv','value': 'BAJAJFINSV'},
                                      {'label': 'Bajaj Finance','value': 'BAJFINANCE'},
                                       {'label': 'Bharti Airtel', 'value': 'BHARTIARTL'},
                                      {'label': 'BPCL', 'value': 'BPCL'},
                                      {'label': 'Britannia Inds', 'value': 'BRITANNIA'},
                                      {'label': 'Cipla', 'value': 'CIPLA'},
                                      {'label': 'Coal India', 'value': 'COALINDIA'},
                                      {'label': 'Dr. Reddys Lab', 'value': 'DRREDDY'},
                                      {'label': 'Eicher Motors', 'value': 'EICHERMOT'},
                                      {'label': 'GAIL India', 'value': 'GAIL'},
                                      {'label': 'Grasim Industries', 'value': 'GRASIM'},
                                      {'label': 'HCL Tech.', 'value': 'HCLTECH'},
                                      {'label': 'ICICI Bank', 'value': 'ICICIBANK'},
                                      {'label': 'Indusind Bank', 'value': 'INDUSINDBK'},
                                      {'label': 'Indus Tower', 'value': 'INFRATEL'},
                                      {'label': 'Infosys', 'value': 'INFY'},
                                      {'label': 'Indian Oil Corp.', 'value': 'IOC'},
                                      {'label': 'ITC', 'value': 'ITC'},
                                      {'label': 'JSW Steel', 'value': 'JSWSTEEL'},
                                      {'label': 'Kotak Mahindra Bank', 'value': 'KOTAKBANK'},
                                      {'label': 'Larsen & Toubro Limited', 'value': 'LT'},
                                      {'label': 'Mahindra & Mahindra', 'value': 'M&M'},
                                      {'label': 'Maruti Suzuki India', 'value': 'MARUTI'},
                                      {'label': 'Nestle', 'value': 'NESTLEIND'},
                                      {'label': 'NTPC', 'value': 'NTPC'},
                                      {'label': 'ONGC', 'value': 'ONGC'},
                                      {'label': 'Power Grid Corp.', 'value': 'POWERGRID'},
                                      {'label': 'Reliance Industries', 'value': 'RELIANCE'},
                                      {'label': 'SBI Bank', 'value': 'SBIN'},
                                      {'label': 'Shree Cement', 'value': 'SHREECEM'},
                                      {'label': 'Sun Pharma Inds.', 'value': 'SUNPHARMA'},
                                      {'label': 'Tata Motors', 'value': 'TATAMOTORS'},
                                      {'label': 'Tata Steel', 'value': 'TATASTEEL'},
                                      {'label': 'TCS', 'value': 'TCS'},
                                      {'label': 'Tech Mahindra', 'value': 'TECHM'},
                                      {'label': 'Titan', 'value': 'TITAN'},
                                      {'label': 'Ultratech Cement', 'value': 'ULTRACEMCO'},
                                      {'label': 'Uniphos Enterprises', 'value': 'UNIPHOS'},
                                      {'label': 'UPL', 'value': 'UPL'},
                                      {'label': 'Vedanta', 'value': 'VEDL'},
                                      {'label': 'Wipro', 'value': 'WIPRO'},
                                      {'label': 'Zee Entertainment', 'value': 'ZEEL'},
                                     ], 
                             multi=True,value=['ASIANPAINT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"ASIANPAINT": "Asian Paint","BAJAJ-AUTO": "Bajaj Auto","BAJAJFINSV": "Bajaj Finserv",
                "AXISBANK": "Axis Bank","ADANIPORTS": "Adani Ports",
                "BAJFINANCE": "Bajaj Finance","BHARTIARTL": "Bharti Airtel","BPCL": "BPCL",
                "BRITANNIA": "Britannia Inds","CIPLA": "Cipla",
                "COALINDIA": "Coal India","DRREDDY": "Dr. Reddys Lab","EICHERMOT": "Eicher Motors",
                "GAIL": "GAIL India","GRASIM": "Grasim Industries","HCLTECH": "HCL Tech.","ICICIBANK": "ICICI Bank",
                "INDUSINDBK": "Indusind Bank","INFRATEL": "Indus Tower","INFY": "Infosys","IOC": "Indian Oil Corp.",
                "ITC": "ITC","JSWSTEEL": "JSW Steel","KOTAKBANK": "Kotak Mahindra Bank","LT": "Larsen & Toubro Limited",
                "M&M": "Mahindra & Mahindra","MARUTI": "Maruti Suzuki India","NESTLEIND": "Nestle","NTPC": "NTPC",
                "ONGC": "ONGC","POWERGRID": "Power Grid Corp.","RELIANCE": "Reliance Industries","SBIN": "SBI Bank",
                "SHREECEM": "Shree Cement","SUNPHARMA": "Sun Pharma Inds.","TATAMOTORS": "Tata Motors","TATASTEEL": "Tata Steel",
                "TCS": "TCS","TECHM": "Tech Mahindra","TITAN": "Titan","ULTRACEMCO": "Ultratech Cement","UNIPHOS": "Uniphos Enterprises",
                "UPL": "UPL","VEDL": "Vedanta","WIPRO": "Wipro","ZEEL": "Zee Entertainment",
               }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                        y=df[df["Symbol"] == stock]["High"],
                        mode='lines', opacity=0.7, 
                        name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
            go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                        y=df[df["Symbol"] == stock]["Low"],
                        mode='lines', opacity=0.6,
                        name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"ASIANPAINT": "Asian Paint","BAJAJ-AUTO": "Bajaj Auto","BAJAJFINSV": "Bajaj Finserv",
                "AXISBANK": "Axis Bank","ADANIPORTS": "Adani Ports",
                "BAJFINANCE": "Bajaj Finance","BHARTIARTL": "Bharti Airtel","BPCL": "BPCL",
                "BRITANNIA": "Britannia Inds","CIPLA": "Cipla",
                "COALINDIA": "Coal India","DRREDDY": "Dr. Reddys Lab","EICHERMOT": "Eicher Motors",
                "GAIL": "GAIL India","GRASIM": "Grasim Industries","HCLTECH": "HCL Tech.","ICICIBANK": "ICICI Bank",
                "INDUSINDBK": "Indusind Bank","INFRATEL": "Indus Tower","INFY": "Infosys","IOC": "Indian Oil Corp.",
                "ITC": "ITC","JSWSTEEL": "JSW Steel","KOTAKBANK": "Kotak Mahindra Bank","LT": "Larsen & Toubro Limited",
                "M&M": "Mahindra & Mahindra","MARUTI": "Maruti Suzuki India","NESTLEIND": "Nestle","NTPC": "NTPC",
                "ONGC": "ONGC","POWERGRID": "Power Grid Corp.","RELIANCE": "Reliance Industries","SBIN": "SBI Bank",
                "SHREECEM": "Shree Cement","SUNPHARMA": "Sun Pharma Inds.","TATAMOTORS": "Tata Motors","TATASTEEL": "Tata Steel",
                "TCS": "TCS","TECHM": "Tech Mahindra","TITAN": "Titan","ULTRACEMCO": "Ultratech Cement","UNIPHOS": "Uniphos Enterprises",
                "UPL": "UPL","VEDL": "Vedanta","WIPRO": "Wipro","ZEEL": "Zee Entertainment",
               }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x=df[df["Symbol"] == stock]["Date"],
                        y=df[df["Symbol"] == stock]["Volume"],
                        mode='lines', opacity=0.7,
                        name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=False)


# In[ ]:




