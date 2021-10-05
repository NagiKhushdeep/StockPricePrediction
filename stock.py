#importing the required libraries
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#defining start date and current date
start_date = "2015-01-01"
current_date = date.today().strftime("%Y-%m-%d")

#setting the title of the web application
st.title('Stock Price Forecasting')

#defining the stock tickers whose price can be predicted
stocks = ('TATACONSUM.NS','TCS.NS','GRASIM.NS','ONGC.NS','NESTLEIND.NS','COALINDIA.NS','RELIANCE.NS','CIPLA.NS','TATASTEEL.NS','BRITANNIA.NS','MARUTI.NS','TITAN.NS','ITC.NS','BHARTIARTL.NS','NTPC.NS','ULTRACEMCO.NS','INDUSINDBK.NS','WIPRO.NS','LT.NS','TECHM.NS','ICICIBANK.NS','KOTAKBANK.NS','HINDALCO.NS','SHREECEM.NS','HDFCLIFE.NS','BAJAJ-AUTO.NS','HEROMOTOCO.NS','BAJFINANCE.NS','MM.NS','BAJAJFINSV.NS')

#creating a select box to display the above list of stock tickers
selected_stock = st.selectbox('Select dataset for prediction', stocks)

#adding a slider to chose the number of years of prediction
no_years = st.slider('Years of prediction:', 1, 5)
period = no_years * 365

#function to load data of the selected stock from yahoo finance using the yfinance library
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, current_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Please wait...Loading data!') 
data = load_data(selected_stock)
data_load_state.text('Data loaded successfully!')

st.subheader('Raw data')
st.write(data.tail())

#plotting raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

#predict stock prices using the Prophet function of fbprohet library
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

#displaying the forecaste data and plot
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {no_years} years')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)
