import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
# Importing data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")  # Corrected date format

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOGL", "MSFT", "GME")  # Tuple
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 5)  # Storing the data whenever users interact
period = n_years * 365

# Loading data
@st.cache_data
def load_data(ticker):
    # Caching the data so we don't have to download/run code again
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # Gives the date in first column
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

# Plot data, Streamlit can handle pandas DataFrame
st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

#Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()  #m for model
m.fit(df_train) #df for data frame
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail()) #last ten rows

#continuing plotting plotting forecast
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)  #not plolty fig