import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Load model and set up Streamlit header
model = load_model("Stock Predictions Model.keras")
#st.set_page_config(page_title='Stock Market Predictor', layout='wide')
st.title(":chart_with_upwards_trend: Stock Trend Analysis")
st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>', unsafe_allow_html=True)


# Function to fetch stock data
def get_stock_data(stock, start, end):
    return yf.download(stock, start, end)


# Sidebar for user input
st.sidebar.header(':roller_coaster: Stock Market Predictor')

# Example list of stock symbols (expanded with international and Indian stocks)
common_stock_symbols = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'INTC', 'AMD', 'CSCO', 'ADBE', 'PYPL',
    'PUM', 'BAC', 'ADS', 'V', 'JPM', 'WMT', 'MA', 'CRM', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'BA', 'XOM',
    'TM', 'HMC', 'NVS', 'TSM', 'RIO', 'BP', 'UL', 'IBM', 'ORCL', 'JD', 'C', 'WFC', 'VZ', 'T', 'GE', 'LMT',
    'INFY', 'TATASTEEL.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 'SBIN.NS','TCS.NS', 'COALINDIA.NS', 'TITAN.NS'
]

# Dropdown menu for selecting a stock symbol
stock = st.sidebar.selectbox('Select Stock Symbol', common_stock_symbols, index=0)

start = st.sidebar.date_input('Start Date', pd.to_datetime('2013-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('2023-12-05'))
st.sidebar.markdown("> It’s far better to buy a wonderful company at a fair price than a fair company at a wonderful price. - Warren Buffett")

# Function to fetch live stock data and create a small chart for last month's price
def get_live_stock_data(stock_symbols):
    data = {}
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        
        # Fetch last month's data
        stock_data = stock.history(period='100mo')['Close']
        
        # Get current price
        current_price = stock.history(period='1d')['Close'][0]
        
        data[symbol] = {
            'Stock Prices': stock_data,
            'Current Price': current_price
        }
    return data

st.header(":globe_with_meridians: Live Stock Prices")

with st.expander("Fetch Live Data"):
    # Get user input for stock symbols
    stocks_input = st.text_input("Enter stock symbols separated by commas (e.g., AAPL,GOOGL,MSFT)", "AAPL,GOOGL,MSFT")
    stock_symbols = [symbol.strip().upper() for symbol in stocks_input.split(',')]

    # Toggle button to get live data
    toggle_live_data = st.button("Get Live Data")

    # Boolean variable to track if data is fetched or not
    data_fetched = False

    if toggle_live_data:
        if not data_fetched:
            live_stock_data = get_live_stock_data(stock_symbols)

            # Display live data and last month chart for each stock
            for symbol, stock_data in live_stock_data.items():
                col1, col2 = st.columns([1, 1])

                # Display current price in the first column
                with col1:
                    st.write(f"**{symbol}**")
                    st.write("Current Price:")
                    st.write(stock_data['Current Price'])

                    # Calculate profit/loss in percentage based on yesterday's price
                    yesterday_close = stock_data['Stock Prices'][-2]  # Get yesterday's closing price
                    profit_loss_percent = ((stock_data['Current Price'] - yesterday_close) / yesterday_close) * 100
                    st.write("Profit/Loss :")
                    st.write(f"  {profit_loss_percent:.2f}%")

                # Display chart in the second column
                with col2:
                    chart_data = go.Scatter(x=stock_data['Stock Prices'].index, y=stock_data['Stock Prices'], mode='lines', name='Stock Prices')
                    layout = {
                        'title': 'Stock Prices',
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Price'},
                        'width': 250,  # Adjust the width of the chart
                        'height': 263  # Increased height by 5%
                    }
                    st.plotly_chart({'data': [chart_data], 'layout': layout})

                st.write("-----------")

            # Set data_fetched to True after fetching data
            data_fetched = True
        else:
            pass  # Do nothing if data has already been fetched


# Fetch stock data
data = get_stock_data(stock, start, end)
# Additional stock information
#st.subheader('Additional Stock Information')


# Calculations for additional information
max_high = data['High'].max()
min_low = data['Low'].min()
total_volume = data['Volume'].sum()


# Displaying stock data and additional information side by side
col1, col2 = st.columns([2, 1])


with col1:
    st.header('Stock Data')
    st.write(data)


with col2:
    st.header(' ')
    st.markdown("---")
    st.write(f"**Highest Price:** {max_high}")
    st.write(f"**Lowest Price:** {min_low}")
    st.write(f"**Total Volume:** {total_volume}")
    st.markdown("---", unsafe_allow_html=True)

# Calculate growth from start date to end date
start_price = data['Close'].iloc[0]
end_price = data['Close'].iloc[-1]
growth = ((end_price - start_price) / start_price) * 100


# Display growth percentage
st.subheader('Company Growth from Start Date to End Date')
st.write(f"The company's stock price grew by {growth:.2f}% from {start} to {end}.")


# Create Plotly line chart for stock price
price_data = data.reset_index()
fig = px.line(price_data, x='Date', y='Close', title='Stock Price Growth', labels={'Close': 'Stock Price'})
fig.update_traces(mode='lines+markers')
fig.update_layout(
    hovermode='x',  # Show values on hover
    xaxis=dict(title='Date'),
    yaxis=dict(title='Stock Price'),
)
st.plotly_chart(fig)


# Data preprocessing
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])


# Initialize session state
session_state = st.session_state
if 'show_ma_plot' not in session_state:
    session_state.show_ma_plot = False


# Plotting
if st.button('Moving Averages :arrow_down_small:'):
    session_state.show_ma_plot = not session_state.show_ma_plot


if session_state.show_ma_plot:
    st.subheader('Price vs Moving Averages')
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    # Create Plotly line chart for Moving Averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Original Price'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA 50 days'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA 100 days'))
    fig_ma.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA 200 days'))

    fig_ma.update_layout(
        title='Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    st.plotly_chart(fig_ma)
    plt.clf()  # Clear the figure after displaying


# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))


past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


x = []
y = []


for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])


x, y = np.array(x), np.array(y)


predict = model.predict(x)


scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale


# Display predicted and actual prices in a horizontal table
st.subheader('Predicted vs Actual Prices')
data_display = pd.DataFrame({
    'Predicted Price': predict.flatten(),
    'Original Price': y.flatten()
})
st.dataframe(data_display.T)  # Display the dataframe transposed for a horizontal view



# Plot predicted vs actual prices using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(predict)), y=predict.flatten(), mode='lines+markers', name='Predicted Price'))
fig.add_trace(go.Scatter(x=np.arange(len(y)), y=y.flatten(), mode='lines+markers', name='Original Price'))
fig.update_layout(title='Original Price vs Predicted Price', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig)

# Get the last 100 days from the test data and scale it for prediction
last_100_days = data_test.tail(100)
data_test_scaled = scaler.fit_transform(last_100_days)

# Reshape the data for prediction
x = []
x.append(data_test_scaled[-100:])
x = np.array(x)

# Predict the next 10 days' prices
predicted_prices = []
for _ in range(10):
    prediction = model.predict(x)
    predicted_prices.append(prediction[0])

    # Update x to include the new prediction and drop the oldest day
    new_data = np.array([[[prediction[0][0]]]])
    x = np.concatenate((x[:, 1:, :], new_data), axis=1)

# Scale back the predicted prices to the original range
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Generate dates for the next 10 days
last_date = data.index[-1]
next_10_days = pd.date_range(start=last_date + pd.DateOffset(1), periods=10)

# Display the predicted prices for the next 10 days
st.subheader('Predicted Prices for Next 10 Days :heavy_dollar_sign:')
predicted_prices_df = pd.DataFrame({
    'Date': next_10_days,
    'Predicted Price': predicted_prices.flatten()
})
st.write(predicted_prices_df)

# Assuming you have 'predicted_prices' and 'next_10_days' from your code snippet

# Create a Plotly line chart for predicted prices over the next 10 days
fig = go.Figure()

fig.add_trace(go.Scatter(x=next_10_days, y=predicted_prices.flatten(), mode='lines+markers', name='Predicted Prices'))

fig.update_layout(
    title='Predicted Prices for Next 10 Days',
    xaxis_title='Date',
    yaxis_title='Price'
)

st.plotly_chart(fig)

import re

def save_contact_info(name, email, message):
    with open('contact_info.txt', 'a') as file:
        file.write(f"Name: {name}\n")
        file.write(f"Email: {email}\n")
        file.write(f"Message: {message}\n")
        file.write("\n")  # Add a separator between entries

def validate_email(email):
    # Using a simple regex pattern for email validation
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email)


# Define columns for About and Contact Me sections
col1, col2 = st.columns([1, 1])

# About Section
with col1:
  st.header(":memo: Contact Me")
  name = st.text_input("Your Name", key='name')
  email = st.text_input("Your Email", key='email')
  message = st.text_area("Message", key='message')

  if st.button("Submit"):
    if name.strip() == '' or not validate_email(email) or message.strip() == '':
      st.warning("Please provide a valid email and fill in all required fields.")
    else:
      save_contact_info(name, email, message)
      st.success("Message sent successfully!")

# Contact Me Section
with col2:
  st.header(":page_with_curl: About")
  st.write("This is the Stock Trend Analysis app.")
  st.write("Here, you can analyze stock historical data, fetch live data, and predict future prices.")
