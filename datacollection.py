import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# DEFINE PARAMETERS

# Set the time period to pull data for
start_date = dt.datetime(2005, 1, 1)  # Capture the pre-2008 and post-2008 periods
end_date = dt.datetime.now()  # Up to the current date

# Define the tickers for the assets of interest
# SPY for S&P 500 ETF, AGG for bonds, and a commodity ETF like GLD for gold
assets_tickers = ['SPY', 'AGG', 'GLD']

# COLLECT ASSET DATA

# Initialize a dictionary to store dataframes
historical_data = {}

# Loop over the tickers and pull data
for ticker in assets_tickers:
    asset_data = yf.download(ticker, start=start_date, end=end_date)
    historical_data[ticker] = asset_data['Adj Close']  # Adjusted Close accounts for dividends and splits

# Convert the dictionary to a DataFrame
assets_df = pd.DataFrame(historical_data)

# GET ECONOMIC INDICATORS DATA

# Federal funds rate
fed_funds_rate = pdr.get_data_fred('DFF', start_date, end_date)
# Consumer Price Index (to measure inflation)
cpi_data = pdr.get_data_fred('CPIAUCSL', start_date, end_date)
# GDP growth rate - Quarterly data
gdp_growth = pdr.get_data_fred('A191RL1Q225SBEA', start_date, end_date)

# DATA PREPROCESSING

# Resample the CPI and GDP data to monthly frequency if necessary, using forward-fill to propagate the last valid observation
cpi_data_monthly = cpi_data.resample('M').ffill()
gdp_growth_monthly = gdp_growth.resample('M').ffill()

# Combine all economic data into a single DataFrame
economic_indicators_df = pd.DataFrame({
    'FedFunds': fed_funds_rate['DFF'],
    'CPI': cpi_data_monthly['CPIAUCSL'],
    'GDPGrowth': gdp_growth_monthly['A191RL1Q225SBEA']
})

# VISUALIZING DATA

# Plotting the Adjusted Close price for the SPY ETF
assets_df['SPY'].plot(title='SPY ETF Adjusted Close Price', figsize=(10, 6))
plt.show()

# # Plotting the Federal Funds Rate
# economic_indicators_df['FedFunds'].plot(title='Federal Funds Rate', figsize=(10, 6))
# plt.show()

# # Plotting the CPI
# economic_indicators_df['CPI'].plot(title='Consumer Price Index (CPI)', figsize=(10, 6))
# plt.show()

# # Plotting the GDP Growth
# economic_indicators_df['GDPGrowth'].plot(title='GDP Growth Rate', figsize=(10, 6))
# plt.show()

# Plotting the economic indicators
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plotting the Federal Funds Rate
axs[0].plot(fed_funds_rate.index, fed_funds_rate['DFF'])
axs[0].set_title('Federal Funds Rate')

# Plotting the CPI
axs[1].plot(cpi_data_monthly.index, cpi_data_monthly['CPIAUCSL'])
axs[1].set_title('Consumer Price Index (CPI)')

# Plotting the GDP Growth
axs[2].plot(gdp_growth_monthly.index, gdp_growth_monthly['A191RL1Q225SBEA'])
axs[2].set_title('GDP Growth Rate')

# Display the plot
plt.tight_layout()
plt.show()


# SAVING DATA
assets_df.to_csv('assets_historical_data.csv')
economic_indicators_df.to_csv('economic_indicators_data.csv')


