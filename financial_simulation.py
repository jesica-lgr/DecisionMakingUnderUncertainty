import pandas as pd
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from dateutil import relativedelta

# Constants
ASSETS = ['SPY', 'AGG', 'GLD']
ROLLING_WINDOW_SIZE = 30
RISK_FREE_RATE = 0.02
INITIAL_INVESTMENT = 1000  # Example initial investment amount
USER_AGE = 12  # Example user age
SPLIT_DATE = '2019-01-01'  # Train-test split date

# Portafolio in USD
initial_portfolio_USD = {'SPY': 1000, 'AGG': 1000, 'GLD': 1000}

# Load datasets
assets_df = pd.read_csv('assets_historical_data.csv', index_col='Date', parse_dates=True)
indicators_df = pd.read_csv('economic_indicators_data.csv', index_col='Date', parse_dates=True)

# Forward-fill the economic indicators data to align with the assets data
#indicators_df = indicators_df.asfreq('D').ffill().reindex(assets_df.index).ffill()

# Combine assets and indicators dataframes
combined_df = pd.concat([assets_df, indicators_df], axis=1)

# Normalize asset prices using MinMaxScaler
scaler = MinMaxScaler()
combined_df[ASSETS] = scaler.fit_transform(combined_df[ASSETS])

# Calculate daily returns for assets
for asset in ASSETS:
    combined_df[f'{asset}_returns'] = combined_df[asset].pct_change()

# Calculate rolling returns and standard deviation for the portfolio
combined_df['rolling_returns'] = combined_df[[f'{asset}_returns' for asset in ASSETS]].mean(axis=1).rolling(window=ROLLING_WINDOW_SIZE).mean()
combined_df['rolling_std_dev'] = combined_df[[f'{asset}_returns' for asset in ASSETS]].std(axis=1).rolling(window=ROLLING_WINDOW_SIZE).std()

# Drop NaN values after calculations
combined_df.dropna(inplace=True)

# Initializing portfolio with the number of shares in each asset using the initial USD money.
initial_portfolio = {
    'SPY': initial_portfolio_USD['SPY']/combined_df['SPY'][0],
    'AGG': initial_portfolio_USD['AGG']/combined_df['AGG'][0],
    'GLD': initial_portfolio_USD['GLD']/combined_df['GLD'][0],
}

# Split into trainig and testing
split_date = '2019-01-01'
train_df_200 = combined_df[combined_df.index < split_date]
test_df_200 = combined_df[combined_df.index >= split_date]

# Reset index to convert the 'Date' index into a column
train_df_reset = train_df_200.reset_index()
test_df_reset = test_df_200.reset_index()

# Save the training and testing datasets with the 'Date' column
train_df_reset.to_csv('training_dataset_V3_1goal.csv', index=False)
test_df_reset.to_csv('testing_dataset_V3_1goal.csv', index=False)


# Define actions and initialize Q-table
actions = [-0.1, -0.05, 0 , 0.05, 0.1 ] # Buy, hold, and sell
action_combinations = [
    (0, 0, 0),
    (-0.1, 0, 0.1),
    (-0.1, 0.1, 0),
    (0.1, 0, -0.1),
    (0.1, -0.1, 0),
    (-0.1, 0.05, 0.05),
    (0.1, -0.05, -0.05),
    (0.05, -0.1, 0.05),
    (-0.05, 0.1, -0.05),
    (0.05, 0.05, -0.1),
    (-0.05, -0.05, 0.1)
]
print('ACTION COMBINATIONS LIST: {}'.format(action_combinations))
num_actions = len(action_combinations)
print('NUM ACTIONS: {}'.format(num_actions))
num_states = len(train_df_200)

# Initialize the Q-table
Q_table = pd.DataFrame(np.zeros((num_states, len(action_combinations))), columns=[str(act) for act in action_combinations])

# Update the portfolio based on the action taken
def update_portfolio_distribution(current_market_state, current_portafolio_state, action):
    current_portafolio_state_copy = current_portafolio_state.copy()
    negative_assets = []
    positive_assets = []
    total_positive_percentage = 0
    for asset_index in range(len(action)):
        asset_action = action[asset_index]
        if asset_action < 0:
            negative_assets.append((asset_index, abs(asset_action)))
        if asset_action > 0:
            positive_assets.append((asset_index, asset_action))
            total_positive_percentage += asset_action

    money_to_add = 0

    for tuple in negative_assets:
        asset_index, asset_action = tuple
        shares_to_reduce = current_market_state[ASSETS[asset_index]]*asset_action
        shares_value = current_market_state[ASSETS[asset_index]]*shares_to_reduce
        current_portafolio_state_copy[ASSETS[asset_index]] -= shares_to_reduce
        money_to_add += shares_value

    for tuple in positive_assets:
        asset_index, asset_action = tuple
        relative_positive_percentage = asset_action/total_positive_percentage
        money_to_add_to_asset = money_to_add * relative_positive_percentage
        shares_to_add = money_to_add_to_asset/current_market_state[ASSETS[asset_index]]
        current_portafolio_state_copy[ASSETS[asset_index]] += shares_to_add

    print(f"Updated Portfolio: {current_portafolio_state_copy}")
    return current_portafolio_state_copy

def get_portfolio_value(current_portfolio, current_market_state):
    # Get the latest available prices for each asset
    current_portfolio_value = 0
    for asset_index, asset in enumerate(ASSETS):
        asset_price = current_market_state[asset]
        ## print("Asset: {}, Asset Price: {}".format(asset, asset_price))
        ## print("Current portfolio shares of {}: {}".format(asset, current_portfolio[asset]))
        ## print('Current portfolio value in USD of {} : {}'.format(asset,asset_price*current_portfolio[asset]))
        current_portfolio_value += asset_price*current_portfolio[asset]
    ## print('Current portfolio value:', current_portfolio_value)
    return current_portfolio_value


# Function to compute the reward
# Target contains the value and the date.
def compute_reward(current_market_state, target, current_portfolio_value, total_months):
    target_date, target_value = target
    ## print('Init compute reward. Target date:  {}, target value: {}'.format(type(target_date), target_value))

    # Sharpe Ratio component
    sharpe_ratio_reward = rolling_sharpe_ratio(current_market_state['rolling_returns'], current_market_state['rolling_std_dev'])
    ## print('Sharpe ratio reward: {}'.format(sharpe_ratio_reward))

    # Progress towards goals
    progress_reward = 0
    remaining_balance = current_portfolio_value - target_value
    ## print('Remaining balance USD: {}'.format(remaining_balance))
    current_date = datetime.strptime(str(current_market_state.name), '%Y-%m-%d %H:%M:%S')
    ## print('Current date:', current_date, type(current_date))
    delta_to_target_date = relativedelta.relativedelta(target_date, current_date)
    ## print('delta_to_target_date:', delta_to_target_date.years)
    remaining_months_to_goal = delta_to_target_date.years * 12 + delta_to_target_date.months
    ## print('Remaining months: {}'.format(remaining_months_to_goal))
    if remaining_balance <0:
        progress_reward = remaining_balance * 1/(remaining_months_to_goal+1)
    else:
        progress_reward = remaining_balance * 1/(total_months - remaining_months_to_goal +1)

    total_reward = progress_reward + sharpe_ratio_reward
    ## print('Combined reward: {}'.format(total_reward))
    print(f"Reward: {total_reward}, Portfolio Value: {current_portfolio_value}, Target: {target}")
    return total_reward


def rolling_sharpe_ratio(returns, std_dev):
    epsilon = 1e-8  # A small number to avoid division by zero
    excess_returns = returns - RISK_FREE_RATE

    if pd.isna(returns) or pd.isna(std_dev) or std_dev == 0:
        return 0  # Return 0 reward in case of invalid inputs

    return excess_returns / (std_dev + epsilon)

# Define epsilon-greedy strategy for action selection
def choose_action_using_epsilon_greedy(state_index, epsilon):
    if np.random.uniform(0, 1) > epsilon:
        action_index = np.argmax(Q_table.iloc[state_index])
    else:
        action_index = np.random.choice(num_actions)
    return action_index

def update_Q_table(state_index, action_index, reward, new_state_index, alpha, gamma):
    # When we are in the final state
    if new_state_index >= num_states:
        Q_table.iloc[state_index, action_index] = reward
        return
    
    best_future_q = np.max(Q_table.iloc[new_state_index])
    Q_table.iloc[state_index, action_index] = (1 - alpha) * Q_table.iloc[state_index, action_index] + alpha * (reward + gamma * best_future_q)


start_year = combined_df.index[0].year
def get_years_since_start(date):
    return date.year - start_year


total_rewards = []
final_rewards = []

# Main training loop
def train_q_learning():
    epsilon = 0.5  # Exploration rate
    alpha = 0.5  # Learning rate
    gamma = 0.1  # Discount factor
    epochs = 200  # Number of training epochs

    # Define target values for each goal
    target_values = (datetime.strptime(SPLIT_DATE, '%Y-%m-%d'), 200000)
    

    # final_rewards = []
    first_date = train_df_200.index[0]
    last_date = datetime.strptime(SPLIT_DATE, '%Y-%m-%d')
    delta_to_all_months = relativedelta.relativedelta(last_date, first_date)
    all_months = delta_to_all_months.years * 12 + delta_to_all_months.months
    # print('all_months:', all_months)

    total_rewards = []
    portfolio_values_per_epoch = []  # List to store portfolio values per epoch
    portfolio_distribution_per_epoch = []  # List to store portfolio distribution per epoch

    for epoch in range(epochs):
        state_index = 0  # Reset state index at the start of each epoch
        total_reward = 0
        current_portfolio = initial_portfolio.copy()
        print('Number of epoch:{}'.format(epoch))
        portfolio_values = []

        for _, current_market_state in train_df_200.iterrows():
            print('+++++++++++++++++++++++')
            print('+++++++++++++++++++++++')
        
            updated_portfolio = current_portfolio.copy()
            action_index = choose_action_using_epsilon_greedy(state_index, epsilon)
            action = action_combinations[action_index]
            ## print('action index:{}'.format(action_index))
            print('Current action to take: {}'.format(action))

            #print('Past Portfolio: {}'.format(current_portfolio))
            updated_portfolio = update_portfolio_distribution(current_market_state, current_portfolio, action)
            updated_portfolio_value = get_portfolio_value(updated_portfolio,current_market_state)
            portfolio_values.append(updated_portfolio_value)
            print('Updated portfolio value USD: {}'.format(updated_portfolio_value))
            # reward = compute_reward(current_market_state, target_values, current_portfolio_value)
            reward = compute_reward(current_market_state, target_values, updated_portfolio_value, all_months)

            print(f"Epoch: {epoch}, State: {state_index}, Action: {action}, Reward: {reward}, Portfolio Value: {updated_portfolio_value}")

            new_state_index = state_index + 1 if state_index + 1 < len(train_df_200) else state_index
            update_Q_table(state_index, action_index, reward, new_state_index, alpha, gamma)
            state_index = new_state_index  # Move to the next state
            print(f"Epoch: {epoch}, State: {state_index}, Action: {action}, Reward: {reward}")
            total_reward += reward
            portfolio_distribution_per_epoch.append(updated_portfolio.copy())
        
        final_rewards.append(reward)
        total_rewards.append(total_reward)
        print(f'Epoch {epoch} Total Reward: {total_reward}')
        portfolio_values_per_epoch.append(portfolio_values)

    # Assuming 3 assets, plot the distribution for each asset
    spy_allocations = [dist['SPY'] for dist in portfolio_distribution_per_epoch]
    agg_allocations = [dist['AGG'] for dist in portfolio_distribution_per_epoch]
    gld_allocations = [dist['GLD'] for dist in portfolio_distribution_per_epoch]
    
    # Plotting total rewards per epoch
    pd.Series(total_rewards).plot(title="Total Rewards per Epoch")
    plt.xlabel('Epoch'); plt.ylabel('Total Reward'); plt.show()
    plt.savefig('total_rewards_per_epoch_V3_0505200.png')
    #Q_table.to_csv('Q_table_V3_300.csv')
    pd.Series(final_rewards).plot(title="Final Rewards per Epoch")
    plt.xlabel('Epoch'); plt.ylabel('Final Reward'); plt.show()
    plt.savefig('final_rewards_per_epoch_V3_0505200.png')

    # Calculate average portfolio value per epoch and plot
    average_portfolio_values = [np.mean(values) for values in portfolio_values_per_epoch]
    plt.figure(figsize=(10, 5))
    plt.plot(average_portfolio_values, label='Average Portfolio Value per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Portfolio Value')
    plt.title('Average Portfolio Value per Epoch Over Training')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(spy_allocations, label='SPY Allocation')
    plt.plot(agg_allocations, label='AGG Allocation')
    plt.plot(gld_allocations, label='GLD Allocation')
    plt.xlabel('Epoch')
    plt.ylabel('Asset Allocation')
    plt.title('Asset Allocation in Portfolio Over Training')
    plt.legend()
    plt.show()


print('==============================================')
print('==============================================')
print('==============================================')
print('==============================================')
# Execute the training
train_q_learning()


# Compute average reward per epoch and plot
average_rewards = [total / len(train_df_200) for total in total_rewards]
plt.figure(figsize=(10, 5))
plt.plot(average_rewards, label='Average Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Reward')
plt.title('Average Reward per Epoch Over Training')
plt.legend()
plt.show()



# Convert the Q-table to a DataFrame
Q_table_df = pd.DataFrame(Q_table)
# Save the Q-table to a CSV file
Q_table.to_csv('Q_table_V3_0505200.csv')

# Plot the Q-table
plt.imshow(Q_table, cmap='hot', interpolation='nearest')
plt.show()
