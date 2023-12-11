# DecisionMakingUnderUncertainty

# Goal-Based Investment Portfolio Optimization Using Reinforcement Learning

## Project Overview

In this project, I develop a Goal-Based Investment Portfolio Optimization Tool utilizing Reinforcement Learning (RL) techniques for the final project of the AA228: Dcesision Making Under Uncertainty course at Stanford University (Fall 2023). The tool is designed to allocate assets in a way that maximizes returns under uncertain financial market conditions. By integrating personal financial goals into the optimization process, this tool also serves as an educational resource for youth to learn and engage in financial planning.

## Key Features

- **Reinforcement Learning Implementation**: Utilizes Q-learning and an epsilon-greedy strategy for decision-making under uncertainty.
- **Portfolio Optimization**: Aims to achieve the highest possible returns while considering the risk and time horizon of the investment.
- **Educational Tool (future) **: Offers insights into financial planning and investment strategies for young individuals.

## How It Works

1. **State Space**: Includes current portfolio distribution, financial/economic market state, and time left until the goal deadline.
2. **Action Space**: Encompasses buying, selling, and holding various asset classes (SPY 500 stocks, treasury bonds, and gold).
3. **Reward Model**: Combines the Sharpe ratio for risk-adjusted returns with a measure of progress towards the financial goal.

## Getting Started

1. **Clone the Repository**: Use `git clone` to clone this repository to your local machine.
2. **Data Files**: Ensure all necessary data files (`assets_historical_data.csv` and `economic_indicators_data.csv`) are placed in the correct directory.

## Usage

Run the 'datacollection.py' script to get the data and then run the 'financial_simulation.py' script to execute the portfolio optimization model. The script will output the portfolio's performance metrics and show various plots for in-depth analysis.

Graphs and tables illustrate the learning progress of the RL agent.


## Future Work
- **Fix reward implementation**: Fix asset allocation and total rewards per epoch.
- **Risk Profile Personalization**: Future updates aim to incorporate personalized strategies based on the user's risk profile.
- **Multi-goal Optimization**: Extend the model to optimize for multiple financial goals simultaneously.

## Contact

Jesica GonzaleZ - [jesgonz@stanford.edu]
