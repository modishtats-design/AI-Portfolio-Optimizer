import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Financial data class
class FinancialData:
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']
        self.data = {}
        
    def fetch_data(self, period="2y"):
        """Fetch historical stock data using yfinance"""
        st.write("Fetching market data...")
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                self.data[symbol] = hist
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
        
        # Calculate returns
        self.returns = {}
        for symbol, df in self.data.items():
            if not df.empty:
                self.returns[symbol] = df['Close'].pct_change().dropna()
        
        st.success("Data fetched successfully!")
        return self.data, self.returns

# LSTM Model for time series forecasting
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=50, output_dim=1, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

# Dataset for LSTM training
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=30):
        self.data = data
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+self.sequence_length]
        return torch.FloatTensor(seq), torch.FloatTensor([target])

# Simple RL Agent for portfolio optimization
class PortfolioAgent:
    def __init__(self, n_assets, learning_rate=0.01):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets  # Start with equal weights
        self.learning_rate = learning_rate
        self.portfolio_history = []
        
    def update_weights(self, returns, risk_free_rate=0.02):
        """Update portfolio weights based on returns"""
        # Calculate portfolio performance
        portfolio_return = np.sum(self.weights * returns)
        
        # Simple reinforcement learning: increase weights of assets with positive returns
        # and decrease weights of assets with negative returns
        adjustment = np.where(returns > 0, self.learning_rate, -self.learning_rate)
        
        # Update weights
        new_weights = self.weights + adjustment
        
        # Ensure weights sum to 1 and are non-negative
        new_weights = np.clip(new_weights, 0, 1)
        new_weights /= np.sum(new_weights)
        
        self.weights = new_weights
        self.portfolio_history.append(self.weights.copy())
        
        return portfolio_return

# Multi-Agent System for portfolio optimization
class MultiAgentSystem:
    def __init__(self, n_assets, n_agents=3):
        self.n_assets = n_assets
        self.n_agents = n_agents
        
        # Create agents with different objectives
        self.return_agent = PortfolioAgent(n_assets, learning_rate=0.03)  # Focus on returns
        self.risk_agent = PortfolioAgent(n_assets, learning_rate=0.01)    # Focus on risk reduction
        self.diversification_agent = PortfolioAgent(n_assets, learning_rate=0.02)  # Focus on diversification
        
        self.agent_weights = np.array([0.4, 0.3, 0.3])  # Weighting of each agent's recommendation
        
    def get_portfolio_weights(self, returns, volatility):
        """Get portfolio weights from all agents and combine them"""
        # Each agent updates based on different aspects of the data
        return_agent_weights = self.return_agent.update_weights(returns)
        risk_agent_weights = self.risk_agent.update_weights(-volatility)  # Minimize volatility
        diversification_agent_weights = self.diversification_agent.update_weights(
            np.ones(self.n_assets) / self.n_assets)  # Encourage diversification
        
        # Combine agent recommendations
        combined_weights = (self.agent_weights[0] * self.return_agent.weights +
                           self.agent_weights[1] * self.risk_agent.weights +
                           self.agent_weights[2] * self.diversification_agent.weights)
        
        return combined_weights

# Monte Carlo Simulation
class MonteCarloSimulator:
    def __init__(self, returns_data, n_simulations=1000):
        self.returns_data = returns_data
        self.n_simulations = n_simulations
        self.symbols = list(returns_data.keys())
        
    def simulate_portfolio(self, weights, days=252, initial_investment=10000):
        """Run Monte Carlo simulation for a given portfolio"""
        # Calculate mean returns and covariance matrix
        returns_matrix = pd.DataFrame(self.returns_data).values.T
        mean_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)
        
        # Generate random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, size=(self.n_simulations, days)
        )
        
        # Calculate portfolio values over time
        portfolio_values = np.zeros((self.n_simulations, days))
        for i in range(self.n_simulations):
            portfolio_values[i, 0] = initial_investment
            for j in range(1, days):
                daily_return = np.sum(weights * simulated_returns[i, j, :])
                portfolio_values[i, j] = portfolio_values[i, j-1] * (1 + daily_return)
        
        return portfolio_values
    
    def calculate_var(self, portfolio_values, confidence_level=0.95):
        """Calculate Value at Risk"""
        final_values = portfolio_values[:, -1]
        var = np.percentile(final_values, (1 - confidence_level) * 100)
        return var
    
    def calculate_cvar(self, portfolio_values, confidence_level=0.95):
        """Calculate Conditional Value at Risk"""
        final_values = portfolio_values[:, -1]
        var = self.calculate_var(portfolio_values, confidence_level)
        cvar = final_values[final_values <= var].mean()
        return cvar

# Portfolio Optimizer
class PortfolioOptimizer:
    def __init__(self, returns_data):
        self.returns_data = returns_data
        self.symbols = list(returns_data.keys())
        self.returns_matrix = pd.DataFrame(returns_data).values.T
        self.mean_returns = np.mean(self.returns_matrix, axis=1)
        self.cov_matrix = np.cov(self.returns_matrix)
        
    def optimize_sharpe_ratio(self):
        """Optimize portfolio for maximum Sharpe ratio"""
        n_assets = len(self.symbols)
        
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = n_assets * [1. / n_assets]
        
        result = minimize(negative_sharpe_ratio, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def optimize_min_variance(self):
        """Optimize portfolio for minimum variance"""
        n_assets = len(self.symbols)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = n_assets * [1. / n_assets]
        
        result = minimize(portfolio_variance, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def optimize_max_return(self, max_volatility=0.2):
        """Optimize portfolio for maximum return with volatility constraint"""
        n_assets = len(self.symbols)
        
        def negative_portfolio_return(weights):
            return -np.sum(self.mean_returns * weights)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: max_volatility - portfolio_volatility(x)}
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = n_assets * [1. / n_assets]
        
        result = minimize(negative_portfolio_return, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x

# Streamlit App
def main():
    
    st.title("🤖 AI-Driven Financial Portfolio Optimizer")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Sidebar for user inputs
    st.sidebar.header("Investment Preferences")
    initial_investment = st.sidebar.number_input("Initial Investment ($)", 
                                               value=10000, min_value=1000, step=1000)
    risk_tolerance = st.sidebar.slider("Risk Tolerance", 1, 10, 5)
    investment_horizon = st.sidebar.selectbox("Investment Horizon", 
                                            ["Short-term (1-2 years)", "Medium-term (3-5 years)", "Long-term (5+ years)"])
    
    # Fetch data
    if st.button("Fetch Market Data") or st.session_state.data_loaded:
        financial_data = FinancialData()
        data, returns = financial_data.fetch_data()
        st.session_state.data = data
        st.session_state.returns = returns
        st.session_state.data_loaded = True
        
        # Show returns data
        st.subheader("Asset Returns")
        returns_df = pd.DataFrame(returns)
        st.dataframe(returns_df.describe())
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        fig = px.imshow(returns_df.corr(), text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.data_loaded:
        data = st.session_state.data
        returns = st.session_state.returns
        
        # Portfolio Optimization
        st.header("Portfolio Optimization")
        
        optimizer = PortfolioOptimizer(returns)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Optimize for Sharpe Ratio"):
                weights = optimizer.optimize_sharpe_ratio()
                st.subheader("Sharpe Ratio Optimized Portfolio")
                for i, symbol in enumerate(returns.keys()):
                    st.write(f"{symbol}: {weights[i]:.2%}")
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(labels=list(returns.keys()), values=weights)])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("Optimize for Minimum Variance"):
                weights = optimizer.optimize_min_variance()
                st.subheader("Minimum Variance Portfolio")
                for i, symbol in enumerate(returns.keys()):
                    st.write(f"{symbol}: {weights[i]:.2%}")
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(labels=list(returns.keys()), values=weights)])
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.button("Optimize for Maximum Return"):
                max_vol = 0.1 + (risk_tolerance / 10) * 0.3  # Adjust based on risk tolerance
                weights = optimizer.optimize_max_return(max_vol)
                st.subheader(f"Max Return Portfolio (Volatility ≤ {max_vol:.1%})")
                for i, symbol in enumerate(returns.keys()):
                    st.write(f"{symbol}: {weights[i]:.2%}")
                
                # Pie chart
                fig = go.Figure(data=[go.Pie(labels=list(returns.keys()), values=weights)])
                st.plotly_chart(fig, use_container_width=True)
        
        # Monte Carlo Simulation
        st.header("Risk Analysis with Monte Carlo Simulation")
        
        # Get weights from user or use equal weights
        st.subheader("Portfolio Weights for Simulation")
        weights = {}
        cols = st.columns(len(returns))
        for i, (symbol, col) in enumerate(zip(returns.keys(), cols)):
            weights[symbol] = col.number_input(symbol, value=1.0/len(returns), 
                                             min_value=0.0, max_value=1.0, step=0.05, key=f"weight_{symbol}")
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            for symbol in weights:
                weights[symbol] /= total
        
        if st.button("Run Monte Carlo Simulation"):
            simulator = MonteCarloSimulator(returns)
            weight_array = np.array([weights[symbol] for symbol in returns.keys()])
            
            portfolio_values = simulator.simulate_portfolio(weight_array, initial_investment=initial_investment)
            
            # Calculate risk metrics
            var = simulator.calculate_var(portfolio_values)
            cvar = simulator.calculate_cvar(portfolio_values)
            
            st.subheader("Risk Metrics")
            st.metric("Value at Risk (95% confidence)", f"${var:,.2f}")
            st.metric("Conditional Value at Risk (95% confidence)", f"${cvar:,.2f}")
            
            # Plot simulation results
            st.subheader("Monte Carlo Simulation Results")
            fig = go.Figure()
            
            for i in range(min(100, len(portfolio_values))):  # Plot first 100 simulations
                fig.add_trace(go.Scatter(
                    y=portfolio_values[i], 
                    mode='lines',
                    line=dict(width=1, color='blue'),
                    opacity=0.1,
                    showlegend=False
                ))
            
            # Add mean line
            mean_values = np.mean(portfolio_values, axis=0)
            fig.add_trace(go.Scatter(
                y=mean_values,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Portfolio Value'
            ))
            
            fig.update_layout(
                title="Monte Carlo Simulation of Portfolio Value",
                xaxis_title="Days",
                yaxis_title="Portfolio Value ($)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Multi-Agent RL Section
        st.header("Multi-Agent Portfolio Optimization")
        
        if st.button("Run Multi-Agent Optimization"):
            st.write("Running multi-agent optimization...")
            
            # Create multi-agent system
            n_assets = len(returns)
            multi_agent = MultiAgentSystem(n_assets)
            
            # Convert returns to numpy array for processing
            returns_array = pd.DataFrame(returns).values
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns_array, axis=0)
            
            # Run simulation
            portfolio_values = [initial_investment]
            weights_history = []
            
            for i in range(len(returns_array)):
                # Get current returns
                current_returns = returns_array[i]
                
                # Get portfolio weights from multi-agent system
                weights = multi_agent.get_portfolio_weights(current_returns, volatility)
                weights_history.append(weights)
                
                # Calculate portfolio return
                portfolio_return = np.sum(weights * current_returns)
                
                # Update portfolio value
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
            
            # Plot performance
            st.subheader("Multi-Agent Portfolio Performance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=portfolio_values,
                mode='lines',
                name='Multi-Agent Portfolio Value'
            ))
            
            # Compare with benchmark (equal weights)
            benchmark_weights = np.ones(n_assets) / n_assets
            benchmark_values = [initial_investment]
            
            for i in range(len(returns_array)):
                daily_return = np.sum(benchmark_weights * returns_array[i])
                benchmark_values.append(benchmark_values[-1] * (1 + daily_return))
            
            fig.add_trace(go.Scatter(
                y=benchmark_values,
                mode='lines',
                name='Equal Weight Benchmark'
            ))
            
            fig.update_layout(
                title="Multi-Agent Portfolio vs Benchmark",
                xaxis_title="Days",
                yaxis_title="Portfolio Value ($)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show final portfolio allocation
            st.subheader("Final Multi-Agent Portfolio Allocation")
            final_weights = weights_history[-1]
            
            fig = go.Figure(data=[go.Pie(
                labels=list(returns.keys()),
                values=final_weights
            )])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show agent contributions
            st.subheader("Agent Contributions")
            agent_names = ['Return Agent', 'Risk Agent', 'Diversification Agent']
            agent_contributions = multi_agent.agent_weights
            
            fig = go.Figure(data=[go.Pie(
                labels=agent_names,
                values=agent_contributions
            )])
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()