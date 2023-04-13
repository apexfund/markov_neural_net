import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

#  prepare input data and target variables for the neural net
def prepare_data(prices, steps):
    X = []
    y = []
    for i in range(len(prices)-steps):
        X.append(prices[i:i+steps])
        y.append(prices[i+steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

api_endpoint = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'outputsize': 'full',
    'datatype': 'json',
    'apikey': 'SAXMJP1U7ZN7OR11',
    'symbol': input("Enter in your stock: ")
}

# api call
response = requests.get(api_endpoint, params=params)
data = response.json()['Time Series (Daily)']
df = pd.DataFrame(data).transpose()
df.index = pd.to_datetime(df.index)
df = df.astype(float)

# States based on different price levels
states = np.arange(0, 2 * max(df['4. close']), 10)

# freq of transitions between states
transition_counts = np.zeros((len(states), len(states)))
for i in range(1, len(df)):
    current_price = df['4. close'][i]
    previous_price = df['4. close'][i-1]
    current_state = np.argmin(np.abs(states - current_price))
    previous_state = np.argmin(np.abs(states - previous_price))
    transition_counts[previous_state, current_state] += 1

# transition prob
transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

# future prices based on the transition matrix
current_price = df['4. close'][-1]
num_steps = 200
simulated_prices = []
for i in range(num_steps):
    current_state = np.argmin(np.abs(states - current_price))
    next_state = np.random.choice(len(states), p=transition_probs[current_state])
    next_price = states[next_state]
    simulated_prices.append(next_price)
    current_price = next_price

# data prep for the neural network
X, y = prepare_data(simulated_prices, 5)

# define and compile the neural network
model = Sequential()
model.add(Dense(256, input_shape=(5,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

# train neural net
model.fit(X, y, epochs=200, batch_size=64, verbose=0)

# Predict future prices using the trained neural network
current_prices = simulated_prices[-5:]
predictions = []
for i in range(num_steps):
    X = np.array([current_prices])
    next_price = model.predict(X)[0][0]
    predictions.append(next_price)
    simulated_prices.append(next_price)
    current_prices = simulated_prices[-5:]

plt.style.use('seaborn-whitegrid')
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(simulated_prices)), simulated_prices, linewidth=2, color='navy')
ax.set_xlabel('Days')
ax.set_ylabel('Simulated Price')
ax.set_title(f"Simulated Future Prices for AAPL", fontweight='bold')
ax.legend(['Simulated Price'], loc='upper right')
ax.tick_params(axis='both', which='major', labelsize=12)
plt.show()