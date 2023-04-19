import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error

# prepare input data and target variables for the neural net
def prepare_data(prices, steps):
    X = []
    y = []
    for i in range(len(prices)-steps):
        X.append(prices[i:i+steps])
        y.append(prices[i+steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

# set up API call parameters
api_endpoint = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY_ADJUSTED',
    'outputsize': 'full',
    'datatype': 'json',
    'apikey': 'SAXMJP1U7ZN7OR11',
    'symbol': input("Enter in your stock: ")
}

# make API call and prepare data
response = requests.get(api_endpoint, params=params)
data = response.json()['Time Series (Daily)']
df = pd.DataFrame(data).transpose()
df.index = pd.to_datetime(df.index)
df = df.astype(float)

# create states based on different price levels
max_price = max(df['4. close'])
min_price = min(df['4. close'])
states = np.arange(min_price, 2 * max_price, 10)

# calculate frequency of transitions between states
transition_counts = np.zeros((len(states), len(states)))
for i in range(1, len(df)):
    current_price = df['4. close'][i]
    previous_price = df['4. close'][i-1]
    current_state = np.argmin(np.abs(states - current_price))
    previous_state = np.argmin(np.abs(states - previous_price))
    transition_counts[previous_state, current_state] += 1

# calculate transition probabilities
transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

# simulate future prices based on the transition matrix
current_price = df['4. close'][-1]
num_steps = 100
simulated_prices = []
for i in range(num_steps):
    current_state = np.argmin(np.abs(states - current_price))
    next_state = np.random.choice(len(states), p=transition_probs[current_state])
    next_price = states[next_state]
    simulated_prices.append(next_price)
    current_price = next_price

# prepare data for the neural network
X, y = prepare_data(simulated_prices, 5)

# create neural network
model = Sequential([
    Dense(256, input_shape=(5,), activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# compile the neural network
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

# define the training data
n_train = int(0.8 * len(simulated_prices))
X_train = []
y_train = []
for i in range(n_train - num_steps):
    X_train.append(simulated_prices[i:i+num_steps])
    y_train.append(simulated_prices[i+num_steps])
X_train = np.array(X_train)
y_train = np.array(y_train)


# train the neural network
history = model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0)

# make predictions using the trained neural network
current_prices = simulated_prices[-5:]
predictions = []
for i in range(num_steps):
    X = np.array([current_prices])
    next_price = model.predict(X)[0][0]
    predictions.append(next_price)
    simulated_prices.append(next_price)

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

MAE = np.mean(np.abs(predictions - simulated_prices[-num_steps:]))
MSE = np.mean(np.square(predictions - simulated_prices[-num_steps:]))
print(f'MAE: {MAE}')
print(f'MSE: {MSE}')

current_prices = simulated_prices[-5:]
predictions = []
for i in range(num_steps):
    X = np.array([current_prices])
    next_price = model.predict(X)[0][0]
    predictions.append(next_price)
    simulated_prices.append(next_price)
    current_prices = simulated_prices[-5:]

plt.figure(figsize=(12, 6))
plt.plot(range(len(simulated_prices)), simulated_prices, label='Simulated Price')
plt.plot(range(len(simulated_prices) - num_steps, len(simulated_prices)), predictions, label='Predicted Price')
plt.title(f'Simulated and Predicted Future Prices for {params["symbol"]}')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

MAE = mean_squared_error(simulated_prices[-num_steps:], predictions)
MSE = np.mean(np.abs(predictions - simulated_prices[-num_steps:]))
print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
