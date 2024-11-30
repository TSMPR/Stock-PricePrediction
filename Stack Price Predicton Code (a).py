
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
dates = pd.date_range(start="2023-01-01", periods=100)
prices = 100 + np.random.randn(100).cumsum()  # Simulate random price fluctuations
data = pd.DataFrame({'Date': dates, 'Price': prices})
data['Day'] = np.arange(len(data))  # Create a numerical 'day' feature

X = data[['Day']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Price'], label='Actual Price')
plt.plot(data['Date'], y_pred, label='Predicted Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
  
print("\nSummary:")
print(f"Model used: Linear Regression")
print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")
print(f"R-squared score: {model.score(X_test, y_test)}") # Evaluate the model
print(f"Data generated:\n {data.head()}")
