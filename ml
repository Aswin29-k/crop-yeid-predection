


# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"C:\Users\kaswi\Downloads\csv"
df = pd.read_csv(file_path)

# Print the first few rows of the dataset
print(df.head())

# Define the features (X) and target variable (y)
X = df.drop("Crop_Yield", axis=1)  # Assuming 'Crop_Yield' is the target variable
y = df["Crop_Yield"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)
print(y_pred)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
