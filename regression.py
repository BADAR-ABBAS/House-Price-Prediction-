import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
df = pd.read_csv('House Price Prediction Dataset.csv')

# 2. One-Hot Encode Categorical Features properly
categorical_cols = ['Location', 'Condition', 'Garage']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Separate Features (X) and Target (y)
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# 4. Split Train and Test BEFORE Scaling (Prevents Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit Scaler strictly on Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Fit Model & Evaluate
model = LinearRegression()
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

print(f"R2 Score: {r2_score(y_test, pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")