import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load dataset (Wine Quality)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# 2. Separate features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# --- PROFESSIONAL ADD-ON: SCALING ---
# Important: Linear models perform much better when data is on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize model
model = LinearRegression()

# 4. Perform K-Fold Cross-Validation (The "Practice Exam")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')

print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Mean R2 Score: {np.mean(cv_scores):.4f}")

# --- THE MISSING STEP: FINAL FITTING ---
# Now we train the model on the ENTIRE dataset so it is ready for the app
model.fit(X_scaled, y)

# 5. Save the 'Brain' (Model) and the 'Filter' (Scaler)
with open('wine_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('wine_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler successfully fitted and saved!")