import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('diabetes.csv')  # Ensure this CSV is in the same folder

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
