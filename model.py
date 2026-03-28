import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1 - Data load karo
df = pd.read_csv('HeartDiseaseTrain-Test.csv')

# Step 2 - Columns select karo
features = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate']
X = df[features]
y = df['target']

# Step 3 - Train aur Test split karo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 - Model banao aur train karo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5 - Accuracy check karo
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6 - Model save karo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")