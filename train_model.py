import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Load dataset
data = pd.read_csv("train.csv")

# Encode Location
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Select features
X = data[
    [
        "Area",
        "Location",
        "No. of Bedrooms",
        "New/Resale",
        "Gymnasium",
        "Car Parking",
        "Indoor Games",
        "Jogging Track",
    ]
]

# Target
y = data["Price"] / 1000000  # app.py multiplies by 1e6

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("New model.pkl created successfully!")