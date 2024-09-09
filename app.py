from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize FastAPI
app = FastAPI()

# Load and train the model on startup
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define an endpoint to make predictions via query parameters
@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    # Validate inputs
    if any([sepal_length <= 0, sepal_width <= 0, petal_length <= 0, petal_width <= 0]):
        raise HTTPException(status_code=400, detail="Input values must be greater than 0.")
    
    # Convert input data from query parameters into a numpy array for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction using the trained model
    prediction = model.predict(features)
    
    # Map the predicted class back to the target names
    predicted_class = iris.target_names[prediction[0]]
    
    return {"prediction": predicted_class}

# Define a root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Iris Classifier API is up and running!"}