# import the Data
# Clean the Data
# Split the Data into Training/Test Sets
# Create a Model 
# Train the Model 
# Make the Predictions
# Evaluate and improve 

import pandas as pd;

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)
predictiions = model.predict([[21, 1], [22, 0]])
print(predictiions)
