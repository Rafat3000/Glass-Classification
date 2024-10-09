
# Glass Classification using Neural Network

## Dataset Information
The dataset used for this model is related to the classification of different types of glass based on various physical and chemical attributes. The attributes include:
- RI: Refractive Index
- Na: Sodium content
- Mg: Magnesium content
- Al: Aluminum content
- Si: Silicon content
- K: Potassium content
- Ca: Calcium content
- Ba: Barium content
- Fe: Iron content
- Type: Type of glass (Target)

## Libraries and Data Loading
```python
import pandas as pd

# Load Data 
file_path = "/kaggle/input/glass/glass.csv"
data = pd.read_csv(file_path) 
data.head()
Sample output:

RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	Type
1.5210	13.64	4.49	1.10	71.78	0.06	8.75	0.0	0.0	1
1.5176	13.89	3.60	1.36	72.73	0.48	7.83	0.0	0.0	1
1.5161	13.53	3.55	1.54	72.99	0.39	7.78	0.0	0.0	1
python
Copy code
# Check distribution of the target variable
data['Type'].value_counts()
Output:

yaml
Copy code
Type
2    76
1    70
7    29
3    17
5    13
6     9
Name: count, dtype: int64
Data Preprocessing

Splitting Features and Target
python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split Features and Target
X = data.drop('Type', axis=1)
y = data['Type'] - 1  # Adjust target to zero-based indexing

# Scaling features using Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
Building the Neural Network

Using TensorFlow and Keras to build an Artificial Neural Network (ANN) for classification.

python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build Sequential model
model = Sequential()

# Add input layer and first hidden layer
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Add more hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# Add output layer with 7 units (for the 7 glass types) and softmax activation
model.add(Dense(7, activation='softmax'))

# Model summary
model.summary()
Model Compilation
python
Copy code
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Training the Model
python
Copy code
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
Sample training output:

arduino
Copy code
Epoch 1/100
14/14 ━━━━━━━━━━━━━━━━━━━━ 2s 25ms/step - accuracy: 0.2974 - loss: 1.8536 - val_accuracy: 0.4286 - val_loss: 1.5107
Epoch 2/100
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6038 - loss: 1.3588 - val_accuracy: 0.4857 - val_loss: 1.3663
...
Epoch 100/100
14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9985 - loss: 0.0211 - val_accuracy: 0.6857 - val_loss: 4.4292
Conclusion

The model achieves a high training accuracy, though the validation accuracy suggests overfitting. Further hyperparameter tuning or regularization might improve the model's performance on unseen data.

vbnet
Copy code

This Markdown structure covers the key sections of the code, including data loading, preprocessing, model building, training, and evaluation.





