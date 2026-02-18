# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/69eca247-4a7f-49b7-8cf7-3c1d21a57b76" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Pavitra J
### Register Number: 212224110043
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('sample.csv (1).xls')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

schaler = MinMaxScaler()
X_train = schaler.fit_transform(X_train)
X_test = schaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: Pavitra J
# Register Number: 212224110043
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
pavitra = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(pavitra.parameters(), lr=0.001)

# Name: Pavitra J
# Register Number: 212224110043
def train_model(pavitra, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(pavitra(X_train), y_train)
        loss.backward()
        optimizer.step()


        # Append loss inside the loop
        jisha.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(pavitra, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(pavitra(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(pavitra.history)

import matplotlib.pyplot as plt
print("\nName: Pavitra J")
print("Register Number:212224110043")
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = pavitra(torch.tensor(schaler.transform(X_n1_1), dtype=torch.float32)).item()
print("\nName: Pavitra J")
print("Register Number:212224110043")
print(f'Prediction: {prediction}')


```
## Dataset Information

<img width="473" height="372" alt="image" src="https://github.com/user-attachments/assets/df7997aa-a92d-4cfd-99b9-71b3803e93e4" />


## OUTPUT
<img width="356" height="233" alt="image" src="https://github.com/user-attachments/assets/fd7a6158-b360-4271-a546-859627668056" />


### Training Loss Vs Iteration Plot

<img width="689" height="540" alt="image" src="https://github.com/user-attachments/assets/c86f64ff-8077-440b-8303-fe277995f5a3" />


### New Sample Data Prediction

<img width="297" height="73" alt="image" src="https://github.com/user-attachments/assets/edf334ed-aed0-4c82-89b8-dd4a891b84fe" />


## RESULT

Successfully executed the code to develop a neural network regression model.

