# DL- Developing a Recurrent Neural Network Model for Stock Prediction
# NAME: ADITHYA M
# REG NO: 21224230008
## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

STOCK PREDICTION

## DESIGN STEPS

## STEP 1:
Load the training and test datasets containing historical Google stock prices using the Pandas library.

## STEP 2:
Extract the Close price column from both datasets and normalize the values using the MinMaxScaler. The scaler is fitted only on the training data to avoid data leakage.

## STEP 3:
Convert the normalized stock prices into sequential data. A sequence length of 60 is used, where the previous 60 closing prices are used to predict the next closing price.

## STEP 4:
Convert the generated sequences into PyTorch tensors and create a DataLoader for batch-wise training.

## STEP 5:
Develop an RNN model using PyTorch with two RNN layers and 64 hidden units. The output from the final time step is passed through a fully connected layer to predict the stock price.

## STEP 6:
Train the RNN model using the Mean Squared Error loss function and Adam optimizer. Predict the stock prices for the test dataset, convert the predictions back to the original scale, and compare them with the actual stock prices using graphical visualization.





## PROGRAM

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df_train = pd.read_csv("Exp-5_train.csv")
df_test = pd.read_csv("Exp-5_test.csv")

print("Training Dataset:")
print(df_train.head())

print("\nTest Dataset:")
print(df_test.head())


# ==========================================
# STEP 2: PREPROCESS DATA
# ==========================================

train_prices = df_train["Close"].values.reshape(-1, 1)
test_prices = df_test["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)


# ==========================================
# STEP 3: CREATE SEQUENCES
# ==========================================

def create_sequences(data, seq_length):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(data[i:i + seq_length])

        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


seq_length = 60

X_train, y_train = create_sequences(
    scaled_train,
    seq_length
)

X_test, y_test = create_sequences(
    scaled_test,
    seq_length
)


# ==========================================
# STEP 4: CONVERT TO PYTORCH TENSORS
# ==========================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ==========================================
# CREATE DATALOADER
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# ==========================================
# STEP 5: DEFINE RNN MODEL
# ==========================================

class RNNModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ):

        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self, x):

        out, _ = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# ==========================================
# STEP 6: INITIALIZE MODEL
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = RNNModel().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# STEP 7: TRAIN MODEL
# ==========================================

epochs = 50

train_losses = []

model.train()

for epoch in range(epochs):

    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    train_losses.append(
        average_loss
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.6f}"
    )


# ==========================================
# STEP 8: PLOT TRAINING LOSS
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Training Loss Over Epochs"
)

plt.grid(True)

plt.show()


# ==========================================
# STEP 9: MAKE PREDICTIONS
# ==========================================

model.eval()

with torch.no_grad():

    predicted = model(
        X_test_tensor.to(device)
    ).cpu().numpy()

    actual = y_test_tensor.cpu().numpy()


# Convert back to original prices

predicted_prices = scaler.inverse_transform(
    predicted
)

actual_prices = scaler.inverse_transform(
    actual
)


# ==========================================
# STEP 10: PLOT RESULTS
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="True Stock Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Stock Price"
)

plt.xlabel("Time")

plt.ylabel("Stock Price")

plt.title(
    "True Stock Price vs Predicted Stock Price using RNN"
)

plt.legend()

plt.grid(True)

plt.show()


# ==========================================
# STEP 11: DISPLAY PREDICTIONS
# ==========================================

results = pd.DataFrame({

    "True Stock Price":
        actual_prices.flatten(),

    "Predicted Stock Price":
        predicted_prices.flatten()

})

print("\nFirst 20 Predictions:")

print(results.head(20))


# ==========================================
# STEP 12: EVALUATION METRICS
# ==========================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

mse = mean_squared_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(mse)

print("\nModel Evaluation:")

print("MAE:", mae)

print("MSE:", mse)

print("RMSE:", rmse)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="825" height="452" alt="image" src="https://github.com/user-attachments/assets/30a7e364-e8ec-42ff-b774-1606dba070aa" />


## True Stock Price, Predicted Stock Price vs time

<img width="873" height="451" alt="image" src="https://github.com/user-attachments/assets/d6e9e9e8-2a26-4129-84f7-f48883226ca0" />


### Predictions
<img width="477" height="660" alt="image" src="https://github.com/user-attachments/assets/71a850c3-9160-4060-87b6-a3f84874adc3" />


## RESULT
The Recurrent Neural Network (RNN) model was successfully developed and trained using historical Google stock closing price data
