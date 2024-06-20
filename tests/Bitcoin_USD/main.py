import sys
sys.path.append(".")
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import random

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from DeltaAdaGrad import DeltaAdaGrad

# Model definition
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # hidden state
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out
    
def random_seed(seed: int = 48763):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Define the root mean squared error function
def rmse(predictions, targets):
    return torch.sqrt(((predictions - targets) ** 2).mean())

# Function to split sequences
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(input_sequences): break
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def train(n_epochs, model, optimizer, loss_fn, X_train, y_train, X_test, y_test,scheduler):
    train_losses = []
    test_losses = []
    train_rmses = []
    test_rmses = []
    
    best_epoch = -1
    best_loss = float("inf")
    best_rmse = 100
    model.train()
    for epoch in tqdm(range(n_epochs)):
        outputs = model(X_train)  # forward pass
        optimizer.zero_grad()  # calculate the gradient, manually setting to 0
        loss = loss_fn(outputs, y_train)  # obtain the loss function
        loss.backward()  # calculates the loss of the loss function

        try:
            optimizer.step(loss=loss)
        except:
            optimizer.step()

        # Compute training RMSE
        train_rmse = rmse(outputs, y_train)
        
        # Evaluate test data
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds, y_test)
            test_rmse = rmse(test_preds, y_test)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_rmses.append(train_rmse.item())
        test_rmses.append(test_rmse.item())
        
        if test_rmse < best_rmse:
            best_loss = test_loss
            best_epoch = epoch
            best_rmse = test_rmse

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Tloss: {loss.item():1.5f}, Vloss: {test_loss.item():1.5f}, T_rmse: {train_rmse.item():1.5f}, V_rmse: {test_rmse.item():1.5f}, Best rmse: {best_rmse:1.5f}")
        
        if scheduler is not None:
            scheduler.step()
    print(f"Best epoch: {best_epoch}, Best rmse: {best_rmse:1.5f}")
    return train_losses, test_losses, train_rmses, test_rmses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        choices=["AdaGrad", "Adam", "DeltaAdaGrad"],
        required=True,
        help="The Name of Optimizer.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=10000,
        help="The Number of Epoch.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The Learning Rate.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=48763,
        help="The Random Seed.",
    )

    args = parser.parse_args()

    random_seed(seed=args.seed)

    # Check if GPU is available
    device = torch.device('cpu')
    # Check if the device is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
    if torch.backends.mps.is_available():
        device = torch.device('mps')   

    print("Using device:",device)

    # Load and preprocess the data
    df = pd.read_csv('tests/Bitcoin_USD/input/BTC-USD-2.csv', parse_dates=True)
    df.drop(columns=['Adj Close'], inplace=True)

    # Separate date column if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    X, y = df.drop(columns=['Close']), df.Close.values

    mm = MinMaxScaler()
    ss = StandardScaler()

    # Ensure all columns in X are numerical
    X_trans = ss.fit_transform(X.select_dtypes(include=[np.number]))
    y_trans = mm.fit_transform(y.reshape(-1, 1))
    X_ss, y_mm = split_sequences(X_trans, y_trans, 100, 50)

    total_samples = len(X)
    train_test_cutoff = round(0.90 * total_samples)

    X_train, X_test= X_ss[:-150], X_ss[-150:]
    y_train, y_test = y_mm[:-150], y_mm[-150:]


    # Convert to pytorch tensors
    X_train_tensors = torch.Tensor(X_train)
    X_test_tensors = torch.Tensor(X_test)

    y_train_tensors = torch.Tensor(y_train)
    y_test_tensors = torch.Tensor(y_test)

    # Reshaping to rows, timestamps, features
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 100, X_train_tensors.shape[2]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 100, X_test_tensors.shape[2]))

    X_check, y_check = split_sequences(X, y.reshape(-1, 1), 100, 50)
    X_check[-1][0:4]

    n_epochs = args.epoch
    learning_rate = args.learning_rate

    input_size = X_train.shape[2]  # number of features
    hidden_size = 2  # number of features in hidden state
    num_layers = 1  # number of stacked lstm layers

    num_classes = 50  

    model = LSTM(num_classes, input_size, hidden_size, num_layers)
    model.to(device)
    loss_fn = torch.nn.MSELoss()  

    if args.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(
            params = model.parameters(),
            lr = args.learning_rate,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr=args.learning_rate,
        )
    elif args.optimizer == "DeltaAdaGrad":
        optimizer = DeltaAdaGrad(
            params = model.parameters(),
            lr = args.learning_rate,
        )
    else:
        raise ValueError(f"The optimizer {args.optimizer} is not supported.")

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_losses, test_losses, train_rmses, test_rmses = train(n_epochs=n_epochs,
                                                                    model=model,
                                                                    optimizer=optimizer,
                                                                    loss_fn=loss_fn,
                                                                    X_train=X_train_tensors_final.to(device),
                                                                    y_train=y_train_tensors.to(device),
                                                                    X_test=X_test_tensors_final.to(device),
                                                                    y_test=y_test_tensors.to(device),
                                                                    scheduler=None)


    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation Loss')
    plt.legend()
    plt.savefig(f"outputs/Bitcoin_USD/{args.optimizer}_loss_epoch_{args.epoch}_lr_{args.learning_rate}.png")
    plt.close()

    # Plot RMSE
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(test_rmses, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and validation RMSE')
    plt.legend()
    plt.savefig(f"outputs/Bitcoin_USD/{args.optimizer}_acc_epoch_{args.epoch}_lr_{args.learning_rate}.png")
    plt.close()

