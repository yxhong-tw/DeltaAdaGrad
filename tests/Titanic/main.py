# Set the path.
# -----
import sys

sys.path.append(".")
# -----

import argparse
import os
import random
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

from matplotlib.collections import EventCollection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict

from DeltaAdaGrad import DeltaAdaGrad

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "C", "Q", "S", np.nan]
TARGET = "Survived"


class PytorchDataSet(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        normarize_scaler: StandardScaler,
    ) -> None:
        if "Survived" not in df.columns:
            df["Survived"] = 9999

        self.df = df
        x = np.array(self.df[FEATURES])
        y = np.array(self.df[TARGET])

        x = normarize_scaler.transform(x)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"X": self.x[idx], "Y": self.y[idx]}


def random_seed(seed: int = 48763):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def data_preprocessing(df: pd.DataFrame, test: pd.DataFrame):
    dmean = df["Age"].dropna().mean()

    df["Age"] = df["Age"].fillna(value=dmean)
    test["Age"] = test["Age"].fillna(value=dmean)

    le = LabelEncoder()
    le.fit(df["Sex"])

    df["Sex"] = le.transform(df["Sex"])
    test["Sex"] = le.transform(test["Sex"])

    dfall = pd.concat(objs=[df, test], axis=0)
    dfall2 = pd.get_dummies(data=dfall["Embarked"], dummy_na=True)
    dfall = pd.concat(objs=[dfall, dfall2], axis=1)

    train = dfall.iloc[:len(df), :]
    test = dfall.iloc[len(df):, :]

    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=48763)

    folds = train.copy()
    for n, (_, val_index) in enumerate(Fold.split(folds, folds["Survived"])):
        folds.loc[val_index, "fold"] = int(n)

    folds["fold"] = folds["fold"].astype(dtype=int)

    p_train = folds[folds["fold"] != 0]
    p_val = folds[folds["fold"] == 0]

    p_train = p_train.reset_index(drop=True)
    p_val = p_val.reset_index(drop=True)

    normarize_scaler = StandardScaler()
    normarize_scaler.fit(np.array(train[FEATURES]))

    train_dataset = PytorchDataSet(
        df=p_train,
        normarize_scaler=normarize_scaler,
    )
    val_dataset = PytorchDataSet(
        df=p_val,
        normarize_scaler=normarize_scaler,
    )
    test_dataset = PytorchDataSet(
        df=test,
        normarize_scaler=normarize_scaler,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=256 * 2,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=256 * 2,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features=len(FEATURES), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train_once(train_dataloader: DataLoader, model: Net):
    total_loss = 0
    total_acc = 0

    model.train()

    for a in train_dataloader:
        train_x = a["X"].to(device)
        train_y = a["Y"].to(device)

        optimizer.zero_grad()

        output = model(x=train_x)

        loss = criterion(output, train_y)
        loss.backward()

        total_loss += loss.item()
        total_acc += accuracy_score(
            y_true=train_y.cpu(),
            y_pred=torch.max(output.data.cpu(), 1)[1],
        )

        try:
            optimizer.step(loss=loss)
        except:
            optimizer.step()
    # -----

    total_loss = total_loss / len(train_dataloader)
    total_acc = total_acc / len(train_dataloader)

    return model, total_loss, total_acc


def val_once(val_dataloader, model):
    total_loss = 0
    total_acc = 0

    model.eval()

    for a in val_dataloader:
        with torch.no_grad():
            val_x = a["X"].to(device)
            val_y = a["Y"].to(device)

            output = model(x=val_x)
            total_acc += accuracy_score(
                y_true=val_y.cpu(),
                y_pred=torch.max(output.data.cpu(), 1)[1],
            )

            loss = criterion(output, val_y)
            total_loss += loss.item()

    total_loss = total_loss / len(val_dataloader)
    total_acc = total_acc / len(val_dataloader)

    return total_loss, total_acc


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
        default=1000,
        help="The Number of Epoch.",
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

    df = pd.read_csv(filepath_or_buffer="tests/Titanic/data/train.csv")
    test = pd.read_csv(filepath_or_buffer="tests/Titanic/data/test.csv")

    train_dataloader, val_dataloader, test_dataloader = \
        data_preprocessing(df=df, test=test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.01)
    elif args.optimizer == "DeltaAdaGrad":
        optimizer = DeltaAdaGrad(params=model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_epoch = -1
    best_loss = float("inf")
    best_val_acc = 0
    all_res = []
    for epoch in tqdm(range(args.epoch)):
        model, train_loss, train_acc = train_once(
            train_dataloader=train_dataloader,
            model=model,
        )
        val_loss, val_acc = val_once(
            val_dataloader=val_dataloader,
            model=model,
        )

        if val_acc > best_val_acc:
            best_epoch = epoch
            best_loss = val_loss
            best_val_acc = val_acc

        all_res.append([epoch, train_loss, val_loss, train_acc, val_acc])
        print(
            f"Epoch: {epoch}, TLoss: {train_loss}, VLoss: {val_loss}, TAcc: {train_acc}, VAcc: {val_acc}, best: {best_val_acc}"
        )

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Loss: {best_loss}")
    print(f"Best VAcc: {best_val_acc}")

    epochs = [x[0] for x in all_res]
    train_losses = [x[1] for x in all_res]
    val_losses = [x[2] for x in all_res]

    plt.plot(epochs, train_losses, "r", label="Train Loss")
    plt.plot(epochs, val_losses, "b", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"outputs/Titanic/{args.optimizer}_loss_{args.epoch}.png")
    plt.clf()

    train_accs = [x[3] for x in all_res]
    val_accs = [x[4] for x in all_res]
    plt.plot(epochs, train_accs, "r", label="Train Accuracy")
    plt.plot(epochs, val_accs, "b", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(f"outputs/Titanic/{args.optimizer}_acc_{args.epoch}.png")
    plt.clf()
