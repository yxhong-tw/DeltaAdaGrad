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
from datasets import load_dataset

from matplotlib.collections import EventCollection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from DeltaAdaGrad import DeltaAdaGrad

def random_seed(seed: int = 48763):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

def data_preprocessing(dataset:Dataset):
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']
    batch_size = 16

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def train_once(train_dataloader: DataLoader, model):
    total_loss = []
    total_acc = []

    model.train()
    predictions, true_labels = [], []
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        total_loss.append(loss.item())

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())
        total_acc.append(accuracy_score(true_labels, predictions))

        try:
            optimizer.step(loss=loss)
        except:
            optimizer.step()
    # -----

    # total_loss = total_loss / len(train_dataloader)
    # total_acc = accuracy_score(true_labels, predictions)

    return model, total_loss, total_acc


def val_once(val_dataloader, model):
    total_loss = []
    total_acc = []

    model.eval()
    predictions, true_labels = [], []
    for batch in val_dataloader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss.append(loss.item())

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
            total_acc.append(accuracy_score(true_labels, predictions))
            
    # total_loss = total_loss / len(val_dataloader)
    # total_acc = accuracy_score(true_labels, predictions)

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

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-5,
        help="The Learning Rate.",
    )

    args = parser.parse_args()

    random_seed(seed=args.seed)
    os.environ['CUDA_VISIBLE_DEVICES']='1'

    dataset = load_dataset('glue', 'sst2')
    model_id = "google-bert/bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    train_dataloader, val_dataloader, test_dataloader = \
        data_preprocessing(dataset=dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    #     print("using at least two gpus")

    if args.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "DeltaAdaGrad":
        optimizer = DeltaAdaGrad(params=model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

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

        # if val_acc > best_val_acc:
        #     best_epoch = epoch
        #     best_loss = val_loss
        #     best_val_acc = val_acc

        all_res.append([epoch, train_loss, val_loss, train_acc, val_acc])
        # print(
        #     f"Epoch: {epoch}, TLoss: {train_loss}, VLoss: {val_loss}, TAcc: {train_acc}, VAcc: {val_acc}"
        # )
    # best_loss = min([x[2] for x in all_res])
    # best_val_acc = max([x[4] for x in all_res])                                                                                                                                                                                                                                                                                                                                                     
    # print(f"Best Epoch: {best_epoch}")
    # print(f"Best Loss: {best_loss}")
    # print(f"Best VAcc: {best_val_acc}")

    # epochs = [x[0] for x in all_res]
    train_losses = [loss for sublist in all_res for loss in sublist[1]]
    val_losses = [loss for sublist in all_res for loss in sublist[2]]
    train_batch_len = range(len(train_losses))
    val_batch_len = range(len(val_losses))

    print("min train_losses: ", min(train_losses))
    print("min val_losses: ", min(val_losses))

    if not os.path.exists("outputs/GLUE/sst2"):
        os.makedirs("outputs/GLUE/sst2")

    plt.plot(train_batch_len, train_losses, "r", label="Train Loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"outputs/GLUE/sst2/{args.optimizer}_train_loss_{args.learning_rate}.png")
    plt.clf()

    plt.plot(val_batch_len, val_losses, "b", label="Validation Loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"outputs/GLUE/sst2/{args.optimizer}_val_loss_{args.learning_rate}.png")
    plt.clf()

    train_accs = [acc for sublist in all_res for acc in sublist[3]]
    val_accs = [acc for sublist in all_res for acc in sublist[4]]

    print("best train_accs: ", max(train_accs))
    print("best val_accs: ", max(val_accs))

    plt.plot(train_batch_len, train_accs, "r", label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(f"outputs/GLUE/sst2/{args.optimizer}_train_acc_{args.learning_rate}.png")
    plt.clf()

    plt.plot(val_batch_len, val_accs, "b", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(f"outputs/GLUE/sst2/{args.optimizer}_val_acc_{args.learning_rate}.png")
    plt.clf()
