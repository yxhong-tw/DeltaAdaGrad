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

from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from DeltaAdaGrad import DeltaAdaGrad


def random_seed(seed: int = 48763):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_function_sst2_type(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
    )


def tokenize_function_mrpc_type(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
    )


def data_preprocessing(dataset: Dataset, args: argparse.Namespace):
    if args.dataset == "sst2" or args.dataset == "cola":
        tokenized_datasets = dataset.map(
            tokenize_function_sst2_type,
            batched=True,
        )
    elif args.dataset == "mrpc":
        tokenized_datasets = dataset.map(
            tokenize_function_mrpc_type,
            batched=True,
        )
    else:
        raise ValueError(f"The dataset {args.dataset} is not supported.")

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=(args.batch_size * 2),
        shuffle=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=(args.batch_size * 2),
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def train_once(
    train_dataloader: DataLoader,
    model: AutoModelForSequenceClassification,
    optimizer: torch.optim.Optimizer,
):
    total_loss = 0
    total_acc = 0

    model.train()

    predictions, true_labels = [], []
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

        try:
            optimizer.step(loss=loss)
        except:
            optimizer.step()

    total_loss /= len(train_dataloader)
    total_acc = accuracy_score(true_labels, predictions)

    return model, optimizer, total_loss, total_acc


def val_once(val_dataloader, model):
    total_loss = 0
    total_acc = 0

    model.eval()

    predictions, true_labels = [], []
    for batch in val_dataloader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    total_loss /= len(val_dataloader)
    total_acc = accuracy_score(y_true=true_labels, y_pred=predictions)

    return total_loss, total_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["sst2", "cola", "mrpc"],
        help="The Name of GLUE Dataset.",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        choices=["AdaGrad", "Adam", "DeltaAdaGrad"],
        required=True,
        help="The Name of Optimizer.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The Batch Size.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=10,
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_id = "google-bert/bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    dataset = load_dataset("glue", args.dataset)
    train_dataloader, val_dataloader, _ = data_preprocessing(
        dataset=dataset,
        args=args,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model.to(device)

    if args.optimizer == "AdaGrad":
        optimizer = torch.optim.Adagrad(
            params=model.parameters(),
            lr=args.learning_rate,
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
        )
    elif args.optimizer == "DeltaAdaGrad":
        optimizer = DeltaAdaGrad(
            params=model.parameters(),
            lr=args.learning_rate,
        )
    else:
        raise ValueError(f"The optimizer {args.optimizer} is not supported.")

    min_lr = 0.00001
    scheduler = StepLR(optimizer=optimizer, step_size=3, gamma=0.1)

    best_epoch = -1
    best_loss = float("inf")
    best_val_acc = 0
    all_res = []

    for epoch in tqdm(range(args.epoch)):
        model, optimizer, train_loss, train_acc = train_once(
            train_dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
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
            f"Epoch: {epoch}, TLoss: {train_loss}, VLoss: {val_loss}, TAcc: {train_acc}, VAcc: {val_acc}"
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr}")
        if current_lr < min_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = min_lr

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Loss: {best_loss}")
    print(f"Best VAcc: {best_val_acc}")

    epochs = [x[0] for x in all_res]
    train_losses = [x[1] for x in all_res]
    val_losses = [x[2] for x in all_res]

    if not os.path.exists(f"outputs/GLUE/{args.dataset}"):
        os.makedirs(f"outputs/GLUE/{args.dataset}")

    plt.plot(epochs, train_losses, "r", label="Train Loss")
    plt.plot(epochs, val_losses, "b", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(
        f"outputs/GLUE/{args.dataset}/{args.optimizer}_loss_epoch_{args.epoch}_lr_{args.learning_rate}.png"
    )
    plt.clf()

    train_accs = [x[3] for x in all_res]
    val_accs = [x[4] for x in all_res]

    plt.plot(epochs, train_accs, "r", label="Train Accuracy")
    plt.plot(epochs, val_accs, "b", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(
        f"outputs/GLUE/{args.dataset}/{args.optimizer}_acc_epoch_{args.epoch}_lr_{args.learning_rate}.png"
    )
    plt.clf()
