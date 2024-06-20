# Set the path.
# -----
import sys

sys.path.append(".")
# -----

import argparse
import torch
import matplotlib.pyplot as plt
import random
import os
import numpy as np

from tqdm import tqdm
from DeltaAdaGrad import DeltaAdaGrad
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score

from dataloader import load_data
from model import Cnn
from DeltaAdaGrad import DeltaAdaGrad


def random_seed(seed: int = 48763):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(device, model, dataloader_train, dataloader_eval, optimizer, criterion, n_epochs, scheduler=None):

    total_train_loss =[]
    total_eval_loss = []

    total_train_accuracy = []
    total_eval_accuracy = []
    n_epochs = args.epoch
    best_epoch = -1
    best_loss = float("inf")
    best_val_acc = 0


    model.train()
    for epoch in tqdm(range(n_epochs)):
        pred_y_train = []
        label_train = []

        pred_y_eval = []
        label_eval = []

        train_loss_list = []
        eval_loss_list = []

        for data, label in dataloader_train:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            
            try:
                optimizer.step(loss=loss)
            except:
                optimizer.step()

            pred_y = output.argmax(dim=1)
            pred_y_train+=pred_y.to('cpu').tolist()
            label_train+=label.to('cpu').tolist()
            train_loss_list.append(loss.item())

        total_train_accuracy.append(accuracy_score(label_train, pred_y_train))
        total_train_loss.append(sum(train_loss_list)/len(train_loss_list))


        with torch.no_grad():

            for data, label in dataloader_eval:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)

                pred_y = output.argmax(dim=1)
                pred_y_eval+=pred_y.to('cpu').tolist()
                label_eval+=label.to('cpu').tolist()
                eval_loss_list.append(loss.item())
            
            total_eval_accuracy.append(accuracy_score(label_eval, pred_y_eval))
            total_eval_loss.append(sum(eval_loss_list)/len(eval_loss_list))


        if total_eval_accuracy[epoch] > best_val_acc:
            best_epoch = epoch
            best_loss = total_eval_loss[epoch]
            best_val_acc = total_eval_accuracy[epoch]

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch: {epoch}, Tloss: {total_train_loss[epoch]}, TAcc:{total_train_accuracy[epoch]}, Vloss: {total_eval_loss[epoch]}, VAcc:{total_eval_accuracy[epoch]}")
    
    print(f"Best epoch: {best_epoch}, Best loss: {best_loss}, Best Accuracy: {best_val_acc}")

    return total_train_loss, total_eval_loss, total_train_accuracy, total_eval_accuracy



if __name__ == "__main__":
    device = torch.device('cpu')
    # Check if the device is available
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if torch.backends.mps.is_available():
        device = torch.device('mps')   

    print("Using device:",device)

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
        "-b",
        "--batch_size",
        type=int,
        default=100,
        help="The Batch Size.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=100,
        help="The Number of Epoch.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=5e-3,
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

    model = Cnn().to(device)
    
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

    criterion = torch.nn.CrossEntropyLoss()
    dataloader_train, dataloader_eval = load_data("tests/Cat_Dog_Panda/animals", args.batch_size)  
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # start model training 
    total_train_loss, total_eval_loss, total_train_accuracy, total_eval_accuracy = train(device, model, dataloader_train, dataloader_eval, optimizer, criterion, args.epoch, scheduler=None)

    plt.plot(total_train_loss, label='train_loss')
    plt.plot(total_eval_loss, label='eval_loss')    
    plt.title(f"Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'outputs/Cat_Dog_Panda/{args.optimizer}_loss_epoch_{args.epoch}_lr_{args.learning_rate}.png')
    plt.close()

    plt.plot(total_train_accuracy, label='train_accuracy')
    plt.plot(total_eval_accuracy, label='eval_accuracy')
    plt.title(f"Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'outputs/Cat_Dog_Panda/{args.optimizer}_acc_epoch_{args.epoch}_lr_{args.learning_rate}.png')
    plt.close()
