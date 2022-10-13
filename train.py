import mlflow.pytorch
import logging
import os
import torch
import torchvision
from argparse import ArgumentParser
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from torchvision import datasets, transforms
import mlflow

class Classifer(nn.Module):
    def __init__(self):
        super(Classifer, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        return self.head(x)

def MnistDataset(Dataset):
    def __init__(self, df):
        super(MnistDataset, self).__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__():
        pass

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Mnist Example")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_size", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    params = {
        "lr": args.lr,
        "data_size": args.data_size,
        "img_size": 256,
        "batch_size": args.batch_size,
    }

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    

    model = Classifer()
    optimizer =  torch.optim.SGD(model.parameters(), params["lr"], momentum=0.0)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    
    df_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=transform
        )
    df_train, df_val = random_split(df_train, [params["data_size"], len(df_train) - params["data_size"]])

    dl_train = DataLoader(df_train, batch_size=params["batch_size"])
    dl_val = DataLoader(df_val, batch_size=params["batch_size"])

    mlflow.start_run(run_name="BaseModel")
    # mlflow.pytorch.autolog()
    lossfn = nn.CrossEntropyLoss()
    for e in range(3):

        model.train()
        accs = []
        for imgs, labels in tqdm(dl_train):
            bs = imgs.shape[0]
            imgs = imgs.view(bs, -1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = lossfn(logits, labels)
            loss.backward()
            optimizer.step()
            output = torch.softmax(logits, 1)
            _, y_hat = torch.max(output, dim=1)
            acc = (y_hat.cpu() == labels.cpu()).float().mean()
            accs.append(acc)
        
        mlflow.log_metric(f"ACC", np.mean(accs))
        print(np.mean(accs))

        with open("train.log", "a") as f:
            f.write(f"Epoch {e} Acc={np.mean(accs):.4f}\n")

    mlflow.log_artifact(local_path="train.log")
    del params["data_size"]
    mlflow.log_params(params)
    # mlflow.pytorch.save_state_dict(model.state_dict(), ".")
    mlflow.pytorch.log_model(model, "model")
