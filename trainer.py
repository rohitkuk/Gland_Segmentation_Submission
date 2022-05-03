
from utils.metrics import Metric
from utils.utils import (
    seeding,
    create_dir,
    prepare_dataset,
    plot,
    combine_img_target_pred
)
from losses.focal_loss import FocalLoss
from losses.diceloss import DiceLoss, DiceBCELoss
from models.unet_plus_plus import UNetPlusPlus
from models.att_unet import AttU_Net
from models.unet import UNet
from data.dataloader import GlasDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import random
import os
import time
import warnings
warnings.filterwarnings("ignore")


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.train_x = sorted(glob("dataset/train/imgs/*"))
        self.valid_x = sorted(glob("dataset/test/imgs/*"))
        self.train_data = GlasDataset(self.train_x)
        self.valid_data = GlasDataset(self.valid_x)

        # Defining the Hyper Parameters
        self.num_workers = 0
        self.batch_size = {"train": 1, "val": 1}
        self.lr = 1e-4
        self.modes = ["train", "val"]
        self.num_epochs = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = DiceBCELoss()

        # Setting up the model
        self.best_loss = float("inf")
        self.net = model
        self.net = self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.dataloaders = {
            "train": DataLoader(dataset=self.train_data, batch_size=self.batch_size['train'], shuffle=False, num_workers=self.num_workers),
            "val": DataLoader(dataset=self.valid_data, batch_size=self.batch_size['val'], shuffle=False, num_workers=self.num_workers)
        }
        # Defining the Containers
        self.losses = {mode: [] for mode in self.modes}
        self.iou_scores = {mode: [] for mode in self.modes}
        self.dice_scores = {mode: [] for mode in self.modes}

    def iterate(self, epoch, mode):
        running_loss = 0.0
        counter = 0
        dataloader = self.dataloaders[mode]
        total_batches = len(dataloader)
        metrics = Metric(mode)

        self.net.train() if mode == "train" else self.net.eval()
        for i, (x, y) in enumerate(dataloader):
            counter += 1
            image, mask = x.float().to(self.device), y.float().to(self.device)
            if mode == "train":
                self.optimizer.zero_grad()
            outputs = self.net(image)
            loss = self.criterion(outputs, mask)
            running_loss += loss.item()
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            metrics.update(mask, outputs)

        epoch_loss = running_loss/counter
        dice, iou = metrics.get_metrics()
        metrics.log(mode, epoch, epoch_loss)

        self.losses[mode].append(epoch_loss)
        self.dice_scores[mode].append(dice)
        self.iou_scores[mode].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss  # for val loss

    def start(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                # self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("############ Improved Model --- Saving.. ############")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, f"./{self.net.name()}_best_model.pth")

        print("############ Training Completed !! ############ ")

    def predict(self, num_imgs):
        for i in range(num_imgs):
            data = self.train_data.__getitem__(i)
            img = data[0].unsqueeze(0).to(device=self.device)
            mask = data[1].squeeze(0)*255.0
            output = self.net(img)
            output = torch.squeeze(output)
            output = torch.sigmoid(output)
            output = output > 0.5
            output = output * 255.0
            rgb = (np.transpose(data[0].detach().cpu() * 255.0, (1, 2, 0)))
            gray = mask.detach().cpu()
            pred = output.detach().cpu()
            comb = combine_img_target_pred(rgb, gray, pred)
            cv2.imwrite(f"{self.net.name()}_Files/combined_{i}.jpeg", comb)

    def create_plots(self):
        plot(self.losses, f"{self.net.name()} BCE loss")
        plot(self.dice_scores, f"{self.net.name()} Dice score")
        plot(self.iou_scores, f"{self.net.name()} IoU score")


if __name__ == "__main__":
    import sys
    seeding(42)
    prepare_dataset(root='dataset')
    if sys.argv[1] == "unet":
        model = UNet()
    elif sys.argv[1] == "att_unet":
        model = AttU_Net()
    elif sys.argv[1] == "unet_plus_plus":
        model = UNetPlusPlus()
    create_dir(model.name()+"_Files")
    model_trainer = Trainer(model)
    model_trainer.start()
    model_trainer.predict(50)
    model_trainer.create_plots()
