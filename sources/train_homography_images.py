import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from models import Autoencoder, OriginalHomography
from dataAugmentation import run
import argparse
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from ignite.metrics import Accuracy, Recall, Precision
from ignite.engine import (Engine, Events)
from alive_progress import alive_bar
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torch.autograd import profiler
import time
import zipfile
import tarfile
from tqdm import tqdm
import csv
import pandas as pd
from homography_labels import validate_pair, chceck_for_trasparent_bg

# Diana Maxima Drzikova

def createdataset(data, df, batch_size):

    lenght = len(df)

    x_train = data[:int(lenght*0.8)]
    x_val = data[int(lenght*0.8):]

    x_train = torch.from_numpy(np.array(x_train)).type(torch.FloatTensor)
    x_val = torch.from_numpy(np.array(x_val)).type(torch.FloatTensor)

    y_train = df[:int(lenght*0.8)].values
    y_val = df[int(lenght*0.8):].values

    y_train = torch.stack([torch.from_numpy(np.array(arr)) for arr in y_train]).type(torch.FloatTensor)
    y_val = torch.stack([torch.from_numpy(np.array(arr)) for arr in y_val]).type(torch.FloatTensor)

    print(x_val.shape, y_val.shape, y_train.shape, x_train.shape)

    train_dataset0 = TensorDataset(x_train, y_train)
    train_dataloader0 = DataLoader(train_dataset0, batch_size=batch_size)
    val_dataset0 = TensorDataset(x_val, y_val)
    val_dataloader0 = DataLoader(val_dataset0, batch_size=batch_size)

    return train_dataloader0, val_dataloader0


def train(data, pairs, df):

    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = OriginalHomography()
    model = model.to(device)
    epochs = 1000
    criterion = nn.MSELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 64

    train_loader, validation_loader = createdataset(data, df, batch_size)

    train_loss = []
    val_loss = []

    with tqdm(total=epochs*(len(train_loader)), leave=False, unit='clip') as pbar:
        for i in range(epochs):
            for batch_data, batch_labels in train_loader:
                model.train()
                optimizer.zero_grad()

                input = batch_data.permute(0,3,1,2).float()
                input.requires_grad = True
                batch_labels.requires_grad = True

                input = batch_data.to(device)

                output = model(input)

                output = output.to('cpu')

                loss0 = criterion(output, batch_labels)

                loss0.backward()
                optimizer.step()

            for batch_data, batch_labels in validation_loader:
                model.eval()
                optimizer.zero_grad()

                input = batch_data.permute(0,3,1,2).float()

                input = batch_data.to(device)

                output = model(input)

                output = output.detach().cpu()

                loss1 = criterion(output, batch_labels)

            train_loss.append(loss0.item())
            val_loss.append(loss1.item())
            pbar.set_postfix(loss0=loss0.item(), loss1=loss1.item())
            pbar.update()
    
    fig, ax = plt.subplots(1,2)

    ax[0].plot(val_loss)
    ax[0].set_title("Loss function Validation")
    ax[1].set_title("Loss function Training")
    ax[1].plot(train_loss)
    plt.tight_layout()
    plt.savefig("lossHM.png")

    torch.save(model.state_dict, "homography.pth")


def pair_images(images, opt):

    pair_data = []

    with alive_bar(len(images)) as bar:
        for i, im in enumerate(images):
            #print("Processing files:", im)

            image0, image1 = Image.open(im[0]).resize([128,128]), Image.open(im[1]).resize([128,128])
            image0, image1 = chceck_for_trasparent_bg(image0), chceck_for_trasparent_bg(image1)
            
            if validate_pair(im[0], im[1]):
                pair_data.append([np.array(image0),np.array(image1)])
            else:
                print(f"Pair {im[0], im[1]} skipped. Not real pair.")

            bar()

    return pair_data


def imagesort(item):
    """ sort images from input file

    Args:
        item (string): name of an image

    Returns:
        int: image number
        int: pair number
    """
    start = item.split('/')[1][3:]
    end = start.split('_')[0]
    end1 = start.split('_')[1][0]
    return int(end), int(end1)

def convert(df):
    df["matrix"] = [eval(df["matrix"][i]) for i in range(len(df['matrix']))]
    return df["matrix"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input', type=str, default='imagestitching.txt',
        help='Path to the list of image for augmentation')

    parser.add_argument(
        '--matrixes', type=str, default='imagestitching_pairs.csv',
        help='Path to the list of image for augmentation')
    
    opt = parser.parse_args()

    df = convert(pd.read_csv(opt.matrixes, delimiter='\t'))

    images = []
    images_name = []

    with open(opt.input, 'r') as f:
        for img in f:
            images.append(img[:-1])

    images.sort(key=imagesort)

    try:
        pairs = np.array(images).reshape(len(images)//2, 2)
    except:
        print("Missing pairs.")
        exit(1)


    print(f"Images loaded. Finall number of pairs: {len(pairs)}.")
    
    data = pair_images(pairs, opt)

    train(data, pairs, df)
