import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from models import OriginalHomography
from dataAugmentation import run
import argparse
from tqdm import tqdm
import pandas as pd

# Diana Maxima Drzikova

def train(df):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OriginalHomography()

    model = model.to(device)

    criterion = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 64
    epochs = 10

    print(f"Criterion {criterion}, Optimizer {optimizer}, learning rate {learning_rate}")

    lenght = len(df['matrix'])

    print(lenght*0.8)

    x_train_m0 = df['matches0'][:int(lenght*0.8)]
    x_train_m1 = df['matches1'][:int(lenght*0.8)]

    x_val_m0 = df['matches0'][int(lenght*0.8):]
    x_val_m1 = df['matches1'][int(lenght*0.8):]

    x_val_m0 = x_val_m0.reset_index(drop=True)
    x_val_m1 = x_val_m1.reset_index(drop=True)
    
    y_train = df['matrix'][:int(lenght*0.8)]
    y_val = df['matrix'][int(lenght*0.8):]
    y_val = y_val.reset_index(drop=True)

    print("Number of pairs train:", len(x_train_m0),type(x_train_m0))
    print("Number of pairs val:", len(x_val_m0))

    train_loss = []
    val_loss = []

    with tqdm(total=epochs*(len(x_train_m0)), leave=False, unit='clip') as pbar:
        for i in range(epochs):
            for index, data in enumerate(x_train_m0):
                model.train()
                optimizer.zero_grad()

                mkp0 = x_train_m0[index]
                mkp1 = x_train_m1[index]
                matrix = y_train[index]

                mkp = [mkp0, mkp1]
                input = torch.tensor(np.array(mkp)).type(torch.FloatTensor)

                input = input.unsqueeze(0)
                input.requires_grad = True

                input = input.to(device)

                output = model(input)

                output = output.to('cpu')

                matrix = torch.tensor(np.array(matrix)).type(torch.FloatTensor)
                matrix.requires_grad = True

                #print(matrix.is_cuda, output.is_cuda)

                loss = criterion(output, matrix)

                loss.backward()
                optimizer.step()

                #print(loss.item())


                pbar.set_postfix(loss=loss.item())
                pbar.update()

            with torch.no_grad():
                for index_1, data in enumerate(x_val_m0):
                    optimizer.zero_grad()
                    model.eval()


                    mkp0 = x_val_m0[index_1]
                    mkp1 = x_val_m1[index_1]
                    matrix = y_val[index_1]

                    if len(mkp0) < 30:
                        continue

                    mkp = [mkp0, mkp1]
                    input = torch.tensor(np.array(mkp)).type(torch.FloatTensor)

                    input = input.unsqueeze(0)
                    input.requires_grad = True

                    input = input.to(device)

                    output = model(input)

                    output = output.to('cpu')

                    matrix = torch.tensor(np.array(matrix)).type(torch.FloatTensor)
                    matrix.requires_grad = True

                    loss1 = criterion(output, matrix)

            val_loss.append(loss1.item())
            train_loss.append(loss.item())



    fig, ax = plt.subplots(1,2)

    ax[0].plot(val_loss)
    ax[0].set_title("Loss function Validation")
    ax[1].set_title("Loss function Training")
    ax[1].plot(train_loss)
    plt.tight_layout()
    plt.savefig("lossHM.png")


    torch.save(model.state_dict, "homography.pth")

    
    pass

def convert(df):
    df["matrix"] = [eval(df["matrix"][i]) for i in range(len(df['matrix']))]
    df["matches0"] = [eval(df["matches0"][i]) for i in range(len(df['matches0']))]
    df["matches1"] = [eval(df["matches1"][i]) for i in range(len(df['matches1']))]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input', type=str, default='imagestitching_pairs.csv',
        help='Path to the list of image for augmentation')
    
    opt = parser.parse_args()

    df = convert(pd.read_csv(opt.input, delimiter='\t'))

    #for i in df["matches0"]:
    #print(len(i))

    train(df)


