import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder
import argparse
import cv2
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import profiler
import tarfile
import datetime
from tqdm import tqdm

# Diana Maxima Drzikova

class EarlyStop:
    def __init__(self, patience=1, min_diff=0):
        self.min_val_loss = np.inf
        self.patience = patience
        self.min_diff = min_diff
        self.cnt = 0

    def early_stop(self, loss):

        if loss < self.min_val_loss:
            self.min_val_loss = loss
            self.cnt = 0
        elif loss > (self.min_val_loss + self.min_diff):
            self.cnt += 1
            if self.cnt >= self.patience:
                return True
        return False

def train(opt):
    """unsupervised training on detecting features on microscopic images
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    archive = tarfile.open('images.tar.gz', 'r') #5345

    model = Autoencoder().to(device)
    print("Model loaded on device:", device)

    criterion = nn.MSELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Criterion {criterion}, Optimizer {optimizer}, learning rate {learning_rate}")


    images = []
    # loading images
    for member in archive.getmembers():
        if member.isfile():
            f = archive.extractfile(member)
            image_data = f.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            images.append(image)   

    print(f"Loaded {len(images)} images.")
    leng = len(images) - 1

    train_index = int(leng*0.8) 
    val_index = int(leng*0.2) 

    x_train = images[:train_index]
    x_val = images[train_index:train_index+val_index]

    x_train = torch.from_numpy(np.array(x_train).astype('float16')/255).type(torch.FloatTensor)
    x_val = torch.from_numpy(np.array(x_val).astype('float16')/255).type(torch.FloatTensor)

    x_train = torch.tensor(np.array(x_train)).type(torch.FloatTensor)
    x_val = torch.tensor(np.array(x_val)).type(torch.FloatTensor)

    train_dataset = TensorDataset(x_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    val_dataset = TensorDataset(x_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    loss_train = []
    loss_val = []

    early_stopper = EarlyStop()
    epoch = 30

    with tqdm(total=epoch*(len(train_dataloader)), leave=False, unit='clip') as pbar:
        for j in range(epoch):
            print(f"Start of epoch{j}")
            for i, data in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()

                input = torch.cat(data, dim=0).to(device)

                input = input.unsqueeze(1)

                input.requires_grad = True

                output,_ = model(input)

                output, input = output.to('cpu'), input.to('cpu')

                loss = criterion(output, input)

                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    print(f"epoch:{j} sample: {i} Loss item", loss.item())

                pbar.update()
                
            loss_train.append(loss.item())


            print(f"End of epoch{j}")

            with torch.no_grad():
                for k, data in enumerate(val_dataloader):
                    model.eval()

                    input = torch.cat(data, dim=0).to(device)
                    input = input.unsqueeze(1)

                    output, _ = model(input)

                    output, input = output.to('cpu'), input.to('cpu')

                    loss = criterion(output, input)

                    if k % 1000 == 0:
                        print(f"epoch:{k} sample: {i} Loss item", loss.item())

            loss_val.append(loss.item())

            if early_stopper.early_stop(loss.item()):   
                print(f"Early stop on {epoch}.")  
                break



    time = datetime.datetime.now()

    fix, ax = plt.subplots(2,1)


    ax[0].set_title("Training loss")
    ax[0].plot(loss_train)
    ax[1].set_title("Validation loss")
    ax[1].plot(loss_val)
    ax[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(f"loss{time}.png")

    torch.save(model.state_dict(),f"autoencoder__{time}.pth")

    archive.close()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    opt = parser.parse_args()

    train(opt)
    
