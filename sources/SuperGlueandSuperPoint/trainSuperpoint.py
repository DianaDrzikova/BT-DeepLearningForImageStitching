import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from superpoint import SuperPoint
import argparse
import cv2
import numpy as np
from autoencoder import Autoencoder
import matplotlib.pyplot as plt
import tarfile
from tqdm import tqdm

# Diana Maxima Drzikova

config = {
    'superpoint': {
        'nms_radius': 5,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048
    },
}
class EarlyStop:
    def __init__(self, patience=5, min_diff=1e-04):
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

def getLables(image_paths, model, device):

    keypoints_list = []
    descriptors_list = []
    score_list = []

    autoencoder = Autoencoder()
    state = torch.load("weights/autoencoder150k.pth")
    autoencoder.state_dict(state)
    autoencoder.eval()
    autoencoder = autoencoder.to(device)
    cnt = 0
    for image in image_paths:

        image = torch.tensor(image).type(torch.FloatTensor)
        image = image.to(device)
        image = image.unsqueeze(0).unsqueeze(0)

        output, features = autoencoder(image)

        features = features.detach().cpu()
        output = output.detach().cpu().numpy()
        image = image.to('cpu')       

        upsampled_features1 = nn.functional.interpolate(features, size=[64,64], mode='bilinear', align_corners=True)
        new_t_output1 = torch.where(upsampled_features1[0] > 4.5, torch.ones_like(upsampled_features1[0]), torch.zeros_like(upsampled_features1[0])).cpu().numpy()

        score_list.append(output/255)
        cnt+=1

        if cnt == 20000:
            break

    autoencoder = autoencoder.to('cpu')
    
    # Save the keypoints and descriptors to an npz file
    #np.savez('score.npz', score=score_list)

    return keypoints_list, score_list, descriptors_list

def retrain():
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    archive = tarfile.open('../../datasets/datasetC/images.tar.gz', 'r')
    images = []
    # loading images
    for member in archive.getmembers():
        if member.isfile():
            f = archive.extractfile(member)
            image_data = f.read()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            images.append(image)   


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Running inference on device \"{device}\"')

    model = SuperPoint(config.get('superpoint', {})).to(device)

    model.load_state_dict(torch.load('weights/superpoint_v1.pth'))

    loss_fn = nn.MSELoss()

    early_stopper = EarlyStop()

    # Define the optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    keypoints, scores, descriptors = getLables(images, model, device)

    print(len(images[:int(len(scores)*0.8)]), len(scores[:int(len(scores)*0.8)]))

    images_train = torch.tensor(np.array(images[:int(len(scores)*0.8)])).float()
    images_train = torch.unsqueeze(images_train, 1)  # add new dimension at index 1
    scores_train = torch.tensor(np.array(scores[:int(len(scores)*0.8)]))

    images_val = torch.tensor(np.array(images[int(len(scores)*0.8):len(scores)])).float()
    images_val = torch.unsqueeze(images_val, 1)  # add new dimension at index 1
    scores_val = torch.tensor(np.array(scores[int(len(scores)*0.8):]))


    print(images_val.shape ,scores_val.shape)

    train_dataset = TensorDataset(images_train, scores_train)
    val_dataset = TensorDataset(images_val, scores_val)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=64)

    for name, param in model.named_parameters():
        if "Pb" in name or "Pa" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print(len(train_loader),len(scores))
    array = []
    array1 = []
    loss_sum = 0
#
    with tqdm(total=5*len(train_loader), leave=False, unit='clip') as pbar:
        for epoch in range(5):
            print('Epoch', epoch + 1)

            model.train()
            cnt = 0
            loss_sum = 0
            loss_sum0 = 0
            for i, (image, scores) in enumerate(train_loader):
                optimizer.zero_grad()

                image = image.to(device)
                image.requires_grad = True

                output = model({'image': image})

                output = {k: v[0].cpu() for k, v in output.items()}

                scores_pred = output["fullscores"].detach().cpu().numpy()

                scores_pred = torch.tensor(scores_pred)

                scores_pred.requires_grad = True

                scores.requires_grad = True

                loss = loss_fn(scores_pred, scores)

                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                loss_sum0 = 0
                
                for j, (image, score) in enumerate(val_loader):
                    model.eval()

                    #image.requires_grad = True
                    image = image.to(device)

                    output = model({'image': image})

                    scores_pred = output["fullscores"].detach().cpu().numpy()
                    scores_pred = torch.tensor(scores_pred)

                    loss0 = loss_fn(scores_pred, score)
                    loss_sum0 += loss0.item()

                if (cnt + 1) % 100 == 0:
                    print('Iteration', cnt + 1, 'loss', loss0.item())
                    array.append(loss.item())
                    array1.append(loss0.item())

                cnt += 1

                pbar.update()

                stop = False
                print(loss_sum0/(len(val_loader)) , loss_sum/(i+1))
                if early_stopper.early_stop(loss_sum0/(len(val_loader))):   
                    print(f"Early stop on {epoch}.")  
                    stop = True
                    break
            if stop:
                break

    torch.save(model.state_dict(), 'weights/superpoint_v1_new.pth')

    fig, ax = plt.subplots(2,1)

    ax[0].plot(array)
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation loss")
    ax[1].plot(array1)
    ax[-1].set_xlabel("Samples")
    plt.tight_layout()
    plt.savefig("train.png")

    

retrain()
