import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder
from superpoint import SuperPoint
import argparse
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Diana Maxima Drzikova

# function from SuperGlue implementation
def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def evaluate(opt):
    torch.manual_seed(42)
    np.random.seed(42)
    
    image_input = Image.open("../datasets/keypointevaluate/square.png").convert('L').resize([400,400])
    image_input_1 = Image.open("../datasets/keypointevaluate/dataset3_long_g0001_t0000_s00010.tif").convert('L').resize([600,400])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    superpointimg = frame2tensor(np.array(image_input), device)
    superpointimg1 = frame2tensor(np.array(image_input_1), device)

    ########### autoencoder ##########


    model = Autoencoder().to(device)
    state = torch.load("../pretrained_models/autoencoder150k.pth") 
    model.load_state_dict(state)

    model.eval()

    img = np.array(image_input).astype("float16")
    img1 = np.array(image_input_1).astype("float16")

    image = torch.from_numpy(img/255).type(torch.FloatTensor).to(device)
    image1 = torch.from_numpy(img1/255).type(torch.FloatTensor).to(device)

    image, image1 = image.unsqueeze(0), image1.unsqueeze(0)

    output, features = model(image)
    output1, features1 = model(image1)

    output, image, features = output.to('cpu'), image.to('cpu'), features.to('cpu')
    output1, image1, features1 = output1.to('cpu'), image1.to('cpu'), features1.to('cpu')

    output = output.squeeze(0).detach().cpu().numpy()
    output1 = output1.squeeze(0).detach().cpu().numpy()
    features = features.detach().cpu()
    features1 = features1.detach().cpu()

    upsampled_features = nn.functional.interpolate(features.unsqueeze(0), size=[400,400], mode='bilinear', align_corners=True)
    new_t_output = torch.where(upsampled_features > 4.5, torch.ones_like(upsampled_features), torch.zeros_like(upsampled_features)).cpu().numpy()

    upsampled_features1 = nn.functional.interpolate(features1.unsqueeze(0), size=[400,600], mode='bilinear', align_corners=True)
    new_t_output1 = torch.where(upsampled_features1 > 4.5, torch.ones_like(upsampled_features1), torch.zeros_like(upsampled_features1)).cpu().numpy()


    ########### superpoint ##########

    model1 = SuperPoint().to(device)
    model1.eval()

    out = model1({"image": superpointimg})
    out1 = model1({"image": superpointimg1})

    scores = out["fullscores"].detach().cpu()#.numpy()
    new_scores = torch.where(scores > 0.005, torch.ones_like(scores), torch.zeros_like(scores)).cpu().numpy()

    scores1 = out1["fullscores"].detach().cpu()#.numpy()
    new_scores1 = torch.where(scores1 > 0.005, torch.ones_like(scores1), torch.zeros_like(scores1)).cpu().numpy()

    ############# image plot ############

    mse = torch.mean((torch.from_numpy(output) - torch.from_numpy(np.array(image_input))) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    print("PSNR square:", psnr)

    mse = torch.mean((torch.from_numpy(output1) - torch.from_numpy(np.array(image_input_1))) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    print("PSNR microscopic:", psnr)

    #from sklearn.metrics import average_precision_score
    #avg_precision = average_precision_score(new_t[0].ravel(), combined_features.ravel())
    #print(avg_precision)

    fig, ax = plt.subplots(2,2, figsize = (10,8), gridspec_kw={'height_ratios': [1,2]})

    ax[0][0].imshow(image_input, cmap='gray')
    ax[0][1].imshow(image_input, cmap='gray')

    ax[1][0].imshow(image_input_1, cmap='gray')
    ax[1][1].imshow(image_input_1, cmap='gray')

    x,y = [],[]
    x1,y1 = [],[]

    x2,y2 = [],[]
    x3,y3 = [],[]

    for i in range(len(img)):
        for j in range(len(img)):
            if new_t_output[0][0][j][i]:
                x.append(j)
                y.append(i)
            if new_scores[0][j][i]:
                x2.append(j)
                y2.append(i)

    print(img1.shape)
    print(new_t_output1.shape)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if new_t_output1[0][0][i][j]:
                x1.append(j)
                y1.append(i)
            if new_scores1[0][i][j]:
                x3.append(j)
                y3.append(i)
            
    ax[0][0].scatter(x,y, color="green", s=10)
    ax[0][0].set_title("Neural Network", fontsize=20)
    ax[0][1].scatter(x2,y2, color="green", s=30)
    ax[0][1].set_title("SuperPoint", fontsize=20)

    ax[1][0].scatter(x1,y1, color="green", s=3)
    ax[1][0].set_title("Neural Network", fontsize=20)
    ax[1][1].scatter(x3,y3, color="green", s=3)
    ax[1][1].set_title("Superpoint", fontsize=20)

    plt.tight_layout()
    for i in range(2):
        for j in range(2):
            ax[i][j].axis('off')
            ax[i][j].axis('tight')
            ax[i][j].axis('image')

    plt.show()

    ################CLUSTERS##############

    clusters_num = opt.clusters

    kmeans = KMeans(n_clusters=clusters_num)
    clusters = kmeans.fit_predict(output)
    clusters1 = kmeans.fit_predict(output1)


    # Compute the overall silhouette score
    silhouette_avg = silhouette_score(output, clusters)
    silhouette_avg1 = silhouette_score(output1, clusters1)

    ch = calinski_harabasz_score(output, clusters)
    ch1 = calinski_harabasz_score(output1, clusters1)

    print("Silhouette Score Square:", silhouette_avg)
    print("Calinski-Harabasz Score Square:", ch)
    print("Silhouette Score Microscopic:", silhouette_avg1)
    print("Calinski-Harabasz Score Microscopic:", ch1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--clusters', type=int, default=10,
        help='Number of clusters for Silhouette Score and Calinski-Harabasz Score.')
    
    opt = parser.parse_args()

    evaluate(opt)
    
