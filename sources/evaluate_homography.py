import torch
import numpy as np
import matplotlib.pyplot as plt
from models import OriginalHomography
import cv2

# Diana Maxima Drzikova

img1 = cv2.imread("../datasets/evaluation_pipeline/dataset3_long_g0001_t0000_s00010.tif")
img2 = cv2.imread("../datasets/evaluation_pipeline/dataset3_long_g0001_t0001_s00010.tif")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#img1_0 = cv2.resize(img1, (128, 128))
#img2_0 = cv2.resize(img2, (128, 128))

img1 = cv2.resize(img1, (128, 128))
img2 = cv2.resize(img2, (128, 128))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

state = torch.load("../pretrained_models/homography.pth")

model = OriginalHomography()

model = model.to(device)

inpt = [np.array(img1),np.array(img2)]
input = torch.tensor(inpt).type(torch.FloatTensor)
input = input.unsqueeze(0)
input = input.to(device)

output = model(input)

print(output)
output = output.detach().cpu().numpy()

try:
    result = cv2.warpPerspective(img2, np.linalg.inv(output[0]), (400,400)) 

    result[0:img2.shape[0], 0:img2.shape[1]] = img2
except:
    print("could not warp the image.")
    plt.show()

plt.imshow(result)
plt.title("Output")
plt.show()


