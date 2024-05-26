import skimage
from skimage import color
from PIL import Image, ImageFilter, ImageEnhance
from scipy import misc
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from scipy.ndimage import zoom
import math
import fire
from alive_progress import alive_bar
import argparse
import math
import torch
from crop import anglecut, crop
import tarfile
import os
import io
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from superpoint import SuperPoint
from superglue import SuperGlue

# Diana Maxima Drzikova

def savekeypoints(path, img, keypoints, data, scores):
    """ packing up the data

    Args:
        path (str): file destination
        img (list): all images names
        keypoints (list): all keypoints
        data (list): all images
        scores (list): all scores

    Returns:
        dictionary: grouped data
    """

    out_matches = {'image': img, 'keypoints': keypoints, 'data': data, 'scores': scores}
    
    return out_matches

def createLabels(img):

    """ create labels using SuperPoint neural network
    Args:
        img (array): input image
    Returns:
        array: keypoints of image
        array: descriptors of image
    """

    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SuperPoint()
    model_state = torch.load('superpoint_v1.pth')

    # Set the model to evaluation mode
    model.state_dict(model_state)
    model.eval()

    imgtmp = Image.fromarray(img.astype(np.uint8)).resize((400, 600))

    input = torch.from_numpy(np.array(imgtmp)/255.).float()[None, None].to(device)

    model = model.to(device)
    output = model(input)

    keypoints = [tensor.detach().cpu() for tensor in output['keypoints']]
    descriptors = [tensor.detach().cpu() for tensor in output['descriptors']]

    return keypoints, descriptors


def augment(img, i, index, blr, brg, rot):
    """ image augmenting

    Args:
        img (image): image to be augmented
        i (int): number of image
        index (int): position in pair
        blr (bool): apply blur
        brg (bool): apply brightness
        rot (bool): apply rotation

    Returns:
        array: augmented image
        array: keypoints
        array: scores
    """

    new_width  = 400
    new_height = 600

    import time
    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    random.seed(int(t) % 2**32)

    orig_img = np.array(img).astype(np.uint8)

    #keypoints,scores = createLabels(img)
    
    # gaussblur
    if blr:
        img = Image.fromarray(img.astype(np.uint8))
        blur = random.randint(5,15)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur)) 
        print("GaussianBlur radius:", blur)
        img = np.array(img)
        noisy_img = img

        mean_original = np.mean(orig_img)
        std_original = np.std(orig_img)
        mean_noisy = np.mean(noisy_img)
        std_noisy = np.std(noisy_img)
        power_signal = std_original ** 2
        power_noise = (std_noisy ** 2) - (std_original ** 2)
        snr = power_signal / power_noise

        print('SNR:', snr)

        # Calculate PSNR
        max_pixel_value = 255
        mse = np.mean((orig_img - noisy_img) ** 2)
        psnr = 10 * np.log10(max_pixel_value ** 2 / mse)

        print('PSNR:', psnr)
        print(f"Saving img{i}_{index}.png")
    
    # brightness
    if brg:
        img = Image.fromarray(img.astype(np.uint8))
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.5,2)
        img = enhancer.enhance(factor)
        img = np.array(img)
        print("Brightness enhance:", str(round(factor,3))+"x")
    
    # rotation
    if rot:
        angle = random.randint(-12,12)

        if angle != 0:
            img = Image.fromarray(img.astype(np.uint8))
            
            imgtmp = np.array(img)
            
            image_rotated = img.rotate(angle, expand=True)

            yp, xp = np.array(image_rotated).shape[0], np.array(image_rotated).shape[1]

            y, x = np.array(image_rotated).shape[0], np.array(image_rotated).shape[1]
            image_rotated = np.array(image_rotated)

            #left, top, right, bottom = crop(image_rotated, x - 1, y - 1, angle)
            left, top, right, bottom = anglecut(angle, x - 1, y - 1, xp, yp)

            img = np.array(image_rotated)

            if left and top and right and bottom:
                print("Rotating angle:", angle, image_rotated.shape)
                img = image_rotated[top:bottom,left:right]
            #else:
                #print("Without rotation, angle:", angle)


    img = Image.fromarray(img.astype(np.uint8)).resize((400, 600)) 

    image.imsave(f"augmented/img{i}_{index}.png", img, cmap="gray")

    return [], np.array(img), [0]

def balance(input, keypoint, scores):
    """ creating a zip file with images for keypoint detection training

    Args:
        input (list): all images
        keypoint (array): keypoints of images
        scores (array): scores of images
    """

    parsed = []

    for a in range(len(input)):
        for b in range(len(input[a])):
            inpt = input[a][b]
            kpt = keypoint[a][b][0]
    
            x,y = [],[]

            for i in range(len(kpt)):
                x.append(kpt[i][0].item()) # index 0 is x
                y.append(kpt[i][1].item()) # index 1 is y

            for i in range(len(kpt)):

                y1 = int(y[i])-32
                y2 = int(y[i])+32
                x1 = int(x[i])-32
                x2 = int(x[i])+32

                if y1 < 0:
                    y1 = 0
                if y2 > inpt.shape[0]:
                    y2 = inpt.shape[0] - 1

                if x1 < 0:
                    x1 = 0
                if x2 > inpt.shape[1]:
                    x2 = inpt.shape[1] - 1

                newimg = inpt[y1:y2,x1:x2]

                newimg = Image.fromarray(np.array(newimg).astype(np.uint8))

                arr = np.array(newimg)/255

                binary_arr = np.zeros_like(arr)
                binary_arr[arr > 0.2] = 1

                light = np.sum(binary_arr)

                if np.array(newimg).shape[0] < 64 or np.array(newimg).shape[1] < 64:
                    newimg = Image.fromarray(np.array(newimg).astype(np.uint8)).resize((64, 64))
                    newimg = np.array(newimg)


                if light/(64*64) <= 0.2 or light/(64*64) >= 0.6:
                    continue

                parsed.append(newimg)

    # Write the image data to a zip file
    with tarfile.open('images_paris_new.tar.gz', 'w:gz') as tar:
        for i, data in enumerate(parsed):
            img = np.array(data).astype(np.uint8)
            # Add the image to the tar file
            tarinfo = tarfile.TarInfo(f'image_{i}.png')
            img_bytes = cv2.imencode('.png', img)[1].tobytes()
            tarinfo.size = len(img_bytes)
            tar.addfile(tarinfo, fileobj=BytesIO(img_bytes))
    
        print(f"Successfully created images_paris_new.tar.gz, {len(parsed)} images")

    pass



def inHalf(img, i, blr, brg, rot, num, split):
    """ process the images

    Args:
        img (list): all images
        i (int): image number
        blr (bool): apply blur
        brg (bool): apply brightness
        rot (bool): apply rotation
        num (int): number of similar pairs
        split (bool): for creating image stitching dataset

    Returns:
        list: image names
        list: all images
        list: keypoints
        list: scores
    """
    random.seed(42)

    data = np.asarray(img)
    index = 0

    image_names = []
    keypoints = []
    data_images = []
    scores = []

    # split parameter creates synthetic pairs of given image
    if split:
        height, width = data.shape

        width_cutoff = width // 2
        padd_right = random.randint(width//6,width//4)
        padd_left = random.randint(width//6,width//4)

        print(padd_right - padd_left)

        data = np.transpose(data)

        s1 = data[:width_cutoff+padd_right]
        s2 = data[width_cutoff-padd_left:]

        s1 = np.transpose(s1)
        s2 = np.transpose(s2)

        halves = [s1,s2]

        for j in range(num):
            for k in range(len(halves)):

                output = augment(halves[k], i, index, blr, brg, rot)
                
                keypoints.append(output[0])
                data_images.append(output[1])
                image_names.append(f"imagestitchingdataset/img{i}_{index}")
                scores.append(output[2])
                index += 1

    else: # simple data augmentation 
        for j in range(num):
            output = augment(data, j, index, blr, brg, rot)
            keypoints.append(output[0])
            data_images.append(output[1])
            image_names.append(f"imagestitchingdataset/img{i}_{index}")
            scores.append(output[2])

    return image_names, data_images, keypoints, scores

def run(images, opt):

    blr = opt.blur
    brg = opt.brightness
    rot = opt.rotation
    num = opt.num
    split = opt.split
    outputdir = opt.output

    image_names = []
    data_images = []
    keypoints = []
    scores = []

    with alive_bar(len(images)) as bar:
        for i, im in enumerate(images):
            print("File:", im)

            img = Image.open(im)#.convert('LA')

            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                alpha = img.convert('RGBA').split()[-1]
                img = img.convert('L').crop(alpha.getbbox())
            else:
                img = img.convert('L')

            
            IM, DI, KP, SC = inHalf(img, i, blr, brg, rot, num, split)
            image_names.append(IM)
            data_images.append(DI)
            keypoints.append(KP)
            scores.append(SC)
            bar()
    
    image_names = np.array(image_names).flatten()

    data_images = np.array(data_images, dtype="object")

    keypoints = np.array(keypoints, dtype="object")
    
    return savekeypoints(outputdir, image_names, keypoints, data_images, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input', type=str, default='dataset.txt',
        help='Path to the list of image for augmentation')
    
    parser.add_argument(
        '--output', type=str, default='keypoints.npz',
        help='Path to the file with saved keypoints')
    
    
    parser.add_argument(
        '--blur', action='store_true',
        help='Add blur to image')

    parser.add_argument(
        '--brightness', action='store_true',
        help='Change brightness in image')
    
    parser.add_argument(
        '--rotation', action='store_true',
        help='Change image rotation')

    parser.add_argument(
        '--num', type=int, default=1,
        help='Number of sets for each image. Default 1')

    parser.add_argument(
        '--split', action='store_true',
        help='Determining whether dataset will be used for image stitching.')
    
    
    opt = parser.parse_args()
    images = []

    with open(opt.input, 'r') as f:
        for img in f:
            images.append(img[:-1])
    
    print("Images loaded.")

    run(images, opt)