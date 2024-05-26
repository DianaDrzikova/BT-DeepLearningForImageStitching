import skimage
from skimage import color
from PIL import Image, ImageFilter, ImageEnhance
from scipy import misc
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import zoom
from alive_progress import alive_bar
import argparse
import math
import torch
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from superpoint import SuperPoint
from superglue import SuperGlue
import csv

# Diana Maxima Drzikova

def save_matches(matches_path, kpts0, kpts1, matches, conf):
    """ save matches to npz file

    Args:
        matches_path (string): file destination
        kpts0 (array): keypoints of left images for pair
        kpts1 (array): keypoints of right images for pair
        matches (array): matches between images
        conf (array): match confidence
    """
    out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                            'matches': matches, 'match_confidence': conf}
    np.savez(str(matches_path), **out_matches)

def loadmodels():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    supergluemodel = SuperGlue()
    superpointmodel = SuperPoint()

    model_state_sp = torch.load('superpoint_v1.pth')
    model_state_sg = torch.load('superglue_outdoor.pth')

    superpointmodel.state_dict(model_state_sp)
    superpointmodel.eval()

    supergluemodel.state_dict(model_state_sg)
    supergluemodel.eval()


    for param in supergluemodel.parameters():
        param.grad = None
    for param in superpointmodel.parameters():
        param.grad = None

    return superpointmodel, supergluemodel


def generate_descriptors(image0, image1):
    """ generating keypoints, descriptors and matches with SuperPoint and SuperGlue

    Args:
        image0 (arary): left image in the pair
        image1 (arary): right image in the pair
    
    Returns:
        matrix: homography matrix
    """
    superpointmodel, supergluemodel = loadmodels()

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image0, cmap='gray')
    ax[1].imshow(image1, cmap='gray')

    ax[0].axis('off')
    ax[1].axis('off')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    supergluemodel, superpointmodel = supergluemodel.to(device) ,superpointmodel.to(device)

    image0 = np.asarray(image0, dtype=np.uint8)
    image1 = np.asarray(image1, dtype=np.uint8)

    imageprint0, imageprint1 = image0, image1

    image0 = torch.from_numpy(np.array(image0)/255.).float()[None, None].to(device)
    image1 = torch.from_numpy(np.array(image1)/255.).float()[None, None].to(device)

    output_sp_0 = superpointmodel({'image': image0})
    output_sp_1 = superpointmodel({'image': image1})

    output_sp_0_out = {k: v[0].detach().cpu().numpy() for k, v in output_sp_0.items()}
    output_sp_1_out = {k: v[0].detach().cpu().numpy() for k, v in output_sp_1.items()}


    kpt0, kpt1 = output_sp_0_out["keypoints"], output_sp_1_out["keypoints"]
    desc0, desc1 = output_sp_0_out["descriptors"], output_sp_1_out["descriptors"]

    print("SuperPoint successfull.")

    images_data = {"image0": image0, "image1": image1}

    pred = {}
    pred = {**pred, **{k+'0': v for k, v in output_sp_0.items()}}
    pred = {**pred, **{k+'1': v for k, v in output_sp_1.items()}}

    data = {**images_data, **pred}

    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])

    matches = supergluemodel(data)

    print("SuperGlue successfull.")

    pred = {k: v[0].detach().cpu().numpy() for k, v in matches.items()}
    output_sp_0 = {k: v[0].detach().cpu().numpy() for k, v in output_sp_0.items()}
    output_sp_1 = {k: v[0].detach().cpu().numpy() for k, v in output_sp_1.items()}
    image0, image1 = image0.to('cpu'), image1.to('cpu')

    mtch0 = pred["matches0"]
    mtch1 = pred["matches1"]
    conf = pred['matching_scores0']

    

    import matplotlib.cm as cm
    valid = mtch0 > -1
    print(valid.shape, kpt0.shape)
    mkpts0 = kpt0[valid]
    mkpts1 = kpt1[mtch0[valid]]
    mconf = conf[valid]
    color = cm.jet(mconf)


    #plot_matches(mkpts0, mkpts1, color)

    ax[0].scatter(mkpts0[:,0], mkpts0[:,1], cmap='gray')
    ax[1].scatter(mkpts1[:,0], mkpts1[:,1], cmap='gray')

    plt.show()
    H = None
    store = True
    try:
        H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        print(type(H))
    except:
        store = False

    supergluemodel, superpointmodel = supergluemodel.to('cpu') ,superpointmodel.to('cpu')

    desc0 = np.transpose(desc0)[valid]
    desc1 = np.transpose(desc1)[mtch0[valid]]

    return H, desc0, desc1, store

def generate_output():
    
    pass


def chceck_for_trasparent_bg(img):
    """ if image contains transparent background, remove it

    Args:
        img (image): image that is validated

    Returns:
        image: image without background
    """
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        img = img.convert('L').crop(alpha.getbbox())
    return img

def validate_pair(image0, image1):
    """ checking for valid pairs

    Args:
        image0 (image): left image of the pair
        image1 (image): right image of the pair

    Returns:
        bool: validation result
    """


    start0 = image0.split('/')[1][3:]
    end0 = start0.split('_')[0]

    start1 = image1.split('/')[1][3:]
    end1 = start1.split('_')[0]
    
    if int(end0) != int(end1):
        return False
    
    return True


def extract_matrixes(images, opt):
    """ processing pairs of images

    Args:
        images (list): list of pairs
    """

    #sp, sg = loadmodels()

    with alive_bar(len(images)) as bar, open (opt.output_data, 'w', newline='\n') as file:
        csvwriter = csv.writer(file, delimiter='\t')
        csvwriter.writerow(["matrix", "matches0", "matches1"])

        for i, im in enumerate(images):
            print("Processing files:", im)

            image0, image1 = Image.open(im[0]).resize([128,128]), Image.open(im[1]).resize([128,128])

            image0, image1 = chceck_for_trasparent_bg(image0), chceck_for_trasparent_bg(image1)
            
            if validate_pair(im[0], im[1]):
                H, m0, m1, store = generate_descriptors(image0, image1)
                if store and H is not None:
                    H = H.astype(float)
                    m0 = m0.astype(float)
                    m1 = m1.astype(float)
                    csvwriter.writerow([H.tolist(),m0.tolist(),m1.tolist()])
                else:
                    print(f"Homography matrix was not found for pair {im[0], im[1]}")

            else:
                print(f"Pair {im[0], im[1]} skipped. Not real pair.")
            bar()

        file.close()

    pass




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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image augmentation for generating dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--input', type=str, default='imagestitching.txt',
        help='Path to the list of image for augmentation')
    
    parser.add_argument(
        '--output_list', type=str, default='imagestitching_pairs_new.txt',
        help='Path to the list of image for augmentation')
    
    parser.add_argument(
        '--output_data', type=str, default='imagestitching_pairs_new.csv',
        help='Path to the list of image for augmentation')

    opt = parser.parse_args()

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

    extract_matrixes(pairs, opt)
    