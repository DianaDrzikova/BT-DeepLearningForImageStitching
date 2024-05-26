#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#   Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#   Edited by: Diana Maxima Drzikova
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib
import sys

from SuperGlueandSuperPoint.matching import Matching
from SuperGlueandSuperPoint.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='../datasets/evaluation_pipeline/assets.txt', #default="assets/scannet_sample_pairs_with_gt.txt", default='assets.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='../datasets/evaluation_pipeline', #default='assets/scannet_sample_images/', default='testImages',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='.',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[840, 680],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--rows', type=int, default=2,
        help='Sum of grid. (Must be positive)')
    parser.add_argument(
        '--cols', type=int, default=2,
        help='Sum of grid. (Must be positive)')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.rows < 1 or opt.cols < 1:
        raise ValueError('Stitching images can be done on two and more images.')

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)

    do_match = True
    do_eval = opt.eval
    do_viz = opt.viz
    do_viz_eval = opt.eval and opt.viz

    for i, pair in enumerate(pairs): #all rows


        all_rows = opt.rows
        all_cols = opt.cols
        
        if opt.rows*opt.cols != len(pair): # missmatch between pair lenght and set row and col arguments
            images = pair
            all_rows = len(images)
            all_cols = 1
        elif len(pair) > 2:
            images = pair[:(opt.rows*opt.cols)]
        else:
            print('For the stitching process at least two images are required: {}'.format(input_dir/images[j]))
            exit(1)

        matches_path = output_dir / '{}_matches.npz'.format(f"grid{i}")
        eval_path = output_dir / '{}_evaluation.npz'.format(f"grid{i}")
        viz_path = output_dir / '{}_matches.{}'.format(f"grid{i}", opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_evaluation.{}'.format(f"grid{i}", opt.viz_extension)

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished images {:5} of {:5}'.format(i, len(images)))
            continue
        
        image_data = []

        # loading the images
        for j in range(len(images)):
            image_data.append(read_image(input_dir / images[j], device, opt.resize, 0, opt.resize_float))
        
        # checking loading
        for j in range(len(image_data)):
            if image_data[i][0] is None:
                print('Problem reading image for grid: {}'.format(input_dir/images[j]))
                exit(1)

        timer.update('load_image')

        inp = []
        names = []

        for j in range(all_rows*all_cols):

            inp.append(image_data[j][1])
            names.append(images[j])

        inp = [inp[i].to('cpu') for i in range(len(inp))]


        if do_match:
            
            new_inp = np.array(inp).reshape(all_rows,all_cols) # column x row
            new_names = np.array(names).reshape(all_cols,all_rows)

            pred = []
            coordinates = []
            second_coors = []
            previous_imgsize = []

            blackbg = np.zeros((opt.resize[0]*3,opt.resize[0]*3))
            warpedimg = []
            newwrapedimg = []
            cords = []
            new_cords = []


            for k in range(all_rows*all_cols-1): # stitching process

                inp0, inp1 = torch.tensor(np.array(inp[0])).to(device), torch.tensor(np.array(inp[1])).to(device)
                if k != 0: 
                    inp0 = inp0.unsqueeze(0).unsqueeze(0)
                    
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                kp0 = pred["keypoints0"] 
                kp1 = pred["keypoints1"] 
                matches = pred["matches0"] 
                mconf = pred["matching_scores0"] 

                color = cm.jet(mconf)  
                valid = matches > -1
                mkpts0 = kp0[valid]
                mkpts1 = kp1[matches[valid]]

                inp0, inp1 = inp0.to('cpu'), inp1.to('cpu')

                inp0 = inp0.squeeze(0).squeeze(0)
                inp1 = inp1.squeeze(0).squeeze(0)

                inp0 = np.float32(inp0)
                inp1 = np.float32(inp1)

                H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
                result = cv2.warpPerspective(inp1, np.linalg.inv(H), (opt.resize[0]*2,opt.resize[1]*2)) 

                if len(names) == all_rows*all_cols: # store first image
                    coords = cv2.findNonZero(inp0)

                    x, y, w, h = cv2.boundingRect(coords)

                    result1 = inp0[y:y+h, x:x+w]
                    cords.append([x,y])

                    new_cords.append([0,0])

                    warpedimg.append(inp0)
                    newwrapedimg.append(result1)

                # warping the image
                warpedimg.append(cv2.warpPerspective(inp1, np.linalg.inv(H), (opt.resize[0]*2,opt.resize[1]*2)))

                coords = cv2.findNonZero(warpedimg[-1])

                x, y, w, h = cv2.boundingRect(coords)
        
                result1 = warpedimg[-1][y:y+h, x:x+w]
                cords.append([x,y]) 
                newwrapedimg.append(inp1)

                mk = [x,y]
                mkk = np.ones(3)

                mkk[0], mkk[1] = mk[0], mk[1]
                xyz = np.dot(H, mkk)
                xyz[0] = xyz[0] / xyz[2]
                xyz[1] = xyz[1] / xyz[2]
                xyz[2] = xyz[2] / xyz[2]

                new = [xyz[0], xyz[1]]

                if x+xyz[0] < 0:
                    xyz[0] = 0
                if y+xyz[1] < 0:
                    xyz[1] = 0

                new_cords.append([xyz[0], xyz[1]])  # new coords are not used, they were just an experiment to enhance
                                                    # transformation of the points

                result[0:inp0.shape[0], 0:inp0.shape[1]] = inp0

                # update the first image and shorten the array of images
                if k == 0: 
                    inp = [result] + inp[k+2:] 
                    names = [f"newimg{k}.png"] + names[k+2:]
                else:
                    inp = [result] + inp[k+1:]
                    names = [f"newimg{k}.png"] + names[k+1:]
                

            # plotting on black background
            blackbg[cords[0][1]:warpedimg[0].shape[0], cords[0][1]:warpedimg[0].shape[1]] = newwrapedimg[0]

            for j in range(1,len(newwrapedimg)):
                blackbg[cords[j][1]:cords[j][1]+newwrapedimg[j].shape[0], cords[j][0]:cords[j][0]+newwrapedimg[j].shape[1]] = newwrapedimg[j]
            
            coords = cv2.findNonZero(blackbg)

            x, y, w, h = cv2.boundingRect(coords)
        
            result = blackbg[y:y+h, x:x+w]
            
            plt.figure(figsize=(12,12))
            plt.imshow(result, cmap="gray")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/match_grid/stitched_{i}.png")
