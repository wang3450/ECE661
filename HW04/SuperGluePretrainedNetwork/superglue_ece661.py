#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 6 12:05:34 2022

This wrapper on superglue is created purely for the purpose of academic
instructions for ECE661-Computer Vision at Purdue University, West Lafayette.

For appropriate usage of superglue, please read the license terms and
conditions at the github webpage cited below.

Cite:
    github: https://github.com/magicleap/SuperGluePretrainedNetwork
    paper: @article{DBLP:journals/corr/abs-1911-11763,
              author    = {Paul{-}Edouard Sarlin and
                           Daniel DeTone and
                           Tomasz Malisiewicz and
                           Andrew Rabinovich},
              title     = {SuperGlue: Learning Feature Matching with Graph
              Neural Networks},
              journal   = {CoRR},
              volume    = {abs/1911.11763},
              year      = {2019},
              url       = {http://arxiv.org/abs/1911.11763},
              eprinttype = {arXiv},
              eprint    = {1911.11763},
              timestamp = {Tue, 03 Dec 2019 20:41:07 +0100},
              biburl    = {https://dblp.org/rec/journals/corr/abs-1911-11763
              .bib},
              bibsource = {dblp computer science bibliography, https://dblp.org}
            }

@author: Rahul Deshmukh
       Email: deshmuk5@purdue.edu
       Robot Vision Lab
       School of Electrical and Computer Engineering,
       Purdue University, West Lafayette, IN, US
"""

import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.matching import Matching
from models.utils import process_resize, read_image


class SuperGlue(object):
    def __init__(self):
        super(SuperGlue, self).__init__()
        self.matcher = None
        self.device = None
        self.config = None
        self.resize = None

    @classmethod
    def create(cls,
               force_gpu=False,
               nms_radius=4,
               keypoint_threshold=0.005,
               max_keypoints=-1,
               superglue_wts='indoor',  # 'outdoor'
               sinkhorn_iterations=20,
               match_threshold=0.2,
               resize=[640, 480]
               ):
        det = cls()
        det.set_device_as_gpu(force_gpu=force_gpu)
        det.set_config(nms_radius,
                       keypoint_threshold,
                       max_keypoints,
                       superglue_wts,
                       sinkhorn_iterations,
                       match_threshold)
        det.matcher = Matching(det.config).eval().to(det.device)
        det.resize = resize
        return det

    def set_device_as_gpu(self, force_gpu=True):
        self.device = 'cuda' if torch.cuda.is_available() and force_gpu else \
            'cpu'

    def set_config(self, nms_radius,
                   keypoint_threshold,
                   max_keypoints,
                   superglue_wts,
                   sinkhorn_iterations,
                   match_threshold):
        self.config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue_wts,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

    @torch.no_grad()
    def detectAndCompute(self, img):
        """
        Returns superpoint keypoints, scorees and descriptor
        :param img: str path to image
        :return:
                 keypoints: [Num_kp, 2] array
                 scores: [Num_kp] array
                 descriptors: [256, Num_kps ] array
        """
        image, inp, scales = self.read_img(img)
        # image: [H,W] gray scale image as numpy array dtype=uint8
        # inp: [1,1,H,W] image tensor dtype=float
        # scales: (scale_width, scale_ht) tuple of float. multiply with this
        # scale
        # to get coordinates in original image size

        data = self.matcher.superpoint({'image': inp})
        kp = data['keypoints'][0].numpy() * np.array(scales)
        scores = data['scores'][0].numpy()
        desc = data['descriptors'][0].numpy()
        return kp, scores, desc

    @torch.no_grad()
    def match(self, img0, img1):
        """
        match using superglue
        :param img0, img1: str path to images
        return:
            mkpts0: [N,2] numpy array of matching keypoints in img0 (x,y)
            mkpts1: [N,2] numpy array of matching keypoints in img1 (x,y)
            mconf: [N] numpy array of matching confidence probabilities
        """
        image0, inp0, scales0 = self.read_img(img0)
        image1, inp1, scales1 = self.read_img(img1)
        pred = self.matcher({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        # convert mkpts back to original image size using scales
        mkpts0 = np.array(mkpts0) * np.array(scales0)
        mkpts1 = np.array(mkpts1) * np.array(scales1)
        return mkpts0, mkpts1, mconf

    def read_img(self, img_path):
        """
        :param img_path: str full path to image file
        :return: img [H,W] grayscale image as np array \in [0-1]
        """
        image, inp, scales = read_image(img_path, self.device, self.resize, 0,
                                        False)
        return image, inp, scales


def read_rgb_img(img_path):
    im = Image.open(img_path)
    img = np.array(im)
    return img


def plot_keypoints(img0: str, img1: str, kp0, kp1, mkpts0, mkpts1, plt_name):
    ms = 0.5
    lw = 0.5
    img0 = read_rgb_img(img0)
    img1 = read_rgb_img(img1)
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    img = np.zeros((np.max([h0, h1]), w0 + w1, 3), dtype=np.uint8)
    img[:h0, :w0] = img0
    img[:h1, w0:] = img1

    plt.figure()
    plt.imshow(img)
    # plot all keypoints in red
    for ikp0 in range(kp0.shape[0]):
        plt.plot(kp0[ikp0, 0], kp0[ikp0, 1], 'r.', markersize=ms)
    for ikp1 in range(kp1.shape[0]):
        plt.plot(kp1[ikp1, 0] + w0, kp1[ikp1, 1], 'r.', markersize=ms)
    # plot all matching keypoints in green lines
    for im in range(mkpts0.shape[0]):
        x0, y0 = mkpts0[im, :]
        x1, y1 = mkpts1[im, :]
        x1 += w0
        plt.plot((x0, x1), (y0, y1), '--gx', linewidth=lw, markersize=ms)
    plt.axis('off')
    plt.savefig(plt_name, bbox_inches='tight', pad_inches=0, dpi=300)
    return


if __name__ == "__main__":
    img0 = sys.argv[1]
    img1 = sys.argv[2]
    outDir = sys.argv[3]

    assert os.path.exists(img0)
    assert os.path.exists(img1)
    assert os.path.exists(outDir)

    img0_base = os.path.basename(img0).split('.')[0]
    img1_base = os.path.basename(img1).split('.')[0]

    # initialize detector
    detector = SuperGlue.create()

    # detect and compute points using superpoint
    kp0, score0, descriptor0 = detector.detectAndCompute(img0)
    kp1, score1, descriptor1 = detector.detectAndCompute(img1)

    # compute matches using superpoint + superglue
    mkpts0, mkpts1, matching_confidence = detector.match(img0, img1)

    # plot matches
    plt_name = os.path.join(outDir,
                            img0_base + '_and_' + img1_base +
                            '_superglue_matches.png')
    plot_keypoints(img0, img1, kp0, kp1, mkpts0, mkpts1, plt_name)
