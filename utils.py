import os
from scipy.io import loadmat
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.misc import toimage
import glob
import matplotlib.image as mpimg

# CONSTANTS

FILE_SEPERATOR = '/'

CODE_WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
WORKING_DIR = os.path.dirname( CODE_WORKING_DIR ) + FILE_SEPERATOR + 'lab5'

BERKELEY_IMG_DIR = os.path.join(WORKING_DIR, 'BSR/BSDS500/data/images')
BERKELEY_GT_DIR = os.path.join(WORKING_DIR, 'BSR/BSDS500/data/groundTruth')

RANDOM_SEED = 0


DEFAULT_N_CLUSTERS = 8
K_MEANS_DEFAULT_MAX_ITER = 50

n_clusters = [3, 5, 7, 9, 11]
max_iter = [5, 10, 20, 30, 40, 50]

# METHODS

def get_image(img, category): 
    return mpimg.imread(BERKELEY_IMG_DIR + FILE_SEPERATOR 
                                       + category 
                                       + FILE_SEPERATOR 
                                       + img 
                                       + ".jpg")

def get_segment(img, category):
    return loadmat(BERKELEY_GT_DIR + FILE_SEPERATOR 
                                   + category 
                                   + FILE_SEPERATOR 
                                   + img 
                                   + ".mat")

def visualize_image(img, category):
    mat = get_segment(img, category)
    original_img = get_image(img, category)
    fig=plt.figure(figsize=(10, 15))
    plt.imshow(original_img)
    # display(jpgfile)
    
    # print(mat.keys())
    # print(mat['groundTruth'][0, 0][0, 0][0]) # segmentation from first human
    # print(mat['groundTruth'][0, 0][0, 0][1]) # Boundaries from first human

    human_participants_num = mat['groundTruth'].shape[1]
    fig=plt.figure(figsize=(15, 20))
    plt.rcParams.update({'font.size': 18})
    for i in range(human_participants_num - 1):
        # Load segmentation from ith human
        segm = np.array(mat['groundTruth'][0, i][0, 0][0])
        # Load Boundaries from ith human
        bound = np.array(mat['groundTruth'][0, i][0, 0][1])
        fig.add_subplot(human_participants_num, 2, (i * 2 + 1))
        plt.axis('off')
        plt.imshow(segm)
        fig.add_subplot(human_participants_num, 2, (i * 2 + 1) + 1)
        plt.axis('off')
        plt.imshow(bound)
    plt.show()

# visualize_image('22090', 'train')

def visualize_result(original_image, segmented_image, gt):
    """
    This functino will visualize the result of a segmentation algorithm against
    its ground truth image.
    """
    rows = 3
    cols = gt.shape[0]
    fig = plt.figure()

    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title('original image')
    plt.axis('off')
    plt.imshow(original_image)
    ax = fig.add_subplot(rows, cols, 2)
    ax.set_title('segmented image')
    plt.axis('off')
    plt.imshow(segmented_image)
    for m in range(cols):
        ax = fig.add_subplot(rows, cols, cols + i)
        ax.set_title('ground truth segmentation')
        plt.axis('off')
        plt.imshow(gt[i, :, :, 0])

    for m in range(cols):
        ax = fig.add_subplot(rows, cols, (cols * 2) + i)
        ax.set_title('ground truth boundries')
        plt.axis('off')
        plt.imshow(gt[i, :, :, 1])

def get_gt_image(img_name, category):
    mat = get_segment(img_name, category)
    human_participants_num = mat['groundTruth'].shape[1]
    
    gt = [] # ((human_participants_num, height, width, 2))
    for i in range(human_participants_num):
        # Load segmentation from ith human
        segm = np.array(mat['groundTruth'][0, i][0, 0][0])
        # Load Boundaries from ith human
        bound = np.array(mat['groundTruth'][0, i][0, 0][1])
        gt.append(np.dstack([segm, bound]))
    gt = np.array(gt)
    return gt

def load_images(category, limit=False, max_limit=5):
    """
    load images from path for given category set (test, train, val)
    """
    X = [] # images data
    y = [] # ground truth (labels)
    count = 0
    for root, dirnames, filenames in os.walk(f'{BERKELEY_IMG_DIR}/{category}'):
        for image_name in filenames:
            if limit and count >= max_limit:
                break
            count += 1
            if image_name.split('.')[1] != 'jpg':
                continue
            image = mpimg.imread(os.path.join(root, image_name))
            X.append(image)
            y.append(get_gt_image(image_name.split('.')[0], category))
    return np.array(X), np.array(y)
    
# X, y = load_images('test')
# print(y.shape)
# print(X.shape)
# plt.imshow(y[0][0][:, :, 0])
