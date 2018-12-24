import os

FILE_SEPERATOR = '/'

CODE_WORKING_DIR = os.path.dirname( os.path.realpath( __file__ ) )
WORKING_DIR = os.path.dirname( CODE_WORKING_DIR ) + FILE_SEPERATOR + 'lab5'

BERKELEY_IMG_DIR = os.path.join(WORKING_DIR, 'BSR/BSDS500/data/images')
BERKELEY_GT_DIR = os.path.join(WORKING_DIR, 'BSR/BSDS500/data/groundTruth')

RANDOM_SEED = 0


DEFAULT_N_CLUSTERS = 8
K_MEANS_DEFAULT_MAX_ITER = 50

n_clusters = [3, 5, 7, 9, 11]
max_iter = [5, 10, 20, 30, 40, 50]


