import os
import sys
import random
import warnings


from tqdm import tqdm
from itertools import chain

import numpy as np
import pandas as pd

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '/Users/Paul/Dropbox/Work/Kaggle_Nuclei/Kaggle_Nuclei_Counting/1_Data/stage1_train/'
TEST_PATH = '/Users/Paul/Dropbox/Work/Kaggle_Nuclei/Kaggle_Nuclei_Counting/1_Data/stage1_test/'
META_PATH =  '/Users/Paul/Dropbox/Work/Kaggle_Nuclei/Kaggle_Nuclei_Counting/1_Data/Training_MetaData.csv'


print(TRAIN_PATH)




Training_MetaData = pd.read_csv(META_PATH, index_col=0)

print(Training_MetaData.head())

#warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
train_ids_2 = Training_MetaData['ID'].tolist()
test_ids = next(os.walk(TEST_PATH))[1]

#print(train_ids)



# print(Training_MetaData[['ID']].head())



# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

# for n in enumerate(train_ids):
#     print(n)

for n in enumerate(train_ids_2):
    print(n)


# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#     path = TRAIN_PATH + id_
#     print(path)
    # img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # X_train[n] = img
    # mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    # for mask_file in next(os.walk(path + '/masks/'))[2]:
    #     mask_ = imread(path + '/masks/' + mask_file)
    #     mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
    #                                   preserve_range=True), axis=-1)
    #     mask = np.maximum(mask, mask_)
    # Y_train[n] = mask

# # Get and resize test images
# X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# sizes_test = []
# print('Getting and resizing test images ... ')
# sys.stdout.flush()
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#     path = TEST_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#     sizes_test.append([img.shape[0], img.shape[1]])
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_test[n] = img

print('Done!')