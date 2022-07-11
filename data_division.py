import contextlib
import os
from sklearn.utils import shuffle
import shutil
import cv2
import tensorflow as tf
import numpy as np

normal_path = 'TB_Chest_Radiography_Database/Normal'
tb_path = 'TB_Chest_Radiography_Database/Tuberculosis'

train_dir = 'train'
test_dir = 'test'
val_dir = 'val'
        
def setup():
    def create_folder(path):
        '''
        create folder:
        if folder exists, delete files and folder
        else create the folder
        '''
        if os.path.exists(path):
            try:
                for d in os.listdir(path):
                    for files in os.listdir(os.path.join(path, d)):
                        os.remove(os.path.join(path, d, files))
                    os.rmdir(os.path.join(path, d))
            except Exception:
                with contextlib.suppress(Exception):
                    for d in os.listdir(path):
                        os.remove(os.path.join(path, d))
            os.rmdir(path)
        os.mkdir(path)
    
    create_folder(train_dir)
    create_folder(test_dir)
    create_folder(val_dir)
    create_folder(os.path.join(train_dir, 'normal'))
    create_folder(os.path.join(train_dir, 'tb'))
    create_folder(os.path.join(val_dir, 'normal'))
    create_folder(os.path.join(val_dir, 'tb'))
    create_folder(os.path.join(test_dir, 'normal'))
    create_folder(os.path.join(test_dir, 'tb'))

def reorganize_dataset():
    '''create data split and move images accordingly'''

    # Get all images as a list
    normal_img_names = os.listdir(normal_path)
    tb_img_names = os.listdir(tb_path)

    normal_img_names = shuffle(normal_img_names, random_state = 0)
    tb_img_names = shuffle(tb_img_names, random_state = 0)
    print('Number of normal images:', len(normal_img_names))
    print('Number of tb images:', len(tb_img_names))

    # Split images in a 70, 15, 15 split between training, validation and testing for each
    normal = {'train': normal_img_names[:int(len(normal_img_names) * 0.7)], 
            'val': normal_img_names[int(len(normal_img_names) * 0.7): int(len(normal_img_names)*0.85)],
            'test': normal_img_names[int(len(normal_img_names) * 0.85):]}
    tb = {'train': tb_img_names[:int(len(tb_img_names) * 0.7)], 
            'val': tb_img_names[int(len(tb_img_names) * 0.7): int(len(tb_img_names)*0.85)],
            'test': tb_img_names[int(len(tb_img_names) * 0.85):]}

    def move_to_folder(d, parent_path, tag):
        for key, names in d.items():
            for name in names:
                path = os.path.join(parent_path, name)
                shutil.copy(path, os.path.join(key, tag))

    move_to_folder(normal, normal_path, 'normal')
    move_to_folder(tb, tb_path, 'tb')

def balance():
    '''Grab images from training dataset, augment them and balance training set'''
    
    normal_folder = 'train/normal'
    tb_folder = 'train/tb'
    
    img_names = list(os.listdir(tb_folder))
    np.random.shuffle(img_names)
    print(len(img_names))
    
    diff = int(abs(len(os.listdir(normal_folder)) - len(img_names)))

    print(diff)
    
    for i in range(diff):
        name = img_names[i%len(img_names)]
        image = cv2.imread(os.path.join(tb_folder, name))
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.6, 1)
        image = tf.image.random_saturation(image, 0, 1)
        image = tf.image.random_flip_left_right(image)
        image = tf.keras.preprocessing.image.random_zoom(image, (0.8, 1))
        cv2.imwrite(os.path.join(tb_folder, f'augment{i}.png'), image)
        
if __name__ == '__main__':
    balance()