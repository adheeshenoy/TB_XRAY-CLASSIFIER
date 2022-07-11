import tensorflow.keras.preprocessing.image as tkpi
from PIL import Image, ImageTk
import os
import sys

def resize_image(file_path, image_size = 256):
    '''resize images to display in app'''
    im = Image.open(file_path)
    im = im.resize((image_size, image_size))
    return ImageTk.PhotoImage(image=im)

def font(text_size = 16):
    return ('roman', text_size, 'bold')

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)