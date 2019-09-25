import pygame
import numpy as np
import sys
from keras import backend as K

size = width, height = 1280, 720
speed = [1, 1]
black = 255, 255, 255


if __name__ == "__main__":

    print(K.tensorflow_backend._get_available_gpus())