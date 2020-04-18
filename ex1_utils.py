"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import cv2
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 322219015


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1:
        img = cv2.imread(filename, 0)
        data = np.asarray(img)

    elif representation == 2:
        img = cv2.imread(filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = np.asarray(img_rgb)

    else:
        return None

    normal_data = data/data.max()
    return normal_data


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation == 1:
        img = cv2.imread(filename, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    elif representation == 2:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()

    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    r2y = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.211, -0.523, 0.311]])
    yiq = np.zeros(imgRGB.shape)
    for i in range(0, np.size(imgRGB, 0)):
        for j in range(0, np.size(imgRGB, 1)):
            yiq[i, j, 0] = r2y[0, 0] * imgRGB[i, j, 0] + r2y[0, 1] * imgRGB[i, j, 1] + r2y[0, 2] * imgRGB[i, j, 2]
            yiq[i, j, 1] = r2y[1, 0] * imgRGB[i, j, 0] + r2y[1, 1] * imgRGB[i, j, 1] + r2y[1, 2] * imgRGB[i, j, 2]
            yiq[i, j, 2] = r2y[2, 0] * imgRGB[i, j, 0] + r2y[2, 1] * imgRGB[i, j, 1] + r2y[2, 2] * imgRGB[i, j, 2]

    return yiq


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    r2y = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.275, -0.321],
                     [0.211, -0.523, 0.311]])

    y2r = np.linalg.inv(r2y)

    rgb = np.zeros(imgYIQ.shape)
    for i in range(0, np.size(imgYIQ, 0)):
        for j in range(0, np.size(imgYIQ, 1)):
            rgb[i, j, 0] = y2r[0, 0] * imgYIQ[i, j, 0] + y2r[0, 1] * imgYIQ[i, j, 1] + y2r[0, 2] * imgYIQ[i, j, 2]
            rgb[i, j, 1] = y2r[1, 0] * imgYIQ[i, j, 0] + y2r[1, 1] * imgYIQ[i, j, 1] + y2r[1, 2] * imgYIQ[i, j, 2]
            rgb[i, j, 2] = y2r[2, 0] * imgYIQ[i, j, 0] + y2r[2, 1] * imgYIQ[i, j, 1] + y2r[2, 2] * imgYIQ[i, j, 2]

    return rgb


def histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
