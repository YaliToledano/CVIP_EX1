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

    normal_data = data.min()+((data-data.min())/(data.max()-data.min()))
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
        :return: (imgEq,histOrg,histEQ)
    """

    imgnew = simpScale(imgOrig)

    if imgOrig.ndim == 3:
        yiq = transformRGB2YIQ(imgnew)
        yiq[:, :, 0], histOrg, histEQ = histogram_Equalize(yiq[:, :, 0])

        return simpScale(transformYIQ2RGB(yiq)), np.zeros((256, 3)), np.zeros((256, 3))
    else:
        return histogram_Equalize(imgnew)


def histogram_Equalize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an gray image
    """
    imgnew = simpScale(imgOrig)
    imghist = calHist(imgnew)
    cumsum = calCumSum(imghist)  # normalized cumulative sum

    for i in range(0, np.size(imgnew, 0)):
        for j in range(0, np.size(imgnew, 1)):
            imgnew[i, j] = 255 * cumsum[imgnew[i, j]]

    return imgnew, imghist, calHist(imgnew)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
    imgnew = simpScale(imOrig)
    if imOrig.ndim == 3:
        yiq = transformRGB2YIQ(imgnew)
        listYIQ, listmst = quantize_Image(yiq[:, :, 0], nQuant, nIter)
        listq = []
        for x in range(0, nIter):
            yiq[:, :, 0] = listYIQ[x]
            listq.append(simpScale(transformYIQ2RGB(yiq)))

        return listq, listmst
    else:
        return quantize_Image(imgnew)


def quantize_Image(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    imgnew = simpScale(imOrig)
    hist = calHist(imgnew)
    init = 255 / nQuant
    listq = []
    listmst = []
    borders = np.zeros(nQuant + 1)
    for x in range(0, nQuant + 1):
        borders[x] = x * init

    for i in range(0, nIter):
        quantimage = imgnew
        borders = np.rint(borders).astype(int)
        print(borders)
        meanavgs = np.zeros(nQuant)
        for j in range(0, nQuant):
            meanavgs[j] = meanavg(np.arange(borders[j], borders[j + 1] + 1), hist[borders[j]:borders[j + 1] + 1])
        for c in range(0, nQuant):
            quantimage[(quantimage >= borders[c]) & (quantimage <= borders[c + 1])] = meanavgs[c]
            quantimage = np.rint(quantimage).astype(int)

        for j in range(1, nQuant):
            borders[j] = (meanavgs[j - 1] + meanavgs[j]) / 2

        listq.append(quantimage)
        mst = np.square(np.subtract(imgnew, quantimage)).mean()
        listmst.append(mst)

    return listq, listmst


def meanavg(intens: np.ndarray, vals: np.ndarray) -> int:
    val = (intens * vals).sum() / vals.sum()
    if np.isnan(val):
        return 0
    return val


def simpScale(img: np.ndarray) -> np.ndarray:
    img = ((img-img.min())/(img.max()-img.min()))*255
    return np.rint(img).astype(int)


def calHistColor(color_img: np.ndarray) -> np.ndarray:
    hist = np.zeros((256, 3))
    for c in range(0, 3):
        hist[:, c] = calHist(color_img[:, :, c])
        return hist


def calHist(img: np.ndarray) -> np.ndarray:
    img_flat = img.ravel()
    hist = np.zeros(256)
    for pix in img_flat:
        hist[pix] += 1

    return hist


def calCumSum(arr: np.array) -> np.ndarray:
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
    return cum_sum / max(cum_sum)


def divideHist(hist: np.array, borders_k: int) -> np.array:
    borders = np.zeros(borders_k + 1)
    borders[0] = 0
    borders[borders_k] = np.size(hist, 0) - 1
    sumdiv = hist.sum() / borders_k
    counter = 0
    y = 0
    z = 1
    for x in hist:
        if counter >= sumdiv:
            borders[z] = y
            z += 1
            counter = 0
        counter += x
        y += 1

    return borders    