import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import math
import matplotlib.pyplot as plt


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211551072


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each point
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    vector = np.array([[1, 0, -1]])
    I_X = cv2.filter2D(im2, -1, vector, borderType=cv2.BORDER_REPLICATE)
    I_Y = cv2.filter2D(im2, -1, vector.T, borderType=cv2.BORDER_REPLICATE)
    I_T = im2 - im1

    u_v = []
    x_y = []

    for i in range(win_size // 2, im1.shape[0], step_size):
        for j in range(win_size // 2, im1.shape[1], step_size):
            sample_I_X = I_X[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_Y = I_Y[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]
            sample_I_T = I_T[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1]

            sample_I_X = sample_I_X.flatten()
            sample_I_Y = sample_I_Y.flatten()
            sample_I_T = sample_I_T.flatten()

            n = len(sample_I_X)

            sum_IX_squared = sum(sample_I_X[h] ** 2 for h in range(n))
            sum_IX_IY = sum(sample_I_X[h] * sample_I_Y[h] for h in range(n))
            sum_IY_squared = sum(sample_I_Y[h] ** 2 for h in range(n))

            sum_IX_IT = sum(sample_I_X[h] * sample_I_T[h] for h in range(n))
            sum_IY_IT = sum(sample_I_Y[h] * sample_I_T[h] for h in range(n))

            A = np.array([[sum_IX_squared, sum_IX_IY], [sum_IX_IY, sum_IY_squared]])
            B = np.array([[-sum_IX_IT], [-sum_IY_IT]])

            eigen_val, eigen_vec = np.linalg.eig(A)
            eig_val1 = eigen_val[0]
            eig_val2 = eigen_val[1]

            if eig_val1 < eig_val2:
                eig_val1, eig_val2 = eig_val2, eig_val1

            if eig_val2 <= 1 or eig_val1 / eig_val2 >= 100:
                continue

            vector_u_v = np.linalg.inv(A) @ B
            u = vector_u_v[0][0]
            v = vector_u_v[1][0]

            x_y.append([j, i])
            u_v.append([u, v])

    return np.array(x_y), np.array(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    uv_return = []
    xy_return = []
    img1_pyr = gaussianPyr(img1, k)
    img2_pyr = gaussianPyr(img2, k)

    # Calculate optical flow for the last pyramid level
    x_y_prev, u_v_prev = opticalFlow(img1_pyr[-1], img2_pyr[-1], stepSize, winSize)
    x_y_prev = list(x_y_prev)
    u_v_prev = list(u_v_prev)

    for i in range(1, k):
        # Calculate optical flow for the current pyramid level
        x_y_i, uv_i = opticalFlow(img1_pyr[-1 - i], img2_pyr[-1 - i], stepSize, winSize)
        uv_i = list(uv_i)
        x_y_i = list(x_y_i)

        # Update uv according to the formula
        for j in range(len(x_y_prev)):
            x_y_prev[j] = [element * 2 for element in x_y_prev[j]]
            u_v_prev[j] = [element * 2 for element in u_v_prev[j]]

        # Update u_v_prev based on the location of movements
        for j in range(len(x_y_i)):
            try:
                index = x_y_prev.index(x_y_i[j])
                u_v_prev[index] += uv_i[j]
            except ValueError:
                x_y_prev.append(x_y_i[j])
                u_v_prev.append(uv_i[j])

    # Convert uv and xy to a 3-dimensional array
    arr3d = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    for x, y in x_y_prev:
        if [y, x] in x_y_prev:
            index = x_y_prev.index([y, x])
            arr3d[x, y] = u_v_prev[index]
    return arr3d


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """

    xy, uv = opticalFlow(im1, im2, step_size=20, win_size=5)
    u_values = uv[:, [0]]
    u_values = list(u_values.T[0])
    u_values = np.array(u_values)

    v_values = uv[:, [1]]
    v_values = list(v_values.T[0])
    v_values = np.array(v_values)

    min_difference = sys.maxsize
    translation_matrix = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]], dtype=np.float)

    for i in range(len(u_values)):
        t_ui = u_values[i]
        t_vi = v_values[i]

        translation_matrix_i = np.array([[1, 0, t_ui],
                                         [0, 1, t_vi],
                                         [0, 0, 1]], dtype=np.float)

        image_i = cv2.warpPerspective(im1, translation_matrix_i, im1.shape[::-1])
        mse = np.square(im2 - image_i).mean()

        if mse < min_difference:
            min_difference = mse
            translation_matrix = translation_matrix_i

    return translation_matrix


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    xy, uv = opticalFlow(im1, im2, step_size=20, win_size=5)
    new_xy = xy.copy()
    angle_list = []
    new_xy = new_xy.astype(float)

    for i in range(len(xy)):
        new_xy[i] += uv[i]
        angle_list.append(calculate_angle(xy[i], (0, 0), new_xy[i]))

    angle_list = np.array(angle_list)
    median_angle = np.median(angle_list)

    translation_matrix = findTranslationCorr(im1, im2)
    t_x = translation_matrix[0][2]
    t_y = translation_matrix[1][2]

    rigid_mat = np.float32([[np.cos(np.radians(median_angle)), -np.sin(np.radians(median_angle)), t_x],
                            [np.sin(np.radians(median_angle)), np.cos(np.radians(median_angle)), t_y],
                            [0, 0, 1]])

    return rigid_mat


def calculate_angle(first_point, third_point, second_point=(0, 0)):
    x1, y1 = first_point[0] - second_point[0], first_point[1] - second_point[1]
    x2, y2 = third_point[0] - second_point[0], third_point[1] - second_point[1]

    atan1 = math.atan2(y1, x1)
    atan2 = math.atan2(y2, x2)

    if atan1 < 0:
        atan1 += math.pi
    if atan2 < 0:
        atan2 += math.pi

    if atan1 <= atan2:
        return atan2 - atan1
    else:
        return math.pi / 3 + atan2 - atan1


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """

    x1, y1, x2, y2 = find_Xs_Ys_corr(im1, im2)
    translation_matrix = np.float32([[1, 0, x2 - x1 - 1], [0, 1, y2 - y1 - 1], [0, 0, 1]])
    return translation_matrix

def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """

    x1, y1, x2, y2 = find_Xs_Ys_corr(im1, im2)
    theta = calculate_angle((x2, y2), (0, 0), (x1, y1))
    transformation_matrix = np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0],
        [0, 0, 1]
    ])
    inverse_matrix = np.linalg.inv(transformation_matrix)
    rotated_image = cv2.warpPerspective(im2, inverse_matrix, im2.shape[::-1])
    x, y, x2, y2 = find_Xs_Ys_corr(im1, rotated_image)
    t_x = x2 - x - 1
    t_y = y2 - y - 1

    return np.float32([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta)), t_x],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta)), t_y / 6],
        [0, 0, 1]
    ])


def find_Xs_Ys_corr(image1, image2):
    """
    :param image1: input image 1 in grayscale format.
    :param image2: image 1 after Translation.
    :return: X's and Y's to find correlation.
    """

    padding_size = np.max(image1.shape) // 2
    padded_image1 = np.fft.fft2(np.pad(image1, padding_size))
    padded_image2 = np.fft.fft2(np.pad(image2, padding_size))
    product = padded_image1 * padded_image2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(product))
    correlation = result_full.real[1 + padding_size:-padding_size + 1, 1 + padding_size:-padding_size + 1]
    y1, x1 = np.unravel_index(np.argmax(correlation), correlation.shape)
    y2, x2 = np.array(image2.shape) // 2

    return x1, y1, x2, y2


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """

    if len(im1.shape) > 2:
        image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warped_image = np.zeros(image2.shape)
    for row in range(1, image2.shape[0]):
        for col in range(1, image2.shape[1]):
            # Find the coordinates of the point in image1 using the inverse transformation.
            current_vector = np.array([row, col, 1]) @ np.linalg.inv(T)
            dx, dy = int(round(current_vector[0])), int(round(current_vector[1]))

            # If the point has valid coordinates in image1, put the pixel in the warped image.
            if 0 <= dx < image1.shape[0] and 0 <= dy < image1.shape[1]:
                warped_image[row, col] = image1[dx, dy]

    return warped_image


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    gauss_pyramid = [img]

    for _ in range(1, levels):
        curr_img = cv2.resize(gauss_pyramid[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        gauss_pyramid.append(cv2.GaussianBlur(curr_img, (5, 5), 0))

    return gauss_pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    pyr = []
    kernel_size = 5
    kernel_sigma = int(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)
    kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = np.multiply(kernel, kernel.transpose()) * 4

    gaussian_pyr = gaussianPyr(img, levels)

    for i in range(levels - 1):
        pyr_img = gaussian_pyr[i + 1]
        extended_pic = np.zeros((pyr_img.shape[0] * 2, pyr_img.shape[1] * 2))
        extended_pic[::2, ::2] = pyr_img
        extend_level = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        curr_level = gaussian_pyr[i] - extend_level
        pyr.append(curr_level)

    pyr.append(gaussian_pyr[-1])

    return pyr

def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel) * 4

    cur_layer = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        cur_layer_shape = (cur_layer.shape[0] * 2, cur_layer.shape[1] * 2)
        extended_pic = np.zeros(cur_layer_shape)
        extended_pic[::2, ::2] = cur_layer
        cur_layer = cv2.filter2D(extended_pic, -1, kernel, borderType=cv2.BORDER_REPLICATE) + lap_pyr[i]

    return cur_layer



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    firstLaplacianPyr = laplaceianReduce(img_1, levels=levels)
    secondLaplacianPyr = laplaceianReduce(img_2, levels=levels)
    gaussPyrMask = gaussianPyr(mask, levels=levels)

    # Create a new blended pyramid by iterating over each level and blending the images using the mask
    blended_pyr = []
    for i in range(len(gaussPyrMask)):
        blended_level = gaussPyrMask[i] * firstLaplacianPyr[i] + (1 - gaussPyrMask[i]) * secondLaplacianPyr[i]
        blended_pyr.append(blended_level)

    # Expand the blended pyramid to build the desired image
    blended_img = laplaceianExpand(blended_pyr)

    # Create the naive blend image by copying the first image and replacing the masked regions with the second image
    naive_img = img_1.copy()
    naive_img[mask == 0] = img_2[mask == 0]

    return naive_img, blended_img