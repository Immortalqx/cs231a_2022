import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''


def lls_eight_point_alg(points1, points2):
    # 想法：
    # 核心公式：p^T*F*p'=0，可以转化为：Wf=0
    # 这里点的数目非常多，N>8，所以通过SVD进行求解！

    # 定义计算矩阵W的每一行的函数
    def compute_row(point1, point2):
        u1 = point1[0]
        v1 = point1[1]
        u2 = point2[0]
        v2 = point2[1]
        return np.array([u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, 1])

    # 计算矩阵W
    point_num = points1.shape[0]
    W = np.zeros((point_num, 9))

    for i in range(point_num):
        W[i] = compute_row(points1[i], points2[i])

    # 计算矩阵F_hat
    # SVD分解求f
    _, _, V_T = np.linalg.svd(W)
    f = V_T[-1, :]
    # normalize |f|=1
    f = f / np.sqrt(np.sum(f ** 2))
    # turn to Matrix F_hat
    F_hat = f.reshape((3, 3))

    # 计算矩阵F
    # 对F_hat进行SVD分解
    U, s_hat, V_T = np.linalg.svd(F_hat)
    # 取前两个奇异值
    s = np.zeros((3, 3))
    s[0][0] = s_hat[0]
    s[1][1] = s_hat[1]
    # 计算F
    # FIXME: 为什么这里要加上转置？
    # return U.dot(s).dot(V_T)
    return U.dot(s).dot(V_T).T


'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''


def normalized_eight_point_alg(points1, points2):
    # 0.compute T and T' for image 1 and 2
    # T包括2D的平移和缩放，形式如下：
    #           s 0 t1
    #       T = 0 s t2
    #           0 0 1

    # 定义函数计算T
    def compute_T(points):
        # 转到非齐次坐标系
        points_uv = points[:, 0:2]
        # 计算中心点的位置
        mean = np.mean(points_uv, axis=0)
        # 平移
        points_center = points_uv - mean
        # 计算缩放的比例（缩放到2个pixel）
        scale = np.sqrt(2 / (np.sum(points_center ** 2) / points_uv.shape[0] * 1.0))
        # 计算T
        return np.array([[scale, 0, -mean[0] * scale],
                         [0, scale, -mean[1] * scale],
                         [0, 0, 1]])

    T_1 = compute_T(points1)
    T_2 = compute_T(points2)

    # 1.normalize coordinates in images 1 and 2
    points1_ = T_1.dot(points1.T).T
    points2_ = T_2.dot(points2.T).T

    # 2.use the eight-point algorithm to compute F_q
    F_q = lls_eight_point_alg(points1_, points2_)

    # 3.de-normalize F_q: F=T^T*F_q*T'
    # FIXME: 由于原八点法结果转置，这里也要跟着转置，但是为什么？（为什么和推导的公式不一样，为什么前面错了这里也要被影响）
    # return T_1.T.dot(F_q).dot(T_2)
    return T_2.T.dot(F_q).dot(T_1)


'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''


def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a, b, c = line
            xs = [1, im.shape[1] - 1]
            ys = [(-c - a * x) / b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x, y, _ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1])) ** 2, 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')


'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''


def compute_distance_to_epipolar_lines(points1, points2, F):
    # 思路：
    # 这里要计算点与极线之间的平均距离
    # 极线 l = F*p', l' = F^T*p

    # 计算得一组极线
    l = F.T.dot(points2.T)
    # 参考网上的，写的很简便！！！
    # distance from point(x0, y0) to line: Ax + By + C = 0 is
    # |Ax0 + By0 + C| / sqrt(A^2 + B^2)
    d = np.mean(np.abs(np.sum(l * points1.T, axis=0)) / np.sqrt(l[0, :] ** 2 + l[1, :] ** 2))
    return d


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-' * 80)
        print("Set:", im_set)
        print('-' * 80)

        # Read in the data
        im1 = imread(im_set + '/image1.jpg')
        im2 = imread(im_set + '/image2.jpg')
        points1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
              compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
              compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
               for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
              F_normalized)
        print("Distance to lines in image 1 for normalized:", \
              compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
              compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
