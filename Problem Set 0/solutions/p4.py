# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def part_a():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.
    # Hint: use io.imread to read in the image file

    img1 = None

    # BEGIN YOUR CODE HERE

    img1 = io.imread("image1.jpg", as_gray=True)
    u, s, v = np.linalg.svd(img1, full_matrices=False)
    # 注意，这里的s是一维的向量！
    # 有需要的话，可以通过下面的方式转换为对角矩阵
    # s = np.diag(s)

    # END YOUR CODE HERE
    return u, s, v


def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE

    # TODO
    #  既然是 the best，那么如何评估best？
    #  这里可以仅仅通过u、s、v来评价某个特征值效果最好吗？
    u_1 = u[:, 0].reshape(len(u[0]), 1)
    s_1 = s[0]
    v_1 = v[0, :].reshape(1, len(v[0]))
    rank1approx = np.dot(u_1, s_1 * v_1)

    # 绘图
    plt.close()
    plt.title("the best rank 1 approximation ")
    plt.imshow(rank1approx, cmap=plt.cm.Greys_r)
    plt.savefig("result/p4b.png")

    # END YOUR CODE HERE
    return rank1approx


def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE

    # MxM MxN NxN = MxN
    # Mx20 20x20 20xN = MxN
    u_20 = u[:, :20].reshape(len(u[0]), 20)
    s_20 = np.diag(s)[:20, :20].reshape(20, 20)
    v_20 = v[:20, :].reshape(20, len(v[0]))
    rank20approx = np.dot(u_20, np.dot(s_20, v_20))

    # 绘图
    plt.close()
    plt.title("the best rank 20 approximation ")
    plt.imshow(rank20approx, cmap=plt.cm.Greys_r)
    plt.savefig("result/p4c.png")

    # END YOUR CODE HERE
    return rank20approx


if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)
