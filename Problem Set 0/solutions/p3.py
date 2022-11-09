# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.
    # Hint: use io.imread to read in the files

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE

    img1 = io.imread("image1.jpg")
    img2 = io.imread("image2.jpg")

    # END YOUR CODE HERE
    return img1, img2


def normalize_img(img):
    img = img.astype('double')

    newMax = 1
    newMin = 0
    imgMax = np.max(img)
    imgMin = np.min(img)

    newImg = (img - imgMin) * (newMax - newMin) / (imgMax - imgMin) + newMin
    return newImg


def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE

    img1 = normalize_img(img1)
    img2 = normalize_img(img2)

    # 绘图
    plt.subplot(1, 2, 1)
    plt.title("image1")
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.title("image2")
    plt.imshow(img2)
    plt.savefig("result/p3b.png")

    # END YOUR CODE HERE
    return img1, img2


def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None

    # BEGIN YOUR CODE HERE

    sumImage = normalize_img(img1 + img2)

    # 绘图
    plt.close()
    plt.title("Summation of Two Images")
    plt.imshow(sumImage)
    plt.savefig("result/p3c.png")

    # END YOUR CODE HERE
    return sumImage


def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    # 默认img1和img2是一样的size，不考虑缩放
    imgRow = np.shape(img1)[0]
    imgCol = np.shape(img1)[1]
    # 创建半张图像size的矩阵，用于拼接成filter
    # TODO 应当处理奇数和偶数的两种情况！
    #  如果这里imgCol是奇数（实际上并不是），会导致最中间一行为0
    #  11000 + 00011
    maskOne = np.ones((imgRow, round(imgCol / 2), 3))
    maskZero = np.zeros((imgRow, imgCol - round(imgCol / 2), 3))
    # 拼接完整的filter
    filterLeft = np.concatenate((maskOne, maskZero), axis=1)
    filterRight = np.concatenate((maskZero, maskOne), axis=1)
    # 计算结果
    newImage1 = np.multiply(img1, filterLeft) + np.multiply(img2, filterRight)
    newImage1 = normalize_img(newImage1)

    # 绘图
    plt.close()
    plt.title("half img1 + half img2")
    plt.imshow(newImage1)
    plt.savefig("result/p3d.png")

    # END YOUR CODE HERE
    return newImage1


def part_e(img1, img2):
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE

    newImage2 = np.zeros(np.shape(img1))
    for i in range(np.shape(img1)[0]):
        if i % 2 != 0:
            newImage2[i] = img1[i]
        else:
            newImage2[i] = img2[i]

    # 绘图
    plt.close()
    plt.title("odd img1 + even img2")
    plt.imshow(newImage2)
    plt.savefig("result/p3e.png")

    # END YOUR CODE HERE
    return newImage2


def part_f(img1, img2):
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE

    # 默认img1和img2是一样的size，不考虑缩放
    imgRow = np.shape(img1)[0]
    imgCol = np.shape(img1)[1]

    # 创建mask
    # TODO 没有考虑奇数的情况了，如果是奇数，mask的创建会稍微复杂一点
    # 奇数[0,1]循环imgRow/2次，img1
    maskOdd = np.tile(np.concatenate((np.zeros((1, imgCol, 3)), np.ones((1, imgCol, 3))), axis=0), (imgRow // 2, 1, 1))
    # 偶数[1,0]循环imgRow/2次，img2（从第0行开始，0为偶数）
    maskEven = np.tile(np.concatenate((np.ones((1, imgCol, 3)), np.zeros((1, imgCol, 3))), axis=0), (imgRow // 2, 1, 1))
    # 计算结果
    newImage3 = np.multiply(img1, maskOdd) + np.multiply(img2, maskEven)

    # 绘图
    plt.close()
    plt.title("odd img1 + even img2")
    plt.imshow(newImage3)
    plt.savefig("result/p3f.png")

    # END YOUR CODE HERE
    return newImage3


def part_g(img):
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.
    # Hint: use np.dot and the standard formula for converting RGB to grey
    # greyscale = R*0.299 + G*0.587 + B*0.114

    # BEGIN YOUR CODE HERE

    # 定义对每一个像素点进行转换操作的函数
    def rgb2gray(pixel):
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

    # 遍历每一个像素点
    temp = img
    for r in range(len(temp)):
        for c in range(len(temp[r])):
            img[r][c] = rgb2gray(temp[r][c])

    # 绘图
    plt.close()
    plt.title("Grayscale Image")
    plt.imshow(img)
    plt.savefig("result/p3g.png")

    # END YOUR CODE HERE
    return img


if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    img = part_g(newImage3)
