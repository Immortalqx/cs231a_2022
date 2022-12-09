# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''


def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    # Hint: reshape your values such that you have PM=p,
    # and use np.linalg.lstsq or np.linalg.pinv to solve for M.
    # See https://apimirror.com/numpy~1.11/generated/numpy.linalg.pinv
    #
    # Our solution has shapes for M=(8,), P=(48,8), and p=(48,)
    # Alternatively, you can set things up such that M=(4,2), P=(24,4), and p=(24,2)
    # Lastly, reshape and add the (0,0,0,1) row to M to have it be (3,4)

    # BEGIN YOUR CODE HERE

    # 思考：
    #   三张图像都是，每行2个像素点，然后12行。已经是一一对应了，直接求解就好了！
    #   real_xy是棋盘格的二维世界坐标，结合题中给出了Z（不是深度！），等于是有了三维世界坐标。
    #   所以这里的情况是，相机不动（M矩阵不变），然后有两个棋盘格的数据，等于是求一个超定方程！
    # 数学表述：
    #   对于超定方程 p = MP, 求M
    #   不过这种形式不好求解，所以可以根据ppt的方法，展开矩阵M为向量m，求解P'm=0
    #   但是更简单的应该是根据[u,v]^T=[m_1*P/m_3*p, m_2*P/m_3*P]来求解，这里的矩阵要好构造一些，也不用考虑m_3的问题。

    # 下面获取p和获取P有更简便的代码，如下所示：
    #   A_temp = np.ones((dims[0], 1));
    #   A1 = np.concatenate((real_XY, 0 * A_temp, A_temp), axis=1)
    #   b1 = front_image[:, 0]
    #   A2 = np.concatenate((real_XY, 150 * A_temp, A_temp), axis=1)
    #   b2 = back_image[:, 0]
    #   A = np.concatenate((A1, A2), axis=0)
    #   b = np.concatenate((b1, b2), axis=0)

    # 获取匹配的点的数目
    img_num1 = front_image.shape[0]
    img_num2 = back_image.shape[0]
    # img_num2 = 0

    # ================= 获取p =================
    # 初始化（注意p是3x1齐次坐标）
    p = np.zeros((2, img_num1 + img_num2))
    # 把x,y赋值进去
    for i in range(img_num1):
        p[:, i] = front_image[i, :].T
    for j in range(img_num2):
        p[:, j + img_num1] = back_image[j, :].T
    # 转化为齐次坐标
    p_ones = np.ones((1, p.shape[1]))
    p = np.vstack((p, p_ones)).T

    # ================= 获取P =================
    # 初始化（注意P是4x1齐次坐标）
    P = np.zeros((2, img_num1 + img_num2))
    # 把X, Y赋值进去
    for i in range(img_num1):
        P[:, i] = real_XY[i, :].T
    for j in range(img_num2):
        P[:, j + img_num1] = real_XY[j, :].T
    # 把Z赋值进去
    Z = np.zeros((1, P.shape[1]))
    for k in range(Z.shape[1]):
        # 第二张图就150，不然就默认为0
        if k >= img_num1:
            Z[:, k] = 150
    P = np.vstack((P, Z))
    # 转化为齐次坐标
    P_ones = np.ones((1, P.shape[1]))
    P = np.vstack((P, P_ones)).T

    # ================= 计算M =================
    # 根据m_1*P=u 求解m_1
    m_1 = np.linalg.lstsq(P, p.T[0], rcond=None)[0]
    # 根据m_2*P=v 求解m_2
    m_2 = np.linalg.lstsq(P, p.T[1], rcond=None)[0]
    # 由于是仿射变换，所以m_3可以直接指定
    m_3 = np.array([0, 0, 0, 1])
    M = np.vstack((m_1, m_2, m_3))

    return M

    # END YOUR CODE HERE


'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''


def rms_error(camera_matrix, real_XY, front_image, back_image):
    # BEGIN YOUR CODE HERE

    # 思考：
    #  这里直接套公式就可以了，先是求p'=MP，再根据p与p'求RMS

    # ================= 获取P =================
    P_ones = np.ones((real_XY.shape[0], 1));
    P_1 = np.concatenate((real_XY, 0 * P_ones, P_ones), axis=1)
    P_2 = np.concatenate((real_XY, 150 * P_ones, P_ones), axis=1)
    P = np.concatenate((P_1, P_2), axis=0)

    # ================= 获取p =================
    p_u = np.concatenate((front_image[:, 0], back_image[:, 0]), axis=0)
    p_v = np.concatenate((front_image[:, 1], back_image[:, 1]), axis=0)
    p = np.vstack((p_u, p_v)).T

    # ================= 计算p' =================
    p_pred = camera_matrix.dot(P.T)[:2].T

    # ================= 计算RMS =================
    x_ = (p.T[0] - p_pred.T[0]) ** 2
    y_ = (p.T[1] - p_pred.T[1]) ** 2
    RMS = np.sqrt((np.sum(x_ + y_) / x_.shape[0]))

    return RMS

    # END YOUR CODE HERE


if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
