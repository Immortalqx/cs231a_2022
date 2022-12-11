# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''


def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE

    # 定义一个求解直线一般方程的函数（这里避开了除法，且进行归一化）
    def compute_line_param(p_1, p_2):
        a = p_1[1] - p_2[1]
        b = -p_1[0] + p_2[0]
        c = p_1[0] * p_2[1] - p_1[1] * p_2[0]
        param = np.array([a, b, c])
        return param / np.max(param)

    param_1 = compute_line_param(points[0], points[1])
    param_2 = compute_line_param(points[2], points[3])
    temp = np.cross(param_1, param_2)

    return temp[:2] / temp[2]

    # END YOUR CODE HERE


'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''


def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE

    # 思路：
    # 三个vanishing points对应的三条平行线应该是互相垂直的
    # 根据ppt里面的内容，可以使用下面的公式计算出w：
    #   v_1^T*w*v_2=0
    #   v_1^T*w*v_3=0
    #   v_2^T*w*v_3=0
    # 由于w = (K*K^T)^{-1}，题中可以指定camera has zero skew and square pixels，所以w有4个未知数：w_1, w_4, w_5, w_6
    # 再根据course notes的介绍，由于w up tp scale，所以三个方程组是可以求出来的，在求解得到w之后可利用Cholesky decomposition求解K
    # Cholesky decomposition: https://zh.wikipedia.org/wiki/%E7%A7%91%E5%88%97%E6%96%AF%E5%9F%BA%E5%88%86%E8%A7%A3

    # 代码的思路：
    # 1.先把计算w的方程组转化成Ax=0的形式，其中
    #   A[0]=[x_1*x_2+y_1*y_2, x_1+x_2, y_1+y_2, 1](A[1], A[2]类似), x=[w_1, w_4, w_5, w_6]
    #   这里只需要计算出矩阵A即可！

    # 计算矩阵A每一行的数据
    def compute_row(v_1, v_2):
        return np.array([v_1[0] * v_2[0] + v_1[1] * v_2[1], v_1[0] + v_2[0], v_1[1] + v_2[1], 1])

    A = np.zeros((3, 4))
    A[0] = compute_row(vanishing_points[0], vanishing_points[1])
    A[1] = compute_row(vanishing_points[0], vanishing_points[2])
    A[2] = compute_row(vanishing_points[1], vanishing_points[2])

    # 2.利用SVD分解求解出w的一个解
    #   A不是方阵，不能用solve，但是可以做SVD分解求得x。
    #   为什么Ax=0的解是V的最后一列：https://blog.csdn.net/rourouwanzi1992/article/details/124752738
    _, _, V_T = np.linalg.svd(A)
    x = V_T[-1, :]
    w = np.array([[x[0], 0, x[1]],
                  [0, x[0], x[2]],
                  [x[1], x[2], x[3]]])

    # 3.利用Cholesky decomposition求解K
    #   w = (K*K^T)^{-1} = (K^T)^{-1}*K^{-1}，正好是一个下三角一个上三角
    K_T_inv = np.linalg.cholesky(w)
    K = np.linalg.inv(K_T_inv.T)

    # 4.normalize K
    return K / K[2, 2]

    # END YOUR CODE HERE


'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''


def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE

    # 关键问题：如何根据平面上的两个灭点，获取平面的方程，从而求得两个平面的夹角？
    # 解决方案：
    # 通过两个灭点得到平面在图像中的地平线方程l，则K_T*l就是这个平面的法线（而且是三维世界中的）
    # 这部分对应的笔记：https://immortalqx.github.io/2022/10/28/cs231a-lecture4/#Vanishing-points-and-planes

    # 求解地平线方程
    def compute_line_param(point_pair):
        a = point_pair[0][1] - point_pair[1][1]
        b = -point_pair[0][0] + point_pair[1][0]
        c = point_pair[0][0] * point_pair[1][1] - point_pair[0][1] * point_pair[1][0]
        param = np.array([a, b, c])
        return param / np.max(param)

    # 求解平面的法线
    n_1 = K.T.dot(compute_line_param(vanishing_pair1))
    n_2 = K.T.dot(compute_line_param(vanishing_pair2))

    # 计算夹角
    cos_angle = (n_1.dot(n_2)) / ((np.sqrt(n_1.dot(n_1))) * (np.sqrt(n_2.dot(n_2))))
    return (np.arccos(cos_angle) / math.pi) * 180

    # END YOUR CODE HERE


'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''


def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE

    # 关键问题：
    #  通过三组对应的灭点，和相机内参，求解两张图片的旋转矩阵
    # 思考：
    #  Q：前两组点都是同一个平面上的，能不能直接通过平面的法线计算出旋转矩阵呢？
    #     比如第一张图作为世界坐标系，法线n,第二张图作为从世界坐标系旋转得到的相机坐标系，法线n'=Rn
    #  A：可以不过没必要，根据v=Kd就能得到实际中平行线的方向向量d，这里可以有三组d来求R，简单又准确！
    # 解决方案：
    #  得到两张图中，三条平行线的方向向量，即D_1和D_2，根据D_2=R*D_1求解R。
    #  R=D_2*D_1^{-1} （不过这里不是很清楚D_1是否一定可逆。。。）

    # 定义一个求解d的函数
    def compute_d(v_p):
        # 转化到齐次坐标系
        v_p = np.hstack((v_p, 1))
        # 计算d
        d = np.linalg.inv(K).dot(v_p)
        # normalize
        return d / np.linalg.norm(d)

    # 计算第一张图像
    D_1 = np.zeros((3, 3))
    for i in range(3):
        D_1[i] = compute_d(vanishing_points1[i])
    D_1 = D_1.T

    # 计算第二张图像
    D_2 = np.zeros((3, 3))
    for i in range(3):
        D_2[i] = compute_d(vanishing_points2[i])
    D_2 = D_2.T

    # 计算旋转矩阵R
    return D_2.dot(np.linalg.inv(D_1))

    # END YOUR CODE HERE


if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674, 1826], [2456, 1060], [1094, 1340], [1774, 1086]]))
    v2 = compute_vanishing_point(np.array([[674, 1826], [126, 1056], [2456, 1060], [1940, 866]]))
    v3 = compute_vanishing_point(np.array([[1094, 1340], [1080, 598], [1774, 1086], [1840, 478]]))

    v1b = compute_vanishing_point(np.array([[314, 1912], [2060, 1040], [750, 1378], [1438, 1094]]))
    v2b = compute_vanishing_point(np.array([[314, 1912], [36, 1578], [2060, 1040], [1598, 882]]))
    v3b = compute_vanishing_point(np.array([[750, 1378], [714, 614], [1438, 1094], [1474, 494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n", compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0], [0, 2438.0, 986.0], [0, 0, 1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094, 1340], [1774, 1086], [1080, 598], [1840, 478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2],
                                         K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]),
                                                              K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z, y, x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
