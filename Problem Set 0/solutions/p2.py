# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt


def part_a():
    # ===== Problem 2a =====
    # Define and return Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE

    M = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [0, 2, 2]])

    # 这里实际上是错误的，题目中是3x1，而这里是1x3
    # 由于设置成3x1处理比较复杂，这里就简化成1x3做运算了
    a = np.array([1, 1, 0])
    b = np.array([-1, 2, 5])
    c = np.array([0, 2, 3, 2])

    # END YOUR CODE HERE
    return M, a, b, c


def part_b(a, b):
    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE

    # aDotb = a.dot(b)
    aDotb = np.dot(a, b)

    # END YOUR CODE HERE
    return aDotb


def part_c(a, b):
    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    aProdb = None

    # BEGIN YOUR CODE HERE

    # aProdb = a * b
    aProdb = np.multiply(a, b)

    # END YOUR CODE HERE
    return aProdb


def part_d(a, b, M):
    # ===== Problem 2d =====
    # Find (a^T b)Ma

    result = None

    # BEGIN YOUR CODE HERE

    result = np.dot(a, b) * np.dot(M, a)

    # END YOUR CODE HERE
    return result


def part_e(a, M):
    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function tile() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE

    # 直接重复4次，然后reshape
    # a = np.tile(a, 4).reshape(4, -1)
    # 第一维（行）复制4次，第二维（列）复制1次
    a = np.tile(a, (4, 1))
    newM = np.multiply(M, a)

    # END YOUR CODE HERE
    return newM


def part_f(M):
    # ===== Problem 2f =====
    # Without using a loop, sort all of the values
    # of M in increasing order and plot them.

    sortedM = None

    # BEGIN YOUR CODE HERE

    sortedM = np.sort(M, axis=None)

    # # plot
    # N = len(sortedM)
    # x = range(N)
    # # plot settings
    # plt.xlabel("Vector Index")
    # plt.ylabel("Values")
    # plt.title("Plot of Values in Sorted M")
    # plt.plot(x, sortedM, 'ro--')
    # plt.axis([0, N, np.min(sortedM), np.max(sortedM)])
    # plt.show()

    # END YOUR CODE HERE

    plt.bar(range(12), np.squeeze(list(sortedM)))
    plt.savefig('result/p2f.png')

    return sortedM


if __name__ == '__main__':
    M, a, b, c = part_a()
    aDotb = part_b(a, b)
    print("Problem 2b:%s" % str(aDotb))
    mult = part_c(a, b)
    print("Problem 2c:%s" % str(mult))
    ans = part_d(a, b, M)
    print("Problem 2d:%s" % str(ans))
    newM = part_e(a, M)
    print("Problem 2e:%s" % str(newM))
    part_f(newM)
