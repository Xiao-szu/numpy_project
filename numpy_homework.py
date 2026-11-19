"""
NumPy 数值计算练习
只能使用使用 Python + NumPy
"""

import numpy as np # 代码开头添加


def np_e1():
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])

    print("ndim =", A.ndim, "| shape =", A.shape, "| size =", A.size)

    row2 = A[1]                 # 下标 1 即第 2 行
    last2 = A[:, -2:]           # 最后两列会变成 (3,2)
    print("第2行 A[1]      =", row2, "shape =", row2.shape)
    print("最后2列 A[:,-2:] =\n", last2, "shape =", last2.shape)

    B = A.copy()                # A复制为B，避免修改影响 A
    B = np.where(B < 0, 0, B)  # 负数置 0，正负判断向量化完成
    print("B(负数置0) =\n", B, "shape =", B.shape)
    print("A 是否保持不变:", np.array_equal(A, np.array([[0.20, -0.50, 1.10, 0.40],
                                                        [0.80, 0.30, -0.70, 0.60],
                                                        [-0.40, 0.90, 0.10, -0.20]])))
    return A



if __name__ == "__main__":
    np_e1()