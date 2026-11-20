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

def np_e2():
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    had = a * b
    dot = a @ b
    print("a * b =", had, "shape =", had.shape)
    print("a @ b =", round(float(dot), 6), "shape =", np.asarray(dot).shape)

    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    print("||a||  =", round(norm_a, 6), "| np.linalg.norm =", round(np.linalg.norm(a), 6))
    print("||b||  =", round(norm_b, 6), "| np.linalg.norm =", round(np.linalg.norm(b), 6))
    print("核对:", np.allclose(norm_a, np.linalg.norm(a)) and np.allclose(norm_b, np.linalg.norm(b)))
    return a, b


if __name__ == "__main__":
    # np_e1()
    np_e2()