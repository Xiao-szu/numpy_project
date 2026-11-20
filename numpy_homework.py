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

def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a 与 b 必须是同维的一维向量")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("输入存在零向量，夹角无定义")
    cos_theta = np.clip((a @ b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def np_e3():
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    print("angle_deg(a, b) =", round(angle_deg(a, b), 6), "度")
    try:  # 任一输入为零向量时抛出 ValueError
        angle_deg(np.zeros(2), b)
    except ValueError as e:
        print("零向量输入 -> ValueError:", e)
    return angle_deg(a, b)

def np_e4():
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    d = p - q
    euclid = np.sqrt(np.sum(d ** 2))
    manhat = np.sum(np.abs(d))
    print("差向量 p - q =", d, "shape =", d.shape)
    print("欧氏距离   =", round(euclid, 6), "(核对 np.linalg.norm:", round(np.linalg.norm(d), 6), ")")
    print("曼哈顿距离 =", round(manhat, 6))
    return euclid, manhat

def unit_vector(a):
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("零向量没有方向，无法单位化")
    return a / n


def np_e5():
    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print("单位向量 u =", u, "shape =", u.shape)
    print("||u|| =", round(np.linalg.norm(u), 6), "| allclose(1):", np.allclose(np.linalg.norm(u), 1.0))
    print("原数组未被修改:", np.array_equal(a, np.array([-1.20, 0.40, 0.80])))
    try:
        unit_vector(np.zeros(3))
    except ValueError as e:
        print("零向量输入 -> ValueError:", e)
    return u

def np_e6():
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])

    col_sum = A.sum(axis=0)     # 跨 2 行累加 -> 每列一个和
    row_mean = A.mean(axis=1)   # 跨 3 列平均 -> 每行一个均值
    print("A.sum(axis=0)  =", col_sum, "shape =", col_sum.shape)
    print("A.mean(axis=1) =", np.round(row_mean, 6), "shape =", row_mean.shape)
    print("整表最大值 max =", A.max())
    print("说明：axis=0 的结果按列对齐 A[:,0]/A[:,1]/A[:,2]；axis=1 的结果按行对齐 A[0,:]/A[1,:]")
    return col_sum, row_mean

def np_e7():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    Y = X * scale + bias
    print("Y = X * scale + bias =\n", np.round(Y, 6), "shape =", Y.shape)
    print("广播检查：scale 视作 (1,3) 复制成 (2,3)，与 X 的每一行逐列对齐")
    return Y

def np_e8():
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10

    batch = np.stack([a, b], axis=0)          # (2,3,2,2)
    print("batch shape =", batch.shape, "(B, C, H, W)")
    blue = batch[:, 2, :, :]                  # 取 C=2 通道
    print("蓝色通道 batch[:,2] =\n", np.round(blue, 6), "shape =", blue.shape)
    batch_hwc = np.transpose(batch, (0, 2, 3, 1))
    print("转为 (B,H,W,C) 后 shape =", batch_hwc.shape)
    return batch

if __name__ == "__main__":
    # np_e1()
    # np_e2()
    # np_e3()
    # np_e4()
    # np_e5()
    # np_e6()
    # np_e7()
    np_e8()