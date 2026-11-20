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


def angles_to(query, X):
    query = np.asarray(query, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if query.ndim != 1 or X.ndim != 2 or X.shape[1] != query.shape[0]:
        raise ValueError("angles_to 需要列数匹配，收到 %s 与 %s"
                         % (query.shape, X.shape))
    nq = float(np.linalg.norm(query))
    if nq == 0.0:
        raise ValueError("angles_to 的 query 不能为零向量")
    nX = np.linalg.norm(X, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (X @ query) / (nX * nq)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return np.where(nX == 0.0, np.nan, ang)


def np_m1():
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])

    ang = angles_to(query, X)
    print("1) angles_to(query, X) shape =", ang.shape, "度 =", ang)
    print("   第 4 行为零向量输出 nan 结果为", np.isnan(ang[3]))
    # ref = np.array([angle_deg(X[i], query) if np.linalg.norm(X[i]) else np.nan
    #                 for i in range(X.shape[0])])
    # print("   与遍历X各行的循环实现一致 =", np.allclose(ang, ref, atol=TOL, equal_nan=True))
    try:
        angles_to(np.zeros(3), X)
    except ValueError as e:
        print("   边界输入 query 为零向量 -> ValueError:", e)
    try:
        angles_to(query, X[:, :2])
    except ValueError as e:
        print("   边界输入 列数不匹配 -> ValueError:", e)
    return ang

def top_k_cosine(query, gallery, k):
    query = np.asarray(query, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    if query.ndim != 1 or gallery.ndim != 2 or gallery.shape[1] != query.shape[0]:
        raise ValueError("top_k_cosine 需要 query(d,) 与 gallery(N,d)，收到 %s 与 %s"
                         % (query.shape, gallery.shape))
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k 必须是正整数，收到 %r" % (k,))
    nG = np.linalg.norm(gallery, axis=1)
    n_valid = int((nG > 0.0).sum())
    if k <= 0 or k > n_valid:
        raise ValueError("k 必须是 1~%d（有效行数）的正整数，收到 %d" % (n_valid, k))
    nq = float(np.linalg.norm(query))
    if nq == 0.0:
        raise ValueError("top_k_cosine 的 query 不能为零向量")
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (gallery @ query) / (nG * nq)
    sim = np.where(nG > 0.0, sim, -np.inf)
    order = np.argsort(-sim, kind="stable")
    idx = order[:k]
    return idx, sim[idx]


def np_m2():
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
    k = 3

    idx, sim = top_k_cosine(query, gallery, k)
    print("1) k =", k, "| 选中行索引 =", idx, "| 相似度（降序）=", sim)
    print("   选中 ID =", [str(ids[i]) for i in idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        sim_all = (gallery @ query) / (np.linalg.norm(gallery, axis=1) * np.linalg.norm(query))
    print("   全库相似度 =", np.array2string(sim_all, precision=6),
          "（零向量行为 nan）")
    print("   img_a 与 img_b 同为 1.0，稳定排序保持原行顺序 =",
          list(idx[:2]) == [0, 1])
    for bad in (5, 0, -1, 2.5):
        try:
            top_k_cosine(query, gallery, bad)
        except ValueError as e:
            print("   边界输入 k =", bad, "-> ValueError:", e)
    try:
        top_k_cosine(np.zeros(3), gallery, k)
    except ValueError as e:
        print("   边界输入 零 query -> ValueError:", e)
    return idx, sim

def np_m3():
    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])

    diff = A[:, None, :] - B[None, :, :]
    print("1) 广播差数组 A[:,None,:] - B[None,:,:] shape =", diff.shape,
          "= (M, N, d)，无双重循环")

    D = np.linalg.norm(diff, axis=-1)
    print("2) 距离矩阵 D shape =", D.shape, "（D[i,j] = A[i] 到 B[j] 的欧氏距离）")

    print("3)   D =\n", D)
    nearest = np.argmin(D, axis=1)
    print(" A 每行最近的 B 行索引 =", nearest,
          "->", [B[j].tolist() for j in nearest])

    print("   边界输入 空 A(0,2) -> D shape =",
          np.linalg.norm(np.zeros((0, 2))[:, None, :] - B[None, :, :], axis=-1).shape)
    return D, nearest

def np_m4():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])
    snapshot = [arr.copy() for arr in (X, W1, b1, W2, b2)]

    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)
    Y = H @ W2.T + b2

    print("1) Z = X @ W1.T + b1   shape =", Z.shape, "值 =\n", Z)
    print("2) H = np.maximum(Z, 0.0) shape =", H.shape, "值 =\n", H)
    print("3) Y = H @ W2.T + b2   shape =", Y.shape, "值 =\n", Y)
    print("   形状流转：X(2,3) @ W1.T(3,2) -> Z(2,2) -> H(2,2) @ W2.T(2,2) -> Y(2,2)")
    manual = float(H[1] @ W2[1] + b2[1])
    print("   给定参数未被修改 =",
          all(np.array_equal(s, v) for s, v in zip(snapshot, (X, W1, b1, W2, b2))))
    try:
        _ = X @ W1
    except ValueError as e:
        print("   边界输入 漏写 .T -> ValueError:", e)
    return Z, H, Y



if __name__ == "__main__":
    print("\n")
    print("NumPy 数值计算练习：NP-E1 结果")
    print("\n")
    np_e1()

    print("\n")
    print("NumPy 数值计算练习：NP-E2 结果")
    print("\n")
    np_e2()

    print("\n")
    print("NumPy 数值计算练习：NP-E3 结果")
    print("\n")
    np_e3()

    print("\n")
    print("NumPy 数值计算练习：NP-E4 结果")
    print("\n")
    np_e4()

    print("\n")
    print("NumPy 数值计算练习：NP-E5 结果")
    print("\n")
    np_e5()

    print("\n")
    print("NumPy 数值计算练习：NP-E6 结果")
    print("\n")
    np_e6()

    print("\n")
    print("NumPy 数值计算练习：NP-E7 结果")
    print("\n")
    np_e7()

    print("\n")
    print("NumPy 数值计算练习：NP-E8 结果")
    print("\n")
    np_e8()

    print("\n")
    print("NumPy 数值计算练习：NP-M1 结果")
    print("\n")
    np_m1()

    print("\n")
    print("NumPy 数值计算练习：NP-M2 结果")
    print("\n")
    np_m2()

    print("\n")
    print("NumPy 数值计算练习：NP-M3 结果")
    print("\n")
    np_m3()

    print("\n")
    print("NumPy 数值计算练习：NP-M4 结果")
    print("\n")
    np_m4()