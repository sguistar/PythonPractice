import copy

EPSILON = 1e-10


def get_matrix_rank(matrix):
    A = copy.deepcopy(matrix)

    if not A or not A[0]:
        return 0

    m = len(A)
    n = len(A[0])
    col = 0
    row = 0

    while row < m and col < n:
        # --- 步骤 A: 寻找当前列的主元 (Pivot) ---
        # 从当前行 row 开始，向下寻找第 col 列中绝对值最大的元素
        pivot_row = row
        max_val = abs(A[row][col])

        for i in range(row + 1, m):
            if abs(A[i][col]) > max_val:
                max_val = abs(A[i][col])
                pivot_row = i

        # --- 步骤 B: 检查主元是否为 0 (或接近 0) ---
        if max_val < EPSILON:
            col += 1
            continue

        # --- 步骤 C: 交换行，将主元行移到当前行 (row) ---
        if pivot_row != row:
            A[row], A[pivot_row] = A[pivot_row], A[row]

        # --- 步骤 D: (可选，但有助于RREF) 将主元行归一化 ---
        # 我们这里为了求秩，可以跳过归一化，
        # 但为了消元方便，我们还是把主元变为 1
        pivot_value = A[row][col]
        if abs(pivot_value) > EPSILON:
            for j in range(col, n):
                A[row][j] /= pivot_value

        # --- 步骤 E: 消元 (Elimination) ---
        # 将当前行 (row) 下方的所有行的第 col 列变为 0
        for i in range(m):
            if i == row:
                continue

            factor = A[i][col]
            for j in range(col, n):
                A[i][j] -= factor * A[row][j]

        # --- 步骤 F: 移动到下一个主元位置 ---
        # 成功处理了这一行 (row)，我们将行和列都推进
        row += 1
        col += 1

    # 循环结束后，row 的值就是我们找到的主元数量，即矩阵的秩
    # (因为每成功找到并处理一个主元，row 才会 +1)
    rank = row
    return rank


n = int(input())
for _ in range(n):
    A = []
    for _ in range(4):
        a, b, c, d = map(int, input().split())
        A.append([a, b, c, d])
    r_A = get_matrix_rank(A)
    n = len(A)
    if r_A == n:
        print(1)
    else:
        print('INF')
