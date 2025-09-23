def FMM(dt, s):  # 正向最大匹配算法 start -> end
    result = []
    max_len = max([len(i) for i in dt])  # 选取字典里长度最大的字符串
    start = 0
    while start != len(s):  # 判断列表不为空，建立循环
        index = start + max_len  # 从0开始正向索引最大长度的字符串
        if index > len(s):  # 判断是否溢出列表
            index = len(s)
        for _ in range(max_len):
            t = s[start:index]  # t是切片
            if t in dt or len(t) == 1:
                result.append(t)
                start = index
                break
            index -= 1  # 为了保证算法能够扫描到所有字符
    return result


def RMM(dt, s):  # 反向最大匹配算法 end -> start
    result = []
    max_len = max([len(i) for i in dt])  # 选取字典里长度最大的字符串
    start = len(s)
    while start != 0:  # 判断列表不为空，建立循环
        index = start - max_len  # 从列表最后开始索引最大长度的字符串
        if index < 0:  # 判断是否溢出列表
            index = 0
        for _ in range(max_len):
            t = s[index:start]  # t是切片
            if t in dt or len(t) == 1:
                result.insert(0, t)  # 在最前面插入
                start = index
                break
            index += 1
    return result


def BM(dt, s):  # 双向最大切词
    r1 = FMM(dt, s)
    r2 = RMM(dt, s)
    if len(r1) == len(r2):
        if r1 == r2:
            return r1
        else:
            r1_cnt = len([i for i in r1 if len(i) == 1])
            r2_cnt = len([i for i in r2 if len(i) == 1])
            return r1 if r1_cnt < r2_cnt else r2
    else:
        return r1 if len(r1) < len(r2) else r2


dt1 = ["研究", "研究生", "生命", "命", "的", "起源"]
s1 = "研究生命的起源"

dt2 = ["I", "like", "sam", "sung", "samsung", "mobile", "ice", "cream"]
s2 = "Ilikesamsungmobileicecream"
print(FMM(dt1, s1))
print(RMM(dt1, s1))
print(BM(dt1, s1))

print(FMM(dt2, s2))
print(RMM(dt2, s2))
print(BM(dt2, s2))
