import sys

# 1..19 的英文单词（小写）
WORDS = [
    "one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen"
]

# 预先计算每个单词需要的字母集合
WORD_SETS = [set(w) for w in WORDS]
print(WORD_SETS)

def solve():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    t = int(data[0].strip())
    out_lines = []
    for i in range(1, 1 + t):
        s = data[i].strip()
        available = set(s)  # 字母可重复使用 -> 只需集合包含关系
        cnt = 0
        for ws in WORD_SETS:
            print([ws,available])
            if ws <= available:
                cnt += 1
        out_lines.append(str(cnt))
    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    solve()