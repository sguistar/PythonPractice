n = int(input())
for _ in range(n):
    num = input()
    res = set()
    for c in num:
        res.add(int(c))
    for i in range(1,len(num)):
        res.add(int(num[:i]))
        res.add(int(num[i:]))
        for j in range(i+1,len(num)+1):
                res.add(int(num[i:j]))

    res.add(int(num))
    res = sorted(res)
    print(*res)