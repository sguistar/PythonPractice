T = int(input())
for _ in range(T):
    a, b, c = map(int, input().split())
    d = b - a #公差
    n = (c - a) // d + 1 #项数
    s = n * (a + c) // 2
    print(s)
    
     