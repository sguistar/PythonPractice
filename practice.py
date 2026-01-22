s = input()
length = len(s)
l = 0
t = ''.join(sorted(s))
if t == s:
    print('Yes')
else:
    while l < length and s[l] == t[l]:
        l += 1

    r = length - 1

    while r >= 0 and s[r] == t[r]:
        r -= 1

    s2 = s[:l] + s[l:r + 1][::-1] + s[r + 1:]
    if s2 == t:
        print('Yes')
    else:
        print('No')


        