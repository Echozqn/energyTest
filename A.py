
def solution(A,B):
    h1,m1 = A.split(':')
    h2,m2 = B.split(':')
    h1,m1,h2,m2 = int(h1),int(m1),int(h2),int(m2)
    if h2 < h1:
        h2 += 24

    ans = 0
    if h1 < h2:
        ans += (60 - m1) // 15
        ans += (h2 - h1 - 1) * 4
        ans += m2 // 15
    elif m1 < m2:
        ans += m2 // 15 - (m1+14) // 15

    return ans

print(solution('12:01',"12:44"))
print(solution('20:00',"06:00"))
print(solution('00:00',"23:59"))