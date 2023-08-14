
def lucky_money(money,giftees):
    num = min(money // 8,giftees)
    for i in range(num,0,-1):
        res = giftees - i
        val = money - i * 8
        if val == 4 and res == 1:
            continue
        if res > val:
            continue
        return i
    return 0

print(lucky_money(2500,2))
print(lucky_money(24,4))
print(lucky_money(7,2))

def compute_multiples(n):
    ans = 0
    for i in range(3,n):
        if i % 3 == 0 or i % 5 == 0 or i % 7 == 0:
            ans += i
    return ans

print(compute_multiples(11))