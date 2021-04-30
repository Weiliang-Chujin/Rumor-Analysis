n = int(input())
name = []
id = []
score = []
for i in range(n):
    na,idd,sc = input().split(" ")
    name.append(na)
    id.append(idd)
    score.append(sc)
ma = max(score)
mi = min(score)
print(max(score))
print(score)
ma_i = score.index(ma)
mi_i = score.index(mi)
print(name[ma_i]+' '+id[ma_i])
print(name[mi_i]+' '+id[mi_i])