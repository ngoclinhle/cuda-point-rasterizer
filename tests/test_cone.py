import numpy as np

window = np.zeros((7,7))
x,y = 3,3

def sector_from_ij(i, j):
    tr = i>=0 and j < 0
    br = i>0 and j >= 0
    bl = i<= 0 and j>0
    tl = i<0 and j <= 0
    if tr and i < -j: return 0
    if tr and i>=-j: return 1
    if br and i>j: return 2
    if br and i<=j: return 3
    if bl and -i<j: return 4
    if bl and -i>=j: return 5
    if tl and -i>-j: return 6
    if tl and -i<=-j: return 7
    return -1

for i in range(-3,4):
    for j in range(-3,4):
        window[y+j,x+i] = sector_from_ij(i,j)

print(window.reshape(-1))