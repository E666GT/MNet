W=int(input(("图片尺寸")))
F=int(input(("Kernal Size")))
P=int(input(("Padding（单侧）")))
S=int(input(("Stride")))

N = (W-F+2*P)/S+1
print(N)