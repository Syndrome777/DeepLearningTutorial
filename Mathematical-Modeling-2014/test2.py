#!/usr/bin/python
#coding=utf-8
# 数学建模：单辆矫运车装车方案，前四问
# 输入为：矫运车长度，宽度
# 输出为：装车方案

# 高度超过1.7米的乘用车只能装在1-1、1-2型下层
# 纵向及横向的安全车距均至少为0.1米

Put = [0,1,0,1,0,0]
#长 上宽 下宽
Truck = [19.1,24.4,19.1]
#长 宽 高
Car = [4.71,3.715,4.73]


for i in range(0,6):
	if Put[i] == 0:
		for j in range(0,int(Truck[i/2]/Car[0])+2):
			for k in range(0,int(Truck[i/2]/Car[1])+2):
				if j*Car[0]+k*Car[1] > Truck[i/2]:
					if k > 0 :
						print(i,j,k-1)
					break
	else:
		for j in range(0,int(Truck[i/2]/Car[0])+2):
			for k in range(0,int(Truck[i/2]/Car[1])+2):
				for l in range(0,int(Truck[i/2]/Car[2])+2):
					if j*Car[0]+k*Car[1]+l*Car[2] > Truck[i/2]:
						if l > 0 :
							print(i,j,k,l-1)
						break







 
