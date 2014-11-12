#!/usr/bin/python
#coding=utf-8

import string

datafile = open("car.txt")


n = 37
m = 2
mat = [[0]*m for i in range(n)]

i = 0
car = datafile.readline()
while car:
	car_data = car.strip('\n').split("	")
	j = 0
	for items in car_data:
		#字符串转换成数字
		data1 = string.atoi(items)
		mat[i][j] = data1
		j = j + 1
		#print data1
	i = i + 1
	car = datafile.readline()




for i in range(n):
	for j in range(m):
		print mat[i][j],
	print 



from sklearn.cluster import KMeans

kmeans = KMeans(init='k-means++', n_clusters = 4, n_init = 10)

kmeans.fit(mat)

result = kmeans.predict(mat)

print result







	




