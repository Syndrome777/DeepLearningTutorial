# coding=utf-8

#---------------------------------------
#   程序：百度贴吧爬虫
#   版本：0.1
#   作者：why
#   日期：2013-05-14
#   语言：Python 2.7
#   操作：输入带分页的地址，去掉最后面的数字，设置一下起始页数和终点页数。
#   功能：下载对应页码内的所有页面并存储为html文件。
#---------------------------------------
 
import string, urllib
 
#定义百度函数
def baidu_tieba(url,begin_page,end_page):   
    for i in range(begin_page, end_page+1):
        sName = string.zfill(i,5) + '.txt'#自动填充成六位的文件名
        print '正在下载第' + str(i) + '个网页，并将其存储为' + sName + '......'
        f = open(sName,'w+')
        m = urllib.urlopen(url + str(i)).read()

        #print m

        f.write(m)
        f.close()
 
 
#-------- 在这里输入参数 ------------------

# 这个是山东大学的百度贴吧中某一个帖子的地址
#bdurl = 'http://tieba.baidu.com/p/2296017831?pn='
#iPostBegin = 1
#iPostEnd = 10

#bdurl = str(raw_input(u'请输入贴吧的地址，去掉pn=后面的数字：\n'))
bdurl = 'http://tieba.baidu.com/p/2296017831?pn='
#begin_page = int(raw_input(u'请输入开始的页数：\n'))
#end_page = int(raw_input(u'请输入终点的页数：\n'))
begin_page = 1
end_page = 5
#-------- 在这里输入参数 ------------------
 

#调用
baidu_tieba(bdurl,begin_page,end_page)

response = urllib.urlopen('http://www.baidu.com/')  
html = response.read()  
print html  
