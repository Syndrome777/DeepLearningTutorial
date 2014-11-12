
import re
import urllib


def getHtml(url):
	page = urllib.urlopen(url)
	html = page.read()
	return html

def getImg(html):
	reg = r"src='+(.*?\.jpg)+' width"
	imgre = re.compile(reg)
	imgList = re.findall(imgre,html)
	x = 0
	for imgurl in imgList:
		print imgurl
		#urllib.urlretrieve(imgurl,'%s.jpg' % x)
		x+=1


#a = raw_input()

html = getHtml("http://tieba.baidu.com/p/2844418574?pn=2")
getImg(html)



