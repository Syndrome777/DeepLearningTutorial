#coding: utf-8


###多线程 Multithreading

import threading
TOTAL = 0
MY_LOCK = threading.Lock()
class CountThread(threading.Thread):
    def run(self):
        global TOTAL
        for i in range(100):
            MY_LOCK.acquire()
            TOTAL = TOTAL + 1
            MY_LOCK.release()
        print('%s\n' % (TOTAL))
a = CountThread()
b = CountThread()
a.start()
b.start()



text1 = ["你是","我今生","唯一的挚爱","你是我今生唯一的挚爱啊啊啊啊"]
print text1.count("你是")+1




