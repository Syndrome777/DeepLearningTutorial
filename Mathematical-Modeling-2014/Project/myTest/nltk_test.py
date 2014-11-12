#coding:utf-8

word_num = [0,0,1,2,2,3,3,3,3,3,3,3,3,3,4]
word = ["你是","我今生","唯一的挚爱","你是我今生唯一的挚爱啊啊啊啊"]

TEST = "他说你是我今生唯一的挚爱"

T_len = len(TEST)/3
print T_len
s = T_len

while s > 0:
	flag = 0
	print TEST[0:s*3]
	for i in range(word_num[s]-1,word_num[s+1]):
		print word[i]+"sss"
		if TEST[0:s*3] == word[i]:
			print word[i] + "XXXXXX"
			flag = 1
	if flag == 1:
		TEST = TEST[s*3:]
		s = len(TEST)/3
	else:
		s -= 1
	if s == 1:
		print TEST[:s*3] + "ZZZZZZZ"
		TEST = TEST[s*3:]
		s = len(TEST)/3
		

import random
def guess(player):
	declare = 'You enter number not between 1 and 99!'
	number = int(raw_input('Player %s - Enter a number between 1 and 99:' % player))
	if number < 1:
		print declare
	elif number > 99:
		print declare
	else:
		pass
	return number
	
def game():
	i = 1
	count = [0,0,0]
	falg = True
	rambom_num = random.randrange(1,99)
	while falg:
		for player in range(0,3):
			number = guess(player + 1)
			count[player] = i
			if number > rambom_num:
				print 'Your guess is too high!'
			elif number < rambom_num:
				print 'Your guess is too low!'
			else:
				print '--------------------------------------'
				print 'Your made the right guess!'
				print 'The secret number is %s' % number
				for p in range(0,len(count)):
					print 'Player %s - Total number of guesses: %s' % (p + 1,count[p])
				falg = False
				break
		i = i + 1
 
game()
	