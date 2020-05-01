from os import listdir
from os.path import isfile, join
import random

train = .7
dev = .3
'''
onlyfiles = [f for f in listdir("../twitter_sentiment/") if isfile(join("../twitter_sentiment/", f)) and ".txt" in f]


for file in onlyfiles:
	train_file = open(f'../twitter_sentiment/{file[:-4]}_train.txt', 'w')
	dev_file = open(f'../twitter_sentiment/{file[:-4]}_dev.txt', 'w')
	test_file = open(f'../twitter_sentiment/{file[:-4]}_test.txt', 'w')

	count = len(open(f'../twitter_sentiment/{file}').readlines(  ))
	lines = open(f'../twitter_sentiment/{file}').read().splitlines()
	lines = random.sample(lines, len(lines))
	train_count = 0
	dev_count = 0
	for l in lines:
		if train_count / count < train:
			train_file.write(file[0] + ',' + file[1] + ',' + l + '\n')
			train_count += 1
		elif dev_count / count < dev:
			dev_file.write(file[0] + ',' + file[1] + ',' + l + '\n')
			dev_count += 1
		else:
			test_file.write(file[0] + ',' + file[1] + ',' + l + '\n')
	#print(train_count, dev_count, count - train_count - dev_count)

	train_file.close()
	dev_file.close()
	test_file.close()
'''
onlyfiles = [f for f in listdir("../twitter_sentiment/") if isfile(join("../twitter_sentiment/", f)) and "Twitter" not in f and ".txt" in f]


for file in onlyfiles:
	train_file = open(f'../twitter_sentiment/{file[:-4]}_trimmed.txt', 'w')
	dev_file = open(f'../twitter_sentiment/{file[:-4]}_dev.txt', 'w')
	#test_file = open(f'../twitter_sentiment/{file[:-4]}_test.txt', 'w')

	count = len(open(f'../twitter_sentiment/{file}').readlines(  ))
	lines = open(f'../twitter_sentiment/{file}').read().splitlines()
	lines = random.sample(lines, len(lines))
	train_count = 0
	dev_count = 0
	for l in lines:
		if train_count / count < train:
			train_file.write(l + '\n')
			train_count += 1
		else:
			dev_file.write(l + '\n')
			dev_count += 1

	#print(train_count, dev_count, count - train_count - dev_count)

	train_file.close()
	dev_file.close()
	#test_file.close()