import os

train = "../twitter_sentiment/semeval_train_trimmed.txt"
test = "../twitter_sentiment/semeval_train_dev.txt"
epochs = 30
classifier = "CNN"
lr = 1e-3
batch_size = 100

kernel_sizes = ["1", "1 3", "1 3 5", "3", "3 5", "5", "1 3 5 7"]
emb_size = [16, 32, 64, 128]
dropout = [0, .1, .5]
hidden_sizes = ["32", "64", "128", "32 32", "64 64", "128 128", "32 32 32", "64 64 64", "128 128 128", "32 64 128", "128 64 32"]

completed_runs = {"k" : [], "e" : [], "d" : [], "h" : []}

with open("master_out.txt", 'r') as f:
	for line in f:
		line = line.strip("\n").split(":")[1][1:].split("]")[:-1]
		completed_runs["k"].append(line[0][1:])
		line = line[1].split("[")
		completed_runs["h"].append(line[1])
		line = line[0].split(" ")[1:-1]
		completed_runs["e"].append(line[0])
		completed_runs["d"].append(line[1])
f.close()

w = open("master_out.txt", "a")

i = 180
for k in kernel_sizes:
	for e in emb_size:
		for d in dropout:
			for h in hidden_sizes:
				if k not in completed_runs["k"] and e not in completed_runs["e"] and d not in completed_runs["d"] and h not in completed_runs["h"]:
					os.system(f'python3 handleTwitterSent.py --train {train} --test {test} --emb-size {e} --hidden-sizes {h} --kernel-sizes {k} --dropout {d} --classifier {classifier} --epochs {epochs} --lr {lr} --batch_size {batch_size} > ./runs/output_{i}.txt')
					i += 1
					w.write(f'completed runs : [{k}] {e} {d} [{h}]\n')
				print(f"completed runs : [{k}] {e} {d} [{h}]")
w.close()