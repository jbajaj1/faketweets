import os

train = "../twitter_sentiment/semeval_train_trimmed.txt"
test = "../twitter_sentiment/semeval_train_dev.txt"
epochs = 30
classifier = "CNN"
lr = 1e-3
batch_size = 100

kernel_sizes = ["1", "1 3", "1 3 5", "3", "3 5"]
emb_size = [32, 64, 128]
dropout = [0, .1]
hidden_sizes = ["32", "64", "128", "32 32", "64 64", "128 128"]


i = 0
for k in kernel_sizes:
	for e in emb_size:
		for d in dropout:
			for h in hidden_sizes:
				os.system(f'python3 handleTwitterSent.py --train {train} --test {test} --emb-size {e} --hidden-sizes {h} --kernel-sizes {k} --dropout {d} --classifier {classifier} --epochs {epochs} --lr {lr} --batch_size {batch_size} > ./runs/output_{i}.txt')
				print(f"completed runs : [{k}] {e} {d} [{h}]")
				i += 1
