import glob
import os

for file in glob.glob('*.txt'):
    with open(file, 'r') as f:
        data = f.read().split("\n")

        print(data[-5:-1])
