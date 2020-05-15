

print("Classifier,Weighted,Epochs,EmbeddingSize,HiddenSize,NumLayers,KernelSize,Dropout,BatchSize,LearningRate,Accuracy,F1,NegF1,PosF1")
with open("weighted_cnn_results.txt", 'r') as r:
	for idx, line in enumerate(r):
		line = line.strip('\n')[1:-1]
		line = line.split("]")
		data = line[-1]
		line = line[:-1]
		val_str = ""
		for idx_line, value in enumerate(line):
			if idx_line == 5:
				val_str += "N/A,"
			if value != "":
				value = value.split("[")[-1]
				if "," in value:
					value = "[" + value + "]"
					value = value.replace(", ", "|")
				val_str += value + ","

		data = data.split(": ")[1:]
		for idx_data, d in enumerate(data):
			d = d.split(",")
			if idx_data != len(data) -1:
				val_str += d[0][:-1] + ","
			else:
				val_str += d[0][:-1]

		print(val_str)
		