import numpy as np

LABEL_PATH = "./label_class.npy"

data = np.load(LABEL_PATH, allow_pickle=True)
print(type(data))
print(data)
