import os
import random
import numpy as np
from tqdm import tqdm

from MAIN.my_constans import DATA_PATH

path = DATA_PATH / "flatten"

names = path.iterdir()

names = filter(lambda x: x.name[0] == "0", names)

n = 277_524 - 78_786

names = np.random.choice(list(names), n, replace=False)

for name in tqdm(names):
    os.remove(name)