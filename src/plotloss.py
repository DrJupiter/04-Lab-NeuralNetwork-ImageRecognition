import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys

filename = sys.argv[1]

data = pd.read_csv(filename, header=None,delim_whitespace=True)
data = np.array(data).flatten()

print(data)


fig, ax = plt.subplots()

ax.plot(range(0,int(data.shape[0])), data[:], '-o')

plt.show()
