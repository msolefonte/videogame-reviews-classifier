import matplotlib.pyplot as plt
import numpy as np
from lib.loader import get_data
from lib.utils.common import calculate_distribution

x, y, names = get_data()

plt.xlabel('Rating')
plt.ylabel('Number of reviews')
distribution = calculate_distribution(y)
bar_range = np.array(range(len(distribution)))
plt.bar(bar_range - 0.35 / 2, distribution, width=0.35,
        label='Input Data', tick_label=range(len(distribution)))

plt.legend()
plt.savefig(f'images/input_data_distribution.jpg')
plt.clf
