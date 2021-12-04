import matplotlib.pyplot as plt
import numpy as np
from lib.loader import get_data
from lib.utils.common import calculate_distribution

# Formatting

plt.rc('font', size=14)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rcParams['figure.constrained_layout.use'] = True


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
