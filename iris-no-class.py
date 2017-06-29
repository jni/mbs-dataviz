import matplotlib.pyplot as plt
import pandas as pd
import seaborn.apionly as sns

flowers = pd.read_csv('data/iris.csv')

g = sns.pairplot(data=flowers, y_vars=['petal_length', 'petal_width'],
                 x_vars=['sepal_length', 'sepal_width'],
                 plot_kws={'linewidth': 0})
plt.show(block=True)
g.fig.savefig('iris-pairs0.png', dpi=300)
