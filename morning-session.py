# IPython log file


import numpy as np
np.random.random()
import pandas as pd
get_ipython().magic('pinfo pd.melt')
get_ipython().magic('pinfo pd.melt')
import seaborn.apionly as sns
import calendar
calendar.month_name
calendar.month_name()
list(calendar.month_name)
pop = pd.read_csv('data/country_populations_by_year.csv')
pop.head()
list(columns)[:10]
list(pop.columns)[:10]
all_years = list(pop.columns)[2:] 
all_years[:10]
all_years2 = [str(i) for i in range(1960, 2011)]
all_years2[:10]
get_ipython().magic('pinfo pd.melt')
meltedpop = pd.melt(pop, id_vars=['Country Name', 'Country Code'],
                    value_vars=all_years,
                    value_name='year',
                    var_name='population')
                    
meltedpop.head()
meltedpop = pd.melt(pop, id_vars=['Country Name', 'Country Code'],
                    value_vars=all_years,
                    value_name='population',
                    var_name='year')
                    
meltedpop.head()
china_or_india_rows = ((meltedpop['Country Name'] == 'China') |
                       (meltedpop['Country Name'] == 'India'))
                       
china_or_india_rows.head()
popcnin = meltedpop.loc[china_or_india_rows]
popcnin.shape
popcnin.head()
import seaborn.apionly as sns
get_ipython().magic('pinfo sns.pointplot')
sns.pointplot(data=popcnin, x='year', y='population')
sns.pointplot(data=popcnin, x='year', y='population', hue='Country Name')
_32
45
_34
_32.figure
_32.figure.legends
_32.legend_
get_ipython().magic('pinfo _32.legend_.draggable')
_32.legend_.draggable(True)
_32.get_xticklabels()
ipython_green = np.array([1, 77, 1]) / 255
import matplotlib.pyplot as plt
plt.cm.colors.rgb2hex(ipython_green)
get_ipython().magic('pinfo plt.cm.colors.rgb2hex')
type(meltedpop['Country Name'])
