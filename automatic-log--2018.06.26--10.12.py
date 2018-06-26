# IPython log file


import matplotlib.pyplot as plt
plt.plot([1, 3, 5, 2])
get_ipython().magic('pwd ')
get_ipython().magic('cd Dropbox/mbs-datasets\\ \\(1\\)')
import os
os.listdir()
import pandas as pd
cars = pd.read_csv('cars.csv')
cars.plot(kind='scatter', x='weight', y='mpg')
fig, axes = plt.subplots(nrows=1, ncols=2)
cars.columns
cars.plot(kind='scatter', x='weight', y='mpg', ax=axes[0])
cars.plot(kind='scatter', x='horsepower', y='acceleration', ax=axes[1])
type(axes)
axes.shape
fig.tight_layout()
cars.head()
origin_dict = {1: 'USA', 2: 'Europe', 3: 'Japan'}
origin_dict[2]
origin_dict.get(2)
cars['origin name'] = cars['origin'].apply(origin_dict.get)
cars.head()
get_ipython().magic('pinfo pd.melt')
pops = pd.read_csv('country_populations_by_year.csv')
pops.head()
pops.columns[:5]
get_ipython().magic('pinfo pd.melt')
pops_tidy = pd.melt(pops, id_vars=['Country Name', 'Country Code'],
                    var_name='Year', value_name='Population')
                    
pops_tidy.head()
peak = pd.read_csv('peak_sunlight_hours.csv')
peak.head()
peak = pd.read_csv('peak_sunlight_hours.csv', skip_rows=1)
peak = pd.read_csv('peak_sunlight_hours.csv', skiprows=1)
peak.head()
peak_tidy = pd.melt(peak, id_vars=['City', 'Country or US State'],
                    var_name='Month', value_name='Sunlight hours')
                    
peak_tidy.head()
import seaborn as sns
get_ipython().magic('pinfo sns.pointplot')
sns.pointplot(data=peak_tidy, x='Month', y='Sunlight hours', hue='City')
sns.pointplot(data=peak_tidy, x='Month', y='Sunlight hours', hue='City')
australian_rows = peak_tidy['Country or US State'] == 'Australia'
type(australian_rows)
australian_rows.dtype
peak_oz = peak_tidy[australian_rows]
peak_tidy.shape
peak_oz.shape
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
with plt.style.use('dark_background'):
    sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
    
with plt.style.context('dark_background'):
    sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
    
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
plt.style.available
plt.style.use('seaborn-bright')
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
plt.style.use('seaborn-white')
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
plt.cm.colors.rgb2hex((48, 107, 161))
plt.cm.colors.rgb2hex((48/255, 107/255, 161/255))
plt.cm.colors.rgb2hex((109/255, 110/255, 111/255))
plt.cm.colors.rgb2hex((255/255, 205/255, 59/255))
with plt.style.context('test.style'):
    sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
    
with plt.style.context('test.style'):
    sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
    
with plt.style.context('test.style'):
    sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
    
get_ipython().magic('pinfo cars.plot')
sns.pointplot(data=peak_oz, x='Month', y='Sunlight hours', hue='City')
get_ipython().magic('pinfo cars.plot')
for origin, data in cars.groupby('origin name'):
    ax = data.plot(kind='scatter', x='weight', y='mpg', label=origin)
    
fig, ax = plt.subplots()
for origin, data in cars.groupby('origin name'):
    data.plot(kind='scatter', x='weight', y='mpg', label=origin, ax=ax)
    
fig, ax = plt.subplots()
for origin, data in cars.groupby('origin name'):
    ax.scatter(x=cars['weight'], y=cars['mpg'], label=origin)
    
for origin, data in cars.groupby('origin name'):
    ax.scatter(x=data['weight'], y=data['mpg'], label=origin)
    
fig, ax = plt.subplots()
for origin, data in cars.groupby('origin name'):
    ax.scatter(x=data['weight'], y=data['mpg'], label=origin)
    
ax.legend()
ax.legend_
ax.legend_.draggable
ax.legend_.draggable(True)
colordat = pd.read_csv('colormap_data.csv')
colordat.head()
get_ipython().magic('pinfo sns.stripplot')
get_ipython().magic('pinfo sns.stripplot')
colordat['error'] = colordat['user_value'] - colordat['real_value']
sns.stripplot(data=colordat, x='colormap', y='error', hue='colormap')
sns.stripplot(data=colordat, x='colormap', y='error', hue='colormap', jitter=True)
from skimage import data
cam = data.camera()
plt.imshow(cam)
plt.imshow(cam, cmap='viridis')
cars.head()
cars['color'] = (cars['year'] - cars['year'].min()) / (cars['year'].max() - cars['year'].min())
cars['colorvalue'] = (cars['year'] - cars['year'].min()) / (cars['year'].max() - cars['year'].min())
cars['color'] = cars['colorvalue'].apply(plt.cm.viridis)
fig, ax = plt.subplots()
ax.scatter(x=cars['weight'], y=cars['mpg'], c=cars['color'])
