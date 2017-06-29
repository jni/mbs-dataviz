# IPython log file


from astroML import datasets
X = datasets.fetch_great_wall()
A = datasets.fetch_moving_objects()
X.shape
plt.scatter(*X.T)
plt.scatter(*X.T, s=1, c='k')
plt.scatter(X[:, 1], X[:, 0], s=1, c='k')
X.shape
fig, ax = plt.subplots()
ax.set_facecolor('black')
fig, ax = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
for a in ax:
    a.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_color('w')
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
for a in ax.ravel():
    a.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_color('w')
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
for a in ax.ravel():
    a.set_facecolor('black')
    for spine in a.spines.values():
        spine.set_color('w')
    for tick in a.xaxis.get_major_ticks() + a.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
A.shape
A[:2]
def manual_color(mag_a, mag_i, mag_z, a_crit=-0.1):
    red = np.ones_like(mag_i)
    green = 0.5 * 10**(-2 * (mag_i - mag_z - 0.01))
    blue = 1.5 * 10**(-8 * (mag_a + 0.0))
    green += 10 / (1 + np.exp((mag_a - a_crit) / 0.02))
    rgb = np.vstack((red, green, blue))
    rgb /= np.max(rgb, axis=0)
    return rgb.T
mag_a = A['mag_a']
mag_i = A['mag_i']
mag_z = A['mag_z']
a = data['aprime']
a = A['aprime']
sini = A['sin_iprime']
mag_a += -0.005 + 0.01 * np.random.random(size=mag_a.shape)
mag_i += -0.005 + 0.01 * np.random.random(size=mag_i.shape)
mag_z += -0.005 + 0.01 * np.random.random(size=mag_z.shape)
colors = manual_color(mag_a, mag_i, mag_z)
np.max(mag_a)
ax.shape
get_ipython().magic('pinfo datasets.fetch_moving_objects')
colors[:3]
np.any(np.isinf(colors))
np.any(np.isnan(colors))
invalid = np.any(np.isnan(colors), axis=1)
maga = mag_a[~invalid]
magi = mag_i[~invalid]
magz = mag_z[~invalid]
a = a[~invalid]
sini = sini[~invalid]
ax[0].scatter(maga, magi-magz, c=colors, s=1)
A = datasets.fetch_moving_objects(Parker2008_cuts=True)
A.shape
maga = A['mag_a']
magi = A['mag_i']
magz = A['mag_z']
a = A['aprime']
sini = A['sin_iprime']
maga += -0.005 + 0.01 * np.random.random(size=maga.shape)
magi += -0.005 + 0.01 * np.random.random(size=magi.shape)
magz += -0.005 + 0.01 * np.random.random(size=magz.shape)
colors = manual_color(maga, magi, magz)
fig, ax = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
for a in ax.ravel():
    a.set_facecolor('black')
    for spine in a.spines.values():
        spine.set_color('w')
    for tick in a.xaxis.get_major_ticks() + a.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
ax[0].scatter(maga, magi - magz, c=color, s=1)
ax[0].scatter(maga, magi - magz, c=colors, s=1)
ax[0].set_xlabel('Optical colour (a*)', color='white')
ax[0].set_ylabel('Near-IR colour (i-z)', color='white')
ax[1].scatter(a, sini, c=colors, s=1)
a.shape
a = A['aprime']
ax[1].scatter(a, sini, c=colors, s=1)
ax.set_xlabel('Distance (AU)', color='white')
ax[1].set_xlabel('Distance (AU)', color='white')
ax[1].set_ylabel('Sine of orbital inclination', color='white')
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = np.rad2deg(np.arctan2(x, y))
    hsv = (1, 1, angle)
    from skimage import color
    return color.hsv2rgb(hsv)
np.arctan2(1, 1)
np.rad2deg(np.arctan2(1, 1))
np.rad2deg(np.arctan2(0, 1))
np.rad2deg(np.arctan2(1, 0))
np.rad2deg(np.arctan2(-1, 1))
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = np.rad2deg(np.arctan2(x, y)) + 180
    hsv = np.transpose((np.ones_like(angle), np.ones_like(angle), angle))
    from skimage import color
    return color.hsv2rgb(hsv)
get_ipython().magic('pinfo color.hsv2rgb')
from skimage import color
get_ipython().magic('pinfo color.hsv2rgb')
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = np.rad2deg(np.arctan2(x, y)) + 180
    ones = np.ones_like(angle)
    hsv = np.reshape((ones, ones, angle), (1, -1, 3))
    return color.hsv2rgb(hsv).reshape((-1, 3))

colors2 = auto_color(maga, magi, magz)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = np.rad2deg(np.arctan2(x, y)) + 180
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return color.hsv2rgb(hsv).reshape((-1, 3))

colors2 = auto_color(maga, magi, magz)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = np.rad2deg(np.arctan2(x, y)) / 180
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return color.hsv2rgb(hsv).reshape((-1, 3))
colors2 = auto_color(maga, magi, magz)
ax[0].clear()
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
colors2.shape
colors.shape
colors2.min(axis=0)
colors2.max(axis=0)
colors3 = (colors2 - _85) / (_86 - _85)
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
ax[0].scatter(maga, magi - magz, c=colors3, s=1)
color.rgb2hsv([[[0, 1, 1]]])
color.rgb2hsv(np.array([[[0, 1, 1]]], dtype=float))
color.rgb2hsv([[[0, 1, 1.]]])
color.rgb2hsv([[[1., 1, 1.]]])
color.rgb2hsv([[[0., 0, 1.]]])
color.rgb2hsv([[[1., 0, 1.]]])
h = np.linspace(0, 1, 10000)
s = np.ones_like(h)
v = np.ones_like(h)
h = np.linspace(0, 1, 10000).reshape((100, 100))
s = np.ones_like(h)
v = np.ones_like(h)
rgb = color.hsv2rgb(np.dstack((h, s, v)))
fig2, ax2 = plt.subplots()
ax2.imshow(rgb)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return color.hsv2rgb(hsv).reshape((-1, 3))
colors2 = auto_color(maga, magi, magz)
ax[0].clear()
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
for a in ax.ravel():
    a.set_facecolor('black')
    for spine in a.spines.values():
        spine.set_color('w')
    for tick in a.xaxis.get_major_ticks() + a.yaxis.get_major_ticks():
        for child in tick.get_children():
            child.set_color('w')
np.min(magi)
np.max(magi)
np.min(magi - magz)
np.max(magi - magz)
np.min(maga)
np.max(maga)
auto_color(-0.1, -0.3)
auto_color(-0.1, -0.3, 0)
auto_color(1, 1, 0)
auto_color(0.1, 0.1, 0)
ax[0].scatter(maga, magi - magz, c=colors, s=1)
plt.figure(); plt.hist(colors2[:, 0], bins='auto')
selector = (maga > -0.1) & (maga < -0.05) % (magi - magz < -0.75)
selector = (maga > -0.1) & (maga < -0.05) & (magi - magz < -0.75)
np.sum(selector)
auto_color(maga[selector], magi[selector], magz[selector])
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return angle, color.hsv2rgb(hsv).reshape((-1, 3))
auto_color(maga[selector], magi[selector], magz[selector])
_[0] * 360
maga[selector]
(magi - magz)[selector]
np.arctan2(_129, _130)
np.rad2deg(np.arctan2(_129, _130))
sel2 = (maga > 0.25) & (magi - magz > 0.4)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return 360 * angle - 180, color.hsv2rgb(hsv).reshape((-1, 3))

auto_color(maga[sel2], magi[sel2], magz[sel2])
auto_color(maga[selector], magi[selector], magz[selector])
angles = auto_color(maga, magi, magz)[0]
plt.figure(); plt.hist(angles, bins='auto')
angles, colors2 = auto_color(maga, magi, magz)
plt.figure(); plt.scatter(angles, magi - magz, c=colors2)
plt.figure(); plt.imshow(rgb)
np.max(h)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))

angles, hsvs, colors2 = auto_color(maga, magi, magz)
np.max(hsvs[:, 0])
plt.figure(); plt.hist(hsvs[:, 0], bins='auto')
hsvs.shape
plt.figure(); plt.hist(hsvs[0, :, 0], bins='auto')
x = maga
y = magi - magz
atans = np.arctan2(x, y)
plt.figure(); plt.hist(atans, bins='auto');
ang2s = np.rad2deg(atans)
plt.figure(); plt.hist(ang2s, bins='auto');
ang3s = (180 + ang2s) / 360
plt.figure(); plt.hist(ang3s, bins='auto');
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.reshape((angle, ones, ones), (1, -1, 3))
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.hstack((angle, ones, ones)).reshape(1, -1, 3)
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))

angles, hsvs, colors2 = auto_color(maga, magi, magz)
np.max(hsvs[0, :, 0])
plt.figure(); plt.hist(hsvs[0, :, 0], bins='auto')
get_ipython().set_next_input('hsv = np.stack');get_ipython().magic('pinfo np.stack')
hsv = np.stack((ang3s, np.ones(ang3s.size), np.ones(ang3s.size)), axis=-1)
hsv.shape
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z
    angle = (180 + np.rad2deg(np.arctan2(x, y))) / 360
    ones = np.ones_like(angle)
    hsv = np.stack((angle, ones, ones), axis=-1).reshape(1, -1, 3)
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))

angles, hsvs, colors2 = auto_color(maga, magi, magz)
plt.figure(); plt.hist(hsvs[0, :, 0], bins='auto');
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
ax[1].scatter(a, sini, c=colors2, s=1)
a.shape
a = A['aprime']
aprime = A['aprime']
ax[1].scatter(aprime, sini, c=colors2, s=1)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z + 0.1
    angle = (180 + np.rad2deg(np.arctan2(y, x))) / 360
    ones = np.ones_like(angle)
    hsv = np.stack((angle, ones, ones), axis=-1).reshape(1, -1, 3)
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))
angles, hsvs, colors2 = auto_color(maga, magi, magz)
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
ax[1].scatter(aprime, sini, c=colors2, s=1)
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z + 0.1
    r = np.hypot(x, y)
    rnorm = 0.5 + 0.5 * r / np.max(r)
    angle = (180 + np.rad2deg(np.arctan2(y, x))) / 360
    ones = np.ones_like(angle)
    hsv = np.stack((angle, rnorm, ones), axis=-1).reshape(1, -1, 3)
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
angles, hsvs, colors2 = auto_color(maga, magi, magz)
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
ax[0].clear()
def auto_color(mag_a, mag_i, mag_z):
    x = mag_a
    y = mag_i - mag_z + 0.1
    r = np.hypot(x, y)
    rnorm = 0.5 + 0.5 * (r / np.max(r)) ** (1/3)
    angle = (180 + np.rad2deg(np.arctan2(y, x))) / 360
    ones = np.ones_like(angle)
    hsv = np.stack((angle, rnorm, ones), axis=-1).reshape(1, -1, 3)
    return 360 * angle - 180, hsv, color.hsv2rgb(hsv).reshape((-1, 3))
angles, hsvs, colors2 = auto_color(maga, magi, magz)
ax[0].scatter(maga, magi - magz, c=colors2, s=1)
ax[1].scatter(aprime, sini, c=colors2, s=1)
from math import floor
floor(4.5)
int(4.6)
16.2 // 4
df_asteroids = pd.DataFrame(A)
df_asteroids.shape
df_asteroids.head()
len(set(df_asteroids['moID']))
df_asteroids.set_index('moID', in_place=True)
df_asteroids.set_index('moID', inplace=True)
df_asteroids.head()
df_asteroids.to_csv('/Users/jni/Dropbox/mbs-datasets/asteroid-belt.csv')
np.arcsin(0.3)
np.rad2deg(np.arcsin(0.3))
color.hsv2rgb([0.5, 1.0, 1.0])
color.hsv2rgb([[0.5, 1.0, 1.0]])
np.squeeze(color.hsv2rgb(np.atleast_3d([0.5, 1.0, 1.0])))
from matplotlib import colors as mplcolor
get_ipython().magic('pinfo mplcolor.hsv_to_rgb')
mplcolor.hsv_to_rgb((0., 1., 1.))
mplcolor.hsv_to_rgb((1., 1., 1.))
mplcolor.hsv_to_rgb((0.5, 1., 1.))
np.median(maga)
np.median(magi - magz)
import calendar
calendar.month_name
list(calendar.month_name)
np.random.random()
months = list(calendar.month_name)[1:]
sunlight = pd.read_csv('data/peak_sunlight_hours.csv', skiprows=1, header=0)
sunlight.head()
sun2 = pd.melt(sunlight, id_vars=['Month'], value_vars=months,
               var_name='Peak sunlight hours', value_name='Month')
               
sun2.head()
sun2 = pd.melt(sunlight, value_vars=months,
               var_name='Peak sunlight hours', value_name='Month')
               
sun2.head()
sun2 = pd.melt(sunlight, id_vars=['City', 'Country or US State'],
               value_vars=months,
               var_name='Peak sunlight hours', value_name='Month')
              
sun2.head()
sun2 = pd.melt(sunlight, id_vars=['City', 'Country or US State'],
               value_vars=months,
               var_name='Month', value_name='Peak sunlight hours')
              
sun2.head()
australia_rows = sun2['Country or US State'] == 'Australia'
sun2oz = sun2.loc[australia_rows]
import seaborn.apionly as sns
get_ipython().magic('pinfo sns.pointplot')
sun2['Month'] = pd.to_datetime(sun2['Month'])
sns.barplot(data=sun2, x='Month', y='Peak sunlight hours', hue='City')
sns.barplot(data=sun2oz, x='Month', y='Peak sunlight hours',hue='City')
sns.pointplot(data=sun2oz, x='Month', y='Peak sunlight hours',hue='City')
from bokeh import plotting
TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
plot = plotting.figure(tools=TOOLS)
get_ipython().magic('pinfo plot.scatter')
get_ipython().magic('pinfo plot.line')
plt.color('C1')
get_ipython().magic('pinfo plt.cm.colors')
plt.cm.colors('C0')
plt.cm.colors.to_hex('C0')
for i, (city, data) in enumerate(sun2oz.groupby('City')):
    c = plt.cm.colors.to_hex(f'C{i}')
    plot.line(source=data, x='Month', y='Peak sunlight hours',
              color=c)
              
plotting.output_file('psh.html')
plotting.show(plot)
get_ipython().magic('pwd ')
from bokeh.io import curstate
curstate().autoadd = False
plotting.show(plot)
plotting.output_file('psh.html')
plotting.show(plot)
plot = plotting.figure(tools=TOOLS)
for i, (city, data) in enumerate(sun2oz.groupby('City')):
    c = plt.cm.colors.to_hex(f'C{i}')
    plot.line(source=data, x='Month', y='Peak sunlight hours',
              color=c)
              
plotting.output_file('psh.html')
plotting.show(plot)
plot = plotting.figure(tools=TOOLS)
plot2 = plotting.figure(tools=TOOLS)
curstate().autoadd = False
for i, (city, data) in enumerate(sun2oz.groupby('City')):
    c = plt.cm.colors.to_hex(f'C{i}')
    plot2.line(source=data, x='Month', y='Peak sunlight hours',
               color=c)
              
plotting.output_file('psh2.html')
plotting.show(plot2)
angles.shape
import itertools
markers = list(zip(itertools.repeat(2), itertools.repeat(2), angles))
plt.scatter(maga, magi-magz, marker=markers, s=1)
from matplotlib import markers as mplmarkers
mplmarkers.marker(markers[0])
mplmarkers.MarkerStyle(markers[0])
markers = [mplmarkers.MarkerStyle(m) for m inzip(itertools.repeat(2), itertools.repeat(2), angles)]
markers = [mplmarkers.MarkerStyle(m) for m in zip(itertools.repeat(2), itertools.repeat(2), angles)]
plt.scatter(maga, magi-magz, marker=markers, s=1)
plt.scatter(maga, magi-magz, marker=markers[0], s=1)
plt.scatter(maga, magi-magz, marker=markers[0], s=1)
X.shape
with plt.style.context('dark_background'):
    fig, ax = plt.subplots()
    x, d = X.T
    ax.scatter(x, d, s=0.5, linewidth=0, color=plt.cm.magma(1.0))
    ax.set_xlabel('x position (Mpsec)')
    ax.set_ylabel('Distance from Earth (Mpsec)')
    
with plt.style.context('dark_background'):
    fig, ax = plt.subplots()
    d, x = X.T
    ax.scatter(x, d, s=0.75, linewidth=0, color=plt.cm.magma(1.0))
    ax.set_xlabel('x position (Mpsec)')
    ax.set_ylabel('Distance from Earth (Mpsec)')
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(d), np.max(d))
    
