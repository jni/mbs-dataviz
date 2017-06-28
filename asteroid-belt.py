import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astroML.datasets import fetch_moving_objects
from astroML.plotting.tools import devectorize_axes


def compute_color(mag_a, mag_i, mag_z, a_crit=-0.1):
    """
    Compute the scatter-plot color using code adapted from
    TCL source used in Parker 2008.
    """
    # define the base color scalings
    red = np.ones_like(mag_i)
    green = 0.5 * 10 ** (-2 * (mag_i - mag_z - 0.01))
    blue = 1.5 * 10 ** (-8 * (mag_a + 0.0))

    # enhance green beyond the a_crit cutoff
    green += 10. / (1 + np.exp((mag_a - a_crit) / 0.02))

    # normalize color of each point to its maximum component
    rgb = np.vstack([red, green, blue])
    rgb /= np.max(rgb, axis=0)

    # return an array of RGB colors, which is shape (n_points, 3)
    return rgb.T


#------------------------------------------------------------
# Fetch data, extract the desired quantities, dither colours,
# precise only to 0.01
data = pd.read_csv('data/asteroid-belt.csv')
nrows = data.shape[0]
mag_a = data['mag_a'] - 0.005 + 0.01 * np.random.random(size=nrows)
mag_i = data['mag_i'] - 0.005 + 0.01 * np.random.random(size=nrows)
mag_z = data['mag_z'] - 0.005 + 0.01 * np.random.random(size=nrows)
a = data['aprime']
sini = data['sin_iprime']

# compute RGB color based on magnitudes
color_manual = compute_color(mag_a, mag_i, mag_z)

#------------------------------------------------------------
# set up the plot
with plt.style.context('dark_background'):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax0.scatter(mag_a, mag_i - mag_z, c=color_manual, s=1)

    ax0.set_xlabel('Optical Colour (a*)')
    ax0.set_ylabel('Near-IR Colour (i - z)')

    # plot the orbital parameters plot
    ax1.scatter(a, sini, c=color_manual, s=1)
    ax1.set_xlabel('Distance from the Sun (AU)')
    ax1.set_ylabel('Orbital Inclination (Sine)')

    # Saving the black-background figure requires some extra arguments:
    fig.savefig('asteroids.png', dpi=300)

plt.show(block=True)
