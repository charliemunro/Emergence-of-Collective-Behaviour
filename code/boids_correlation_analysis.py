#!/usr/bin/env python
"""
boids_correlation_analysis.py

63

Performs correlation analysis on Boids simulation output and generates plots:

1. Correlation Function Plot with a fitted curve.
2. LogLog Plot of the correlation function.
3. Spatial Correlation Map.
4. Boid Snapshot with Velocity Vectors.

Usage: python boids_correlation_analysis.py --input simulation_output.npz

--bins 20 --fit exponential

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import argparse
import os

def load_simulation_data(filename):

if not os.path.exists(filename):

raise FileNotFoundError(f"File not found: {filename}")

data = np.load(filename)
positions = data[’positions’]
velocities = data[’velocities’]
metadata = {’num_boids’: int(data[’num_boids’]),
’num_steps’: int(data[’num_steps’]),
’solver’: str(data[’solver’]),
’noise_level’: float(data[’noise_level’])}

return positions, velocities, metadata

def normalize_velocities(velocities):

norms = np.linalg.norm(velocities, axis=1)
norms[norms < 1e-10] = 1.0
return velocities / norms[:, np.newaxis]

def calculate_velocity_fluctuations(velocities):

mean_velocity = np.mean(velocities, axis=0)
return velocities - mean_velocity

def calculate_minimum_distance(p1, p2, box_size):

delta = np.abs(p1 - p2)
delta = np.minimum(delta, box_size - delta)
return np.sqrt(np.sum(delta**2))

def calculate_pairwise_distances(positions, box_size):

n_boids = positions.shape[0]
distances = np.zeros((n_boids, n_boids))
for i in range(n_boids):

for j in range(i+1, n_boids):

dist = calculate_minimum_distance(positions[i], positions[j],

box_size)

64

distances[i, j] = dist
distances[j, i] = dist

return distances

def calculate_velocity_correlations(positions, velocities, box_size,

n_bins=20, max_distance=None):
n_boids = positions.shape[0]
vel_fluct = calculate_velocity_fluctuations(velocities)
normalization = np.mean(np.sum(vel_fluct**2, axis=1))
dists = calculate_pairwise_distances(positions, box_size)
corr_matrix = np.zeros((n_boids, n_boids))
for i in range(n_boids):

for j in range(n_boids):

if i != j:

corr_matrix[i, j] = np.dot(vel_fluct[i], vel_fluct[j]) /

normalization
if max_distance is None:

max_distance = np.min(box_size) / 2.0
bins = np.linspace(0, max_distance, n_bins+1)
bin_centers = 0.5*(bins[1:] + bins[:-1])
binned_corr = np.zeros(n_bins)
counts = np.zeros(n_bins)
for i in range(n_boids):

for j in range(i+1, n_boids):

dist = dists[i, j]
if dist <= max_distance:

bin_idx = np.digitize(dist, bins) - 1
if 0 <= bin_idx < n_bins:

binned_corr[bin_idx] += corr_matrix[i, j]
counts[bin_idx] += 1

valid = counts > 0
binned_corr[valid] /= counts[valid]
return bin_centers, binned_corr

def fit_correlation_function(distances, correlations,

fit_type=’exponential’):
valid = np.isfinite(correlations) & (correlations > 0)
if np.sum(valid) < 3:
return None, None
x_data = distances[valid]
y_data = correlations[valid]
if fit_type == ’exponential’:

def exp_func(r, A, xi):

return A * np.exp(-r/xi)

try:

params, _ = curve_fit(exp_func, x_data, y_data, p0=[1.0,

np.mean(x_data)])

A, xi = params
fitted_values = exp_func(distances, A, xi)
return {’A’: A, ’correlation_length’: xi}, fitted_values

except Exception as e:

65

print("Exponential fit failed:", e)
return None, None

elif fit_type == ’power_law’:

def power_func(r, A, alpha):

return A * np.power(r, -alpha)

non_zero = x_data > 0
if np.sum(non_zero) < 3:
return None, None

try:

params, _ = curve_fit(power_func, x_data[non_zero],

y_data[non_zero], p0=[1.0, 0.5])

A, alpha = params
fitted_values = power_func(distances, A, alpha)
return {’A’: A, ’alpha’: alpha}, fitted_values

except Exception as e:

print("Power law fit failed:", e)
return None, None

else:

raise ValueError("Unknown fit type. Choose ’exponential’ or

’power_law’.")

def calculate_correlation_map(positions, velocities, box_size,

resolution=50):
if isinstance(box_size, (int, float)):

box_size = np.array([box_size, box_size])
norm_velocities = normalize_velocities(velocities)
x = np.linspace(0, box_size[0], resolution)
y = np.linspace(0, box_size[1], resolution)
X, Y = np.meshgrid(x, y)
grid_points = np.column_stack((X.flatten(), Y.flatten()))
correlation_map = np.zeros((resolution, resolution))
tree = cKDTree(positions)
for i, point in enumerate(grid_points):
_, idx = tree.query(point, k=1)
ref_velocity = norm_velocities[idx]
corr_values = np.sum(norm_velocities * ref_velocity, axis=1)
correlation_map[i // resolution, i % resolution] =

np.mean(corr_values)
return correlation_map

def set_plot_style():

plt.rcParams.update({

"figure.facecolor": "white",
"axes.facecolor": "white",
"axes.edgecolor": "black",
"axes.linewidth": 1.2,
"axes.titlesize": 14,
"axes.labelsize": 12,
"xtick.labelsize": 10,
"ytick.labelsize": 10,
"font.family": "sans-serif",

66

"grid.linestyle": ""

})

def plot_correlation_function(distances, correlations, fit_params,

fitted_curve, noise_level):
set_plot_style()
plt.figure()
plt.plot(distances, correlations, ’bo-’, markersize=4, linewidth=1.5,
label=’Data’)
if fit_params is not None:

if ’correlation_length’ in fit_params:

label_str = f"Exponential Fit: =
{fit_params[’correlation_length’]:.2f} m"

elif ’alpha’ in fit_params:

label_str = f"Power Law Fit: = {fit_params[’alpha’]:.2f}"

plt.plot(distances, fitted_curve, ’r--’, linewidth=1.5,

label=label_str)
plt.xlabel(’Distance r (m)’)
plt.ylabel(’Velocity Correlation C(r)’)
plt.title(f’Correlation Function (Noise: {noise_level:.2f})’)
plt.legend()
plt.tight_layout()
plt.show()

def plot_loglog(distances, correlations):

set_plot_style()
plt.figure()
valid = (correlations > 0) & (distances > 0)
if np.any(valid):

plt.loglog(distances[valid], correlations[valid], ’bo-’,

markersize=4, linewidth=1.5)
plt.xlabel(’Distance r (m)’)
plt.ylabel(’Velocity Correlation C(r)’)
plt.title(’Log-Log Plot of Correlation Function’)
plt.tight_layout()
plt.show()

def plot_spatial_map(corr_map, box_size):

set_plot_style()
plt.figure()
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
img = plt.imshow(corr_map, origin=’lower’, extent=[0, box_size[0], 0,
box_size[1]],

cmap=’RdBu_r’, norm=norm)

cbar = plt.colorbar(img)
cbar.set_label(’Velocity Correlation’, fontsize=12)
plt.xlabel(’X (m)’)
plt.ylabel(’Y (m)’)
plt.title(’Spatial Correlation Map’)
plt.tight_layout()
plt.show()

67

def plot_boid_snapshot(positions, velocities, box_size):

set_plot_style()
plt.figure()
plt.scatter(positions[:, 0], positions[:, 1], s=15, color=’blue’,
alpha=0.7)
stride = max(1, positions.shape[0] // 200)
plt.quiver(positions[::stride, 0], positions[::stride, 1],

velocities[::stride, 0], velocities[::stride, 1],
color=’red’, scale=20, width=0.003)

plt.xlabel(’X (m)’)
plt.ylabel(’Y (m)’)
plt.title(’Boid Snapshot with Velocity Vectors’)
plt.xlim(0, box_size[0])
plt.ylim(0, box_size[1])
plt.tight_layout()
plt.show()

def main():

parser = argparse.ArgumentParser(description="Boids Correlation
Analysis")
parser.add_argument(’--input’, type=str, required=True, help="Input
simulation NPZ file")
parser.add_argument(’--bins’, type=int, default=20, help="Number of
distance bins")
parser.add_argument(’--fit’, type=str, default=’exponential’,
choices=[’exponential’, ’power_law’])
args = parser.parse_args()
positions, velocities, metadata = load_simulation_data(args.input)
box_size = np.array([100.0, 100.0])
noise_level = metadata[’noise_level’]
last_positions = positions[-1]
last_velocities = velocities[-1]
distances, correlations =
calculate_velocity_correlations(last_positions, last_velocities,
box_size, n_bins=args.bins)
fit_params, fitted_curve = fit_correlation_function(distances,
correlations, fit_type=args.fit)
corr_map = calculate_correlation_map(last_positions, last_velocities,
box_size)
plot_correlation_function(distances, correlations, fit_params,
fitted_curve, noise_level)
plot_loglog(distances, correlations)
plot_spatial_map(corr_map, box_size)
plot_boid_snapshot(last_positions, last_velocities, box_size)

if __name__ == ’__main__’:

main()

68