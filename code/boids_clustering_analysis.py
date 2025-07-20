#!/usr/bin/env python
"""
boids_clustering_analysis.py

Runs a boids simulation with periodic boundaries, applies DBSCAN clustering

with a custom

distance metric (accounting for periodic boundaries), and generates

publication-quality snapshots

with velocity directions, color-coded by cluster.
Also computes cluster size distributions, largest cluster fraction, and

number of clusters vs noise.

Usage: python boids_clustering_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
from boids_simulation import BoidsSimulation, run_simulation

plt.style.use(’default’)
plt.rcParams.update({’figure.figsize’: (10, 8), ’font.size’: 12,

’axes.grid’: False, ’savefig.dpi’: 300})

def distance(p1, p2, width, height):

dx = abs(p1[0] - p2[0])
dy = abs(p1[1] - p2[1])
dx = min(dx, width - dx)
dy = min(dy, height - dy)
return np.sqrt(dx**2 + dy**2)

def cluster_boids(positions, width, height, perception_radius,

min_samples=3):
eps = perception_radius * 1.2
custom_metric = lambda a, b: distance(a, b, width, height)
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=custom_metric)
labels = dbscan.fit_predict(positions)
return labels

def compute_cluster_metrics(labels, total_boids):

label_counts = Counter(label for label in labels if label != -1)
if label_counts:

sizes = np.array(list(label_counts.values()))
largest_fraction = sizes.max() / total_boids
n_clusters = len(label_counts)

else:

sizes = np.array([])
largest_fraction = 0
n_clusters = 0

69

return sizes, largest_fraction, n_clusters

def plot_boids_snapshot(positions, velocities, labels, noise, sim_width,

sim_height):
fig, ax = plt.subplots()
unique_labels = np.unique(labels)
n_clusters = len(unique_labels[unique_labels != -1])
n_colors = max(n_clusters, 1)
cmap = plt.get_cmap(’tab10’, n_colors)
colors = np.array([cmap(i) for i in range(n_colors)])
color_map = {label: colors[i] for i, label in
enumerate(sorted(unique_labels[unique_labels != -1]))}
color_map[-1] = (0.5, 0.5, 0.5, 1)
boid_colors = [color_map[label] for label in labels]
ax.scatter(positions[:, 0], positions[:, 1], c=boid_colors, s=30,
edgecolor=’k’, zorder=2)
for pos, vel in zip(positions, velocities):

norm = np.linalg.norm(vel)
arrow = vel / norm * 2.0 if norm > 0 else np.zeros_like(vel)
ax.arrow(pos[0], pos[1], arrow[0], arrow[1], head_width=0.8,

head_length=1.0, fc=’k’, ec=’k’, alpha=0.8, zorder=3)
ax.set_title(f"Noise = {noise:.2f} | Clusters: {n_clusters}")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_xlim(0, sim_width)
ax.set_ylim(0, sim_height)
ax.set_aspect(’equal’)
return fig

def run_and_analyze(noise_values, num_steps=300, num_boids=200):

snapshot_data = []
for noise in noise_values:

positions_history, velocities_history, sim_instance =

run_simulation(num_steps=num_steps, num_boids=num_boids, solver=’rk4’,
noise_level=noise)

positions = sim_instance.positions.copy()
velocities = sim_instance.velocities.copy()
labels = cluster_boids(positions, sim_instance.width,

sim_instance.height, sim_instance.perception_radius, min_samples=3)

sizes, largest_fraction, n_clusters =
compute_cluster_metrics(labels, num_boids)

snapshot_data.append({
’noise’: noise,
’positions’: positions,
’velocities’: velocities,
’labels’: labels,
’cluster_sizes’: sizes,
’largest_fraction’: largest_fraction,
’n_clusters’: n_clusters,
’sim_width’: sim_instance.width,
’sim_height’: sim_instance.height

70

})

return snapshot_data

def plot_snapshots(snapshot_data):
for data in snapshot_data:

fig = plot_boids_snapshot(data[’positions’], data[’velocities’],

data[’labels’],

data[’sim_height’])

data[’noise’], data[’sim_width’],

fig.suptitle("Boids Snapshot: Clustering & Velocity Directions",

fontsize=16)

filename = f"boids_snapshot_eta_{data[’noise’]:.2f}.png"
fig.savefig(filename)
print(f"Generated: {filename}")
plt.show()

def plot_cluster_size_distributions(snapshot_data):

plt.figure()
for data in snapshot_data:

if data[’cluster_sizes’].size > 0:

fractions = data[’cluster_sizes’] /
float(np.sum(data[’cluster_sizes’]) + 1e-6)

plt.hist(fractions, bins=20, alpha=0.5, label=f’ =

{data["noise"]:.2f}’, density=True)
plt.xlabel("Cluster Size (Fraction of Population)")
plt.ylabel("Frequency Density")
plt.title("Cluster Size Distributions vs Noise")
plt.legend()
plt.savefig("cluster_size_distributions.png")
print("Generated: cluster_size_distributions.png")
plt.show()

def plot_cluster_metrics(snapshot_data):

noise_vals = [d[’noise’] for d in snapshot_data]
largest_fracs = [d[’largest_fraction’] for d in snapshot_data]
n_clusters = [d[’n_clusters’] for d in snapshot_data]
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(noise_vals, largest_fracs, marker=’o’, linestyle=’-’)
ax1.set_xlabel("Noise ")
ax1.set_ylabel("Largest Cluster Fraction")
ax1.set_title("Largest Cluster Fraction vs Noise")
ax2.plot(noise_vals, n_clusters, marker=’o’, linestyle=’-’,
color=’orange’)
ax2.set_xlabel("Noise ")
ax2.set_ylabel("Number of Clusters")
ax2.set_title("Number of Clusters vs Noise")
plt.tight_layout()
plt.savefig("cluster_metrics_vs_noise.png")
print("Generated: cluster_metrics_vs_noise.png")
plt.show()

71

def main():

noise_values = [0.1, 0.5, 1.0, 1.5, 2.0]
print("Running boids simulation and clustering analysis for noise
values:", noise_values)
snapshot_data = run_and_analyze(noise_values=noise_values,
num_steps=300, num_boids=200)
plot_snapshots(snapshot_data)
plot_cluster_size_distributions(snapshot_data)
plot_cluster_metrics(snapshot_data)
print("Clustering analysis complete!")

if __name__ == ’__main__’:

main()

72