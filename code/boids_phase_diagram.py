#!/usr/bin/env python
"""
boids_phase_diagram.py

Sweeps over noise level () and perception radius (R) in a BoidsSimulation to

compute

the mean order parameter (). Plots a 2D heatmap with contour lines

highlighting the order/disorder boundary.

Usage: python boids_phase_diagram.py
"""

import numpy as np
import matplotlib.pyplot as plt
from boids_simulation import BoidsSimulation

plt.style.use(’default’)
plt.rcParams.update({’figure.figsize’: (8, 6), ’font.size’: 12, ’axes.grid’:

False, ’savefig.dpi’: 300})

def calculate_order_parameter(velocities):

speeds = np.linalg.norm(velocities, axis=1)
speeds[speeds < 1e-10] = 1e-10
norm_vel = velocities / speeds[:, np.newaxis]
return np.linalg.norm(np.mean(norm_vel, axis=0))

def run_simulation(noise, radius, num_steps=500, num_boids=200,

warmup_fraction=0.8):
sim = BoidsSimulation(num_boids=num_boids, solver=’rk4’,
noise_level=noise)
sim.perception_radius = radius
warmup_steps = int(num_steps * warmup_fraction)
order_history = []
for step in range(num_steps):

sim.update()
if step >= warmup_steps:

order_history.append(calculate_order_parameter(sim.velocities))

return np.mean(order_history)

def sweep_parameters(noise_values, radius_values, num_steps=500,

num_boids=200, warmup_fraction=0.8, n_runs=3):
order_map = np.zeros((len(radius_values), len(noise_values)))

62

for i, R in enumerate(radius_values):

for j, eta in enumerate(noise_values):

orders = []
for _ in range(n_runs):

orders.append(run_simulation(eta, R, num_steps, num_boids,

warmup_fraction))

order_map[i, j] = np.mean(orders)
print(f"R={R:.2f}, ={eta:.2f} -> ={order_map[i,j]:.3f}")

return order_map

def plot_phase_diagram(noise_values, radius_values, order_map,

contour_levels=[0.5]):
Rg, Ng = np.meshgrid(noise_values, radius_values)
plt.figure()
im = plt.imshow(order_map, origin=’lower’, aspect=’auto’,

extent=[noise_values[0], noise_values[-1],

radius_values[0], radius_values[-1]],

cmap=’viridis’)

cbar = plt.colorbar(im)
cbar.set_label(’Order Parameter ()’)
CS = plt.contour(Ng, Rg, order_map, levels=contour_levels,
colors=’white’, linewidths=2)
plt.clabel(CS, inline=True, fontsize=10, fmt=’ = %.2f’)
plt.xlabel("Noise Level ()")
plt.ylabel("Perception Radius (m)")
plt.title("Boids Phase Diagram: Order vs. Noise & Radius")
plt.tight_layout()
plt.savefig("boids_phase_diagram.pdf")
plt.close()
print("Saved phase diagram: boids_phase_diagram.pdf")

def main():

noise_values = np.linspace(0.0, 2.0, 9)
radius_values = np.linspace(5.0, 20.0, 7)
num_steps = 500
num_boids = 200
order_map = sweep_parameters(noise_values, radius_values, num_steps,
num_boids, warmup_fraction=0.8, n_runs=3)
plot_phase_diagram(noise_values, radius_values, order_map,
contour_levels=[0.5])

if __name__ == "__main__":

main()