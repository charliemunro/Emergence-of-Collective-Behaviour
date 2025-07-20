#!/usr/bin/env python
"""
boids_phase_transition_analysis.py

Runs boids simulations over a range of noise levels, computes the global

order parameter,

fits the data to a sigmoid, and generates publication-quality plots.
Usage: python boids_phase_transition_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from boids_simulation import BoidsSimulation
import argparse

plt.style.use(’default’)
plt.rcParams[’figure.figsize’] = (10, 6)
plt.rcParams[’font.size’] = 12

def calculate_order_parameter(velocities):

speeds = np.linalg.norm(velocities, axis=1)

54

speeds[speeds < 1e-10] = 1e-10
normalized_vels = velocities / speeds[:, np.newaxis]
return np.linalg.norm(np.mean(normalized_vels, axis=0))

def run_one_simulation(noise, num_steps=800, num_boids=300,

warmup_fraction=0.8):
sim = BoidsSimulation(num_boids=num_boids, solver=’rk4’,
noise_level=noise)
order_history = []
warmup_steps = int(num_steps * warmup_fraction)
for step in range(num_steps):

sim.update()
if step >= warmup_steps:

order_history.append(calculate_order_parameter(sim.velocities))

return np.mean(order_history)

def gather_phase_data(noise_values, num_steps=800, num_boids=300, n_runs=20):

order_means = []
order_stds = []
for noise in noise_values:

orders = [run_one_simulation(noise, num_steps, num_boids) for _ in

range(n_runs)]

order_means.append(np.mean(orders))
order_stds.append(np.std(orders))
print(f"Noise={noise:.3f}: Mean Order={np.mean(orders):.3f},

Std={np.std(orders):.3f}")
return np.array(order_means), np.array(order_stds)

def fit_sigmoid(noise_values, order_means):

def sigmoid(x, x0, k, a, b):

return a + (b - a) / (1.0 + np.exp(-k * (x - x0)))

p0 = [np.median(noise_values), 10.0, 0.0, 1.0]
try:

popt, _ = curve_fit(sigmoid, noise_values, order_means, p0=p0,

maxfev=10000)

critical_noise = popt[0]
return sigmoid, popt, critical_noise

except Exception as e:

print(f"Sigmoid fit failed: {e}")
return None, None, None

def plot_phase_transition(noise_values, order_means, order_stds,
sigmoid_func, popt, R=10.0, savefig=’phase_transition.pdf’):
plt.figure()
plt.errorbar(noise_values, order_means, yerr=order_stds, fmt=’o’,
color=’blue’, capsize=4, label=’Simulation data’)
if sigmoid_func is not None and popt is not None:

x_fit = np.linspace(min(noise_values), max(noise_values), 300)
y_fit = sigmoid_func(x_fit, *popt)
plt.plot(x_fit, y_fit, ’r-’, label=f’Fit (_c {popt[0]:.3f})’)

55

plt.axvline(x=popt[0], color=’r’, linestyle=’--’, label=f’Critical

noise (_c {popt[0]:.3f})’)
plt.xlabel(’Noise level ()’)
plt.ylabel(’Order Parameter ()’)
plt.title(f’Order Parameter vs. Noise (R = {R:.1f} m)’)
plt.legend()
plt.tight_layout()
plt.savefig(savefig, dpi=300)
plt.close()
print(f"Saved phase transition plot as ’{savefig}’")

def plot_time_series_for_noises(example_noises, num_steps=300,

num_boids=200, savefig=’order_time_series.pdf’):
plt.figure()
colors = plt.cm.viridis(np.linspace(0, 1, len(example_noises)))
for i, noise in enumerate(example_noises):

sim = BoidsSimulation(num_boids=num_boids, solver=’rk4’,

noise_level=noise)

order_history = []
for step in range(num_steps):

sim.update()
order_history.append(calculate_order_parameter(sim.velocities))

plt.plot(np.arange(num_steps), order_history, color=colors[i],

label=f’Noise = {noise:.2f}’)
plt.xlabel(’Time Steps’)
plt.ylabel(’Order Parameter ()’)
plt.title(’Order Parameter vs. Time for Selected Noise Levels’)
plt.legend()
plt.tight_layout()
plt.savefig(savefig, dpi=300)
plt.close()
print(f"Saved time-series plot as ’{savefig}’")

def generate_snapshots(selected_noises, num_steps=300, num_boids=200):

for noise in selected_noises:

sim = BoidsSimulation(num_boids=num_boids, solver=’rk4’,

noise_level=noise)

for _ in range(num_steps):

sim.update()

plt.figure()
plt.scatter(sim.positions[:, 0], sim.positions[:, 1], s=20,

color=’#1f77b4’, alpha=0.7)

scale = 2.0
for i in range(num_boids):

plt.arrow(sim.positions[i, 0], sim.positions[i, 1],

sim.velocities[i, 0]*scale, sim.velocities[i, 1]*scale,

head_width=0.8, head_length=1.0, fc=’red’, ec=’red’,

alpha=0.5)

order = calculate_order_parameter(sim.velocities)
plt.xlim(0, sim.width)
plt.ylim(0, sim.height)

56

plt.title(f"Snapshot (Noise = {noise:.2f}, Order
plt.xlabel(’X Position (m)’)
plt.ylabel(’Y Position (m)’)
plt.tight_layout()
filename = f’snapshot_noise_{noise:.2f}.pdf’
plt.savefig(filename, dpi=300)
plt.close()
print(f"Saved snapshot as ’{filename}’")

= {order:.2f})")

def main():

noise_values = np.linspace(0.0, 1.2, 16)
num_steps = 800
num_boids = 300
n_runs = 20
print("Gathering phase transition data...")
order_means, order_stds = gather_phase_data(noise_values, num_steps,
num_boids, n_runs)
sigmoid_func, popt, critical_noise = fit_sigmoid(noise_values,
order_means)
plot_phase_transition(noise_values, order_means, order_stds,
sigmoid_func, popt, R=10.0, savefig=’phase_transition.pdf’)
example_noises = [0.1, 0.5, 1.0]
plot_time_series_for_noises(example_noises, num_steps=300,
num_boids=200, savefig=’order_time_series.pdf’)
snapshot_noises = [0.1, 0.5, 1.0]
generate_snapshots(snapshot_noises, num_steps=300, num_boids=200)
print("Phase Transition Analysis complete!")

if __name__ == ’__main__’:

main()

B.3

solver comparison.py

#!/usr/bin/env python
"""
solver_comparison.py

Compares numerical solvers (Euler, RK4, Verlet) in the Boids simulation.
Generates plots for order parameter evolution, position drift relative to

RK4, boid trajectories,

convergence vs. time step size, and computational efficiency.
Usage: python solver_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from boids_simulation import BoidsSimulation, run_simulation

57

plt.style.use(’default’)
plt.rcParams[’figure.figsize’] = (10, 8)
plt.rcParams[’font.size’] = 12

def calculate_order_parameter(velocities):

normalized_velocities = velocities / (np.linalg.norm(velocities,
axis=1)[:, np.newaxis] + 1e-10)
return np.linalg.norm(np.mean(normalized_velocities, axis=0))

def plot_order_parameter_time(num_steps=500, num_boids=200):

solvers = [’euler’, ’rk4’, ’verlet’]
colors = [’#1f77b4’, ’#ff7f0e’, ’#2ca02c’]
plt.figure()
for i, solver in enumerate(solvers):

positions, velocities, _ = run_simulation(num_steps=num_steps,

num_boids=num_boids, solver=solver)

order_parameters =

np.array([calculate_order_parameter(velocities[step]) for step in
range(num_steps)])

plt.plot(np.arange(num_steps), order_parameters, color=colors[i],

label=solver.upper())
plt.xlabel(’Time Steps’)
plt.ylabel(’Order Parameter ()’)
plt.title(’Evolution of Order Parameter by Solver’)
plt.legend()
plt.tight_layout()
plt.savefig(’order_parameter_vs_time.png’, dpi=300)
plt.close()
print("Generated: order_parameter_vs_time.png")

def plot_position_drift(num_steps=200, num_boids=200,

reference_solver=’rk4’):
comparison_solvers = [’euler’, ’verlet’]
colors = [’#1f77b4’, ’#2ca02c’]
plt.figure()
ref_positions, _, _ = run_simulation(num_steps=num_steps,
num_boids=num_boids, solver=reference_solver)
for i, solver in enumerate(comparison_solvers):

positions, _, _ = run_simulation(num_steps=num_steps,

num_boids=num_boids, solver=solver)
drift = np.zeros(num_steps)
for step in range(num_steps):

drift[step] = np.mean(np.linalg.norm(positions[step] -

ref_positions[step], axis=1))

plt.plot(np.arange(num_steps), drift, color=colors[i],

label=f’{solver.upper()} vs {reference_solver.upper()}’)
plt.xlabel(’Time Steps’)
plt.ylabel(’Mean Position Drift (meters)’)
plt.title(f’Position Drift Relative to {reference_solver.upper()}’)
plt.legend()
plt.tight_layout()

58

plt.savefig(’position_drift.png’, dpi=300)
plt.close()
print("Generated: position_drift.png")

def plot_boid_trajectories(num_steps=200, num_boids=200, boid_idx=0):

solvers = [’euler’, ’rk4’, ’verlet’]
colors = [’#1f77b4’, ’#ff7f0e’, ’#2ca02c’]
markers = [’o’, ’s’, ’^’]
plt.figure()
width = 100.0
height = 100.0
for i, solver in enumerate(solvers):

np.random.seed(42)
positions, _, _ = run_simulation(num_steps=num_steps,

num_boids=num_boids, solver=solver)

trajectory = positions[:, boid_idx]
for j in range(1, len(trajectory)):

dx = trajectory[j, 0] - trajectory[j-1, 0]
dy = trajectory[j, 1] - trajectory[j-1, 1]
if abs(dx) > width/2 or abs(dy) > height/2:

continue

plt.plot([trajectory[j-1, 0], trajectory[j, 0]],

[trajectory[j-1, 1], trajectory[j, 1]], color=colors[i], linewidth=1.5,
alpha=0.8)

plt.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i],

marker=markers[i], s=100, label=f’{solver.upper()} Start’)

plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[i],

marker=markers[i], facecolors=’none’, s=100, label=f’{solver.upper()}
End’)
plt.xlabel(’X Position (meters)’)
plt.ylabel(’Y Position (meters)’)
plt.title(f’Trajectory of Boid #{boid_idx} by Solver’)
plt.legend()
plt.tight_layout()
plt.savefig(’boid_trajectories.png’, dpi=300)
plt.close()
print("Generated: boid_trajectories.png")

def plot_order_parameter_vs_dt(dt_values, num_steps=100, num_boids=200):

solvers = [’euler’, ’rk4’, ’verlet’]
colors = [’#1f77b4’, ’#ff7f0e’, ’#2ca02c’]
markers = [’o’, ’s’, ’^’]
plt.figure()
for i, solver in enumerate(solvers):

order_values = []
order_errors = []
for dt in dt_values:

n_runs = 5
run_orders = []
for run in range(n_runs):

59

boids = BoidsSimulation(num_boids=num_boids, solver=solver,

time_step=dt)

final_orders = []
for _ in range(num_steps):

boids.update()
if _ >= int(0.8 * num_steps):

final_orders.append(calculate_order_parameter(boids.velocities))
run_orders.append(np.mean(final_orders))

order_values.append(np.mean(run_orders))
order_errors.append(np.std(run_orders))

plt.errorbar(dt_values, order_values, yerr=order_errors,

color=colors[i], label=solver.upper(), marker=markers[i], linestyle=’-’,
capsize=4)
plt.xlabel(’Time Step Size (seconds)’)
plt.ylabel(’Order Parameter ()’)
plt.title(’Convergence of Order Parameter with Time Step Size’)
plt.legend()
plt.tight_layout()
plt.savefig(’order_vs_dt.png’, dpi=300)
plt.close()
print("Generated: order_vs_dt.png")

def plot_drift_vs_dt(dt_values, num_steps=100, num_boids=200,

reference_dt=0.01):
solvers = [’euler’, ’rk4’, ’verlet’]
colors = [’#1f77b4’, ’#ff7f0e’, ’#2ca02c’]
markers = [’o’, ’s’, ’^’]
plt.figure()
for i, solver in enumerate(solvers):

drift_values = []
drift_errors = []
np.random.seed(42)
boids_ref = BoidsSimulation(num_boids=num_boids, solver=solver,

time_step=reference_dt)

ref_steps = int(num_steps * (max(dt_values) / reference_dt))
for _ in range(ref_steps):
boids_ref.update()

ref_positions = boids_ref.positions.copy()
for dt in dt_values:

n_runs = 5
run_drifts = []
for run in range(n_runs):
np.random.seed(42)
boids = BoidsSimulation(num_boids=num_boids, solver=solver,

time_step=dt)

steps = int(num_steps * (max(dt_values) / dt))
for _ in range(steps):
boids.update()

drift = np.mean(np.linalg.norm(boids.positions -

ref_positions, axis=1))

60

run_drifts.append(drift)

drift_values.append(np.mean(run_drifts))
drift_errors.append(np.std(run_drifts))

plt.errorbar(dt_values, drift_values, yerr=drift_errors,

color=colors[i], label=solver.upper(), marker=markers[i], linestyle=’-’,
capsize=4)
plt.xlabel(’Time Step Size (seconds)’)
plt.ylabel(’Mean Position Drift (meters)’)
plt.title(f’Position Drift vs Time Step Size (Relative to
dt={reference_dt}s)’)
plt.legend()
plt.tight_layout()
plt.savefig(’drift_vs_dt.png’, dpi=300)
plt.close()
print("Generated: drift_vs_dt.png")

def plot_computation_time(dt_values, num_steps=100, num_boids=200):

solvers = [’euler’, ’rk4’, ’verlet’]
colors = [’#1f77b4’, ’#ff7f0e’, ’#2ca02c’]
markers = [’o’, ’s’, ’^’]
plt.figure()
for i, solver in enumerate(solvers):

time_values = []
for dt in dt_values:

boids = BoidsSimulation(num_boids=num_boids, solver=solver,

time_step=dt)

start_time = time.time()
for _ in range(num_steps):

boids.update()
end_time = time.time()
computation_time = end_time - start_time
time_values.append(computation_time)

plt.plot(dt_values, time_values, color=colors[i],

label=solver.upper(), marker=markers[i], linestyle=’-’)
plt.xlabel(’Time Step Size (seconds)’)
plt.ylabel(’Computation Time (seconds)’)
plt.title(’Computational Efficiency vs Time Step Size’)
plt.legend()
plt.tight_layout()
plt.savefig(’computation_time.png’, dpi=300)
plt.close()
print("Generated: computation_time.png")

def main():

print("Running Solver Comparison Analysis...")
plot_order_parameter_time(num_steps=500, num_boids=200)
plot_position_drift(num_steps=200, num_boids=200)
plot_boid_trajectories(num_steps=200, num_boids=200)
dt_values = [0.01, 0.05, 0.1, 0.2, 0.5]
plot_order_parameter_vs_dt(dt_values, num_steps=100, num_boids=200)
plot_drift_vs_dt(dt_values, num_steps=100, num_boids=200)

61

plot_computation_time(dt_values, num_steps=100, num_boids=200)
print("Solver Comparison Analysis complete!")

if __name__ == ’__main__’:

main()