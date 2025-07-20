#!/usr/bin/env python
"""
boids_simulation.py

Implements boid dynamics using Euler, RK4, or Velocity Verlet solvers.
Includes noise injection, periodic boundaries, and computes the global order

parameter.

Usage: python boids_simulation.py --num_steps 100 --num_boids 500 --solver

rk4 --noise_level 0.1 --output simulation_output.npz

"""

import numpy as np
from scipy.spatial import cKDTree
import argparse

class BoidsSimulation:

def __init__(self, num_boids=500, width=100.0, height=100.0,
time_step=0.1, solver=’rk4’, noise_level=0.1):

self.num_boids = num_boids
self.width = width
self.height = height
self.time_step = time_step
self.solver = solver
self.noise_level = noise_level
self.positions = np.random.rand(num_boids, 2) * np.array([width,

height])

self.velocities = np.random.randn(num_boids, 2)
self.accelerations = np.zeros_like(self.velocities)
self.max_speed = 3.0
self.perception_radius = 10.0
self.separation_weight = 1.5
self.alignment_weight = 1.0
self.cohesion_weight = 1.0

def find_neighbors(self, positions):

tree = cKDTree(positions)
neighbors = tree.query_ball_point(positions,

r=self.perception_radius)
return neighbors

def separation(self, positions, velocities, neighbors=None):

N = positions.shape[0]
diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
dist_sq = np.sum(diff**2, axis=2)

51

np.fill_diagonal(dist_sq, np.inf)
mask = dist_sq <= self.perception_radius**2
denom = np.maximum(dist_sq, 1e-6)
factor = np.where(mask, 1.0 / denom, 0.0)
separations = np.sum(diff * factor[:, :, np.newaxis], axis=1)
return separations

def alignment(self, positions, velocities, neighbors):

alignments = np.zeros_like(velocities)
for i in range(len(positions)):

boid_neighbors = [n for n in neighbors[i] if n != i]
if boid_neighbors:

alignments[i] = np.mean(velocities[boid_neighbors], axis=0)

return alignments

def cohesion(self, positions, velocities, neighbors):

cohesions = np.zeros_like(positions)
for i in range(len(positions)):

boid_neighbors = [n for n in neighbors[i] if n != i]
if boid_neighbors:

cohesions[i] = np.mean(positions[boid_neighbors], axis=0) -

positions[i]

return cohesions

def compute_acceleration(self, positions, velocities):

neighbors = self.find_neighbors(positions)
sep = self.separation(positions, velocities, neighbors)
ali = self.alignment(positions, velocities, neighbors)
coh = self.cohesion(positions, velocities, neighbors)
acceleration = (sep * self.separation_weight +

ali * self.alignment_weight +
coh * self.cohesion_weight)

return acceleration

def update(self):

if self.solver == ’euler’:

acc = self.compute_acceleration(self.positions, self.velocities)
self.velocities += acc * self.time_step
self.positions += self.velocities * self.time_step

elif self.solver == ’rk4’:

def f(pos, vel):

acc = self.compute_acceleration(pos, vel)
return vel, acc

k1_pos, k1_vel = f(self.positions, self.velocities)
k2_pos = self.positions + 0.5 * self.time_step * k1_pos
k2_vel = self.velocities + 0.5 * self.time_step * k1_vel
k2_pos, k2_vel = f(k2_pos, k2_vel)
k3_pos = self.positions + 0.5 * self.time_step * k2_pos
k3_vel = self.velocities + 0.5 * self.time_step * k2_vel
k3_pos, k3_vel = f(k3_pos, k3_vel)
k4_pos = self.positions + self.time_step * k3_pos

52

k4_vel = self.velocities + self.time_step * k3_vel
k4_pos, k4_vel = f(k4_pos, k4_vel)
self.positions += self.time_step * (k1_pos + 2*k2_pos + 2*k3_pos

+ k4_pos) / 6

self.velocities += self.time_step * (k1_vel + 2*k2_vel +

2*k3_vel + k4_vel) / 6

elif self.solver == ’verlet’:

current_acc = self.compute_acceleration(self.positions,

self.velocities)

self.positions += self.velocities * self.time_step + 0.5 *

self.accelerations * self.time_step**2

new_acc = self.compute_acceleration(self.positions,

self.velocities)

self.velocities += 0.5 * (self.accelerations + new_acc) *

self.time_step

self.accelerations = new_acc

if self.noise_level > 0:

self.velocities += np.random.randn(*self.velocities.shape) *

self.noise_level

speeds = np.linalg.norm(self.velocities, axis=1)
over_max = speeds > self.max_speed
if np.any(over_max):

self.velocities[over_max] *= (self.max_speed /

speeds[over_max])[:, np.newaxis]

self.positions %= np.array([self.width, self.height])
return self.positions

def calculate_order_parameter(self):

speeds = np.linalg.norm(self.velocities, axis=1)
speeds[speeds == 0] = 1.0
norm_vel = self.velocities / speeds[:, np.newaxis]
phi = np.linalg.norm(np.mean(norm_vel, axis=0))
return phi

def run_simulation(num_steps=100, num_boids=500, solver=’rk4’,

noise_level=0.1):
sim = BoidsSimulation(num_boids=num_boids, solver=solver,
noise_level=noise_level)
positions_history = np.zeros((num_steps, num_boids, 2))
velocities_history = np.zeros((num_steps, num_boids, 2))
for step in range(num_steps):

positions_history[step] = sim.positions.copy()
velocities_history[step] = sim.velocities.copy()
sim.update()

return positions_history, velocities_history, sim

def main():

parser = argparse.ArgumentParser(description="Run Boids Simulation")
parser.add_argument(’--num_steps’, type=int, default=100)
parser.add_argument(’--num_boids’, type=int, default=500)

53

parser.add_argument(’--solver’, type=str, default=’rk4’,
choices=[’euler’, ’rk4’, ’verlet’])
parser.add_argument(’--noise_level’, type=float, default=0.1)
parser.add_argument(’--output’, type=str,
default=’simulation_output.npz’)
args = parser.parse_args()
print(f"Running simulation with {args.num_boids} boids for
{args.num_steps} steps using {args.solver.upper()} solver...")
positions, velocities, sim_instance =
run_simulation(num_steps=args.num_steps, num_boids=args.num_boids,
solver=args.solver, noise_level=args.noise_level)
np.savez(args.output, positions=positions, velocities=velocities,

num_boids=args.num_boids, num_steps=args.num_steps,
solver=args.solver, noise_level=args.noise_level)

final_order_param = sim_instance.calculate_order_parameter()
avg_speed = np.mean(np.linalg.norm(velocities, axis=2))
max_speed = np.max(np.linalg.norm(velocities, axis=2))
print(f"Simulation complete. Results saved to {args.output}")
print(f"Average speed: {avg_speed:.2f} m/s")
print(f"Maximum speed: {max_speed:.2f} m/s")
print(f"Final order parameter (): {final_order_param:.2f}")

if __name__ == ’__main__’:

main()