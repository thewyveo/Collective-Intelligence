"""
Baseline Lotka-Volterra predator-prey model (discrete-time, agent-based).

Implements the classical (mathematical) Lotka-Volterra equations WITHOUT energy, movement, spatiality, or perception.
- Prey reproduce exponentially each frame (α · N).
- Predators die at a constant rate (γ · P).
- Predation occurs at a rate proportional to N·P (β · N · P).
- Predator reproduction occurs probabilistically upon eating (δ).

This is the SIMPLEST reference implementation used for comparison with more complex models.
"""


from dataclasses import dataclass
from vi import Agent, Config, Simulation
import random
import pandas as pd
import matplotlib.pyplot as plt
from pygame.math import Vector2

# Global population sets
preys = set()
predators = set()

# Image paths (optional visuals)
PREY_IMAGES = ["images/green.png"]
PREDATOR_IMAGES = ["images/triangle5.png"]

# Tracking data
frame_data = {
    "frame": [],
    "prey_count": [],
    "predator_count": [],
}

@dataclass
class BasicLVConfig(Config):
    initial_prey: int = 100
    initial_predators: int = 20

    # Adjusted Lotka-Volterra rates for cyclic behavior
    prey_birth_rate: float = 0.03      # α - reduced from 0.1
    predation_rate: float = 0.001     # β - reduced from 0.005
    predator_death_rate: float = 0.04  # γ - reduced from 0.05
    predator_reproduction_rate: float = 0.4  # δ - increased from 0.01

    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

    movement_speed: float = 20.0  # Movement speed for agents

    80,"{'prey_birth_rate': 0.035, 'predation_rate': 0.0012, 'predator_death_rate': 0.045, 'predator_reproduction_rate': 0.5}",False,118,59,59

class Prey(Agent):
    config: BasicLVConfig
    def on_spawn(self):
        preys.add(self.id)
    def update(self):
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.movement_speed
        self.pos += self.move * self.config.delta_time

class Predator(Agent):
    config: BasicLVConfig
    def on_spawn(self):
        predators.add(self.id)
    def update(self):
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.movement_speed
        self.pos += self.move * self.config.delta_time

class Monitor(Agent):
    config: BasicLVConfig
    def update(self):
        frame = self.simulation.shared.counter
        num_prey = len(preys)
        num_pred = len(predators)

        # --- Prey reproduction (exponential) ---
        # Use probabilistic reproduction instead of deterministic
        for prey_id in list(preys):
            if random.random() < self.config.prey_birth_rate:
                self.simulation.spawn_agent(Prey, images=PREY_IMAGES)

        # --- Predation ---
        # More controlled predation with probabilistic encounters
        if num_prey > 0 and num_pred > 0:
            # Each predator has a chance to catch prey based on prey density
            predator_ids = list(predators)
            for pred_id in predator_ids:
                # Probability of successful hunt depends on prey availability
                hunt_probability = min(self.config.predation_rate * num_prey, 0.5)
                if random.random() < hunt_probability and preys:
                    # Remove a random prey
                    prey_id = random.choice(list(preys))
                    for agent in self.simulation._agents:
                        if agent.id == prey_id:
                            preys.remove(prey_id)
                            agent.kill()
                            break

                    # Predator reproduction upon successful hunt
                    if random.random() < self.config.predator_reproduction_rate:
                        self.simulation.spawn_agent(Predator, images=PREDATOR_IMAGES)

        # --- Predator death (probabilistic) ---
        predator_ids = list(predators)
        for pred_id in predator_ids:
            if random.random() < self.config.predator_death_rate:
                for agent in self.simulation._agents:
                    if agent.id == pred_id:
                        predators.remove(pred_id)
                        agent.kill()
                        break

        # --- Logging ---
        frame_data["frame"].append(frame)
        frame_data["prey_count"].append(len(preys))
        frame_data["predator_count"].append(len(predators))

        if len(preys) == 0 or len(predators) == 0:
            print("Extinction occurred")
            self.simulation.stop()

        if frame > self.config.duration:
            print("Time limit reached")
            self.simulation.stop()

def run_basic_lv():
    cfg = BasicLVConfig()
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
        .run()
    )

    # Export results
    df = pd.DataFrame(frame_data)
    df.to_csv("baseline.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["frame"], df["prey_count"], label="Prey", color="green")
    plt.plot(df["frame"], df["predator_count"], label="Predator", color="red")
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Population")
    plt.title("Classic/Baseline Lotka-Volterra")
    plt.grid(True)
    plt.savefig("baseline.png")
    plt.close()
    print("Plot saved to baseline.png")

import numpy as np
import pandas as pd
from itertools import product
from vi import HeadlessSimulation

# Base good parameters
base_params = {
    "prey_birth_rate": 0.03,
    "predation_rate": 0.001,
    "predator_death_rate": 0.04,
    "predator_reproduction_rate": 0.4,
}

# Extended variation: 5-point grid around each parameter
variation = {
    "prey_birth_rate":       np.linspace(0.025, 0.035, 5),  # α
    "predation_rate":        np.linspace(0.0007, 0.0013, 5), # β
    "predator_death_rate":   np.linspace(0.03, 0.05, 5),     # γ
    "predator_reproduction_rate": np.linspace(0.2, 0.6, 5),  # δ
}

# Prepare combinations
param_names = list(base_params.keys())
param_combinations = list(product(*[variation[p] for p in param_names]))

def run_lv_headless(params):
    preys.clear()
    predators.clear()
    frame_data["frame"].clear()
    frame_data["prey_count"].clear()
    frame_data["predator_count"].clear()

    cfg = BasicLVConfig(
        prey_birth_rate=params[0],
        predation_rate=params[1],
        predator_death_rate=params[2],
        predator_reproduction_rate=params[3],
        duration=6000
    )

    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
        .run()
    )

    last_frame = frame_data["frame"][-1] if frame_data["frame"] else 0
    return {
        "params": dict(zip(param_names, params)),
        "frames_survived": last_frame
    }

# Run search
best_result = {"params": None, "frames_survived": -1}

for i, param_set in enumerate(param_combinations):
    if i % 25 == 0:
        print(f"Progress: {i}/{len(param_combinations)}")
    result = run_lv_headless(param_set)
    if result["frames_survived"] > best_result["frames_survived"]:
        best_result = result
        print(f"current best: {best_result['frames_survived']} frames with params {best_result['params']}")

# Print final best configuration
print("\n✅ Best configuration (longest survival):")
print(f"Parameters: {best_result['params']}")
print(f"Frames survived: {best_result['frames_survived']}")
