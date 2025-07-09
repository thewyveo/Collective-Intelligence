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
from vi import Agent, Config, Simulation, HeadlessSimulation
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
    prey_birth_rate: float = 0.027500000000000004      # α - reduced from 0.1
    predation_rate: float = 0.00085     # β - reduced from 0.005
    predator_death_rate: float = 0.035  # γ - reduced from 0.05
    predator_reproduction_rate: float = 0.5  # δ - increased from 0.01

    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

    movement_speed: float = 20.0  # Movement speed for agents

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
    global frame_data, preys, predators
    frame_data["frame"].clear()
    frame_data["prey_count"].clear()
    frame_data["predator_count"].clear()
    preys.clear()
    predators.clear()

    cfg = BasicLVConfig()
    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
        .run()
    )

    return pd.DataFrame(frame_data)

all_runs = []
for run_id in range(50):
    df = run_basic_lv()
    df["run"] = run_id
    all_runs.append(df)

results_df = pd.concat(all_runs, ignore_index=True)
results_df.to_csv("all_50_runs_baseline.csv", index=False)

print("Saved 50-run results to all_50_runs_baseline.csv")

# Plot all 50 runs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for run_id in results_df["run"].unique():
    run_df = results_df[results_df["run"] == run_id]
    plt.plot(run_df["frame"], run_df["prey_count"], color="green", alpha=0.3)
    plt.plot(run_df["frame"], run_df["predator_count"], color="red", alpha=0.3)

plt.title("Classic/Baseline Lotka-Volterra: 50 Runs")
plt.xlabel("Frame")
plt.ylabel("Population")
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_50_runs.png")
plt.close()
print("Plot saved to baseline_50_runs.png")
