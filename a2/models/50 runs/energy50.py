from dataclasses import dataclass, field
from vi import Agent, Config, Simulation, HeadlessSimulation
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from pygame.math import Vector2

# Image paths
PREY_IMAGES = ["images/green.png"]
PREDATOR_IMAGES = ["images/triangle5.png"]
FOOD_IMAGES = ["images/plus.png"]

frame_data = {
    "frame": [],
    "prey": [],
    "predator": [],
    "food": [],
}

@dataclass
class SimpleEnergyLVConfig(Config):
    initial_prey: int = 100
    initial_predators: int = 20

    # Lotka-Volterra parameters
    prey_birth_rate: float = 0.012
    predation_rate: float = 0.0008
    predator_death_rate: float = 0.035
    predator_reproduction_rate: float = 0.4

    # Energy parameters
    energy_decay_rate: float = 0.06
    energy_gain_from_eating: float = 1.8

    # Food parameters
    initial_food: int = 50
    food_regrowth_rate: float = 0.1
    food_energy_value: float = 0.5
    food_consumption_radius: float = 10.0

    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

    movement_speed: float = 30.0


class Food(Agent):
    config: SimpleEnergyLVConfig
    eaten: bool = False

    def on_spawn(self):
        self.eaten = False

    def update(self):
        pass


class Prey(Agent):
    config: SimpleEnergyLVConfig
    energy: float = 10.0

    def on_spawn(self):
        self.energy = 10.0

    def update(self):
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate

        # Check for nearby food
        for agent in self.simulation._agents:
            if isinstance(agent, Food) and not agent.eaten:
                distance = math.dist(self.pos, agent.pos)
                if distance < self.config.food_consumption_radius:
                    self.energy += self.config.food_energy_value
                    agent.eaten = True
                    break

        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            return

        # Reproduce based on birth rate (modified by energy level)
        if random.random() < self.config.prey_birth_rate * (1 + self.energy) / 2:
            self.reproduce()

        self.move = Vector2(random.uniform(-1, 1),
                            random.uniform(-1, 1)).normalize() * self.config.movement_speed * self.config.delta_time
        self.pos += self.move * self.config.delta_time


class Predator(Agent):
    config: SimpleEnergyLVConfig
    energy: float = 1.0

    def on_spawn(self):
        self.energy = 1.0

    def update(self):
        self.energy -= self.config.energy_decay_rate

        # Distance-based death probability
        nearest_prey_distance = self.find_nearest_prey_distance()

        # Base death probability increases with distance to nearest prey
        # The farther away prey are, the higher chance of death
        distance_death_prob = self.config.predator_death_rate * (1 + nearest_prey_distance / 100)

        # Die if energy depleted or distance-based death
        if self.energy <= 0 or random.random() < distance_death_prob:
            self.kill()
            return

        # Original predation mechanics (unchanged)
        preys = [agent for agent in self.simulation._agents if isinstance(agent, Prey)]
        if preys and random.random() < self.config.predation_rate * len(preys):
            prey = random.choice(preys)
            prey.kill()
            self.energy += self.config.energy_gain_from_eating

            if random.random() < self.config.predator_reproduction_rate and self.energy > 0.25:
                self.reproduce()

        self.move = Vector2(random.uniform(-1, 1),
                            random.uniform(-1, 1)).normalize() * self.config.movement_speed * self.config.delta_time
        self.pos += self.move * self.config.delta_time

    def find_nearest_prey_distance(self) -> float:
        """Calculate distance to nearest prey agent"""
        min_distance = float('inf')
        for agent in self.simulation._agents:
            if isinstance(agent, Prey):
                distance = math.dist(self.pos, agent.pos)
                if distance < min_distance:
                    min_distance = distance
        return min_distance if min_distance != float('inf') else self.config.window_size[
            0]  # Return max distance if no prey


class Monitor(Agent):
    config: SimpleEnergyLVConfig

    def __init__(self, images, simulation):
        super().__init__(images, simulation)
        self.frame_data = {
            "frame": [],
            "prey": [],
            "predator": [],
            "food": [],
        }

    def update(self):
        global frame_data
        frame = self.simulation.shared.counter

        # Regrow food occasionally
        if random.random() < self.config.food_regrowth_rate:
            for agent in self.simulation._agents:
                if isinstance(agent, Food) and agent.eaten and random.random() < 0.1:
                    agent.eaten = False

        # Count populations
        prey_count = sum(1 for a in self.simulation._agents if isinstance(a, Prey))
        predator_count = sum(1 for a in self.simulation._agents if isinstance(a, Predator))
        food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food) and not a.eaten)

        # Store data
        frame_data["frame"].append(frame)
        frame_data["prey"].append(prey_count)
        frame_data["predator"].append(predator_count)
        frame_data["food"].append(food_count)

        # Stop conditions
        if prey_count == 0 or predator_count == 0:
            print("Extinction occurred")
            self.on_destroy()
            self.simulation.stop()
        if frame > self.config.duration:
            print("Time limit reached")
            self.on_destroy()
            self.simulation.stop()

    def on_destroy(self):
        # Save data when simulation ends
        df = pd.DataFrame(frame_data)
        df.to_csv("energy.csv", index=False)

        plt.figure(figsize=(12, 6))
        plt.plot(df["frame"], df["prey"], label="Prey", color="green")
        plt.plot(df["frame"], df["predator"], label="Predator", color="red")
        plt.plot(df["frame"], df["food"], label="Food", color="blue", alpha=0.5)
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Population/Amount")
        plt.title("Semi-Realistic Lotka-Volterra Model Results")
        plt.grid(True)
        plt.savefig("energy.png")
        plt.close()
        print("Plot saved to energy.png")


def run_simple_energy_lv():
    global frame_data
    frame_data["frame"].clear()
    frame_data["prey"].clear()
    frame_data["predator"].clear()
    frame_data["food"].clear()

    cfg = SimpleEnergyLVConfig()

    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
    )

    sim.run()
import pandas as pd

# Holds all results across 50 runs
all_runs = []

# Run simulation 50 times
for run_id in range(50):
    # Run simulation
    run_simple_energy_lv()

    # Load this run's output (from the file that Monitor wrote)
    df = pd.read_csv("energy.csv")
    df["run"] = run_id

    # Add to all_runs list
    all_runs.append(df)

# Combine all runs
results_df = pd.concat(all_runs, ignore_index=True)
results_df.to_csv("all_50_runs.csv", index=False)

print("Saved 50-run results to all_50_runs.csv")

# Plot all 50 runs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for run_id in results_df["run"].unique():
    run_df = results_df[results_df["run"] == run_id]
    plt.plot(run_df["frame"], run_df["prey"], color="green", alpha=0.3)
    plt.plot(run_df["frame"], run_df["predator"], color="red", alpha=0.3)

plt.title("Semi-Realistic Lotka-Volterra: 50 Runs")
plt.xlabel("Frame")
plt.ylabel("Population")
plt.grid(True)
plt.tight_layout()
plt.savefig("semi_realistic_50_runs.png")
plt.close()
print("Plot saved to semi_realistic_50_runs.png")