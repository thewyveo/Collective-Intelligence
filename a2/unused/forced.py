import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import math
import pandas as pd
import numpy as np

# Simplified image paths
PREY_IMAGE = "images/green.png"
PREDATOR_IMAGE = "images/triangle5.png"
FOOD_IMAGE = "images/plus.png"

# Tracking data
frame_data = {
    "frame": [],
    "prey_count": [],
    "predator_count": [],
    "food_count": []
}

# Global counters
food_count = 0
prey_count = 0
predator_count = 0

@dataclass
class CyclicConfig(Config):
    # Core parameters for cycling
    cycle_length: int = 1000  # frames for full predator-prey cycle
    max_prey: int = 200       # carrying capacity for prey
    max_predators: int = 50   # carrying capacity for predators
    
    # Movement parameters
    prey_speed: float = 2.0
    predator_speed: float = 3.5
    
    # Energy parameters
    prey_energy_gain: float = 5.0
    predator_energy_gain: float = 20.0
    
    # Simulation settings
    initial_prey: int = 100
    initial_predators: int = 20
    initial_food: int = 50
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

class CycleController(Agent):
    config: CyclicConfig
    
    def on_spawn(self):
        self.change_image(0)  # invisible controller
        global food_count, prey_count, predator_count
        food_count = self.config.initial_food
        prey_count = self.config.initial_prey
        predator_count = self.config.initial_predators
        
    def update(self):
        global food_count, prey_count, predator_count
        
        current_frame = self.simulation.shared.counter
        
        # Calculate where we are in the cycle (0-2π)
        cycle_position = (current_frame % self.config.cycle_length) / self.config.cycle_length * 2 * math.pi
        
        # Calculate target populations using sine waves (offset by π/2 for predator-prey lag)
        target_prey = int((math.sin(cycle_position) + 1) / 2 * self.config.max_prey)
        target_predators = int((math.sin(cycle_position - math.pi/2) + 1) / 2 * self.config.max_predators)
        
        # Adjust populations to match targets
        self.adjust_populations(target_prey, target_predators)
        
        # Spawn food periodically
        if current_frame % 10 == 0:
            self.simulation.spawn_agent(Food, images=[FOOD_IMAGE])
            food_count += 1
        
        # Record data
        frame_data["frame"].append(current_frame)
        frame_data["prey_count"].append(prey_count)
        frame_data["predator_count"].append(predator_count)
        frame_data["food_count"].append(food_count)
        
        if current_frame % 100 == 0:
            print(f"Frame {current_frame}: Prey={prey_count}, Predators={predator_count}, Food={food_count}")
    
    def adjust_populations(self, target_prey, target_predators):
        global prey_count, predator_count
        
        # Adjust prey population
        current_prey = prey_count
        diff = target_prey - current_prey
        
        if diff > 0:  # Need more prey
            for _ in range(diff):
                if prey_count < self.config.max_prey * 1.2:  # Slightly over target is okay
                    self.simulation.spawn_agent(Prey, images=[PREY_IMAGE])
                    prey_count += 1
        elif diff < 0:  # Need fewer prey
            # Let natural predation handle it first
            if random.random() < 0.1:  # Sometimes force removal
                agents = [a for a in self.simulation._agents if isinstance(a, Prey)]
                if agents:
                    agent = random.choice(agents)
                    agent.kill()
                    prey_count -= 1
        
        # Adjust predator population
        current_predators = predator_count
        diff = target_predators - current_predators
        
        if diff > 0:  # Need more predators
            for _ in range(diff):
                if predator_count < self.config.max_predators * 1.2:
                    self.simulation.spawn_agent(Predator, images=[PREDATOR_IMAGE])
                    predator_count += 1
        elif diff < 0:  # Need fewer predators
            if random.random() < 0.1:  # Higher chance to force removal
                agents = [a for a in self.simulation._agents if isinstance(a, Predator)]
                if agents:
                    agent = random.choice(agents)
                    agent.kill()
                    predator_count -= 1

class Food(Agent):
    def on_spawn(self):
        self.change_image(0)
        self.move = Vector2(0, 0)
        
    def update(self):
        # Food doesn't do anything
        pass

class Prey(Agent):
    config: CyclicConfig
    
    def on_spawn(self):
        self.change_image(0)
        self.energy = 100
        self.move = Vector2(
            random.uniform(-self.config.prey_speed, self.config.prey_speed),
            random.uniform(-self.config.prey_speed, self.config.prey_speed)
        )
        
    def update(self):
        # Simple movement with random changes
        if random.random() < 0.05:
            self.move = Vector2(
                random.uniform(-self.config.prey_speed, self.config.prey_speed),
                random.uniform(-self.config.prey_speed, self.config.prey_speed)
            )
        
        # Eat food if found
        for agent, _ in self.in_proximity_accuracy():
            if isinstance(agent, Food):
                agent.kill()
                global food_count
                food_count -= 1
                self.energy += self.config.prey_energy_gain
                break
        
        # Die occasionally to allow population control
        if random.random() < 0.001:
            self.kill()
            global prey_count
            prey_count -= 1
    
    def change_position(self):
        self.there_is_no_escape()
        self.pos += self.move * self.config.delta_time

class Predator(Agent):
    config: CyclicConfig
    
    def on_spawn(self):
        self.change_image(0)
        self.energy = 100
        self.move = Vector2(
            random.uniform(-self.config.predator_speed, self.config.predator_speed),
            random.uniform(-self.config.predator_speed, self.config.predator_speed)
        )
        
    def update(self):
        # Hunt prey
        for agent, _ in self.in_proximity_accuracy():
            if isinstance(agent, Prey):
                agent.kill()
                global prey_count
                prey_count -= 1
                self.energy += self.config.predator_energy_gain
                break
        
        # Random movement changes
        if random.random() < 0.05:
            self.move = Vector2(
                random.uniform(-self.config.predator_speed, self.config.predator_speed),
                random.uniform(-self.config.predator_speed, self.config.predator_speed)
            )
        
        # Die occasionally to allow population control
        if random.random() < 0.002:
            self.kill()
            global predator_count
            predator_count -= 1
    
    def change_position(self):
        self.there_is_no_escape()
        self.pos += self.move * self.config.delta_time

def run_simulation():
    cfg = CyclicConfig()
    
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=[PREY_IMAGE])
        .batch_spawn_agents(cfg.initial_predators, Predator, images=[PREDATOR_IMAGE])
        .batch_spawn_agents(cfg.initial_food, Food, images=[FOOD_IMAGE])
        .spawn_agent(CycleController, images=["images/transparent.png"])
        .run()
    )
    
    # Plot results
    df = pd.DataFrame(frame_data)
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['frame'], df['prey_count'], label='Prey', color='green')
    plt.plot(df['frame'], df['predator_count'], label='Predators', color='red')
    plt.plot(df['frame'], df['food_count'], label='Food', color='blue', alpha=0.5)
    
    plt.title("Forced Cyclic Predator-Prey Dynamics")
    plt.xlabel("Time (frames)")
    plt.ylabel("Population Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forced_cycles.png')
    plt.close()
    
    print("Simulation complete. Plot saved to forced_cycles.png")

if __name__ == "__main__":
    run_simulation()