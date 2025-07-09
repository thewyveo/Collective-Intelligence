import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import pandas as pd
import math

# Image paths
PREY_IMAGES = ["images/green.png"]
PREDATOR_IMAGES = ["images/triangle5.png"]
FOOD_IMAGES = ["images/plus.png"]

frame_data = {
    "frame": [],
    "prey_count": [],
    "predator_count": [],
    "food_count": []
}

preys = set()
predators = set()
foods = set()

@dataclass
class LotkaVolterraConfig(Config):
    # Energy parameters - adjusted for better oscillations
    prey_energy_gain: float = 25.0  # Slightly reduced from 30.0
    prey_energy_loss: float = 0.2   # Increased from 0.1
    prey_initial_energy: float = 40.0  # Reduced from 50.0
    predator_energy_gain: float = 40.0  # Increased from 30.0 (predators need more incentive to hunt)
    predator_energy_loss: float = 0.3   # Increased from 0.2
    predator_initial_energy: float = 40.0  # Increased from 30.0
    
    # Reproduction parameters - adjusted timing
    prey_reproduce_prob: float = 0.07  # Increased from 0.03
    predator_reproduce_prob: float = 0.02  # Increased from 0.02
    prey_reproduce_threshold: float = 20.0  # Increased from 30.0
    predator_reproduce_threshold: float = 30.0  # Increased from 25.0
    prey_reproduce_cost: float = 15.0  # Increased from 15.0
    predator_reproduce_cost: float = 25.0  # Increased from 20.0
    
    # Food parameters - more stable food supply
    food_spawn_prob: float = 0.02  # Increased from 0.01
    food_spawn_amount: int = 2     # Increased from 1
    initial_food: int = 100        # Increased from 50
    
    # Movement parameters - slightly adjusted
    prey_speed: float = 1.8  # Increased from 1.5
    predator_speed: float = 2.0  # Increased from 2.5
    
    # Interaction parameters
    predation_distance: float = 10.0  # Increased from 10.0
    eating_distance: float = 5.0      # Increased from 5.0
    
    # Simulation parameters
    initial_prey: int = 80   # Reduced from 100
    initial_predators: int = 15  # Reduced from 20
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

class Food(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        foods.add(self.id)
        self.change_image(0)
        self.move = Vector2(0, 0)  # Food doesn't move
    
    def update(self):
        pass

class FoodSpawner(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        self.change_image(0)  # Transparent image
    
    def update(self):
        # Spawn new food randomly
        if random.random() < self.config.food_spawn_prob:
            for _ in range(self.config.food_spawn_amount):
                self.simulation.spawn_agent(Food, images=FOOD_IMAGES)

class Prey(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        preys.add(self.id)
        self.change_image(0)
        self.energy = self.config.prey_initial_energy
    
    def update(self):
        # Energy loss per frame
        self.energy -= self.config.prey_energy_loss
        
        # Death from starvation
        if self.energy <= 0:
            preys.discard(self.id)
            self.kill()
            return
        
        # Check for nearby food
        nearest_food = None
        min_food_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Food) and distance < min_food_distance:
                min_food_distance = distance
                nearest_food = agent
        
        # Eat food if close enough
        if nearest_food and min_food_distance < self.config.eating_distance:
            if nearest_food.id in foods:
                foods.remove(nearest_food.id)
                nearest_food.kill()
                self.energy += self.config.prey_energy_gain
        
        # Reproduction - NetLogo style probability-based
        if (self.energy > self.config.prey_reproduce_threshold and 
            random.random() < self.config.prey_reproduce_prob):
            self.energy -= self.config.prey_reproduce_cost
            self.reproduce()
    
    def change_position(self):
        self.there_is_no_escape()
        
        # Move randomly but with some chance to move toward food
        if random.random() < 0.05 or self.move.length() == 0:
            # Check for nearby food
            nearest_food = None
            min_food_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Food) and distance < min_food_distance:
                    min_food_distance = distance
                    nearest_food = agent
            
            if nearest_food and min_food_distance < 100:  # Move toward food if within vision
                food_direction = (nearest_food.pos - self.pos).normalize()
                self.move = food_direction * self.config.prey_speed
            else:  # Random movement
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_speed
        
        self.pos += self.move * self.config.delta_time

class Predator(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        predators.add(self.id)
        self.change_image(0)
        self.energy = self.config.predator_initial_energy
    
    def update(self):
        # Energy loss per frame
        self.energy -= self.config.predator_energy_loss
        
        # Death from starvation
        if self.energy <= 0:
            predators.discard(self.id)
            self.kill()
            return
        
        # Check for nearby prey
        nearest_prey = None
        min_prey_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Prey) and distance < min_prey_distance:
                min_prey_distance = distance
                nearest_prey = agent
        
        # Eat prey if close enough
        if nearest_prey and min_prey_distance < self.config.predation_distance:
            if nearest_prey.id in preys:
                preys.remove(nearest_prey.id)
                nearest_prey.kill()
                self.energy += self.config.predator_energy_gain
        
        # Reproduction - NetLogo style probability-based
        if (self.energy > self.config.predator_reproduce_threshold and 
            random.random() < self.config.predator_reproduce_prob):
            self.energy -= self.config.predator_reproduce_cost
            self.reproduce()
    
    def change_position(self):
        self.there_is_no_escape()
        
        # Move randomly but with some chance to move toward prey
        if random.random() < 0.05 or self.move.length() == 0:
            # Check for nearby prey
            nearest_prey = None
            min_prey_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Prey) and distance < min_prey_distance:
                    min_prey_distance = distance
                    nearest_prey = agent
            
            if nearest_prey and min_prey_distance < 150:  # Move toward prey if within vision
                prey_direction = (nearest_prey.pos - self.pos).normalize()
                self.move = prey_direction * self.config.predator_speed
            else:  # Random movement
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        self.pos += self.move * self.config.delta_time

class SimulationMonitor(Agent):
    def update(self):
        current_frame = self.simulation.shared.counter
        frame_data["frame"].append(current_frame)
        frame_data["prey_count"].append(len(preys))
        frame_data["predator_count"].append(len(predators))
        frame_data["food_count"].append(len(foods))
        
        if len(preys) == 0 or len(predators) == 0:
            print("Simulation ended due to population collapse")
            self.simulation.stop()

        if current_frame % 10 == 0:
            print(f"Frame {current_frame}: Prey={len(preys)}, Predators={len(predators)}, Food={len(foods)}")

        if current_frame > 5000:
            print("Simulation ended due to time limit")
            self.simulation.stop()

def run_simulation():
    cfg = LotkaVolterraConfig()
    
    # Create and run simulation
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(FoodSpawner, images=["images/transparent.png"])
        .spawn_agent(SimulationMonitor, images=["images/transparent.png"])
        .run()
    )

    # Create DataFrame from collected data
    df = pd.DataFrame(frame_data)
    
    # Save data to CSV
    df.to_csv('lotka_volterra_data.csv', index=False)
    print("Data saved to lotka_volterra_data.csv")
    
    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['frame'], df['prey_count'], label='Prey Population', color='green')
    plt.plot(df['frame'], df['predator_count'], label='Predator Population', color='red')
    plt.plot(df['frame'], df['food_count'], label='Food Count', color='blue', linestyle='--')
    plt.title('Lotka-Volterra Predator-Prey Dynamics with Energy and Food')
    plt.xlabel('Time (frames)')
    plt.ylabel('Population Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lotka_volterra.png')
    plt.close()
    print("Plot saved to lotka_volterra.png")

if __name__ == "__main__":
    run_simulation()