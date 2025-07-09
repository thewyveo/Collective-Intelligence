import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import math
import pandas as pd

# Define image paths
PREY_IMAGES = [
    "images/green.png",    # Prey wandering (0)
    "images/yellow.png",   # Prey fleeing (1)
    "images/red.png"       # Prey being eaten (2)
]
PREDATOR_IMAGES = [
    "images/triangle5.png",  # Hunting (0)
    "images/triangle7.png",  # Lunging (1)
    "images/triangle6.png"   # Eating (2)
]
FOOD_IMAGES = [
    "images/plus.png"       # Food (0)
]

# State constants
PREY_WANDERING = 0
PREY_FLEEING = 1
PREY_EATEN = 2
PREDATOR_HUNTING = 0
PREDATOR_LUNGING = 1
PREDATOR_EATING = 2

frame_data = {
    "frame": [],
    "prey_wandering": [],
    "prey_fleeing": [],
    "predator_hunting": [],
    "predator_lunging": [],
    "predator_eating_prey": [],
    "food_count": []
}

food_count = 15
preys = set()
predators = set()

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 5.0  # Reduced from 5.0
    prey_flee_speed: float = 6.0  # Reduced from 6.0
    prey_vision: float = 100.0
    prey_energy: float = 200.0  # Reduced from 200.0
    prey_energy_consumption: float = 0.1  # Reduced from 0.223
    prey_flee_energy_consumption: float = 0.2  # Reduced from 0.3
    prey_energy_gain: float = 20.0  # Increased from 5.0
    prey_reproduction_energy_threshold: float = 60.0  # Reduced from 75.0
    prey_reproduction_cost: float = 30.0  # Increased from 25.0
    prey_reproduction_radius: float = 30.0
    prey_max_age = 200
    
    # Predator parameters
    predator_speed: float = 5.5  # Reduced from 5.5
    predator_lunge_speed: float = 10.0  # Reduced from 10.0
    predator_vision: float = 140.0
    predator_energy: float = 150.0  # Reduced from 150.0
    predator_energy_consumption: float = 0.15  # Increased from 0.15
    predator_lunge_energy_consumption: float = 0.4  # Reduced from 0.5
    predator_eating_threshold: float = 25.0
    predator_eating_energy: float = 50.0  # Reduced from 50.0
    predator_reproduction_energy_threshold: float = 30.0  # Increased from 30.0
    predator_reproduction_cost: float = 40.0  # Reduced from 70.0
    predator_reproduction_radius: float = 50.0
    predator_max_age = 200

    # Eating parameters
    eating_duration: int = 3  # Frames prey shows being eaten / predator is eating
    
    # Food parameters
    food_spawn_rate: float = 0.1  # Increased from 0.001
    
    # New probabilistic parameters from simple model - adjusted values
    prey_birth_rate: float = 0.2  # Reduced from 0.01
    predation_rate: float = 0.002  # Increased from 0.001
    predator_death_rate: float = 0.4  # Reduced from 0.04
    predator_reproduction_rate: float = 0.4  # Reduced from 0.4
    
    # Simulation parameters
    initial_prey: int = 150  # Reduced from 50
    initial_predators: int = 20  # Reduced from 20
    initial_food: int = 200  # Increased from 15

    # Setting parameters
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

class FoodSpawner(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        self.change_image(0)  # transparent image
        global food_count
        self.config.inital_food = food_count
    
    def update(self):
        if random.random() < self.config.food_spawn_rate:
            self.simulation.spawn_agent(agent_class=Food, images=FOOD_IMAGES)
            global food_count, preys, predators
            food_count += 1

        current_frame = self.simulation.shared.counter

        if current_frame >= len(frame_data["frame"]):
            # Count current states
            prey_wandering = sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.state == PREY_WANDERING)
            prey_fleeing = sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.state == PREY_FLEEING)
            prey_eaten = sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.state == PREY_EATEN)
            
            predator_hunting = sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_HUNTING)
            predator_lunging = sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_LUNGING)
            predator_eating = sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_EATING)
            
            # Record frame data
            frame_data["frame"].append(current_frame)
            frame_data["prey_wandering"].append(prey_wandering)
            frame_data["prey_fleeing"].append(prey_fleeing)
            frame_data["predator_hunting"].append(predator_hunting)
            frame_data["predator_lunging"].append(predator_lunging)
            frame_data["predator_eating_prey"].append(predator_eating)
            frame_data["food_count"].append(food_count)

        if current_frame % 1 == 0:  # Record every 10 frames
            print(f"frame {current_frame}: preys = {len(preys)}, predators = {len(predators)}, food = {food_count}")

        if len(predators) <= 1 or len(preys) <= 1:
            print("Simulation ended due to extinction.")
            self.simulation.stop()

class Food(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        self.change_image(0)  # Only one food image
        self.move = Vector2(0, 0)  # Food does not move

    def update(self):
        pass

class Prey(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        self.state = PREY_WANDERING
        self.energy = self.config.prey_energy
        self.eating_timer = 0
        self.change_image(PREY_WANDERING)
        preys.add(self.id)
        self.type = "prey"
        self.sex = random.choice(["male", "female"])
        self.age = 0
    
    def update(self):
        self.age += 1
        if self.age >= self.config.prey_max_age:
            self.kill()
            if self.id in preys:
                preys.remove(self.id)
            return

        if self.state == PREY_EATEN:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.kill()
                if self.id in preys:
                    preys.remove(self.id)
            return
        
        if self.state == PREY_FLEEING:
            self.energy -= self.config.prey_flee_energy_consumption
        else:
            self.energy -= self.config.prey_energy_consumption
        
        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            if self.id in preys:
                preys.remove(self.id)
            return

        # Find nearest predator and food
        nearest_predator = None
        nearest_food = None
        min_predator_distance = float('inf')
        min_food_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Predator):
                if distance < min_predator_distance:
                    min_predator_distance = distance
                    nearest_predator = agent
            elif isinstance(agent, Food):
                if distance < min_food_distance:
                    min_food_distance = distance
                    nearest_food = agent
        
        # State transitions
        if nearest_predator and min_predator_distance < self.config.prey_vision:
            self.state = PREY_FLEEING
            self.change_image(PREY_FLEEING)
        else:
            self.state = PREY_WANDERING
            self.change_image(PREY_WANDERING)
        
        # Eat food if nearby
        if nearest_food and min_food_distance < 10:  # Eating distance
            nearest_food.kill()
            self.energy += self.config.prey_energy_gain

        # Reproduction - only check occasionally to reduce computation
        if random.random() < 0.05:  # Only check 5% of frames
            for agent, distance in self.in_proximity_accuracy():
                if (isinstance(agent, Prey) and 
                    distance < self.config.prey_reproduction_radius and
                    self.state == PREY_WANDERING and 
                    agent.state == PREY_WANDERING and
                    self.energy > self.config.prey_reproduction_energy_threshold and
                    agent.energy > self.config.prey_reproduction_energy_threshold and
                    ((self.sex == "male" and agent.sex == "female") or (self.sex == "female" and agent.sex == "male"))):
                    
                    if random.random() < self.config.prey_birth_rate:
                        self.reproduce()
                        agent.reproduce()
                        self.energy -= self.config.prey_reproduction_cost
                        agent.energy -= self.config.prey_reproduction_cost
                        break

    def change_position(self):
        if self.state == PREY_EATEN:
            self.move = Vector2(0, 0)
            return
            
        self.there_is_no_escape()
        
        if self.state == PREY_WANDERING:
            nearest_food = None
            min_food_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Food) and distance < min_food_distance:
                    min_food_distance = distance
                    nearest_food = agent
            
            if nearest_food and min_food_distance < self.config.prey_vision:
                # Move toward food
                delta = nearest_food.pos - self.pos
                if delta.length() == 0:
                    food_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                else:
                    food_direction = delta.normalize()
                self.move = food_direction * self.config.prey_speed
            else:
                if random.random() < 0.05 or self.move.length() == 0:
                    angle = random.uniform(0, 2 * math.pi)
                    self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_speed

        else:  # FLEEING
            nearest_predator = None
            min_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Predator):
                    if distance < min_distance:
                        min_distance = distance
                        nearest_predator = agent
            
            if nearest_predator:
                delta = self.pos - nearest_predator.pos
                if delta.length() == 0:
                    flee_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                else:
                    flee_direction = delta.normalize()
                self.move = flee_direction * self.config.prey_flee_speed
            else:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_flee_speed
                self.state = PREY_WANDERING

        self.pos += self.move * self.config.delta_time
    
    def start_being_eaten(self):
        """Transition to being eaten state"""
        self.state = PREY_EATEN
        self.change_image(PREY_EATEN)
        self.eating_timer = self.config.eating_duration


class Predator(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        self.state = PREDATOR_HUNTING
        self.energy = self.config.predator_energy
        self.eating_timer = 0
        self.target_prey = None
        self.change_image(PREDATOR_HUNTING)
        predators.add(self.id)
        self.sex = random.choice(["male", "female"])
        self.age = 0
    
    def update(self):
        self.age += 1
        if self.age >= self.config.predator_max_age:
            self.kill()
            if self.id in predators:
                predators.remove(self.id)
            return

        if self.state == PREDATOR_LUNGING:
            self.energy -= self.config.predator_lunge_energy_consumption
        else:
            self.energy -= self.config.predator_energy_consumption
        
        # Die if energy depleted or based on predator_death_rate
        if (self.energy <= 0 or 
            (random.random() < self.config.predator_death_rate * (1 - self.energy/self.config.predator_energy))):
            self.kill()
            if self.id in predators:
                predators.remove(self.id)
            return
        
        # Reproduction - only check occasionally to reduce computation
        for agent, distance in self.in_proximity_accuracy():
            if (isinstance(agent, Predator) and 
                distance < self.config.predator_reproduction_radius and
                self.state != PREDATOR_EATING and 
                agent.state != PREDATOR_EATING and
                self.energy > self.config.predator_reproduction_energy_threshold and
                agent.energy > self.config.predator_reproduction_energy_threshold and
                ((self.sex == "male" and agent.sex == "female") or (self.sex == "female" and agent.sex == "male"))):
                
                    if random.random() < self.config.predator_reproduction_rate:
                        self.reproduce()
                        agent.reproduce()
                        self.energy -= self.config.predator_reproduction_cost
                        agent.energy -= self.config.predator_reproduction_cost
                        break
        
        # Handle state transitions
        if self.state == PREDATOR_EATING:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.state = PREDATOR_HUNTING
                self.change_image(PREDATOR_HUNTING)
                self.target_prey = None
            return
        
        # Find nearest prey
        nearest_prey = None
        min_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Prey):
                if distance < min_distance:
                    min_distance = distance
                    nearest_prey = agent
        
        # State transitions
        if self.state == PREDATOR_HUNTING:
            if nearest_prey and min_distance < self.config.predator_vision:
                self.target_prey = nearest_prey
                self.state = PREDATOR_LUNGING
                self.change_image(PREDATOR_LUNGING)
        
        elif self.state == PREDATOR_LUNGING:
            if not nearest_prey or min_distance > self.config.predator_vision:
                # Lost sight of prey
                self.state = PREDATOR_HUNTING
                self.change_image(PREDATOR_HUNTING)
                self.target_prey = None
            elif min_distance < self.config.predator_eating_threshold:
                # Probabilistic killing based on predation_rate
                if (nearest_prey.state != PREY_EATEN and 
                    random.random() < self.config.predation_rate):
                    nearest_prey.start_being_eaten()
                    self.energy += self.config.predator_eating_energy
                    self.state = PREDATOR_EATING
                    self.change_image(PREDATOR_EATING)
                    self.eating_timer = self.config.eating_duration
                    self.target_prey = None
            
    def change_position(self):
        if self.state == PREDATOR_EATING:
            self.move = Vector2(0, 0)
        elif self.state == PREDATOR_LUNGING and self.target_prey:
            delta = self.target_prey.pos - self.pos
            if delta.length() == 0:
                lunge_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            else:
                lunge_direction = delta.normalize()
            self.move = lunge_direction * self.config.predator_lunge_speed
        else:
            if random.random() < 0.05 or self.move.length() == 0:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        if self.there_is_no_escape():
            self.move = -self.move
        
        self.pos += self.move * self.config.delta_time

def run_simulation():
    cfg = PredatorPreyConfig()
    
    # Create and run simulation
    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(
            cfg.initial_prey, 
            Prey, 
            images=PREY_IMAGES
        )
        .batch_spawn_agents(
            cfg.initial_predators, 
            Predator, 
            images=PREDATOR_IMAGES
        )
        .batch_spawn_agents(
            cfg.initial_food,
            Food,
            images=FOOD_IMAGES
        )
        .spawn_agent(
            FoodSpawner,
            images=["images/transparent.png"],
        )
        .run()
    )

    # Create DataFrame from collected data
    df = pd.DataFrame(frame_data)

    # Save data to CSV
    df.to_csv('predator_prey_data.csv', index=False)
    print("Data saved to predator_prey_data.csv, plotting...")

    # Plotting species-level dynamics
    df["total_prey"] = df["prey_wandering"] + df["prey_fleeing"]
    df["total_predators"] = df["predator_hunting"] + df["predator_lunging"] + df["predator_eating_prey"]

    plt.figure(figsize=(14, 8))

    plt.plot(df['frame'], df['total_prey'], label='Total Prey', color='green')
    plt.plot(df['frame'], df['total_predators'], label='Total Predators', color='red')
    plt.plot(df['frame'], df['food_count'], label='Food', color='blue', linestyle='--')

    plt.title('Balanced Lotka-Volterra Model with States')
    plt.xlabel('Time (frames)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predator_prey.png')
    plt.close()

    print("Plotting complete. Plot saved to predator_prey.png")

if __name__ == "__main__":
    run_simulation()