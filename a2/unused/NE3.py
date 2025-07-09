import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from pygame.math import Vector2
import copy
from multiprocessing import Pool, cpu_count
from copy import deepcopy

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

preys = set()  
predators = set()  # Global sets to track prey and predator IDs

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 4.5
    prey_flee_speed: float = 6.5
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.1
    prey_flee_energy_consumption: float = 0.15
    prey_energy_gain: float = 35.0
    prey_reproduction_energy_threshold: float = 150.0
    prey_reproduction_cost: float = 50.0
    prey_reproduction_radius: float = 27.5
    
    # Predator parameters
    predator_speed: float = 5.25
    predator_lunge_speed: float = 7.0
    predator_vision: float = 150.0
    predator_energy: float = 110.0
    predator_energy_consumption: float = 0.2
    predator_lunge_energy_consumption: float = 0.4
    predator_eating_threshold: float = 10.0
    predator_eating_energy: float = 75.0
    predator_reproduction_energy_threshold: float = 70.0
    predator_reproduction_cost: float = 44.0
    predator_reproduction_radius: float = 25.0

    # Eating parameters
    eating_duration: int = 15
    
    # Food parameters
    food_spawn_rate: float = 0.2
    
    # Simulation parameters
    initial_prey: int = 40
    initial_predators: int = 15
    initial_food: int = 15
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

class Food(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        self.change_image(0)
        self.move = Vector2(0, 0)
    
    def update(self):
        pass

class FoodSpawner(Agent):
    config: PredatorPreyConfig
    
    def update(self):
        if random.random() < self.config.food_spawn_rate:
            self.simulation.spawn_agent(Food, images=FOOD_IMAGES)

class SimulationMonitor(Agent):
    def on_spawn(self):
        self.simulation.shared.frame_data = {
            "frame": [],
            "prey_count": [],
            "predator_count": [],
            "food_count": []
        }

    def update(self):
        try:
            current_frame = self.simulation.shared.counter
            prey_count = sum(1 for a in self.simulation._agents if isinstance(a, NeuralPrey) and a.alive)
            predator_count = sum(1 for a in self.simulation._agents if isinstance(a, NeuralPredator) and a.alive)
            food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food) and a.alive)

            self.simulation.shared.frame_data["frame"].append(current_frame)
            self.simulation.shared.frame_data["prey_count"].append(prey_count)
            self.simulation.shared.frame_data["predator_count"].append(predator_count)
            self.simulation.shared.frame_data["food_count"].append(food_count)

            if current_frame > 5 and (prey_count == 0 or predator_count == 0):
                self.simulation.stop()

            if current_frame >= self.config.duration:
                self.simulation.stop()

        except Exception as e:
            print(f"[Monitor] Exception in update(): {e}")
            self.simulation.stop()

class NeuralPredatorPreySimulation(Simulation):
    def __init__(self, prey_brain, predator_brain, duration=10000):
        super().__init__(PredatorPreyConfig(duration=duration))
        self.prey_brain = prey_brain
        self.predator_brain = predator_brain
        self.shared.best_prey_brain = prey_brain
        self.shared.best_predator_brain = predator_brain
        self.shared.frame_data = {
            "frame": [],
            "prey_count": [],
            "predator_count": [],
            "food_count": []
        }
    
    def run(self):
        def spawn_neural_prey(**kwargs):
            prey = NeuralPrey(**kwargs)
            prey.brain = self.prey_brain
            return prey
        
        def spawn_neural_predator(**kwargs):
            predator = NeuralPredator(**kwargs)
            predator.brain = self.predator_brain
            return predator
        
        self.batch_spawn_agents(self.config.initial_prey, spawn_neural_prey, images=PREY_IMAGES)
        self.batch_spawn_agents(self.config.initial_predators, spawn_neural_predator, images=PREDATOR_IMAGES)
        self.batch_spawn_agents(self.config.initial_food, Food, images=FOOD_IMAGES)
        self.spawn_agent(FoodSpawner, images=["images/transparent.png"])
        self.spawn_agent(SimulationMonitor, images=["images/transparent.png"])
        super().run()

class PreyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 32)  # Added reproduction output
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)   # Now outputs: [state, move_x, move_y, speed, reproduce]
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x[4] = torch.sigmoid(x[4])  # Reproduction probability
        return x

class PredatorBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 32)   # Added reproduction output
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)   # Now outputs: [state, move_x, move_y, speed, reproduce]
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x[4] = torch.sigmoid(x[4])  # Reproduction probability
        return x

class NeuralPrey(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        global preys
        self.state = PREY_WANDERING
        self.energy = self.config.prey_energy
        self.eating_timer = 0
        self.change_image(PREY_WANDERING)
        preys.add(self.id)
        self.age = 0
        self.brain = copy.deepcopy(self.simulation.shared.best_prey_brain)
        self.last_nn_output = None
    
    def get_nn_state(self):
        nearest_pred, nearest_food = None, None
        pred_distance, food_distance = self.config.prey_vision, self.config.prey_vision
        pred_count = 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPredator) and agent.alive and distance < pred_distance:
                pred_distance, nearest_pred = distance, agent
                pred_count += 1
            elif isinstance(agent, Food) and agent.alive and distance < food_distance:
                food_distance, nearest_food = distance, agent

        pred_dir = (nearest_pred.pos - self.pos).normalize() if nearest_pred else Vector2(0, 0)
        food_dir = (nearest_food.pos - self.pos).normalize() if nearest_food else Vector2(0, 0)

        return torch.tensor([
            min(self.energy / self.config.prey_energy, 1.0),
            1.0 if self.energy < 0.2 * self.config.prey_energy else 0.0,
            pred_distance / self.config.prey_vision,
            pred_dir.x, pred_dir.y,
            min(pred_count / 10.0, 1.0),
            food_distance / self.config.prey_vision,
            food_dir.x, food_dir.y,
            1.0 if nearest_pred and pred_distance < 50 else 0.0,
            1.0 if self.state == PREY_FLEEING else 0.0,
            self.age / 1000.0  # Normalized age
        ], dtype=torch.float32)

    def update(self):
        global preys

        if self.state == PREY_EATEN:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.kill()
                preys.remove(self.id)
            return

        self.age += 1
        if self.age > 700 or self.energy <= 0:
            self.kill()
            preys.remove(self.id)
            return

        # Energy consumption based on state
        if self.state == PREY_FLEEING:
            self.energy -= self.config.prey_flee_energy_consumption
        else:
            self.energy -= self.config.prey_energy_consumption

        # Neural network decision making
        if self.brain is not None:
            with torch.no_grad():
                state = self.get_nn_state()
                output = self.brain(state)
                
                # Update state based on NN output
                self.state = PREY_FLEEING if output[0] > 0 else PREY_WANDERING
                self.change_image(self.state)
                
                # Handle reproduction
                if (output[4] > 0.7 and 
                    self.energy >= self.config.prey_reproduction_energy_threshold):
                    self.energy -= self.config.prey_reproduction_cost
                    child = self.reproduce()
                    child.brain = copy.deepcopy(self.brain)
                    preys.add(child.id)

                # Handle eating food if nearby
                for agent, distance in self.in_proximity_accuracy():
                    if isinstance(agent, Food) and distance < 10:
                        agent.kill()
                        self.energy += self.config.prey_energy_gain
                        self.age -= 100
                        break

    def change_position(self):
        if self.state == PREY_EATEN:
            self.move = Vector2(0, 0)
            return
            
        self.there_is_no_escape()
        
        if self.brain is not None and self.last_nn_output is not None:
            _, move_x, move_y, speed_control, _ = self.last_nn_output
            move_vec = Vector2(move_x.item(), move_y.item())
            
            if move_vec.length() > 0:
                move_vec = move_vec.normalize()
                base_speed = self.config.prey_flee_speed if self.state == PREY_FLEEING else self.config.prey_speed
                speed = base_speed * ((speed_control.item() + 1) / 2)
                self.move = move_vec * speed
        else:
            if random.random() < 0.5:
                # Fallback random movement
                angle = random.uniform(0, 2 * math.pi)
                speed = self.config.prey_flee_speed if self.state == PREY_FLEEING else self.config.prey_speed
                self.move = Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))


        self.pos += self.move * self.config.delta_time
    
    def start_being_eaten(self):
        self.state = PREY_EATEN
        self.change_image(PREY_EATEN)
        self.eating_timer = self.config.eating_duration

class NeuralPredator(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        global predators
        self.brain = copy.deepcopy(self.simulation.shared.best_predator_brain)
        self.state = PREDATOR_HUNTING
        self.energy = self.config.predator_energy
        self.eating_timer = 0
        self.target_prey = None
        self.change_image(PREDATOR_HUNTING)
        predators.add(self.id)
        self.age = 0
        self.last_nn_output = None
    
    def get_nn_state(self):
        nearest_prey = None
        prey_distance = self.config.predator_vision
        prey_count = 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPrey) and agent.alive:
                prey_count += 1
                if distance < prey_distance:
                    prey_distance, nearest_prey = distance, agent

        prey_dir = (nearest_prey.pos - self.pos).normalize() if nearest_prey else Vector2(0, 0)

        return torch.tensor([
            min(self.energy / self.config.predator_energy, 1.0),
            prey_distance / self.config.predator_vision,
            prey_dir.x, prey_dir.y,
            min(prey_count / 20.0, 1.0),
            1.0 if nearest_prey and prey_distance < 30 else 0.0,
            self.age / 1000.0  # Normalized age
        ], dtype=torch.float32)

    def update(self):
        global predators

        if self.state == PREDATOR_EATING:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.state = PREDATOR_HUNTING
                self.change_image(PREDATOR_HUNTING)
                self.target_prey = None
            return

        self.age += 1
        if self.age > 250 or self.energy <= 0:
            self.kill()
            predators.remove(self.id)
            return

        # Energy consumption based on state
        if self.state == PREDATOR_LUNGING:
            self.energy -= self.config.predator_lunge_energy_consumption
        else:
            self.energy -= self.config.predator_energy_consumption

        # Neural network decision making
        if self.brain is not None:
            with torch.no_grad():
                state = self.get_nn_state()
                output = self.brain(state)
                
                # Update state based on NN output
                self.state = PREDATOR_LUNGING if output[0] > 0 else PREDATOR_HUNTING
                self.change_image(self.state)
                
                # Handle reproduction
                if (output[4] > 0.7 and 
                    self.energy >= self.config.predator_reproduction_energy_threshold):
                    self.energy -= self.config.predator_reproduction_cost
                    child = self.reproduce()
                    child.brain = copy.deepcopy(self.brain)
                    predators.add(child.id)

                # Handle eating prey if nearby
                for agent, distance in self.in_proximity_accuracy():
                    if (isinstance(agent, NeuralPrey) and 
                        distance < self.config.predator_eating_threshold and
                        agent.state != PREY_EATEN):
                        agent.start_being_eaten()
                        self.energy += self.config.predator_eating_energy
                        self.state = PREDATOR_EATING
                        self.change_image(PREDATOR_EATING)
                        self.eating_timer = self.config.eating_duration
                        break

    def change_position(self):
        if self.state == PREDATOR_EATING:
            self.move = Vector2(0, 0)
            return
            
        self.there_is_no_escape()
        
        if self.brain is not None and self.last_nn_output is not None:
            _, move_x, move_y, speed_control, _ = self.last_nn_output
            move_vec = Vector2(move_x.item(), move_y.item())
            
            if move_vec.length() > 0:
                move_vec = move_vec.normalize()
                base_speed = self.config.predator_lunge_speed if self.state == PREDATOR_LUNGING else self.config.predator_speed
                speed = base_speed * ((speed_control.item() + 1) / 2)
                self.move = move_vec * speed
        else:
            if random.random() < 0.25:
                # Fallback random movement
                angle = random.uniform(0, 2 * math.pi)
                speed = self.config.predator_lunge_speed if self.state == PREDATOR_LUNGING else self.config.predator_speed
                self.move = Vector2(math.cos(angle), math.sin(angle)) * speed
            else:
                self.move = Vector2(random.uniform(-20, 20), random.uniform(-20, 20))

        self.pos += self.move * self.config.delta_time

def run_final_visual_simulation(best_prey: PreyBrain, best_predator: PredatorBrain, duration: int = 5000):
    cfg = PredatorPreyConfig(duration=duration)
    sim = Simulation(cfg)
    sim.shared.best_prey_brain = copy.deepcopy(best_prey)
    sim.shared.best_predator_brain = copy.deepcopy(best_predator)
    sim.batch_spawn_agents(cfg.initial_prey, NeuralPrey, images=PREY_IMAGES)
    sim.batch_spawn_agents(cfg.initial_predators, NeuralPredator, images=PREDATOR_IMAGES)
    sim.batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
    sim.spawn_agent(FoodSpawner, images=["images/transparent.png"])
    sim.spawn_agent(SimulationMonitor, images=["images/transparent.png"])

    print("[Running Final Visual Simulation...]")
    sim.run()

def evolve_brains(generations=10, population_size=20):
    best_prey = None
    best_predator = None
    best_score = -float("inf")

    for gen in range(generations):
        print(f"Generation {gen}")
        generation_scores = []

        for _ in range(population_size):
            prey_brain = PreyBrain()
            predator_brain = PredatorBrain()

            # Evaluate fitness
            score = evaluate_pair(prey_brain, predator_brain)
            generation_scores.append((score, prey_brain, predator_brain))

        generation_scores.sort(key=lambda x: x[0], reverse=True)
        top_score, best_prey, best_predator = generation_scores[0]
        print(f"Best score in generation {gen}: {top_score}")

    return best_prey, best_predator

def evaluate_pair(prey_brain, predator_brain):
    sim = NeuralPredatorPreySimulation(prey_brain, predator_brain, duration=1000)
    sim.run()
    df = pd.DataFrame(sim.shared.frame_data)
    
    # Example fitness: total number of prey surviving
    final_prey = df["prey_count"].iloc[-1]
    final_predators = df["predator_count"].iloc[-1]

    return final_prey + final_predators  # your fitness function here

if __name__ == "__main__":
    best_prey, best_predator = evolve_brains(generations=20, population_size=10)
    
    run_final_visual_simulation(
        best_prey=best_prey,
        best_predator=best_predator,
        duration=5000
    )