import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass, field
from vi import Agent, Config, Simulation, HeadlessSimulation
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math
from pygame.math import Vector2
import copy
import threading

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
predators = set()
foods = set()

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 3.0
    prey_flee_speed: float = 4.5
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.1
    prey_flee_energy_consumption: float = 0.15
    prey_energy_gain: float = 30.0
    prey_reproduction_energy_threshold: float = 150.0
    prey_reproduction_cost: float = 50.0
    prey_reproduction_radius: float = 20.0
    prey_growth_rate: float = 0.05
    prey_carrying_capacity: float = 100.0
    
    # Predator parameters
    predator_speed: float = 4.0
    predator_lunge_speed: float = 6.0
    predator_vision: float = 150.0
    predator_energy: float = 150.0
    predator_energy_consumption: float = 0.2
    predator_lunge_energy_consumption: float = 0.3
    predator_eating_threshold: float = 10.0
    predator_eating_energy: float = 30.0
    predator_reproduction_energy_threshold: float = 120.0
    predator_reproduction_cost: float = 60.0
    predator_reproduction_radius: float = 20.0
    predator_reproduce_prob: float = 0.05
    eating_distance: float = 10.0

    # Eating parameters
    eating_duration: int = 30
    
    # Food parameters
    food_spawn_rate: float = 0.005
    
    # Simulation parameters
    initial_prey: int = 50
    initial_predators: int = 10
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

class Prey(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        with threading.Lock():
            preys.add(self.id)
        self.change_image(0)
        self.energy = self.config.prey_energy
        self.alive = True
        self.state = PREY_WANDERING
    
    def update(self):
        global preys, foods

        # Energy loss per frame
        self.energy -= self.config.prey_energy_consumption
        
        # Death from starvation
        if self.energy <= 0:
            with threading.Lock():
                if self.id in preys:
                    preys.remove(self.id)
            self.alive = False
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
            if nearest_food.alive:
                nearest_food.alive = False
                nearest_food.kill()
                self.energy += self.config.prey_energy_gain
        
        # LOGISTIC REPRODUCTION
        current_prey_count = sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.alive)
        if current_prey_count > 0:
            growth_prob = self.config.prey_growth_rate * (1 - current_prey_count / self.config.prey_carrying_capacity)
            
            if (current_prey_count < self.config.prey_carrying_capacity and 
                random.random() < growth_prob and 
                self.energy > self.config.prey_reproduction_cost):
                
                self.energy -= self.config.prey_reproduction_cost
                child = self.reproduce()
                with threading.Lock():
                    preys.add(child.id)
    
    def change_position(self):
        self.there_is_no_escape()
        
        if random.random() < 0.05 or self.move.length() == 0:
            nearest_food = None
            min_food_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Food) and distance < min_food_distance:
                    min_food_distance = distance
                    nearest_food = agent
            
            if nearest_food and min_food_distance < 100:
                food_direction = (nearest_food.pos - self.pos).normalize()
                self.move = food_direction * self.config.prey_speed
            else:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_speed
        
        self.pos += self.move * self.config.delta_time
    
    def on_death(self):
        with threading.Lock():
            if self.id in preys:
                preys.remove(self.id)

class Predator(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        with threading.Lock():
            predators.add(self.id)
        self.change_image(0)
        self.energy = self.config.predator_energy
        self.alive = True
        self.state = PREDATOR_HUNTING
    
    def update(self):
        global preys, predators

        # Energy loss per frame
        self.energy -= self.config.predator_energy_consumption
        
        # Death from starvation
        if self.energy <= 0:
            with threading.Lock():
                if self.id in predators:
                    predators.remove(self.id)
            self.alive = False
            self.kill()
            return
        
        # Check for nearby prey
        nearest_prey = None
        min_prey_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Prey) and distance < min_prey_distance:
                min_prey_distance = distance
                nearest_prey = agent
        
        if nearest_prey and min_prey_distance < self.config.eating_distance:
            if nearest_prey.alive:
                nearest_prey.alive = False
                nearest_prey.kill()
                self.energy += self.config.predator_eating_energy

                # Reproduce upon eating
                if self.energy > self.config.predator_reproduction_energy_threshold and random.random() < self.config.predator_reproduce_prob:
                    self.energy -= self.config.predator_reproduction_cost
                    child = self.reproduce()
                    with threading.Lock():
                        predators.add(child.id)
    
    def change_position(self):
        self.there_is_no_escape()
        
        if random.random() < 0.05 or self.move.length() == 0:
            nearest_prey = None
            min_prey_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Prey) and distance < min_prey_distance:
                    min_prey_distance = distance
                    nearest_prey = agent
            
            if nearest_prey and min_prey_distance < 150:
                prey_direction = (nearest_prey.pos - self.pos).normalize()
                self.move = prey_direction * self.config.predator_speed
            else:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        self.pos += self.move * self.config.delta_time
    
    def on_death(self):
        with threading.Lock():
            if self.id in predators:
                predators.remove(self.id)

class SimulationMonitor(Agent):
    def on_spawn(self):
        self.simulation.shared.frame_data = {
            "frame": [],
            "prey_count": [],
            "predator_count": [],
            "food_count": []
        }

    def update(self):
        current_frame = self.simulation.shared.counter
        prey_count = sum(1 for a in self.simulation._agents if isinstance(a, (Prey, NeuralPrey)))
        predator_count = sum(1 for a in self.simulation._agents if isinstance(a, (Predator, NeuralPredator)))
        food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food))
        
        self.simulation.shared.frame_data["frame"].append(current_frame)
        self.simulation.shared.frame_data["prey_count"].append(prey_count)
        self.simulation.shared.frame_data["predator_count"].append(predator_count)
        self.simulation.shared.frame_data["food_count"].append(food_count)
        
        if current_frame > 5 and (prey_count == 0 or predator_count == 0):
            self.simulation.stop()
        if current_frame >= self.config.duration:
            self.simulation.stop()

# Neural Network Definitions
class PreyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class PredatorBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

@dataclass 
class NeuroEvoConfig:
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.1
    mutation_scale: float = 0.2
    crossover_rate: float = 0.7
    elite_size: int = 5
    sim_duration: int = 2000
    n_simulations: int = 3
    prey_test_population_size: int = 5
    predator_test_population_size: int = 5

class NeuroEvolution:
    def __init__(self, config: NeuroEvoConfig):
        self.config = config
        self.prey_population = []
        self.predator_population = []
        self.prey_fitness_history = []
        self.predator_fitness_history = []
        self.best_prey = None
        self.best_predator = None
        self.best_prey_fitness = -float('inf')
        self.best_predator_fitness = -float('inf')
        
        # Initialize populations
        for _ in range(config.population_size):
            self.prey_population.append(PreyBrain())
            self.predator_population.append(PredatorBrain())
        
        # Create test populations
        self.prey_test_population = [PreyBrain() for _ in range(config.prey_test_population_size)]
        self.predator_test_population = [PredatorBrain() for _ in range(config.predator_test_population_size)]
    
    def evaluate_prey(self, prey_net):
        """Evaluate prey against the test predator population"""
        total_fitness = 0.0
        
        for predator in self.predator_test_population:
            for _ in range(self.config.n_simulations):
                sim = NeuralPredatorPreySimulation(
                    prey_brain=prey_net,
                    predator_brain=predator,
                    duration=self.config.sim_duration
                )
                sim.run()
                
                df = pd.DataFrame(sim.shared.frame_data)
                if len(df) < 10:
                    return -1.0
                else:
                    duration = len(df) / self.config.sim_duration
                    prey_alive = df["prey_count"].iloc[-1] > 0
                    prey_std = df["prey_count"].std()
                    stability = 1 / (1 + prey_std)
                    
                    fitness = (
                        0.6 * duration +
                        0.2 * prey_alive +
                        0.2 * stability
                    )
                
                total_fitness += fitness
        
        return total_fitness / (self.config.n_simulations * len(self.predator_test_population))
    
    def evaluate_predator(self, predator_net):
        """Evaluate predators against the test prey population"""
        total_fitness = 0.0
        
        for prey in self.prey_test_population:
            for _ in range(self.config.n_simulations):
                sim = NeuralPredatorPreySimulation(
                    prey_brain=prey,
                    predator_brain=predator_net,
                    duration=self.config.sim_duration
                )
                sim.run()
                
                df = pd.DataFrame(sim.shared.frame_data)
                if len(df) < 10:
                    fitness = 0
                else:
                    duration = len(df) / self.config.sim_duration
                    pred_alive = df["predator_count"].iloc[-1] > 0
                    prey_eaten = df["prey_count"].iloc[0] - df["prey_count"].iloc[-1]
                    max_possible_eaten = df["prey_count"].iloc[0]
                    
                    fitness = (
                        0.5 * duration +
                        0.5 * (prey_eaten / max_possible_eaten if max_possible_eaten > 0 else 0)
                    )
                
                total_fitness += fitness
        
        return total_fitness / (self.config.n_simulations * len(self.prey_test_population))
    
    def evolve(self):
        for gen in tqdm(range(self.config.generations), desc="Evolving"):
            # Evaluate prey population
            prey_fitnesses = []
            for prey in self.prey_population:
                fitness = self.evaluate_prey(prey)
                prey_fitnesses.append(fitness)
                if fitness > self.best_prey_fitness:
                    self.best_prey_fitness = fitness
                    self.best_prey = copy.deepcopy(prey)
            
            # Evaluate predator population
            predator_fitnesses = []
            for predator in self.predator_population:
                fitness = self.evaluate_predator(predator)
                predator_fitnesses.append(fitness)
                if fitness > self.best_predator_fitness:
                    self.best_predator_fitness = fitness
                    self.best_predator = copy.deepcopy(predator)
            
            # Record best fitness
            best_prey_idx = np.argmax(prey_fitnesses)
            best_pred_idx = np.argmax(predator_fitnesses)
            self.prey_fitness_history.append(prey_fitnesses[best_prey_idx])
            self.predator_fitness_history.append(predator_fitnesses[best_pred_idx])
            
            print(f"\nGeneration {gen+1}:")
            print(f"  Best prey fitness: {prey_fitnesses[best_prey_idx]:.4f}")
            print(f"  Best predator fitness: {predator_fitnesses[best_pred_idx]:.4f}")
            
            # Create next generation
            self.prey_population = self._create_next_generation(
                self.prey_population, prey_fitnesses, PreyBrain)
            self.predator_population = self._create_next_generation(
                self.predator_population, predator_fitnesses, PredatorBrain)
            
            # Update test populations occasionally
            if gen % 5 == 0:
                self._update_test_populations()
    
    def _create_next_generation(self, population, fitnesses, net_class):
        """Super elitist approach with mutation based on best fitness individuals"""
        elite_indices = np.argsort(fitnesses)[-self.config.elite_size:]
        elite = [population[i] for i in elite_indices]
        
        new_population = []
        
        # Keep the elite unchanged
        new_population.extend(elite)
        
        # Create mutated copies of the elite
        while len(new_population) < self.config.population_size:
            parent = random.choice(elite)
            child = copy.deepcopy(parent)
            
            # Mutate all parameters
            for param in child.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.config.mutation_scale
                    mask = torch.rand_like(param) < self.config.mutation_rate
                    param.data += noise * mask
                    param.data.clamp_(-5.0, 5.0)  # ⬅️ clamp after mutation
            
            new_population.append(child)
        
        return new_population
    
    def _update_test_populations(self):
        """Update the test populations with current best individuals"""
        # Keep best from current test populations
        self.prey_test_population = sorted(
            self.prey_test_population + [self.best_prey],
            key=lambda x: self.evaluate_prey(x),
            reverse=True
        )[:self.config.prey_test_population_size]
        
        self.predator_test_population = sorted(
            self.predator_test_population + [self.best_predator],
            key=lambda x: self.evaluate_predator(x),
            reverse=True
        )[:self.config.predator_test_population_size]
    
    def visualize_best(self):
        """Run a visualization with the best found individuals"""
        if self.best_prey is None or self.best_predator is None:
            print("No best individuals found yet")
            return
        
        print("\nRunning visualization with best prey and predator...")
        
        cfg = PredatorPreyConfig(duration=5000)
        
        # Create agent spawn functions with access to the brains
        def spawn_neural_prey():
            prey = NeuralPrey(images=PREY_IMAGES)
            prey.brain = self.best_brain
            return prey
        
        def spawn_neural_predator():
            predator = NeuralPredator(images=PREDATOR_IMAGES)
            predator.brain = self.best_brain
            return predator
        
        sim = Simulation(cfg)
        sim.batch_spawn_agents(cfg.initial_prey, spawn_neural_prey)
        sim.batch_spawn_agents(cfg.initial_predators, spawn_neural_predator)
        sim.batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        sim.spawn_agent(FoodSpawner, images=["images/transparent.png"])
        sim.spawn_agent(SimulationMonitor, images=["images/transparent.png"])
        
        # Run the simulation
        sim.run()
        
        # Plot the results
        df = pd.DataFrame(sim.shared.frame_data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df["frame"], df["prey_count"], label="Prey")
        plt.plot(df["frame"], df["predator_count"], label="Predators")
        plt.plot(df["frame"], df["food_count"], label="Food")
        plt.title("Best Evolved Population Dynamics")
        plt.xlabel("Time Steps")
        plt.ylabel("Population Count")
        plt.legend()
        plt.grid(True)
        plt.savefig("best_evolved_population.png")
        plt.close()
        
        # Save fitness history plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.prey_fitness_history, label="Prey Fitness")
        plt.plot(self.predator_fitness_history, label="Predator Fitness")
        plt.title("Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig("fitness_history.png")
        plt.close()

class NeuralPrey(Prey):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = deque(maxlen=5)
        self.brain = None

    def get_state(self):
        nearest_pred = None
        nearest_food = None
        pred_distance = self.config.prey_vision
        food_distance = self.config.prey_vision

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, (Predator, NeuralPredator)) and distance < pred_distance:
                pred_distance = distance
                nearest_pred = agent
            elif isinstance(agent, Food) and distance < food_distance:
                food_distance = distance
                nearest_food = agent

        pred_dir = (nearest_pred.pos - self.pos).normalize() if nearest_pred else Vector2(0, 0)
        food_dir = (nearest_food.pos - self.pos).normalize() if nearest_food else Vector2(0, 0)

        return torch.tensor([
            self.energy / self.config.prey_energy,
            pred_distance / self.config.prey_vision,
            pred_dir.x, pred_dir.y,
            food_distance / self.config.prey_vision,
            food_dir.x, food_dir.y,
            len(self.memory) / 5
        ], dtype=torch.float32)

    def update(self):
        super().update()
        if self.brain is None or self.state == PREY_EATEN:
            return

        if self.state in [PREY_WANDERING, PREY_FLEEING]:
            state = self.get_state()
            with torch.no_grad():
                actions = self.brain(state)

            # Fallback to FSM if actions are near zero vector
            if actions[1].item() == 0 and actions[2].item() == 0:
                return

            if actions[0] > 0.7:
                self.state = PREY_FLEEING
                self.change_image(PREY_FLEEING)

            move_dir = Vector2(actions[1].item() - 0.5, actions[2].item() - 0.5)
            if move_dir.length() != 0:
                speed = self.config.prey_flee_speed if self.state == PREY_FLEEING else self.config.prey_speed
                self.move = move_dir.normalize() * speed

class NeuralPredator(Predator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = None

    def get_state(self):
        nearest_prey = None
        prey_distance = self.config.predator_vision

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, (Prey, NeuralPrey)) and distance < prey_distance:
                prey_distance = distance
                nearest_prey = agent

        prey_dir = (nearest_prey.pos - self.pos).normalize() if nearest_prey else Vector2(0, 0)

        return torch.tensor([
            self.energy / self.config.predator_energy,
            prey_distance / self.config.predator_vision,
            prey_dir.x, prey_dir.y,
            len([a for a in self.simulation._agents if isinstance(a, (Prey, NeuralPrey))]) / 100,
            len([a for a in self.simulation._agents if isinstance(a, (Predator, NeuralPredator))]) / 50
        ], dtype=torch.float32)

    def update(self):
        super().update()
        if self.brain is None or self.state == PREDATOR_EATING:
            return

        if self.state in [PREDATOR_HUNTING, PREDATOR_LUNGING]:
            state = self.get_state()
            with torch.no_grad():
                actions = self.brain(state)

            # Fallback to FSM if output is non-informative
            if actions[1].item() == 0 and actions[2].item() == 0:
                return

            if actions[0] > 0.7:
                self.state = PREDATOR_LUNGING
                self.change_image(PREDATOR_LUNGING)

            move_dir = Vector2(actions[1].item() - 0.5, actions[2].item() - 0.5)
            if move_dir.length() != 0:
                speed = self.config.predator_lunge_speed if self.state == PREDATOR_LUNGING else self.config.predator_speed
                self.move = move_dir.normalize() * speed

class NeuralPredatorPreySimulation(HeadlessSimulation):
    def __init__(self, prey_brain, predator_brain, duration=10000):
        super().__init__(PredatorPreyConfig(duration=duration))
        self.prey_brain = prey_brain
        self.predator_brain = predator_brain
        self.shared.frame_data = {
            "frame": [],
            "prey_count": [],
            "predator_count": [],
            "food_count": []
        }
    
    def run(self):
        # Create agent spawn functions with access to the brains
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

if __name__ == "__main__":
    config = NeuroEvoConfig(
        population_size=30,
        generations=20,
        sim_duration=1500,
        n_simulations=2
    )
    
    neuro_evo = NeuroEvolution(config)
    neuro_evo.evolve()
    
    # Save the best models
    if neuro_evo.best_prey:
        torch.save(neuro_evo.best_prey.state_dict(), "best_prey.pth")
    if neuro_evo.best_predator:
        torch.save(neuro_evo.best_predator.state_dict(), "best_predator.pth")
    
    # Run visualization with the best individuals
    neuro_evo.visualize_best()