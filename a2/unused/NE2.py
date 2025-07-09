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

def evaluate_prey_task(args):
    prey_net, predator_list, config = args
    total_fitness = 0.0
    for idx, predator in enumerate(predator_list):
        for _ in range(config.n_simulations):
            sim = NeuralPredatorPreySimulation(prey_net, predator, config.sim_duration)
            sim.run()
            df = pd.DataFrame(sim.shared.frame_data)
            if len(df) < 10:
                fitness = 0.0
            else:
                survival_time = len(df) / config.sim_duration
                final_prey_count = df["prey_count"].iloc[-1]
                avg_prey_count = df["prey_count"].mean()
                initial_prey_count = df["prey_count"].iloc[0]
                fitness = (
                    0.5 * survival_time +
                    0.3 * (final_prey_count / max(initial_prey_count, 1)) +
                    0.2 * (avg_prey_count / max(initial_prey_count, 1))
                )
            total_fitness += fitness
    return total_fitness / (config.n_simulations * len(predator_list))


def evaluate_predator_task(args):
    predator_net, prey_list, config = args
    total_fitness = 0.0
    for idx, prey in enumerate(prey_list):
        for _ in range(config.n_simulations):
            sim = NeuralPredatorPreySimulation(prey, predator_net, config.sim_duration)
            sim.run()
            df = pd.DataFrame(sim.shared.frame_data)
            if len(df) < 10:
                fitness = 0.0
            else:
                initial_prey = df["prey_count"].iloc[0]
                final_prey = df["prey_count"].iloc[-1]
                prey_eaten = max(0, initial_prey - final_prey)
                survival_time = len(df) / config.sim_duration
                final_predators = df["predator_count"].iloc[-1]
                initial_predators = df["predator_count"].iloc[0]
                fitness = (
                    0.4 * (prey_eaten / max(initial_prey, 1)) +
                    0.3 * survival_time +
                    0.3 * (final_predators / max(initial_predators, 1))
                )
            total_fitness += fitness
    return total_fitness / (config.n_simulations * len(prey_list))

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 5.5
    prey_flee_speed: float = 7.25
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.1
    prey_flee_energy_consumption: float = 0.15
    prey_energy_gain: float = 35.0  # Energy gained from eating food
    prey_reproduction_energy_threshold: float = 150.0
    prey_reproduction_cost: float = 50.0
    prey_reproduction_radius: float = 25.0
    prey_reproduction_probability: float = 0.15
    
    # Predator parameters
    predator_speed: float = 6.25
    predator_lunge_speed: float = 8.0
    predator_vision: float = 150.0
    predator_energy: float = 110.0
    predator_energy_consumption: float = 0.2
    predator_lunge_energy_consumption: float = 0.4
    predator_eating_threshold: float = 10.0
    predator_eating_energy: float = 75.0
    predator_reproduction_energy_threshold: float = 70.0
    predator_reproduction_cost: float = 44.0
    predator_reproduction_radius: float = 25.0  # Distance for reproduction

    # Eating parameters
    eating_duration: int = 22  # Frames prey shows being eaten / predator is eating
    
    # Food parameters
    food_spawn_rate: float = 0.15  # Probability of food spawning per frame
    
    # Simulation parameters
    initial_prey: int = 45
    initial_predators: int = 22
    initial_food: int = 15

    # Setting parameters
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
        global preys
        self.state = PREY_WANDERING
        self.energy = self.config.prey_energy
        self.eating_timer = 0
        self.change_image(PREY_WANDERING)
        preys.add(self.id)
        self.type = "prey"
        self.age = 0
    
    def update(self):
        global preys

        self.age += 1
        if self.age > 700:
            self.kill()
            preys.remove(self.id)
            return
        
        if self.energy <= 0:
            self.kill()
            preys.remove(self.id)
            return

        if self.state == PREY_EATEN:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.kill()
                preys.remove(self.id)
            return
        
        if self.state == PREY_FLEEING:
            self.energy -= self.config.prey_flee_energy_consumption
        else:
            if not self.on_site():
                self.energy -= self.config.prey_energy_consumption
        
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
            self.age -= 100

        # Reproduction - only if another prey is nearby and has enough energy
        if (self.energy >= self.config.prey_reproduction_energy_threshold and 
            (random.random() < self.config.prey_reproduction_probability) or (random.random() < 0.00000025 and self.simulation.shared.counter > 2000)):  # Small chance to check each frame
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Prey):
                    if (distance < self.config.prey_reproduction_radius and 
                        agent.energy >= self.config.prey_reproduction_energy_threshold):
                        # Only one of them reproduces (random choice)
                            self.energy -= self.config.prey_reproduction_cost
                            child = self.reproduce()
                            preys.add(child.id)
                            break
    
    def change_position(self):
        if self.state == PREY_EATEN:
            self.move = Vector2(0, 0)  # Don't move while being eaten
            self.alive = False
            return
            
        self.there_is_no_escape()
        
        if self.state == PREY_WANDERING:
            # Random wandering with occasional food seeking
            # Sometimes move toward food if visible
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
                    food_direction = (nearest_food.pos - self.pos).normalize()
                self.move = food_direction * self.config.prey_speed
            else:
                # Random wandering
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_speed
        else:  # FLEEING
            # Flee from nearest predator
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
                # No predator nearby, fallback movement
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_flee_speed

        #site_center = Vector2(500*3/5, 500/2)
        #if self.state == PREY_WANDERING and self.pos.distance_to(site_center) < 100:
            #if random.random() < 0.9:
            #    self.move = self.move * 0.975
            #else:
            #    self.move = self.move * 1.65

        self.pos += self.move * self.config.delta_time
    
    def start_being_eaten(self):
        """Transition to being eaten state"""
        self.state = PREY_EATEN
        self.change_image(PREY_EATEN)
        self.eating_timer = self.config.eating_duration


class Predator(Agent):
    config: PredatorPreyConfig
    
    def on_spawn(self):
        global predators
        self.state = PREDATOR_HUNTING
        self.energy = self.config.predator_energy
        self.eating_timer = 0
        self.target_prey = None
        self.change_image(PREDATOR_HUNTING)
        predators.add(self.id)
        self.age = 0
        #self.avoided = False  # Track if predator has avoided prey-only site
    
    def update(self):
        global predators
        self.age += 1
        if self.age > 350:
            self.kill()
            predators.remove(self.id)
            return

        if self.state == PREDATOR_LUNGING:
            self.energy -= self.config.predator_lunge_energy_consumption
        else:
            self.energy -= self.config.predator_energy_consumption
        
        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            predators.remove(self.id)
            return
        
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
            if isinstance(agent, Prey) and agent.state != PREY_EATEN:
                if distance < min_distance:
                    min_distance = distance
                    nearest_prey = agent

                # Check for reproduction with nearby predators
                if self.energy >= self.config.predator_reproduction_energy_threshold:
                    for agent, distance in self.in_proximity_accuracy():
                        if isinstance(agent, Predator):
                            if (distance < self.config.predator_reproduction_radius and 
                                agent.energy >= self.config.predator_reproduction_energy_threshold and
                                self.age > 100 and agent.age > 100):
                                # Both predators can reproduce
                                self.energy -= self.config.predator_reproduction_cost
                                agent.energy -= self.config.predator_reproduction_cost
                                child = self.reproduce()
                                predators.add(child.id)
                                if random.random() < 0.5:
                                        child2 = agent.reproduce()
                                        predators.add(child2.id)
                                        if random.random() < 0.5:
                                            child3 = self.reproduce()
                                            predators.add(child3.id)
                                break
        
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
                # Caught prey!
                if nearest_prey.state != PREY_EATEN:  # Only if not already being eaten
                    nearest_prey.start_being_eaten()
                    self.energy += self.config.predator_eating_energy
                    self.state = PREDATOR_EATING
                    self.change_image(PREDATOR_EATING)
                    self.eating_timer = self.config.eating_duration
                    self.target_prey = None       
    
    def change_position(self):
        #if self.avoided:
            #self.avoided = False

        if self.state == PREDATOR_EATING:
            # Stay still while eating
            self.move = Vector2(0, 0)
        elif self.state == PREDATOR_LUNGING and self.target_prey:
            # Lunge toward target prey
            delta = self.target_prey.pos - self.pos
            if delta.length() == 0:
                lunge_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            else:
                lunge_direction = (self.target_prey.pos - self.pos).normalize()
            self.move = lunge_direction * self.config.predator_lunge_speed
        else:
            # Random hunting movement
            angle = random.uniform(0, 2 * math.pi)
            self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        # Redirect predator away if it tries to enter prey-only site
        #future_pos = self.pos + self.move * self.config.delta_time
        #distance_to_site = future_pos.distance_to(Vector2(500*3/4, 500/2))

        #if distance_to_site < 175 or self.on_site():
            # Push predator away from the center of the site
            #avoidance_dir = (future_pos - Vector2(500*3/4, 500/2)).normalize()
            #self.move = avoidance_dir * self.config.predator_speed
            #self.avoided = True

        #if self.avoided:
            #self.move = -self.move

        self.there_is_no_escape()

        self.pos += self.move * self.config.delta_time

# === Patched SimulationMonitor with robust stop condition ===
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
            prey_count = sum(1 for a in self.simulation._agents if isinstance(a, (Prey, NeuralPrey)) and a.alive)
            predator_count = sum(1 for a in self.simulation._agents if isinstance(a, (Predator, NeuralPredator)) and a.alive)
            food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food) and a.alive)

            # Ensure we append to all arrays in the same update
            self.simulation.shared.frame_data["frame"].append(current_frame)
            self.simulation.shared.frame_data["prey_count"].append(prey_count)
            self.simulation.shared.frame_data["predator_count"].append(predator_count)
            self.simulation.shared.frame_data["food_count"].append(food_count)

            if current_frame > 5 and (prey_count == 0 or predator_count == 0):
                print(f"[Monitor] Stopping: prey={prey_count}, predator={predator_count}")
                self.simulation.stop()

            if current_frame >= self.config.duration:
                print("[Monitor] Reached max duration, stopping.")
                self.simulation.stop()

        except Exception as e:
            print(f"[Monitor] Exception in update(): {e}")
            self.simulation.stop()

class NeuralPredatorPreySimulation(Simulation):
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

# Neural Network Definitions
class PreyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh for better gradient flow
        return x

class PredatorBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh for better gradient flow
        return x

@dataclass 
class NeuroEvoConfig:
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    mutation_scale: float = 0.2
    crossover_rate: float = 0.7
    elite_size: int = 5
    sim_duration: int = 5000
    n_simulations: int = 2
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
        self.initial_mutation_rate = self.config.mutation_rate
        self.initial_mutation_scale = self.config.mutation_scale

        
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
                global preys, predators
                preys.clear()  # Clear global prey set for each simulation
                predators.clear()

                sim = NeuralPredatorPreySimulation(
                    prey_brain=prey_net,
                    predator_brain=predator,
                    duration=self.config.sim_duration
                )
                sim.run()
                
                df = pd.DataFrame(sim.shared.frame_data)
                if len(df) < 10:
                    fitness = 0.0
                else:
                    # Reward survival time and maintaining population
                    survival_time = len(df) / self.config.sim_duration
                    final_prey_count = df["prey_count"].iloc[-1]
                    avg_prey_count = df["prey_count"].mean()
                    initial_prey_count = df["prey_count"].iloc[0]
                    
                    # Better fitness function for prey
                    fitness = (
                        0.5 * survival_time +  # Reward surviving longer
                        0.3 * (final_prey_count / max(initial_prey_count, 1)) +  # Final survival rate
                        0.2 * (avg_prey_count / max(initial_prey_count, 1))  # Average population maintenance
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
                    fitness = 0.0
                else:
                    # Reward hunting efficiency and survival
                    initial_prey = df["prey_count"].iloc[0]
                    final_prey = df["prey_count"].iloc[-1]
                    prey_eaten = max(0, initial_prey - final_prey)
                    survival_time = len(df) / self.config.sim_duration
                    final_predator_count = df["predator_count"].iloc[-1]
                    initial_predator_count = df["predator_count"].iloc[0]
                    
                    # Better fitness function for predators
                    hunt_efficiency = prey_eaten / max(initial_prey, 1)
                    survival_rate = final_predator_count / max(initial_predator_count, 1)
                    
                    fitness = (
                        0.4 * hunt_efficiency +  # Reward successful hunting
                        0.3 * survival_time +    # Reward survival
                        0.3 * survival_rate     # Reward population survival
                    )
                
                total_fitness += fitness
        
        return total_fitness / (self.config.n_simulations * len(self.prey_test_population))
    
    def evolve(self):
        for gen in tqdm(range(self.config.generations), desc="Evolving"):
            # Evaluate prey population sequentially
            prey_fitnesses, predator_fitnesses = [], []
            for idx, (prey, pred) in enumerate(zip(self.prey_population, self.predator_population)):
                args = (deepcopy(prey), deepcopy(self.predator_test_population), self.config)
                prey_fitnesses.append(evaluate_prey_task(args))
                print(f"Prey {(idx+1)*5}/{len(self.prey_population)} fitness: {prey_fitnesses[-1]:.4f}")
                args = (deepcopy(pred), deepcopy(self.prey_test_population), self.config)
                predator_fitnesses.append(evaluate_predator_task(args))
                print(f"Predator {(idx+1)*5}/{len(self.predator_population)} fitness: {predator_fitnesses[-1]:.4f}")
            
            # Evaluate predator population sequentially
            #predator_fitnesses = []
            #for pred in self.predator_population:
            #    args = (deepcopy(pred), deepcopy(self.prey_test_population), self.config)
            #    predator_fitnesses.append(evaluate_predator_task(args))
            
            # Record best fitness
            if prey_fitnesses:
                best_prey_idx = np.argmax(prey_fitnesses)
                self.prey_fitness_history.append(prey_fitnesses[best_prey_idx])

                if prey_fitnesses[best_prey_idx] > self.best_prey_fitness:
                    self.best_prey = copy.deepcopy(self.prey_population[best_prey_idx])
                    self.best_prey_fitness = prey_fitnesses[best_prey_idx]

            if predator_fitnesses:
                best_pred_idx = np.argmax(predator_fitnesses)
                self.predator_fitness_history.append(predator_fitnesses[best_pred_idx])

                if predator_fitnesses[best_pred_idx] > self.best_predator_fitness:
                    self.best_predator = copy.deepcopy(self.predator_population[best_pred_idx])
                    self.best_predator_fitness = predator_fitnesses[best_pred_idx]
            
            print(f"\nGeneration {gen+1}:")
            if prey_fitnesses:
                print(f"  Best prey fitness: {max(prey_fitnesses):.4f}")
            if predator_fitnesses:
                print(f"  Best predator fitness: {max(predator_fitnesses):.4f}")
            
            # Create next generation
            self.prey_population = self._create_next_generation(
                self.prey_population, prey_fitnesses, PreyBrain)
            self.predator_population = self._create_next_generation(
                self.predator_population, predator_fitnesses, PredatorBrain)
            
            # Update test populations occasionally
            if gen % 5 == 0:
                self._update_test_populations()

            # Adaptive mutation decay
            decay = 0.95  # Decay factor per generation (tweak as needed)
            self.config.mutation_rate *= decay
            self.config.mutation_scale *= decay

            # Optional: enforce lower bound
            self.config.mutation_rate = max(self.config.mutation_rate, 0.01)
            self.config.mutation_scale = max(self.config.mutation_scale, 0.01)
    
    def _create_next_generation(self, population, fitnesses, net_class):
        """Create next generation using elitism and mutation"""
        # Convert to numpy for easier manipulation
        fitnesses = np.array(fitnesses)
        
        # Handle case where all fitnesses are the same or negative
        if np.all(fitnesses <= 0) or np.std(fitnesses) == 0:
            # If no good solutions, create random population but keep one elite
            best_idx = np.argmax(fitnesses)
            new_population = [copy.deepcopy(population[best_idx])]
            for _ in range(self.config.population_size - 1):
                new_population.append(net_class())
            return new_population
        
        # Get top-N elite indices
        elite_indices = np.argsort(fitnesses)[-self.config.elite_size:]
        elite = [population[i] for i in elite_indices]
        self.elite_archive = elite  # Optional: store for later

        new_population = [copy.deepcopy(population[i]) for i, _ in elite]

        # Normalize fitnesses for selection (handle negative values)
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-8
        
        # Probability-based selection for breeding
        fitness_sum = np.sum(fitnesses)
        if fitness_sum > 0:
            selection_probs = fitnesses / fitness_sum
        else:
            selection_probs = np.ones(len(fitnesses)) / len(fitnesses)
        
        # Create rest of population through mutation and crossover
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(elite) >= 2:
                # Crossover between two elite parents
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1, parent2, net_class)
            else:
                # Select parent based on fitness
                parent_idx = np.random.choice(len(population), p=selection_probs)
                child = copy.deepcopy(population[parent_idx])
            
            # Mutate child
            self._mutate(child)
            new_population.append(child)
        
        return new_population
    
    def _crossover(self, parent1, parent2, net_class):
        """Create child through crossover of two parents"""
        child = net_class()
        
        # Blend parameters from both parents
        for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            if random.random() < 0.5:
                child_param.data.copy_(p1_param.data)
            else:
                child_param.data.copy_(p2_param.data)
        
        return child
    
    def _mutate(self, network):
        """Mutate network parameters"""
        for param in network.parameters():
            if param.requires_grad:
                # Add Gaussian noise to parameters
                noise = torch.randn_like(param) * self.config.mutation_scale
                mask = torch.rand_like(param) < self.config.mutation_rate
                param.data += noise * mask
                # Clamp to prevent extreme values
                param.data.clamp_(-5.0, 5.0)
    
    def _update_test_populations(self):
        """Update the test populations with current best individuals"""
        if self.best_prey is not None:
            # Replace worst in test population with current best
            for i in range(min(len(self.prey_test_population), self.config.elite_size)):
                self.prey_test_population[i] = copy.deepcopy(self.prey_population[self._get_elite_indices(self.prey_fitness_history[-1:], self.prey_population)[i]])

        if self.best_predator is not None:
            #self.predator_test_population[0] = copy.deepcopy(self.best_predator)
            for i in range(min(len(self.predator_test_population), self.config.elite_size)):
                self.predator_test_population[i] = copy.deepcopy(self.predator_population[self._get_elite_indices(self.predator_fitness_history[-1:], self.predator_population)[i]])
    
    def visualize_best(self):
        """Run a visualization with the best found individuals"""
        if self.best_prey is None or self.best_predator is None:
            print("No best individuals found yet")
            return
        
        print("\nRunning visualization with best prey and predator...")
        
        cfg = PredatorPreyConfig(duration=5000)
        
        # Create the simulation
        sim = Simulation(cfg)
        
        # Spawn neural agents with the best brains
        for _ in range(cfg.initial_prey):
            prey = NeuralPrey(simulation=sim, images=PREY_IMAGES)
            prey.brain = copy.deepcopy(self.best_prey)
            sim.spawn_agent_instance(prey)
        
        for _ in range(cfg.initial_predators):
            predator = NeuralPredator(simulation=sim, images=PREDATOR_IMAGES)
            predator.brain = copy.deepcopy(self.best_predator)
            sim.spawn_agent_instance(predator)
        
        # Spawn regular agents
        sim.batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        sim.spawn_agent(FoodSpawner, images=["images/transparent.png"])
        sim.spawn_agent(SimulationMonitor, images=["images/transparent.png"])
        
        # Run the simulation
        sim.run()
        
        # Plot the results
        df = pd.DataFrame(sim.shared.frame_data)
        
        plt.figure(figsize=(12, 8))
        
        # Population dynamics plot
        plt.subplot(2, 1, 1)
        plt.plot(df["frame"], df["prey_count"], label="Prey", color='green')
        plt.plot(df["frame"], df["predator_count"], label="Predators", color='red')
        plt.plot(df["frame"], df["food_count"], label="Food", color='blue')
        plt.title("Best Evolved Population Dynamics")
        plt.xlabel("Time Steps")
        plt.ylabel("Population Count")
        plt.legend()
        plt.grid(True)
        
        # Fitness history plot
        plt.subplot(2, 1, 2)
        plt.plot(self.prey_fitness_history, label="Prey Fitness", color='green')
        plt.plot(self.predator_fitness_history, label="Predator Fitness", color='red')
        plt.title("Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("evolution_results.png", dpi=300, bbox_inches='tight')
        plt.show()

# === Hybrid Neural + FSM Simulation for Prey ===
class NeuralPrey(Prey):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = None
        self.steps_since_decision = 0
        self.decision_interval = 1
        self.confidence = 0.0  # Smoothness factor from FSM to NN

    def get_nn_state(self):
        nearest_pred, nearest_food = None, None
        pred_distance, food_distance = self.config.prey_vision, self.config.prey_vision
        pred_count, prey_count = 0, 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, (Predator, NeuralPredator)) and agent.alive and distance < pred_distance:
                pred_distance, nearest_pred = distance, agent
                pred_count += 1
            elif isinstance(agent, Food) and agent.alive and distance < food_distance:
                food_distance, nearest_food = distance, agent
            elif isinstance(agent, (Prey, NeuralPrey)) and agent.alive and distance < self.config.prey_vision:
                prey_count += 1

        pred_dir = (nearest_pred.pos - self.pos).normalize() if nearest_pred else Vector2(0, 0)
        food_dir = (nearest_food.pos - self.pos).normalize() if nearest_food else Vector2(0, 0)

        return torch.tensor([
            min(self.energy / self.config.prey_energy, 1.0),
            1.0 if self.energy < 0.2 * self.config.prey_energy else 0.0,  # 1: low energy flag
            pred_distance / self.config.prey_vision,
            pred_dir.x, pred_dir.y,
            min(pred_count / 10.0, 1.0),             # 5: number of predators nearby (capped at 10)
            food_distance / self.config.prey_vision,
            food_dir.x, food_dir.y,
            1.0 if nearest_pred and pred_distance < 50 else 0.0,
            1.0 if self.state == PREY_FLEEING else 0.0,  # 9: fleeing state flag
            min(prey_count / 20.0, 1.0),  # 10: number of other prey nearby (capped at 20)
        ], dtype=torch.float32)

    def update(self):
        super().update()
        if not self.alive or self.brain is None:
            return

        self.age += 1
        self.confidence = min(max(self.age / 500.0, 0.0), 1.0)

        self.steps_since_decision += 1
        if self.steps_since_decision >= self.decision_interval:
            self.steps_since_decision = 0
            with torch.no_grad():
                self.last_nn_output = self.brain(self.get_nn_state())

    def change_position(self):
        super().change_position()
        if not hasattr(self, "last_nn_output") or self.brain is None:
            return

        _, move_x, move_y, speed_ctrl = self.last_nn_output
        move_vec = Vector2(move_x.item(), move_y.item())

        if move_vec.length() > 0:
            move_vec = move_vec.normalize()
            base_speed = self.config.prey_flee_speed if self.state == PREY_FLEEING else self.config.prey_speed
            nn_move = move_vec * base_speed * ((speed_ctrl.item() + 1) / 2)
            self.move = self.move.lerp(nn_move, self.confidence)

        if self.state == PREY_EATEN:
            # Don't move while being eaten
            self.move = Vector2(0, 0)

        if self.there_is_no_escape():
            # If there is no escape, reverse direction
            self.move = -self.move

        self.pos += self.move * self.config.delta_time

    #def reproduce(self):
    #    child = super().reproduce()
    #    if isinstance(child, NeuralPrey):
    #        child.brain = copy.deepcopy(self.brain)
    #    return child


# === Hybrid Neural + FSM Simulation for Predator ===

class NeuralPredator(Predator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brain = None
        self.steps_since_decision = 0
        self.decision_interval = 5
        self.confidence = 0.0

    def get_nn_state(self):
        nearest_prey = None
        prey_distance = self.config.predator_vision
        pred_count, prey_count = 0, 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, (Prey, NeuralPrey)) and agent.alive and agent.state != PREY_EATEN:
                prey_count += 1
                if distance < prey_distance:
                    prey_distance, nearest_prey = distance, agent
            elif isinstance(agent, (Predator, NeuralPredator)) and agent.alive and agent.state != PREDATOR_EATING:
                if distance < self.config.predator_vision:
                    pred_count += 1

        prey_dir = (nearest_prey.pos - self.pos).normalize() if nearest_prey else Vector2(0, 0)

        return torch.tensor([
            min(self.energy / self.config.predator_energy, 1.0),
            prey_distance / self.config.predator_vision,
            prey_dir.x, prey_dir.y,
            min(prey_count / 20.0, 1.0),
            1.0 if nearest_prey and prey_distance < 30 else 0.0,
            1.0 if self.state == PREDATOR_LUNGING else 0.0,  # 5: lunging state flag
            min(pred_count / 10.0, 1.0),  # 6: number of predators nearby (capped at 10)
            (self.eating_timer / self.config.eating_duration) if self.state == PREDATOR_EATING else 0.0,  # 7: eating timer normalized
        ], dtype=torch.float32)

    def update(self):
        super().update()
        if not self.alive or self.brain is None:
            return

        self.age += 1
        self.confidence = min(max(self.age / 500.0, 0.0), 1.0)

        self.steps_since_decision += 1
        if self.steps_since_decision >= self.decision_interval:
            self.steps_since_decision = 0
            with torch.no_grad():
                self.last_nn_output = self.brain(self.get_nn_state())

    def change_position(self):
        super().change_position()
        if not hasattr(self, "last_nn_output") or self.brain is None:
            return

        _, move_x, move_y, speed_ctrl = self.last_nn_output
        move_vec = Vector2(move_x.item(), move_y.item())
        if move_vec.length() == 0:
            move_vec = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()

        if move_vec.length() > 0:
            move_vec = move_vec.normalize()
            base_speed = self.config.predator_lunge_speed if self.state == PREDATOR_LUNGING else self.config.predator_speed
            nn_move = move_vec * base_speed * ((speed_ctrl.item() + 1) / 2)
            self.move = self.move.lerp(nn_move, self.confidence)

                # Push away if predator is inside prey-only site
        #future_pos = self.pos + self.move * self.config.delta_time
        #site_center = Vector2(500 * 3 / 4, 500 / 2)

        #if future_pos.distance_to(site_center) < 175 or self.on_site():
        #    avoidance_dir = (future_pos - site_center).normalize()
        #    self.move = avoidance_dir * self.config.predator_speed

        if self.state == PREDATOR_EATING:
            # Stay still while eating
            self.move = Vector2(0, 0)

        if self.there_is_no_escape():
            self.move = -self.move

        self.pos += self.move * self.config.delta_time

    def reproduce(self):
        child = super().reproduce()
        if isinstance(child, NeuralPredator):
            child.brain = copy.deepcopy(self.brain)
        return child


def run_final_visual_simulation(best_prey: PreyBrain, best_predator: PredatorBrain, duration: int = 5000):
    """
    Runs a visual simulation using the best evolved prey and predator brains.

    @param best_prey - Trained PyTorch model for prey behavior
    @param best_predator - Trained PyTorch model for predator behavior
    @param duration - Duration of the simulation in frames (default: 5000)
    """
    cfg = PredatorPreyConfig(duration=duration)

    # Create simulation
    sim = Simulation(cfg)

    # Spawn neural prey with best brain
    for _ in range(cfg.initial_prey):
        prey = NeuralPrey(simulation=sim, images=PREY_IMAGES)
        prey.brain = copy.deepcopy(best_prey)
        sim.spawn_agent_instance(prey)

    # Spawn neural predators with best brain
    for _ in range(cfg.initial_predators):
        predator = NeuralPredator(simulation=sim, images=PREDATOR_IMAGES)
        predator.brain = copy.deepcopy(best_predator)
        sim.spawn_agent_instance(predator)

    # Add food and spawners
    sim.batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
    sim.spawn_agent(FoodSpawner, images=["images/transparent.png"])
    sim.spawn_agent(SimulationMonitor, images=["images/transparent.png"])

    # Run visual simulation
    print("[Running Final Visual Simulation...]")
    sim.run()

if __name__ == "__main__":
    # Evolve and get the best brains
    config = NeuroEvoConfig()
    neuro_evo = NeuroEvolution(config)
    neuro_evo.evolve()

    # Visualize final result
    run_final_visual_simulation(
        best_prey=neuro_evo.best_prey,
        best_predator=neuro_evo.best_predator,
        duration=5000
    )