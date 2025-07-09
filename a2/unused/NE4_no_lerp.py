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
    "images/red.png"  ,     # Prey being eaten (2)
    "images/blue.png",       # Prey conscious wandering (3)
    "images/violet.png"     # Prey conscious fleeing (4)

]
PREDATOR_IMAGES = [
    "images/triangle5.png",  # Predator hunting (0)
    "images/triangle7.png",  # Predator lunging (1)
    "images/triangle6.png",   # Predator eating (2)
    "images/triangle4.png",   # Predator conscious hunting (3)
    "images/triangle8.png"    # Predator conscious lunging (4)
]
FOOD_IMAGES = [
    "images/plus.png"       # Food (0)
]

# State constants
PREY_WANDERING = 0
PREY_FLEEING = 1
PREY_EATEN = 2
PREY_CONSCIOUS_W = 3
PREY_CONSCIOUS_F = 4
PREDATOR_HUNTING = 0
PREDATOR_LUNGING = 1
PREDATOR_EATING = 2
PREDATOR_CONSCIOUS_H = 3
PREDATOR_CONSCIOUS_L = 4

CONSCIOUSNESS_THRESHOLD = 0.75  # Threshold for consciousness

preys = set()  
predators = set()  # Global sets to track prey and predator IDs

all_training_prey_data = []  # Store all training data for prey
all_training_predator_data = []  # Store all training data for predators
run_id_glb = 0  # Global run ID for tracking simulations

def evaluate_prey_task(args):
    prey_net, predator_list, config = args
    total_fitness = 0.0
    for idx, predator in enumerate(predator_list):
        for _ in range(config.n_simulations):
            print(f"Evaluating prey {(idx + 1) * _}/{len(predator_list)*config.n_simulations}")
            global preys, predators, all_training_prey_data, run_id_glb
            preys.clear()
            predators.clear()
            # Clear global sets for each simulation
            sim = NeuralPredatorPreySimulation(prey_net, predator, config.sim_duration)
            sim.run()
            df = pd.DataFrame(sim.shared.frame_data)
            df["run"] = run_id_glb
            run_id_glb += 1
            all_training_prey_data.append(df)
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
            print(f"Evaluating predator {(idx + 1) * _}/{len(prey_list)*config.n_simulations}")
            global preys, predators, all_training_predator_data, run_id_glb
            preys.clear()
            predators.clear()
            sim = NeuralPredatorPreySimulation(prey, predator_net, config.sim_duration)
            sim.run()
            df = pd.DataFrame(sim.shared.frame_data)
            df["run"] = run_id_glb
            run_id_glb += 1
            all_training_predator_data.append(df)
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

def parallel_evaluation(prey_population, predator_test_population, predator_population, prey_test_population, config):
    with Pool(processes=cpu_count()) as pool:
        prey_args = [(copy.deepcopy(prey), copy.deepcopy(predator_test_population), config) for prey in prey_population]
        predator_args = [(copy.deepcopy(pred), copy.deepcopy(prey_test_population), config) for pred in predator_population]

        prey_fitnesses = pool.map(evaluate_prey_task, prey_args)
        predator_fitnesses = pool.map(evaluate_predator_task, predator_args)

    return prey_fitnesses, predator_fitnesses

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 12
    prey_flee_speed: float = 18.5
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.3
    prey_flee_energy_consumption: float = 0.45
    prey_energy_gain: float = 35.0  # Energy gained from eating food
    prey_reproduction_energy_threshold: float = 150.0
    prey_reproduction_cost: float = 50.0
    prey_reproduction_radius: float = 25.0
    prey_reproduction_probability: float = 0.125
    
    # Predator parameters
    predator_speed: float = 15
    predator_lunge_speed: float = 20
    predator_vision: float = 150.0
    predator_energy: float = 110.0
    predator_energy_consumption: float = 0.7
    predator_lunge_energy_consumption: float = 1.3
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
    duration: int = 10

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
    def __init__(self, prey_brain, predator_brain, duration=10):
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
        self.spawn_site("images/circle2.png", x=375, y=500/2)
        super().run()

# Neural Network Definitions - Modified to output state decisions
class PreyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 32)
        self.fc2 = nn.Linear(32, 16)
        # Output: state (2 values for wandering/fleeing), move_x, move_y, speed
        self.fc3 = nn.Linear(16, 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Split output: state decision (2), movement (2), speed (1)
        state_logits = x[:2]  # For state decision
        movement = torch.tanh(x[2:4])  # Movement direction
        speed = torch.sigmoid(x[4])  # Speed control
        return state_logits, movement[0], movement[1], speed

class PredatorBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 32)
        self.fc2 = nn.Linear(32, 16)
        # Output: state (2 values for hunting/lunging), move_x, move_y, speed
        self.fc3 = nn.Linear(16, 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Split output: state decision (2), movement (2), speed (1)
        state_logits = x[:2]  # For state decision
        movement = torch.tanh(x[2:4])  # Movement direction
        speed = torch.sigmoid(x[4])  # Speed control
        return state_logits, movement[0], movement[1], speed

@dataclass 
class NeuroEvoConfig:
    population_size: int = 20
    generations: int = 3
    mutation_rate: float = 0.125
    mutation_scale: float = 0.25
    crossover_rate: float = 0.75
    elite_size: int = 5
    sim_duration: int = 1500
    n_simulations: int = 1
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
    
    def _get_elite_indices(self, fitness_scores, population):
        """
        Get indices of elite individuals based on fitness scores.
        
        Args:
            fitness_scores: List of fitness scores for the population
            population: The population to get elite indices from
        
        Returns:
            List of indices of elite individuals
        """
        if not fitness_scores or len(fitness_scores) == 0:
            return list(range(len(population)))
        
        # Get the most recent fitness scores
        current_fitness = fitness_scores[-1] if isinstance(fitness_scores[0], list) else fitness_scores
        
        # Create list of (fitness, index) pairs
        fitness_index_pairs = [(fitness, i) for i, fitness in enumerate(current_fitness)]
        
        # Sort by fitness (assuming higher is better, reverse if lower is better)
        fitness_index_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted indices
        return [index for _, index in fitness_index_pairs]

    def evolve(self):
        for gen in tqdm(range(self.config.generations), desc="Evolving"):
            # Evaluate prey population sequentially
            prey_fitnesses, predator_fitnesses = parallel_evaluation(
                self.prey_population, self.predator_test_population,
                self.predator_population, self.prey_test_population,
                self.config
            )

            for idx, (prey_fit, pred_fit) in enumerate(zip(prey_fitnesses, predator_fitnesses)):
                print(f"Prey {(idx+1)}/{len(prey_fitnesses)} fitness: {prey_fit:.4f}")
                print(f"Predator {(idx+1)}/{len(predator_fitnesses)} fitness: {pred_fit:.4f}")
            
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
            if gen % 1 == 0:
                self._update_test_populations()

            # Adaptive mutation decay
            decay = 0.95  # Decay factor per generation (tweak as needed)
            self.config.mutation_rate *= decay
            self.config.mutation_scale *= decay

            # Optional: enforce lower bound
            self.config.mutation_rate = max(self.config.mutation_rate, 0.01)
            self.config.mutation_scale = max(self.config.mutation_scale, 0.01)
    
        global all_training_prey_data, all_training_predator_data
        all_runs = all_training_prey_data + all_training_predator_data
        results_df = pd.concat(all_runs, ignore_index=True)

        plt.figure(figsize=(12, 6))
        for run_id in results_df["run"].unique():
            run_df = results_df[results_df["run"] == run_id]
            plt.plot(run_df["frame"], run_df["prey_count"], color="green", alpha=0.3)
            plt.plot(run_df["frame"], run_df["predator_count"], color="red", alpha=0.3)

        plt.title("Neuroevolution Lotka-Volterra: First Generation")
        plt.xlabel("Frame")
        plt.ylabel("Population")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("NE4_one_generation.png")
        plt.legend()
        plt.close()
        print("Plot saved to NE4_one_generation.png")

        all_training_prey_data.clear()
        all_training_predator_data.clear()

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

        # Fix: elite contains PreyBrain/PredatorBrain objects, not tuples
        new_population = [copy.deepcopy(brain) for brain in elite]

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
        """Update test populations with elite individuals from main populations."""
        if not self.prey_fitness_history or not self.predator_fitness_history:
            return
        
        # Get elite indices, but ensure they don't exceed population size
        prey_elite_indices = self._get_elite_indices(self.prey_fitness_history[-1:], self.prey_population)
        predator_elite_indices = self._get_elite_indices(self.predator_fitness_history[-1:], self.predator_population)
        
        # Update prey test population
        for i in range(len(self.prey_test_population)):
            # Use modulo to ensure we don't go out of bounds
            elite_index = prey_elite_indices[i % len(prey_elite_indices)]
            if elite_index < len(self.prey_population):
                self.prey_test_population[i] = copy.deepcopy(self.prey_population[elite_index])
        
        # Update predator test population
        for i in range(len(self.predator_test_population)):
            # Use modulo to ensure we don't go out of bounds
            elite_index = predator_elite_indices[i % len(predator_elite_indices)]
            if elite_index < len(self.predator_population):
                self.predator_test_population[i] = copy.deepcopy(self.predator_population[elite_index])

# === Neural Prey with FSM + NN hybrid logic - NOW WITH STATE CONTROL ===
class NeuralPrey(Agent):
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
        self.brain = None
        self.steps_since_decision = 0
        self.decision_interval = 5
        self.confidence = 0.0  # Smoothness factor from FSM to NN
        self.last_nn_output = None
        self.sex = random.choice(["m", "f"])
        self.just_reproduced = False

    def get_nn_state(self):
        nearest_pred, nearest_prey, nearest_food = None, None, None
        pred_distance, prey_distance, food_distance = self.config.prey_vision, self.config.prey_vision, self.config.prey_vision
        pred_count, prey_count = 0, 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPredator) and agent.alive and distance < pred_distance:
                pred_distance, nearest_pred = distance, agent
                pred_count += 1
            elif isinstance(agent, Food) and agent.alive and distance < food_distance:
                food_distance, nearest_food = distance, agent
            elif isinstance(agent, NeuralPrey) and agent.alive and distance < self.config.prey_vision:
                prey_distance, nearest_prey = distance, agent
                prey_count += 1

        if nearest_pred and (nearest_pred.pos - self.pos).length_squared() > 0:
            pred_dir = (nearest_pred.pos - self.pos).normalize()
        else:
            pred_dir = Vector2(0, 0)
        if nearest_prey and (nearest_prey.pos - self.pos).length_squared() > 0:
            prey_dir = (nearest_prey.pos - self.pos).normalize()
        else:
            prey_dir = Vector2(0, 0)
        if nearest_food and (nearest_food.pos - self.pos).length_squared() > 0:
            food_dir = (nearest_food.pos - self.pos).normalize()
        else:
            food_dir = Vector2(0, 0)

        return torch.tensor([
            self.pos.x, self.pos.y,
            min(self.energy / self.config.prey_energy, 1.0),
            1.0 if self.energy < 0.2 * self.config.prey_energy else 0.0,  # 1: low energy flag
            pred_distance / self.config.prey_vision,
            pred_dir.x, pred_dir.y,
            min(pred_count / 10.0, 1.0),             # 5: number of predators nearby (capped at 10)
            food_distance / self.config.prey_vision,
            food_dir.x, food_dir.y,
            1.0 if (self.state == PREY_FLEEING or self.state == PREY_CONSCIOUS_F) else 0.0,  # 9: fleeing state flag
            min(prey_count / 20.0, 1.0),  # 10: number of other prey nearby (capped at 20)
            prey_dir.x, prey_dir.y,
            prey_distance / self.config.prey_vision,
            0.0 if self.sex == "m" else 1.0,
            0.0 if nearest_prey is None else (0.0 if nearest_prey.sex == "m" else 1.0),
            1.0 if self.just_reproduced else 0.0,  # 12: just reproduced flag
            1.0 if self.on_site() else 0.0,  # 11: on-site flag
            len(preys)
        ], dtype=torch.float32)
    
    def update(self):
        global preys

        self.age += 1
        if self.age > 400:
            self.kill()
            #print("Prey died of old age")
            preys.remove(self.id)
            return
        
        if self.energy <= 0:
            self.kill()
            #print("Prey died of energy loss")
            preys.remove(self.id)
            return

        if self.state == PREY_EATEN:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.kill()
                #print("Prey was eaten")
                preys.remove(self.id)
            return
        
        # Energy consumption
        if self.state == PREY_FLEEING or self.state == PREY_CONSCIOUS_F:
            self.energy -= self.config.prey_flee_energy_consumption
        else:
            if not self.on_site():
                self.energy -= self.config.prey_energy_consumption
            else:
                self.age += 1
        
        # Neural network confidence grows with age
        if self.brain is not None:
            self.confidence = min(max(self.age / 400.0, 0.0), 1.0)
            
            # Update NN decision periodically
            self.steps_since_decision += 1
            if self.steps_since_decision >= self.decision_interval:
                self.steps_since_decision = 0
                with torch.no_grad():
                    self.last_nn_output = self.brain(self.get_nn_state())

        # STATE CONTROL: Confident prey can choose their state intelligently
        if self.brain is not None and hasattr(self, "last_nn_output") and self.last_nn_output is not None:
            state_logits, _, _, _ = self.last_nn_output
            
            if self.confidence > CONSCIOUSNESS_THRESHOLD:
                # Confident prey - use NN to choose state intelligently
                nn_state_probs = F.softmax(state_logits, dim=0)
                if nn_state_probs[0] > nn_state_probs[1]:  # NN prefers wandering
                    self.state = PREY_WANDERING
                    self.change_image(PREY_CONSCIOUS_W)
                else:  # NN prefers fleeing
                    self.state = PREY_CONSCIOUS_F
                    self.change_image(PREY_CONSCIOUS_F)
            else:
                # Not confident enough - use FSM approach
                self.state = self._get_fsm_state()
        else:
            # No NN - pure FSM
            self.state = self._get_fsm_state()
        
        # Find nearest predator and food for other behaviors
        nearest_food = None
        min_food_distance = float('inf')
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, Food):
                if distance < min_food_distance:
                    min_food_distance = distance
                    nearest_food = agent
        
        # Eat food if nearby
        if nearest_food and min_food_distance < 10:  # Eating distance
            nearest_food.kill()
            self.energy += self.config.prey_energy_gain
            self.age -= 100

        # Reproduction - only if another prey is nearby and has enough energy
        if (self.energy >= self.config.prey_reproduction_energy_threshold and 
            (random.random() < self.config.prey_reproduction_probability) or (random.random() < 0.00000025 and self.simulation.shared.counter > 2000)):
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, NeuralPrey):
                    if (distance < self.config.prey_reproduction_radius and 
                        agent.energy >= self.config.prey_reproduction_energy_threshold and self.age > 100 and agent.age > 100 and
                        (self.sex == "m" and agent.sex == "f" or self.sex == "f" and agent.sex == "m")):
                        self.energy -= self.config.prey_reproduction_cost
                        self.just_reproduced = True
                        agent.just_reproduced = True
                        agent.energy -= self.config.prey_reproduction_cost
                        child = self.reproduce()
                        if isinstance(child, NeuralPrey) and self.brain is not None:
                            child.brain = copy.deepcopy(self.brain)
                        preys.add(child.id)
                        break
    
    def _get_fsm_state(self):
        """Get state from FSM logic"""
        # Find nearest predator
        nearest_predator = None
        min_predator_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPredator):
                if distance < min_predator_distance:
                    min_predator_distance = distance
                    nearest_predator = agent
        
        # FSM state decision
        if nearest_predator and min_predator_distance < self.config.prey_vision:
            return PREY_FLEEING
        else:
            return PREY_WANDERING
    
    def change_position(self):
        if self.state == PREY_EATEN:
            self.move = Vector2(0, 0)
            self.alive = False
            return
            
        self.there_is_no_escape()
        
        if self.brain is not None and hasattr(self, "last_nn_output") and self.last_nn_output is not None:
            _, move_x, move_y, speed_ctrl = self.last_nn_output
            move_vec = Vector2(move_x.item(), move_y.item())
            
            if move_vec.length() > 0:
                move_vec = move_vec.normalize()
                # Confident agents have more precise control
                if self.confidence > CONSCIOUSNESS_THRESHOLD:
                    speed = self.config.prey_flee_speed if (self.state == PREY_FLEEING or self.state == PREY_CONSCIOUS_F) else self.config.prey_speed
                    # Full NN control for confident agents
                    self.move = move_vec * speed * ((speed_ctrl.item() + 1) / 2)
                else:
                    self.move = self._get_fsm_movement()
            else:
                self.move = self._get_fsm_movement()
        else:
            self.move = self._get_fsm_movement()

        if self.on_site():
            if random.random() < 0.75:
                self.move = self.move * 0.95
            else:
                self.move = self.move * 1.35

        self.pos += self.move * self.config.delta_time
    
    def _get_fsm_movement(self):
        """Get movement from FSM logic (from NE2)"""
        if self.state == PREY_WANDERING:
            # Random wandering with occasional food seeking
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
                return food_direction * self.config.prey_speed
            else:
                # Random wandering
                angle = random.uniform(0, 2 * math.pi)
                return Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_speed
        else:  # FLEEING
            # Flee from nearest predator
            nearest_predator = None
            min_distance = float('inf')
            
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, NeuralPredator):
                    if distance < min_distance:
                        min_distance = distance
                        nearest_predator = agent
            
            if nearest_predator:
                delta = self.pos - nearest_predator.pos
                if delta.length() == 0:
                    flee_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                else:
                    flee_direction = delta.normalize()
                return flee_direction * self.config.prey_flee_speed
            else:
                # No predator nearby, fallback movement
                angle = random.uniform(0, 2 * math.pi)
                return Vector2(math.cos(angle), math.sin(angle)) * self.config.prey_flee_speed
    
    def start_being_eaten(self):
        """Transition to being eaten state"""
        self.state = PREY_EATEN
        self.change_image(PREY_EATEN)
        self.eating_timer = self.config.eating_duration


# === Neural Predator with FSM + NN hybrid logic - NOW WITH STATE CONTROL ===
class NeuralPredator(Agent):
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
        self.brain = None
        self.steps_since_decision = 0
        self.decision_interval = 5
        self.confidence = 0.0
        self.last_nn_output = None
        self.sex = random.choice(["m", "f"])
        self.just_reproduced = False

    def get_nn_state(self):
        nearest_prey, nearest_pred = None, None
        prey_distance, pred_distance = self.config.predator_vision, self.config.predator_vision
        pred_count, prey_count = 0, 0

        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPrey) and agent.alive and agent.state != PREY_EATEN:
                prey_count += 1
                if distance < prey_distance:
                    prey_distance, nearest_prey = distance, agent
            elif isinstance(agent, NeuralPredator) and agent.alive and agent.state != PREDATOR_EATING:
                if distance < self.config.predator_vision:
                    pred_distance, nearest_pred = distance, agent
                    pred_count += 1

        if nearest_prey and (nearest_prey.pos - self.pos).length_squared() > 0:
            prey_dir = (nearest_prey.pos - self.pos).normalize()
        else:
            prey_dir = Vector2(0, 0)
        if nearest_pred and (nearest_pred.pos - self.pos).length_squared() > 0:
            pred_dir = (nearest_pred.pos - self.pos).normalize()
        else:
            pred_dir = Vector2(0, 0)

        return torch.tensor([
            self.pos.x, self.pos.y,
            min(self.energy / self.config.predator_energy, 1.0),
            prey_distance / self.config.predator_vision,
            prey_dir.x, prey_dir.y,
            min(prey_count / 20.0, 1.0),
            1.0 if nearest_prey is not None and prey_distance < 30 else 0.0,
            1.0 if self.state == PREDATOR_LUNGING else 0.0,  # 5: lunging state flag
            min(pred_count / 20.0, 1.0),  # 6: number of predators nearby (capped at 20)
            pred_dir.x, pred_dir.y,
            pred_distance / self.config.prey_vision,
            0.0 if self.sex == "m" else 1.0,
            0.0 if nearest_pred is None else (0.0 if nearest_pred.sex == "m" else 1.0),
            (self.eating_timer / self.config.eating_duration) if self.state == PREDATOR_EATING else 0.0,  # 7: eating timer normalized
            1.0 if self.just_reproduced else 0.0,
            len(predators)
        ], dtype=torch.float32)
    
    def update(self):
        global predators
        self.age += 1
        if self.age > 350:
            self.kill()
            #print("Predator died of old age")
            predators.remove(self.id)
            return

        if self.state == PREDATOR_LUNGING or self.state == PREDATOR_CONSCIOUS_L:
            self.energy -= self.config.predator_lunge_energy_consumption
        else:
            self.energy -= self.config.predator_energy_consumption
        
        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            #print("Predator died of energy loss")
            predators.remove(self.id)
            return
        
        # Neural network confidence grows with age
        if self.brain is not None:
            self.confidence = min(max(self.age / 350.0, 0.0), 1.0)
            
            # Update NN decision more frequently for conscious predators
            decision_interval = 1 if self.confidence > CONSCIOUSNESS_THRESHOLD else 5
            self.steps_since_decision += 1
            if self.steps_since_decision >= decision_interval:
                self.steps_since_decision = 0
                with torch.no_grad():
                    self.last_nn_output = self.brain(self.get_nn_state())
        
        # Handle eating state
        if self.state == PREDATOR_EATING:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                # Reset to hunting after eating
                if self.brain is not None and hasattr(self, "last_nn_output") and self.last_nn_output is not None and self.confidence > 0.5:
                    state_logits, _, _, _ = self.last_nn_output
                    nn_state_probs = F.softmax(state_logits, dim=0)
                    self.state = PREDATOR_HUNTING if nn_state_probs[0] > nn_state_probs[1] else PREDATOR_LUNGING
                else:
                    self.state = PREDATOR_HUNTING
                self.change_image(self.state)
                self.target_prey = None
            return
        
        # Find nearest prey
        nearest_prey = None
        min_distance = float('inf')
        
        for agent, distance in self.in_proximity_accuracy():
            if isinstance(agent, NeuralPrey) and agent.state != PREY_EATEN:
                if distance < min_distance:
                    min_distance = distance
                    nearest_prey = agent

        # Check for reproduction with nearby predators
        if self.energy >= self.config.predator_reproduction_energy_threshold:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, NeuralPredator):
                    if (distance < self.config.predator_reproduction_radius and 
                        agent.energy >= self.config.predator_reproduction_energy_threshold and
                        self.age > 100 and agent.age > 100 and (self.sex == "m" and agent.sex == "f" or self.sex == "f" and agent.sex == "m")):
                        # Both predators can reproduce
                        self.energy -= self.config.predator_reproduction_cost
                        agent.energy -= self.config.predator_reproduction_cost
                        self.just_reproduced = True
                        agent.just_reproduced = True
                        child = self.reproduce()
                        if isinstance(child, NeuralPredator) and self.brain is not None:
                            child.brain = copy.deepcopy(self.brain)
                        predators.add(child.id)
                        if random.random() < 0.5:
                            child2 = agent.reproduce()
                            if isinstance(child2, NeuralPredator) and agent.brain is not None:
                                child2.brain = copy.deepcopy(agent.brain)
                            predators.add(child2.id)
                            if random.random() < 0.5:
                                child3 = self.reproduce()
                                if isinstance(child3, NeuralPredator) and self.brain is not None:
                                    child3.brain = copy.deepcopy(self.brain)
                                predators.add(child3.id)
                        break
        
        # STATE CONTROL: More aggressive state selection for conscious predators
        if self.brain is not None and hasattr(self, "last_nn_output") and self.last_nn_output is not None:
            state_logits, _, _, _ = self.last_nn_output
            
            if self.confidence > CONSCIOUSNESS_THRESHOLD:
                # Conscious predators make smarter state decisions
                nn_state_probs = F.softmax(state_logits, dim=0)
                
                # More aggressive decision making - prioritize lunging when prey is near
                if nearest_prey and min_distance < self.config.predator_vision * 0.6:  # Lunging range
                    # Use NN output but bias toward lunging when prey is close
                    lunge_prob = nn_state_probs[1] * 1.5  # Boost lunge probability
                    if lunge_prob > 0.7 or min_distance < self.config.predator_vision * 0.3:
                        self.state = PREDATOR_CONSCIOUS_L
                        self.change_image(PREDATOR_CONSCIOUS_L)
                        self.target_prey = nearest_prey
                    else:
                        self.state = PREDATOR_CONSCIOUS_H
                        self.change_image(PREDATOR_CONSCIOUS_H)
                else:
                    self.state = PREDATOR_CONSCIOUS_H
                    self.change_image(PREDATOR_CONSCIOUS_H)
            else:
                # Non-conscious predators use simpler FSM logic
                self.state = self._get_fsm_state(nearest_prey, min_distance)
        else:
            self.state = self._get_fsm_state(nearest_prey, min_distance)
        
        # Handle target assignment
        if (self.state == PREDATOR_LUNGING or self.state == PREDATOR_CONSCIOUS_L) and nearest_prey:
            self.target_prey = nearest_prey
        else:
            self.target_prey = None
        
        # Check for successful catch (more lenient threshold for conscious predators)
        if (self.state == PREDATOR_LUNGING or self.state == PREDATOR_CONSCIOUS_L) and nearest_prey:
            if min_distance < self.config.predator_eating_threshold and nearest_prey.state != PREY_EATEN:
                nearest_prey.start_being_eaten()
                self.energy += self.config.predator_eating_energy
                self.state = PREDATOR_EATING
                self.change_image(PREDATOR_EATING)
                self.eating_timer = self.config.eating_duration
                self.target_prey = None

    
    def _get_fsm_state(self, nearest_prey, min_distance):
        """Get state from FSM logic"""
        if self.state == PREDATOR_HUNTING:
            if nearest_prey and min_distance < self.config.predator_vision:
                return PREDATOR_LUNGING
        elif self.state == PREDATOR_LUNGING:
            if not nearest_prey or min_distance > self.config.predator_vision:
                return PREDATOR_HUNTING
        
        return self.state  # Keep current state if no transition triggered
    
    def change_position(self):
        if self.state == PREDATOR_EATING:
            self.move = Vector2(0, 0)
            return
            
        self.there_is_no_escape()
        
        if self.brain is not None and hasattr(self, "last_nn_output") and self.last_nn_output is not None:
            _, move_x, move_y, speed_ctrl = self.last_nn_output
            move_vec = Vector2(move_x.item(), move_y.item())
            
            if move_vec.length() == 0:
                move_vec = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            
            if move_vec.length() > 0:
                move_vec = move_vec.normalize()
                if self.confidence > CONSCIOUSNESS_THRESHOLD:
                    # Conscious predators have more focused movement
                    if self.state == PREDATOR_CONSCIOUS_L and self.target_prey:
                        # Directly lunge at prey with more precision
                        target_dir = (self.target_prey.pos - self.pos).normalize()
                        move_vec = (move_vec + target_dir * 2.0).normalize()  # Blend NN direction with prey direction
                        speed = self.config.predator_lunge_speed * 1.1  # Slightly faster when conscious
                    else:
                        # More purposeful hunting movement
                        speed = self.config.predator_speed * 1.1
                    
                    self.move = move_vec * speed * ((speed_ctrl.item() + 1) / 2)
                else:
                    # Non-conscious movement
                    speed = self.config.predator_lunge_speed if self.state == PREDATOR_LUNGING else self.config.predator_speed
                    self.move = move_vec * speed * ((speed_ctrl.item() + 1) / 2)
            else:
                self.move = self._get_fsm_movement()
        else:
            self.move = self._get_fsm_movement()

        # Site avoidance remains the same
        future_pos = self.pos + self.move * self.config.delta_time
        distance_to_site = future_pos.distance_to(Vector2(500*3/4, 500/2))

        if distance_to_site < 130 or self.on_site():
            avoidance_dir = (future_pos - Vector2(500*3/4, 500/2)).normalize()
            self.move = avoidance_dir * self.config.predator_speed
            self.avoided = True

        self.pos += self.move * self.config.delta_time
    
    def _get_fsm_movement(self):
        """Get movement from FSM logic (from NE2)"""
        if self.state == PREDATOR_LUNGING and self.target_prey:
            # Lunge toward target prey
            delta = self.target_prey.pos - self.pos
            if delta.length() == 0:
                lunge_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            else:
                lunge_direction = (self.target_prey.pos - self.pos).normalize()
            return lunge_direction * self.config.predator_lunge_speed
        else:
            random_chance = random.random()
            if random_chance < 0.33:
                # Random hunting movement
                angle = random.uniform(0, math.pi)
                return Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
            elif random_chance < 0.66:
                angle = random.uniform(-math.pi, math.pi)
                return Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
            else:
                angle = random.uniform(0, 2 * math.pi)
                return Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed


def run_final_visual_simulation(best_prey: PreyBrain, best_predator: PredatorBrain, duration: int = 5000):
    """
    Runs a visual simulation using the best evolved prey and predator brains.
    Returns data for plotting.
    """
    cfg = PredatorPreyConfig(duration=duration)
    
    # Lists to store simulation data
    final_sim_data = {
        'frame': [],
        'prey_count': [],
        'predator_count': [],
        'food_count': [],
        'prey_states': [],
        'predator_states': []
    }

    # Create simulation
    sim = Simulation(cfg)

    def spawn_neural_prey(**kwargs):
        prey = NeuralPrey(**kwargs)
        prey.brain = best_prey
        return prey
        
    def spawn_neural_predator(**kwargs):
        predator = NeuralPredator(**kwargs)
        predator.brain = best_predator
        return predator

    # Spawn agents
    sim.batch_spawn_agents(cfg.initial_prey, spawn_neural_prey, images=PREY_IMAGES)
    sim.batch_spawn_agents(cfg.initial_predators, spawn_neural_predator, images=PREDATOR_IMAGES)
    sim.batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
    sim.spawn_agent(FoodSpawner, images=["images/transparent.png"])
    
    # Create a custom monitor to collect data
    class FinalSimMonitor(Agent):
        def update(self):
            frame = self.simulation.shared.counter
            prey_agents = [a for a in self.simulation.agents if isinstance(a, NeuralPrey)]
            predator_agents = [a for a in self.simulation.agents if isinstance(a, NeuralPredator)]
            food_agents = [a for a in self.simulation.agents if isinstance(a, Food)]
            
            # Count states
            prey_states = [p.state for p in prey_agents]
            predator_states = [p.state for p in predator_agents]
            
            final_sim_data['frame'].append(frame)
            final_sim_data['prey_count'].append(len(prey_agents))
            final_sim_data['predator_count'].append(len(predator_agents))
            final_sim_data['food_count'].append(len(food_agents))
            final_sim_data['prey_states'].append(prey_states.copy())
            final_sim_data['predator_states'].append(predator_states.copy())

            if len(prey_agents) <= 1:
                # Stop simulation if prey are extinct
                self.simulation.stop()
                print("Simulation ended: Prey extinct.")
            if len(predator_agents) <= 1:
                # Stop simulation if predators are extinct
                self.simulation.stop()
                print("Simulation ended: Predators extinct.")
    
    sim.spawn_agent(FinalSimMonitor, images=["images/transparent.png"])
    sim.spawn_site("images/circle2.png", x=(500/5)+200, y=500/2)

    print("[Running Final Visual Simulation...]")
    sim.run()
    
    return final_sim_data

def plot_final_simulation(final_data):
    """Plot the final simulation data."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Population counts over time
    axes[0, 0].plot(final_data['frame'], final_data['prey_count'], label='Prey', color='green')
    axes[0, 0].plot(final_data['frame'], final_data['predator_count'], label='Predators', color='red')
    axes[0, 0].plot(final_data['frame'], final_data['food_count'], label='Food', color='blue')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Population Counts Over Time')
    axes[0, 0].legend()
    
    # Prey state distribution over time
    prey_wandering = [sum(1 for state in states if state == PREY_WANDERING) for states in final_data['prey_states']]
    prey_fleeing = [sum(1 for state in states if state == PREY_FLEEING) for states in final_data['prey_states']]
    prey_eaten = [sum(1 for state in states if state == PREY_EATEN) for states in final_data['prey_states']]
    prey_conscious_w = [sum(1 for state in states if state == PREY_CONSCIOUS_W) for states in final_data['prey_states']]
    prey_conscious_f = [sum(1 for state in states if state == PREY_CONSCIOUS_F) for states in final_data['prey_states']]
    
    axes[0, 1].plot(final_data['frame'], prey_wandering, label='Wandering', color='lightgreen')
    axes[0, 1].plot(final_data['frame'], prey_fleeing, label='Fleeing', color='yellow')
    axes[0, 1].plot(final_data['frame'], prey_eaten, label='Eaten', color='red')
    axes[0, 1].plot(final_data['frame'], prey_conscious_w, label='Conscious Wandering', color='lightblue')
    axes[0, 1].plot(final_data['frame'], prey_conscious_f, label='Conscious Fleeing', color='orange')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Prey States Over Time')
    axes[0, 1].legend()
    
    # Predator state distribution over time
    pred_hunting = [sum(1 for state in states if state == PREDATOR_HUNTING) for states in final_data['predator_states']]
    pred_lunging = [sum(1 for state in states if state == PREDATOR_LUNGING) for states in final_data['predator_states']]
    pred_eating = [sum(1 for state in states if state == PREDATOR_EATING) for states in final_data['predator_states']]
    pred_conscious_h = [sum(1 for state in states if state == PREDATOR_CONSCIOUS_H) for states in final_data['predator_states']]
    pred_conscious_l = [sum(1 for state in states if state == PREDATOR_CONSCIOUS_L) for states in final_data['predator_states']]
    
    axes[1, 0].plot(final_data['frame'], pred_hunting, label='Hunting', color='darkgreen')
    axes[1, 0].plot(final_data['frame'], pred_lunging, label='Lunging', color='orange')
    axes[1, 0].plot(final_data['frame'], pred_eating, label='Eating', color='darkred')
    axes[1, 0].plot(final_data['frame'], pred_conscious_h, label='Conscious Hunting', color='purple')
    axes[1, 0].plot(final_data['frame'], pred_conscious_l, label='Conscious Lunging', color='pink')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Predator States Over Time')
    axes[1, 0].legend()
    
    # Survival ratio
    total_agents = [p + pr for p, pr in zip(final_data['prey_count'], final_data['predator_count'])]
    survival_ratio = [p/t if t > 0 else 0 for p, t in zip(final_data['prey_count'], total_agents)]
    
    axes[1, 1].plot(final_data['frame'], survival_ratio, color='purple')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Prey Survival Ratio')
    axes[1, 1].set_title('Prey Survival Ratio Over Time')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Evolve and get the best brains
    config = NeuroEvoConfig()
    neuro_evo = NeuroEvolution(config)
    neuro_evo.evolve()

    # Visualize final result
    final_data = run_final_visual_simulation(
        best_prey=neuro_evo.best_prey, 
        best_predator=neuro_evo.best_predator,
        duration=5000
    )
    
    # Plot the final simulation
    plot_final_simulation(final_data)
