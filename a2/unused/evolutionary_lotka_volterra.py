import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import pandas as pd
import math
import numpy as np
from collections import deque
import traceback
import threading

# Image paths
PREY_IMAGES = ["images/green.png"]
PREDATOR_IMAGES = ["images/triangle5.png"]
FOOD_IMAGES = ["images/plus.png"]

# Evolutionary parameters
GENERATIONS = 30  # Increased from 20
POPULATION_SIZE = 24  # Increased from 16
MUTATION_RATE = 0.08  # Reduced from 0.1
CROSSOVER_RATE = 0.7
ELITE_COUNT = 3  # Increased from 2
TOURNAMENT_K = 4  # Increased from 3
PATIENCE = 8  # Increased from 5
MIN_DELTA = 1e-4

frame_data = {
    "frame": [],
    "prey_count": [],
    "predator_count": [],
    "food_count": []
}

preys = set()
predators = set()
foods = set()

def reset_global_state():
    global frame_data, preys, predators, foods
    frame_data = {
        "frame": [],
        "prey_count": [],
        "predator_count": [],
        "food_count": []
    }
    preys = set()
    predators = set()
    foods = set()

def get_actual_counts(simulation):
    prey_count = sum(1 for agent in simulation._agents if isinstance(agent, Prey) and agent.alive)
    predator_count = sum(1 for agent in simulation._agents if isinstance(agent, Predator) and agent.alive)
    food_count = sum(1 for agent in simulation._agents if isinstance(agent, Food) and agent.alive)
    return prey_count, predator_count, food_count

@dataclass
class LotkaVolterraConfig(Config):
    # Energy parameters - adjusted for better oscillations
    prey_energy_gain: float = 30.0  # Increased from 25.0
    prey_energy_loss: float = 0.15  # Reduced from 0.2
    prey_initial_energy: float = 50.0  # Increased from 40.0
    predator_energy_gain: float = 50.0  # Increased from 40.0
    predator_energy_loss: float = 0.25  # Reduced from 0.3
    predator_initial_energy: float = 50.0  # Increased from 40.0
    
    # Logistic growth parameters
    prey_growth_rate: float = 0.08  # Increased from 0.05
    prey_carrying_capacity: float = 250.0  # Increased from 200.0
    prey_reproduce_cost: float = 12.0  # Reduced from 15.0

    # Predator reproduction
    predator_reproduce_prob: float = 0.03  # Increased from 0.02
    predator_reproduce_threshold: float = 35.0  # Increased from 30.0
    predator_reproduce_cost: float = 20.0  # Reduced from 25.0
    
    # Food parameters
    food_spawn_prob: float = 0.025  # Increased from 0.02
    food_spawn_amount: int = 3      # Increased from 2
    initial_food: int = 120         # Increased from 100
    
    # Movement parameters
    prey_speed: float = 2.0  # Increased from 1.8
    predator_speed: float = 2.2  # Increased from 2.0
    
    # Interaction parameters
    predation_distance: float = 12.0  # Increased from 10.0
    eating_distance: float = 6.0      # Increased from 5.0
    
    # Simulation parameters
    initial_prey: int = 100   # Increased from 80
    initial_predators: int = 20  # Increased from 15
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 10000

class Food(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        with threading.Lock():
            foods.add(self.id)
        self.change_image(0)
        self.move = Vector2(0, 0)
        self.alive = True
    
    def update(self):
        pass
    
    def on_death(self):
        with threading.Lock():
            if self.id in foods:
                foods.remove(self.id)

class FoodSpawner(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        self.change_image(0)
    
    def update(self):
        if random.random() < self.config.food_spawn_prob:
            for _ in range(self.config.food_spawn_amount):
                self.simulation.spawn_agent(Food, images=FOOD_IMAGES)

class Prey(Agent):
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        with threading.Lock():
            preys.add(self.id)
        self.change_image(0)
        self.energy = self.config.prey_initial_energy
        self.alive = True
    
    def update(self):
        global preys, foods

        # Energy loss per frame
        self.energy -= self.config.prey_energy_loss
        
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
                self.energy > self.config.prey_reproduce_cost):
                
                self.energy -= self.config.prey_reproduce_cost
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
    config: LotkaVolterraConfig
    
    def on_spawn(self):
        with threading.Lock():
            predators.add(self.id)
        self.change_image(0)
        self.energy = self.config.predator_initial_energy
        self.alive = True
    
    def update(self):
        global preys, predators

        # Energy loss per frame
        self.energy -= self.config.predator_energy_loss
        
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
        
        if nearest_prey and min_prey_distance < self.config.predation_distance:
            if nearest_prey.alive:
                nearest_prey.alive = False
                nearest_prey.kill()
                self.energy += self.config.predator_energy_gain

                # Reproduce upon eating
                if self.energy > self.config.predator_reproduce_threshold and random.random() < self.config.predator_reproduce_prob:
                    self.energy -= self.config.predator_reproduce_cost
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dominance_window = 300
        self.dominance_ratio = 5.0
        self.dominance_threshold = 0.6
        self.dominance_limit = 2000

        self.prey_buffer = deque(maxlen=self.dominance_window)
        self.pred_buffer = deque(maxlen=self.dominance_window)
        self.dominance_counter = 0

    def update(self):
        current_frame = self.simulation.shared.counter
        
        # Get actual counts by checking alive agents
        prey_count = sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.alive)
        predator_count = sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.alive)
        food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food) and a.alive)
        
        # Update frame data
        frame_data["frame"].append(current_frame)
        frame_data["prey_count"].append(prey_count)
        frame_data["predator_count"].append(predator_count)
        frame_data["food_count"].append(food_count)
        
        # Termination conditions
        if current_frame > 5 and (prey_count == 0 or predator_count == 0):
            self.simulation.stop()
            print(f"Extinction at frame {current_frame}")
            
        if current_frame >= self.config.duration:
            self.simulation.stop()

        if current_frame % 50 == 0:
            print(f"Frame {current_frame}: Prey={prey_count}, Predators={predator_count}, Food={food_count}")

        # track counts
        self.prey_buffer.append(prey_count)
        self.pred_buffer.append(predator_count)

        if len(self.prey_buffer) == self.dominance_window:
            dominance_frames = sum(
                1 for prey, pred in zip(self.prey_buffer, self.pred_buffer)
                if prey > self.dominance_ratio * pred or pred > self.dominance_ratio * prey
            )
            if dominance_frames / self.dominance_window >= self.dominance_threshold:
                self.dominance_counter += 1
            else:
                self.dominance_counter = 0

        # 🔥 Early termination condition
        if self.dominance_counter >= self.dominance_limit:
            print("⚠️ SimulationMonitor: Dominance persisted too long. Aborting run.")
            self.simulation.stop()

class EvolutionaryLotkaVolterra:
    def __init__(self):
        self.param_bounds = {
            # Energy parameters
            "prey_energy_gain": (20.0, 40.0),
            "prey_energy_loss": (0.1, 0.25),
            "predator_energy_gain": (30.0, 50.0),
            "predator_energy_loss": (0.15, 0.3),
            
            # Logistic growth parameters
            "prey_growth_rate": (0.05, 0.15),
            "prey_carrying_capacity": (150.0, 350.0),
            "prey_reproduce_cost": (8.0, 15.0),

            # Predator reproduction
            "predator_reproduce_prob": (0.02, 0.05),
            "predator_reproduce_threshold": (25.0, 40.0),
            "predator_reproduce_cost": (15.0, 25.0),
            
            # Food parameters
            "food_spawn_prob": (0.01, 0.1),
            
            # Movement parameters
            "prey_speed": (1.0, 4.0),
            "predator_speed": (2.0, 4.0),
            
            # Interaction parameters
            "predation_distance": (10.0, 15.0),
            "eating_distance": (5.0, 10.0),
        }
        
        self.param_order = list(self.param_bounds.keys())

    def random_genome(self):
        return [random.uniform(*self.param_bounds[k]) for k in self.param_order]
    
    def mutate(self, genome):
        return [
            min(max(g + random.gauss(0, MUTATION_RATE * (self.param_bounds[k][1] - self.param_bounds[k][0])), 
                self.param_bounds[k][0]), self.param_bounds[k][1])
            for g, k in zip(genome, self.param_order)
        ]
    
    def crossover(self, p1, p2):
        return [p1[i] if random.random() < CROSSOVER_RATE else p2[i] 
                for i in range(len(p1))]
    
    def tournament_selection(self, population, scores):
        best = None
        for _ in range(TOURNAMENT_K):
            i = random.randint(0, len(population) - 1)
            if best is None or scores[i] > scores[best]:
                best = i
        return population[best]
    
    def evaluate_fitness(self, genome):
        try:
            reset_global_state()

            params = dict(zip(self.param_order, genome))
            
            # Stability checks
            if (params["prey_energy_gain"] <= params["prey_energy_loss"] * 40 or
                params["predator_energy_gain"] <= params["predator_energy_loss"] * 25):
                return 0.0
                
            frame_data = {"frame": [], "prey_count": [], "predator_count": [], "food_count": []}
            
            @dataclass
            class EvolvedConfig(Config):
                prey_energy_gain: float = params["prey_energy_gain"]
                prey_energy_loss: float = params["prey_energy_loss"]
                prey_initial_energy: float = 50.0
                predator_energy_gain: float = params["predator_energy_gain"]
                predator_energy_loss: float = params["predator_energy_loss"]
                predator_initial_energy: float = 50.0
                
                prey_growth_rate: float = params["prey_growth_rate"]
                prey_carrying_capacity: float = params["prey_carrying_capacity"]
                prey_reproduce_cost: float = params["prey_reproduce_cost"]

                predator_reproduce_prob: float = params["predator_reproduce_prob"]
                predator_reproduce_threshold: float = params["predator_reproduce_threshold"]
                predator_reproduce_cost: float = params["predator_reproduce_cost"]

                food_spawn_prob: float = params["food_spawn_prob"]
                food_spawn_amount: int = 2
                initial_food: int = 100
                
                prey_speed: float = params["prey_speed"]
                predator_speed: float = params["predator_speed"]
                
                predation_distance: float = params["predation_distance"]
                eating_distance: float = params["eating_distance"]
                
                initial_prey: int = 80
                initial_predators: int = 20
                window_size: tuple[int, int] = (1000, 1000)
                delta_time: float = 1.0
                duration: int = 10000
            
            cfg = EvolvedConfig()
            
            sim = (
                HeadlessSimulation(cfg)
                .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
                .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
                .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
                .spawn_agent(FoodSpawner, images=["images/transparent.png"])
                .spawn_agent(SimulationMonitor, images=["images/transparent.png"])
                .run()
            )
            
            df = pd.DataFrame(frame_data)
            
            if len(df) < 10:
                return 0.0
                
            # Enhanced fitness calculation
            duration = len(df)
            prey_alive = 1.0 if df["prey_count"].iloc[-1] > 0 else 0.0
            pred_alive = 1.0 if df["predator_count"].iloc[-1] > 0 else 0.0
            
            # Calculate population stability and oscillations
            def calculate_stability(series):
                if len(series) < 20:
                    return 0.0
                
                # Normalize the series
                norm_series = (series - series.min()) / (series.max() - series.min() + 1e-8)
                
                # Calculate stability metrics
                mean = np.mean(norm_series)
                std = np.std(norm_series)
                autocorr = np.correlate(norm_series - mean, norm_series - mean, mode='full')
                autocorr = autocorr[len(autocorr)//2:] / (len(norm_series) * std**2)
                
                # We want some oscillation but not too much
                oscillation_score = min(0.5, np.mean(autocorr[1:5]))  # Short-term correlation
                
                # Balance between stability and oscillation
                stability_score = 1.0 - min(1.0, std * 2)  # Penalize high variance
                
                return 0.7 * stability_score + 0.3 * oscillation_score
            
            prey_stability = calculate_stability(df["prey_count"])
            pred_stability = calculate_stability(df["predator_count"])
            
            # Calculate interaction score (predator-prey correlation)
            if len(df) > 20:
                corr = np.corrcoef(df["prey_count"], df["predator_count"])[0,1]
                interaction_score = (corr + 1) / 2  # Convert from [-1,1] to [0,1]
            else:
                interaction_score = 0.0

            amplitude_score = np.clip((df["prey_count"].max() - df["prey_count"].min()) / cfg.prey_carrying_capacity, 0, 1)
            
            fitness = (
                0.25 * (duration / cfg.duration) + 
                0.15 * min(prey_alive, pred_alive) + 
                0.2 * (prey_stability + pred_stability) / 2 +
                0.2 * interaction_score +
                0.2 * amplitude_score
            )

            return max(0.0, min(1.0, fitness))
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            traceback.print_exc()
            return 0.0
    
    def run_evolution(self):
        population = [self.random_genome() for _ in range(POPULATION_SIZE)]

        # Known good starting parameters
        population[0] = [
            30.0, 0.15, 50.0,   # prey energy
            40.0, 0.2, 50.0,    # predator energy
            0.08, 250.0, 12.0,  # prey growth
            0.03, 35.0, 20.0,   # predator reproduction
            0.025, 3, 120,      # food
            2.0, 2.2,           # speeds
            12.0, 6.0,          # interaction distances
            100, 20             # initial populations
        ]

        fitness_scores = [self.evaluate_fitness(g) for g in population]
        
        best_ever = max(fitness_scores)
        best_genome = population[fitness_scores.index(best_ever)]
        patience_counter = 0
        
        for gen in range(GENERATIONS):
            print(f"\nGeneration {gen + 1}/{GENERATIONS}")
            
            sorted_indices = sorted(range(len(population)), 
                                  key=lambda i: fitness_scores[i], reverse=True)
            new_population = [population[i] for i in sorted_indices[:ELITE_COUNT]]
            
            while len(new_population) < POPULATION_SIZE:
                p1 = self.tournament_selection(population, fitness_scores)
                p2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            fitness_scores = [self.evaluate_fitness(g) for g in population]
            
            gen_best = max(fitness_scores)
            gen_best_genome = population[fitness_scores.index(gen_best)]
            
            if gen_best > best_ever + MIN_DELTA:
                best_ever = gen_best
                best_genome = gen_best_genome
                patience_counter = 0
                print(f"New best fitness: {best_ever:.4f}")
            else:
                patience_counter += 1
                
            print(f"Generation best: {gen_best:.4f}")
            print(f"Average fitness: {np.mean(fitness_scores):.4f}")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping at generation {gen + 1}")
                break
        
        return best_genome, best_ever

def run_optimized_simulation():
    evo = EvolutionaryLotkaVolterra()
    best_genome, best_fitness = evo.run_evolution()
    
    print("\nBest parameters found:")
    for param, value in zip(evo.param_order, best_genome):
        print(f"{param}: {value:.4f}")
    print(f"Best fitness: {best_fitness:.4f}")
    
    params = dict(zip(evo.param_order, best_genome))
    
    @dataclass
    class OptimizedConfig(Config):
        prey_energy_gain: float = params["prey_energy_gain"]
        prey_energy_loss: float = params["prey_energy_loss"]
        prey_initial_energy: float = 50
        predator_energy_gain: float = params["predator_energy_gain"]
        predator_energy_loss: float = params["predator_energy_loss"]
        predator_initial_energy: float = 50
        
        prey_growth_rate: float = params["prey_growth_rate"]
        prey_carrying_capacity: float = params["prey_carrying_capacity"]
        prey_reproduce_cost: float = params["prey_reproduce_cost"]

        predator_reproduce_prob: float = params["predator_reproduce_prob"]
        predator_reproduce_threshold: float = params["predator_reproduce_threshold"]
        predator_reproduce_cost: float = params["predator_reproduce_cost"]
        
        food_spawn_prob: float = params["food_spawn_prob"]
        food_spawn_amount: int = 2
        initial_food: int = 100
        
        prey_speed: float = params["prey_speed"]
        predator_speed: float = params["predator_speed"]
        
        predation_distance: float = params["predation_distance"]
        eating_distance: float = params["eating_distance"]
        
        initial_prey: int = 80
        initial_predators: int = 20
        window_size: tuple[int, int] = (1000, 1000)
        delta_time: float = 1.0
        duration: int = 10000
    
    reset_global_state()
    
    cfg = OptimizedConfig()
    
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(FoodSpawner, images=["images/transparent.png"])
        .spawn_agent(SimulationMonitor, images=["images/transparent.png"])
        .run()
    )
    
    df = pd.DataFrame(frame_data)
    df.to_csv('optimized_lotka_volterra_data.csv', index=False)
    print("Data saved to optimized_lotka_volterra_data.csv")
    
    plt.figure(figsize=(14, 8))
    plt.plot(df['frame'], df['prey_count'], label='Prey Population', color='green')
    plt.plot(df['frame'], df['predator_count'], label='Predator Population', color='red')
    plt.plot(df['frame'], df['food_count'], label='Food Count', color='blue', linestyle='--')
    plt.title('Optimized Lotka-Volterra Predator-Prey Dynamics')
    plt.xlabel('Time (frames)')
    plt.ylabel('Population Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimized_lotka_volterra.png')
    plt.close()
    print("Plot saved to optimized_lotka_volterra.png")

if __name__ == "__main__":
    run_optimized_simulation()