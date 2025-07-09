# Place at top
import numpy as np
from scipy.signal import find_peaks
import multiprocessing
from copy import deepcopy
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
    prey_speed: float = 5.0
    prey_flee_speed: float = 6.0
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.223
    prey_flee_energy_consumption: float = 0.3
    prey_energy_gain: float = 5.0
    prey_reproduction_energy_threshold: float = 75.0
    prey_reproduction_cost: float = 25.0
    prey_reproduction_radius: float = 30.0
    prey_reproduction_probability: float = 0.1
    prey_max_age = 200
    
    # Predator parameters
    predator_speed: float = 5.5
    predator_lunge_speed: float = 10.0
    predator_vision: float = 140.0
    predator_energy: float = 150.0
    predator_energy_consumption: float = 0.15
    predator_lunge_energy_consumption: float = 0.5
    predator_eating_threshold: float = 5.0
    predator_eating_energy: float = 50.0
    predator_reproduction_energy_threshold: float = 30.0
    predator_reproduction_cost: float = 70.0
    predator_reproduction_radius: float = 20.0
    predator_reproduction_probability: float = 0.5
    predator_max_age = 200

    # Eating parameters
    eating_duration: int = 5  # Frames prey shows being eaten / predator is eating
    
    # Food parameters
    food_spawn_rate: float = 0.001 #0.005  # Probability of food spawning per frame
    
    # Simulation parameters
    initial_prey: int = 50
    initial_predators: int = 15
    initial_food: int = 15

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
            ## i am really proud of myself on this one: i overrode the main violet library's Agent class to include
            ## self.simulation, which didn't exist before, so that we can access the simulation directly within an agent class
            ## there is a self.__simulation attribute in the Agent class, but it is either a) not working or b) not accessible
            ## so i just added it myself -k

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

        if current_frame % 10 == 0:  # Record every 100 frames
            print(f"frame {current_frame}: preys = {len(preys)}, predators = {len(predators)}, food = {food_count}")

        if len(predators) > 5000 and len(preys) > 5000:
            # Kill 15/16 of prey
            prey_agents = [a for a in self.simulation._agents if isinstance(a, Prey)]
            for prey in random.sample(prey_agents, int(len(prey_agents) * 15 / 16)):
                prey.kill()
                preys.discard(prey.id)

            # Kill 15/16 of predators
            predator_agents = [a for a in self.simulation._agents if isinstance(a, Predator)]
            for pred in random.sample(predator_agents, int(len(predator_agents) * 15 / 16)):
                pred.kill()
                predators.discard(pred.id)

            print("1/16")
        elif len(predators) > 5000 and len(preys) < 100:
            print("Simulation ended due to predator overpopulation.")
            self.simulation.stop()

        elif len(preys) > 5000 and len(predators) < 100:
            print("Simulation ended due to prey overpopulation.")
            self.simulation.stop()

        if len(predators) <= 1 or len(preys) <= 1:
            print("Simulation ended due to extinction.")
            self.simulation.stop()
            ## same here

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

        # Reproduction - only if another prey is nearby and has enough energy
        if (self.energy >= self.config.prey_reproduction_energy_threshold and random.random() < self.config.prey_reproduction_probability) and self.state == PREY_WANDERING:  # Small chance to check each frame
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Prey):
                    if (distance < self.config.prey_reproduction_radius and 
                        agent.energy >= self.config.prey_reproduction_energy_threshold and 
                        ((self.sex == "male" and agent.sex == "female") or (self.sex == "female" and agent.sex == "male"))):
                            # Only one of them reproduces (random choice)
                            self.energy -= self.config.prey_reproduction_cost
                            agent.energy -= self.config.prey_reproduction_cost
                            self.reproduce()
                            agent.reproduce()
                            break

    def change_position(self):
        if self.state == PREY_EATEN:
            self.move = Vector2(0, 0)  # Don't move while being eaten
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
                    # Fallback to small random direction
                    food_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
                else:
                    food_direction = delta.normalize()
                self.move = food_direction * self.config.prey_speed
            else:
                # Match predator random movement style
                if random.random() < 0.05 or self.move.length() == 0:
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
                # Fallback random movement if no predator detected but still in fleeing state
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
        
        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            if self.id in predators:
                predators.remove(self.id)
            return
        
        # Check for reproduction with nearby predators
        if self.energy >= self.config.predator_reproduction_energy_threshold and random.random() < self.config.predator_reproduction_probability:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Predator):
                    if (distance < self.config.predator_reproduction_radius and 
                        agent.energy >= self.config.predator_reproduction_energy_threshold and
                        ((self.sex == "male" and agent.sex == "female") or (self.sex == "female" and agent.sex == "male"))):
                        # Both predators can reproduce
                        self.energy -= self.config.predator_reproduction_cost
                        agent.energy -= self.config.predator_reproduction_cost
                        self.reproduce()
                        agent.reproduce()
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
                # Caught prey!
                if nearest_prey.state != PREY_EATEN:  # Only if not already being eaten
                    nearest_prey.start_being_eaten()
                    self.energy += self.config.predator_eating_energy
                    self.state = PREDATOR_EATING
                    self.change_image(PREDATOR_EATING)
                    self.eating_timer = self.config.eating_duration
                    self.target_prey = None
            
    def change_position(self):
        if self.state == PREDATOR_EATING:
            # Stay still while eating
            self.move = Vector2(0, 0)
        elif self.state == PREDATOR_LUNGING and self.target_prey:
            # Lunge toward target prey
            delta = self.target_prey.pos - self.pos
            if delta.length() == 0:
                # Rare fallback to random direction
                lunge_direction = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
            else:
                lunge_direction = delta.normalize()
            self.move = lunge_direction * self.config.predator_lunge_speed
        else:
            # Random hunting movement
            if random.random() < 0.05 or self.move.length() == 0:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        # Keep within bounds
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
            images=["images/transparent.png"],  # Placeholder image for FoodSpawner)
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

    plt.title('Realistic Lotka-Volterra Model')
    plt.xlabel('Time (frames)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predator_prey.png')
    plt.close()

    print("Plotting complete. Plot saved to predator_prey.png")

# Define the parameter space to evolve
PARAM_SPACE = {
    "prey_energy_consumption": (0.15, 0.4),
    "prey_flee_energy_consumption": (0.2, 0.6),
    "prey_reproduction_probability": (0.01, 0.3),
    "predator_energy_consumption": (0.1, 0.3),
    "predator_lunge_energy_consumption": (0.3, 0.7),
    "predator_reproduction_probability": (0.1, 0.6),
    "food_spawn_rate": (0.0005, 0.01),
}

def get_config_from_genes(genes):
    keys = list(PARAM_SPACE.keys())
    cfg = PredatorPreyConfig()
    for i, key in enumerate(keys):
        setattr(cfg, key, genes[i])
    return cfg

def cyclic_score(series):
    # Normalize series
    norm = (series - np.mean(series)) / np.std(series)
    fft = np.fft.fft(norm)
    freqs = np.fft.fftfreq(len(fft))
    amplitudes = np.abs(fft)

    # Exclude DC component (index 0)
    amplitudes[0] = 0
    dominant_idx = np.argmax(amplitudes)
    dominant_amp = amplitudes[dominant_idx]

    # Score is just dominant frequency amplitude
    return dominant_amp

def evaluate_individual(genes, sim_duration=2000):
    global frame_data, food_count, preys, predators
    # Reset globals
    frame_data = {k: [] for k in frame_data}
    food_count = 15
    preys.clear()
    predators.clear()

    cfg = get_config_from_genes(genes)
    cfg.duration = sim_duration

    try:
        sim = (
            HeadlessSimulation(cfg)
            .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
            .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
            .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
            .spawn_agent(FoodSpawner, images=["images/transparent.png"])
            .run()
        )
        df = pd.DataFrame(frame_data)
        
        # Duration component (0-1, 1 is best)
        duration_ratio = len(df) / sim_duration
        
        # Minimum duration threshold (20% of target duration)
        min_duration = sim_duration * 0.2
        if len(df) < min_duration:
            return 0  # Simulation ended too early
        
        df["total_prey"] = df["prey_wandering"] + df["prey_fleeing"]
        df["total_predators"] = df["predator_hunting"] + df["predator_lunging"] + df["predator_eating_prey"]

        # Cyclic behavior score (0-1, 1 is best)
        prey_score = cyclic_score(df["total_prey"].values)
        pred_score = cyclic_score(df["total_predators"].values)
        cyclic_behavior = (prey_score + pred_score) / 2
        
        # Population stability (penalize extreme values)
        max_prey = df["total_prey"].max()
        max_pred = df["total_predators"].max()
        stability = 1 / (1 + abs(max_prey - 500)/500 + abs(max_pred - 100)/100)
        
        # Phase relationship score (predators should lag prey)
        cross_corr = np.correlate(
            (df["total_prey"] - df["total_prey"].mean()).values,
            (df["total_predators"] - df["total_predators"].mean()).values,
            mode='full'
        )
        ideal_lag = 50  # frames
        actual_lag = np.argmax(cross_corr) - len(df) + 1
        phase_score = 1 / (1 + abs(actual_lag - ideal_lag)/ideal_lag)
        
        # Combined fitness (weighted components)
        fitness = (
            0.4 * cyclic_behavior + 
            0.3 * duration_ratio +
            0.2 * stability +
            0.1 * phase_score
        )
        
        # Ensure minimum value (0.1 for any completed simulation)
        return max(0.1, fitness)
        
    except Exception as e:
        print("Error during evaluation:", e)
        return 0

def evolve_population(generations=10, pop_size=20, elite_size=4, mutation_rate=0.2):
    keys = list(PARAM_SPACE.keys())
    bounds = [PARAM_SPACE[k] for k in keys]
    
    def random_individual():
        return [random.uniform(lo, hi) for lo, hi in bounds]

    def mutate(ind):
        return [
            min(max(val + np.random.normal(0, 0.05 * (hi - lo)), lo), hi)
            if random.random() < mutation_rate else val
            for val, (lo, hi) in zip(ind, bounds)
        ]

    population = [random_individual() for _ in range(pop_size)]
    
    for gen in range(generations):
        print(f"\n--- Generation {gen+1} ---")
        with multiprocessing.Pool() as pool:
            scores = pool.map(evaluate_individual, population)

        sorted_pop = [x for _, x in sorted(zip(scores, population), key=lambda p: -p[0])]
        best_score = max(scores)
        print(f"Best fitness: {best_score:.4f}")

        elites = sorted_pop[:elite_size]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            parent = random.choice(elites)
            child = mutate(parent)
            new_pop.append(child)
        population = new_pop

    best_ind = population[0]
    print("\nBest parameter set:")
    for k, v in zip(keys, best_ind):
        print(f"{k}: {v:.4f}")
    return best_ind

if __name__ == "__main__":
    best_genes = evolve_population(generations=8, pop_size=16, elite_size=4)
    
    print("\nRunning final simulation with best parameters...")
    cfg = get_config_from_genes(best_genes)
    cfg.duration = 5000

    frame_data = {k: [] for k in frame_data}
    food_count = 15
    preys.clear()
    predators.clear()

    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(FoodSpawner, images=["images/transparent.png"])
        .run()
    )