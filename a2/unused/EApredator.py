import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import math
import pandas as pd
import numpy as np
from copy import deepcopy

# Image paths
PREY_IMAGES = ["images/green.png", "images/yellow.png", "images/red.png"]
PREDATOR_IMAGES = ["images/triangle5.png", "images/triangle7.png", "images/triangle6.png"]
FOOD_IMAGES = ["images/plus.png"]

# States
PREY_WANDERING, PREY_FLEEING, PREY_EATEN = 0, 1, 2
PREDATOR_HUNTING, PREDATOR_LUNGING, PREDATOR_EATING = 0, 1, 2

# Frame data tracker
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

# For stagnation tracking
last_change_frame = {"prey": 0, "predator": 0}
last_counts = {"prey": None, "predator": None}
STAGNATION_THRESHOLD = 500
penalized_params = {}

@dataclass
class PredatorPreyConfig(Config):
    prey_energy_gain: float = 30.0
    prey_energy_consumption: float = 0.1
    prey_reproduction_energy_threshold: float = 150.0
    predator_energy_consumption: float = 0.2
    predator_eating_energy: float = 30.0
    predator_reproduction_energy_threshold: float = 120.0
    prey_vision: float = 100.0
    predator_vision: float = 150.0
    prey_reproduction_radius: float = 20.0
    predator_reproduction_radius: float = 20.0
    prey_reproduction_probability: float = 0.05
    prey_speed: float = 3.0
    prey_flee_speed: float = 4.5
    prey_energy: float = 150.0
    predator_speed: float = 4.0
    predator_lunge_speed: float = 6.0
    predator_energy: float = 100.0
    predator_lunge_energy_consumption: float = 0.3
    predator_eating_threshold: float = 10.0
    prey_flee_energy_consumption: float = 0.15
    prey_reproduction_cost: float = 50.0
    predator_reproduction_cost: float = 60.0
    eating_duration: int = 30
    food_spawn_rate: float = 0.005
    initial_prey: int = 50
    initial_predators: int = 10
    initial_food: int = 15
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000

class FoodSpawner(Agent):
    config: PredatorPreyConfig

    def on_spawn(self):
        self.change_image(0)
        self.recently_exploded = False
        self.explosion_timer = 0

    def update(self):
        global food_count
        if random.random() < self.config.food_spawn_rate:
            self.simulation.spawn_agent(Food, images=FOOD_IMAGES)
            food_count += 1

        frame = self.simulation.shared.counter
        if frame >= len(frame_data["frame"]):
            frame_data["frame"].append(frame)
            frame_data["prey_wandering"].append(sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.state == PREY_WANDERING))
            frame_data["prey_fleeing"].append(sum(1 for a in self.simulation._agents if isinstance(a, Prey) and a.state == PREY_FLEEING))
            frame_data["predator_hunting"].append(sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_HUNTING))
            frame_data["predator_lunging"].append(sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_LUNGING))
            frame_data["predator_eating_prey"].append(sum(1 for a in self.simulation._agents if isinstance(a, Predator) and a.state == PREDATOR_EATING))
            frame_data["food_count"].append(food_count)

        if frame % 100 == 0:
            print(f"frame {frame}: prey={len(preys)}, predators={len(predators)}, food={food_count}")

        # --- Stagnation detection
        global last_change_frame, last_counts

        if last_counts["prey"] != len(preys):
            last_counts["prey"] = len(preys)
            last_change_frame["prey"] = frame

        if last_counts["predator"] != len(predators):
            last_counts["predator"] = len(predators)
            last_change_frame["predator"] = frame

        # If either species hasn't changed in STAGNATION_THRESHOLD frames, end early
        if (
            frame - last_change_frame["prey"] > STAGNATION_THRESHOLD or
            frame - last_change_frame["predator"] > STAGNATION_THRESHOLD
        ):
            print("Simulation ended due to stagnation.")
            self.simulation.stop()

        # Add imbalance early stop (e.g., >1000 prey vs. <10 predators)
        total_prey = frame_data["prey_wandering"][-1] + frame_data["prey_fleeing"][-1]
        total_pred = frame_data["predator_hunting"][-1] + frame_data["predator_lunging"][-1] + frame_data["predator_eating_prey"][-1]

        if (len(preys) + len(predators)) > 15000:
            if self.recently_exploded:
                if (len(preys) > 1000 and len(predators) == 1) or (len(predators) > 1000 and len(preys) == 1):
                    print("Explosion in population detected, couldn't be fixed. Ending simulation.")
                    self.simulation.stop()
            else:
                # Halve prey
                prey_agents = [a for a in self.simulation._agents if isinstance(a, Prey)]
                for prey in random.sample(prey_agents, len(prey_agents) // 2):
                    prey.kill()
                    preys.discard(prey.id)

                # Halve predators
                predator_agents = [a for a in self.simulation._agents if isinstance(a, Predator)]
                for pred in random.sample(predator_agents, len(predator_agents) // 2):
                    pred.kill()
                    predators.discard(pred.id)

                self.recently_exploded = True
                print(f"Population explosion: reduced population to {len(preys)} preys and {len(predators)} predators. (half for both species)")

                if len(preys) == 1 or len(predators) == 1:
                    print("Simulation ended due to extinction after explosion.")
                    self.simulation.stop()
        else:
            if self.recently_exploded:
                self.explosion_timer -= 1
                if self.explosion_timer <= 0:
                    self.recently_exploded = False
                    self.explosion_timer = 0


        if len(preys) == 0 or len(predators) == 0:
            print("Simulation ended due to extinction.")
            self.simulation.stop()

class Food(Agent):
    config: PredatorPreyConfig

    def on_spawn(self):
        self.change_image(0)
        self.move = Vector2(0, 0)

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
    
    def update(self):
        if self.state == PREY_EATEN:
            self.eating_timer -= 1
            if self.eating_timer <= 0:
                self.kill()
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
        if (self.energy >= self.config.prey_reproduction_energy_threshold and random.random() < self.config.prey_reproduction_probability):  # Small chance to check each frame
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Prey):
                    if (distance < self.config.prey_reproduction_radius and 
                        agent.energy >= self.config.prey_reproduction_energy_threshold):
                        # Only one of them reproduces (random choice)
                            self.energy -= self.config.prey_reproduction_cost
                            self.reproduce()
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
    
    def update(self):
        if self.state == PREDATOR_LUNGING:
            self.energy -= self.config.predator_lunge_energy_consumption
        else:
            self.energy -= self.config.predator_energy_consumption
        
        # Die if energy depleted
        if self.energy <= 0:
            self.kill()
            predators.remove(self.id)
            return
        
        # Check for reproduction with nearby predators
        if self.energy >= self.config.predator_reproduction_energy_threshold:
            for agent, distance in self.in_proximity_accuracy():
                if isinstance(agent, Predator):
                    if (distance < self.config.predator_reproduction_radius and 
                        agent.energy >= self.config.predator_reproduction_energy_threshold):
                        # Both predators can reproduce
                        self.energy -= self.config.predator_reproduction_cost
                        agent.energy -= self.config.predator_reproduction_cost
                        self.reproduce()
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

# EA PARAMS ---
param_bounds = {
    "prey_energy_gain": (10.0, 50.0),
    "prey_energy_consumption": (0.05, 1),
    "prey_reproduction_energy_threshold": (30.0, 100.0),
    "predator_eating_energy": (10.0, 50.0),
    "predator_energy_consumption": (0.1, 1),
    "predator_reproduction_energy_threshold": (30.0, 80.0),
    "prey_vision": (50.0, 200.0),
    "predator_vision": (80.0, 250.0),
    "prey_reproduction_radius": (5.0, 50.0),
    "predator_reproduction_radius": (5.0, 50.0),
    "food_spawn_rate": (0.001, 0.01),
    "prey_reproduction_probability": (0.01, 0.1),
    "eating_duration": (5, 30),
}

# Utility
def mutate_params(params, bounds, scale=0.15):
    mutated = {}
    for k, v in params.items():
        low, high = bounds[k]
        stddev = (high - low) * scale

        if penalized_params.get(k) is not None:
            # Push away from penalized value
            bad_val = penalized_params[k]
            direction = -1 if v > bad_val else 1
            offset = direction * abs(np.random.normal(0, stddev))
            mutated_val = np.clip(v + offset, low, high)
        else:
            # Normal Gaussian mutation
            mutated_val = np.clip(np.random.normal(v, stddev), low, high)

        mutated[k] = mutated_val

    return mutated

# Replace mutate_params() with tracking version
def mutate_params_with_tracking(params, bounds, scale, mutation_history):
    mutated = {}
    changed_keys = []

    for k, v in params.items():
        low, high = bounds[k]
        stddev = (high - low) * scale

        if penalized_params.get(k) is not None:
            bad_val = penalized_params[k]
            direction = -1 if v > bad_val else 1
            offset = direction * abs(np.random.normal(0, stddev))
            mutated_val = np.clip(v + offset, low, high)
        else:
            mutated_val = np.clip(np.random.normal(v, stddev), low, high)

        mutated[k] = mutated_val

        if abs(mutated_val - v) > 1e-6:
            changed_keys.append(k)
            mutation_history[k]["attempts"] += 1

    # Evaluate mutation success externally and update mutation_history[k]["success"] accordingly
    return mutated, changed_keys

def run_simulation(cfg: PredatorPreyConfig):
    global frame_data, preys, predators, food_count

    # Reset all global state
    for key in frame_data:
        frame_data[key].clear()
    preys.clear()
    predators.clear()
    food_count = cfg.initial_food

    sim = (
        HeadlessSimulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(FoodSpawner, images=["images/transparent.png"])
        .run()
    )

    df = pd.DataFrame(frame_data)
    df.to_csv("predator_prey_data.csv", index=False)
    return df

def run_simulation_visual(cfg: PredatorPreyConfig):
    global frame_data, preys, predators, food_count

    # Reset all global state
    for key in frame_data:
        frame_data[key].clear()
    preys.clear()
    predators.clear()
    food_count = cfg.initial_food

    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_prey, Prey, images=PREY_IMAGES)
        .batch_spawn_agents(cfg.initial_predators, Predator, images=PREDATOR_IMAGES)
        .batch_spawn_agents(cfg.initial_food, Food, images=FOOD_IMAGES)
        .spawn_agent(FoodSpawner, images=["images/transparent.png"])
        .run()
    )

    df = pd.DataFrame(frame_data)
    df.to_csv("predator_prey_data.csv", index=False)
    return df


def evaluate_individual(params):
    global penalized_params

    cfg = PredatorPreyConfig(**params)
    try:
        df = run_simulation(cfg)
    except:
        print("Simulation failed.")
        return 0.0

    if len(df) < 500:
        penalized_params = deepcopy(params)
        print("Simulation too short, penalizing params.")
        return 0.0

    prey = df["prey_wandering"] + df["prey_fleeing"]
    preds = df["predator_hunting"] + df["predator_lunging"] + df["predator_eating_prey"]

    if prey.iloc[-1] == 0 or preds.iloc[-1] == 0:
        penalized_params = deepcopy(params)
        print("Simulation ended with extinction, penalizing params.")
        return 0.0
    
    penalized_params = {}

    prey_std = prey.std()
    pred_std = preds.std()
    prey_peaks = len((prey.diff().shift(-1) < 0) & (prey.diff() > 0))
    pred_peaks = len((preds.diff().shift(-1) < 0) & (preds.diff() > 0))

    # Penalize species imbalance (ideal ratio near 1)
    avg_prey = prey.mean()
    avg_pred = preds.mean()
    ratio = avg_prey / (avg_pred + 1e-6)  # Avoid div by 0
    balance_penalty = abs(math.log(ratio))  # Closer to 0 is better

    oscillation_score = compute_oscillation_score(prey, preds)

    fitness = (
        0.4 * len(df) +                         # longevity still matters
        0.3 * (prey_std + pred_std) +           # promote fluctuating populations
        0.2 * (prey_peaks + pred_peaks) +       # promote oscillations
        0.5 * oscillation_score +              # promote oscillations 2
        - 0.2 * balance_penalty               # penalize domination
    )

    return fitness

def compute_oscillation_score(prey, preds, frame_window=200):
    num_frames = len(prey)
    num_windows = num_frames // frame_window

    dominators = []
    for i in range(num_windows):
        start = i * frame_window
        end = start + frame_window
        mean_prey = prey[start:end].mean()
        mean_pred = preds[start:end].mean()
        
        if mean_prey > mean_pred:
            dominators.append("prey")
        elif mean_pred > mean_prey:
            dominators.append("pred")

    # Count changes in dominator
    swaps = 0
    for i in range(1, len(dominators)):
        if dominators[i] != dominators[i - 1]:
            swaps += 1

    return swaps

def run_evolution():
    population_size = 10
    generations = 20

    population = [
        {k: np.random.uniform(*v) for k, v in param_bounds.items()}
        for _ in range(population_size)
    ]

    mutation_history = {
        k: {"success": 0, "attempts": 0} for k in param_bounds
    }

    best = None
    best_fitness = -float("inf")
    last_changed_keys = []

    for gen in range(generations):
        print(f"\n[GENERATION {gen+1}]")
        fitnesses = []

        for i, ind in enumerate(population):
            fitness = evaluate_individual(ind)
            print(f"G{gen+1}:  Individual {i+1}: fitness = {fitness:.2f}")
            fitnesses.append(fitness)

        top_idx = int(np.argmax(fitnesses))
        if fitnesses[top_idx] > best_fitness:
            best = deepcopy(population[top_idx])
            best_fitness = fitnesses[top_idx]
            print("🔥 New best individual found!")
            # Update mutation success count for last changed keys
            if "last_changed_keys" in locals():
                for k in last_changed_keys:
                    mutation_history[k]["success"] += 1

        # Elitism + mutation
        new_population = [deepcopy(best)]

        for _ in range(population_size - 1):
            mutated, changed_keys = mutate_params_with_tracking(best, param_bounds, scale=0.1, mutation_history=mutation_history)
            new_population.append(mutated)
            last_changed_keys.extend(changed_keys)  # Track what was changed

        population = new_population

    print("\n✅ Evolution complete!")
    print("Best parameters:")
    for k, v in best.items():
        print(f"  {k}: {v:.3f}")
    print(f"Final fitness: {best_fitness:.2f}")

if __name__ == "__main__":
    run_evolution()