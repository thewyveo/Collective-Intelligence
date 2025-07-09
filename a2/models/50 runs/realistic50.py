import matplotlib.pyplot as plt
from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
from pygame.math import Vector2
import random
import polars as pl
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
    "prey": [],
    "predator": [],
    "food": []
}

food_count = 15
preys = set()
predators = set()
extinction = False

@dataclass
class PredatorPreyConfig(Config):
    # Prey parameters
    prey_speed: float = 4.5
    prey_flee_speed: float = 7.2
    prey_vision: float = 100.0
    prey_energy: float = 200.0
    prey_energy_consumption: float = 0.1
    prey_flee_energy_consumption: float = 0.15
    prey_energy_gain: float = 35.0  # Energy gained from eating food
    prey_reproduction_energy_threshold: float = 150.0
    prey_reproduction_cost: float = 50.0
    prey_reproduction_radius: float = 27.5
    prey_reproduction_probability: float = 0.999999999999999999999999999
    
    # Predator parameters
    predator_speed: float = 6.25
    predator_lunge_speed: float = 9.5
    predator_vision: float = 150.0
    predator_energy: float = 150.0
    predator_energy_consumption: float = 0.99
    predator_lunge_energy_consumption: float = 2.0
    predator_eating_threshold: float = 10.0
    predator_eating_energy: float = 75.0
    predator_reproduction_energy_threshold: float = 70.0
    predator_reproduction_cost: float = 44.0
    predator_reproduction_radius: float = 25.0  # Distance for reproduction

    # Eating parameters
    eating_duration: int = 10  # Frames prey shows being eaten / predator is eating
    
    # Food parameters
    food_spawn_rate: float = 0.125  # Probability of food spawning per frame
    
    # Simulation parameters
    initial_prey: int = 20
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
            global food_count, extinction
            food_count += 1
            ## i am really proud of myself on this one: i overrode the main violet library's Agent class to include
            ## self.simulation, which didn't exist before, so that we can access the simulation directly within an agent class
            ## there is a self.__simulation attribute in the Agent class, but it is either a) not working or b) not accessible
            ## so i just added it myself -k

        current_frame = self.simulation.shared.counter

        prey = sum(1 for a in self.simulation._agents if isinstance(a, Prey))
        predator = sum(1 for a in self.simulation._agents if isinstance(a, Predator))
        food_count = sum(1 for a in self.simulation._agents if isinstance(a, Food))
        
        # Record frame data
        frame_data["frame"].append(current_frame)
        frame_data["prey"].append(prey)
        frame_data["predator"].append(predator)
        frame_data["food"].append(food_count)

        if len(predators) <= 1 or len(preys) <= 1:
            print("Simulation ended due to extinction.")
            self.simulation.stop()
            extinction = True
            ## same here

        if self.simulation.shared.counter % 100 == 0:
            print(f"Frame {self.simulation.shared.counter}: "
                  f"Prey: {len(preys)}, Predators: {len(predators)}, Food: {food_count}")
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
        self.age = 0
    
    def update(self):

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
            return
            
        self.there_is_no_escape()
        
        if self.state == PREY_WANDERING:
            # Random wandering with occasional food seeking
            if random.random() < 0.05 or self.move.length() == 0:
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

        site_center = Vector2(500*3/5, 500/2)
        if self.state == PREY_WANDERING and self.pos.distance_to(site_center) < 100:
            if random.random() < 0.9:
                self.move = self.move * 0.975
            else:
                self.move = self.move * 1.65

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
        self.age = 0
        self.avoided = False  # Track if predator has avoided prey-only site
    
    def update(self):
        self.age += 1
        if self.age > 420:
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
            if isinstance(agent, Prey):
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
                    if random.random() < 0.05:
                        self.energy -= self.config.predator_reproduction_cost
                        child = self.reproduce()
                        predators.add(child.id)          
    
    def change_position(self):
        if self.avoided:
            self.avoided = False

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
            if random.random() < 0.05 or self.move.length() == 0:
                angle = random.uniform(0, 2 * math.pi)
                self.move = Vector2(math.cos(angle), math.sin(angle)) * self.config.predator_speed
        
        # Redirect predator away if it tries to enter prey-only site
        future_pos = self.pos + self.move * self.config.delta_time
        distance_to_site = future_pos.distance_to(Vector2(500*3/4, 500/2))

        if distance_to_site < 175 or self.on_site():
            # Push predator away from the center of the site
            avoidance_dir = (future_pos - Vector2(500*3/4, 500/2)).normalize()
            self.move = avoidance_dir * self.config.predator_speed
            self.avoided = True

        if self.there_is_no_escape() and not self.avoided:
            self.move = -self.move

        self.pos += self.move * self.config.delta_time


def run_simulation():
    cfg = PredatorPreyConfig()
    
    # Create and run simulation
    sim = (
        Simulation(cfg)
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
        .spawn_site("images/circle2.png", x=500*3/5, y=500/2)
        .run()
    )

    # Create DataFrame from collected data
    df = pd.DataFrame(frame_data)

    # Save data to CSV
    df.to_csv('realistic.csv', index=False)
    print("Data saved to realistic.csv, plotting...")

    plt.figure(figsize=(14, 8))

    plt.plot(df['frame'], df['prey'], label='Total Prey', color='green')
    plt.plot(df['frame'], df['predator'], label='Total Predators', color='red')
    plt.plot(df['frame'], df['food'], label='Food', color='blue', linestyle='--')

    plt.title('Realistic Lotka-Volterra')
    plt.xlabel('Time (frames)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('realistic.png')
    plt.close()

    print("Plotting complete. Plot saved to realistic.png")



import pandas as pd

# Holds all results across 50 runs
all_runs = []

# Run simulation 50 times
runs = 15
for run_id in range(runs):
    print(f"Running simulation {run_id + 1}/{runs}...")

    # Reset frame data for each run
    frame_data["frame"].clear()
    frame_data["prey"].clear()
    frame_data["predator"].clear()
    frame_data["food"].clear()

    # Reset global sets
    preys.clear()
    predators.clear()
    food_count = 15  # Reset food count for each run

    try:
        # Run simulation
        run_simulation()

        # Load this run's output (from the file that Monitor wrote)
        df = pd.read_csv("realistic.csv")
        df["run"] = run_id

        # Add to all_runs list
        if extinction:
            all_runs.append(df)

        extinction = False
    except KeyboardInterrupt as e:
        print(f"Error in run {run_id + 1}: {e}")
        continue

# Combine all runs
results_df = pd.concat(all_runs, ignore_index=True)
results_df.to_csv("all_50_runs.csv", index=False)

print("Saved 50-run results to all_50_runs.csv")

# Plot all 50 runs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

for run_id in results_df["run"].unique():
    run_df = results_df[results_df["run"] == run_id]
    plt.plot(run_df["frame"], run_df["prey"], color="green", alpha=0.3)
    plt.plot(run_df["frame"], run_df["predator"], color="red", alpha=0.3)

plt.title("Realistic Lotka-Volterra: 25 Runs")
plt.xlabel("Frame")
plt.ylabel("Population")
plt.grid(True)
plt.tight_layout()
plt.savefig("realistic_50_runs.png")
plt.close()
print("Plot saved to realistic_50_runs.png")