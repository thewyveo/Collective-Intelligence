# global info (?)

import numpy as np
import random
from vi import HeadlessSimulation, Simulation
from vi import Agent, Vector2, Config
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np

def dominant_oscillation_power(series):
    x = series.to_numpy()  # convert to NumPy array
    n = len(x)
    yf = np.abs(fft(x - np.mean(x)))  # remove DC offset
    power_spectrum = yf[1:n // 2]     # exclude the 0-frequency component
    return np.max(power_spectrum)


# Image paths
SPECIES_A_IMAGES = ["images/triangle2.png"]    # Rock (kills C, killed by B)
SPECIES_B_IMAGES = ["images/triangle7.png"] # Paper (kills A, killed by C)
SPECIES_C_IMAGES = ["images/triangle8.png"]       # Scissors (kills B, killed by A)

species_a = set()
species_b = set()
species_c = set()

frame_data = {
    "frame": [],
    "species_a": [],
    "species_b": [],
    "species_c": [],
}

@dataclass
class RockPaperScissorsConfig(Config):
    # Initial population of each species
    initial_population: int = 100

    # Parameters for species interactions
    birth_rate: float = 0.01
    death_rate: float = 0.005
    predation_rate: float = 0.0006
    reproduction_rate: float = 0.3

    # Energy parameters
    initial_energy: float = 15.0
    energy_decay_rate: float = 0.02
    energy_gain_from_eating: float = 1.0
    
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 5000
    movement_speed: float = 20.0

class SpeciesA(Agent):  # Rock - kills C, killed by B
    config: RockPaperScissorsConfig
    energy: float = 10.0
    
    def on_spawn(self):
        global species_a
        if self.id not in species_a:
            species_a.add(self.id)
        self.energy = self.config.initial_energy
    
    def update(self):
        global species_a, species_c
        kill_rate = self.config.predation_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.death_rate:
            self.kill()
            if self.id in species_a:
                species_a.remove(self.id)
            return
        
        # A kills C (rock crushes scissors)
        species_c_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesC)]
        interaction_prob = kill_rate * len(species_c_agents)
        interaction_prob = min(interaction_prob, 0.5)  # avoid >50% kill chance in low pop cases
        if species_c_agents and random.random() < interaction_prob:
            prey = random.choice(species_c_agents)
            prey.kill()
            if prey.id in species_c:
                species_c.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.reproduction_rate:
                self.reproduce()
                if self.id not in species_a:
                    species_a.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.birth_rate:
            self.reproduce()
            if self.id not in species_a:
                species_a.add(self.id)

        # Random movement
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.movement_speed
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed
        self.pos += self.move * self.config.delta_time

class SpeciesB(Agent):  # Paper - kills A, killed by C
    config: RockPaperScissorsConfig
    energy: float = 10.0
    
    def on_spawn(self):
        global species_b
        if self.id not in species_b:
            species_b.add(self.id)
        self.energy = self.config.initial_energy
    
    def update(self):
        global species_b, species_a
        kill_rate = self.config.predation_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.death_rate:
            self.kill()
            if self.id in species_b:
                species_b.remove(self.id)
            return
        
        # B kills A (paper covers rock)
        species_a_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesA)]
        interaction_prob = kill_rate * len(species_a_agents)
        interaction_prob = min(interaction_prob, 0.5)  # avoid >50% kill chance in low pop cases
        if species_a_agents and random.random() < interaction_prob:
            prey = random.choice(species_a_agents)
            prey.kill()
            if prey.id in species_a:
                species_a.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.reproduction_rate:
                self.reproduce()
                if self.id not in species_b:
                    species_b.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.birth_rate:
            self.reproduce()
            if self.id not in species_b:
                species_b.add(self.id)

        # Random movement
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.movement_speed
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed
        self.pos += self.move * self.config.delta_time

class SpeciesC(Agent):  # Scissors - kills B, killed by A
    config: RockPaperScissorsConfig
    energy: float = 10.0
    
    def on_spawn(self):
        global species_c
        if self.id not in species_c:
            species_c.add(self.id)
        self.energy = self.config.initial_energy
    
    def update(self):
        global species_c, species_b
        kill_rate = self.config.predation_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.death_rate:
            self.kill()
            if self.id in species_c:
                species_c.remove(self.id)
            return
        
        # C kills B (scissors cut paper)
        species_b_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesB)]
        interaction_prob = kill_rate * len(species_b_agents)
        interaction_prob = min(interaction_prob, 0.5)  # avoid >50% kill chance in low pop cases
        if species_b_agents and random.random() < interaction_prob:
            prey = random.choice(species_b_agents)
            prey.kill()
            if prey.id in species_b:
                species_b.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.reproduction_rate:
                self.reproduce()
                if self.id not in species_c:
                    species_c.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.birth_rate:
            self.reproduce()
            if self.id not in species_c:
                species_c.add(self.id)

        # Random movement
        self.move = Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.config.movement_speed
        if self.move.length() > 0:
            self.move = self.move.normalize() * self.config.movement_speed

        self.pos += self.move * self.config.delta_time

class Monitor(Agent):
    config: RockPaperScissorsConfig
    
    def update(self):
        global frame_data

        frame = self.simulation.shared.counter
        
        # Count populations
        species_a_count = sum(1 for a in self.simulation._agents if isinstance(a, SpeciesA))
        species_b_count = sum(1 for a in self.simulation._agents if isinstance(a, SpeciesB))
        species_c_count = sum(1 for a in self.simulation._agents if isinstance(a, SpeciesC))
        
        # Store data
        frame_data["frame"].append(frame)
        frame_data["species_a"].append(species_a_count)
        frame_data["species_b"].append(species_b_count)
        frame_data["species_c"].append(species_c_count)

        # Print status every 100 frames
        #if frame % 100 == 0:
        #    print(f"Frame {frame}: A={species_a_count}, B={species_b_count}, C={species_c_count}")

        # Stop conditions
        extinct_count = sum([species_a_count == 0, species_b_count == 0, species_c_count == 0])
        if extinct_count >= 1:  # Stop if 2 or more species are extinct
            print("Extinction(s) occurred.")
            self.on_destroy()
            self.simulation.stop()
        if frame >= self.config.duration:
            print("Time limit reached")
            self.on_destroy()
            self.simulation.stop()
    
    def on_destroy(self):
        # Save data when simulation ends
        df = pd.DataFrame(frame_data)
        df.to_csv("rock-paper-scissors.csv", index=False)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df["frame"], df["species_a"], label="Species A (Rock)", color="green", linewidth=2)
        plt.plot(df["frame"], df["species_b"], label="Species B (Paper)", color="blue", linewidth=2)
        plt.plot(df["frame"], df["species_c"], label="Species C (Scissors)", color="red", linewidth=2)
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("Population")
        plt.title("Rock-Paper-Scissors Ecosystem\n(A kills C, B kills A, C kills B)")
        plt.grid(True, alpha=0.3)
        plt.savefig("rock-paper-scissors.png", dpi=300, bbox_inches='tight')
        plt.close()
        #print("Plot saved to rock-paper-scissors.png")x

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

# EA parameters
POPULATION_SIZE = 15
GENERATIONS = 30
MUTATION_RATE = 0.125
ELITISM = 6

# Bounds for each parameter
BOUNDS = {
    "death_rate": (0.0001, 0.002),
    "birth_rate": (0.01, 0.1),
    "reproduction_rate": (0.1, 0.5),
    "predation_rate": (0.0001, 0.001),
    "initial_population": (50, 100),
    "energy_gain_from_eating": (1.0, 5.0),
    "energy_decay_rate": (0.01, 0.1),
    "initial_energy": (10.0, 20.0),
    "movement_speed": (10.0, 30.0)
}

def random_individual():
    return {k: random.uniform(*v) for k, v in BOUNDS.items()}

def mutate(individual):
    mutant = individual.copy()
    for k in BOUNDS:
        if random.random() < MUTATION_RATE:
            low, high = BOUNDS[k]
            mutant[k] = np.clip(mutant[k] + np.random.normal(0, (high - low) * 0.1), low, high)
    return mutant

def crossover(p1, p2):
    child = {}
    for k in BOUNDS:
        child[k] = random.choice([p1[k], p2[k]])
    return child

def evaluate(individual, n_repeats=3):
    scores = []

    for _ in range(n_repeats):
        global frame_data, species_a, species_b, species_c
        frame_data = {
            "frame": [],
            "species_a": [],
            "species_b": [],
            "species_c": [],
        }
        species_a.clear()
        species_b.clear()
        species_c.clear()

        cfg = RockPaperScissorsConfig()
        for k, v in individual.items():
            setattr(cfg, k, v)
        cfg.duration = 5000

        sim = (
            HeadlessSimulation(cfg)
            .batch_spawn_agents(int(cfg.initial_population), SpeciesA, images=SPECIES_A_IMAGES)
            .batch_spawn_agents(int(cfg.initial_population), SpeciesB, images=SPECIES_B_IMAGES)
            .batch_spawn_agents(int(cfg.initial_population), SpeciesC, images=SPECIES_C_IMAGES)
            .spawn_agent(Monitor, images=["images/transparent.png"])
        )
        sim.run()

        df = pd.DataFrame(frame_data)
        if len(df) < 100:
            scores.append(0.0)
            continue

        osc_a = dominant_oscillation_power(df["species_a"])
        osc_b = dominant_oscillation_power(df["species_b"])
        osc_c = dominant_oscillation_power(df["species_c"])
        duration_factor = sim.shared.counter / cfg.duration

        # Fitness: encourage oscillation with some duration bonus
        score = (osc_a + osc_b + osc_c) * duration_factor
        scores.append(score)

    # Use minimum of 3 to punish instability, or second worst:
    return sorted(scores)[1]  # more stable than max or average

def evolve():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    global_best_ind = None
    global_best_score = -float("inf")

    for gen in range(GENERATIONS):
        print(f"Generation {gen}...")

        fitnesses = [evaluate(ind) for ind in population]
        scored = list(zip(fitnesses, population))
        scored.sort(reverse=True, key=lambda x: x[0])

        # Update global best if found
        if scored[0][0] > global_best_score:
            global_best_score = scored[0][0]
            global_best_ind = scored[0][1]

        print(f"Best fitness this gen: {scored[0][0]:.2f}, Global best: {global_best_score:.2f}")
        print(f"Best current parameters:")
        for k, v in scored[0][1].items():
            print(f"  {k}: {v:.5f}")

        # New generation starts with best individuals
        next_gen = [scored[i][1] for i in range(ELITISM)]

        # Hyper-elitist: fill rest by mutating best
        while len(next_gen) < POPULATION_SIZE:
            if random.random() < 0.8:
                child = mutate(global_best_ind)  # exploit best
            else:
                # occasional exploration
                p1, p2 = random.sample(population, 2)
                child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen

    print("\nGlobal Best Configuration:")
    for k, v in global_best_ind.items():
        print(f"{k}: {v:.5f}")
    print(f"Final Fitness: {global_best_score:.2f}")

    return global_best_ind, global_best_score

def visualize_best(best_ind, _):
    print("\nRunning best configuration in full simulation mode...")

    global frame_data, species_a, species_b, species_c
    frame_data =  {
        "frame": [],
        "species_a": [],
        "species_b": [],
        "species_c": [],
    }
    species_a.clear()
    species_b.clear()
    species_c.clear()
    
    cfg = RockPaperScissorsConfig()
    cfg.predation_rate = best_ind["predation_rate"]
    cfg.birth_rate = best_ind["birth_rate"]
    cfg.death_rate = best_ind["death_rate"]
    cfg.reproduction_rate = best_ind["reproduction_rate"]
    cfg.energy_gain_from_eating = best_ind["energy_gain_from_eating"]
    cfg.energy_decay_rate = best_ind["energy_decay_rate"]
    cfg.initial_energy = best_ind["initial_energy"]
    cfg.movement_speed = best_ind["movement_speed"]
    cfg.initial_population = int(best_ind["initial_population"])
    cfg.duration = 5000

    sim = (
        Simulation(cfg)
        .batch_spawn_agents(int(cfg.initial_population), SpeciesA, images=SPECIES_A_IMAGES)
        .batch_spawn_agents(int(cfg.initial_population), SpeciesB, images=SPECIES_B_IMAGES)
        .batch_spawn_agents(int(cfg.initial_population), SpeciesC, images=SPECIES_C_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
    )

    sim.run()

    df = pd.DataFrame(frame_data)
    plt.figure(figsize=(12, 6))
    plt.plot(df["frame"], df["species_a"], label="Species A (Rock)", color="green", linewidth=2)
    plt.plot(df["frame"], df["species_b"], label="Species B (Paper)", color="blue", linewidth=2)
    plt.plot(df["frame"], df["species_c"], label="Species C (Scissors)", color="red", linewidth=2)
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Population")
    plt.title("Rock-Paper-Scissors Ecosystem\n(A kills C, B kills A, C kills B)")
    plt.grid(True, alpha=0.3)
    plt.savefig("best-rock-paper-scissors.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to best-rock-paper-scissors.png")

if __name__ == "__main__":
    best_ind, best_fit = evolve()
    visualize_best(best_ind, best_fit)
