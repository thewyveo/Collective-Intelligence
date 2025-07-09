from dataclasses import dataclass
from vi import Agent, Config, Simulation, HeadlessSimulation
import random
import pandas as pd
import matplotlib.pyplot as plt
from pygame.math import Vector2


# Ensure we can access the Simulation object from each Agent instance
# convenience handle inside vi.Agent
#if not hasattr(Agent, "simulation"):
    #Agent.simulation = property(lambda self: self._Agent__simulation)

# Image paths
SPECIES_A_IMAGES = ["images/triangle2.png"]    # Rock (kills C, killed by B)
SPECIES_B_IMAGES = ["images/triangle4.png"]     # Paper (kills A, killed by C)
SPECIES_C_IMAGES = ["images/triangle6.png"]       # Scissors (kills B, killed by A)

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
    initial_species_a = 30
    initial_species_b = 24
    initial_species_c = 6
    
    # Parameters for species interactions
    species_a_birth_rate = 0.02
    species_b_birth_rate = 0.02
    species_c_birth_rate = 0.02

    a_kills_b_rate  = 0.00053
    b_kills_c_rate = 0.00087
    c_kills_a_rate =  0.00053

    self_crowd_coeff = 0.00133
    capacity = 3000
    # Energy parameters
    #initial_energy: float = 12.0
    #energy_decay_rate: float = 0.3
    #energy_gain_from_eating: float = 3.0
    
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 0.02
    duration: int = 5000
    movement_speed: float = 10.0

class SpeciesA(Agent):  # Rock - kills C, killed by B
    config: RockPaperScissorsConfig
    energy: float = 10.0
    
    def on_spawn(self):
        global species_a
        if self.id not in species_a:
            species_a.add(self.id)
        #self.energy = self.config.initial_energy
    
    def update(self):
        global species_a, species_b
        kill_rate = self.config.a_kills_b_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        #self.energy -= self.config.energy_decay_rate
        Ni = sum(1 for ag in self.simulation._agents if isinstance(ag, SpeciesA))
        if random.random() < self.config.self_crowd_coeff * Ni:
            self.kill()
            if self.id in species_a:
                species_a.remove(self.id)
            return

        # 1️⃣  — correct visual comment + prey target  (A kills B)
        species_b_agents = [ag for ag in self.simulation._agents if isinstance(ag, SpeciesB)]
        if species_b_agents and random.random() < kill_rate:  # ← use jittered rate
            prey = random.choice(species_b_agents)
            prey.kill()
            if prey.id in species_b:
                species_b.remove(prey.id)
            #self.energy += self.config.energy_gain_from_eating

        # Reproduce based on birth rate
        if random.random() < self.config.species_a_birth_rate and len([agent for agent in self.simulation._agents if isinstance(agent, SpeciesA)])< self.config.capacity :
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
        #self.energy = self.config.initial_energy
    
    def update(self):
        global species_b, species_c
        kill_rate = self.config.b_kills_c_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        #self.energy -= self.config.energy_decay_rate
        Ni = sum(1 for ag in self.simulation._agents if isinstance(ag, SpeciesB))
        if random.random() < self.config.self_crowd_coeff * Ni:
            self.kill()
            if self.id in species_b:
                species_b.remove(self.id)
            return
        
        # B kills C (paper covers rock)
        species_c_agents = [agent for agent in self.simulation._agents if isinstance(agent, SpeciesC)]
        if species_c_agents and random.random() < self.config.b_kills_c_rate:
            prey = random.choice(species_c_agents)
            prey.kill()
            if prey.id in species_c:
                species_c.remove(prey.id)
            #self.energy += self.config.energy_gain_from_eating
        
        # Reproduce based on birth rate
        if random.random() < self.config.species_b_birth_rate and len([agent for agent in self.simulation._agents if isinstance(agent, SpeciesB)])< self.config.capacity:
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
        #self.energy = self.config.initial_energy
    
    def update(self):
        global species_c, species_b
        kill_rate = self.config.c_kills_a_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)

        Ni = sum(1 for ag in self.simulation._agents if isinstance(ag, SpeciesC))
        if random.random() < self.config.self_crowd_coeff * Ni:
            self.kill()
            if self.id in species_c:
                species_c.remove(self.id)
            return
        
        # C kills B (scissors cut paper)
        species_a_agents = [agent for agent in self.simulation._agents if isinstance(agent, SpeciesA)]
        if species_a_agents and random.random() < self.config.c_kills_a_rate:
            prey = random.choice(species_a_agents)
            prey.kill()
            if prey.id in species_a:
                species_a.remove(prey.id)
            #self.energy += self.config.energy_gain_from_eating

        # Reproduce based on birth rate
        if random.random() < self.config.species_c_birth_rate and len([agent for agent in self.simulation._agents if isinstance(agent, SpeciesC)])< self.config.capacity:
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
            print("Multiple extinctions occurred")
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
        print("Plot saved to rock-paper-scissors.png")

def run_rock_paper_scissors(frame_data):
    cfg = RockPaperScissorsConfig()
    
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_species_a, SpeciesA, images=SPECIES_A_IMAGES)
        .batch_spawn_agents(cfg.initial_species_b, SpeciesB, images=SPECIES_B_IMAGES)
        .batch_spawn_agents(cfg.initial_species_c, SpeciesC, images=SPECIES_C_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
    )
    
    sim.run()

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
    print("Plot saved to rock-paper-scissors.png")

import numpy as np
import random
from vi import HeadlessSimulation, Simulation
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

# EA CONFIG
POPULATION_SIZE = 15
GENERATIONS = 30
MUTATION_RATE = 0.125
ELITISM = 6
BOUNDS = {
    "species_a_birth_rate": (0.005, 0.05),
    "species_b_birth_rate": (0.005, 0.05),
    "species_c_birth_rate": (0.005, 0.05),
    "a_kills_b_rate": (0.0001, 0.002),
    "b_kills_c_rate": (0.0001, 0.002),
    "c_kills_a_rate": (0.0001, 0.002),
    "self_crowd_coeff": (0.0001, 0.005),
    "movement_speed": (5.0, 30.0)
}


def dominant_oscillation_power(series):
    x = series.to_numpy()
    n = len(x)
    yf = np.abs(fft(x - np.mean(x)))
    power_spectrum = yf[1:n // 2]
    return np.max(power_spectrum)


def random_individual():
    return {k: random.uniform(*v) for k, v in BOUNDS.items()}


def mutate(ind):
    mutant = ind.copy()
    for k in BOUNDS:
        if random.random() < MUTATION_RATE:
            low, high = BOUNDS[k]
            mutant[k] = np.clip(mutant[k] + np.random.normal(0, (high - low) * 0.1), low, high)
    return mutant


def crossover(p1, p2):
    return {k: random.choice([p1[k], p2[k]]) for k in BOUNDS}


def evaluate(ind, n_repeats=3):
    scores = []
    for _ in range(n_repeats):
        global frame_data, species_a, species_b, species_c
        frame_data = {"frame": [], "species_a": [], "species_b": [], "species_c": []}
        species_a.clear()
        species_b.clear()
        species_c.clear()

        cfg = RockPaperScissorsConfig()
        for k, v in ind.items():
            setattr(cfg, k, v)
        cfg.duration = 5000

        sim = (
            HeadlessSimulation(cfg)
            .batch_spawn_agents(cfg.initial_species_a, SpeciesA, images=SPECIES_A_IMAGES)
            .batch_spawn_agents(cfg.initial_species_b, SpeciesB, images=SPECIES_B_IMAGES)
            .batch_spawn_agents(cfg.initial_species_c, SpeciesC, images=SPECIES_C_IMAGES)
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
        scores.append((osc_a + osc_b + osc_c) * duration_factor)

    return sorted(scores)[1]


def evolve():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    global_best, best_score = None, -float("inf")

    for gen in range(GENERATIONS):
        print(f"Generation {gen}...")
        fitnesses = [evaluate(ind) for ind in population]
        scored = sorted(zip(fitnesses, population), reverse=True, key=lambda x: x[0])

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            global_best = scored[0][1]

        print(f"Gen best: {scored[0][0]:.2f} | Global best: {best_score:.2f}")
        for k, v in scored[0][1].items():
            print(f"  {k}: {v:.5f}")

        next_gen = [s[1] for s in scored[:ELITISM]]
        while len(next_gen) < POPULATION_SIZE:
            if random.random() < 0.8:
                child = mutate(global_best)
            else:
                p1, p2 = random.sample(population, 2)
                child = mutate(crossover(p1, p2))
            next_gen.append(child)
        population = next_gen

        print(f"Best current parameters in generation {gen+1}/{GENERATIONS}:")
        for k, v in global_best.items():
            print(f"  {k}: {v:.5f}")

    return global_best, best_score


def visualize_best(best_ind):
    print("\nVisualizing best individual...")
    global frame_data, species_a, species_b, species_c
    frame_data = {"frame": [], "species_a": [], "species_b": [], "species_c": []}
    species_a.clear()
    species_b.clear()
    species_c.clear()

    cfg = RockPaperScissorsConfig()
    for k, v in best_ind.items():
        setattr(cfg, k, v)
    cfg.duration = 5000

    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_species_a, SpeciesA, images=SPECIES_A_IMAGES)
        .batch_spawn_agents(cfg.initial_species_b, SpeciesB, images=SPECIES_B_IMAGES)
        .batch_spawn_agents(cfg.initial_species_c, SpeciesC, images=SPECIES_C_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
    )

    sim.run()
    df = pd.DataFrame(frame_data)
    plt.plot(df["frame"], df["species_a"], label="A", color="red")
    plt.plot(df["frame"], df["species_b"], label="B", color="blue")
    plt.plot(df["frame"], df["species_c"], label="C", color="green")
    plt.legend()
    plt.title("Best Rock-Paper-Scissors Dynamics")
    plt.xlabel("Frame")
    plt.ylabel("Population")
    plt.grid(True)
    plt.savefig("rps_ea_best.png")
    plt.close()
    print("Saved to rps_ea_best.png")


if __name__ == "__main__":
    best, score = evolve()
    visualize_best(best)