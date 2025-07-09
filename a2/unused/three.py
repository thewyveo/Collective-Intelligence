from dataclasses import dataclass, field
from vi import Agent, Config, Simulation, HeadlessSimulation
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from pygame.math import Vector2

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
    initial_species_a = initial_species_b = initial_species_c = int(67.32057)
    
    # Parameters for species interactions
    species_a_birth_rate = species_b_birth_rate = species_c_birth_rate = 0.01
    species_a_death_rate = species_b_death_rate = species_c_death_rate = 0.00012
    a_kills_c_rate = b_kills_a_rate = c_kills_b_rate = 0.001
    species_a_reproduction_rate = species_b_reproduction_rate = species_c_reproduction_rate = 0.30214

    # Energy parameters
    initial_energy: float = 12.20441
    energy_decay_rate: float = 0.05904
    energy_gain_from_eating: float = 3.6346
    
    window_size: tuple[int, int] = (1000, 1000)
    delta_time: float = 1.0
    duration: int = 1000
    movement_speed: float = 23.32018

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
        kill_rate = self.config.a_kills_c_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.species_a_death_rate:
            self.kill()
            if self.id in species_a:
                species_a.remove(self.id)
            return
        
        # A kills C (rock crushes scissors)
        species_c_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesC)]
        if species_c_agents and random.random() < self.config.a_kills_c_rate:
            prey = random.choice(species_c_agents)
            prey.kill()
            if prey.id in species_c:
                species_c.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.species_a_reproduction_rate:
                self.reproduce()
                if self.id not in species_a:
                    species_a.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.species_a_birth_rate:
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
        kill_rate = self.config.b_kills_a_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.species_b_death_rate:
            self.kill()
            if self.id in species_b:
                species_b.remove(self.id)
            return
        
        # B kills A (paper covers rock)
        species_a_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesA)]
        if species_a_agents and random.random() < self.config.b_kills_a_rate:
            prey = random.choice(species_a_agents)
            prey.kill()
            if prey.id in species_a:
                species_a.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.species_b_reproduction_rate:
                self.reproduce()
                if self.id not in species_b:
                    species_b.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.species_b_birth_rate:
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
        kill_rate = self.config.a_kills_c_rate + random.uniform(-0.0001, 0.0001)
        kill_rate = max(0, kill_rate)
        # Energy decreases each time step
        self.energy -= self.config.energy_decay_rate
        
        # Die if energy depleted or natural death
        if self.energy <= 0 or random.random() < self.config.species_c_death_rate:
            self.kill()
            if self.id in species_c:
                species_c.remove(self.id)
            return
        
        # C kills B (scissors cut paper)
        species_b_agents = [agent for agent in self.simulation.agents if isinstance(agent, SpeciesB)]
        if species_b_agents and random.random() < self.config.c_kills_b_rate:
            prey = random.choice(species_b_agents)
            prey.kill()
            if prey.id in species_b:
                species_b.remove(prey.id)
            self.energy += self.config.energy_gain_from_eating
            
            if random.random() < self.config.species_c_reproduction_rate:
                self.reproduce()
                if self.id not in species_c:
                    species_c.add(self.id)
        
        # Reproduce based on birth rate
        if random.random() < self.config.species_c_birth_rate:
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
        if frame % 100 == 0:
            print(f"Frame {frame}: A={species_a_count}, B={species_b_count}, C={species_c_count}")

        # Stop conditions
        extinct_count = sum([species_a_count == 0, species_b_count == 0, species_c_count == 0])
        if extinct_count >= 2:  # Stop if 2 or more species are extinct
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

def run_rock_paper_scissors():
    cfg = RockPaperScissorsConfig()
    
    sim = (
        Simulation(cfg)
        .batch_spawn_agents(cfg.initial_species_a, SpeciesA, images=SPECIES_A_IMAGES)
        .batch_spawn_agents(cfg.initial_species_b, SpeciesB, images=SPECIES_B_IMAGES)
        .batch_spawn_agents(cfg.initial_species_c, SpeciesC, images=SPECIES_C_IMAGES)
        .spawn_agent(Monitor, images=["images/transparent.png"])
    )
    
    sim.run()

if __name__ == "__main__":
    run_rock_paper_scissors()