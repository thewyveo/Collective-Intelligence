import numpy as np
import pygame
import random
import sys
import math
from matplotlib import pyplot as plt

# Simulation parameters
DIAMETER = 300            # Diameter of the circular agar plate
CELL_SIZE = 2              # Pixels per cell
SPECIES = 3                # Number of species (rock, paper, scissors)
EMPTY = 0                  # Value for empty space

# Toxin parameters from the paper
# Strain R (Red) is strongest, G (Green) intermediate, B (Blue) weakest
TOXIN_STRENGTH = {
    1: 4.17,  # Strain R (Red) - strongest
    2: 1.53,  # Strain G (Green) - intermediate
    3: 1.0   # Strain B (Blue) - weakest
}

# Base interaction rates
BASE_REPRODUCTION_RATE = 0.40947575465628583
BASE_DEATH_RATE = 0.004332903579439568
BASE_DOMINATION_RATE = 0.8642648072136341  # Base rate at which one species dominates another

# Colors matching the paper
COLORS = [
    (255, 255, 255),       # Empty - white (agar)
    (255, 0, 0),           # Species 1 - red (Strain R)
    (0, 255, 0),           # Species 2 - green (Strain G)
    (0, 0, 255)            # Species 3 - blue (Strain B)
]

class BacterialToxinSimulation:
    def __init__(self, init_mode='grid', spacing=10):
        radius = DIAMETER // 2
        self.radius = radius
        self.center = (radius, radius)
        self.grid = np.zeros((DIAMETER, DIAMETER), dtype=int)
        self.next_grid = np.zeros_like(self.grid)
        self.population_history = []

        if init_mode == 'grid':
            self.initialize_grid_pattern(spacing)
        elif init_mode == 'vortex':
            self.initialize_vortex_pattern()
        elif init_mode == 'concentric':
            self.initialize_concentric_rings()
        elif init_mode == 'noise':
            self.initialize_noise_pattern()
        else:
            self.initialize_random()


        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((DIAMETER * CELL_SIZE, DIAMETER * CELL_SIZE))
        pygame.display.set_caption("Bacterial Toxin-Mediated RPS Simulation")
        self.clock = pygame.time.Clock()

    def initialize_random(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    rand_val = random.random()
                    threshold = 1.0 / (SPECIES + 1)  # +1 to leave space for EMPTY
                    assigned = False
                    for s in range(1, SPECIES + 1):
                        if rand_val < s * threshold:
                            self.grid[i, j] = s
                            assigned = True
                            break
                    if not assigned:
                        self.grid[i, j] = EMPTY
                else:
                    self.grid[i, j] = -1  # Outside agar

    def initialize_grid_pattern(self, spacing):
        strain = 1
        for i in range(0, DIAMETER, spacing):
            for j in range(0, DIAMETER, spacing):
                if self.is_in_circle(i, j):
                    self.grid[i, j] = strain
                    strain = strain % SPECIES + 1  # Cycle through 1→2→3→1...
                else:
                    self.grid[i, j] = -1

    def initialize_vortex_pattern(self, density=0.05):
        total_species = SPECIES
        angle_per_species = 2 * math.pi / total_species
        
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    self.grid[i, j] = -1
                    continue

                dx = i - self.center[0]
                dy = j - self.center[1]
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi

                species_index = int(angle // angle_per_species) + 1  # Species are 1-indexed
                if random.random() < density:
                    self.grid[i, j] = species_index
                else:
                    self.grid[i, j] = EMPTY

    def initialize_concentric_rings(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    dx = i - self.center[0]
                    dy = j - self.center[1]
                    dist = math.hypot(dx, dy)
                    ring = int(dist / (self.radius / SPECIES))
                    self.grid[i, j] = (ring % SPECIES) + 1
                else:
                    self.grid[i, j] = -1

    def initialize_noise_pattern(self):
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if self.is_in_circle(i, j):
                    noise_val = (math.sin(i * 0.1) + math.cos(j * 0.1)) * 0.5
                    species = int((noise_val + 1) / 2 * SPECIES)
                    self.grid[i, j] = (species % SPECIES) + 1
                else:
                    self.grid[i, j] = -1
        
    def is_in_circle(self, x, y):
        # Check if coordinates are within the circular agar plate
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx*dx + dy*dy <= self.radius*self.radius
    
    def get_interaction_rate(self, attacker, defender):
        """Calculate the interaction rate based on toxin strengths"""
        # From the paper: R kills G, G kills B, B kills R
        interaction_map = {
            (1, 2): TOXIN_STRENGTH[1],  # R kills G
            (2, 3): TOXIN_STRENGTH[2],  # G kills B
            (3, 1): TOXIN_STRENGTH[3]  # B kills R
        }

        # Normalize the interaction rates relative to the weakest toxin
        base_rate = interaction_map.get((attacker, defender), 0)
        return BASE_DOMINATION_RATE * (base_rate / TOXIN_STRENGTH[3])
    
    def update(self):
        self.next_grid = np.copy(self.grid)
        
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    continue
                
                current = self.grid[i, j]
                
                # Death process
                if current != EMPTY and random.random() < BASE_DEATH_RATE:
                    self.next_grid[i, j] = EMPTY
                    continue
                
                # Reproduction and competition
                if current != EMPTY:
                    # Find a random neighbor
                    ni, nj = self.get_random_neighbor(i, j)
                    if not self.is_in_circle(ni, nj):
                        continue
                    
                    neighbor = self.grid[ni, nj]
                    
                    # Reproduction to empty space
                    if neighbor == EMPTY and random.random() < BASE_REPRODUCTION_RATE:
                        self.next_grid[ni, nj] = current
                    
                    # Toxin-mediated dominance interactions
                    elif neighbor != EMPTY and neighbor != current:
                        # Get the appropriate interaction rate based on toxin strengths
                        domination_rate = self.get_interaction_rate(current, neighbor)
                        
                        if random.random() < domination_rate:
                            self.next_grid[ni, nj] = current
        
        self.grid = self.next_grid
        
        # Record population counts
        counts = [np.sum(self.grid == i) for i in range(SPECIES + 1)]
        self.population_history.append(counts[1:])  # Exclude empty count
        
    def get_random_neighbor(self, i, j):
        # Get random neighbor (no periodic boundaries)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        di, dj = random.choice(directions)
        ni, nj = i + di, j + dj
        
        # Ensure we stay within grid bounds
        ni = max(0, min(DIAMETER - 1, ni))
        nj = max(0, min(DIAMETER - 1, nj))
        
        return ni, nj
    
    def draw(self):
        # Draw agar plate (white circle)
        pygame.draw.circle(self.screen, (255, 255, 255), 
                          (self.center[0] * CELL_SIZE, self.center[1] * CELL_SIZE), 
                          self.radius * CELL_SIZE)
        
        # Draw bacteria
        for i in range(DIAMETER):
            for j in range(DIAMETER):
                if not self.is_in_circle(i, j):
                    continue
                
                species = self.grid[i, j]
                if species != EMPTY:
                    pygame.draw.rect(self.screen, COLORS[species], 
                                   (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
    
    def run(self, generations):
        running = True
        for _ in range(generations):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
                
            self.update()
            self.draw()
            self.clock.tick(60)
            if _ % 100 == 0:
                print(f"Generation {_ + 1}/{generations} completed")
        
        pygame.quit()
        self.plot_population_history()
    
    def plot_population_history(self):
        history = np.array(self.population_history)
        plt.figure(figsize=(10, 6))
        plt.plot(history[:, 0], 'r-', label='Strain R (Strongest)')
        plt.plot(history[:, 1], 'g-', label='Strain G (Intermediate)')
        plt.plot(history[:, 2], 'b-', label='Strain B (Weakest)')
        plt.title('Population Dynamics with Asymmetric Toxin Strengths')
        plt.xlabel('Time Steps')
        plt.ylabel('Population Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('bacteria_diff.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Population dynamics plot saved as 'bacteria_diff.png'")


if __name__ == "__main__":
    print("Starting simulation with toxin-mediated interactions")
    print("Toxin strength ratios (from paper):")
    print(f"Strain R (Red): {TOXIN_STRENGTH[1]} (strongest)")
    print(f"Strain G (Green): {TOXIN_STRENGTH[2]} (intermediate)")
    print(f"Strain B (Blue): {TOXIN_STRENGTH[3]} (weakest, reference)")

    # Change 'init_mode' to 'random' or 'grid'
    sim = BacterialToxinSimulation(init_mode='grid')
    sim.run(5000)


