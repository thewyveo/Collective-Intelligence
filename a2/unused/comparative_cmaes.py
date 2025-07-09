# CMA-ES integrated refactoring
import cma
import numpy as np
from dataclasses import dataclass, field
from vi import Simulation, HeadlessSimulation, Window, Config
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import random
from Assignment_1.a2.unused.evolutionary_lotka_volterra import (
    reset_global_state, frame_data, Prey, Predator, Food, FoodSpawner,
    SimulationMonitor, PREY_IMAGES, PREDATOR_IMAGES, FOOD_IMAGES
)
from pygame.math import Vector2
import threading
import traceback
from tqdm import tqdm
from multiprocessing import Pool

class HybridEvoCMAESLotka:
    def __init__(self):
        self.param_bounds = {
            "prey_energy_gain": (25.0, 35.0),  # Reduced range
            "prey_energy_loss": (0.12, 0.2),   # Tighter range
            "predator_energy_gain": (35.0, 45.0),
            "predator_energy_loss": (0.18, 0.25),
            "prey_growth_rate": (0.06, 0.12),   # More stable growth
            "prey_carrying_capacity": (200.0, 300.0),
            "prey_reproduce_cost": (10.0, 15.0),
            "predator_reproduce_prob": (0.025, 0.04),
            "predator_reproduce_threshold": (30.0, 38.0),
            "predator_reproduce_cost": (18.0, 22.0),
            "food_spawn_prob": (0.02, 0.05),    # More reasonable food spawn
            "prey_speed": (1.5, 2.5),          # Reduced range
            "predator_speed": (2.0, 3.0),      # Predators should be faster
            "predation_distance": (12.0, 15.0), # Slightly increased
            "eating_distance": (6.0, 8.0),      # Reduced range
        }
        self.param_order = list(self.param_bounds.keys())

        # Evolutionary parameters
        self.generations = 30
        self.population_size = 20
        self.elite_size = 3
        self.mutation_rate = 0.15
        self.tournament_size = 5
        self.n_simulations = 3  # Number of simulations per individual
        
        # CMA-ES parameters
        self.cma_popsize = 10
        self.cma_sigma = 0.25

    def evaluate_individual(self, genome):
        """Evaluate an individual by running multiple simulations"""
        fitnesses = []
        for _ in range(self.n_simulations):
            fitness = self._evaluate_fitness_once(genome)
            fitnesses.append(fitness)
        return np.mean(fitnesses)

    def create_next_generation(self, population, fitnesses):
        """Create next generation using tournament selection and mutation"""
        new_population = []
        
        # Keep elites
        elite_indices = np.argsort(fitnesses)[:self.elite_size]
        new_population.extend([population[i] for i in elite_indices])
        
        # Tournament selection
        while len(new_population) < self.population_size:
            # Select parents through tournament
            tournament_indices = np.random.choice(
                len(population), 
                size=self.tournament_size, 
                replace=False
            )
            tournament_fitness = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parent = population[winner_idx]
            
            # Create offspring with mutation
            offspring = self.mutate(parent.copy())
            new_population.append(offspring)
        
        return new_population

    def mutate(self, genome):
        """Apply mutation to a genome"""
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                lower, upper = self.param_bounds[self.param_order[i]]
                # Gaussian mutation centered on current value
                genome[i] = np.clip(
                    genome[i] + random.gauss(0, (upper - lower) * 0.1),
                    lower,
                    upper
                )
        return genome

    def run_hybrid_optimization(self):
        # Initialize population using CMA-ES
        lower = np.array([self.param_bounds[k][0] for k in self.param_order])
        upper = np.array([self.param_bounds[k][1] for k in self.param_order])
        x0 = (lower + upper) / 2
        
        # CMA-ES for initial population
        es = cma.CMAEvolutionStrategy(
            x0, 
            self.cma_sigma, 
            {'popsize': self.cma_popsize, 'bounds': [lower, upper]}
        )
        
        # Initial population
        population = es.ask()
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Evaluate all individuals in parallel
            with Pool() as pool:
                fitnesses = list(tqdm(
                    pool.imap(self.evaluate_individual, population),
                    total=len(population),
                    desc=f"Generation {gen+1}"
                ))
            
            # Find best individual
            best_idx = np.argmin(fitnesses)
            best_genome = population[best_idx]
            best_fitness = fitnesses[best_idx]
            best_fitness_history.append(best_fitness)
            
            print(f"\nGeneration {gen+1}: Best fitness = {-best_fitness:.4f}")
            
            # Create next generation
            population = self.create_next_generation(population, fitnesses)
            
            if gen % 5 == 0:  # Every 5 generations
                cma_solutions = es.ask()  # ask for new candidates
                with Pool() as pool:
                    cma_fitnesses = list(tqdm(
                        pool.imap(self.evaluate_individual, cma_solutions),
                        total=len(cma_solutions),
                        desc=f"CMA-ES Injection {gen+1}"
                    ))
                es.tell(cma_solutions, cma_fitnesses)  # ✅ Correct: same length
                
                # Replace worst individuals with CMA-ES solutions
                worst_indices = np.argsort(fitnesses)[-len(cma_solutions):]
                for i, idx in enumerate(worst_indices):
                    population[idx] = cma_solutions[i]
                    fitnesses[idx] = cma_fitnesses[i]  # Optional: keep fitness array updated

            
            # Early stopping if no improvement
            if gen > 10 and (best_fitness_history[-10] - best_fitness_history[-1]) < 0.01:
                print("Early stopping - no improvement in last 10 generations")
                break
        
        # Final evaluation of best genome
        final_fitness = self.evaluate_individual(best_genome)
        print(f"\nFinal best fitness: {-final_fitness:.4f}")
        
        # Return best parameters
        best_params = dict(zip(self.param_order, best_genome))
        return best_params

    def _evaluate_fitness_once(self, genome):
        try:
            reset_global_state()
            params = dict(zip(self.param_order, genome))

            # Stability check - reject obviously bad parameters
            if (params["prey_energy_gain"] <= params["prey_energy_loss"] * 40 or
                params["predator_energy_gain"] <= params["predator_energy_loss"] * 25):
                return 1.0  # CMA minimizes, so higher is worse

            @dataclass
            class EvolvedConfig:
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
                predator_speed: float = max(params["predator_speed"], params["prey_speed"] + 0.3)
                predation_distance: float = max(params["predation_distance"], params["eating_distance"] + 2.0)
                eating_distance: float = params["eating_distance"]
                initial_prey: int = 80
                initial_predators: int = 20
                window_size: tuple[int, int] = (1000, 1000)
                delta_time: float = 1.0
                duration: int = 20000
                seed: int = random.randint(0, 999999)
                window: Window = field(default_factory=lambda: Window(1000, 1000))
                radius: int = 30
                movement_speed: float = 1.0
                image_rotation: bool = True

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
            
            if len(df) < 50:  # Require minimum simulation length
                return 1.0

            # Improved fitness calculation
            duration = len(df) / cfg.duration
            
            # Survival bonus (both must survive)
            survival = float(df["prey_count"].iloc[-1] > 0 and df["predator_count"].iloc[-1] > 0)
            
            # Stability metrics
            def normalized_fft(series):
                norm = (series - series.mean()) / (series.std() + 1e-8)
                fft = np.abs(np.fft.rfft(norm))
                return fft[:10]  # Only consider low frequencies
            
            prey_fft = normalized_fft(df["prey_count"])
            pred_fft = normalized_fft(df["predator_count"])
            
            # Interaction metric
            cross_corr = np.correlate(
                (df["prey_count"] - df["prey_count"].mean()).values,
                (df["predator_count"] - df["predator_count"].mean()).values,
                mode='same'
            )
            interaction = np.max(cross_corr) / (df["prey_count"].std() * df["predator_count"].std() + 1e-8)
            
            # Calculate fitness components
            fitness = (
                0.3 * duration + 
                0.3 * survival +
                0.2 * (np.mean(prey_fft[1:3]) + np.mean(pred_fft[1:3])) / 2 +  # Low freq oscillations
                0.2 * np.clip(interaction, 0, 1)
            )
            
            # Penalize extreme population ratios
            max_ratio = max(
                df["prey_count"].max() / (df["predator_count"].mean() + 1e-5),
                df["predator_count"].max() / (df["prey_count"].mean() + 1e-5)
            )
            if max_ratio > 10:
                fitness *= 0.5
            
            return -fitness  # CMA minimizes

        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            traceback.print_exc()
            return 1.0

    def run(self):
        # Initial parameter values (midpoint of bounds)
        lower = np.array([self.param_bounds[k][0] for k in self.param_order])
        upper = np.array([self.param_bounds[k][1] for k in self.param_order])
        x0 = (lower + upper) / 2
        
        # CMA-ES options
        opts = cma.CMAOptions()
        opts.set("bounds", [lower, upper])
        opts.set("popsize", 20)
        opts.set("maxiter", 50)
        opts.set("verbose", -1)  # Reduce output
        
        # Run optimization
        es = cma.CMAEvolutionStrategy(x0, 0.25, opts)
        
        with tqdm(total=opts['maxiter'], desc="CMA-ES Optimization") as pbar:
            while not es.stop():
                solutions = es.ask()
                fitnesses = self.evaluate_fitness(solutions)
                es.tell(solutions, fitnesses)
                es.disp()
                pbar.update(1)
                
                # Save current best
                best_idx = np.argmin(fitnesses)
                best_genome = solutions[best_idx]
                best_fitness = -fitnesses[best_idx]
                pbar.set_postfix({"best": f"{best_fitness:.4f}"})

        # Final results
        best_genome = es.result.xbest
        best_fitness = -es.result.fbest
        best_params = dict(zip(self.param_order, best_genome))

        print("\n=== Best Parameters ===")
        for k, v in best_params.items():
            print(f"{k}: {v:.4f}")
        print(f"Best fitness: {best_fitness:.4f}")
        
        return best_params
    
def run_final(best_params):
    """Run final simulation with best parameters"""
    reset_global_state()
    
    @dataclass
    class FinalConfig:
        prey_energy_gain: float = best_params["prey_energy_gain"]
        prey_energy_loss: float = best_params["prey_energy_loss"]
        prey_initial_energy: float = 50.0
        predator_energy_gain: float = best_params["predator_energy_gain"]
        predator_energy_loss: float = best_params["predator_energy_loss"]
        predator_initial_energy: float = 50.0
        prey_growth_rate: float = best_params["prey_growth_rate"]
        prey_carrying_capacity: float = best_params["prey_carrying_capacity"]
        prey_reproduce_cost: float = best_params["prey_reproduce_cost"]
        predator_reproduce_prob: float = best_params["predator_reproduce_prob"]
        predator_reproduce_threshold: float = best_params["predator_reproduce_threshold"]
        predator_reproduce_cost: float = best_params["predator_reproduce_cost"]
        food_spawn_prob: float = best_params["food_spawn_prob"]
        food_spawn_amount: int = 2
        initial_food: int = 100
        prey_speed: float = best_params["prey_speed"]
        predator_speed: float = max(best_params["predator_speed"], best_params["prey_speed"] + 0.3)
        predation_distance: float = max(best_params["predation_distance"], best_params["eating_distance"] + 2.0)
        eating_distance: float = best_params["eating_distance"]
        initial_prey: int = 80
        initial_predators: int = 20
        window_size: tuple[int, int] = (1000, 1000)
        delta_time: float = 1.0
        duration: int = 20000
        seed: int = random.randint(0, 999999)
        window: Window = field(default_factory=lambda: Window(1000, 1000))
        radius: int = 30
        movement_speed: float = 1.0
        image_rotation: bool = True

    cfg = FinalConfig()
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
    
    # Save data to CSV
    df.to_csv('comparative_cmaes.csv', index=False)
    print("Data saved to comparative_cmaes.csv")
    
    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['frame'], df['prey_count'], label='Prey Population', color='green')
    plt.plot(df['frame'], df['predator_count'], label='Predator Population', color='red')
    plt.plot(df['frame'], df['food_count'], label='Food Count', color='blue', linestyle='--')
    plt.title('Comparative/Hybrid Evolution CMA-ES Lotka-Volterra Model')
    plt.xlabel('Time (frames)')
    plt.ylabel('Population Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparative_cmaes.png')
    plt.close()
    print("Plot saved to comparative_cmaes.png")

if __name__ == "__main__":
    optimizer = HybridEvoCMAESLotka()
    best_params = optimizer.run_hybrid_optimization()
    run_final(best_params)