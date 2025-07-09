import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess

# Configuration
CSV_FILE = "rps_50_runs_mean.csv"  # Your CSV file with 'run' column
MAX_FRAME = 5_000                     # Maximum frame to show
FRAC = 0.08                           # LOWESS smoothing fraction
SECONDS_PER_FRAME = 0.02              # Time per frame
ALPHA = 0.3                           # Line transparency

def plot_smoothed_simulations():
    # Load data and clean column names (strip whitespace)
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()  # Remove any whitespace from column names
    
    # Filter by frame and group by run
    df = df.query("frame < @MAX_FRAME")
    runs = df['run'].unique()
    
    plt.figure(figsize=(12, 6))
    colors = {'species_a': 'tab:green', 'species_b': 'tab:blue', 'species_c': 'tab:red'}
    
    # Process each run
    for run_id in runs[:20]:  # Limit to first 50 runs if needed
        run_df = df[df['run'] == run_id]
        
        # Smooth each species
        for species in colors:
            smoothed = lowess(run_df[species], run_df['frame'], frac=FRAC, return_sorted=False)
            plt.plot(run_df['frame'], smoothed, color=colors[species], alpha=ALPHA, lw=1)
    
    # Create legend
    legend_elements = [Line2D([0], [0], color=colors[col], lw=2, label=col.replace('_', ' ').title())
                      for col in colors]
    
    plt.xlabel("Time (frames)")
    plt.ylabel("Population count (smoothed)")
    plt.title(f"Smoothed population trajectories for {len(runs)} simulations")
    plt.legend(handles=legend_elements)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig("smoothed_population_trajectories.png", dpi=300)
    plt.close()

plot_smoothed_simulations()