import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configuration
CSV_FILE = "combined_population_data.csv"  # Your CSV file with 'Run' column
MAX_FRAME = 5_000                      # Maximum frame to show
ALPHA = 0.3                            # Line transparency
LINE_WIDTH = 0.8                        # Line width for plots

def plot_raw_simulations():
    # Load data and clean column names
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()  # Remove any whitespace
    
    # Filter by frame and group by run
    df = df.query("Frame < @MAX_FRAME")
    runs = df['Run'].unique()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    colors = {'Strain_R': 'tab:red', 'Strain_G': 'tab:green', 'Strain_B': 'tab:blue'}
    
    # Process each run (first 20 only)
    for run_id in runs[:20]:
        run_df = df[df['Run'] == run_id]
        
        # Plot raw data for each strain
        for strain in colors:
            plt.plot(run_df['Frame'], 
                    run_df[strain], 
                    color=colors[strain], 
                    alpha=ALPHA, 
                    lw=LINE_WIDTH)
    
    # Create legend
    legend_elements = [Line2D([0], [0], color=colors[col], lw=2, label=col.replace('_', ' '))
                      for col in colors]
    
    plt.xlabel("Time (frames)")
    plt.ylabel("Population count (raw)")
    plt.title(f"Bacterial Simulation: 20 Runs")
    plt.legend(handles=legend_elements)
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig("raw_population_trajectories.png", dpi=300)
    plt.close()

plot_raw_simulations()