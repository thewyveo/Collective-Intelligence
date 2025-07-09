import pandas as pd

# Load the results CSV
df = pd.read_csv("all_50_runs_energy.csv")

# Group by run ID and get the max frame (final frame reached)
final_frames = df.groupby("run")["frame"].max().reset_index()
final_frames.columns = ["run", "final_frame"]

# Save to new CSV
final_frames.to_csv("final_frames_per_run.csv", index=False)

print("Saved final frame counts to final_frames_per_run.csv")