#!/usr/bin/env python3
"""
Enhanced hyperparameter sweep with plotting for learning rate sensitivity.
"""

import subprocess
import itertools
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from pathlib import Path

def parse_tensorboard_logs(runs_dir, algorithm, learning_rate):
    """Parse TensorBoard logs for a specific algorithm and learning rate."""
    # Look for runs with the pattern: {algorithm}_lr{learning_rate}_seed{seed}
    pattern = os.path.join(runs_dir, f"*{algorithm}_lr{learning_rate}*")
    run_dirs = glob.glob(pattern)
    
    if not run_dirs:
        print(f"No {algorithm} runs found for lr={learning_rate} in {runs_dir}")
        return { 
            'episodic_returns': [], 
            'timesteps': [],
            'actual_performance': [],
            'performance_timesteps': []
        }

    print(f"Found {len(run_dirs)} {algorithm} runs for lr={learning_rate}")

    all_episodic_returns = []
    all_timesteps = []
    all_actual_performance = []
    all_performance_timesteps = []
    
    for run_dir in run_dirs:
        event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
        if not event_files:
            continue
        event_file = max(event_files, key=os.path.getctime)
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            ea = EventAccumulator(event_file)
            ea.Reload()

            # Extract charts/episodic_return for this run
            if 'charts/episodic_return' in ea.Tags()['scalars']:
                events = ea.Scalars('charts/episodic_return')
                run_returns = []
                run_steps = []
                for event in events:
                    run_returns.append(float(event.value))
                    run_steps.append(int(event.step))
                
                all_episodic_returns.append(run_returns)
                all_timesteps.append(run_steps)
            
            # Extract charts/actual_performance for this run
            if 'charts/actual_performance' in ea.Tags()['scalars']:
                events = ea.Scalars('charts/actual_performance')
                run_performance = []
                run_perf_steps = []
                for event in events:
                    run_performance.append(float(event.value))
                    run_perf_steps.append(int(event.step))
                
                all_actual_performance.append(run_performance)
                all_performance_timesteps.append(run_perf_steps)
                
        except Exception as e:
            print(f"Error parsing {event_file}: {e}")
            continue

    return { 
        'episodic_returns': all_episodic_returns, 
        'timesteps': all_timesteps,
        'actual_performance': all_actual_performance,
        'performance_timesteps': all_performance_timesteps
    }

def parse_final_performance_from_tensorboard(runs_dir, algorithm, learning_rate):
    """Parse final actual performance from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Look for runs with the pattern: {algorithm}_lr{learning_rate}_seed{seed}
        pattern = os.path.join(runs_dir, f"*{algorithm}_lr{learning_rate}*")
        run_dirs = glob.glob(pattern)
        
        if not run_dirs:
            return None
        
        # Get the most recent run directory
        latest_run_dir = max(run_dirs, key=os.path.getctime)
        
        # Find TensorBoard event files
        event_files = glob.glob(os.path.join(latest_run_dir, "events.out.tfevents.*"))
        if not event_files:
            return None
        
        # Use the most recent event file
        event_file = max(event_files, key=os.path.getctime)
        
        # Parse TensorBoard logs
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Look for actual performance (this is what we want for final performance)
        if 'charts/actual_performance' in ea.Tags()['scalars']:
            events = ea.Scalars('charts/actual_performance')
            if events:
                return float(events[-1].value)  # Get the last (final) actual performance value
        
        # Fallback: look for final mean episodic return if actual performance not available
        elif 'final/mean_episodic_return' in ea.Tags()['scalars']:
            events = ea.Scalars('final/mean_episodic_return')
            if events:
                return float(events[-1].value)
        
        # Fallback: look for regular episodic return and get the last value
        elif 'charts/episodic_return' in ea.Tags()['scalars']:
            events = ea.Scalars('charts/episodic_return')
            if events:
                return float(events[-1].value)  # Get the last value
        
    except Exception as e:
        print(f"Error parsing TensorBoard logs: {e}")
        pass
    
    return None

def aggregate_episodic_returns(data, algorithm_name, learning_rate):
    """Aggregate episodic returns across runs and create time-binned statistics."""
    if not data or not data['episodic_returns']:
        return None
    
    all_episodic_returns = data['episodic_returns']
    all_timesteps = data['timesteps']
    
    # Find the maximum timestep across all runs
    max_timestep = 0
    for run_steps in all_timesteps:
        if run_steps:
            max_timestep = max(max_timestep, max(run_steps))
    
    if max_timestep == 0:
        return None
    
    # Create time bins for aggregation
    num_bins = 50
    bin_edges = np.linspace(0, max_timestep, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Aggregate data across all runs for each time bin
    binned_returns = [[] for _ in range(num_bins)]
    
    for run_returns, run_steps in zip(all_episodic_returns, all_timesteps):
        for return_val, step in zip(run_returns, run_steps):
            bin_idx = min(int(step / max_timestep * num_bins), num_bins - 1)
            binned_returns[bin_idx].append(return_val)
    
    # Calculate statistics for each bin
    means = []
    stds = []
    counts = []
    
    for bin_data in binned_returns:
        if bin_data:
            means.append(np.mean(bin_data))
            stds.append(np.std(bin_data))
            counts.append(len(bin_data))
        else:
            means.append(0)
            stds.append(0)
            counts.append(0)
    
    return {
        'timesteps': bin_centers,
        'means': np.array(means),
        'stds': np.array(stds),
        'counts': np.array(counts),
        'algorithm': algorithm_name,
        'learning_rate': learning_rate
    }

def aggregate_actual_performance(data, algorithm_name, learning_rate):
    """Aggregate actual performance across runs and create time-binned statistics."""
    if not data or not data['actual_performance']:
        return None
    
    all_performance = data['actual_performance']
    all_timesteps = data['performance_timesteps']
    
    # Find the maximum timestep across all runs
    max_timestep = 0
    for run_steps in all_timesteps:
        if run_steps:
            max_timestep = max(max_timestep, max(run_steps))
    
    if max_timestep == 0:
        return None
    
    # Create time bins for aggregation
    num_bins = 50
    bin_edges = np.linspace(0, max_timestep, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Aggregate data across all runs for each time bin
    binned_performance = [[] for _ in range(num_bins)]
    
    for run_performance, run_steps in zip(all_performance, all_timesteps):
        for perf_val, step in zip(run_performance, run_steps):
            bin_idx = min(int(step / max_timestep * num_bins), num_bins - 1)
            binned_performance[bin_idx].append(perf_val)
    
    # Calculate statistics for each bin
    means = []
    stds = []
    counts = []
    
    for bin_data in binned_performance:
        if bin_data:
            means.append(np.mean(bin_data))
            stds.append(np.std(bin_data))
            counts.append(len(bin_data))
        else:
            means.append(0)
            stds.append(0)
            counts.append(0)
    
    return {
        'timesteps': bin_centers,
        'means': np.array(means),
        'stds': np.array(stds),
        'counts': np.array(counts),
        'algorithm': algorithm_name,
        'learning_rate': learning_rate
    }

def create_episodic_return_plot_by_lr(algorithm_data, algorithm, output_dir):
    """Create plot for mean episodic returns over time for different learning rates."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))
    
    for i, lr in enumerate(learning_rates):
        if lr in algorithm_data and algorithm_data[lr]:
            aggregated = aggregate_episodic_returns(algorithm_data[lr], algorithm, lr)
            if aggregated:
                timesteps = aggregated['timesteps']
                means = aggregated['means']
                stds = aggregated['stds']
                counts = aggregated['counts']
                
                # Filter out empty bins
                valid_mask = counts > 0
                timesteps = timesteps[valid_mask]
                means = means[valid_mask]
                stds = stds[valid_mask]
                counts = counts[valid_mask]
                
                if len(means) > 0:
                    # Calculate 95% confidence intervals
                    sem = stds / np.sqrt(counts)
                    ci_lower = means - 1.96 * sem
                    ci_upper = means + 1.96 * sem
                    
                    ax.plot(timesteps, means, color=colors[i], label=f'LR={lr:.0e}', linewidth=2)
                    ax.fill_between(timesteps, ci_lower, ci_upper, color=colors[i], alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episodic Return')
    ax.set_title(f'{algorithm.upper()} - Episodic Return vs Learning Rate\n(95% confidence intervals)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{algorithm}_episodic_return_by_lr.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved episodic return plot: {plot_path}")

def create_actual_performance_plot_by_lr(algorithm_data, algorithm, output_dir):
    """Create plot for mean actual performance over time for different learning rates."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))
    
    for i, lr in enumerate(learning_rates):
        if lr in algorithm_data and algorithm_data[lr]:
            aggregated = aggregate_actual_performance(algorithm_data[lr], algorithm, lr)
            if aggregated:
                timesteps = aggregated['timesteps']
                means = aggregated['means']
                stds = aggregated['stds']
                counts = aggregated['counts']
                
                # Filter out empty bins
                valid_mask = counts > 0
                timesteps = timesteps[valid_mask]
                means = means[valid_mask]
                stds = stds[valid_mask]
                counts = counts[valid_mask]
                
                if len(means) > 0:
                    # Calculate 95% confidence intervals
                    sem = stds / np.sqrt(counts)
                    ci_lower = means - 1.96 * sem
                    ci_upper = means + 1.96 * sem
                    
                    ax.plot(timesteps, means, color=colors[i], label=f'LR={lr:.0e}', linewidth=2)
                    ax.fill_between(timesteps, ci_lower, ci_upper, color=colors[i], alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Actual Performance')
    ax.set_title(f'{algorithm.upper()} - Actual Performance vs Learning Rate\n(95% confidence intervals)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{algorithm}_actual_performance_by_lr.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved actual performance plot: {plot_path}")

def create_hyperparameter_plots(results, output_dir):
    """Create hyperparameter sensitivity plots."""
    
    for algo in algorithms:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        lrs = []
        mean_perfs = []
        std_perfs = []
        
        for lr in learning_rates:
            if results[algo][lr]:  # If we have data for this learning rate
                perfs = results[algo][lr]
                lrs.append(lr)
                mean_perfs.append(np.mean(perfs))
                std_perfs.append(np.std(perfs))
        
        if lrs:  # If we have data
            # Plot 1: Mean performance vs Learning Rate
            ax1.errorbar(lrs, mean_perfs, yerr=std_perfs, marker='o', capsize=5, 
                        capthick=2, linewidth=2, markersize=8)
            ax1.set_xlabel("Learning Rate", fontsize=12)
            ax1.set_ylabel("Mean Episodic Return", fontsize=12)
            ax1.set_title(f"{algo.upper()} - Performance vs Learning Rate", fontweight='bold')
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plot for each learning rate
            data_for_box = [results[algo][lr] for lr in lrs if results[algo][lr]]
            ax2.boxplot(data_for_box, labels=[f"{lr:.0e}" for lr in lrs if results[algo][lr]])
            ax2.set_xlabel("Learning Rate", fontsize=12)
            ax2.set_ylabel("Episodic Return", fontsize=12)
            ax2.set_title(f"{algo.upper()} - Performance Distribution", fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Find best learning rate
            best_idx = np.argmax(mean_perfs)
            best_lr = lrs[best_idx]
            best_perf = mean_perfs[best_idx]
            
            print(f"\n🎯 Best learning rate for {algo.upper()}: {best_lr:.0e} (performance: {best_perf:.2f})")
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{algo}_hyperparameter_sensitivity.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {plot_path}")
            plt.show()
    
    # Save results summary
    summary_path = os.path.join(output_dir, "hyperparameter_results.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved: {summary_path}")

# Define parameter sweep
algorithms = ["dqn", "ppo"]
learning_rates = [1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2]
seeds = [1, 2, 3, 4] 

env_id = "PointMassDiscrete-v0"
total_timesteps = 100000  # Adjust as needed

# Make experiment folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"hparam_sweep_{timestamp}"
os.makedirs(base_dir, exist_ok=True)

# Store results for plotting
results = {algo: {lr: [] for lr in learning_rates} for algo in algorithms}
tensorboard_data = {algo: {lr: None for lr in learning_rates} for algo in algorithms}

print(f"🚀 Running hyperparameter sweep for {algorithms}")
print(f"Learning rates: {learning_rates}")
print(f"Seeds per combination: {seeds}")

for algo, lr, seed in itertools.product(algorithms, learning_rates, seeds):
    run_name = f"{algo}_lr{lr}_seed{seed}"
    print(f"\nRunning {algo.upper()} with lr={lr}, seed={seed}...")

    # Build command with custom run name that includes learning rate
    run_name_with_lr = f"{algo}_lr{lr}_seed{seed}"
    cmd = [
        ".venv/bin/python", f"cleanrl/{algo}.py",
        "--env-id", env_id,
        "--learning-rate", str(lr),
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--run-name", run_name_with_lr,  # This will create runs with lr in the name
    ]

    # Run experiment
    log_file = os.path.join(base_dir, f"{run_name}.log")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        # Write output to log file
        with open(log_file, 'w') as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        if result.returncode == 0:
            # Parse final performance from TensorBoard logs
            final_performance = parse_final_performance_from_tensorboard("runs", algo, lr)
            if final_performance is not None:
                results[algo][lr].append(final_performance)
                print(f"  ✅ Final performance: {final_performance:.2f}")
            else:
                print(f"  ⚠️  Could not parse performance from TensorBoard")
                # Debug: show available run directories
                pattern = f"*{algo}_lr{lr}*"
                run_dirs = glob.glob(os.path.join("runs", pattern))
                print(f"  🔍 Available runs: {run_dirs}")
        else:
            print(f"  ❌ Failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout")

# Parse TensorBoard logs for time-series analysis
print(f"\n📊 Parsing TensorBoard logs for time-series analysis...")
runs_dir = "runs"  # Default TensorBoard logs directory

for algo in algorithms:
    for lr in learning_rates:
        print(f"Parsing {algo.upper()} logs for lr={lr:.0e}...")
        tensorboard_data[algo][lr] = parse_tensorboard_logs(runs_dir, algo, lr)

# Create hyperparameter sensitivity plots
create_hyperparameter_plots(results, base_dir)

# Create time-series plots for each algorithm
for algo in algorithms:
    print(f"\n📈 Creating time-series plots for {algo.upper()}...")
    create_episodic_return_plot_by_lr(tensorboard_data[algo], algo, base_dir)
    create_actual_performance_plot_by_lr(tensorboard_data[algo], algo, base_dir)

print(f"\n✅ Hyperparameter sweep completed!")
print(f"📊 Check {base_dir} for plots and results")