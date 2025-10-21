#!/usr/bin/env python3
"""
Script to parse TensorBoard logs and create comparison plots for DQN vs PPO.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict
import json

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_tensorboard_logs(runs_dir, algorithm):
    """Parse TensorBoard logs and aggregate charts/episodic_return across all runs."""
    print(f"Parsing {algorithm} logs from {runs_dir}")

    pattern = os.path.join(runs_dir, f"*{algorithm}*")
    run_dirs = glob.glob(pattern)
    if not run_dirs:
        print(f"No {algorithm} runs found in {runs_dir}")
        return { 
            'episodic_returns': [], 
            'timesteps': [],
            'actual_performance': [],
            'performance_timesteps': []
        }

    print(f"Found {len(run_dirs)} {algorithm} runs")

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

def parse_stdout_logs(run_dir, algorithm):
    """Fallback stub. Not used when analyzing only final means."""
    return { 'final_means': [] }

def aggregate_actual_performance(data, algorithm_name):
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
        'algorithm': algorithm_name
    }

def aggregate_episodic_returns(data, algorithm_name):
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
        'algorithm': algorithm_name
    }

def create_episodic_return_plot(dqn_data, ppo_data, output_dir):
    """Create plot for mean episodic returns over time with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Process DQN data
    dqn_aggregated = aggregate_episodic_returns(dqn_data, 'DQN')
    if dqn_aggregated:
        timesteps = dqn_aggregated['timesteps']
        means = dqn_aggregated['means']
        stds = dqn_aggregated['stds']
        counts = dqn_aggregated['counts']
        
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
            
            ax.plot(timesteps, means, 'b-', label='DQN', linewidth=2)
            ax.fill_between(timesteps, ci_lower, ci_upper, color='blue', alpha=0.2)
    
    # Process PPO data
    ppo_aggregated = aggregate_episodic_returns(ppo_data, 'PPO')
    if ppo_aggregated:
        timesteps = ppo_aggregated['timesteps']
        means = ppo_aggregated['means']
        stds = ppo_aggregated['stds']
        counts = ppo_aggregated['counts']
        
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
            
            ax.plot(timesteps, means, 'r-', label='PPO', linewidth=2)
            ax.fill_between(timesteps, ci_lower, ci_upper, color='red', alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episodic Return')
    ax.set_title('Mean Episodic Return Over Time\n(10 runs with 95% confidence intervals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episodic_returns_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'episodic_returns_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print("Episodic return plot saved")

def create_final_performance_comparison(dqn_data, ppo_data, output_dir):
    """Create box plot comparing final performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect final episodic returns from all runs
    dqn_final_returns = []
    ppo_final_returns = []
    
    if dqn_data and dqn_data['episodic_returns']:
        for run_returns in dqn_data['episodic_returns']:
            if run_returns:
                dqn_final_returns.append(run_returns[-1])  # Last return from each run
    
    if ppo_data and ppo_data['episodic_returns']:
        for run_returns in ppo_data['episodic_returns']:
            if run_returns:
                ppo_final_returns.append(run_returns[-1])  # Last return from each run
    
    # Create box plot
    data_to_plot = []
    labels = []
    
    if dqn_final_returns:
        data_to_plot.append(dqn_final_returns)
        labels.append('DQN')
    
    if ppo_final_returns:
        data_to_plot.append(ppo_final_returns)
        labels.append('PPO')
    
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_ylabel('Final Episodic Return')
        ax.set_title('Final Performance Comparison\n(Box plot shows median, quartiles, and outliers)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'final_performance_comparison.pdf'), bbox_inches='tight')
        plt.close()
        
        print("Final performance comparison plot saved")
    else:
        print("No final returns available to plot")

def create_actual_performance_plot(dqn_data, ppo_data, output_dir):
    """Create plot for mean actual performance over time with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Process DQN data
    dqn_aggregated = aggregate_actual_performance(dqn_data, 'DQN')
    if dqn_aggregated:
        timesteps = dqn_aggregated['timesteps']
        means = dqn_aggregated['means']
        stds = dqn_aggregated['stds']
        counts = dqn_aggregated['counts']
        
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
            
            ax.plot(timesteps, means, 'b-', label='DQN', linewidth=2)
            ax.fill_between(timesteps, ci_lower, ci_upper, color='blue', alpha=0.2)
    
    # Process PPO data
    ppo_aggregated = aggregate_actual_performance(ppo_data, 'PPO')
    if ppo_aggregated:
        timesteps = ppo_aggregated['timesteps']
        means = ppo_aggregated['means']
        stds = ppo_aggregated['stds']
        counts = ppo_aggregated['counts']
        
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
            
            ax.plot(timesteps, means, 'r-', label='PPO', linewidth=2)
            ax.fill_between(timesteps, ci_lower, ci_upper, color='red', alpha=0.2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Actual Performance')
    ax.set_title('Mean Actual Performance Over Time\n(10 runs with 95% confidence intervals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'actual_performance_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print("Actual performance plot saved")

def main():
    parser = argparse.ArgumentParser(description='Create comparison plots from TensorBoard logs')
    parser.add_argument('--runs-dir', type=str, default='runs',
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--plots-dir', type=str, default='comparison_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("Parsing TensorBoard logs...")
    
    # Parse DQN data
    dqn_data = parse_tensorboard_logs(args.runs_dir, "dqn")
    
    # Parse PPO data
    ppo_data = parse_tensorboard_logs(args.runs_dir, "ppo")
    
    if not dqn_data and not ppo_data:
        print("No data found to create plots")
        return
    
    print("Creating comparison plots...")
    
    # Create episodic return plot with confidence intervals
    create_episodic_return_plot(dqn_data, ppo_data, args.plots_dir)
    
    # Create actual performance plot with confidence intervals
    create_actual_performance_plot(dqn_data, ppo_data, args.plots_dir)
    
    # Create final performance comparison
    
    print(f"All plots saved to {args.plots_dir}")

if __name__ == "__main__":
    main()
