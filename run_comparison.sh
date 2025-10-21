#!/bin/bash

# Complete comparison analysis for PointMass assignment
# Runs 10 independent experiments with paired random seeds for DQN and PPO

set -e  # Exit on any error

echo "=== PointMass Comparison Analysis ==="
echo "Running 10 independent experiments with paired random seeds..."

# Check if we're in the right directory
if [ ! -f "cleanrl/dqn.py" ]; then
    echo "Error: Please run this script from the CS839-Fall2025 directory"
    exit 1
fi

# Create directories
mkdir -p comparison_plots
mkdir -p runs

# Define seeds for paired experiments
# SEEDS=(1 2 3 4 5 6 7 8 9 10)
SEEDS=(1 2 3 4)

TOTAL_TIMESTEPS=500000

echo "Running experiments with seeds: ${SEEDS[@]}"
echo "Total timesteps per experiment: $TOTAL_TIMESTEPS"

# Function to run a single experiment
run_experiment() {
    local algorithm=$1
    local seed=$2
    local timesteps=$3
    
    echo ""
    echo "=== Running $algorithm with seed $seed ==="
    
    source .venv/bin/activate && python -m cleanrl.${algorithm} \
        --seed $seed \
        --env-id PointMassDiscrete-v0 \
        --total-timesteps $timesteps
    
    if [ $? -eq 0 ]; then
        echo "✓ $algorithm seed $seed completed successfully"
    else
        echo "✗ $algorithm seed $seed failed"
        exit 1
    fi
}

# Run DQN experiments
echo ""
echo "=== Running DQN Experiments ==="
for seed in "${SEEDS[@]}"; do
    run_experiment "dqn" $seed $TOTAL_TIMESTEPS
done

# Run PPO experiments  
echo ""
echo "=== Running PPO Experiments ==="
for seed in "${SEEDS[@]}"; do
    run_experiment "ppo" $seed $TOTAL_TIMESTEPS
done

echo ""
echo "=== All Experiments Completed ==="
echo "Creating comparison plots..."

# Parse tensorboard logs and create plots
source .venv/bin/activate && python create_plots.py \
    --runs-dir runs \
    --plots-dir comparison_plots

if [ $? -eq 0 ]; then
    echo "✓ Comparison plots created successfully"
else
    echo "✗ Failed to create comparison plots"
    exit 1
fi

echo ""
echo "=== Analysis Complete ==="
echo "Results saved to:"
echo "  - comparison_plots/episodic_returns_comparison.png (mean episodic return over time with 95% CI)"
echo "  - comparison_plots/actual_performance_comparison.png (mean actual performance over time with 95% CI)"
echo "  - comparison_plots/final_performance_comparison.png (final performance box plot)"
echo ""
echo "Tensorboard logs saved to:"
echo "  - runs/ directory"
echo ""
echo "You can view tensorboard logs with:"
echo "  tensorboard --logdir runs"
