#!/bin/bash

#############################################################################
# Monte Carlo Loop Runner Script
#
# This script runs the Monte Carlo backtesting system multiple times in a 
# loop. Each run will continue to build upon the previous state, allowing
# for extended parameter exploration and optimization.
#
# Usage: ./run_monte_carlo.sh <number_of_runs> [search_strategy]
#
# Examples:
#   ./run_monte_carlo.sh 5                    # Run 5 times with default strategy
#   ./run_monte_carlo.sh 10 explore          # Run 10 times with exploration
#   ./run_monte_carlo.sh 3 exploit           # Run 3 times with exploitation
#   ./run_monte_carlo.sh 7 explore-exploit   # Run 7 times with dynamic strategy
#############################################################################

# Function to display usage information
show_usage() {
    echo "Usage: $0 <number_of_runs> [search_strategy] [--verbose]"
    echo ""
    echo "Arguments:"
    echo "  number_of_runs    Number of Monte Carlo runs to execute (required)"
    echo "  search_strategy   Search strategy: explore, exploit, or explore-exploit (optional)"
    echo "  --verbose         Show detailed normalized score breakdown (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 5                       # Run 5 times with default strategy"
    echo "  $0 10 explore             # Run 10 times with exploration strategy"
    echo "  $0 3 exploit --verbose    # Run 3 times with exploitation and verbose output"
    echo "  $0 7 explore-exploit      # Run 7 times with dynamic strategy"
    echo "  $0 2 --verbose            # Run 2 times with default strategy and verbose output"
    echo ""
    echo "Note: Each run builds upon the previous state for continuous optimization."
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Validate number of arguments
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Error: Invalid number of arguments"
    echo ""
    show_usage
    exit 1
fi

# Parse arguments to handle --verbose flag in any position
NUM_RUNS=""
SEARCH_STRATEGY=""
VERBOSE_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --verbose)
            VERBOSE_FLAG="--verbose"
            ;;
        explore|exploit|explore-exploit)
            SEARCH_STRATEGY="$arg"
            ;;
        *)
            if [[ -z "$NUM_RUNS" ]]; then
                NUM_RUNS="$arg"
            else
                echo "Error: Unknown argument '$arg'"
                show_usage
                exit 1
            fi
            ;;
    esac
done

# Validate that NUM_RUNS was provided
if [[ -z "$NUM_RUNS" ]]; then
    echo "Error: Number of runs is required"
    show_usage
    exit 1
fi

# Validate that NUM_RUNS is a positive integer
if ! [[ "$NUM_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of runs must be a positive integer"
    echo "Provided: '$NUM_RUNS'"
    exit 1
fi

# Validate search strategy if provided
if [[ -n "$SEARCH_STRATEGY" ]]; then
    case "$SEARCH_STRATEGY" in
        explore|exploit|explore-exploit)
            echo "Using search strategy: $SEARCH_STRATEGY"
            ;;
        *)
            echo "Error: Invalid search strategy '$SEARCH_STRATEGY'"
            echo "Valid options: explore, exploit, explore-exploit"
            exit 1
            ;;
    esac
fi

# Set PYTHONPATH to current directory for proper module resolution
export PYTHONPATH=$(pwd)

# Display run configuration
echo "=============================================================="
echo "Monte Carlo Loop Runner"
echo "=============================================================="
echo "Number of runs: $NUM_RUNS"
echo "Search strategy: ${SEARCH_STRATEGY:-default (explore-exploit)}"
echo "Verbose mode: ${VERBOSE_FLAG:-disabled}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================================="
echo ""

# Initialize counters and timing
START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Main execution loop
for ((i=1; i<=NUM_RUNS; i++)); do
    echo ""
    echo "--------------------------------------------------------------"
    echo "Starting Monte Carlo Run $i of $NUM_RUNS"
    echo "Time: $(date)"
    echo "--------------------------------------------------------------"
    
    # Build command with optional search strategy and verbose flag
    CMD="uv run python run_monte_carlo.py"
    
    if [[ -n "$SEARCH_STRATEGY" ]]; then
        CMD="$CMD --search $SEARCH_STRATEGY"
    fi
    
    if [[ -n "$VERBOSE_FLAG" ]]; then
        CMD="$CMD $VERBOSE_FLAG"
    fi
    
    echo "Executing: $CMD"
    echo ""
    
    # Execute the Monte Carlo run
    if $CMD; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "✓ Run $i completed successfully"
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        echo ""
        echo "✗ Run $i failed with exit code $?"
        
        # Ask user if they want to continue on failure
        echo ""
        read -p "Continue with remaining runs? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping execution at user request"
            break
        fi
    fi
    
    # Show progress
    COMPLETED=$((SUCCESS_COUNT + FAILURE_COUNT))
    echo "Progress: $COMPLETED/$NUM_RUNS runs completed ($SUCCESS_COUNT successful, $FAILURE_COUNT failed)"
    
    # Add delay between runs to prevent system overload
    if [[ $i -lt $NUM_RUNS ]]; then
        echo "Waiting 5 seconds before next run..."
        sleep 5
    fi
done

# Calculate execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

# Display final summary
echo ""
echo "=============================================================="
echo "Monte Carlo Loop Runner - Final Summary"
echo "=============================================================="
echo "Total runs requested: $NUM_RUNS"
echo "Successful runs: $SUCCESS_COUNT"
echo "Failed runs: $FAILURE_COUNT"
echo "Total execution time: ${MINUTES}m ${SECONDS}s"
echo "End time: $(date)"

# Display per-run average if successful runs exist
if [[ $SUCCESS_COUNT -gt 0 ]]; then
    AVG_TIME_PER_RUN=$((TOTAL_TIME / SUCCESS_COUNT))
    AVG_MINUTES=$((AVG_TIME_PER_RUN / 60))
    AVG_SECONDS=$((AVG_TIME_PER_RUN % 60))
    echo "Average time per successful run: ${AVG_MINUTES}m ${AVG_SECONDS}s"
fi

echo "=============================================================="

# Exit with appropriate code
if [[ $FAILURE_COUNT -eq 0 ]]; then
    echo "✓ All runs completed successfully!"
    exit 0
else
    echo "⚠ Some runs failed. Check logs for details."
    exit 1
fi