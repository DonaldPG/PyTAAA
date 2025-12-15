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
#   ./run_monte_carlo.sh 5 --fp-duration=7   # Run 5 times with 7-year focus periods
#   ./run_monte_carlo.sh 3 --fp-year-min=2000 --fp-year-max=2020  # Custom year range
#############################################################################

# Function to display usage information
show_usage() {
    echo "Usage: $0 <number_of_runs> [search_strategy] [options]"
    echo ""
    echo "Arguments:"
    echo "  number_of_runs    Number of Monte Carlo runs to execute (required)"
    echo "  search_strategy   Search strategy: explore, exploit, or explore-exploit (optional)"
    echo ""
    echo "Options:"
    echo "  --verbose              Show detailed normalized score breakdown (optional)"
    echo "  --reset                Reset saved state after each run (optional)"
    echo "  --json=<path>          Specify JSON configuration file path (optional)"
    echo "  --randomize            Use randomized normalization values (optional)"
    echo "  --fp-duration=<years>  Focus period duration in years (default: 5)"
    echo "  --fp-year-min=<year>   Minimum year for focus period start (default: 1995)"
    echo "  --fp-year-max=<year>   Maximum year for focus period start (default: 2021)"
    echo ""
    echo "Examples:"
    echo "  $0 5                       # Run 5 times with default strategy"
    echo "  $0 10 explore             # Run 10 times with exploration strategy"
    echo "  $0 3 exploit --verbose    # Run 3 times with exploitation and verbose output"
    echo "  $0 7 explore-exploit      # Run 7 times with dynamic strategy"
    echo "  $0 2 --verbose            # Run 2 times with default strategy and verbose output"
    echo "  $0 5 --reset              # Run 5 times, resetting state after each run"
    echo "  $0 3 explore --reset      # Run 3 times with exploration, resetting after each"
    echo "  $0 4 --json=config.json   # Run 4 times with JSON configuration file"
    echo "  $0 2 --randomize          # Run 2 times with randomized normalization values"
    echo "  $0 3 exploit --randomize  # Run 3 times with exploitation and randomization"
    echo "  $0 5 --fp-duration=7      # Run 5 times with 7-year focus periods"
    echo "  $0 3 --fp-year-min=2000 --fp-year-max=2020  # Custom year range"
    echo ""
    echo "Note: Each run builds upon the previous state for continuous optimization."
    echo "      Use --reset to start fresh exploration from each run."
    echo "      Focus period parameters override JSON config values for the run."
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Validate number of arguments - Allow more arguments for all combinations
if [[ $# -lt 1 ]]; then
    echo "Error: Invalid number of arguments"
    echo ""
    show_usage
    exit 1
fi

# Parse arguments to handle all flags in any position
NUM_RUNS=""
SEARCH_STRATEGY=""
VERBOSE_FLAG=""
RESET_FLAG=""
JSON_FLAG=""
RANDOMIZE_FLAG=""
FP_DURATION=""
FP_YEAR_MIN=""
FP_YEAR_MAX=""

for arg in "$@"; do
    case "$arg" in
        --verbose)
            VERBOSE_FLAG="--verbose"
            ;;
        --reset)
            RESET_FLAG="--reset"
            ;;
        --json=*)
            JSON_FLAG="$arg"  # Preserve the full --json=path format
            ;;
        --json)
            # Handle --json as separate argument (next arg should be path)
            JSON_FLAG="--json"
            ;;
        --randomize)
            RANDOMIZE_FLAG="--randomize"
            ;;
        --fp-duration=*)
            FP_DURATION="${arg#*=}"
            ;;
        --fp-year-min=*)
            FP_YEAR_MIN="${arg#*=}"
            ;;
        --fp-year-max=*)
            FP_YEAR_MAX="${arg#*=}"
            ;;
        explore|exploit|explore-exploit)
            SEARCH_STRATEGY="$arg"
            ;;
        *)
            # Check if this might be a JSON path following --json
            if [[ "$JSON_FLAG" == "--json" && -z "$NUM_RUNS" ]]; then
                JSON_FLAG="--json $arg"
            elif [[ -z "$NUM_RUNS" ]]; then
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
echo "Reset state after each run: ${RESET_FLAG:-disabled}"
echo "JSON configuration: ${JSON_FLAG:-none}"
echo "Randomize normalization: ${RANDOMIZE_FLAG:-disabled}"
echo "Focus period duration: ${FP_DURATION:-default (5 years)}"
echo "Focus period year range: ${FP_YEAR_MIN:-1995} to ${FP_YEAR_MAX:-2021}"
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
    
    # Build command with optional search strategy, verbose flag, JSON flag, and randomize flag
    CMD="uv run python run_monte_carlo.py"
    
    if [[ -n "$SEARCH_STRATEGY" ]]; then
        CMD="$CMD --search $SEARCH_STRATEGY"
    fi
    
    if [[ -n "$VERBOSE_FLAG" ]]; then
        CMD="$CMD $VERBOSE_FLAG"
    fi

    if [[ -n "$JSON_FLAG" ]]; then
        CMD="$CMD $JSON_FLAG"
    fi

    if [[ -n "$RANDOMIZE_FLAG" ]]; then
        CMD="$CMD $RANDOMIZE_FLAG"
    fi
    
    # Add focus period parameters if specified
    if [[ -n "$FP_DURATION" ]]; then
        CMD="$CMD --fp-duration=$FP_DURATION"
    fi
    
    if [[ -n "$FP_YEAR_MIN" ]]; then
        CMD="$CMD --fp-year-min=$FP_YEAR_MIN"
    fi
    
    if [[ -n "$FP_YEAR_MAX" ]]; then
        CMD="$CMD --fp-year-max=$FP_YEAR_MAX"
    fi
    
    echo "Executing: $CMD"
    echo ""
    
    # Execute the Monte Carlo run
    if $CMD; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo "✓ Run $i completed successfully"
        
        # Reset saved state if --reset flag is provided
        if [[ -n "$RESET_FLAG" ]]; then
            echo "Resetting saved state for next run..."
            if uv run python modify_saved_state.py reset --no-confirm --no-backup; then
                echo "✓ State reset completed"
            else
                echo "⚠ State reset failed (continuing anyway)"
            fi
        fi
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