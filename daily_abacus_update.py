#!/usr/bin/env python3
"""
Daily abacus portfolio update system.

Automatically detects active trading model from holdings files, updates stock
prices only if needed, and generates web content for HTML dashboard display.
Uses centralized JSON configuration for all model switching scripts.
"""

import os
import sys

# Set environment variables before any other imports
os.environ['MPLBACKEND'] = 'Agg'  # Set matplotlib backend via environment
os.environ['MPLCONFIGDIR'] = '/tmp'  # Prevent matplotlib config directory issues

# Redirect stderr temporarily to suppress early matplotlib messages
import io
import contextlib

# Create context manager to suppress stderr during imports
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Suppress matplotlib backend selection messages during import
with suppress_stderr():
    import matplotlib
    matplotlib.use('Agg', force=True)  # Force backend before pyplot import
    import matplotlib.pyplot as plt

# Now safe to import other modules
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Configure matplotlib logging after import
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)

# Suppress verbose numba debug logging
logging.getLogger('numba').setLevel(logging.WARNING)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*Using non-interactive Agg backend.*")
warnings.filterwarnings("ignore", message=".*no display found.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

from functions.GetParams import get_json_params
from run_pytaaa import run_pytaaa


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration with matplotlib suppression."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress noisy third-party loggers regardless of verbose setting
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)  # Pillow image library
    
    return logging.getLogger(__name__)


def suppress_matplotlib_output():
    """Suppress matplotlib backend selection messages."""
    # Redirect matplotlib's backend selection messages
    import matplotlib.backends.backend_agg
    matplotlib.backends.backend_agg.FigureCanvasAgg._get_output_dpi = lambda self, dpi: dpi or 100

    # Set matplotlib to quiet mode
    plt.ioff()  # Turn off interactive mode
    
    # Suppress specific matplotlib messages
    original_print = print
    def quiet_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        if 'no display found' in message.lower() or 'using non-interactive agg backend' in message.lower():
            return  # Suppress these specific messages
        original_print(*args, **kwargs)
    
    # Temporarily replace print during matplotlib operations
    import builtins
    builtins.__original_print__ = original_print  # Save for numba compatibility
    builtins.print = quiet_print


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load and validate JSON configuration file."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def validate_config_structure(config: Dict[str, Any]) -> None:
    """Ensure all required configuration parameters exist."""
    logger = logging.getLogger(__name__)
    
    required_sections = [
        'models',
        'Valuation',
        'web_output_dir'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate models section
    if 'model_choices' not in config['models']:
        raise ValueError("Missing 'model_choices' in models configuration")
    
    if 'base_folder' not in config['models']:
        raise ValueError("Missing 'base_folder' in models configuration")
    
    # Validate Valuation section
    required_valuation = ['performance_store']
    for item in required_valuation:
        if item not in config['Valuation']:
            raise ValueError(f"Missing '{item}' in Valuation configuration")
    
    # Validate model_selection section for performance metric weights
    if 'model_selection' not in config:
        raise ValueError("Missing 'model_selection' section in JSON configuration")
    
    if 'performance_metrics' not in config['model_selection']:
        raise ValueError("Missing 'performance_metrics' section in model_selection configuration")
    
    # Validate required performance metric weights
    required_weights = [
        'sharpe_ratio_weight',
        'sortino_ratio_weight',
        'max_drawdown_weight',
        'avg_drawdown_weight',
        'annualized_return_weight'
    ]
    
    performance_metrics = config['model_selection']['performance_metrics']
    missing_weights = [w for w in required_weights if w not in performance_metrics]
    
    if missing_weights:
        raise ValueError(f"Missing required performance metric weights in JSON configuration: {', '.join(missing_weights)}")
    
    # Validate metric blending configuration
    if 'metric_blending' not in config:
        raise ValueError("Missing 'metric_blending' section in JSON configuration")
    
    metric_blending = config['metric_blending']
    required_blending_params = ['enabled', 'full_period_weight', 'focus_period_weight']
    missing_blending = [p for p in required_blending_params if p not in metric_blending]
    
    if missing_blending:
        raise ValueError(f"Missing required metric blending parameters in JSON configuration: {', '.join(missing_blending)}")
    
    logger.debug("Configuration structure validation passed")


def detect_active_trading_model(config: Dict[str, Any]) -> Optional[str]:
    """
    Detect active trading model from abacus holdings file.
    
    Reads the 'trading_model:' line from PyTAAA_holdings.params file
    in the abacus data store.
    """
    logger = logging.getLogger(__name__)
    
    performance_store = config['Valuation']['performance_store']
    holdings_file = os.path.join(performance_store, 'PyTAAA_holdings.params')
    
    if not os.path.exists(holdings_file):
        logger.warning(f"Holdings file not found: {holdings_file}")
        logger.warning("Cannot detect active trading model")
        return None
    
    try:
        with open(holdings_file, 'r') as f:
            lines = f.readlines()
        
        # Look for trading_model line (usually at the end)
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('trading_model:'):
                trading_model = line.split(':', 1)[1].strip()
                logger.info(f"Detected active trading model: {trading_model}")
                return trading_model
        
        logger.warning("No 'trading_model:' line found in holdings file")
        return None
        
    except Exception as e:
        logger.error(f"Error reading holdings file: {e}")
        return None


def resolve_template_path(template_path: str, base_folder: str, 
                         data_file: str) -> str:
    """
    Resolve template-based path with {base_folder} and {data_file} 
    placeholders.
    """
    return template_path.format(
        base_folder=base_folder,
        data_file=data_file
    )


def get_data_source_from_trading_model(trading_model: str) -> Dict[str, str]:
    """
    Map trading model to data source paths.
    
    Returns dictionary with symbols_file and data_folder paths
    based on the trading model.
    """
    logger = logging.getLogger(__name__)
    
    # Define mapping from trading model to data source
    model_to_source = {
        'naz100_pine': 'naz100',
        'naz100_hma': 'naz100', 
        'naz100_pi': 'naz100',
        'sp500_hma': 'sp500',
        'sp500_pine': 'sp500',
        'cash': 'abacus'  # Default to abacus for cash
    }
    
    data_source = model_to_source.get(trading_model, 'abacus')
    logger.info(f"Trading model '{trading_model}' maps to data source: {data_source}")
    
    # Define paths for each data source
    if data_source == 'naz100':
        return {
            'symbols_file': '/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_Symbols.txt',
            'data_folder': '/Users/donaldpg/pyTAAA_data/Naz100',
            'company_names_file': '/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_companyNames.txt',
            'stockList': 'Naz100'  # This is the key parameter for TAfunctions.py
        }
    elif data_source == 'sp500':
        return {
            'symbols_file': '/Users/donaldpg/pyTAAA_data/SP500/symbols/SP500_Symbols.txt',
            'data_folder': '/Users/donaldpg/pyTAAA_data/SP500',
            'company_names_file': '/Users/donaldpg/pyTAAA_data/SP500/symbols/SP500_companyNames.txt',
            'stockList': 'SP500'  # This is the key parameter for TAfunctions.py
        }
    else:  # abacus default
        return {
            'symbols_file': '/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/symbols/abacus_symbols.txt',
            'data_folder': '/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus',
            'company_names_file': None,  # No company names file for combined abacus
            'stockList': 'Naz100'  # Default to Naz100 for abacus fallback
        }


def update_config_with_data_source(config: Dict[str, Any], 
                                  trading_model: str) -> Dict[str, Any]:
    """
    Update configuration to point to correct data source based on trading model.
    
    This dynamically switches between naz100, sp500, or abacus data sources
    based on the active trading model detected from holdings file.
    
    IMPORTANT: This only updates the data source routing (symbols_file, stockList).
    The performance_store remains pointing to the abacus data store so the abacus
    portfolio maintains its own holdings, status, and configuration.
    """
    logger = logging.getLogger(__name__)
    
    # Get data source paths for the trading model
    data_source_paths = get_data_source_from_trading_model(trading_model)
    
    # Create a copy of config to avoid modifying original
    updated_config = config.copy()
    
    # Update the Valuation section with correct symbols file and stockList
    # BUT keep performance_store pointing to abacus data store
    if 'Valuation' in updated_config:
        updated_config['Valuation'] = updated_config['Valuation'].copy()
        updated_config['Valuation']['symbols_file'] = data_source_paths['symbols_file']
        updated_config['Valuation']['stockList'] = data_source_paths['stockList']
        
        # DO NOT change performance_store - it should always point to abacus
        # The abacus portfolio maintains its own holdings and configuration
        logger.info(f"Updated symbols_file to: {data_source_paths['symbols_file']}")
        logger.info(f"Updated stockList to: {data_source_paths['stockList']}")
        logger.info(f"Keeping performance_store as: {updated_config['Valuation']['performance_store']}")
    
    # Update data_paths section if it exists
    if 'data_paths' in updated_config:
        updated_config['data_paths'] = updated_config['data_paths'].copy()
        updated_config['data_paths']['filename'] = data_source_paths['symbols_file']
        
        logger.info(f"Updated data_paths filename to: {data_source_paths['symbols_file']}")
    
    return updated_config


def update_config_with_active_model(config: Dict[str, Any], 
                                   active_model: str) -> Dict[str, Any]:
    """
    Update configuration with active model data paths.
    
    This now includes both model-specific data validation AND
    data source routing based on the trading model.
    
    IMPORTANT: This loads the active model's Valuation section to get
    the correct uptrendSignalMethod and other model-specific parameters.
    """
    logger = logging.getLogger(__name__)
    
    if not active_model:
        logger.warning("No active model detected, using default configuration")
        return config
    
    # First, update data source routing based on trading model
    config = update_config_with_data_source(config, active_model)
    
    model_choices = config['models']['model_choices']
    base_folder = config['models']['base_folder']
    
    if active_model not in model_choices:
        logger.error(f"Active model '{active_model}' not found in model_choices")
        raise ValueError(f"Unknown trading model: {active_model}")
    
    # Handle cash model (empty path)
    if active_model == 'cash' or model_choices[active_model] == "":
        logger.info("Active model is 'cash' - using default configuration")
        return config
    
    # Resolve template path for active model
    template_path = model_choices[active_model]
    data_file = config['monte_carlo']['data_files']['actual']
    
    resolved_path = resolve_template_path(template_path, base_folder, data_file)
    logger.info(f"Resolved active model path: {resolved_path}")
    
    # Verify the resolved path exists
    if not os.path.exists(resolved_path):
        logger.error(f"Active model data file not found: {resolved_path}")
        raise FileNotFoundError(f"Active model data file not found: {resolved_path}")
    
    # Load the active model's JSON configuration to get its Valuation section
    # This ensures we use the correct uptrendSignalMethod and other parameters
    model_data_store = os.path.dirname(resolved_path)
    model_json_files = [
        os.path.join(model_data_store, f"pytaaa_{active_model}.json"),
        os.path.join(os.path.dirname(model_data_store), f"pytaaa_{active_model}.json"),
        os.path.join(base_folder, active_model, f"pytaaa_{active_model}.json")
    ]
    
    model_config_loaded = False
    for model_json_path in model_json_files:
        if os.path.exists(model_json_path):
            try:
                with open(model_json_path, 'r') as f:
                    model_config = json.load(f)
                
                # Copy the Valuation section from the model's config
                if 'Valuation' in model_config:
                    # Preserve critical abacus paths BEFORE updating
                    abacus_performance_store = config['Valuation']['performance_store']
                    
                    # Get webpage - check multiple sources with priority order
                    abacus_webpage = config['Valuation'].get('webpage')
                    if not abacus_webpage and 'web_output_dir' in config:
                        abacus_webpage = config['web_output_dir']
                    
                    logger.info(f"BEFORE update - abacus_webpage: {abacus_webpage}")
                    logger.info(f"BEFORE update - performance_store: {abacus_performance_store}")
                    
                    # Update with model's Valuation section
                    config['Valuation'].update(model_config['Valuation'])
                    
                    # Restore abacus-specific paths (these should ALWAYS be preserved)
                    config['Valuation']['performance_store'] = abacus_performance_store
                    
                    # CRITICAL: Always ensure webpage is set
                    if abacus_webpage:
                        config['Valuation']['webpage'] = abacus_webpage
                        logger.info(f"AFTER update - Preserved webpage directory: {abacus_webpage}")
                    elif 'web_output_dir' in config:
                        # Fallback to web_output_dir if webpage was never set
                        config['Valuation']['webpage'] = config['web_output_dir']
                        logger.info(f"AFTER update - Set webpage from web_output_dir: {config['web_output_dir']}")
                    else:
                        logger.error("No webpage directory available - this will cause errors!")
                        raise ValueError("webpage directory not configured in JSON")
                    
                    logger.info(f"AFTER update - performance_store: {config['Valuation']['performance_store']}")
                    logger.info(f"Loaded Valuation section from {model_json_path}")
                    logger.info(f"Using uptrendSignalMethod: {config['Valuation'].get('uptrendSignalMethod')}")
                    model_config_loaded = True
                    break
            except Exception as e:
                logger.warning(f"Could not load model config from {model_json_path}: {e}")
    
    if not model_config_loaded:
        logger.warning(f"Could not find JSON config for model {active_model}, using abacus Valuation section")
    
    logger.debug(f"Configuration updated for active model: {active_model}")
    return config


def ensure_required_files(config: Dict[str, Any]) -> None:
    """
    Create initial PyTAAA files if they don't exist.
    
    Creates basic structure for:
    - PyTAAA_status.params
    - PyTAAA_holdings.params
    - PyTAAA_ranks.params
    """
    logger = logging.getLogger(__name__)
    
    performance_store = config['Valuation']['performance_store']
    
    # Ensure performance store directory exists
    os.makedirs(performance_store, exist_ok=True)
    logger.debug(f"Ensured directory exists: {performance_store}")
    
    # Required files with default content
    required_files = {
        'PyTAAA_status.params': '10000.0',  # Default portfolio value
        'PyTAAA_holdings.params': '''stocks\tshares\tbuyprice\tranks
CASH\t10000\t1.00\t1
trading_model: cash''',
        'PyTAAA_ranks.params': 'CASH\t1'
    }
    
    for filename, default_content in required_files.items():
        filepath = os.path.join(performance_store, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(default_content)
            logger.info(f"Created initial file: {filename}")
        else:
            logger.debug(f"File already exists: {filename}")


def check_if_update_needed(config: Dict[str, Any], 
                          active_model: str) -> bool:
    """
    Check if stock price update is needed using existing PyTAAA market logic.
    
    Returns True if:
    - No active model detected (always update abacus)
    - Active model is 'cash' (always update abacus)  
    - Market is open and quotes haven't been updated today
    
    Returns False if:
    - Market is closed (weekends/holidays/after hours)
    - Market is open but quotes are already current
    
    This uses the existing PyTAAA market checking logic that properly
    handles weekends, holidays, and trading hours.
    """
    logger = logging.getLogger(__name__)
    
    # Always update if no active model or cash model
    if not active_model or active_model == 'cash':
        logger.info("Update needed - no active model or cash model")
        return True
    
    # Use the existing PyTAAA market checking logic
    try:
        from functions.CheckMarketOpen import get_MarketOpenOrClosed
        
        # get_MarketOpenOrClosed() returns strings like:
        # " Markets are closed" for weekends/holidays/after hours
        # " Markets are open" or "close in X hours" during trading hours
        market_status = get_MarketOpenOrClosed()
        logger.debug(f"Market status: '{market_status}'")
        
        # If market is closed, no update needed
        if "closed" in market_status.lower():
            logger.info(f"Update not needed - {market_status.strip()}")
            return False
        
        # If market is open, check if quotes are current
        logger.info(f"Market is open - {market_status.strip()}")
        
        # Get the HDF5 file for the active model's data source
        data_source_paths = get_data_source_from_trading_model(active_model)
        symbols_file = data_source_paths['symbols_file']
        symbol_directory = os.path.dirname(symbols_file)
        symbols_filename = os.path.basename(symbols_file)
        
        # Determine HDF5 filename based on symbols file
        if "Naz100" in symbols_filename:
            hdf5_filename = os.path.join(symbol_directory, "Naz100_Symbols_.hdf5")
        elif "SP500" in symbols_filename:
            hdf5_filename = os.path.join(symbol_directory, "SP500_Symbols_.hdf5")
        else:
            # Look for any HDF5 file in the directory
            import glob
            hdf5_files = glob.glob(os.path.join(symbol_directory, "*.hdf5"))
            if hdf5_files:
                hdf5_filename = hdf5_files[0]
            else:
                logger.info("Update needed - no HDF5 file found")
                return True
        
        if not os.path.exists(hdf5_filename):
            logger.info(f"Update needed - HDF5 file not found: {hdf5_filename}")
            return True
        
        # Check if HDF5 file was modified today
        from datetime import datetime, date
        file_mtime = datetime.fromtimestamp(os.path.getmtime(hdf5_filename))
        file_date = file_mtime.date()
        today = date.today()
        
        if file_date < today:
            logger.info(f"Update needed - HDF5 file last modified {file_date}, today is {today}")
            return True
        else:
            logger.info(f"Update not needed - HDF5 file is current (modified: {file_date})")
            return False
            
    except Exception as e:
        logger.warning(f"Could not check market status or quote freshness: {e}")
        logger.info("Update needed - unable to determine market status")
        return True


def generate_web_content(config: Dict[str, Any]) -> None:
    """
    Generate web content for HTML dashboard.
    
    The run_pytaaa function handles web page generation through
    writeWebPage function. This ensures web output goes to the
    configured web_output_dir.
    """
    logger = logging.getLogger(__name__)
    
    web_output_dir = config.get('web_output_dir')
    if web_output_dir:
        os.makedirs(web_output_dir, exist_ok=True)
        logger.info(f"Web content will be generated in: {web_output_dir}")
    else:
        logger.warning("No web_output_dir configured")


def create_temporary_config_file(config: Dict[str, Any]) -> str:
    """
    Create a temporary JSON configuration file with updated paths.
    
    This allows us to pass the dynamically updated configuration
    to run_pytaaa() which expects a file path.
    """
    logger = logging.getLogger(__name__)
    
    # Create temporary file in same directory as script
    import tempfile
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='daily_abacus_temp_')
    
    try:
        # Write updated config to temporary file
        with os.fdopen(temp_fd, 'w') as temp_file:
            json.dump(config, temp_file, indent=4)
        
        logger.debug(f"Created temporary config file: {temp_path}")
        return temp_path
        
    except Exception as e:
        # Clean up on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


def cleanup_temporary_file(temp_path: str) -> None:
    """Clean up temporary configuration file."""
    logger = logging.getLogger(__name__)
    
    try:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary file {temp_path}: {e}")


def generate_web_content_only(config: Dict[str, Any]) -> None:
    """
    Generate web content without running full PyTAAA update.
    
    This runs only the web generation parts of PyTAAA without
    updating stock quotes, for when quotes are current but 
    web pages need refreshing.
    """
    logger = logging.getLogger(__name__)
    
    try:
        from functions.WriteWebPage_pi import writeWebPage
        from functions.GetParams import get_holdings, get_status, get_webpage_store
        from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
        
        # Create temporary config file for web generation
        temp_config_file = create_temporary_config_file(config)
        logger.info("generate_web_content_only: Created temporary config file")
        
        try:
            # Ensure web output directory exists
            web_output_dir = config.get('web_output_dir')
            if web_output_dir:
                os.makedirs(web_output_dir, exist_ok=True)
                print(f"Web output directory ensured: {web_output_dir}")
            
            # Get current portfolio data for web page
            holdings = get_holdings(temp_config_file)
            status_value = get_status(temp_config_file)
            logger.info("generate_web_content_only: Created temporary config file")
            logger.info("generate_web_content_only: Retrieved holdings and status value")
            
            # Convert status_value to float if it's a string
            if isinstance(status_value, str):
                try:
                    status_value = float(status_value)
                except (ValueError, TypeError):
                    status_value = 10000.0  # Default fallback value
            
            # Generate basic web content with current data
            regulartext = f"<p>Portfolio value: ${status_value:,.2f}</p>"
            boldtext = f"Current portfolio status as of {datetime.now().strftime('%Y-%m-%d')}"
            headlinetext = "Daily Abacus Portfolio Update - Web Refresh"
            lastdate = datetime.now().strftime('%Y-%m-%d')
            print(f"{headlinetext}\nfor date: {lastdate}")
            logger.info("generate_web_content_only: Initialized web content variables")
            
            # Extract portfolio data from holdings
            last_symbols_text = []
            last_symbols_weight = []  
            last_symbols_price = []
            
            if holdings and len(holdings) > 0:
                for holding in holdings:
                    if len(holding) >= 4:  # Ensure we have enough data
                        symbol = holding[0]
                        shares_str = holding[1]
                        price_str = holding[2]
                        
                        # Safely convert strings to floats
                        try:
                            shares = float(shares_str) if shares_str else 0.0
                        except (ValueError, TypeError):
                            shares = 0.0
                        
                        try:
                            price = float(price_str) if price_str else 0.0
                        except (ValueError, TypeError):
                            price = 0.0
                        
                        last_symbols_text.append(symbol)
                        last_symbols_weight.append(shares * price / status_value if status_value > 0 else 0.0)
                        last_symbols_price.append(price)
            
            print("Generating HTML dashboard and PNG plots...")
            
            # Generate web page with current data - this creates all HTML and PNG files
            writeWebPage(
                regulartext, boldtext, headlinetext, lastdate,
                last_symbols_text, last_symbols_weight, last_symbols_price,
                temp_config_file
            )
            print("Returned from writeWebPage...")
            
            # Verify files were created
            verify_web_files_created(temp_config_file)
            
            logger.info("Web content generated successfully")
            
        finally:
            cleanup_temporary_file(temp_config_file)
            
    except Exception as e:
        logger.error(f"Error generating web content: {e}")
        raise e


def verify_web_files_created(json_fn: str) -> None:
    """
    Verify that expected web files were created and report status.
    """
    logger = logging.getLogger(__name__)
    
    try:
        from functions.GetParams import get_webpage_store
        webpage_dir = get_webpage_store(json_fn)
        
        # Expected files
        expected_files = [
            "pyTAAAweb.html",
            "PyTAAA_stock-chart-blue.png", 
            "PyTAAA_backtest.png",
            "PyTAAA_backtest_updated.png"
        ]
        
        # Check which files exist
        created_files = []
        missing_files = []
        
        for filename in expected_files:
            filepath = os.path.join(webpage_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                created_files.append(f"{filename} ({file_size} bytes)")
            else:
                missing_files.append(filename)
        
        # Report results
        print(f"\n=== Web File Generation Summary ===")
        print(f"Web directory: {webpage_dir}")
        print(f"Files created: {len(created_files)}")
        
        for file_info in created_files:
            print(f"  ✅ {file_info}")
        
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for filename in missing_files:
                print(f"  ❌ {filename}")
        
        # Check for any PNG files that were generated by plot functions
        if os.path.exists(webpage_dir):
            all_files = os.listdir(webpage_dir)
            png_files = [f for f in all_files if f.endswith('.png')]
            
            if png_files:
                print(f"Generated PNG files: {len(png_files)}")
                for png_file in sorted(png_files):
                    filepath = os.path.join(webpage_dir, png_file)
                    file_size = os.path.getsize(filepath)
                    print(f"  📊 {png_file} ({file_size} bytes)")
        
        print("=== End Summary ===\n")
        
    except Exception as e:
        logger.warning(f"Could not verify web files: {e}")


def main() -> None:
    """Main entry point for daily abacus update."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Daily abacus portfolio update with active model detection'
    )
    parser.add_argument(
        '--json', 
        required=True,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging with matplotlib suppression
    logger = setup_logging(args.verbose)
    
    # Suppress matplotlib output at startup
    suppress_matplotlib_output()
    
    temp_config_file = None
    
    try:
        print("=== Daily Abacus Portfolio Update ===")
        print(f"Configuration file: {args.json}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load and validate configuration
        logger.info("Loading configuration file...")
        config = load_config_file(args.json)
        
        logger.info("Validating configuration structure...")
        validate_config_structure(config)
        
        # Ensure required files exist
        logger.info("Ensuring required files exist...")
        ensure_required_files(config)
        
        # Detect active trading model
        logger.info("Detecting active trading model...")
        active_model = detect_active_trading_model(config)
        
        if active_model:
            print(f"Active trading model: {active_model}")
        else:
            print("No active trading model detected - using abacus defaults")
        
        # Update configuration with active model data
        logger.info("Updating configuration with active model...")
        updated_config = update_config_with_active_model(config, active_model)
        
        # Show the data source being used
        data_source_paths = get_data_source_from_trading_model(active_model or 'cash')
        print(f"Using data source: {data_source_paths['symbols_file']}")
        
        # Prepare web content generation directory
        generate_web_content(config)
        
        # Check if stock quote update is needed
        logger.info("Checking if stock price update is needed...")
        update_needed = check_if_update_needed(config, active_model)
        

        # print("Stock price update needed - proceeding with full update")
        
        # Create temporary config file with updated paths
        logger.info("Creating temporary configuration file...")
        temp_config_file = create_temporary_config_file(updated_config)
        
        # Run main PyTAAA update process with updated config
        logger.info("Running PyTAAA update process...")
        print("\n--- PyTAAA Update Process ---")
        
        run_pytaaa(temp_config_file)
        
        print("--- PyTAAA Update Complete ---\n")
        
        # Update abacus backtest portfolio values in PyTAAA_status.params
        logger.info("Updating abacus backtest portfolio values...")
        print("\n--- Updating Abacus Backtest Values ---")
        
        try:
            from functions.abacus_backtest import write_abacus_backtest_portfolio_values
            
            # Use saved lookbacks or defaults from config
            success = write_abacus_backtest_portfolio_values(
                json_config_path=args.json,
                lookbacks=None  # Will auto-detect from saved state or config
            )
            
            if success:
                print("✓ Abacus backtest portfolio values updated successfully")
                logger.info("Abacus backtest portfolio values updated")
            else:
                print("⚠ Abacus backtest update skipped or failed")
                logger.warning("Abacus backtest update was not successful")
                
        except Exception as e:
            logger.error(f"Failed to update abacus backtest values: {e}")
            print(f"ERROR: Failed to update abacus backtest values: {e}")
        
        print("--- Abacus Backtest Update Complete ---\n")

        # if update_needed:
        #     print("Stock price update needed - proceeding with full update")
            
        #     # Create temporary config file with updated paths
        #     logger.info("Creating temporary configuration file...")
        #     temp_config_file = create_temporary_config_file(updated_config)
            
        #     # Run main PyTAAA update process with updated config
        #     logger.info("Running PyTAAA update process...")
        #     print("\n--- PyTAAA Update Process ---")
            
        #     run_pytaaa(temp_config_file)
            
        #     print("--- PyTAAA Update Complete ---\n")
            
        # else:
        #     print("Stock prices are current - updating web pages only")
        #     logger.info("Skipping stock update - updating web pages only")
            
        #     # Generate web content even when stock quotes don't need updating
        #     logger.info("Generating web content...")
        #     print("\n--- Web Page Generation ---")
            
        #     # Suppress matplotlib messages during web generation
        #     suppress_matplotlib_output()
        #     generate_web_content_only(updated_config)
            
        #     print("--- Web Page Generation Complete ---\n")
        
        print("=== Daily Abacus Update Complete ===")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        print(f"ERROR: Daily update failed: {e}")
        sys.exit(1)
        
    finally:
        # Clean up temporary file
        if temp_config_file:
            cleanup_temporary_file(temp_config_file)


if __name__ == '__main__':
    main()