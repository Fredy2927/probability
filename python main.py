import argparse
import json
import sys
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Any, Optional

# --- RICH IMPORTS ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
except ImportError:
    print("Error: This script requires the 'rich' library.")
    print("Please run: pip install rich")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. Market Data Model
# -----------------------------------------------------------------------------

@dataclass
class MarketParameters:
    S0: float       # Current Price
    K: float        # Target Price
    T: float        # Time (years)
    r: float        # Risk-free rate (decimal)
    mu: float       # Expected Return (decimal)
    sigma: float    # Volatility (decimal)
    q: float        # Dividend yield (decimal)
    mode: str       # 'RISK_NEUTRAL' or 'REAL_WORLD'

    def get_drift(self) -> float:
        if self.mode == "RISK_NEUTRAL":
            return self.r - self.q
        else:
            return self.mu - self.q

# -----------------------------------------------------------------------------
# 2. Math Engines
# -----------------------------------------------------------------------------

def calculate_analytical_probs(params: MarketParameters) -> Dict[str, float]:
    if params.T <= 0: return {'finish_above': 0.0, 'touch_anytime': 0.0}
    
    drift_rate = params.get_drift()
    mu_adj = drift_rate - 0.5 * params.sigma**2
    
    ln_K_S0 = np.log(params.K / params.S0)
    sqrt_T = np.sqrt(params.T)
    
    # Finish Above (Digital)
    d2 = (-ln_K_S0 + mu_adj * params.T) / (params.sigma * sqrt_T)
    prob_finish = stats.norm.cdf(d2)

    # Touch Anytime (Barrier)
    alpha = mu_adj
    d1_touch = (-ln_K_S0 + alpha * params.T) / (params.sigma * sqrt_T)
    d2_touch = (-ln_K_S0 - alpha * params.T) / (params.sigma * sqrt_T)
    
    try:
        term2 = np.exp((2 * alpha / params.sigma**2) * ln_K_S0) * stats.norm.cdf(d2_touch)
    except OverflowError:
        term2 = 0.0

    prob_touch = stats.norm.cdf(d1_touch) + term2
    
    return {
        'finish_above': min(max(prob_finish, 0.0), 1.0),
        'touch_anytime': min(max(prob_touch, 0.0), 1.0)
    }

def run_monte_carlo(params: MarketParameters, n_paths: int, steps_per_day: int) -> Dict[str, Any]:
    if params.T <= 0: return {'prob_finish': 0.0, 'prob_touch': 0.0}

    total_steps = int(max(10, params.T * 365.25 * steps_per_day))
    dt = params.T / total_steps
    sqrt_dt = np.sqrt(dt)
    
    drift_rate = params.get_drift()
    drift_step = (drift_rate - 0.5 * params.sigma**2) * dt
    vol_step = params.sigma * sqrt_dt

    dw = np.random.normal(0, 1, (total_steps, n_paths))
    log_returns = np.cumsum(drift_step + vol_step * dw, axis=0)
    
    target_log = np.log(params.K / params.S0)
    
    hits_finish = log_returns[-1, :] >= target_log
    hits_touch = np.max(log_returns, axis=0) >= target_log
    
    return {
        'prob_finish': np.mean(hits_finish),
        'prob_touch': np.mean(hits_touch)
    }

# -----------------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------------

def parse_date_str(date_str: str) -> date:
    """Parses dates like '09 Feb 2026' or '9 jan 2025'."""
    fmt = "%d %b %Y"
    try:
        # Title case handles 'feb' -> 'Feb' automatically
        return datetime.strptime(date_str.strip().title(), fmt).date()
    except ValueError:
        raise ValueError(f"Date format must be 'DD Mon YYYY' (e.g., '09 Feb 2026'). Got: {date_str}")

def calculate_time_years(start_str: Optional[str], expiry_str: str) -> float:
    expiry = parse_date_str(expiry_str)
    
    if start_str:
        start = parse_date_str(start_str)
    else:
        start = date.today()
        
    delta = (expiry - start).days
    return delta / 365.25 if delta > 0 else 0.0

def load_config(path):
    with open(path, 'r') as f: return json.load(f)

def create_template(path):
    template = {
        "stock_symbol": "NFLX",
        "current_price": 82.2,
        "target_price": 90.0,
        "start_date": "09 Feb 2026",
        "expiration_date": "13 Feb 2026",
        "model_mode": "REAL_WORLD", 
        "capm_settings": {"beta": 1.71, "market_return_percent": 10.0},
        "volatility_annualized_percent": 38.0,
        "risk_free_rate_percent": 4.0,
        "dividend_yield_percent": 0.0,
        "simulation_settings": {"n_paths": 100000, "n_steps_per_day": 7}
    }
    with open(path, 'w') as f: json.dump(template, f, indent=4)

# -----------------------------------------------------------------------------
# 4. Rich Display Logic
# -----------------------------------------------------------------------------

def display_dashboard(config, params, an_res, mc_res, T):
    console = Console()
    
    # 1. Header with Symbol and Time
    symbol_text = Text(f"{config.get('stock_symbol', 'UNK')} ANALYSIS", style="bold white on blue", justify="center")
    
    # 2. Scenario Table
    scenario_table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    scenario_table.add_column("Metric", style="cyan")
    scenario_table.add_column("Value", style="bold white")
    
    scenario_table.add_row("Current Price", f"${params.S0:,.2f}")
    scenario_table.add_row("Target Price", f"${params.K:,.2f}")
    scenario_table.add_row("Time Horizon", f"{T*365.25:.1f} days ({T:.4f} yrs)")
    scenario_table.add_row("Start Date", config.get('start_date', 'Today'))
    scenario_table.add_row("Expiry Date", config.get('expiration_date'))

    # 3. Model Inputs Table
    model_table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    model_table.add_column("Metric", style="magenta")
    model_table.add_column("Value", style="bold white")
    
    model_table.add_row("Volatility (IV)", f"{params.sigma:.2%}")
    model_table.add_row("CAPM Drift", f"{params.mu:.2%} / yr")
    model_table.add_row("Risk Free Rate", f"{params.r:.2%}")
    model_table.add_row("Simulated Paths", f"{config['simulation_settings']['n_paths']:,}")

    # 4. Results Table (The Main Event)
    results_table = Table(title="Probability: Finish > Target @ Expiry", box=box.ROUNDED)
    results_table.add_column("Method", justify="center", style="yellow")
    results_table.add_column("Probability", justify="center", style="bold green")

    results_table.add_row("Analytical (Exact)", f"{an_res['finish_above']:.2%}")
    results_table.add_row("Monte Carlo (Sim)", f"{mc_res['prob_finish']:.2%}")
    
    # 5. Barrier Table (Secondary Info)
    touch_table = Table(title="Probability: Touch Target (Anytime)", box=box.ROUNDED)
    touch_table.add_column("Method", justify="center", style="dim yellow")
    touch_table.add_column("Probability", justify="center", style="dim green")
    touch_table.add_row("Monte Carlo", f"{mc_res['prob_touch']:.2%}")

    # Assemble Layout
    console.print("\n")
    console.print(Panel(symbol_text, border_style="blue"))
    
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    
    grid.add_row(
        Panel(scenario_table, title="Scenario", border_style="cyan"),
        Panel(model_table, title="Model Assumptions", border_style="magenta")
    )
    console.print(grid)
    
    console.print(Panel(results_table, border_style="green", padding=(1, 2)))
    console.print(touch_table)
    console.print("\n")

# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.json")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        create_template(args.config)
        print("Config created. Edit percentages and dates (Format: 09 Feb 2026) and run again.")
        return

    try:
        # Conversion
        rf_dec = config['risk_free_rate_percent'] / 100.0
        vol_dec = config['volatility_annualized_percent'] / 100.0
        div_dec = config.get('dividend_yield_percent', 0.0) / 100.0
        
        capm_cfg = config.get('capm_settings', {})
        rm_dec = capm_cfg.get('market_return_percent', 8.0) / 100.0
        beta = capm_cfg.get('beta', 1.0)

        # CAPM & Params
        mu = rf_dec + beta * (rm_dec - rf_dec)
        start_str = config.get("start_date")
        T = calculate_time_years(start_str, config["expiration_date"])
        
        params = MarketParameters(
            S0=config['current_price'],
            K=config['target_price'],
            T=T, r=rf_dec, mu=mu, sigma=vol_dec, q=div_dec,
            mode=config.get("model_mode", "REAL_WORLD")
        )

        # Run Calculations
        an_res = calculate_analytical_probs(params)
        s = config.get('simulation_settings', {})
        mc_res = run_monte_carlo(params, s.get('n_paths', 100000), s.get('n_steps_per_day', 7))

        # Render Dashboard
        display_dashboard(config, params, an_res, mc_res, T)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()