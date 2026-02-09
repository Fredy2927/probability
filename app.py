import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import date

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Probability Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LIGHT THEME CSS ---
# Forces a clean white look regardless of system settings
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Metrics Box Styling (Light Grey Cards) */
    div[data-testid="metric-container"] {
        background-color: #F0F2F6;
        border: 1px solid #D1D5DB;
        padding: 15px;
        border-radius: 10px;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Metric Label (e.g. Time Horizon) */
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        color: #4B5563 !important; /* Dark Grey */
        font-weight: 500;
    }

    /* Metric Value (e.g. 4 Days) */
    div[data-testid="metric-container"] > div[data-testid="stMetricValue"] > div {
        color: #111827 !important; /* Almost Black */
        font-weight: 700;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MATH LOGIC
# -----------------------------------------------------------------------------

@dataclass
class MarketParameters:
    S0: float; K: float; T: float; r: float; mu: float; sigma: float; mode: str

    def get_drift(self) -> float:
        return self.r if self.mode == "RISK_NEUTRAL" else self.mu

def calculate_analytical(params: MarketParameters):
    if params.T <= 0: return 0.0, 0.0
    
    drift = params.get_drift()
    mu_adj = drift - 0.5 * params.sigma**2
    ln_K_S0 = np.log(params.K / params.S0)
    sqrt_T = np.sqrt(params.T)
    
    # 1. Finish Above
    d2 = (-ln_K_S0 + mu_adj * params.T) / (params.sigma * sqrt_T)
    prob_finish = stats.norm.cdf(d2)

    # 2. Touch Anytime
    alpha = mu_adj
    d1_touch = (-ln_K_S0 + alpha * params.T) / (params.sigma * sqrt_T)
    d2_touch = (-ln_K_S0 - alpha * params.T) / (params.sigma * sqrt_T)
    
    try:
        term2 = np.exp((2 * alpha / params.sigma**2) * ln_K_S0) * stats.norm.cdf(d2_touch)
        prob_touch = stats.norm.cdf(d1_touch) + term2
    except:
        prob_touch = 0.0
        
    return prob_finish, prob_touch

def run_simulation(params: MarketParameters, n_paths, steps_per_day=7):
    if params.T <= 0: return 0.0, 0.0, None

    total_steps = int(max(10, params.T * 365.25 * steps_per_day))
    dt = params.T / total_steps
    drift = params.get_drift()
    
    drift_step = (drift - 0.5 * params.sigma**2) * dt
    vol_step = params.sigma * np.sqrt(dt)

    dw = np.random.normal(0, 1, (total_steps, n_paths))
    log_returns = np.cumsum(drift_step + vol_step * dw, axis=0)
    
    # Add S0 to start of paths
    price_paths = params.S0 * np.exp(np.vstack([np.zeros((1, n_paths)), log_returns]))
    
    hits_finish = price_paths[-1, :] >= params.K
    hits_touch = np.max(price_paths, axis=0) >= params.K
    
    return np.mean(hits_finish), np.mean(hits_touch), price_paths

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    
    symbol = st.text_input("Stock Symbol", "NFLX")
    
    st.caption("PRICE & TARGET")
    c1, c2 = st.columns(2)
    S0 = c1.number_input("Current ($)", value=82.20, step=0.10)
    K = c2.number_input("Target ($)", value=90.00, step=0.10)
    
    st.caption("TIMELINE")
    d1 = st.date_input("Start Date", value=date(2026, 2, 9))
    d2 = st.date_input("Expiry Date", value=date(2026, 2, 13))
    
    st.caption("MODEL PARAMETERS")
    mode = st.selectbox("Drift Mode", ["REAL_WORLD (CAPM)", "RISK_NEUTRAL"])
    
    if "REAL_WORLD" in mode:
        beta = st.number_input("Beta", value=1.71, step=0.01)
        mkt_ret_pct = st.number_input("Market Return (%)", value=10.0, step=0.1)
        rf_pct = st.number_input("Risk Free Rate (%)", value=4.0, step=0.1)
        
        # Calculate CAPM
        mu_pct = rf_pct + beta * (mkt_ret_pct - rf_pct)
        st.success(f"📈 Expected Return: {mu_pct:.2f}%")
        mu_input = mu_pct / 100.0
    else:
        rf_pct = st.number_input("Risk Free Rate (%)", value=4.0, step=0.1)
        mu_input = 0.0

    iv_pct = st.number_input("Implied Volatility (%)", value=38.0, step=0.5)
    
    st.markdown("---")
    n_paths = st.slider("Sim Paths", 1000, 100000, 20000)
    
    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------

# Header
st.title(f"📊 Quant Analysis: {symbol}")

# Time Calc
days_diff = (d2 - d1).days
T = days_diff / 365.25

if days_diff <= 0:
    st.error("⚠️ Expiry must be after Start Date")
    st.stop()

# Setup Params
rf = rf_pct / 100.0
sigma = iv_pct / 100.0
calc_mode = "REAL_WORLD" if "REAL_WORLD" in mode else "RISK_NEUTRAL"
params = MarketParameters(S0, K, T, rf, mu_input, sigma, calc_mode)

# --- Top Metric Row ---
m1, m2, m3, m4 = st.columns(4)

m1.metric("Time Horizon", f"{days_diff} Days", f"{T:.4f} Years")
m2.metric("Distance to Target", f"${K - S0:.2f}", f"{(K/S0 - 1)*100:.2f}%")
m3.metric("Annualized Drift", f"{params.get_drift():.2%}", calc_mode)
m4.metric("Implied Volatility", f"{iv_pct:.1f}%", "Input")

st.markdown("---")

if run_btn:
    with st.spinner("Crunching numbers..."):
        # Run Math
        an_finish, an_touch = calculate_analytical(params)
        mc_finish, mc_touch, paths = run_simulation(params, n_paths)
        
        # Determine Colors based on probability
        def prob_color(p):
            if p > 0.50: return "#16a34a" # Green
            if p > 0.15: return "#d97706" # Orange
            return "#dc2626"              # Red

        # --- RESULTS SECTION ---
        col_res1, col_res2 = st.columns(2)
        
        # Result Card 1
        with col_res1:
            st.markdown(
                f"""
                <div style="background-color: #F9FAFB; padding: 20px; border-radius: 10px; border: 1px solid #E5E7EB; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #6B7280; margin:0; font-size: 18px;">Finish Above (Expiry)</h3>
                    <h1 style="color: {prob_color(mc_finish)}; font-size: 48px; margin: 10px 0;">{mc_finish:.2%}</h1>
                    <p style="color: #4B5563; margin:0; font-size: 14px;">Analytical: {an_finish:.2%}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Result Card 2
        with col_res2:
            st.markdown(
                f"""
                <div style="background-color: #F9FAFB; padding: 20px; border-radius: 10px; border: 1px solid #E5E7EB; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                    <h3 style="color: #6B7280; margin:0; font-size: 18px;">Touch Target (Anytime)</h3>
                    <h1 style="color: {prob_color(mc_touch)}; font-size: 48px; margin: 10px 0;">{mc_touch:.2%}</h1>
                    <p style="color: #4B5563; margin:0; font-size: 14px;">Analytical: {an_touch:.2%}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        st.write("") # Spacer

        # --- CHART SECTION ---
        st.subheader("📉 Simulation Visualizer")
        
        # Plot Logic (Subset for speed)
        subset = paths[:, :100]
        x_axis = np.linspace(0, days_diff, subset.shape[0])
        
        fig = go.Figure()

        # Add individual paths
        for i in range(subset.shape[1]):
            hit = np.max(subset[:, i]) >= K
            # Green if it touches target, faint grey if not
            color = 'rgba(22, 163, 74, 0.5)' if hit else 'rgba(107, 114, 128, 0.1)'
            width = 1.5 if hit else 1
            
            fig.add_trace(go.Scatter(
                x=x_axis, y=subset[:, i], mode='lines', 
                line=dict(color=color, width=width), 
                hoverinfo='skip'
            ))

        # Target Line
        fig.add_hline(y=K, line_color="#DC2626", line_width=2, line_dash="dash", annotation_text=f"Target ${K}", annotation_position="top right")
        # Start Line
        fig.add_hline(y=S0, line_color="#2563EB", line_width=2, annotation_text=f"Start ${S0}", annotation_position="bottom right")

        # Chart Styling to match LIGHT theme
        fig.update_layout(
            title="Projected Paths (Green = Hit / Grey = Miss)",
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
            font=dict(color="#111827"),    # Black text
            xaxis=dict(title="Days", gridcolor="#E5E7EB"),
            yaxis=dict(title="Price ($)", gridcolor="#E5E7EB"),
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter your parameters in the sidebar to begin.")