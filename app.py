import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Page Configuration (Wide layout for better viewing)
# ==========================================
st.set_page_config(page_title="Regionserien Analytics Dashboard", layout="wide", page_icon="⚾")

# Team ID Mapping Table
TEAM_IDS = {
    
    "Stockholm B": "36163",
    "Stockholm J": "36168",
    "Alby Stars": "36162",
    "Enköping": "36171",
    "Enskede": "36165",
    "Sundbyberg": "36164",
    "Sundbyberg junior": "36169"
}

# ==========================================
# 2. Data Fetching and Preprocessing
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    # Official WBSC Stats API URL
    url = "https://stats.baseboll-softboll.se/api/v1/stats/events/2025-regionserien-baseboll/index?section=players&stats-section=batting&team=&round=&split=&split=&language=en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://stats.baseboll-softboll.se/'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        # Check if response is valid JSON
        if "application/json" not in response.headers.get("Content-Type", ""):
            st.error("Server returned non-JSON format. Please try again later.")
            return pd.DataFrame()
            
        json_data = response.json()
        df = pd.json_normalize(json_data['data'], sep='_')
        
        # 26 Basic Numerical Features
        numeric_cols = ['g', 'gs', 'ab', 'r', 'h', 'double', 'triple', 'hr', 'rbi', 'tb', 
                        'avg', 'slg', 'obp', 'ops', 'bb', 'hbp', 'so', 'gdp', 'sf', 'sh', 'sb', 'cs']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Normalize Rate Stats (AVG, OBP, SLG, OPS) by 1000
        for rate_col in ['avg', 'slg', 'obp', 'ops']:
            if rate_col in df.columns:
                df[rate_col] = df[rate_col] / 1000
        
        # Clean Player Names (Remove HTML Tags)
        def clean_name(text):
            return " ".join(re.sub(r'<[^>]*>', ' ', str(text)).split())
        df['name_clean'] = df['name'].apply(clean_name)
        
        return df
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. Main Logic and Metric Calculations
# ==========================================
st.title("⚾ Real-time Baseball Analytics System")

df_all = load_data()

if not df_all.empty:
    # Top Control Panel
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_team = st.selectbox("Select Team to Analyze:", list(TEAM_IDS.keys()))
    with col2:
        min_ab = st.slider("Minimum At Bats (AB) Filter:", 1, 50, 10) # Filter AB >= 10
    
    # Filter by Team and AB
    target_id = TEAM_IDS[selected_team]
    df_team = df_all[(df_all['teamid'].astype(str) == target_id) & (df_all['ab'] >= min_ab)].copy()
    
    if df_team.empty:
        st.warning(f"No players found with AB >= {min_ab}.")
    else:
        # Safe Division Helper
        def safe_divide(numerator, denominator):
            return np.where(denominator == 0, 0, numerator / denominator)

        # Advanced Metrics Calculations
        df_team['pa'] = df_team['ab'] + df_team['bb'] + df_team['hbp'] + df_team['sf'] + df_team['sh']
        
        babip_denom = df_team['ab'] - df_team['so'] - df_team['hr'] + df_team['sf']
        df_team['babip'] = safe_divide(df_team['h'] - df_team['hr'], babip_denom)
        
        df_team['iso_val'] = df_team['slg'] - df_team['avg']
        df_team['k_pct'] = safe_divide(df_team['so'], df_team['pa'])
        df_team['bb_pct'] = safe_divide(df_team['bb'], df_team['pa'])
        df_team['rc'] = safe_divide((df_team['h'] + df_team['bb']) * df_team['tb'], (df_team['ab'] + df_team['bb']))
        df_team['gpa_val'] = (1.8 * df_team['obp'] + df_team['slg']) / 4

        # ==========================================
        # 4. Dual Tab Interface (English)
        # ==========================================
        tab1, tab2 = st.tabs(["📋 Basic Stats", "🔬 Advanced Analytics"])
        
        # --- Tab 1: Basic Stats (Rate Stats moved to front) ---
        with tab1:
            st.subheader(f"📋 Raw Performance Data: {selected_team}")
            # Rearranged columns as requested
            basic_columns = [
                'name_clean', 'ab', 'avg', 'obp', 'slg', 'ops', # Primary Rate Stats first
                'g', 'gs', 'r', 'h', 'double', 'triple', 'hr', 'rbi', 
                'tb', 'bb', 'hbp', 'so', 'gdp', 'sf', 'sh', 'sb', 'cs'
            ]
            
            # Filter valid columns to avoid KeyError
            available_basic_cols = [c for c in basic_columns if c in df_team.columns]
            
            st.dataframe(
                df_team[available_basic_cols].style.format({
                    'avg': '{:.3f}', 'obp': '{:.3f}', 'slg': '{:.3f}', 'ops': '{:.3f}'
                }),
                use_container_width=True,
                hide_index=True
            )

        # --- Tab 2: Advanced Analytics ---
        with tab2:
            st.subheader("🔬 Advanced Sabermetrics")
            
            adv_cols = ['name_clean', 'pa', 'babip', 'iso_val', 'k_pct', 'bb_pct', 'rc', 'gpa_val']
            st.dataframe(
                df_team.sort_values('rc', ascending=False)[adv_cols].style.format({
                    'babip': '{:.3f}', 'iso_val': '{:.3f}', 'k_pct': '{:.1%}', 'bb_pct': '{:.1%}', 
                    'rc': '{:.2f}', 'gpa_val': '{:.3f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.divider()
            
            # Visualization Section
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("**📊 Runs Created (RC) Ranking**")
                top_10_rc = df_team.sort_values('rc', ascending=False).head(10)
                fig_rc, ax_rc = plt.subplots(figsize=(8, 5))
                sns.barplot(data=top_10_rc, x='rc', y='name_clean', palette='magma', ax=ax_rc)
                ax_rc.set_xlabel("Estimated Runs Created (RC)")
                ax_rc.set_ylabel("")
                st.pyplot(fig_rc)
                
            with col_chart2:
                st.markdown("**🎯 Plate Discipline: K% vs BB%**")
                fig_disc, ax_disc = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df_team, x='k_pct', y='bb_pct', size='rc', hue='gpa_val', sizes=(80, 400), alpha=0.7, ax=ax_disc)
                
                # Label all players
                for _, row in df_team.iterrows():
                    ax_disc.text(row['k_pct'] + 0.003, row['bb_pct'], row['name_clean'], fontsize=8)

                ax_disc.axhline(df_team['bb_pct'].mean(), color='red', linestyle='--', alpha=0.5, label='Team Avg BB%')
                ax_disc.axvline(df_team['k_pct'].mean(), color='blue', linestyle='--', alpha=0.5, label='Team Avg K%')
                
                # Format legend to upper right
                ax_disc.legend(loc='upper right', fontsize='x-small', title='Metrics')
                
                ax_disc.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax_disc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                st.pyplot(fig_disc)