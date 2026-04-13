import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ======== Please add these two lines ========
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# New: Pitching data loading and calculation
# ==========================================
@st.cache_data(ttl=3600)
def load_pitching_data():
    # Remove the team parameter from the URL to fetch pitching data for all teams at once
    url = "https://stats.baseboll-softboll.se/api/v1/stats/events/2025-regionserien-baseboll/index?section=players&stats-section=pitching&team=&round=&split=&split=&language=en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        json_data = response.json()
        df = pd.json_normalize(json_data['data'], sep='_')
        
        # Define columns that need to be converted to numeric
        numeric_cols = [
            'pitch_win', 'pitch_loss', 'era', 'pitch_appear', 'pitch_gs', 'pitch_save', 'pitch_cg', 'pitch_sho', 'pitch_ip', 'pitch_h', 'pitch_r', 
            'pitch_er', 'pitch_bb', 'pitch_so', 'pitch_double', 'pitch_triple', 'pitch_hr', 
            'pitch_ab', 'bavg', 'pitch_wp', 'pitch_hbp', 'pitch_bk', 'pitch_sfa', 'pitch_sha', 'pitch_ground', 'pitch_fly', 'pitch_whip'
        
        ]
        

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Normalize bavg by dividing by 1000
        if 'bavg' in df.columns:
            df['bavg'] = df['bavg'] / 1000

        # Clean names
        def clean_name(text):
            return " ".join(re.sub(r'<[^>]*>', ' ', str(text)).split())
        df['name_clean'] = df['name'].apply(clean_name)
        
        # Process innings pitched (IP) conversion: In baseball, 10.1 innings represents 10 and 1/3 innings
        def convert_ip(ip):
            ip_str = str(ip)
            if '.' in ip_str:
                full, partial = ip_str.split('.')
                return float(full) + (float(partial) / 3.0)
            return float(ip_str)
            
        df['ip_calc'] = df['pitch_ip'].apply(convert_ip)
        
        # Safe division function
        def safe_divide(numerator, denominator):
            return np.where(denominator == 0, 0, numerator / denominator)

        # ----------------------------------------
        # Calculate advanced metrics (Advanced Metrics)
        # ----------------------------------------
        
        # 1. Total Batters Faced (TBF) estimate
        # TBF = AB + BB + HBP + SF + SH
        df['tbf'] = df['pitch_ab'] + df['pitch_bb'] + df['pitch_hbp'] + df['pitch_sfa'] + df['pitch_sha']
        
        # 2. FIP (Fielding Independent Pitching)
        # Formula: (13*HR + 3*(BB+HBP) - 2*SO) / IP + C (Constant C is usually about 3.15)
        fip_constant = 3.15
        fip_numerator = (13 * df['pitch_hr']) + (3 * (df['pitch_bb'] + df['pitch_hbp'])) - (2 * df['pitch_so'])
        df['fip'] = safe_divide(fip_numerator, df['ip_calc']) + fip_constant
        
        # 3. Opponent BABIP (Batting Average on Balls in Play)
        # Formula: (H - HR) / (AB - SO - HR + SF)
        babip_denom = df['pitch_ab'] - df['pitch_so'] - df['pitch_hr'] + df['pitch_sfa']
        df['opp_babip'] = safe_divide((df['pitch_h'] - df['pitch_hr']), babip_denom)
        
        # 4. K%, BB%, K-BB%
        df['k_pct'] = safe_divide(df['pitch_so'], df['tbf'])
        df['bb_pct'] = safe_divide(df['pitch_bb'], df['tbf'])
        df['k_minus_bb_pct'] = df['k_pct'] - df['bb_pct']
        # 👇 ADD THIS NEW CALCULATION 👇
        # 5. Ground/Fly Ratio (GB/FB)
        df['gb_fb_ratio'] = safe_divide(df['pitch_ground'], df['pitch_fly'])

        return df
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading pitching data: {e}")
        return pd.DataFrame()
    
# ==========================================
# 3. Helper Functions
# ==========================================
def rename_pitching_cols(df):
    """Renames columns by removing 'pitch_' prefix and applying custom labels."""
    mapping = {col: col.replace('pitch_', '') for col in df.columns if col.startswith('pitch_')}
    mapping.update({
        'ip_calc': 'IP',
        'bavg': 'Opp AVG',
        'name_clean': 'Name',
        'k_pct': 'K%',
        'bb_pct': 'BB%',
        'k_minus_bb_pct': 'K-BB%',
        'opp_babip': 'BABIP',
        'fip': 'FIP',
        'tbf': 'TBF',
        'whip': 'WHIP'
    })
    return df.rename(columns=mapping)
# ==========================================
# 4. Main Logic and Metric Calculations
# ==========================================
st.title("⚾ 2025 Baseball Analytics System")

# 1. Define safe_divide globally once
def safe_divide(numerator, denominator):
    return np.where(denominator == 0, 0, numerator / denominator)

# 2. Load both datasets
df_batting = load_data()          
df_pitching = load_pitching_data() 

if not df_batting.empty:
    # --- CALCULATE LEAGUE-WIDE BATTING METRICS FIRST ---
    # This ensures Tab 7 can calculate percentiles correctly
    df_batting['pa'] = df_batting['ab'] + df_batting['bb'] + df_batting['hbp'] + df_batting['sf'] + df_batting['sh']
    
    babip_denom = df_batting['ab'] - df_batting['so'] - df_batting['hr'] + df_batting['sf']
    df_batting['babip'] = safe_divide(df_batting['h'] - df_batting['hr'], babip_denom)
    
    df_batting['iso'] = df_batting['slg'] - df_batting['avg'] # Changed from iso_val to iso
    df_batting['k_pct'] = safe_divide(df_batting['so'], df_batting['pa'])
    df_batting['bb_pct'] = safe_divide(df_batting['bb'], df_batting['pa'])
    df_batting['rc'] = safe_divide((df_batting['h'] + df_batting['bb']) * df_batting['tb'], (df_batting['ab'] + df_batting['bb']))
    df_batting['gpa'] = (1.8 * df_batting['obp'] + df_batting['slg']) / 4 # Changed from gpa_val to gpa

    # 3. Top Control Panel
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_team = st.selectbox("Select Team to Analyze:", list(TEAM_IDS.keys()))
    with col2:
        min_ab = st.slider("Minimum At Bats (AB) Filter:", 1, 50, 10)
    
    # 4. Filter by Team and AB (df_team now inherits the calculated columns)
    target_id = TEAM_IDS[selected_team]
    df_team = df_batting[(df_batting['teamid'].astype(str) == target_id) & (df_batting['ab'] >= min_ab)].copy()
    
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
        # 5. Dual Tab Interface (English)
        # ==========================================
        # Correct way to write (add the 3rd title)
        # Replace with writing containing 5 tabs
        # 加入第 6 個標題 "📖 數據字典"
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Batting Basic", 
        "🔬 Batting Adv", 
        "🥎 Pitching Basic", 
        "🤖 Pitching ML", 
        "⚔️ BP Sim", 
        "📖 Glossary",
        "👤 Player Profile"
    ])
        
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
                sns.barplot(data=top_10_rc, x='rc', y='name_clean', palette='viridis', ax=ax_rc)
                ax_rc.set_xlabel("Estimated Runs Created (RC)")
                ax_rc.set_ylabel("")
                st.pyplot(fig_rc)
                
            with col_chart2:
                st.markdown("**🎯 Plate Discipline: K% vs BB%**")
                fig_disc, ax_disc = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df_team, x='k_pct', y='bb_pct', size='rc', hue='gpa_val', sizes=(80, 400), alpha=0.7, ax=ax_disc)
                ax_disc.legend(bbox_to_anchor=(1.02, 1), loc='upper left',fontsize='small',borderaxespad=0.)
                # Label all players
                for _, row in df_team.iterrows():
                    ax_disc.text(row['k_pct'] + 0.003, row['bb_pct'], row['name_clean'], fontsize=8)

                ax_disc.axhline(df_team['bb_pct'].mean(), color='red', linestyle='--', alpha=0.5, label='Team Avg BB%')
                ax_disc.axvline(df_team['k_pct'].mean(), color='blue', linestyle='--', alpha=0.5, label='Team Avg K%')
                
                
                
                ax_disc.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax_disc.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                st.pyplot(fig_disc)
        with tab3:
            st.subheader(f"🥎 Pitching Dashboard: {selected_team}")
                
            df_pitching_all = load_pitching_data()
            
            if not df_pitching_all.empty:
                target_id = TEAM_IDS[selected_team]
                df_team_pitchers = df_pitching_all[df_pitching_all['teamid'].astype(str) == target_id].copy()
                
                if df_team_pitchers.empty:
                    st.warning(f"No pitching stats found for {selected_team}.")
                else:
                    # Calculate WHIP
                    df_team_pitchers['whip'] = safe_divide((df_team_pitchers['pitch_bb'] + df_team_pitchers['pitch_h']), df_team_pitchers['ip_calc'])
                    
                    # -------------------------
                    # 1. Basic Pitching Data
                    # -------------------------
                    st.markdown("### 📋 Basic Pitching Stats")
                    st.info("Includes W/L, ERA, Innings Pitched (IP), Hits, Walks, Strikeouts, and WHIP.")
                    
                    basic_pitching_cols = [
                        'name_clean', 'pitch_win', 'pitch_loss', 'era', 'pitch_appear', 'pitch_gs', 
                        'pitch_save', 'pitch_cg', 'pitch_sho', 'ip_calc', 'pitch_h', 'pitch_r', 
                        'pitch_er', 'pitch_bb', 'pitch_so', 'pitch_double', 'pitch_triple', 'pitch_hr', 
                        'bavg', 'pitch_wp', 'pitch_hbp', 'pitch_bk', 'pitch_sfa', 'pitch_sha', 
                        'pitch_ground', 'pitch_fly', 'whip' 
                    ]
                    rename_map = {col: col.replace('pitch_', '') for col in basic_pitching_cols}
                    rename_map.update({'bavg': 'Opp AVG', 'ip_calc': 'IP', 'whip': 'WHIP'})
                    st.dataframe(
                        df_team_pitchers[basic_pitching_cols].rename(columns=rename_map).style.format({
                            'era': '{:.2f}', 
                            'IP': '{:.1f}', 
                            'WHIP': '{:.2f}', 
                            'Opp AVG': '{:.3f}' # 注意這裡要改成對應新的名稱
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.divider()
                    
                    # -------------------------
                    # 2. Advanced Pitching Data
                    # -------------------------
                    st.markdown("### 🔬 Advanced Sabermetrics")
                    st.info("**FIP**: Fielding Independent Pitching | **Opp BABIP**: Batting Average on Balls In Play | **K-BB%**: Strikeout minus Walk percentage (evaluates command and dominance)")
                    
                    adv_pitching_cols = [
                        'name_clean', 'tbf', 'fip', 'opp_babip', 'k_pct', 'bb_pct', 'k_minus_bb_pct'
                    ]
                    adv_rename_map = {col: col.replace('pitch_', '') for col in adv_pitching_cols}
                    adv_rename_map.update({
                        'tbf': 'TBF', 'fip': 'FIP', 'opp_babip': 'BABIP', 
                        'k_pct': 'K%', 'bb_pct': 'BB%', 'k_minus_bb_pct': 'K-BB%'
                    })
                    
                    st.dataframe(
                        df_team_pitchers[adv_pitching_cols].rename(columns=adv_rename_map).style.format({
                            'fip': '{:.2f}', 
                            'opp_babip': '{:.3f}', 
                            'k_pct': '{:.1%}', 
                            'bb_pct': '{:.1%}', 
                            'k_minus_bb_pct': '{:.1%}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
            with tab4:
                # 1. Get league-wide pitching data (filter out samples with innings pitched less than 5)
                df_pitchers_all = load_pitching_data()
                df_pitchers_all = df_pitchers_all[df_pitchers_all['ip_calc'] >= 5].copy() 

                if not df_pitchers_all.empty:
                    st.subheader(f"🤖 Machine Learning Analysis (Pitching Side) - {selected_team}")
                    
                   # ==========================================
                    # Enhanced Machine Learning Model (4 Clusters)
                    # ==========================================
                    # 1. Add gb_fb_ratio to the training features
                    features = ['k_pct', 'bb_pct', 'opp_babip', 'fip', 'gb_fb_ratio']
                    X = df_pitchers_all[features].fillna(0)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # 2. Execute K-Means clustering (Now upgraded to 4 clusters)
                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                    df_pitchers_all['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # 3. Dynamic Labeling: Safely label clusters based on their mathematical centroids
                    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
                    style_map = {}
                    
                    for i, row in cluster_centers.iterrows():
                        if row['bb_pct'] == cluster_centers['bb_pct'].max() and row['fip'] > 4.0:
                            style_map[i] = "Wild/Struggling"
                        elif row['gb_fb_ratio'] == cluster_centers['gb_fb_ratio'].max():
                            style_map[i] = "Groundball Specialist"
                        elif row['k_pct'] == cluster_centers['k_pct'].max():
                            style_map[i] = "Power/Strikeout"
                        else:
                            style_map[i] = "Finesse/Control"
                            
                    df_pitchers_all['style'] = df_pitchers_all['cluster'].map(style_map)
                    
                    # Calculate luck index (ERA minus FIP)
                    df_pitchers_all['era_minus_fip'] = df_pitchers_all['era'] - df_pitchers_all['fip']
                    
                    # Filter for the selected team
                    target_id = TEAM_IDS[selected_team]
                    df_team_pitchers = df_pitchers_all[df_pitchers_all['teamid'].astype(str) == target_id].copy()

                    if df_team_pitchers.empty:
                        st.warning(f"Currently {selected_team} has no pitcher data that meets the conditions (innings pitched >= 5).")
                    else:
                        st.markdown("### 1. Enhanced Pitcher Style Clustering")
                        
                        fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
                        
                        # Plot: GB/FB Ratio (X-axis) vs K% (Y-axis)
                        sns.scatterplot(data=df_team_pitchers, x='gb_fb_ratio', y='k_pct', hue='style', s=150, ax=ax_cluster)
                        
                        # Add player name labels
                        for _, row in df_team_pitchers.iterrows():
                            ax_cluster.text(row['gb_fb_ratio'] + 0.05, row['k_pct'] + 0.002, row['name_clean'], fontsize=9)

                        # Update Chart formatting
                        ax_cluster.set_title(f"Pitcher Styles ({selected_team}): GB/FB Ratio vs Strikeout Rate")
                        ax_cluster.set_xlabel("Groundball/Flyball Ratio (GB/FB)")
                        ax_cluster.set_ylabel("Strikeout Rate (K%)")
                        
                        # Format Y-axis as percentage (X-axis stays as standard float ratio)
                        ax_cluster.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                        
                        # Move legend outside the plot so it doesn't cover data points
                        ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                        st.pyplot(fig_cluster)
                        
                        st.divider()
                        
                        st.markdown("### 2. ERA Unexploded Bomb Detection (ERA vs FIP)")
                        st.info("Compare the pitcher's actual ERA with advanced independent ERA (FIP). If ERA is much lower than FIP, it means luck is a big factor, with risk of exploding in the future; otherwise, strength is underestimated.")
                        
                        df_team_pitchers = df_team_pitchers.sort_values('era_minus_fip', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.error("🚨 Extremely lucky / unexploded bomb zone (ERA lower than FIP)")
                            # Filter pitchers with ERA less than FIP (negative values)
                            lucky_pitchers = df_team_pitchers[df_team_pitchers['era_minus_fip'] < 0]
                            st.dataframe(lucky_pitchers[['name_clean', 'era', 'fip', 'era_minus_fip']].style.format({
                                'era': '{:.2f}', 'fip': '{:.2f}', 'era_minus_fip': '{:.2f}'
                            }), hide_index=True)
                            
                        with col2:
                            st.success("💎 Extremely unlucky / strength underestimated (ERA higher than FIP)")
                            # Filter pitchers with ERA greater than FIP (positive values)
                            unlucky_pitchers = df_team_pitchers[df_team_pitchers['era_minus_fip'] >= 0]
                            st.dataframe(unlucky_pitchers[['name_clean', 'era', 'fip', 'era_minus_fip']].style.format({
                                'era': '{:.2f}', 'fip': '{:.2f}', 'era_minus_fip': '{:.2f}'
                            }), hide_index=True)




               
            with tab5:
                st.subheader("⚔️ Pitcher vs Batter: Matchup Expected Value Simulation")
                st.info("Combine the batter's extra-base hit ability (ISO) with the pitcher's advanced independent data (FIP, K%), predict the probability of specific outcomes in the at-bat.")
                df_batters_all = load_data()
                df_pitchers_all = load_pitching_data()

                if not df_batters_all.empty and not df_pitchers_all.empty:
                    df_batters_all['pa'] = df_batters_all['ab'] + df_batters_all['bb'] + df_batters_all['hbp'] + df_batters_all['sf'] + df_batters_all['sh']
                    df_batters_all['iso_val'] = df_batters_all['slg'] - df_batters_all['avg']
                    df_batters_all['k_pct'] = safe_divide(df_batters_all['so'], df_batters_all['pa'])
                    
                    df_batters_filtered = df_batters_all[df_batters_all['pa'] >= 10]
                    df_pitchers_filtered = df_pitchers_all[df_pitchers_all['tbf'] >= 10]

                    col_sel1, col_sel2 = st.columns(2)
                    
                    with col_sel1:
                        st.markdown("### 🦇 Select Batter (Offense)")
                        b_team = st.selectbox("Batter's team", list(TEAM_IDS.keys()), key="bat_team")
                        b_players = df_batters_filtered[df_batters_filtered['teamid'].astype(str) == TEAM_IDS[b_team]]
                        
                        if b_players.empty:
                            st.warning("The team currently has no qualified batters.")
                            selected_batter_name = None
                        else:
                            selected_batter_name = st.selectbox("Select batter", b_players['name_clean'].tolist())
                            batter_stats = b_players[b_players['name_clean'] == selected_batter_name].iloc[0]

                    with col_sel2:
                        st.markdown("### ⚾ Select Pitcher (Defense)")
                        p_team = st.selectbox("Pitcher's team", list(TEAM_IDS.keys()), key="pit_team")
                        p_players = df_pitchers_filtered[df_pitchers_filtered['teamid'].astype(str) == TEAM_IDS[p_team]]
                        
                        if p_players.empty:
                            st.warning("The team currently has no qualified pitchers.")
                            selected_pitcher_name = None
                        else:
                            selected_pitcher_name = st.selectbox("Select pitcher", p_players['name_clean'].tolist())
                            pitcher_stats = p_players[p_players['name_clean'] == selected_pitcher_name].iloc[0]

                    st.divider()

                    # If both batter and pitcher are successfully selected, proceed with calculation
                if selected_batter_name and selected_pitcher_name:
                    # 1. Get league averages (as baseline)
                    lg_avg_k_pct = df_batters_filtered['k_pct'].mean()
                    lg_avg_hr_rate = safe_divide(df_batters_filtered['hr'].sum(), df_batters_filtered['ab'].sum())
                    lg_avg_ba = safe_divide(df_batters_filtered['h'].sum(), df_batters_filtered['ab'].sum()) # New: League average batting average
                    
                    # Avoid extreme values or divide by 0 protection mechanism
                    lg_avg_k_pct = lg_avg_k_pct if lg_avg_k_pct > 0 else 0.20
                    lg_avg_hr_rate = lg_avg_hr_rate if lg_avg_hr_rate > 0 else 0.03
                    lg_avg_ba = lg_avg_ba if lg_avg_ba > 0 else 0.250
                    
                    # 2. Extract both sides' data
                    b_k = batter_stats['k_pct']
                    b_iso = batter_stats['iso_val']
                    b_avg = batter_stats['avg'] # New: Batter's batting average
                    
                    p_k = pitcher_stats['k_pct']
                    p_fip = pitcher_stats['fip']
                    p_opp_avg = safe_divide(pitcher_stats['pitch_h'], pitcher_stats['pitch_ab']) # New: Pitcher's opponent batting average
                    
                    # --- 3. Prediction formula calculation ---
                    # Strikeout rate prediction
                    pred_k = min((b_k * p_k) / lg_avg_k_pct, 0.95) 
                    
                    # Home run prediction
                    pred_hr = min(max((b_iso * 0.2) * (pitcher_stats['fip'] / 3.15), 0), 0.30)

                    # New: Hit probability prediction (using Bill James' Log-5 formula)
                    if lg_avg_ba > 0 and lg_avg_ba < 1:
                        log5_num = (b_avg * p_opp_avg) / lg_avg_ba
                        log5_den = log5_num + ((1 - b_avg) * (1 - p_opp_avg)) / (1 - lg_avg_ba)
                        pred_hit = log5_num / log5_den if log5_den > 0 else 0
                    else:
                        pred_hit = 0

                    st.markdown(f"#### 🏟️ Matchup simulation results: **{selected_batter_name}** vs **{selected_pitcher_name}**")
                    
                    # Expand to 5 display columns
                    res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                    
                    # Strikeout prediction display
                    k_diff = pred_k - lg_avg_k_pct
                    res_col1.metric("Predicted strikeout rate (K%)", f"{pred_k:.1%}", f"{k_diff*100:.1f}% vs league average", delta_color="inverse")
                    
                    # New: Hit prediction display
                    hit_diff = pred_hit - lg_avg_ba
                    res_col2.metric("Predicted hit rate (xBA)", f"{pred_hit:.1%}", f"{hit_diff*100:.1f}% vs league average", delta_color="normal")
                    
                    # Home run prediction display
                    hr_diff = pred_hr - lg_avg_hr_rate
                    res_col3.metric("Predicted home run rate (HR%)", f"{pred_hr:.1%}", f"{hr_diff*100:.1f}% vs league average", delta_color="normal")
                    
                    # Key matchup indicators
                    res_col4.metric("Batter's extra-base hit threat (ISO)", f"{b_iso:.3f}")
                    res_col5.metric("Pitcher's independent ERA (FIP)", f"{p_fip:.2f}")

                    # Advanced tactical advice tips (integrating hit prediction)
                    st.markdown("---")
                    if pred_hr > 0.08:
                        st.error("🚨 **High risk warning (extra-base hit)**: This batter has an extremely high home run rate against this pitcher! If it's a critical moment, consider **intentional walk (IBB)** or pitching to avoid.")
                    elif pred_hit > 0.350:
                        st.warning("⚠️ **High on-base risk (hit)**: The batter has a high expected value for hits, infield defense should shift back or position based on batter's hitting tendencies (Shift).")
                    elif pred_k > 0.35:
                        st.success("🎯 **Strikeout advantage**: The pitcher has an extremely high strikeout rate advantage in this matchup, can boldly use putaway pitches to attack the strike zone!")
                    else:
                        st.info("⚖️ **Even matchup**: This is a standard pitcher vs batter matchup, the result will largely depend on the day's control and contact.")

            with tab6:
                st.subheader("📖 Term Dictionary & Algorithms")
                st.info("Here you can find definitions for all sabermetrics and the inner workings of our ML/Simulation models.")

                col_dict1, col_dict2 = st.columns(2)

                with col_dict1:
                    st.markdown("### 🦇 Batting Metrics")
                    
                    with st.expander("📊 Basic Batting Stats", expanded=True):
                        st.markdown("""
                        * **G (Games Played)**: Total number of games the player has participated in.
                        * **GS (Games Started)**: Total number of games the player started in the starting lineup.
                        * **AB (At Bats)**: Official plate appearances, excluding walks, hit-by-pitches, and sacrifices.
                        * **R (Runs)**: Number of times the player safely crossed home plate to score.
                        * **H (Hits)**: Total number of successful hits (singles, doubles, triples, and home runs).
                        * **2B (Doubles)**: Number of hits resulting in the batter safely reaching second base.
                        * **3B (Triples)**: Number of hits resulting in the batter safely reaching third base.
                        * **HR (Home Runs)**: Number of hits where the batter circles all bases and scores.
                        * **RBI (Runs Batted In)**: Number of runs scored as a direct result of the batter's action at the plate.
                        * **TB (Total Bases)**: The total number of bases gained by a batter through their hits (Single=1, Double=2, Triple=3, HR=4).
                        * **AVG (Batting Average)**: Hits divided by At Bats (H / AB). The traditional metric for a player's hitting ability.
                        * **SLG (Slugging Percentage)**: Measures a batter's power by calculating total bases divided by at-bats (TB / AB).
                        * **OBP (On-Base Percentage)**: How frequently a batter reaches base (Hits + Walks + Hit By Pitch).
                        * **OPS (On-Base Plus Slugging)**: OBP + SLG. A comprehensive metric for a batter's overall offensive production.
                        * **BB (Walks)**: Free passes to first base awarded after receiving four balls.
                        * **HBP (Hit By Pitch)**: Times the batter was awarded first base after being struck by a pitched ball.
                        * **SO (Strikeouts)**: Number of times the batter struck out.
                        * **GDP (Grounded Into Double Play)**: Number of times the batter hit a ground ball that resulted in multiple outs.
                        * **SF (Sacrifice Flies)**: Fly balls hit to the outfield that result in a baserunner scoring from third base.
                        * **SH (Sacrifice Hits / Bunts)**: Successful bunts that advance a baserunner while resulting in the batter being put out.
                        * **SB (Stolen Bases)**: Number of times the runner successfully advanced to the next base without the aid of a hit, putout, or error.
                        * **CS (Caught Stealing)**: Number of times the runner was tagged out while attempting to steal a base.
                        """)

                    with st.expander("🔬 Advanced Batting Stats"):
                        st.markdown("""
                        * **PA (Plate Appearances)**: Total number of times the batter steps into the box.
                        * **BABIP (Batting Average on Balls In Play)**: The rate at which batted balls (excluding home runs and strikeouts) fall for hits. Often used to measure luck.
                        * **ISO (Isolated Power)**: SLG minus AVG. Measures a batter's pure power. Values > .200 indicate a strong power hitter.
                        * **K% (Strikeout Rate)**: SO divided by PA.
                        * **BB% (Walk Rate)**: BB divided by PA. An excellent indicator of plate discipline and batting eye.
                        * **RC (Runs Created)**: A metric created by Bill James estimating how many runs a player has contributed to their team.The baseline formula is $$RC = \\frac{(H + BB) \\times TB}{AB + BB}$$which mathematically multiplies the "on-base factor" (Hits plus Walks) by the "advancement factor" (Total Bases), and then divides the result by the "opportunity factor" (At Bats plus Walks). By combining these elements, the formula calculates offensive efficiency; for example, if a player finishes the season with an RC of 80, it means their individual offensive production at the plate was directly responsible for generating approximately 80 runs for their team throughout the year.
                        * **GPA (Gross Production Average)**: Similar to OPS, The formula is $$ GPA = \\frac{1.8 \\times OBP + SLG}{4}$$which corrects the primary flaw of OPS by multiplying On-Base Percentage (OBP) by a weight of 1.8. This adjustment reflects the historical statistical reality that avoiding an out and getting on base is approximately 1.8 times more valuable to run creation than hitting for power (SLG). 
                        """)

                with col_dict2:
                    st.markdown("### ⚾ Pitching Metrics")

                    with st.expander("📋 Basic Pitching Stats", expanded=True):
                        st.markdown("""
                        * **W (Wins)**: The number of games where the pitcher was credited with the victory.
                        * **L (Losses)**: The number of games where the pitcher was credited with the defeat.
                        * **ERA (Earned Run Average)**: The average number of earned runs a pitcher gives up per 9 innings pitched.
                        * **APP / G (Appearances)**: Total number of games the pitcher has pitched in.
                        * **GS (Games Started)**: Total number of games the pitcher started.
                        * **SV (Saves)**: Awarded to a relief pitcher who successfully finishes a game for the winning team under close-score conditions.
                        * **CG (Complete Games)**: Number of games where the starting pitcher pitches the entire game without being replaced.
                        * **SHO (Shutouts)**: A complete game pitched without allowing the opposing team to score a single run.
                        * **IP (Innings Pitched)**: The number of innings a pitcher has completed (each out represents 1/3 of an inning).
                        * **H (Hits Allowed)**: Total number of hits given up by the pitcher.
                        * **R (Runs Allowed)**: Total runs given up by the pitcher, including both earned and unearned runs.
                        * **ER (Earned Runs)**: Runs allowed that were directly attributed to the pitcher, not the result of defensive errors.
                        * **BB (Walks Allowed)**: Free passes given to batters (Base on Balls).
                        * **SO / K (Strikeouts)**: Number of batters struck out by the pitcher.
                        * **2B (Doubles Allowed)**: Number of two-base hits given up by the pitcher.
                        * **3B (Triples Allowed)**: Number of three-base hits given up by the pitcher.
                        * **HR (Home Runs Allowed)**: Number of home runs given up by the pitcher.
                        * **AB (At Bats Against)**: Official at-bats registered by batters against the pitcher.
                        * **Opp AVG / BAVG (Opponent Batting Average)**: The collective batting average of opposing hitters against this pitcher.
                        * **WP (Wild Pitches)**: Pitches thrown so wildly that the catcher cannot handle them, allowing a runner to advance.
                        * **HBP (Hit By Pitch)**: Number of times the pitcher hit a batter with a pitch, awarding them first base.
                        * **BK (Balks)**: Illegal pitching motions that result in baserunners being automatically awarded the next base.
                        * **SFA (Sacrifice Flies Allowed)**: Number of sacrifice flies hit against the pitcher.
                        * **SHA (Sacrifice Hits Allowed)**: Number of sacrifice bunts successfully laid down against the pitcher.
                        * **Ground (Ground Balls / Ground Outs)**: Number of batted balls hit on the ground or outs recorded via grounders.
                        * **Fly (Fly Balls / Fly Outs)**: Number of batted balls hit in the air or outs recorded via fly balls.
                        * **WHIP (Walks and Hits Per Inning Pitched)**: The average number of baserunners a pitcher allows per inning ((BB + H) / IP).
                        """)

                    with st.expander("🔬 Advanced Pitching Stats"):
                        st.markdown("""
                        * **TBF (Total Batters Faced)**: The total number of batters the pitcher has pitched to.
                        * **FIP (Fielding Independent Pitching)**: Estimates a pitcher's ERA based *only* on outcomes they control (strikeouts, walks, hit-by-pitches, home runs). It excludes defense and luck.
                        * **Opp BABIP**: The opponent's Batting Average on Balls In Play. A very high number suggests bad luck or poor defense behind the pitcher.
                        * **K% / BB%**: Strikeout Rate / Walk Rate against total batters faced.
                        * **K-BB% (Strikeout minus Walk Rate)**: A core modern metric for evaluating a pitcher's pure dominance and command.
                        """)

                st.divider()
                
                st.markdown("### 🤖 Models & Algorithms")

                with st.expander("🧠 Pitching Analytics [ML]", expanded=True):
                    st.markdown("""
                    #### 1. Pitcher Style Clustering
                    #### 1. Enhanced Pitcher Style Clustering
                    * **Algorithm**: **K-Means Clustering (4 Clusters)**
                    * **Logic**: The system extracts five core features from all pitchers: `K%`, `BB%`, `Opp BABIP`, `FIP`, and the newly added `GB/FB Ratio` (Groundball/Flyball Ratio). Using K-Means unsupervised learning, it projects all pitchers into a multi-dimensional space and automatically finds 4 distinct cluster centroids. The system uses dynamic mathematical labeling to ensure accurate categorization regardless of shifting league averages.
                    * **Categories**: 
                        * **Power/Strikeout**: Pitchers who possess the highest strikeout rates (K%). They overpower hitters to miss bats, though they may occasionally be prone to fly balls.
                        * **Groundball Specialist**: Pitchers with the highest Groundball-to-Flyball ratios (GB/FB). They rely heavily on sinkers or breaking balls down in the zone to induce weak grounders and double plays, effectively keeping the ball in the park.
                        * **Finesse/Control**: Pitchers who excel at limiting walks and managing contact. They maintain solid independent metrics (FIP) without necessarily relying on elite strikeout numbers.
                        * **Wild/Struggling**: Pitchers exhibiting the highest walk rates (BB%) combined with elevated FIPs (> 4.0), indicating severe command issues and difficulty preventing runs.
                    
                    #### 2. ERA Regression Risk Detection
                    * **Algorithm**: **Luck Index Calculation (ERA minus FIP)**
                    * **Logic**: Calculates the difference between ERA and FIP. ERA is easily influenced by the defense behind the pitcher or sheer luck, whereas FIP isolates the pitcher's true performance.
                    * **Application**: If ERA is significantly lower than FIP (negative value), the system flags them as "Lucky", warning of future regression. If ERA is higher, it suggests their true skill is currently undervalued due to bad luck.
                    """)

                with st.expander("⚔️ BP Sim (Matchup Expected Value)", expanded=True):
                    st.markdown("""
                    The Matchup Simulation is built upon the **Log-5 Method** (introduced by baseball historian Bill James) combined with the interaction of modern sabermetrics.

                    #### 1. Predicted xBA (Expected Batting Average)
                    * **Core Formula**: `Log-5 Method`
                    * **Logic**: It takes the Batter's AVG, Pitcher's Opponent AVG, and the League AVG, and inputs them into the Log-5 equation:
                    $$Expected = \\frac{\\frac{Bat \\times Pit}{Lg}}{\\frac{Bat \\times Pit}{Lg} + \\frac{(1-Bat) \\times (1-Pit)}{(1-Lg)}}$$
                    * **Meaning**: If a .300 batter faces an elite pitcher who only allows a .200 average, Log-5 calculates the mathematically expected probability of a hit occurring when these two forces collide.

                    #### 2. Predicted K% (Strikeout Probability)
                    * **Core Formula**: `(Batter K% × Pitcher K%) / League K%`
                    * **Logic**: Combines the batter's tendency to strike out with the pitcher's ability to induce strikeouts, weighted against the league baseline. 

                    #### 3. Predicted HR% (Home Run Probability)
                    * **Core Formula**: `(Batter ISO × 0.2) × (Pitcher FIP / 3.15)`
                    * **Logic**: Uses the batter's Isolated Power (ISO) as a baseline for strength, and multiplies it by the ratio of the pitcher's FIP. This acts as a proxy model to evaluate the "spark" generated when a batter's raw power meets a pitcher's overall suppressive ability.
                    """)
            # ==========================================
            # --- Tab 7: 👤 Individual Player Profile ---
            # ==========================================
            with tab7:
                st.subheader("👤 Individual Player Intelligence Profile")
                st.info("Select a player to generate a comprehensive AI-driven scouting report.")

                # Merge names from both datasets to ensure we see everyone
                batting_names = df_batting['name_clean'].unique().tolist() if not df_batting.empty else []
                pitching_names = df_pitching['name_clean'].unique().tolist() if not df_pitching.empty else []
                all_players = sorted(list(set(batting_names + pitching_names)))
                
                selected_p = st.selectbox("Search & Select Player:", all_players, key="profile_select")

                if selected_p:
                    p_col1, p_col2 = st.columns([1, 1.5])
                    
                    # These will now work because df_batting and df_pitching were defined globally
                    b_data = df_batting[df_batting['name_clean'] == selected_p]
                    p_data = df_pitching[df_pitching['name_clean'] == selected_p]
                    with p_col1:
                        st.markdown(f"## {selected_p}")
                        # 顯示球隊資訊
                        team_name = b_data['teamcode'].iloc[0] if not b_data.empty else p_data['teamcode'].iloc[0]
                        st.markdown(f"**Team:** {team_name} | **Primary Role:** {'Two-way' if not b_data.empty and not p_data.empty else ('Batter' if not b_data.empty else 'Pitcher')}")
                        
                        # -------------------------
                        # 機器學習邏輯：計算百分位數 (Percentile Rank)
                        # -------------------------
                        def get_rank_label(val, series, inverse=False):
                            if series.empty: return "N/A"
                            percentile = (series < val).mean() if not inverse else (series > val).mean()
                            if percentile >= 0.90: return "🌟 Elite (Top 10%)"
                            if percentile >= 0.75: return "✅ Great (Top 25%)"
                            if percentile >= 0.40: return "🔵 Average"
                            return "⚠️ Below Avg"

                        # --- B. 打擊優劣勢分析 ---
                        if not b_data.empty:
                            st.divider()
                            st.markdown("### 🦇 Batting Scouting Report")
                            row = b_data.iloc[0]
                            
                            # 計算 GPA, ISO, K% 的全聯盟排名
                            gpa_l = get_rank_label(row['gpa'], df_batting['gpa'])
                            iso_l = get_rank_label(row['iso'], df_batting['iso'])
                            k_l = get_rank_label(row['k_pct'], df_batting['k_pct'], inverse=True) # 三振率越低越好
                            
                            st.write(f"**Overall Production (GPA):** {gpa_l}")
                            st.write(f"**Power Threat (ISO):** {iso_l}")
                            st.write(f"**Plate Discipline (K%):** {k_l}")
                            
                            # AI 建議 (簡單邏輯判斷)
                            st.markdown("**💡 Tactical Advice:**")
                            if row['k_pct'] > df_batting['k_pct'].mean() and row['iso'] > df_batting['iso'].mean():
                                st.warning("Classic Power Hitter: High reward, but high strikeout risk. Focus on contact in 2-strike counts.")
                            elif row['obp'] > df_batting['obp'].mean() and row['iso'] < df_batting['iso'].mean():
                                st.success("On-base Specialist: Excellent at drawing walks. Ideal for Lead-off or #2 spot.")
                            else:
                                st.info("Balanced Profile: Adaptable to various spots in the lineup.")

                        # --- C. 投球優劣勢分析 ---
                        if not p_data.empty:
                            st.divider()
                            st.markdown("### ⚾ Pitching Scouting Report")
                            p_row = p_data.iloc[0]
                            
                            fip_l = get_rank_label(p_row['fip'], df_pitching['fip'], inverse=True) # FIP越低越好
                            kbb_l = get_rank_label(p_row['k_minus_bb_pct'], df_pitching['k_minus_bb_pct'])
                            
                            st.write(f"**True Dominance (FIP):** {fip_l}")
                            st.write(f"**Command (K-BB%):** {kbb_l}")

                            st.markdown("**💡 Coaching Insight:**")
                            if p_row['era'] > p_row['fip'] + 3.0:
                                st.success("Luck Factor: You are pitching better than your ERA suggests. Don't change your routine!")
                            elif p_row['k_minus_bb_pct'] > df_pitching['k_minus_bb_pct'].mean():
                                st.success("Strikeout Artist: High ability to finish hitters. Use your put-away pitch early.")

                    with p_col2:
                        # --- D. 雷達圖可視化 (Radar Chart) ---
                        if not b_data.empty:
                            st.markdown("### 📊 Skill Attribute Map")
                            # 準備雷達圖數據 (標準化為 0-100)
                            categories = ['AVG', 'OBP', 'SLG', 'K% (Inv)', 'BB%']
                            
                            # 計算百分位數作為雷達圖的分數
                            def get_p_val(val, series, inv=False):
                                return (series < val).mean() * 100 if not inv else (series > val).mean() * 100

                            stats_val = [
                                get_p_val(row['avg'], df_batting['avg']),
                                get_p_val(row['obp'], df_batting['obp']),
                                get_p_val(row['slg'], df_batting['slg']),
                                get_p_val(row['k_pct'], df_batting['k_pct'], True),
                                get_p_val(row['bb_pct'], df_batting['bb_pct'])
                            ]
                            
                            # 繪製簡單的雷達圖
                            fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                            stats_val += stats_val[:1] # 閉合圖形
                            angles += angles[:1]
                            
                            ax_radar.fill(angles, stats_val, color='red', alpha=0.25)
                            ax_radar.plot(angles, stats_val, color='red', linewidth=2)
                            ax_radar.set_yticklabels([]) # 隱藏圓圈標籤
                            ax_radar.set_xticks(angles[:-1])
                            ax_radar.set_xticklabels(categories)
                            st.pyplot(fig_radar)