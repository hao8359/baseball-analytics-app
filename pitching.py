import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ======== 請加上這兩行 ========
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
# 新增：投手數據載入與計算
# ==========================================
@st.cache_data(ttl=3600)
def load_pitching_data():
    # 移除 URL 中的 team 參數，一次抓取所有球隊的投手數據
    url = "https://stats.baseboll-softboll.se/api/v1/stats/events/2025-regionserien-baseboll/index?section=players&stats-section=pitching&team=&round=&split=&split=&language=en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        json_data = response.json()
        df = pd.json_normalize(json_data['data'], sep='_')
        
        # 定義需要轉為數值的欄位
        numeric_cols = [
            'pitch_win', 'pitch_loss', 'era', 'pitch_appear', 'pitch_gs', 'pitch_save', 
            'pitch_ip', 'pitch_h', 'pitch_r', 'pitch_er', 'pitch_bb', 'pitch_so', 
            'pitch_hr', 'pitch_ab', 'pitch_wp', 'pitch_hbp', 'pitch_sfa', 'pitch_sha'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        # 清理姓名
        def clean_name(text):
            return " ".join(re.sub(r'<[^>]*>', ' ', str(text)).split())
        df['name_clean'] = df['name'].apply(clean_name)
        
        # 處理投球局數 (IP) 轉換：棒球的 10.1 局代表 10又 1/3 局
        def convert_ip(ip):
            ip_str = str(ip)
            if '.' in ip_str:
                full, partial = ip_str.split('.')
                return float(full) + (float(partial) / 3.0)
            return float(ip_str)
            
        df['ip_calc'] = df['pitch_ip'].apply(convert_ip)
        
        # 安全除法函數
        def safe_divide(numerator, denominator):
            return np.where(denominator == 0, 0, numerator / denominator)

        # ----------------------------------------
        # 計算進階指標 (Advanced Metrics)
        # ----------------------------------------
        
        # 1. 總面對打席數 (Total Batters Faced, TBF) 估算
        # TBF = AB + BB + HBP + SF + SH
        df['tbf'] = df['pitch_ab'] + df['pitch_bb'] + df['pitch_hbp'] + df['pitch_sfa'] + df['pitch_sha']
        
        # 2. FIP (Fielding Independent Pitching)
        # 公式: (13*HR + 3*(BB+HBP) - 2*SO) / IP + C (常數 C 通常約為 3.15)
        fip_constant = 3.15
        fip_numerator = (13 * df['pitch_hr']) + (3 * (df['pitch_bb'] + df['pitch_hbp'])) - (2 * df['pitch_so'])
        df['fip'] = safe_divide(fip_numerator, df['ip_calc']) + fip_constant
        
        # 3. Opponent BABIP (被場內安打率)
        # 公式: (H - HR) / (AB - SO - HR + SF)
        babip_denom = df['pitch_ab'] - df['pitch_so'] - df['pitch_hr'] + df['pitch_sfa']
        df['opp_babip'] = safe_divide((df['pitch_h'] - df['pitch_hr']), babip_denom)
        
        # 4. K%, BB%, K-BB%
        df['k_pct'] = safe_divide(df['pitch_so'], df['tbf'])
        df['bb_pct'] = safe_divide(df['pitch_bb'], df['tbf'])
        df['k_minus_bb_pct'] = df['k_pct'] - df['bb_pct']

        return df
    except Exception as e:
        import streamlit as st
        st.error(f"Error loading pitching data: {e}")
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
        # 正確的寫法 (加入第 3 個標題)
        tab1, tab2, tab3 = st.tabs(["📋 Basic Stats", "🔬 Advanced Analytics", "🤖 ML Analytics"])
        
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
            # 假設您已取得篩選後的投手數據 df_pitchers (例如 IP >= 10)
            df_pitchers = load_pitching_data()
            df_pitchers = df_pitchers[df_pitchers['ip_calc'] >= 5].copy() # 過濾樣本數太少的投手

            if not df_pitchers.empty:
                st.subheader("🤖 機器學習分析 (投手端)")
                
                # ---------------------------------
                # A. K-Means 投手角色分群
                # ---------------------------------
                st.markdown("### 1. 投手風格分群 (Clustering)")
                # 使用 K%, BB%, 被全壘打率(HR/9), 滾飛比的替代指標等進行分群
                features = ['k_pct', 'bb_pct', 'opp_babip', 'fip']
                
                # 填補缺失值並標準化
                X = df_pitchers[features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 設定 3 個群集 (強力型、效率型、未知型)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df_pitchers['cluster'] = kmeans.fit_predict(X_scaled)
                
                # 定義群集標籤 (這需要根據實際群集中心點去微調邏輯，此為示意)
                cluster_mapping = {0: "Power Pitchers (強力型)", 1: "Efficiency/Ground (效率型)", 2: "Wild/Struggling (控球不穩型)"}
                df_pitchers['style'] = df_pitchers['cluster'].map(cluster_mapping)
                
                # 畫出分群散佈圖 (K% vs BB%)
                fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df_pitchers, x='k_pct', y='bb_pct', hue='style', s=100, ax=ax_cluster)
                ax_cluster.set_title("Pitcher Styles: K% vs BB%")
                st.pyplot(fig_cluster)
                
                # ---------------------------------
                # B. XGBoost 預期防禦率 (xERA / 預測 ERA)
                # ---------------------------------
                st.markdown("### 2. 防禦率「未爆彈」偵測 (XGBoost xERA)")
                st.info("比較投手的實際 ERA 與機器學習透過進階數據(FIP, K%, BB%, BABIP)算出的預期防禦率，尋找運氣極佳或極差的投手。")
                
                # [訓練模型概念]
                # 在實際應用中，你需要過去幾年的歷史數據來 train XGBoost
                # 這裡示範如何宣告模型並進行預測：
                # model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                # X_train = historical_data[['fip', 'k_pct', 'bb_pct', 'opp_babip']]
                # y_train = historical_data['era_second_half']
                # model.fit(X_train, y_train)
                # df_pitchers['xERA_pred'] = model.predict(df_pitchers[['fip', 'k_pct', 'bb_pct', 'opp_babip']])
                
                # (如果沒有歷史數據，可以先用 FIP 與 ERA 的差值做簡單回歸比較)
                df_pitchers['era_minus_fip'] = df_pitchers['era'] - df_pitchers['fip']
                df_pitchers = df_pitchers.sort_values('era_minus_fip', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error("🚨 運氣太好 (ERA低但FIP高，隨時可能爆發)")
                    st.dataframe(df_pitchers.head(5)[['name_clean', 'teamcode', 'era', 'fip', 'era_minus_fip']])
                with col2:
                    st.success("💎 運氣極差 (ERA高但FIP低，實力其實不錯)")
                    st.dataframe(df_pitchers.tail(5)[['name_clean', 'teamcode', 'era', 'fip', 'era_minus_fip']])