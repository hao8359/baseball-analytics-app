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
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Basic Stats", "🔬 Advanced Analytics", "🤖 ML Analytics", "live bp simulation"])
        
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
            # 1. 取得全聯盟的投手數據 (過濾掉投球局數小於 5 局的樣本)
            df_pitchers_all = load_pitching_data()
            df_pitchers_all = df_pitchers_all[df_pitchers_all['ip_calc'] >= 5].copy() 

            if not df_pitchers_all.empty:
                st.subheader(f"🤖 機器學習分析 (投手端) - {selected_team}")
                
                # ==========================================
                # 先用「全聯盟」的數據來訓練模型，確保準確度
                # ==========================================
                features = ['k_pct', 'bb_pct', 'opp_babip', 'fip']
                X = df_pitchers_all[features].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 執行 K-Means 分群
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df_pitchers_all['cluster'] = kmeans.fit_predict(X_scaled)
                cluster_mapping = {0: "Power Pitchers ", 1: "Efficiency/Ground ", 2: "Wild/Struggling s"}
                df_pitchers_all['style'] = df_pitchers_all['cluster'].map(cluster_mapping)
                
                # 計算運氣指數 (ERA 減去 FIP)
                df_pitchers_all['era_minus_fip'] = df_pitchers_all['era'] - df_pitchers_all['fip']
                
                # ==========================================
                # 這裡最重要：將預測完的結果，過濾出「當前選擇的球隊」
                # ==========================================
                target_id = TEAM_IDS[selected_team]
                df_team_pitchers = df_pitchers_all[df_pitchers_all['teamid'].astype(str) == target_id].copy()

                if df_team_pitchers.empty:
                    st.warning(f"目前 {selected_team} 沒有符合條件 (投球局數 >= 5) 的投手數據。")
                else:
                    st.markdown("### 1. 投手風格分群 (Clustering)")
                    
                    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
                    
                    # 畫出該隊投手的散佈圖
                    sns.scatterplot(data=df_team_pitchers, x='k_pct', y='bb_pct', hue='style', s=150, ax=ax_cluster)
                    
                    # 在點旁邊加上球員名字標籤，方便辨識
                    for _, row in df_team_pitchers.iterrows():
                        ax_cluster.text(row['k_pct'] + 0.005, row['bb_pct'], row['name_clean'], fontsize=10)

                    ax_cluster.set_title(f"Pitcher Styles ({selected_team}): K% vs BB%")
                    # 格式化 X 軸和 Y 軸為百分比
                    ax_cluster.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    ax_cluster.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    st.pyplot(fig_cluster)
                    
                    st.divider()
                    
                    st.markdown("### 2. 防禦率「未爆彈」偵測 (ERA vs FIP)")
                    st.info("比較投手的實際 ERA 與進階獨立防禦率 (FIP)。若 ERA 遠低於 FIP，代表運氣成分居多，未來有爆掉的風險；反之則代表實力被低估。")
                    
                    df_team_pitchers = df_team_pitchers.sort_values('era_minus_fip', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.error("🚨 運氣極佳 / 未爆彈區 (ERA 比 FIP 低)")
                        # 篩選出 ERA 小於 FIP 的投手 (負值)
                        lucky_pitchers = df_team_pitchers[df_team_pitchers['era_minus_fip'] < 0]
                        st.dataframe(lucky_pitchers[['name_clean', 'era', 'fip', 'era_minus_fip']].style.format({
                            'era': '{:.2f}', 'fip': '{:.2f}', 'era_minus_fip': '{:.2f}'
                        }), hide_index=True)
                        
                    with col2:
                        st.success("💎 運氣極差 / 實力被低估 (ERA 比 FIP 高)")
                        # 篩選出 ERA 大於 FIP 的投手 (正值)
                        unlucky_pitchers = df_team_pitchers[df_team_pitchers['era_minus_fip'] >= 0]
                        st.dataframe(unlucky_pitchers[['name_clean', 'era', 'fip', 'era_minus_fip']].style.format({
                            'era': '{:.2f}', 'fip': '{:.2f}', 'era_minus_fip': '{:.2f}'
                        }), hide_index=True)
            with tab4:
                st.subheader("⚔️ 投手 vs 打者：對戰期望值模擬")
                st.info("結合打者的長打能力 (ISO) 與投手的進階獨立數據 (FIP, K%)，預測該打席發生特定結果的機率。")

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
                        st.markdown("### 🦇 選擇打者 (Offense)")
                        b_team = st.selectbox("打者所屬球隊", list(TEAM_IDS.keys()), key="bat_team")
                        b_players = df_batters_filtered[df_batters_filtered['teamid'].astype(str) == TEAM_IDS[b_team]]
                        
                        if b_players.empty:
                            st.warning("該隊目前無符合條件的打者。")
                            selected_batter_name = None
                        else:
                            selected_batter_name = st.selectbox("選擇打者", b_players['name_clean'].tolist())
                            batter_stats = b_players[b_players['name_clean'] == selected_batter_name].iloc[0]

                    with col_sel2:
                        st.markdown("### ⚾ 選擇投手 (Defense)")
                        p_team = st.selectbox("投手所屬球隊", list(TEAM_IDS.keys()), key="pit_team")
                        p_players = df_pitchers_filtered[df_pitchers_filtered['teamid'].astype(str) == TEAM_IDS[p_team]]
                        
                        if p_players.empty:
                            st.warning("該隊目前無符合條件的投手。")
                            selected_pitcher_name = None
                        else:
                            selected_pitcher_name = st.selectbox("選擇投手", p_players['name_clean'].tolist())
                            pitcher_stats = p_players[p_players['name_clean'] == selected_pitcher_name].iloc[0]

                    st.divider()

                    # 如果打者跟投手都有順利選到，就進行計算
                if selected_batter_name and selected_pitcher_name:
                    # 1. 取得聯盟平均值 (作為基準點)
                    lg_avg_k_pct = df_batters_filtered['k_pct'].mean()
                    lg_avg_hr_rate = safe_divide(df_batters_filtered['hr'].sum(), df_batters_filtered['ab'].sum())
                    lg_avg_ba = safe_divide(df_batters_filtered['h'].sum(), df_batters_filtered['ab'].sum()) # 新增：聯盟平均打擊率
                    
                    # 避免極端值或除以 0 的保護機制
                    lg_avg_k_pct = lg_avg_k_pct if lg_avg_k_pct > 0 else 0.20
                    lg_avg_hr_rate = lg_avg_hr_rate if lg_avg_hr_rate > 0 else 0.03
                    lg_avg_ba = lg_avg_ba if lg_avg_ba > 0 else 0.250
                    
                    # 2. 擷取雙方數據
                    b_k = batter_stats['k_pct']
                    b_iso = batter_stats['iso_val']
                    b_avg = batter_stats['avg'] # 新增：打者打擊率
                    
                    p_k = pitcher_stats['k_pct']
                    p_fip = pitcher_stats['fip']
                    p_opp_avg = safe_divide(pitcher_stats['pitch_h'], pitcher_stats['pitch_ab']) # 新增：投手被打擊率
                    
                    # --- 3. 預測公式計算 ---
                    # 三振率預測
                    pred_k = min((b_k * p_k) / lg_avg_k_pct, 0.95) 
                    
                    # 全壘打預測
                    pred_hr = min(max((b_iso * 0.2) * (pitcher_stats['fip'] / 3.15), 0), 0.30)

                    # 新增：安打機率預測 (使用 Bill James 的 Log-5 公式)
                    if lg_avg_ba > 0 and lg_avg_ba < 1:
                        log5_num = (b_avg * p_opp_avg) / lg_avg_ba
                        log5_den = log5_num + ((1 - b_avg) * (1 - p_opp_avg)) / (1 - lg_avg_ba)
                        pred_hit = log5_num / log5_den if log5_den > 0 else 0
                    else:
                        pred_hit = 0

                    st.markdown(f"#### 🏟️ 對決模擬結果：**{selected_batter_name}** vs **{selected_pitcher_name}**")
                    
                    # 擴充為 5 個顯示欄位
                    res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
                    
                    # 三振預測顯示
                    k_diff = pred_k - lg_avg_k_pct
                    res_col1.metric("預測三振機率 (K%)", f"{pred_k:.1%}", f"{k_diff*100:.1f}% vs 聯盟平均", delta_color="inverse")
                    
                    # 新增：安打預測顯示
                    hit_diff = pred_hit - lg_avg_ba
                    res_col2.metric("預測安打機率 (xBA)", f"{pred_hit:.1%}", f"{hit_diff*100:.1f}% vs 聯盟平均", delta_color="normal")
                    
                    # 全壘打預測顯示
                    hr_diff = pred_hr - lg_avg_hr_rate
                    res_col3.metric("預測全壘打機率 (HR%)", f"{pred_hr:.1%}", f"{hr_diff*100:.1f}% vs 聯盟平均", delta_color="normal")
                    
                    # 關鍵對決指標
                    res_col4.metric("打者長打威脅 (ISO)", f"{b_iso:.3f}")
                    res_col5.metric("投手獨立防禦率 (FIP)", f"{p_fip:.2f}")

                    # 進階戰術建議提示 (整合安打預測)
                    st.markdown("---")
                    if pred_hr > 0.08:
                        st.error("🚨 **高風險警告 (長打)**：這名打者對戰此投手的開轟機率極高！若為關鍵時刻，建議考慮**敬遠保送 (IBB)** 或配球盡量閃躲。")
                    elif pred_hit > 0.350:
                        st.warning("⚠️ **高上壘風險 (安打)**：打者擊出安打的期望值很高，內野守備建議稍微退深或針對打者擊球習性佈陣 (Shift)。")
                    elif pred_k > 0.35:
                        st.success("🎯 **三振優勢**：投手在此對決中擁有極高的三振率優勢，可以大膽使用決勝球 (Putaway Pitch) 攻擊好球帶！")
                    else:
                        st.info("⚖️ **勢均力敵**：這是一個標準的投打對決，結果將很大程度取決於當天的臨場控球與擊球掌握度 (Contact)。")