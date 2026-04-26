# ============================================================
# GST Revenue Anomaly Detection — Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GST Anomaly Detector — Tamil Nadu",
                   page_icon="🔍", layout="wide",
                   initial_sidebar_state="collapsed")

EPSILON = 1e-9

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e0e0e0; }
.block-container { padding: 1rem 1rem 2rem 1rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #d0d0d0 !important; }

.hero {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99,179,237,0.25); border-radius: 16px;
    padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero h1 {
    font-size: clamp(1.2rem, 4vw, 1.9rem); font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #f687b3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0; line-height: 1.3;
}
.hero .sub { color: #a0aec0; font-size: clamp(0.75rem, 2vw, 0.88rem); margin: 0; }

.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.75rem; margin-bottom: 1.2rem; }
@media(max-width:640px){ .metric-grid { grid-template-columns: repeat(2,1fr); } }
.metric-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem 0.8rem; text-align: center;
    backdrop-filter: blur(10px); transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 24px rgba(0,0,0,0.3); }
.metric-card .label { font-size: clamp(0.6rem,2vw,0.72rem); font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #718096; margin-bottom: 0.3rem; }
.metric-card .value { font-size: clamp(1.4rem,5vw,2rem); font-weight: 700; line-height: 1; }
.metric-card.total  .value { color: #63b3ed; }
.metric-card.normal .value { color: #68d391; }
.metric-card.suspect .value { color: #fc8181; }
.metric-card.risk   .value { color: #f6ad55; }
.metric-card.revenue .value { color: #b794f4; }

[data-testid="stTabs"] button {
    color: #a0aec0 !important; font-weight: 600;
    font-size: clamp(0.68rem,2.5vw,0.82rem); padding: 0.4rem 0.7rem;
    border-radius: 8px 8px 0 0; white-space: nowrap;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #63b3ed !important; border-bottom: 2px solid #63b3ed !important;
    background: rgba(99,179,237,0.08) !important;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow-x: auto !important; border: 1px solid rgba(255,255,255,0.08); -webkit-overflow-scrolling: touch; }

[data-testid="stRadio"] label, [data-testid="stRadio"] label p,
.stRadio span, [data-baseweb="radio"] span, [data-baseweb="radio"] label,
label, label p { color: #f0f0f0 !important; font-weight: 500 !important; }

[data-testid="stSelectbox"] label, [data-testid="stSelectbox"] label p { color: #f0f0f0 !important; }
[data-testid="stSlider"] label { color: #a0aec0 !important; }
h2, h3 { color: #e2e8f0 !important; font-size: clamp(1rem,3.5vw,1.4rem) !important; }

.stDownloadButton > button {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important; font-weight: 600 !important; width: 100% !important;
}
[data-testid="stAlert"] {
    border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.04) !important; color: #cbd5e0 !important;
}
.insight-card {
    background: rgba(99,179,237,0.07); border-left: 3px solid #63b3ed;
    border-radius: 0 10px 10px 0; padding: 0.9rem 1rem; margin-top: 0.8rem;
    color: #a0aec0; font-size: clamp(0.78rem,2.5vw,0.88rem); line-height: 1.6;
}
.explain-pill {
    display: inline-block; background: rgba(252,129,129,0.12);
    color: #fc8181; border: 1px solid rgba(252,129,129,0.3);
    border-radius: 6px; padding: 3px 8px; font-size: 0.72rem; margin: 2px;
}
@media(max-width:480px){
    [data-testid="stRadio"] > div { flex-direction: column !important; gap: 0.4rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 GST Revenue Anomaly Detection — Tamil Nadu</h1>
    <p class="sub">Multi-source anomaly detection framework that cross-validates declared GST turnover
    against TANGEDCO electricity consumption, EPFO employment data, and e-way bill freight movement
    to identify businesses likely under-reporting their taxable turnover.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Model Settings")
contamination = st.sidebar.slider("Expected % suspicious", 5, 30, 15, 5) / 100
n_estimators  = st.sidebar.slider("Trees (n_estimators)", 50, 300, 100, 50)
zscore_thresh = st.sidebar.slider("Z-score alert threshold (σ)", 1.0, 3.5, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.sidebar.markdown("""
<div style='font-size:0.75rem;color:#718096;line-height:1.7;margin-top:0.5rem'>
<b style='color:#a0aec0'>Required columns:</b><br>
gstin · pan · tangedco_account<br>
epfo_id · metrowater_id<br>
business_name · district<br>
nic_code · industry_type<br>
declared_turnover · electricity_units<br>
employee_count · freight_movement<br>
water_consumption · gstr_mismatch_pct<br>
eway_bill_count · eway_bill_value<br>
<i>(audit_outcome optional)</i>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/tamilnadu_gst_data.csv')

df_raw = pd.read_csv(uploaded_file) if uploaded_file else load_data()

# ── Pipeline ──────────────────────────────────────────────────
@st.cache_data
def run_pipeline(data, contamination, n_estimators):
    df = data.copy()

    # ── Safe defaults for optional columns ──
    for col, default in [
        ('water_consumption',  0.0),
        ('gstr_mismatch_pct',  0.0),
        ('eway_bill_count',    0),
        ('eway_bill_value',    0.0),
        ('metrowater_id',      'UNKNOWN'),
        ('audit_outcome',      'NOT_AUDITED'),
        ('true_evader',        False),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── Column name aliases (handle different naming conventions) ──
    col_aliases = {
        'turnover':           'declared_turnover',
        'declared_turnover':  'declared_turnover',
        'electricity':        'electricity_units',
        'electricity_units':  'electricity_units',
        'employees':          'employee_count',
        'employee_count':     'employee_count',
        'freight':            'freight_movement',
        'freight_movement':   'freight_movement',
        'water':              'water_consumption',
        'water_consumption':  'water_consumption',
        'gstr_mismatch':      'gstr_mismatch_pct',
        'gstr_mismatch_pct':  'gstr_mismatch_pct',
        'eway_value':         'eway_bill_value',
        'eway_bill_value':    'eway_bill_value',
        'eway_count':         'eway_bill_count',
        'eway_bill_count':    'eway_bill_count',
        'gstin':              'gstin',
        'business':           'business_name',
        'business_name':      'business_name',
        'district':           'district',
        'industry':           'industry_type',
        'industry_type':      'industry_type',
        'nic':                'nic_code',
        'nic_code':           'nic_code',
    }
    df.columns = [col_aliases.get(c.lower().strip(), c.lower().strip()) for c in df.columns]

    # ── Ensure nic_code and industry_type exist ──
    if 'nic_code' not in df.columns:
        df['nic_code'] = 'UNKNOWN'
    if 'industry_type' not in df.columns:
        df['industry_type'] = 'Unknown'
    if 'business_name' not in df.columns:
        df['business_name'] = df.get('gstin', pd.Series(range(len(df)))).astype(str)
    if 'gstin' not in df.columns:
        df['gstin'] = ['BIZ' + str(i).zfill(3) for i in range(len(df))]
    if 'district' not in df.columns:
        df['district'] = 'Unknown'

    # ── Safe defaults for ALL required numeric columns ──
    for col, default in [
        ('declared_turnover',  0.0),
        ('electricity_units',  0.0),
        ('employee_count',     0.0),
        ('freight_movement',   0.0),
        ('water_consumption',  0.0),
        ('gstr_mismatch_pct',  0.0),
        ('eway_bill_count',    0.0),
        ('eway_bill_value',    0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    numeric_cols = ['declared_turnover','electricity_units','employee_count',
                    'freight_movement','water_consumption','gstr_mismatch_pct',
                    'eway_bill_count','eway_bill_value']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median() if df[col].notna().any() else 0, inplace=True)

    df['electricity_to_turnover_ratio'] = df['electricity_units']  / (df['declared_turnover'] + EPSILON)
    df['employees_to_turnover_ratio']   = df['employee_count']     / (df['declared_turnover'] + EPSILON)
    df['freight_to_turnover_ratio']     = df['freight_movement']   / (df['declared_turnover'] + EPSILON)
    df['water_to_turnover_ratio']       = df['water_consumption']  / (df['declared_turnover'] + EPSILON)
    df['eway_value_to_turnover_ratio']  = df['eway_bill_value']    / (df['declared_turnover'] + EPSILON)
    # gstr_mismatch_pct used directly as a feature

    ratio_cols = [
        'electricity_to_turnover_ratio',
        'employees_to_turnover_ratio',
        'freight_to_turnover_ratio',
        'water_to_turnover_ratio',
        'eway_value_to_turnover_ratio',
        'gstr_mismatch_pct',
    ]

    for col in ratio_cols:
        gm = df.groupby('nic_code')[col].transform('mean')
        gs = df.groupby('nic_code')[col].transform('std').replace(0, EPSILON)
        df[f'{col}_zscore'] = (df[col] - gm) / gs

    zscore_cols  = [c + '_zscore' for c in ratio_cols]
    feature_cols = ['declared_turnover'] + zscore_cols

    df['anomaly_label'] = 1
    df['anomaly_score'] = 0.0
    for nic, grp in df.groupby('nic_code'):
        if len(grp) < 3: continue
        # Class-imbalance aware contamination
        local_cont = max(0.05, min(0.40, contamination))
        X = StandardScaler().fit_transform(grp[feature_cols].fillna(0))
        m = IsolationForest(n_estimators=n_estimators, contamination=local_cont, random_state=42)
        m.fit(X)
        df.loc[grp.index, 'anomaly_label'] = m.predict(X)
        df.loc[grp.index, 'anomaly_score'] = m.score_samples(X)

    raw = df['anomaly_score']
    df['risk_score'] = (100*(1-(raw-raw.min())/(raw.max()-raw.min()+EPSILON))).round(2)
    df['status'] = df['anomaly_label'].map({1:'✅ Normal', -1:'🚨 Suspicious'})

    def explain(row):
        if row['anomaly_label'] == 1:
            return "No significant anomaly"
        r = []
        t = 2.0
        checks = [
            ('electricity_to_turnover_ratio_zscore', '⚡ Electricity'),
            ('employees_to_turnover_ratio_zscore',   '👥 Employees'),
            ('freight_to_turnover_ratio_zscore',     '🚚 Freight'),
            ('water_to_turnover_ratio_zscore',       '💧 Water usage'),
            ('eway_value_to_turnover_ratio_zscore',  '📦 E-way bill value'),
        ]
        for col, label in checks:
            if col in row and row[col] > t:
                r.append(f"{label} {row[col]:.1f}σ above industry avg")
        if row.get('gstr_mismatch_pct', 0) > 10:
            r.append(f"📋 GSTR mismatch {row['gstr_mismatch_pct']:.1f}%")
        return " | ".join(r) if r else "Overall pattern inconsistent with declared turnover"

    df['explanation'] = df.apply(explain, axis=1)
    return df

df = run_pipeline(df_raw, contamination, n_estimators)

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.03)',
    font=dict(family='Inter', color='#a0aec0', size=11),
    title_font=dict(color='#e2e8f0', size=14),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    margin=dict(l=40, r=20, t=50, b=60), autosize=True,
)
CMAP = {'✅ Normal':'#68d391', '🚨 Suspicious':'#fc8181'}

# ── Metrics ───────────────────────────────────────────────────
total     = len(df)
susp      = int((df['anomaly_label']==-1).sum())
norm      = int((df['anomaly_label']==1).sum())
avg_risk  = df[df['anomaly_label']==-1]['risk_score'].mean()
est_rev   = f"₹{susp * 50:,}L"   # ₹50L avg per evader (conservative)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card total">  <div class="label">Total Businesses</div><div class="value">{total}</div></div>
  <div class="metric-card normal"> <div class="label">✅ Normal</div>       <div class="value">{norm}</div></div>
  <div class="metric-card suspect"><div class="label">🚨 Suspicious</div>   <div class="value">{susp}</div></div>
  <div class="metric-card risk">   <div class="label">Avg Risk Score</div>  <div class="value">{avg_risk:.1f}</div></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Overview", "📋 Audit List", "📊 Risk Charts",
    "🔬 Industry Analysis", "📈 Model Evaluation", "📥 Download"
])

# ── Tab 1: Overview ───────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        dist_risk = df[df['district'] != 'Unknown'].groupby('district')['risk_score'].mean().reset_index().sort_values('risk_score', ascending=False)
        if dist_risk.empty:
            dist_risk = df.groupby('district')['risk_score'].mean().reset_index().sort_values('risk_score', ascending=False)
        fig = px.bar(dist_risk, x='district', y='risk_score', color='risk_score',
                     color_continuous_scale='RdYlGn_r', title='Average Risk Score by District',
                     labels={'risk_score':'Avg Risk Score','district':'District'})
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        ind_susp = df[df['anomaly_label']==-1].groupby('industry_type').size().reset_index(name='count').sort_values('count', ascending=False)
        fig2 = px.bar(ind_susp, x='industry_type', y='count', color='count',
                      color_continuous_scale='Reds', title='Suspicious Businesses by Industry',
                      labels={'industry_type':'Industry','count':'Count'})
        fig2.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    <b style="color:#63b3ed">Context:</b><br>
    Tamil Nadu has ~15 lakh GST-registered dealers generating ₹60,000 crore+ annual SGST revenue.
    Conservative estimates suggest <b style="color:#fc8181">10–15% of potential GST revenue is lost to evasion annually</b>.
    Current manual audit covers only 0.5–1% of businesses. This system enables data-driven prioritisation
    across all registered dealers using multi-source operational signals.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Audit List ─────────────────────────────────────────
with tab2:
    st.markdown("### 🚨 Prioritised Audit List")
    filter_opt = st.radio("Filter", ["All", "🚨 Suspicious Only", "✅ Normal Only"], horizontal=True)
    dist_filter = st.multiselect("Filter by District", sorted(df['district'].dropna().unique()), default=[])
    ind_filter  = st.multiselect("Filter by Industry", sorted(df['industry_type'].dropna().unique()), default=[])

    display = df.copy()
    if "Suspicious" in filter_opt: display = display[display['anomaly_label']==-1]
    elif "Normal"    in filter_opt: display = display[display['anomaly_label']==1]
    if dist_filter: display = display[display['district'].isin(dist_filter)]
    if ind_filter:  display = display[display['industry_type'].isin(ind_filter)]

    show_cols = ['gstin','business_name','district','industry_type',
                 'declared_turnover','risk_score','status','explanation']
    styled = (
        display[show_cols].sort_values('risk_score', ascending=False).reset_index(drop=True)
        .style
        .background_gradient(subset=['risk_score'], cmap='RdYlGn_r', vmin=0, vmax=100)
        .format({'declared_turnover':'₹{:,.0f}', 'risk_score':'{:.1f}'})
        .map(lambda v: 'color:#fc8181;font-weight:600' if '🚨' in str(v)
                  else 'color:#68d391;font-weight:600', subset=['status'])
    )
    st.dataframe(styled, use_container_width=True, height=450)

# ── Tab 3: Risk Charts ────────────────────────────────────────
with tab3:
    fig_bar = px.bar(
        df.sort_values('risk_score', ascending=False).head(40),
        x='business_name', y='risk_score', color='status',
        color_discrete_map=CMAP, text='risk_score',
        title='Top 40 Businesses by Risk Score',
        labels={'business_name':'Business','risk_score':'Risk Score'}
    )
    fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside', textfont_size=9)
    fig_bar.add_hline(y=50, line_dash='dot', line_color='#f6ad55',
                      annotation_text='Threshold', annotation_font_color='#f6ad55')
    fig_bar.update_layout(**PLOT_LAYOUT, xaxis_tickangle=-45, bargap=0.2)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    signal = st.selectbox("Scatter signal", [
        'electricity_units','employee_count','freight_movement',
        'water_consumption','gstr_mismatch_pct','eway_bill_value'
    ], format_func=lambda x: x.replace('_',' ').title())
    fig_sc = px.scatter(df, x='declared_turnover', y=signal, color='status',
                        color_discrete_map=CMAP, size='risk_score', size_max=25,
                        hover_data=['business_name','district','industry_type','risk_score'],
                        title=f'{signal.replace("_"," ").title()} vs Declared Turnover',
                        labels={'declared_turnover':'Declared Turnover (₹)'})
    fig_sc.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 4: Industry Analysis ──────────────────────────────────
with tab4:
    st.markdown("### Industry-Stratified Z-Score Analysis")
    st.markdown("""
    <div class="insight-card">
    <b style="color:#63b3ed">Why industry stratification matters:</b>
    A software company and a steel foundry with the same declared turnover have very different
    electricity consumption. Raw ratios are misleading — Z-scores within each NIC code group
    reveal true anomalies relative to industry peers.
    </div>
    """, unsafe_allow_html=True)

    # ── Scatter: 2 clean traces, no symbol clutter ──
    fig_z = go.Figure()
    for label, color, name in [('✅ Normal','#68d391','Normal'), ('🚨 Suspicious','#fc8181','Suspicious')]:
        s = df[df['status'] == label]
        fig_z.add_trace(go.Scatter(
            x=s['electricity_to_turnover_ratio_zscore'],
            y=s['freight_to_turnover_ratio_zscore'],
            mode='markers',
            name=name,
            marker=dict(
                color=color,
                size=(s['risk_score'].clip(lower=8) / 5 + 8).tolist(),
                opacity=0.88,
                line=dict(color='rgba(255,255,255,0.25)', width=1)
            ),
            customdata=s[['business_name','district','industry_type','risk_score']].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "District: %{customdata[1]}<br>"
                "Industry: %{customdata[2]}<br>"
                "Risk Score: %{customdata[3]:.1f}<br>"
                "Elec Z: %{x:.2f} | Freight Z: %{y:.2f}"
                "<extra></extra>"
            )
        ))
    fig_z.add_hline(y=zscore_thresh, line_dash='dot', line_color='#f6ad55',
                    annotation_text=f'{zscore_thresh}σ', annotation_font_color='#f6ad55',
                    annotation_position='top right')
    fig_z.add_vline(x=zscore_thresh, line_dash='dot', line_color='#f6ad55')
    LEGEND_STYLE = dict(bgcolor='rgba(255,255,255,0.06)', bordercolor='rgba(255,255,255,0.15)',
                        borderwidth=1, font=dict(color='#e2e8f0', size=13),
                        orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    layout_z = {k: v for k, v in PLOT_LAYOUT.items() if k != 'legend'}
    fig_z.update_layout(
        **layout_z,
        title='Industry-Adjusted Signal Mismatch (Z-scores)',
        xaxis_title='Electricity / Turnover Z-score',
        yaxis_title='Freight / Turnover Z-score',
        legend=LEGEND_STYLE
    )
    st.plotly_chart(fig_z, use_container_width=True)

    # ── Box plot per industry ──
    ratio_choice = st.selectbox("Select ratio", [
        'electricity_to_turnover_ratio_zscore',
        'employees_to_turnover_ratio_zscore',
        'freight_to_turnover_ratio_zscore',
        'water_to_turnover_ratio_zscore',
        'eway_value_to_turnover_ratio_zscore',
        'gstr_mismatch_pct_zscore',
    ], format_func=lambda x: x.replace('_zscore','').replace('_',' ').title())

    fig_box = px.box(df, x='industry_type', y=ratio_choice, color='status',
                     color_discrete_map=CMAP, points='outliers',
                     title=f'{ratio_choice.replace("_zscore","").replace("_"," ").title()} by Industry',
                     labels={'industry_type':'Industry'})
    layout_box = {k: v for k, v in PLOT_LAYOUT.items() if k != 'legend'}
    fig_box.update_layout(**layout_box, xaxis_tickangle=-30, legend=LEGEND_STYLE)
    st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 5: Model Evaluation ───────────────────────────────────
with tab5:
    st.markdown("### 📈 Evaluation Against Historical Audit Outcomes")
    audited = df[df['audit_outcome'] != 'NOT_AUDITED'].copy()

    if len(audited) > 0:
        audited['true_label'] = (audited['audit_outcome'] == 'EVADER').astype(int)
        audited['pred_label'] = (audited['anomaly_label'] == -1).astype(int)

        prec = precision_score(audited['true_label'], audited['pred_label'], zero_division=0)
        rec  = recall_score(audited['true_label'], audited['pred_label'], zero_division=0)
        cm   = confusion_matrix(audited['true_label'], audited['pred_label'])

        m1, m2, m3 = st.columns(3)
        m1.metric("Audit Hit Rate (Precision)", f"{prec:.1%}")
        m2.metric("Evader Catch Rate (Recall)", f"{rec:.1%}")
        m3.metric("Audited Businesses", len(audited))

        # Confusion matrix heatmap
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=['Predicted Clean','Predicted Evader'],
            y=['Actual Clean','Actual Evader'],
            colorscale='RdYlGn_r', showscale=False,
            text=cm, texttemplate='%{text}', textfont=dict(size=18, color='white')
        ))
        fig_cm.update_layout(title='Confusion Matrix (Audited Subset)',
                              paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#a0aec0'), title_font_color='#e2e8f0',
                              margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        <div class="insight-card">
        <b style="color:#63b3ed">Success Metric:</b><br>
        The system evaluates <b>audit hit rate improvement</b> — the proportion of flagged businesses
        found to have under-declared turnover, compared to the current manual selection method's hit rate
        of ~0.5–1%. Higher precision = more efficient use of limited audit resources.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No audited businesses in this dataset. Upload data with 'audit_outcome' column to see evaluation metrics.")

# ── Tab 6: Download ───────────────────────────────────────────
with tab6:
    st.markdown("### 📥 Export Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Results as CSV", csv, 'gst_anomaly_results.csv', 'text/csv')

    top_audit = df[df['anomaly_label']==-1][['gstin','business_name','district','industry_type','declared_turnover','risk_score','explanation']].sort_values('risk_score', ascending=False)
    audit_csv = top_audit.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Audit Priority List", audit_csv, 'audit_priority_list.csv', 'text/csv')

    st.markdown("#### Preview — Top 10 Priority Audit Cases")
    st.dataframe(top_audit.head(10).reset_index(drop=True), use_container_width=True)

    st.markdown("""
    <div class="insight-card">
    <b style="color:#63b3ed">Strategic Impact:</b><br>
    A 10% improvement in audit hit rate would translate to
    <b style="color:#b794f4">₹500–1,000 crore of additional annual revenue recovery</b> for Tamil Nadu.
    This system enables systematic coverage of all 15 lakh registered dealers vs. the current
    manual 0.5–1% sample.
    </div>
    """, unsafe_allow_html=True)
