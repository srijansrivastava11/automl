import pickle, warnings, io, json, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    mean_absolute_error, mean_absolute_percentage_error,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ═══════════════════════════════════════════
# PAGE CONFIG & STYLES
# ═══════════════════════════════════════════
st.set_page_config(page_title="AutoML POC v1.0", page_icon="⚡", layout="wide")
st.markdown("""
<style>
    .title{font-size:2.3rem;font-weight:800;color:#0f172a;text-align:center;margin-bottom:0;letter-spacing:-0.5px}
    .sub{text-align:center;color:#64748b;margin-bottom:1.2rem;font-size:1rem}
    .pill{display:inline-block;padding:5px 14px;border-radius:999px;font-size:0.82rem;margin:2px 4px;font-weight:600}
    .good{background:#dcfce7;color:#166534}.warn{background:#fef9c3;color:#854d0e}.bad{background:#fee2e2;color:#991b1b}
    .info-pill{background:#dbeafe;color:#1e40af}
    .kpi-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04)}
    .kpi-val{font-size:1.5rem;font-weight:700;color:#0f172a;margin:0}
    .kpi-label{font-size:0.78rem;color:#64748b;margin:0;text-transform:uppercase;letter-spacing:0.5px}
    .tooltip-box{background:#f0f9ff;border-left:3px solid #3b82f6;padding:0.6rem 1rem;border-radius:0 8px 8px 0;margin:0.5rem 0;font-size:0.88rem;color:#1e3a5f}
    div[data-testid="stMetricValue"]{font-size:1.3rem}
    .stProgress>div>div>div>div{background:linear-gradient(90deg,#3b82f6,#8b5cf6)}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">⚡ AutoML POC — v1.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload → Profile → KPI → Clean → Anomalies → Model → Evaluate → Ask AI  → Download</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SETTINGS (inline)
# ═══════════════════════════════════════════
with st.expander("⚙️ Configuration & Quick Guide", expanded=False):
    cfg_l, cfg_r = st.columns(2)
    with cfg_l:
        speed_mode = st.radio("Speed Mode", ["Fast","Balanced"], index=0, horizontal=True,
                              help="**Fast**=smaller trees, capped previews. **Balanced**=larger models, PSI on.")
    with cfg_r:
        st.markdown("**Steps:** Upload → Profile → KPI → Clean → Anomalies → Model → Evaluate → AI Chat → Download")
    st.caption("v1.0 — JSON Expand + %/$/ Currency Parsing + Claude AI Chat")

is_fast = (speed_mode == "Fast")
MAX_PREVIEW = 200 if is_fast else 400
HEATMAP_CAP = 25 if is_fast else 40
RF_TREES = 100 if is_fast else 250
DT_DEPTH = 10 if is_fast else 12
PSI_DEFAULT = not is_fast
CV_DEFAULT = False

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════
def make_unique_columns(cols):
    seen, new_cols, mapping = {}, [], []
    for c in list(cols):
        base = str(c).strip()
        if base not in seen:
            seen[base] = 1; new_cols.append(base); mapping.append((base, base))
        else:
            seen[base] += 1; nn = "{}__{}".format(base, seen[base]); new_cols.append(nn); mapping.append((base, nn))
    return new_cols, mapping

def safe_csv(df): return df.to_csv(index=False).encode("utf-8")

def _sanitize_for_display(df):
    """Fix mixed-type columns that crash pyarrow/st.dataframe."""
    d=df.copy()
    for c in d.columns:
        if d[c].dtype=="object":
            # Check for mixed types (e.g., datetime + string in same column)
            types=set(type(v).__name__ for v in d[c].dropna().head(50))
            if len(types)>1:
                d[c]=d[c].astype(str)
    return d

def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
    return "{:.2f}%".format(val)

def fmt_num(val, d=2):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
    if abs(val) >= 1e6: return "{:,.0f}".format(val)
    return "{:,.{}f}".format(val, d)

# ─── Loaders ───
@st.cache_data(show_spinner=False)
def load_csv(f):
    try: return pd.read_csv(f)
    except UnicodeDecodeError:
        f.seek(0); return pd.read_csv(f, encoding="latin-1")

@st.cache_data(show_spinner=False)
def get_excel_sheets(f): return pd.ExcelFile(f).sheet_names

@st.cache_data(show_spinner=False)
def load_excel(f, sheet=0): return pd.read_excel(f, sheet_name=sheet)

def load_file(uf, sheet=None):
    n = uf.name.lower()
    if n.endswith(".csv"): return load_csv(uf)
    if n.endswith((".xlsx", ".xls")): return load_excel(uf, sheet or 0)
    raise ValueError("Unsupported file type.")

# ─── % / $ / Currency cleaning ───
def _clean_numeric_string(series):
    """
    Strips %, $, €, £, commas, spaces, accounting parens from string values.
    '45.2%' → 45.2 | '$1,200' → 1200 | '(500)' → -500
    """
    s = series.astype(str).str.strip()
    had_pct = s.str.contains('%', na=False).any()
    # Remove currency symbols and whitespace
    s = s.str.replace(r'[\$€£¥₹₩₿\s]', '', regex=True)
    # Remove percent sign
    s = s.str.replace('%', '', regex=False)
    # Accounting negative: (123) → -123
    paren_mask = s.str.fullmatch(r'\([\d,\.]+\)', na=False)
    s = s.str.replace(r'[()]', '', regex=True)
    # Remove commas
    s = s.str.replace(',', '', regex=False)
    # Apply negative for parenthesised values
    s = s.where(~paren_mask, '-' + s)
    return s, had_pct

# ─── JSON expansion ───
def _is_json_like(series, sample_n=50):
    sample = series.dropna().head(sample_n).astype(str)
    if len(sample) == 0: return False
    ct = 0
    for v in sample:
        v = v.strip()
        if (v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']')):
            try: json.loads(v); ct += 1
            except: pass
    return ct >= max(1, len(sample) * 0.3)

def expand_json_columns(df_in):
    """Detect & expand JSON-string columns into flat columns."""
    df = df_in.copy(); expansions = []
    for col in df.select_dtypes(include=["object"]).columns.tolist():
        if not _is_json_like(df[col]): continue
        rows = []
        for val in df[col]:
            if pd.isna(val): rows.append({}); continue
            try:
                obj = json.loads(str(val).strip())
                if isinstance(obj, dict): rows.append(obj)
                elif isinstance(obj, list): rows.append({"{}_{}".format(col, i): v for i, v in enumerate(obj)})
                else: rows.append({})
            except: rows.append({})
        exp = pd.json_normalize(rows, sep="_")
        if exp.shape[1] == 0: continue
        renames = {nc: "{}_{}".format(col, nc) for nc in exp.columns}
        exp = exp.rename(columns=renames); exp.index = df.index
        for nc in exp.columns: df[nc] = exp[nc]
        expansions.append((col, list(exp.columns)))
        df = df.drop(columns=[col])
    return df, expansions

# ─── Type conversion (NO @st.cache_data — avoids stale cache bugs) ───
def auto_convert_types(df_in, dt_thresh=0.5, num_thresh=0.5):
    """
    Auto-detect and convert object columns to datetime or numeric.
    Handles: '45%', '$1,200', '(500)', '€3.500', plain numbers.
    Returns: (df, changes_list, pct_columns_list)
    """
    df = df_in.copy(); changes = []; pct_cols = []
    for col in df.columns:
        if df[col].dtype != "object": continue
        orig = "object"
        # 1) Try datetime
        try:
            conv = pd.to_datetime(df[col], errors="coerce")
            if conv.notna().sum() >= dt_thresh * len(df):
                df[col] = conv; changes.append((col, orig, "datetime64[ns]")); continue
        except: pass
        # 2) Try numeric WITH % / $ / currency cleaning
        try:
            cleaned, had_pct = _clean_numeric_string(df[col])
            conv = pd.to_numeric(cleaned, errors="coerce")
            if conv.notna().sum() >= num_thresh * len(df):
                df[col] = conv
                tag = str(conv.dtype) + (" (was %)" if had_pct else "")
                changes.append((col, orig, tag))
                if had_pct: pct_cols.append(col)
                continue
        except: pass
    return df, changes, pct_cols

# ─── Missing values ───
def handle_missing(df, strategy):
    d = df.copy()
    num = d.select_dtypes(include=np.number).columns.tolist()
    cat = d.select_dtypes(include=["object","category"]).columns.tolist()
    if strategy == "Drop rows": return d.dropna()
    if strategy == "Forward Fill": return d.ffill()
    if strategy == "Backward Fill": return d.bfill()
    if strategy in ("Mean + Mode","Median + Mode"):
        fn = "mean" if strategy.startswith("Mean") else "median"
        for c in num: d[c] = d[c].fillna(getattr(d[c], fn)())
        for c in cat:
            mode = d[c].mode(); d[c] = d[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        return d
    return d

# ─── Profiling ───
@st.cache_data(show_spinner=False)
def compute_profile(df):
    rows = []
    for col in df.columns:
        s = df[col]; nc = int(s.isna().sum()); npct = round(nc/max(len(s),1)*100,1)
        row = {"Column":col,"Type":str(s.dtype),"Nulls":nc,"Null %":npct,"Unique":int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            row.update({"Mean":round(float(d.get("mean",0)),2),"Std":round(float(d.get("std",0)),2),
                        "Min":round(float(d.get("min",0)),2),"Max":round(float(d.get("max",0)),2),
                        "Skew":round(float(s.skew()),2) if len(s)>2 else 0})
        else: row.update({"Mean":None,"Std":None,"Min":None,"Max":None,"Skew":None})
        rows.append(row)
    return pd.DataFrame(rows)

# ─── KPI Insights ───
@st.cache_data(show_spinner=False)
def compute_kpis(df):
    kpis = []
    num_c = df.select_dtypes(include=np.number).columns.tolist()
    cat_c = df.select_dtypes(include=["object","category"]).columns.tolist()
    dt_c  = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    tc = len(df)*len(df.columns)
    comp = round((1-df.isna().sum().sum()/max(tc,1))*100,1)
    kpis += [{"cat":"Overview","name":"Records","val":"{:,}".format(len(df)),"icon":"📊"},
             {"cat":"Overview","name":"Features","val":str(len(df.columns)),"icon":"📐"},
             {"cat":"Overview","name":"Completeness","val":"{}%".format(comp),"icon":"✅"},
             {"cat":"Overview","name":"Duplicates","val":"{:,}".format(int(df.duplicated().sum())),"icon":"🔄"}]
    for col in num_c[:6]:
        s = df[col].dropna()
        if len(s)==0: continue
        t,a,m,sd = float(s.sum()),float(s.mean()),float(s.median()),float(s.std())
        kpis += [{"cat":"Numeric — "+col,"name":"Sum","val":fmt_num(t,0),"icon":"➕"},
                 {"cat":"Numeric — "+col,"name":"Avg","val":fmt_num(a),"icon":"📈"},
                 {"cat":"Numeric — "+col,"name":"Median","val":fmt_num(m),"icon":"📊"},
                 {"cat":"Numeric — "+col,"name":"Std","val":fmt_num(sd),"icon":"📉"}]
        if a!=0: kpis.append({"cat":"Numeric — "+col,"name":"CV%","val":"{}%".format(round(abs(sd/a)*100,1)),"icon":"🔀"})
    for col in cat_c[:4]:
        s=df[col].dropna()
        if len(s)==0: continue
        vc=s.value_counts(); tv,tc2=str(vc.index[0]),int(vc.iloc[0])
        kpis += [{"cat":"Cat — "+col,"name":"Top","val":tv,"icon":"🏷️"},
                 {"cat":"Cat — "+col,"name":"Freq","val":"{:,} ({:.1f}%)".format(tc2,tc2/len(s)*100),"icon":"📊"},
                 {"cat":"Cat — "+col,"name":"Cardinality","val":str(int(s.nunique())),"icon":"🔢"}]
    for col in dt_c[:2]:
        s=df[col].dropna()
        if len(s)==0: continue
        kpis += [{"cat":"Time — "+col,"name":"Range","val":"{} → {}".format(s.min().strftime("%Y-%m-%d"),s.max().strftime("%Y-%m-%d")),"icon":"📅"},
                 {"cat":"Time — "+col,"name":"Days","val":"{:,}".format((s.max()-s.min()).days),"icon":"⏱️"}]
    if len(num_c)>=2:
        try:
            cr=df[num_c[:20]].corr().abs(); np.fill_diagonal(cr.values,0); mx=cr.max().max()
            if mx>0.3:
                idx=cr.stack().idxmax()
                kpis += [{"cat":"Correlation","name":"Top Pair","val":"{} ↔ {}".format(idx[0],idx[1]),"icon":"🔗"},
                         {"cat":"Correlation","name":"r","val":"{:.3f}".format(mx),"icon":"📐"}]
        except: pass
    return kpis

# ─── Anomaly ───
def zscore_flags(s,z=3.0):
    s2=pd.to_numeric(s,errors="coerce"); mu,sd=s2.mean(),s2.std(ddof=0)
    if sd==0 or np.isnan(sd): return pd.Series(False,index=s.index)
    return ((s2-mu)/sd).abs()>z
def iqr_flags(s,k=1.5):
    s2=pd.to_numeric(s,errors="coerce"); q1,q3=s2.quantile(0.25),s2.quantile(0.75); iq=q3-q1
    if iq==0 or np.isnan(iq): return pd.Series(False,index=s.index)
    return (s2<q1-k*iq)|(s2>q3+k*iq)
def apply_anomaly_action(df,mask,action,scope=None):
    d=df.copy()
    if scope is None: scope=d.select_dtypes(include=np.number).columns.tolist()
    if action=="Remove rows": return d.loc[~mask].copy()
    if action=="Keep": return d
    for c in scope:
        if c in d.columns and pd.api.types.is_numeric_dtype(d[c]): d.loc[mask,c]=np.nan
    if action=="Forward Fill": return d.ffill()
    if action=="Backward Fill": return d.bfill()
    for c in scope:
        if c in d.columns and pd.api.types.is_numeric_dtype(d[c]):
            d[c]=d[c].fillna(d[c].mean() if action=="Mean" else d[c].median())
    return d
@st.cache_data(show_spinner=False)
def run_isoforest(ndf,cont):
    X=ndf.copy()
    for c in X.columns: X[c]=X[c].fillna(X[c].median())
    return IsolationForest(contamination=cont,random_state=42,n_jobs=-1).fit_predict(X)

# ─── Modeling ───
def regression_metrics(yt,yp):
    return {"R2":float(r2_score(yt,yp)),"RMSE":float(np.sqrt(mean_squared_error(yt,yp))),
            "MAE":float(mean_absolute_error(yt,yp)),"MAPE":float(mean_absolute_percentage_error(yt,yp)*100)}
def ma_forecast(yt,n,w=7): return np.array([float(pd.Series(yt).tail(min(w,len(yt))).mean())]*n)
def psi_numeric(tr,te,bins=10):
    t2=pd.to_numeric(tr,errors="coerce").dropna(); e2=pd.to_numeric(te,errors="coerce").dropna()
    if t2.empty or e2.empty: return np.nan
    edges=np.unique(t2.quantile(np.linspace(0,1,bins+1)).values)
    if len(edges)<3: return np.nan
    tc=np.histogram(t2,bins=edges)[0].astype(float); ec=np.histogram(e2,bins=edges)[0].astype(float)
    tp=np.clip(tc/tc.sum(),1e-6,None); ep=np.clip(ec/ec.sum(),1e-6,None)
    return float(np.sum((ep-tp)*np.log(ep/tp)))

# ─── LLM Chat (RAG) ───
def build_data_context(df, max_rows=30):
    """Build a concise data summary for LLM context."""
    lines = []
    lines.append("DATASET SUMMARY: {} rows × {} columns".format(len(df), len(df.columns)))
    lines.append("\nCOLUMN TYPES:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())
        uniq = int(df[col].nunique(dropna=True))
        lines.append("  {} — type={}, nulls={}, unique={}".format(col, dtype, nulls, uniq))

    # Numeric stats
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        lines.append("\nNUMERIC STATISTICS:")
        desc = df[num_cols].describe().round(2).to_string()
        lines.append(desc)

    # Categorical value counts (top 5 per col)
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        lines.append("\nCATEGORICAL TOP VALUES:")
        for col in cat_cols[:8]:
            vc = df[col].value_counts().head(5)
            vals = ", ".join(["{}: {}".format(k, v) for k, v in vc.items()])
            lines.append("  {} → {}".format(col, vals))

    # Sample rows
    lines.append("\nSAMPLE DATA (first {} rows):".format(min(max_rows, len(df))))
    lines.append(df.head(max_rows).to_string(max_colwidth=40))

    return "\n".join(lines)

def ask_claude(question, data_context, model_context, api_key):
    """Send question + data context to Claude API."""
    client = Anthropic(api_key=api_key)
    system_prompt = """You are an expert data analyst assistant. The user has uploaded a dataset and may have trained ML models on it.
Answer questions about the data using the context provided. Be specific with numbers, column names, and insights.
If the user asks about model results, refer to the model context. Be concise and actionable.
Format numbers nicely. If you're not sure about something, say so."""

    user_msg = """DATA CONTEXT:
{}

{}

USER QUESTION: {}""".format(data_context, model_context, question)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}]
    )
    return response.content[0].text


# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════
for k in ["raw_df","df_clean","df_final","type_changes","pct_cols","json_expansions",
          "results_df","predictions_df","best_bundle_bytes","best_name","preds_store",
          "task_type","trained_models","chat_history"]:
    if k not in st.session_state: st.session_state[k] = None
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ═══════════════════════════════════════════
# STEP 1 — UPLOAD
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 1️⃣ Upload Data")
st.markdown('<div class="tooltip-box">📌 <b>What to do:</b> Upload a CSV or Excel file (.csv .xlsx .xls). The tool auto-detects columns, types, JSON, and percentage/currency strings.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv","xlsx","xls"],
                                 help="Drag & drop or click to browse.")
if uploaded_file is None:
    st.info("👆 Upload a file to begin."); st.stop()

sheet_choice = None
if uploaded_file.name.lower().endswith((".xlsx",".xls")):
    try: sheets = get_excel_sheets(uploaded_file); sheet_choice = st.selectbox("📑 Sheet", sheets, index=0)
    except Exception as e: st.error(str(e)); st.stop()

try: raw_df = load_file(uploaded_file, sheet=sheet_choice)
except Exception as e: st.error(str(e)); st.stop()

# Drop 'Unnamed' index columns from bad CSV exports
unnamed_cols=[c for c in raw_df.columns if str(c).startswith("Unnamed")]
if unnamed_cols:
    raw_df=raw_df.drop(columns=unnamed_cols)

had_dupes = raw_df.columns.duplicated().any()
new_cols, col_map = make_unique_columns(raw_df.columns)
raw_df.columns = new_cols; st.session_state.raw_df = raw_df

c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows","{:,}".format(raw_df.shape[0]))
c2.metric("Columns",str(raw_df.shape[1]))
c3.metric("Numeric",str(len(raw_df.select_dtypes(include=np.number).columns)))
c4.metric("Categorical",str(len(raw_df.select_dtypes(include=["object","category"]).columns)))
if had_dupes: st.warning("⚠️ Duplicate column names renamed automatically.")
with st.expander("👀 Preview raw data",expanded=False):
    st.dataframe(_sanitize_for_display(raw_df.head(MAX_PREVIEW)),width="stretch")

# ═══════════════════════════════════════════
# STEP 2 — PROFILING
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 2️⃣ Data Profiling")
st.markdown('<div class="tooltip-box">📌 Check <b>Null %</b>, <b>Skew</b>, and <b>Unique</b> counts. High null % → needs cleaning. High skew → consider transforms.</div>', unsafe_allow_html=True)
profile_df = compute_profile(raw_df)
high_null = profile_df[profile_df["Null %"]>30]
if len(high_null)>0: st.markdown("<span class='pill warn'>⚠️ {} cols >30% nulls</span>".format(len(high_null)),unsafe_allow_html=True)
else: st.markdown("<span class='pill good'>✅ No high-null columns</span>",unsafe_allow_html=True)
with st.expander("📋 Profile Table",expanded=True):
    st.dataframe(profile_df.style.background_gradient(subset=["Null %"],cmap="YlOrRd",vmin=0,vmax=100),width="stretch",height=min(400,40+len(profile_df)*35))

# Null chart
np_ = raw_df.isnull().mean()*100; cn = np_[np_>0].sort_values(ascending=False).head(20)
if len(cn)>0:
    st.markdown("#### 🕳️ Missing Values")
    if HAS_PLOTLY:
        fig=px.bar(x=cn.index,y=cn.values,labels={"x":"Column","y":"Null %"},color=cn.values,color_continuous_scale="YlOrRd")
        fig.update_layout(height=300,margin=dict(t=30,b=40),showlegend=False); st.plotly_chart(fig,width="stretch")

# Distributions
npc = raw_df.select_dtypes(include=np.number).columns.tolist()[:6]
if npc and HAS_PLOTLY:
    st.markdown("#### 📊 Distributions")
    fig=make_subplots(rows=1,cols=len(npc),subplot_titles=npc)
    clrs=["#3b82f6","#8b5cf6","#06b6d4","#f59e0b","#ef4444","#10b981"]
    for i,col in enumerate(npc): fig.add_trace(go.Histogram(x=raw_df[col].dropna(),nbinsx=30,marker_color=clrs[i%6],showlegend=False,opacity=0.85),1,i+1)
    fig.update_layout(height=280,margin=dict(t=40,b=20)); st.plotly_chart(fig,width="stretch")

# Box plots for numeric columns
if npc and HAS_PLOTLY:
    st.markdown("#### 📦 Box Plots — Outlier Overview")
    fig=go.Figure()
    for col in npc:
        fig.add_trace(go.Box(y=raw_df[col].dropna(),name=col,boxmean='sd',marker_color=clrs[npc.index(col)%6]))
    fig.update_layout(height=350,margin=dict(t=30,b=30),showlegend=False); st.plotly_chart(fig,width="stretch")

# Data type composition pie chart + skewness bar
prof_cols=st.columns(2)
with prof_cols[0]:
    st.markdown("#### 🧩 Column Type Composition")
    dtype_counts=profile_df["Type"].apply(lambda x: "Numeric" if "int" in x or "float" in x else ("Datetime" if "datetime" in x else "Categorical")).value_counts()
    if HAS_PLOTLY:
        fig=px.pie(names=dtype_counts.index,values=dtype_counts.values,color_discrete_sequence=["#3b82f6","#f59e0b","#10b981","#8b5cf6"],hole=0.4)
        fig.update_layout(height=300,margin=dict(t=20,b=20)); st.plotly_chart(fig,width="stretch")
    else:
        fig,ax=plt.subplots(figsize=(4,4)); ax.pie(dtype_counts.values,labels=dtype_counts.index,autopct='%1.0f%%',colors=["#3b82f6","#f59e0b","#10b981"])
        st.pyplot(fig); plt.close()

with prof_cols[1]:
    skew_data=profile_df[profile_df["Skew"].notna()].sort_values("Skew",key=abs,ascending=False).head(10)
    if len(skew_data)>0:
        st.markdown("#### 📐 Skewness (top 10)")
        if HAS_PLOTLY:
            colors=["#ef4444" if abs(v)>2 else "#f59e0b" if abs(v)>1 else "#10b981" for v in skew_data["Skew"]]
            fig=go.Figure(go.Bar(x=skew_data["Column"],y=skew_data["Skew"],marker_color=colors))
            fig.add_hline(y=2,line_dash="dash",line_color="#ef4444",annotation_text="Heavy skew")
            fig.add_hline(y=-2,line_dash="dash",line_color="#ef4444")
            fig.update_layout(height=300,margin=dict(t=20,b=40)); st.plotly_chart(fig,width="stretch")
        else:
            fig,ax=plt.subplots(figsize=(5,3)); ax.bar(skew_data["Column"],skew_data["Skew"],color="#f59e0b")
            ax.axhline(2,color="red",ls="--"); ax.axhline(-2,color="red",ls="--"); plt.xticks(rotation=45,ha="right"); st.pyplot(fig); plt.close()

# Scatter matrix for top numeric columns
if len(npc)>=2 and HAS_PLOTLY:
    with st.expander("🔗 Scatter Matrix (top numeric columns)",expanded=False):
        scatter_cols=npc[:4]  # limit to 4 for readability
        fig=px.scatter_matrix(raw_df[scatter_cols].dropna().head(2000),dimensions=scatter_cols,opacity=0.4,
                              color_discrete_sequence=["#6366f1"])
        fig.update_traces(diagonal_visible=True,marker=dict(size=3))
        fig.update_layout(height=500,margin=dict(t=30,b=20)); st.plotly_chart(fig,width="stretch")

# ═══════════════════════════════════════════
# STEP 3 — KPI
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 3️⃣ KPI Insights")
st.markdown('<div class="tooltip-box">📌 Auto-detected metrics: sums, averages, top categories, date ranges, correlations.</div>',unsafe_allow_html=True)
kpi_list = compute_kpis(raw_df)
if kpi_list:
    kdf=pd.DataFrame(kpi_list)
    for cat in kdf["cat"].unique():
        ck=kdf[kdf["cat"]==cat]; st.markdown("##### {}".format(cat))
        cols=st.columns(min(len(ck),4))
        for i,(_,r) in enumerate(ck.iterrows()):
            with cols[i%len(cols)]:
                st.markdown('<div class="kpi-card"><p class="kpi-label">{} {}</p><p class="kpi-val">{}</p></div>'.format(r["icon"],r["name"],r["val"]),unsafe_allow_html=True)
        st.markdown("")
    # Cat distributions
    cc=[c for c in raw_df.select_dtypes(include=["object","category"]).columns if 2<=raw_df[c].nunique()<=20][:4]
    if cc and HAS_PLOTLY:
        st.markdown("#### 🏷️ Categorical Distributions")
        cols=st.columns(min(len(cc),2))
        for i,col in enumerate(cc):
            with cols[i%len(cols)]:
                vc=raw_df[col].value_counts().head(10)
                fig=px.bar(x=vc.index.astype(str),y=vc.values,labels={"x":col,"y":"Count"},title=col,color_discrete_sequence=["#6366f1"])
                fig.update_layout(height=280,margin=dict(t=40,b=30)); st.plotly_chart(fig,width="stretch")

    # Time-series trend lines (if datetime columns exist)
    dt_kpi=raw_df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    num_kpi=raw_df.select_dtypes(include=np.number).columns.tolist()
    if dt_kpi and num_kpi and HAS_PLOTLY:
        st.markdown("#### 📈 Time Trends")
        ts_col=dt_kpi[0]
        trend_cols=num_kpi[:3]
        df_sorted=raw_df[[ts_col]+trend_cols].dropna().sort_values(ts_col)
        if len(df_sorted)>0:
            fig=go.Figure()
            tcolors=["#3b82f6","#ef4444","#10b981"]
            for j,nc in enumerate(trend_cols):
                fig.add_trace(go.Scatter(x=df_sorted[ts_col],y=df_sorted[nc],mode="lines",name=nc,line=dict(color=tcolors[j%3])))
            fig.update_layout(height=350,margin=dict(t=30,b=40),xaxis_title=ts_col,legend=dict(orientation="h",yanchor="bottom",y=1.02)); st.plotly_chart(fig,width="stretch")

    # Correlation heatmap preview
    num_for_corr=raw_df.select_dtypes(include=np.number).columns.tolist()[:12]
    if len(num_for_corr)>=2 and HAS_PLOTLY:
        with st.expander("🔥 Quick Correlation Heatmap",expanded=False):
            corr_m=raw_df[num_for_corr].corr()
            fig=px.imshow(corr_m,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
            fig.update_layout(height=450,margin=dict(t=20,b=20)); st.plotly_chart(fig,width="stretch")

# ═══════════════════════════════════════════
# STEP 4 — CLEANING
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 4️⃣ Data Cleaning")
st.markdown('<div class="tooltip-box">📌 <b>What to do:</b> Set detection thresholds. <b>Columns with %, $, €, £ values</b> are auto-converted to numeric. <b>JSON columns</b> are auto-expanded. Choose missing value strategy, then click Apply.</div>',unsafe_allow_html=True)

cA,cB=st.columns(2)
with cA: dt_threshold=st.slider("Datetime detect threshold",0.3,0.9,0.5,0.05,help="Min % parseable as dates to auto-convert.")
with cB: num_threshold=st.slider("Numeric detect threshold",0.3,0.9,0.5,0.05,help="Min % parseable as numbers (after stripping %/$) to auto-convert.")
missing_strategy=st.selectbox("Missing value strategy",["Keep as is","Drop rows","Forward Fill","Backward Fill","Mean + Mode","Median + Mode"],index=5,
    help="Drop=remove nulls. FFill/BFill=propagate. Mean/Median=statistical fill.")
remove_dups=st.checkbox("Remove duplicate rows",value=True)

if st.button("✅ Apply Cleaning",type="primary",width="stretch"):
    with st.spinner("Cleaning — expanding JSON → converting types → handling missing values..."):
        # A) Expand JSON columns
        df_j, json_exp = expand_json_columns(raw_df)
        st.session_state.json_expansions = json_exp
        # B) Auto-convert types (handles %, $, currency, commas)
        df0, changes, pct_cols = auto_convert_types(df_j, dt_thresh=dt_threshold, num_thresh=num_threshold)
        st.session_state.type_changes = changes
        st.session_state.pct_cols = pct_cols
        # C) Handle missing
        df1 = handle_missing(df0, missing_strategy)
        if remove_dups: df1 = df1.drop_duplicates()
        st.session_state.df_clean = df1; st.session_state.df_final = df1.copy()
        for k in ["results_df","predictions_df","best_bundle_bytes","best_name","preds_store","task_type","trained_models"]:
            st.session_state[k] = None

if st.session_state.df_clean is None:
    st.info("👆 Click **Apply Cleaning** to continue."); st.stop()

df_clean=st.session_state.df_clean
bef,aft=raw_df.shape[0],df_clean.shape[0]
c1,c2,c3=st.columns(3)
c1.metric("Before","{:,}".format(bef)); c2.metric("After","{:,}".format(aft),delta="{:,}".format(aft-bef) if aft!=bef else "0")
c3.metric("Nulls Left","{:,}".format(int(df_clean.isna().sum().sum())))

# Before/After nulls comparison chart
if HAS_PLOTLY:
    null_before=raw_df.isnull().sum(); null_after=df_clean.isnull().sum()
    cols_with_change=null_before[null_before>0].index.tolist()[:15]
    if cols_with_change:
        with st.expander("📊 Before vs After — Null Counts",expanded=False):
            comp_df=pd.DataFrame({"Column":cols_with_change,"Before":[int(null_before[c]) for c in cols_with_change],
                                  "After":[int(null_after.get(c,0)) for c in cols_with_change]})
            fig=go.Figure()
            fig.add_trace(go.Bar(x=comp_df["Column"],y=comp_df["Before"],name="Before",marker_color="#ef4444",opacity=0.7))
            fig.add_trace(go.Bar(x=comp_df["Column"],y=comp_df["After"],name="After",marker_color="#10b981",opacity=0.7))
            fig.update_layout(barmode="group",height=320,margin=dict(t=30,b=40),legend=dict(orientation="h",yanchor="bottom",y=1.02))
            st.plotly_chart(fig,width="stretch")

# Show type conversions
if st.session_state.type_changes:
    with st.expander("🔄 Type conversions ({})".format(len(st.session_state.type_changes))):
        st.dataframe(pd.DataFrame(st.session_state.type_changes,columns=["Column","From","To"]),width="stretch")
# Show JSON expansions
if st.session_state.get("json_expansions"):
    with st.expander("🔗 JSON Columns Expanded"):
        for oc,ncs in st.session_state.json_expansions:
            st.markdown("**{}** → `{}`".format(oc,  "`, `".join(ncs[:10])))
# Show % conversions
if st.session_state.get("pct_cols"):
    st.markdown("<span class='pill info-pill'>📐 Converted from %: {}</span>".format(", ".join(st.session_state.pct_cols[:10])),unsafe_allow_html=True)

st.download_button("⬇️ Download cleaned CSV",data=safe_csv(df_clean),file_name="cleaned.csv",mime="text/csv",width="stretch")
with st.expander("👀 Preview cleaned",expanded=False): st.dataframe(_sanitize_for_display(df_clean.head(MAX_PREVIEW)),width="stretch")
date_cols=df_clean.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

# ═══════════════════════════════════════════
# STEP 4.5 — COLUMN MANAGER (Delete & Filter)
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 🗂️ Column Manager — Delete & Filter")
st.markdown('<div class="tooltip-box">📌 <b>Delete columns</b> you don\'t need, and <b>filter rows</b> by column values. Changes apply to the working dataset used in all downstream steps.</div>',unsafe_allow_html=True)

# Use df_final if already set (from previous interactions), else df_clean
if st.session_state.df_final is not None:
    _cm_df = st.session_state.df_final.copy()
else:
    _cm_df = df_clean.copy()

# ── 4.5a: Delete Columns ──
with st.expander("🗑️ Delete Columns", expanded=False):
    cols_to_drop = st.multiselect(
        "Select columns to remove from dataset",
        _cm_df.columns.tolist(),
        default=[],
        help="These columns will be permanently removed from the working dataset."
    )
    if cols_to_drop:
        st.warning("⚠️ Will remove **{}** column(s): `{}`".format(len(cols_to_drop), "`, `".join(cols_to_drop)))

# ── 4.5b: Filter Rows ──
with st.expander("🔍 Filter Rows", expanded=False):
    st.markdown("Add one or more filters. Rows matching **all** conditions are kept.")

    if "cm_filters" not in st.session_state:
        st.session_state.cm_filters = []

    filter_col = st.selectbox("Column to filter on", ["— select —"] + _cm_df.columns.tolist(), key="cm_fcol")

    if filter_col != "— select —":
        col_dtype = str(_cm_df[filter_col].dtype)

        # Numeric filter
        if pd.api.types.is_numeric_dtype(_cm_df[filter_col]):
            col_min = float(_cm_df[filter_col].min()) if not _cm_df[filter_col].isna().all() else 0.0
            col_max = float(_cm_df[filter_col].max()) if not _cm_df[filter_col].isna().all() else 1.0
            fc1, fc2 = st.columns(2)
            with fc1:
                f_op = st.selectbox("Condition", ["between", ">=", "<=", ">", "<", "==", "!="], key="cm_fop")
            with fc2:
                if f_op == "between":
                    f_range = st.slider("Range", min_value=col_min, max_value=col_max, value=(col_min, col_max), key="cm_frange")
                else:
                    f_val = st.number_input("Value", value=col_min, key="cm_fval")
            if st.button("➕ Add numeric filter", key="cm_addnum"):
                if f_op == "between":
                    st.session_state.cm_filters.append({"col": filter_col, "type": "numeric", "op": "between", "val": f_range})
                else:
                    st.session_state.cm_filters.append({"col": filter_col, "type": "numeric", "op": f_op, "val": f_val})

        # Datetime filter
        elif pd.api.types.is_datetime64_any_dtype(_cm_df[filter_col]):
            dt_min = _cm_df[filter_col].min()
            dt_max = _cm_df[filter_col].max()
            if pd.notna(dt_min) and pd.notna(dt_max):
                fc1, fc2 = st.columns(2)
                with fc1:
                    d_start = st.date_input("From", value=dt_min.date(), key="cm_dstart")
                with fc2:
                    d_end = st.date_input("To", value=dt_max.date(), key="cm_dend")
                if st.button("➕ Add date filter", key="cm_adddt"):
                    st.session_state.cm_filters.append({"col": filter_col, "type": "datetime", "op": "between", "val": (str(d_start), str(d_end))})
            else:
                st.info("Column has no valid dates to filter on.")

        # Categorical / object filter
        else:
            unique_vals = _cm_df[filter_col].dropna().unique().tolist()
            if len(unique_vals) <= 200:
                f_selected = st.multiselect("Keep rows where value is in:", unique_vals, default=unique_vals[:min(5, len(unique_vals))], key="cm_fcat")
                f_exclude = st.checkbox("Invert (exclude selected instead)", key="cm_fexclude")
                if st.button("➕ Add category filter", key="cm_addcat"):
                    st.session_state.cm_filters.append({"col": filter_col, "type": "categorical", "op": "exclude" if f_exclude else "include", "val": f_selected})
            else:
                f_contains = st.text_input("Keep rows containing (text search):", key="cm_ftext")
                f_case = st.checkbox("Case sensitive", value=False, key="cm_fcase")
                if st.button("➕ Add text filter", key="cm_addtxt"):
                    st.session_state.cm_filters.append({"col": filter_col, "type": "text", "op": "contains", "val": f_contains, "case": f_case})

    # Show active filters
    if st.session_state.cm_filters:
        st.markdown("**Active Filters:**")
        for i, f in enumerate(st.session_state.cm_filters):
            if f["type"] == "numeric" and f["op"] == "between":
                desc = "📐 **{}** between {} and {}".format(f["col"], f["val"][0], f["val"][1])
            elif f["type"] == "numeric":
                desc = "📐 **{}** {} {}".format(f["col"], f["op"], f["val"])
            elif f["type"] == "datetime":
                desc = "📅 **{}** from {} to {}".format(f["col"], f["val"][0], f["val"][1])
            elif f["type"] == "categorical":
                action = "exclude" if f["op"] == "exclude" else "include"
                desc = "🏷️ **{}** {} {} value(s)".format(f["col"], action, len(f["val"]))
            elif f["type"] == "text":
                desc = "🔤 **{}** contains '{}'".format(f["col"], f["val"])
            else:
                desc = str(f)
            st.markdown("{}. {}".format(i + 1, desc))

        if st.button("🗑️ Clear all filters", key="cm_clearfilters"):
            st.session_state.cm_filters = []
            st.rerun()

# ── Apply button ──
if cols_to_drop or st.session_state.get("cm_filters"):
    if st.button("✅ Apply Column Changes & Filters", type="primary", width="stretch", key="cm_apply"):
        work_df = _cm_df.copy()
        # Drop columns
        if cols_to_drop:
            work_df = work_df.drop(columns=[c for c in cols_to_drop if c in work_df.columns])
            st.success("🗑️ Dropped {} column(s)".format(len(cols_to_drop)))
        # Apply filters
        for f in st.session_state.get("cm_filters", []):
            if f["col"] not in work_df.columns:
                continue
            before_len = len(work_df)
            if f["type"] == "numeric":
                if f["op"] == "between":
                    work_df = work_df[(work_df[f["col"]] >= f["val"][0]) & (work_df[f["col"]] <= f["val"][1])]
                elif f["op"] == ">=":
                    work_df = work_df[work_df[f["col"]] >= f["val"]]
                elif f["op"] == "<=":
                    work_df = work_df[work_df[f["col"]] <= f["val"]]
                elif f["op"] == ">":
                    work_df = work_df[work_df[f["col"]] > f["val"]]
                elif f["op"] == "<":
                    work_df = work_df[work_df[f["col"]] < f["val"]]
                elif f["op"] == "==":
                    work_df = work_df[work_df[f["col"]] == f["val"]]
                elif f["op"] == "!=":
                    work_df = work_df[work_df[f["col"]] != f["val"]]
            elif f["type"] == "datetime":
                work_df = work_df[(work_df[f["col"]] >= pd.Timestamp(f["val"][0])) & (work_df[f["col"]] <= pd.Timestamp(f["val"][1]))]
            elif f["type"] == "categorical":
                if f["op"] == "include":
                    work_df = work_df[work_df[f["col"]].isin(f["val"])]
                else:
                    work_df = work_df[~work_df[f["col"]].isin(f["val"])]
            elif f["type"] == "text":
                case_flag = f.get("case", False)
                work_df = work_df[work_df[f["col"]].astype(str).str.contains(f["val"], case=case_flag, na=False)]
            st.info("🔍 Filter on **{}**: {:,} → {:,} rows".format(f["col"], before_len, len(work_df)))

        st.session_state.df_final = work_df
        st.session_state.cm_filters = []
        # Reset downstream
        for k in ["results_df", "predictions_df", "best_bundle_bytes", "best_name", "preds_store", "task_type", "trained_models"]:
            st.session_state[k] = None
        st.success("✅ Applied! Dataset is now **{:,} rows × {:,} columns**".format(work_df.shape[0], work_df.shape[1]))
        st.rerun()
else:
    st.info("No columns selected to drop and no filters added. Dataset unchanged.")

# Show current working dataset shape
_wdf = st.session_state.df_final if st.session_state.df_final is not None else df_clean
st.markdown("<span class='pill good'>📋 Working dataset: {:,} rows × {:,} cols</span>".format(_wdf.shape[0], _wdf.shape[1]), unsafe_allow_html=True)
with st.expander("👀 Preview working dataset", expanded=False):
    st.dataframe(_wdf.head(MAX_PREVIEW), width="stretch")

# ═══════════════════════════════════════════
# STEP 5 — ANOMALY DETECTION
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 5️⃣ Anomaly Detection (Optional)")
st.markdown('<div class="tooltip-box">📌 <b>IsolationForest</b> = multivariate scan. <b>Z-score/IQR</b> = single-column. Choose action for detected outliers.</div>',unsafe_allow_html=True)
df_final=st.session_state.df_final if st.session_state.df_final is not None else df_clean
enable_anom=st.checkbox("Enable anomaly detection",value=False)
if enable_anom:
    scope=st.radio("Scope",["Entire Dataset (IsolationForest)","Specific Column (Z/IQR)"],horizontal=True)
    if scope.startswith("Entire"):
        ndf=df_final.select_dtypes(include=np.number)
        if ndf.shape[1]<2: st.warning("Need ≥2 numeric columns.")
        else:
            cont=st.slider("Anomaly rate %",1,20,5)/100.0
            action=st.selectbox("Action",["Keep","Remove rows","Forward Fill","Backward Fill","Mean","Median"])
            if st.button("🚨 Run IsolationForest",width="stretch"):
                with st.spinner("Detecting..."):
                    preds=run_isoforest(ndf,cont); mask=(preds==-1)
                    st.write("Found **{:,}** anomalies ({:.1f}%).".format(int(mask.sum()),mask.sum()/max(len(df_final),1)*100))
                    df2=df_final.copy(); df2["__anomaly__"]=mask.astype(int)
                    df2=apply_anomaly_action(df2,mask,action,scope=ndf.columns.tolist())
                    st.session_state.df_final=df2; df_final=df2; st.success("Applied.")
    else:
        ncols=df_final.select_dtypes(include=np.number).columns.tolist()
        if not ncols: st.warning("No numeric columns.")
        else:
            col_ch=st.selectbox("Column",ncols); method=st.radio("Method",["Z-score","IQR"],horizontal=True)
            action=st.selectbox("Action",["Keep","Remove rows","Forward Fill","Backward Fill","Mean","Median"])
            if method=="Z-score": z_th=st.slider("Z threshold",2.0,5.0,3.0,0.1)
            else: iqr_k=st.slider("IQR multiplier",1.0,3.0,1.5,0.1)
            if st.button("🚨 Run Detection",width="stretch"):
                with st.spinner("Detecting..."):
                    am=zscore_flags(df_final[col_ch],z=z_th) if method=="Z-score" else iqr_flags(df_final[col_ch],k=iqr_k)
                    st.write("Found **{:,}** in **{}**.".format(int(am.sum()),col_ch))
                    df2=df_final.copy(); df2["__anom_{}__".format(col_ch)]=am.astype(int)
                    df2=apply_anomaly_action(df2,am,action,scope=[col_ch])
                    st.session_state.df_final=df2; df_final=df2; st.success("Applied.")
                    if HAS_PLOTLY: fig=px.box(df_final,y=col_ch,color_discrete_sequence=["#6366f1"]); fig.update_layout(height=280); st.plotly_chart(fig,width="stretch")
else: st.info("Anomaly detection OFF.")
st.markdown("<span class='pill good'>✅ Ready — {:,} rows × {:,} cols</span>".format(df_final.shape[0],df_final.shape[1]),unsafe_allow_html=True)

# ═══════════════════════════════════════════
# STEP 6 — MODELING
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown("## 6️⃣ Model Training")
st.markdown('<div class="tooltip-box">📌 Pick <b>Target (Y)</b> and <b>Features (X)</b>. Auto-detects Regression vs Classification. Adjust test %, then click Train.</div>',unsafe_allow_html=True)

all_cols=df_final.columns.tolist()
if len(all_cols)<2: st.error("Need ≥2 columns."); st.stop()

target_col=st.selectbox("🎯 Target (Y)",all_cols,index=len(all_cols)-1,help="Column to predict.")
feat_cands=[c for c in all_cols if c!=target_col and not c.startswith("__anom")]
feature_cols=st.multiselect("📊 Features (X)",feat_cands,default=feat_cands[:min(10,len(feat_cands))],help="Input columns. Fewer = faster.")
if not feature_cols: st.warning("Pick ≥1 feature."); st.stop()

t_nunique=int(df_final[target_col].nunique(dropna=True))
task_type="Classification" if (str(df_final[target_col].dtype) in ["object","category"]) or t_nunique<=15 else "Regression"
st.markdown("<span class='pill info-pill'>🔍 <b>{}</b> — {} unique values</span>".format(task_type,t_nunique),unsafe_allow_html=True)

# Heatmap
with st.expander("🔥 Correlation Heatmap",expanded=False):
    sel_num=[c for c in feature_cols+[target_col] if c in df_final.select_dtypes(include=np.number).columns.tolist()]
    if len(sel_num)>=2 and HAS_PLOTLY:
        corr=df_final[sel_num[:HEATMAP_CAP]].corr()
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
        fig.update_layout(height=480,margin=dict(t=30,b=20)); st.plotly_chart(fig,width="stretch")
    else: st.info("Need ≥2 numeric columns + Plotly.")

# TS mode
ts_mode=False; date_col=None
dt_cols_m=df_final.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
if dt_cols_m:
    ts_mode=st.checkbox("📅 Time-series mode",value=False)
    if ts_mode: date_col=st.selectbox("Order by",dt_cols_m)

c1,c2,c3=st.columns(3)
with c1: test_size=st.slider("Test %",10,50,20,5)
with c2:
    use_cv=st.checkbox("Cross-validation",value=CV_DEFAULT)
    cv_folds=st.selectbox("Folds",[3,4,5],index=0) if use_cv else 3
with c3: run_drift=st.checkbox("PSI Drift",value=PSI_DEFAULT)

# Prepare
mcl=list(set(feature_cols+[target_col]+([date_col] if ts_mode and date_col else [])))
model_df=df_final[mcl].copy().dropna(subset=[target_col])
if ts_mode and date_col: model_df=model_df.sort_values(date_col)
X=model_df[feature_cols].copy(); y=model_df[target_col].copy()

# Force target to numeric for regression (handles datetime/mixed targets)
if task_type=="Regression":
    if pd.api.types.is_datetime64_any_dtype(y):
        y=y.astype(np.int64)//10**9  # Convert datetime to unix timestamp
    else:
        y=pd.to_numeric(y,errors="coerce")
        model_df=model_df.loc[y.dropna().index]; X=X.loc[y.dropna().index]; y=y.dropna()

for c in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[c]):
        X[c+"_year"]=X[c].dt.year; X[c+"_month"]=X[c].dt.month; X[c+"_day"]=X[c].dt.day; X[c+"_dow"]=X[c].dt.dayofweek
        X=X.drop(columns=[c])

label_encoders={}
for c in X.select_dtypes(include=["object","category"]).columns:
    le=LabelEncoder(); X[c]=le.fit_transform(X[c].astype(str)); label_encoders[c]=le
target_le=None
if task_type=="Classification" and str(y.dtype) in ["object","category"]:
    target_le=LabelEncoder(); y=pd.Series(target_le.fit_transform(y.astype(str)),name=target_col)

if ts_mode:
    si=int(len(X)*(1-test_size/100.0))
    if si<=0 or si>=len(X): st.error("Not enough data."); st.stop()
    X_train,X_test=X.iloc[:si],X.iloc[si:]; y_train,y_test=y.iloc[:si],y.iloc[si:]
else:
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size/100.0,random_state=42)

m1,m2,m3=st.columns(3); m1.metric("Samples","{:,}".format(len(X))); m2.metric("Train","{:,}".format(len(X_train))); m3.metric("Test","{:,}".format(len(X_test)))

# Target distribution + Train/Test split visuals
if HAS_PLOTLY:
    tv1,tv2=st.columns(2)
    with tv1:
        st.markdown("##### 🎯 Target Distribution")
        if task_type=="Regression":
            fig=px.histogram(x=y.values,nbins=40,color_discrete_sequence=["#8b5cf6"],labels={"x":target_col})
            fig.update_layout(height=280,margin=dict(t=20,b=30),showlegend=False); st.plotly_chart(fig,width="stretch")
        else:
            vc=y.value_counts().sort_index()
            lbls=[str(target_le.inverse_transform([v])[0]) if target_le else str(v) for v in vc.index]
            fig=px.bar(x=lbls,y=vc.values,labels={"x":"Class","y":"Count"},color=lbls,color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=280,margin=dict(t=20,b=30),showlegend=False); st.plotly_chart(fig,width="stretch")
    with tv2:
        st.markdown("##### ✂️ Train / Test Split")
        fig=px.pie(names=["Train","Test"],values=[len(X_train),len(X_test)],color_discrete_sequence=["#3b82f6","#f59e0b"],hole=0.45)
        fig.update_layout(height=280,margin=dict(t=20,b=20)); st.plotly_chart(fig,width="stretch")
scaler=StandardScaler(); X_train_sc=scaler.fit_transform(X_train); X_test_sc=scaler.transform(X_test)

if task_type=="Regression":
    models={"Linear Regression":LinearRegression(),"Decision Tree":DecisionTreeRegressor(random_state=42,max_depth=DT_DEPTH),
            "Random Forest":RandomForestRegressor(n_estimators=RF_TREES,random_state=42,n_jobs=-1)}
    if HAS_HGB: models["HistGradientBoosting"]=HistGradientBoostingRegressor(random_state=42)
    else: models["GradientBoosting"]=GradientBoostingRegressor(random_state=42)
else:
    models={"Logistic Regression":LogisticRegression(max_iter=1500,random_state=42),
            "Decision Tree":DecisionTreeClassifier(random_state=42,max_depth=DT_DEPTH),
            "Random Forest":RandomForestClassifier(n_estimators=RF_TREES,random_state=42,n_jobs=-1)}
    if HAS_HGB: models["HistGradientBoosting"]=HistGradientBoostingClassifier(random_state=42)
    else: models["GradientBoosting"]=GradientBoostingClassifier(random_state=42)

if st.button("🏋️ Train Models",type="primary",width="stretch"):
    with st.spinner("Training..."):
        results=[]; preds_store={}
        if ts_mode and task_type=="Regression":
            bp=ma_forecast(y_train,len(y_test),w=min(7,len(y_train))); bn="Baseline: MA(7)"
            m=regression_metrics(y_test,bp)
            results.append({"Model":bn,"R²":round(m["R2"],4),"RMSE":round(m["RMSE"],4),"MAE":round(m["MAE"],4),"MAPE":fmt_pct(m["MAPE"]),"CV":None,"Status":"✅"})
            preds_store[bn]=bp
        progress=st.progress(0.0); names=list(models.keys())
        for i,name in enumerate(names):
            mdl=models[name]; progress.progress(float(i)/max(len(names),1))
            scaled=name in ["Linear Regression","Logistic Regression"]
            Xtr=X_train_sc if scaled else X_train; Xte=X_test_sc if scaled else X_test
            try:
                mdl.fit(Xtr,y_train); yp=mdl.predict(Xte); preds_store[name]=yp
                cv_sc=None
                if use_cv:
                    kf=KFold(n_splits=cv_folds,shuffle=not ts_mode,random_state=42 if not ts_mode else None)
                    sc="neg_root_mean_squared_error" if task_type=="Regression" else "accuracy"
                    cv_sc=float(np.mean(cross_val_score(mdl,Xtr,y_train,cv=kf,scoring=sc))*(-1 if task_type=="Regression" else 1))
                if task_type=="Regression":
                    m=regression_metrics(y_test,yp)
                    results.append({"Model":name,"R²":round(m["R2"],4),"RMSE":round(m["RMSE"],4),"MAE":round(m["MAE"],4),
                                   "MAPE":fmt_pct(m["MAPE"]),"CV":round(cv_sc,4) if cv_sc else None,"Status":"✅"})
                else:
                    acc=accuracy_score(y_test,yp); prec=precision_score(y_test,yp,average="weighted",zero_division=0)
                    rec=recall_score(y_test,yp,average="weighted",zero_division=0); f1=f1_score(y_test,yp,average="weighted",zero_division=0)
                    results.append({"Model":name,"Accuracy":fmt_pct(acc*100),"Precision":fmt_pct(prec*100),
                                   "Recall":fmt_pct(rec*100),"F1":fmt_pct(f1*100),"CV":fmt_pct(cv_sc*100) if cv_sc else None,"Status":"✅"})
            except Exception as e:
                if task_type=="Regression": results.append({"Model":name,"R²":None,"RMSE":None,"MAE":None,"MAPE":None,"CV":None,"Status":"❌ "+str(e)})
                else: results.append({"Model":name,"Accuracy":None,"Precision":None,"Recall":None,"F1":None,"CV":None,"Status":"❌ "+str(e)})
        progress.progress(1.0)
        st.session_state.results_df=pd.DataFrame(results); st.session_state.task_type=task_type
        st.session_state.preds_store=preds_store; st.session_state.trained_models=models
        best_name=None
        if task_type=="Regression":
            v=st.session_state.results_df.dropna(subset=["RMSE"])
            if not v.empty: best_name=v.loc[v["RMSE"].idxmin(),"Model"]
        else:
            rr=st.session_state.results_df.copy(); rr["_a"]=rr["Accuracy"].apply(lambda x: float(str(x).replace("%","")) if x and x!="N/A" else 0)
            v=rr[rr["_a"]>0]
            if not v.empty: best_name=v.loc[v["_a"].idxmax(),"Model"]
        st.session_state.best_name=best_name
        if best_name and best_name in preds_store:
            po=model_df.loc[X_test.index].copy(); po["y_true"]=y_test.values; po["y_pred"]=preds_store[best_name]
            if task_type=="Classification" and target_le:
                po["y_true_label"]=target_le.inverse_transform(y_test.values.astype(int))
                po["y_pred_label"]=target_le.inverse_transform(preds_store[best_name].astype(int))
            st.session_state.predictions_df=po
        else: st.session_state.predictions_df=None
        if best_name and best_name in models:
            st.session_state.best_bundle_bytes=pickle.dumps({"model_name":best_name,"task_type":task_type,"model":models[best_name],
                "scaler":scaler,"feature_cols":X_train.columns.tolist(),"label_encoders":label_encoders,"target_encoder":target_le,"ts_mode":ts_mode,"date_col":date_col})
        else: st.session_state.best_bundle_bytes=None

# ═══════════════════════════════════════════
# STEP 7 — RESULTS & EVALUATION
# ═══════════════════════════════════════════
if st.session_state.results_df is not None:
    st.markdown("---")
    st.markdown("## 7️⃣ Results & Evaluation")
    st.markdown('<div class="tooltip-box">📌 <b>MAPE</b> = Mean Absolute % Error (12.98% → ~13% off on avg, lower=better). <b>R²</b> = variance explained (1=perfect). <b>RMSE</b> = root mean squared error. <b>F1</b> = precision+recall balance.</div>',unsafe_allow_html=True)
    task_type=st.session_state.task_type; preds_store=st.session_state.preds_store or {}

    st.markdown("### 🏆 Leaderboard")
    st.dataframe(st.session_state.results_df,width="stretch",hide_index=True)
    if st.session_state.best_name: st.markdown("<span class='pill good'>🏆 Best: <b>{}</b></span>".format(st.session_state.best_name),unsafe_allow_html=True)

    # Comparison charts
    st.markdown("### 📊 Model Comparison")
    rdf=st.session_state.results_df.copy()
    if task_type=="Regression" and HAS_PLOTLY:
        vld=rdf.dropna(subset=["RMSE"]).sort_values("RMSE")
        if not vld.empty:
            fig=make_subplots(rows=1,cols=3,subplot_titles=["RMSE ↓","R² ↑","MAE ↓"])
            clrs=["#10b981" if n==st.session_state.best_name else "#3b82f6" for n in vld["Model"]]
            fig.add_trace(go.Bar(x=vld["Model"],y=vld["RMSE"],marker_color=clrs,showlegend=False),1,1)
            fig.add_trace(go.Bar(x=vld["Model"],y=vld["R²"],marker_color=clrs,showlegend=False),1,2)
            fig.add_trace(go.Bar(x=vld["Model"],y=vld["MAE"],marker_color=clrs,showlegend=False),1,3)
            fig.update_layout(height=380,margin=dict(t=50,b=40)); st.plotly_chart(fig,width="stretch")
    elif task_type!="Regression" and HAS_PLOTLY:
        rdf["_a"]=rdf["Accuracy"].apply(lambda x: float(str(x).replace("%","")) if x and x!="N/A" else 0)
        rdf["_f"]=rdf["F1"].apply(lambda x: float(str(x).replace("%","")) if x and x!="N/A" else 0)
        vld=rdf[rdf["_a"]>0].sort_values("_a",ascending=False)
        if not vld.empty:
            fig=make_subplots(rows=1,cols=2,subplot_titles=["Accuracy % ↑","F1 % ↑"])
            clrs=["#10b981" if n==st.session_state.best_name else "#6366f1" for n in vld["Model"]]
            fig.add_trace(go.Bar(x=vld["Model"],y=vld["_a"],marker_color=clrs,showlegend=False),1,1)
            fig.add_trace(go.Bar(x=vld["Model"],y=vld["_f"],marker_color=clrs,showlegend=False),1,2)
            fig.update_layout(height=380,margin=dict(t=50,b=40)); st.plotly_chart(fig,width="stretch")

    # Deep dive
    best=st.session_state.best_name
    if best and best in preds_store:
        st.markdown("### 🔬 Deep Dive — {}".format(best))
        ypb=preds_store[best]
        if task_type=="Regression":
            c1,c2=st.columns(2)
            with c1:
                st.markdown("##### Actual vs Predicted")
                if HAS_PLOTLY:
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=y_test.values.astype(float),y=ypb.astype(float),mode="markers",marker=dict(size=5,color="#6366f1",opacity=0.6),name="Pred"))
                    mn,mx=min(float(y_test.min()),float(ypb.min())),max(float(y_test.max()),float(ypb.max()))
                    fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(dash="dash",color="#ef4444"),name="Perfect"))
                    fig.update_layout(height=400,xaxis_title="Actual",yaxis_title="Predicted",margin=dict(t=30,b=40)); st.plotly_chart(fig,width="stretch")
            with c2:
                st.markdown("##### Residual Distribution")
                res=y_test.values.astype(float)-ypb.astype(float)
                if HAS_PLOTLY:
                    fig=px.histogram(x=res,nbins=40,color_discrete_sequence=["#f59e0b"],labels={"x":"Residual"})
                    fig.add_vline(x=0,line_dash="dash",line_color="red"); fig.update_layout(height=400,showlegend=False); st.plotly_chart(fig,width="stretch")
            if HAS_PLOTLY:
                st.markdown("##### Residuals vs Predicted")
                fig=go.Figure(); fig.add_trace(go.Scatter(x=ypb,y=res,mode="markers",marker=dict(size=4,color="#06b6d4",opacity=0.5),showlegend=False))
                fig.add_hline(y=0,line_dash="dash",line_color="red"); fig.update_layout(height=330,xaxis_title="Predicted",yaxis_title="Residual"); st.plotly_chart(fig,width="stretch")
            if ts_mode and HAS_PLOTLY:
                st.markdown("##### 📈 Forecast vs Actual")
                fig=go.Figure(); fig.add_trace(go.Scatter(y=y_test.values,mode="lines",name="Actual",line=dict(color="#3b82f6")))
                fig.add_trace(go.Scatter(y=ypb,mode="lines",name="Predicted",line=dict(color="#ef4444",dash="dot")))
                fig.update_layout(height=330); st.plotly_chart(fig,width="stretch")
        else:  # Classification
            c1,c2=st.columns(2)
            labels=target_le.classes_ if target_le else sorted(y_test.unique())
            with c1:
                st.markdown("##### Confusion Matrix")
                cm=confusion_matrix(y_test,ypb)
                if HAS_PLOTLY:
                    fig=px.imshow(cm,text_auto=True,x=[str(l) for l in labels],y=[str(l) for l in labels],color_continuous_scale="Blues",labels=dict(x="Predicted",y="Actual"))
                    fig.update_layout(height=400); st.plotly_chart(fig,width="stretch")
            with c2:
                st.markdown("##### Classification Report")
                rpt=classification_report(y_test,ypb,output_dict=True,target_names=[str(l) for l in labels] if target_le else None,zero_division=0)
                st.dataframe(pd.DataFrame(rpt).T.round(3),width="stretch")
            if HAS_PLOTLY and len(labels)<=20:
                cd=[{"Class":str(l),"Precision":rpt[str(l)]["precision"],"Recall":rpt[str(l)]["recall"],"F1":rpt[str(l)]["f1-score"]} for l in labels if str(l) in rpt]
                if cd:
                    cdf=pd.DataFrame(cd); fig=go.Figure()
                    fig.add_trace(go.Bar(x=cdf["Class"],y=cdf["Precision"],name="Prec",marker_color="#3b82f6"))
                    fig.add_trace(go.Bar(x=cdf["Class"],y=cdf["Recall"],name="Rec",marker_color="#f59e0b"))
                    fig.add_trace(go.Bar(x=cdf["Class"],y=cdf["F1"],name="F1",marker_color="#10b981"))
                    fig.update_layout(barmode="group",height=350); st.plotly_chart(fig,width="stretch")

    # Feature importance
    if best and best in (st.session_state.trained_models or {}):
        tm=st.session_state.trained_models[best]; imp=None
        if hasattr(tm,"feature_importances_"): imp=tm.feature_importances_
        elif hasattr(tm,"coef_"):
            imp=np.abs(tm.coef_).flatten()
            if len(imp)!=len(X_train.columns): imp=None
        if imp is not None and len(imp)==len(X_train.columns):
            st.markdown("### 🎯 Feature Importance")
            fi=pd.DataFrame({"Feature":X_train.columns,"Importance":imp}).sort_values("Importance",ascending=False).head(20)
            if HAS_PLOTLY:
                fig=px.bar(fi,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale="Viridis")
                fig.update_layout(height=max(300,len(fi)*25),yaxis=dict(autorange="reversed")); st.plotly_chart(fig,width="stretch")

            # Treemap of feature importance
            if HAS_PLOTLY and len(fi)>2:
                with st.expander("🌳 Feature Importance Treemap",expanded=False):
                    fig=px.treemap(fi,path=["Feature"],values="Importance",color="Importance",color_continuous_scale="Viridis")
                    fig.update_layout(height=400,margin=dict(t=20,b=20)); st.plotly_chart(fig,width="stretch")

    # ─── Additional Model Comparison Visuals ───
    if preds_store and len(preds_store)>1 and HAS_PLOTLY:
        st.markdown("### 📈 Advanced Model Analysis")

        if task_type=="Regression":
            # Error distribution comparison across models (violin)
            with st.expander("🎻 Error Distribution by Model (Violin)",expanded=False):
                err_data=[]
                for mname,ypred in preds_store.items():
                    errs=y_test.values-ypred
                    for e in errs: err_data.append({"Model":mname,"Error":float(e)})
                if err_data:
                    edf=pd.DataFrame(err_data)
                    fig=px.violin(edf,x="Model",y="Error",color="Model",box=True,points="outliers",
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.add_hline(y=0,line_dash="dash",line_color="red")
                    fig.update_layout(height=400,margin=dict(t=30,b=40),showlegend=False); st.plotly_chart(fig,width="stretch")

            # Cumulative absolute error chart
            with st.expander("📉 Cumulative Error — Best Model",expanded=False):
                if best and best in preds_store:
                    abs_err=np.abs(y_test.values-preds_store[best])
                    sorted_err=np.sort(abs_err)
                    cumul=np.cumsum(sorted_err)/np.sum(sorted_err)*100
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(cumul))),y=cumul,mode="lines",fill="tozeroy",
                                             line=dict(color="#3b82f6"),fillcolor="rgba(59,130,246,0.15)"))
                    fig.add_hline(y=80,line_dash="dash",line_color="#f59e0b",annotation_text="80% of error")
                    fig.update_layout(height=320,xaxis_title="Sorted Predictions",yaxis_title="Cumulative Error %",margin=dict(t=20,b=40))
                    st.plotly_chart(fig,width="stretch")

            # QQ Plot — normality of residuals
            with st.expander("📊 QQ Plot — Residual Normality",expanded=False):
                if best and best in preds_store:
                    residuals=y_test.values-preds_store[best]
                    sorted_res=np.sort(residuals)
                    n=len(sorted_res)
                    theoretical=np.array([float(np.random.normal(0,1)) for _ in range(n)])
                    theoretical.sort()
                    # Use scipy if available, else simple normal quantiles
                    try:
                        from scipy import stats as sp_stats
                        theoretical=sp_stats.norm.ppf(np.linspace(0.01,0.99,n))
                    except ImportError:
                        theoretical=np.linspace(-3,3,n)
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=theoretical,y=sorted_res,mode="markers",marker=dict(size=4,color="#8b5cf6",opacity=0.6),showlegend=False))
                    mn2,mx2=min(theoretical.min(),sorted_res.min()),max(theoretical.max(),sorted_res.max())
                    fig.add_trace(go.Scatter(x=[mn2,mx2],y=[mn2,mx2],mode="lines",line=dict(dash="dash",color="#ef4444"),name="Normal"))
                    fig.update_layout(height=380,xaxis_title="Theoretical Quantiles",yaxis_title="Sample Quantiles",margin=dict(t=20,b=40))
                    st.plotly_chart(fig,width="stretch")

            # Prediction Error Band (sorted actual vs predicted with band)
            with st.expander("🎯 Prediction vs Actual — Sorted with Error Band",expanded=False):
                if best and best in preds_store:
                    sort_idx=np.argsort(y_test.values)
                    ya_sorted=y_test.values[sort_idx]; yp_sorted=preds_store[best][sort_idx]
                    err_band=np.abs(ya_sorted-yp_sorted)
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(y=ya_sorted,mode="lines",name="Actual",line=dict(color="#3b82f6")))
                    fig.add_trace(go.Scatter(y=yp_sorted,mode="lines",name="Predicted",line=dict(color="#ef4444",dash="dot")))
                    fig.add_trace(go.Scatter(y=yp_sorted+err_band,mode="lines",line=dict(width=0),showlegend=False))
                    fig.add_trace(go.Scatter(y=yp_sorted-err_band,mode="lines",line=dict(width=0),fill="tonexty",
                                             fillcolor="rgba(239,68,68,0.1)",name="Error Band"))
                    fig.update_layout(height=380,xaxis_title="Sorted Index",yaxis_title=target_col,margin=dict(t=20,b=40))
                    st.plotly_chart(fig,width="stretch")

        else:  # Classification extra visuals
            # Radar chart comparing models
            with st.expander("🕸️ Radar Chart — Model Comparison",expanded=False):
                radar_data=[]
                for _,row in st.session_state.results_df.iterrows():
                    if row.get("Status","")!="✅": continue
                    rd={"Model":row["Model"]}
                    for metric in ["Accuracy","Precision","Recall","F1"]:
                        v=row.get(metric,"0%")
                        rd[metric]=float(str(v).replace("%","")) if v and v!="N/A" else 0
                    radar_data.append(rd)
                if radar_data:
                    rdf2=pd.DataFrame(radar_data)
                    metrics=["Accuracy","Precision","Recall","F1"]
                    fig=go.Figure()
                    rcolors=["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6"]
                    for j,(_,r) in enumerate(rdf2.iterrows()):
                        vals=[r[m] for m in metrics]+[r[metrics[0]]]
                        fig.add_trace(go.Scatterpolar(r=vals,theta=metrics+[metrics[0]],fill="toself",
                                      name=r["Model"],line=dict(color=rcolors[j%5]),opacity=0.6))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),height=420,margin=dict(t=40,b=20))
                    st.plotly_chart(fig,width="stretch")

            # Class balance in test set
            if target_le and HAS_PLOTLY:
                with st.expander("⚖️ Class Balance — Test Set",expanded=False):
                    test_vc=pd.Series(y_test).value_counts().sort_index()
                    test_labels=[str(target_le.inverse_transform([v])[0]) for v in test_vc.index]
                    fig=px.pie(names=test_labels,values=test_vc.values,color_discrete_sequence=px.colors.qualitative.Pastel,hole=0.4)
                    fig.update_layout(height=320,margin=dict(t=20,b=20)); st.plotly_chart(fig,width="stretch")

    # PSI
    if run_drift:
        with st.expander("🧭 PSI Drift"):
            dcols=[c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])][:50]
            pr=[(c,psi_numeric(X_train[c],X_test[c])) for c in dcols]
            pdf=pd.DataFrame(pr,columns=["Feature","PSI"]).sort_values("PSI",ascending=False)
            st.dataframe(pdf,width="stretch",hide_index=True)
            if HAS_PLOTLY and len(pdf)>0:
                fig=px.bar(pdf.head(20),x="Feature",y="PSI",color="PSI",color_continuous_scale="YlOrRd")
                fig.add_hline(y=0.25,line_dash="dash",line_color="red",annotation_text="Significant")
                fig.add_hline(y=0.1,line_dash="dash",line_color="orange",annotation_text="Moderate")
                fig.update_layout(height=330); st.plotly_chart(fig,width="stretch")

    # ═══════════════════════════════════════════
    # STEP 8 — ASK AI (Claude RAG)
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 8️⃣ Ask AI About Your Data")
    st.markdown('<div class="tooltip-box">📌 <b>What to do:</b> Ask any question about your uploaded data, cleaning results, or model performance. Claude analyzes your data context (schema, stats, sample rows, model results) and gives specific answers.</div>',unsafe_allow_html=True)

    # ── Retrieve API key from Streamlit secrets ──
    api_key = st.secrets.get("CLAUD_KEY", None)

    if not api_key:
        st.error("⚠️ `ANTHROPIC_API_KEY` not found in Streamlit secrets. Add it to `.streamlit/secrets.toml` or your Streamlit Cloud secrets.")
        st.code('# .streamlit/secrets.toml\nANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
    elif not HAS_ANTHROPIC:
        st.error("Please install the anthropic package: `pip install anthropic`")
    else:
        # Build context
        data_ctx = build_data_context(df_final)
        model_ctx = ""
        if st.session_state.results_df is not None:
            model_ctx = "\nMODEL RESULTS:\n" + st.session_state.results_df.to_string(index=False)
            if st.session_state.best_name:
                model_ctx += "\nBest model: " + st.session_state.best_name

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask about your data... (e.g. 'What are the top 3 insights?' or 'Why is MAPE high?')")
        # Chat input
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Claude is analyzing your data..."):
                    try:
                        response = ask_claude(prompt, data_ctx, model_ctx, api_key)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        err_msg = "Error: {}".format(str(e))
                        st.error(err_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

        if st.session_state.chat_history and st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

    # ═══════════════════════════════════════════
    # STEP 9 — DOWNLOADS
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 9️⃣ Downloads")
    st.markdown('<div class="tooltip-box">📌 Download predictions CSV and/or trained model bundle (.pkl).</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2)
    with d1:
        if st.session_state.predictions_df is not None:
            st.download_button("⬇️ Predictions CSV",data=safe_csv(st.session_state.predictions_df),file_name="predictions.csv",mime="text/csv",width="stretch")
        else: st.info("No predictions.")
    with d2:
        if st.session_state.best_bundle_bytes is not None:
            st.download_button("⬇️ Model Bundle (.pkl)",data=st.session_state.best_bundle_bytes,file_name="best_model.pkl",mime="application/octet-stream",width="stretch")
        else: st.info("No model bundle.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#94a3b8;font-size:0.82rem;padding:1rem 0'>AutoML POC v1.0 — Fast + KPI + JSON Expand + %/$ Parsing + Claude AI | Button-driven • Cached • Plotly</div>",unsafe_allow_html=True)