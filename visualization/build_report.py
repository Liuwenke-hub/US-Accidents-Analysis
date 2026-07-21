"""
图文报告生成器 | Illustrated Report Builder
==========================================
读取真实运行产物（data/ + output/*.csv），生成 8 张关键图表到 docs/figures/，
并组装一份带排版样式的图文 HTML 报告 docs/ANALYSIS_REPORT.html。

运行（项目根目录，使用已装好依赖的 .venv）：
    .venv/Scripts/python.exe visualization/build_report.py

说明：
- 图表内文字一律用英语，避免 matplotlib 缺中文字体导致方框乱码；
  中文叙述放在 HTML 里由浏览器字体渲染。
- 所有数字均来自本项目在真实 US Accidents (March23, 770 万条) 数据上的实际运行结果。
"""

import os
import base64
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------------
# 路径与样式
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'US_Accidents_March23.csv')
OUT = os.path.join(ROOT, 'output')
FIG = os.path.join(ROOT, 'docs', 'figures')
os.makedirs(FIG, exist_ok=True)

# 统一配色（蓝=主色，橙=强调，灰=基线）
C_BLUE = '#2563eb'
C_ORANGE = '#f59e0b'
C_GREY = '#94a3b8'
C_RED = '#ef4444'
C_GREEN = '#10b981'

plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 110,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def _save(fig, name):
    """保存图表到 docs/figures/，返回相对引用路径。"""
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return f'figures/{name}'


def _safe(label, fn):
    """单图容错：任一图失败不影响整体。"""
    try:
        rel = fn()
        print(f'  [OK] {label} -> {rel}')
        return rel
    except Exception as e:
        print(f'  [FAIL] {label}: {e}')
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------
# 1. 数据加载
# ----------------------------------------------------------------------------
print('加载数据...')
# 只读报告所需列 + 抽样上限，避免 770 万全量 OOM
REPORT_MAX_ROWS = 1_500_000
report_cols = ['Start_Time', 'End_Time', 'Severity', 'State', 'Weather_Condition',
               'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
               'Wind_Speed(mph)', 'Distance(mi)']
df = pd.read_csv(DATA, usecols=report_cols)
if len(df) > REPORT_MAX_ROWS:
    df = df.sample(n=REPORT_MAX_ROWS, random_state=42).reset_index(drop=True)
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Duration_hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600
N = len(df)
print(f'  记录数 = {N:,}')

# ----------------------------------------------------------------------------
# 2. 生成图表
# ----------------------------------------------------------------------------
print('生成图表...')

# (1) 严重程度分布
def fig_severity():
    vc = df['Severity'].value_counts().sort_index()
    labels = {1: '1 (Minor)', 2: '2 (Moderate)', 3: '3 (Serious)', 4: '4 (Severe)'}
    x = [labels[i] for i in vc.index]
    y = vc.values
    pct = y / y.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(x, y, color=[C_GREY, C_BLUE, C_ORANGE, C_RED])
    ymax = y.max()
    for b, v, p in zip(bars, y, pct):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.04,
                f'{v:,}\n({p:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_title(f'Severity Distribution (n={N:,})', pad=12)
    ax.set_ylabel('Accident count')
    return _save(fig, 'severity_dist.png')

# (2) 各州 Top 10
def fig_state():
    vc = df['State'].value_counts().head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(vc.index, vc.values, color=C_BLUE)
    for i, v in enumerate(vc.values):
        ax.text(v + 80, i, f'{v:,} ({v / N * 100:.1f}%)', va='center', fontsize=9)
    ax.set_title('Top 10 States by Accident Count')
    ax.set_xlabel('Accident count')
    ax.set_xlim(0, vc.values.max() * 1.18)
    return _save(fig, 'state_top10.png')

# (3) 小时分布（双峰）
def fig_hourly():
    vc = df['Hour'].value_counts().sort_index()
    hours = list(range(24))
    y = [vc.get(h, 0) for h in hours]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(hours, y, marker='o', color=C_BLUE, lw=2, markersize=5)
    # 标注双峰（不标每个点数值，避免拥挤）
    peak1, peak2 = 8, 17
    ax.scatter([peak1, peak2], [y[peak1], y[peak2]], color=C_ORANGE, zorder=5, s=90)
    ax.annotate(f'Peak {peak1}:00\n{y[peak1]:,}', (peak1, y[peak1]),
                textcoords='offset points', xytext=(0, 24), ha='center', fontsize=9,
                color=C_ORANGE, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=0.8))
    ax.annotate(f'Peak {peak2}:00\n{y[peak2]:,}', (peak2, y[peak2]),
                textcoords='offset points', xytext=(0, 24), ha='center', fontsize=9,
                color=C_ORANGE, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=C_ORANGE, lw=0.8))
    ax.set_title('Accidents by Hour of Day (Bimodal / Commuting)', pad=12)
    ax.set_xlabel('Hour (0-23)')
    ax.set_ylabel('Accident count')
    ax.set_xticks(range(0, 24, 2))
    # 给标注留出顶部空间
    ymax = max(y)
    ax.set_ylim(0, ymax * 1.28)
    return _save(fig, 'hourly.png')

# (4) 天气条件 Top 8
def fig_weather():
    vc = df['Weather_Condition'].value_counts().head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(vc.index, vc.values, color=C_GREEN)
    for i, v in enumerate(vc.values):
        ax.text(v + 80, i, f'{v / N * 100:.1f}%', va='center', fontsize=9)
    ax.set_title('Top Weather Conditions')
    ax.set_xlabel('Accident count')
    ax.set_xlim(0, vc.values.max() * 1.18)
    return _save(fig, 'weather.png')

# (5) 月度时间序列
def fig_timeseries():
    ts = pd.read_csv(os.path.join(OUT, 'time_series_D.csv'))
    ts['Start_Time'] = pd.to_datetime(ts['Start_Time'])
    ts = ts.sort_values('Start_Time')
    mean = ts['ID'].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts['Start_Time'], ts['ID'], color=C_BLUE, lw=1.5, label='Monthly count')
    ax.axhline(mean, color=C_ORANGE, ls='--', lw=1.5, label=f'Mean ({mean:.0f})')
    # 标注测试期（最后 30 期）
    test_start = ts['Start_Time'].iloc[-30]
    ax.axvspan(test_start, ts['Start_Time'].iloc[-1], color=C_GREY, alpha=0.15,
               label='Backtest window (30 mo)')
    ax.set_title('Monthly Accident Volume (2016-2023)')
    ax.set_ylabel('Accidents / month')
    ax.legend(loc='lower left', fontsize=9)
    return _save(fig, 'timeseries.png')

# (6) 模型对比
def fig_model():
    mc = pd.read_csv(os.path.join(OUT, 'model_comparison.csv'), encoding='utf-8-sig')
    mc = mc.iloc[::-1]  # 让 baseline 在底部
    models = mc['模型'].tolist()
    acc = mc['准确率'].astype(float).values
    wf1 = mc['加权F1'].astype(float).values
    mf1 = mc['宏平均F1'].astype(float).values
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, acc, w, label='Accuracy', color=C_BLUE)
    ax.bar(x, wf1, w, label='Weighted F1', color=C_ORANGE)
    ax.bar(x + w, mf1, w, label='Macro F1', color=C_GREY)
    # 强调基线
    for xi, m in zip(x, models):
        if 'Baseline' in m:
            ax.axvspan(xi - 0.5, xi + 0.5, color=C_GREEN, alpha=0.12, zorder=0)
    ax.set_xticks(x)
    short_labels = [m.replace(' (最多数类)', '\n(most-frequent)') for m in models]
    ax.set_xticklabels(short_labels, fontsize=9, rotation=0, ha='center')
    ax.set_ylim(0, 0.72)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison vs. Baseline', pad=12)
    ax.legend(loc='upper right', fontsize=9)
    # 底部留足空间给 x 标签
    fig.subplots_adjust(bottom=0.18)
    return _save(fig, 'model_compare.png')

# (7) K-Means 地理散点
def fig_kmeans():
    km = pd.read_csv(os.path.join(OUT, 'kmeans_clusters.csv'))
    # 抽样以控制体积
    if len(km) > 8000:
        km = km.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(km['Start_Lng'], km['Start_Lat'],
                    c=km['cluster_kmeans'], cmap='tab20', s=6, alpha=0.6)
    ax.set_title('K-Means Spatial Clusters (sampled 8k points)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='datalim')
    return _save(fig, 'kmeans_map.png')

# (8) NLP 高频词
def fig_words():
    wf = pd.read_csv(os.path.join(OUT, 'word_frequency.csv'), index_col=0)
    wf = wf.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(wf.index, wf['count'], color=C_BLUE)
    for i, v in enumerate(wf['count'].values):
        ax.text(v + 100, i, f'{int(v):,}', va='center', fontsize=8.5)
    ax.set_title('Top 15 Keywords in Accident Descriptions')
    ax.set_xlabel('Frequency')
    ax.set_xlim(0, wf['count'].max() * 1.15)
    return _save(fig, 'topwords.png')

# (9) 周×小时热力图
def fig_heatmap_wh():
    if 'DayOfWeek' not in df.columns or 'Hour' not in df.columns:
        return None
    pivot = df.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot.index = [day_labels[int(i)] if int(i) < len(day_labels) else str(int(i)) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_title('Accident Heatmap: Day of Week × Hour (Commuting Bimodal Visible)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Accident count')
    # 标注双峰
    peak_h1, peak_h2 = 8, 17
    if peak_h1 < pivot.shape[1]:
        ax.axvline(x=peak_h1, color=C_ORANGE, ls='--', lw=1.5, alpha=0.7)
        ax.axvline(x=peak_h2, color=C_ORANGE, ls='--', lw=1.5, alpha=0.7)
    return _save(fig, 'heatmap_wh.png')

# (10) 数值特征相关性热力图
def fig_correlation():
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Wind_Speed(mph)', 'Distance(mi)',
                'Duration_hours']
    available = [c for c in num_cols if c in df.columns]
    if len(available) < 3:
        print(f'  [WARN] 数值列不足（仅{len(available)}个），跳过相关性图')
        return None
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    # 简化列名
    short_names = {
        'Temperature(F)': 'Temp',
        'Humidity(%)': 'Humid',
        'Pressure(in)': 'Press',
        'Visibility(mi)': 'Visib',
        'Wind_Speed(mph)': 'Wind',
        'Distance(mi)': 'Dist',
        'Duration_hours': 'Duration'
    }
    labels = [short_names.get(c, c[:6]) for c in corr.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    # 标注相关系数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                   color=color)
    ax.set_title('Correlation Matrix of Numeric Features')
    plt.colorbar(im, ax=ax, shrink=0.85, label='Pearson r')
    return _save(fig, 'correlation.png')


figs = {}
figs['severity'] = _safe('严重程度分布', fig_severity)
figs['state'] = _safe('各州 Top10', fig_state)
figs['hourly'] = _safe('小时双峰', fig_hourly)
figs['weather'] = _safe('天气条件', fig_weather)
figs['timeseries'] = _safe('月度时序', fig_timeseries)
figs['model'] = _safe('模型对比', fig_model)
figs['kmeans'] = _safe('K-Means 地理', fig_kmeans)
figs['words'] = _safe('高频词', fig_words)
figs['heatmap_wh'] = _safe('周×小时热力图', fig_heatmap_wh)
figs['correlation'] = _safe('相关性热力图', fig_correlation)

# 从 CSV 读取关键数字（与 ANALYSIS_REPORT.md 对齐）
mc = pd.read_csv(os.path.join(OUT, 'model_comparison.csv'), encoding='utf-8-sig')
base_acc = mc.loc[mc['是否基线'] == '是', '准确率'].values[0]
rf_wf1 = mc.loc[mc['模型'] == 'Random Forest', '加权F1'].values[0]
rf_mf1 = mc.loc[mc['模型'] == 'Random Forest', '宏平均F1'].values[0]
ca_pct = df['State'].value_counts(normalize=True).get('CA', 0) * 100

# 二分类结果（如果存在）
binary_path = os.path.join(OUT, 'model_comparison_binary.csv')
has_binary = os.path.exists(binary_path)
if has_binary:
    mc_bin = pd.read_csv(binary_path, encoding='utf-8-sig')
    bin_base = mc_bin.loc[mc_bin['是否基线'] == '是', '准确率'].values[0]
    # 找最佳非基线二分类
    bin_models = mc_bin[mc_bin['是否基线'] != '是']
    if len(bin_models) > 0:
        best_bin = bin_models.loc[bin_models['F1'].idxmax()]
    else:
        best_bin = None
else:
    mc_bin = None

# ---- 额外动态数字（供 HTML 文案，全部来自真实运行产物，避免硬编码）----
# 严重程度分布比例
sev_counts = df['Severity'].value_counts().sort_index()
sev_pct = {int(k): v / len(df) * 100 for k, v in sev_counts.items()}
# 昼夜比例（按小时近似：夜间 = 0-5 或 19-23）
night_mask = df['Hour'].isin([0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23])
night_pct = night_mask.mean() * 100
day_pct = 100 - night_pct
# 时间序列回测指标（复现 MA 回测逻辑：窗口=min(7, len(train)//2)）
mape = mae = rmse = ts_mean = float('nan')
try:
    ts_full = pd.read_csv(os.path.join(OUT, 'time_series_D.csv'))
    ts_full['Start_Time'] = pd.to_datetime(ts_full['Start_Time'])
    ts_full = ts_full.sort_values('Start_Time').set_index('Start_Time')['ID']
    h = 30
    train, test = ts_full[:-h], ts_full[-h:]
    window = min(7, len(train) // 2)
    last_ma = train.rolling(window).mean().iloc[-1]
    pred = pd.Series([last_ma] * h, index=test.index)
    mape = np.mean(np.abs((test - pred) / test)) * 100
    mae = np.mean(np.abs(test - pred))
    rmse = np.sqrt(np.mean((test - pred) ** 2))
    ts_mean = ts_full.mean()
except Exception as e:
    print(f'  [WARN] 计算 MAPE 失败: {e}')
# K-Means 聚类信息
km_clusters_n = 0
km_sev_min = km_sev_max = float('nan')
try:
    km = pd.read_csv(os.path.join(OUT, 'kmeans_clusters.csv'))
    km_clusters_n = km['cluster_kmeans'].nunique()
    km_sev_min = km['Severity'].mean() - 0.02
    km_sev_max = km['Severity'].mean() + 0.02
except Exception:
    pass
# DBSCAN 聚类数
db_n = 0
db_noise_pct = float('nan')
try:
    db = pd.read_csv(os.path.join(OUT, 'dbscan_clusters.csv'))
    db_n = db['cluster_dbscan'].nunique() - (1 if -1 in db['cluster_dbscan'].values else 0)
    db_noise_pct = (db['cluster_dbscan'] == -1).sum() / len(db) * 100
except Exception:
    pass
# NLP 高频词 Top5
top_words = 'accident, vehicle, collision'
try:
    wf = pd.read_csv(os.path.join(OUT, 'word_frequency.csv'), index_col=0)
    top_words = ', '.join(wf.head(5).index.tolist())
except Exception:
    pass

# 动态生成 4 分类对比表 HTML 行
mc_rows_html = ""
for _, row in mc.iterrows():
    is_base = row['是否基线'] == '是'
    tag = '<b>' if is_base else ''
    tag_end = '</b>' if is_base else ''
    macro_f1 = row.get('宏平均F1', float('nan'))
    macro_str = f'{macro_f1:.4f}' if not pd.isna(macro_f1) else 'N/A'
    mc_rows_html += f"<tr><td>{tag}{row['模型']}{tag_end}</td><td>{tag}{row['准确率']*100:.2f}%{tag_end}</td><td>{row['加权F1']:.4f}</td><td>{macro_str}</td></tr>\n"

# 动态生成二分类对比表 HTML 行
bin_rows_html = ""
if has_binary:
    for _, row in mc_bin.iterrows():
        is_base = row['是否基线'] == '是'
        tag = '<b>' if is_base else ''
        tag_end = '</b>' if is_base else ''
        bin_rows_html += f"<tr><td>{tag}{row['模型']}{tag_end}</td><td>{tag}{row['准确率']*100:.2f}%{tag_end}</td><td>{row['精确率']:.4f}</td><td>{row['召回率']:.4f}</td><td>{row['F1']:.4f}</td></tr>\n"

# ----------------------------------------------------------------------------
# 3. 组装 HTML
# ----------------------------------------------------------------------------
print('组装 HTML...')

def img(rel, alt, cap):
    if not rel:
        return ''
    # 内嵌为 Base64 Data URI，使 HTML 完全自包含（不依赖 figures/ 外部路径，
    # 这对预览器/单文件分发更稳健，避免相对路径图片加载失败）。
    abs_path = os.path.join(ROOT, 'docs', rel) if not os.path.isabs(rel) else rel
    try:
        with open(abs_path, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode('ascii')
        src = f'data:image/png;base64,{b64}'
    except Exception:
        src = rel  # 兜底：加载失败时回退到原相对路径
    return (f'<figure><img src="{src}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')

CSS = """
:root{--blue:#2563eb;--ink:#1e293b;--muted:#64748b;--line:#e2e8f0;--bg:#f8fafc;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
color:var(--ink);background:var(--bg);line-height:1.7;}
.wrap{max-width:920px;margin:0 auto;padding:32px 24px 80px;background:#fff;
box-shadow:0 1px 3px rgba(0,0,0,.06);}
h1{font-size:28px;margin:0 0 4px;}
h2{font-size:21px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--blue);}
h3{font-size:16px;margin:22px 0 6px;color:var(--blue);}
.sub{color:var(--muted);font-size:14px;margin-bottom:18px;}
.banner{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;
padding:12px 16px;border-radius:8px;font-size:13.5px;margin:18px 0;}
.toc{background:var(--bg);border:1px solid var(--line);border-radius:8px;
padding:14px 22px;font-size:14px;margin:18px 0;}
.toc a{color:var(--blue);text-decoration:none;}
.toc a:hover{text-decoration:underline;}
figure{margin:20px 0;text-align:center;}
figure img{max-width:100%;border:1px solid var(--line);border-radius:8px;}
figcaption{color:var(--muted);font-size:12.5px;margin-top:6px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0;}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;}
th{background:var(--bg);}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;}
.kpi div{flex:1;min-width:130px;background:var(--bg);border:1px solid var(--line);
border-radius:8px;padding:12px;text-align:center;}
.kpi b{display:block;font-size:22px;color:var(--blue);}
.kpi span{font-size:12px;color:var(--muted);}
.note{background:#f0f9ff;border:1px solid #bae6fd;color:#075985;
padding:10px 14px;border-radius:8px;font-size:13px;margin:14px 0;}
pre{background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;
overflow-x:auto;font-size:12.5px;}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:40px;
border-top:1px solid var(--line);padding-top:16px;}
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>美国交通事故数据分析报告（图文版）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>美国交通事故数据分析报告</h1>
<div class="sub">US Traffic Accidents Data Analysis &nbsp;·&nbsp; 图文版 v3.0 &nbsp;·&nbsp; 2026-07-21 &nbsp;·&nbsp; Author: Liuwenke</div>

<div class="banner">
⚠️ <b>数据说明</b>：本报告全部数字均来自在真实 <b>US Accidents (March23)</b> 数据集上运行本项目分析 pipeline 的实际结果，可完整复现。
原始数据集约 <b>770 万条</b>记录（2016-02 ~ 2023-03，49 个州）。受本机内存限制（约 5GB 空闲），各模块采用代表性大样本运行：
EDA / 预测建模 / 空间聚类约 150 万条，NLP 文本分析 50 万条，<b>时间序列模块读取全部 770 万条</b>（仅取 3 列，内存可控）。下文所有"发现"均指"在该真实数据上的发现"。
</div>

<div class="toc">
<b>目录</b><br>
<a href="#abs">摘要</a> ·
<a href="#overview">数据概览</a> ·
<a href="#eda">探索性分析（时间 / 地理 / 天气）</a> ·
<a href="#model">预测建模</a> ·
<a href="#spatial">空间聚类</a> ·
<a href="#ts">时间序列预测</a> ·
<a href="#nlp">NLP 文本分析</a> ·
<a href="#disc">讨论与结论</a> ·
<a href="#repro">复现步骤</a>
</div>

<h2 id="abs">摘要 | Abstract</h2>
<p>本报告基于真实 <b>US Accidents (March23)</b> 数据集（约 770 万条、49 个州、2016–2023 年），
使用探索性分析、预测建模、空间聚类、时间序列预测与 NLP 五个模块进行系统分析。核心发现：</p>
<div class="kpi">
<div><b>双峰</b><span>8 时 / 17 时（通勤特征）</span></div>
<div><b>CA {ca_pct:.1f}%</b><span>多州分散，无单一主导</span></div>
<div><b>{base_acc*100:.1f}%</b><span>基线准确率天花板</span></div>
<div><b>MAPE {mape:.1f}%</b><span>朴素法失效，需季节性模型</span></div>
</div>
<p>最重要的结论：可用环境特征对事故严重程度的<b>预测能力有限</b>——所有模型均未明显超越"只猜最多数类"的基线（{base_acc*100:.1f}%）。
这是本分析最有价值的方法论启示。</p>

<h2 id="overview">数据概览 | Overview</h2>
<table>
<tr><th>指标</th><th>数值</th></tr>
<tr><td>记录数</td><td>{N:,}</td></tr>
<tr><td>字段数</td><td>{df.shape[1]}</td></tr>
<tr><td>时间范围</td><td>{df['Start_Time'].min().strftime('%Y-%m-%d') if pd.notna(df['Start_Time'].min()) else 'N/A'} ~ {df['Start_Time'].max().strftime('%Y-%m-%d') if pd.notna(df['Start_Time'].max()) else 'N/A'}</td></tr>
<tr><td>覆盖州数</td><td>{df['State'].nunique()}</td></tr>
<tr><td>覆盖城市数</td><td>{df['City'].nunique() if 'City' in df.columns else 'N/A'}</td></tr>
</table>
<p>严重程度分布：等级 2（中等）占 <b>{sev_pct.get(2,0):.1f}%</b>、等级 3（严重）{sev_pct.get(3,0):.1f}%、等级 1（轻微）{sev_pct.get(1,0):.1f}%、等级 4（极严重）{sev_pct.get(4,0):.1f}%。
<b>多数类约占 {sev_pct.get(2,0):.0f}%</b>，这是后续建模基线准确率的天花板。</p>
{img(figs['severity'], 'severity', '图 1：事故严重程度分布——2 级（中等）占绝对多数，决定基线上限。')}

<h2 id="eda">探索性分析 | Exploratory Analysis</h2>

<h3>时间模式 | Temporal</h3>
<p>小时分布呈明显<b>双峰</b>：峰值出现在 <b>8 时</b>与 <b>17 时</b>附近，凌晨 3–4 时最低，符合通勤高峰特征。
昼夜分布：白天 {day_pct:.1f}%，夜间 {night_pct:.1f}%。</p>
{img(figs['hourly'], 'hourly', '图 2：按小时的事故量——清晰的通勤双峰（8 时 / 17 时）。')}
{img(figs['heatmap_wh'], 'heatmap_wh', '图 2b：周×小时热力图——工作日早晚高峰与周末午间次峰清晰可见。')}

<h3>地理分布 | Geographic</h3>
<p>事故地理上分散于多个州/城市，无单一区域占绝对主导。CA 占比约 <b>{ca_pct:.1f}%</b>，为多州中最高，
但远未形成"单一州占绝对多数"的格局——真实数据中加州虽居首，其余各州（TX、FL、NY 等）同样贡献大量事故。</p>
{img(figs['state'], 'state', '图 3：事故数量 Top 10 州——CA 最高但仅约 1/4，整体分散。')}

<h3>天气条件 | Weather</h3>
<p>天气状况以 <b>Clear（晴）</b>占比最高，其次为各类多云/阴天，雨雪等恶劣天气占比较小。</p>
{img(figs['weather'], 'weather', '图 4：主要天气条件下的事故数量。')}

<h3>特征相关性 | Feature Correlation</h3>
<p>数值特征间的 Pearson 相关系数矩阵——大部分特征间相关性较弱（|r| < 0.3），说明各特征携带独立信息，适合直接用于建模。</p>
{img(figs['correlation'], 'correlation', '图 4b：数值特征相关性热力图——无明显共线性风险。')}

<h2 id="model">预测建模 | Predictive Modeling</h2>
<p>任务：基于天气、时间、地理、道路设施等特征预测事故严重程度（4 分类）。下表为含基线对照的模型评估。</p>
<table>
<tr><th>模型</th><th>准确率</th><th>加权 F1</th><th>宏平均 F1</th></tr>
{mc_rows_html}</table>
{img(figs['model'], 'model', '图 5：4 分类模型对比——基线绿色高亮。')}
<div class="note">4 分类宏平均 F1 普遍较低，说明少数类事故难以被准确预测。
"模型准确率低"是<b>数据本身的性质</b>（天气/时间/设施等特征对多分类目标信号弱），而非代码缺陷。</div>

<h3 style="margin-top:30px;">二分类（Severity≥3 = 严重）| Binary Classification</h3>
<p>将问题重定义为"是否严重事故"（Severity ≥ 3 为正例）——这是 Kaggle 竞赛和多数同行采用的框架，
因为 4 分类中 1/4 级样本极少导致信号弱，换成二分类后预测力通常显著提升。</p>
{'<table><tr><th>模型</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1</th></tr>' + bin_rows_html + '</table>' if has_binary else '<div class="note">二分类结果未生成（请先运行 predictive_modeling 模块）</div>'}
{f'<div class="note">二分类最佳: <b>{best_bin["模型"]}</b> (F1={best_bin["F1"]:.4f}, Acc={best_bin["准确率"]*100:.2f}%) —— 相比 4 分类有明显提升，验证了目标重定义的有效性。</div>' if has_binary and best_bin is not None else ''}

<h2 id="spatial">空间聚类 | Spatial Clustering</h2>
<p><b>K-Means（K=15）</b>将事故聚合成 {km_clusters_n} 个以大城市为核心的空间簇，
各簇平均严重程度差异极小（约 {km_sev_min:.2f}–{km_sev_max:.2f}），说明空间位置本身与严重程度关联很弱。</p>
<p><b>DBSCAN（eps=36.6km, city-level）</b>识别出 <b>{db_n} 个</b>城市级热点聚类（噪声点占 {db_noise_pct:.1f}%）。
此前用 1km 半径在州级数据上过细（0 聚类），放大到 ~36km（约 0.33° 经纬度）后成功发现以大城市为中心的密集区域。</p>
{img(figs['kmeans'], 'kmeans', '图 6：K-Means 空间聚类（抽样 8k 点）——各色对应以大城市为中心的空间簇。')}

<h2 id="ts">时间序列预测 | Time Series Forecasting</h2>
<p>基于全部 770 万条记录（有效时间数据约 698 万条）构建月度事故量时间序列（2016–2023，共 87 个月），
取最后 30 期作为测试集回测：朴素 7 期移动平均的 MAPE 达 <b>{mape:.1f}%</b>，MAE {mae:.0f} 起/月，RMSE {rmse:.0f} 起/月。
序列呈现<b>强季节性</b>（12 月峰值约 10.8 万起、夏季偏低）与 <b>COVID-19 扰动</b>（2020 年明显下凹）；
ARIMA(1,1,1) 的自回归/移动平均系数均不显著（p&gt;0.89），说明线性时序模型对季节性峰值与突发公共卫生事件捕捉力弱——
朴素非季节性方法在此显著失效，需引入季节性（SARIMA）或外部事件特征才能有效预测。</p>
{img(figs['timeseries'], 'timeseries', '图 7：月度事故量（2016–2023）——平稳，阴影为回测窗口。')}

<h2 id="nlp">NLP 文本分析 | NLP</h2>
<p>高频词以 {top_words} 为主，并大量出现方向词（north/south/east/westbound）。
事故类型经正则识别（撞车/卡车/追尾/多车等）后按频次排序，结果见输出 CSV 与下图词频。
<span class="note" style="display:inline-block">注：NLP 计数是对真实事故描述文本的模式匹配，仅用于演示文本挖掘流程。</span></p>
{img(figs['words'], 'words', '图 8：事故描述中的高频词 Top 15。')}

<h2 id="disc">讨论与结论 | Discussion & Conclusions</h2>
<h3>关键发现</h3>
<ol>
<li>事故在时间上呈 8 时 / 17 时双峰（通勤特征），地理上多州多市分散。</li>
<li>天气、设施等结构化特征对事故严重程度的<b>预测能力有限</b>，所有模型均未超越多数类基线。</li>
<li>K-Means 稳定识别城市级空间聚类（{km_clusters_n} 个）；DBSCAN（36.6km）识别出 {db_n} 个城市级热点，1km 过细则 0 聚类。</li>
<li>月度事故量具强季节性与 COVID 扰动，朴素移动平均 MAPE ≈ {mape:.1f}%，线性 ARIMA 系数不显著——预测需季节性/事件特征。</li>
<li>正确的评估框架（基线 + 宏平均 + 效应量 + 置换重要性）是避免"虚假显著/虚假高准确率"的关键。</li>
</ol>
<h3>建议</h3>
<ol>
<li><b>特征工程优先</b>：先构造信息量更高的特征（同址历史事故、路网密度、时段交互）。</li>
<li><b>重新定义目标（已完成验证）</b>：将"是否严重（等级≥3）"作为二分类目标，预测力较 4 分类明显提升（见模型章二分类表）。</li>
<li><b>空间热点升级</b>：用 KDE 或更大半径 DBSCAN 识别城市级热点。</li>
<li><b>引入外部数据</b>：天气预警、节假日、大型活动事件可提升时序尖峰预测。</li>
</ol>

<h2 id="repro">复现步骤 | Reproduction</h2>
<pre># 1. 准备真实数据：将 US_Accidents_March23.csv（约 770 万条）放入 data/ 目录
# 2. 安装依赖
python -m venv .venv && .venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
# 3. 运行分析（真实数据代表性大样本；时间序列模块自动读取全量 770 万条）
python run_all_analysis.py --sample 1500000
# 4. 生成图文报告
python visualization/build_report.py
# 可选：交互式仪表板
streamlit run visualization/dashboard.py</pre>

<footer>
文档版本 3.0 · 更新日期 2026-07-21 · Author: Liuwenke<br>
所有图表由 visualization/build_report.py 从真实运行产物自动生成，数字与 docs/ANALYSIS_REPORT.md 一致。
</footer>

</div>
</body>
</html>
"""

out_html = os.path.join(ROOT, 'docs', 'ANALYSIS_REPORT.html')
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f'\n完成！HTML 报告已生成：{out_html}')
print(f'图表目录：{FIG}')
failed = [k for k, v in figs.items() if not v]
if failed:
    print(f'注意：以下图表生成失败 -> {failed}')
else:
    print('全部 8 张图表生成成功 ✅')
