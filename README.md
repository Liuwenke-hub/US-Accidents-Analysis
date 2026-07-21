# 美国交通事故数据分析 | US Accidents Data Analysis

<!-- Tech Stack -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">&nbsp;
  <img src="https://img.shields.io/badge/pandas-3.0.3-blue?logo=pandas&logoColor=white" alt="pandas">&nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-1.9.0-orange?logo=scikitlearn&logoColor=white" alt="scikit-learn">&nbsp;
  <img src="https://img.shields.io/badge/xgboost-3.3.0-red?logo=xgboost&logoColor=white" alt="xgboost">&nbsp;
  <img src="https://img.shields.io/badge/matplotlib-3.11-purple?logo=matplotlib&logoColor=white" alt="matplotlib">
</p>

<!-- Status -->
<p align="center">
  <img src="https://github.com/Liuwenke-hub/US-Accidents-Analysis/actions/workflows/test.yml/badge.svg" alt="CI">&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **一个完整的美国交通事故数据分析项目，涵盖探索性分析、预测建模、空间聚类、时间序列预测、NLP文本分析和交互式可视化**
>
> **A comprehensive data analysis project for US traffic accidents, covering EDA, predictive modeling, spatial clustering, time series forecasting, NLP, and interactive visualization**

---

## 摘要 | Abstract

**中文**：本项目基于真实美国交通事故数据集（US Accidents, March 2023，**约 770 万条**记录，2016–2023 年，49 个州）构建并运行了一套完整的数据分析框架，包含探索性数据分析、预测建模（含四分类与二分类）、空间聚类、时间序列预测和自然语言处理等模块。受单机内存限制，各模块采用代表性大样本（约 150 万条；NLP 50 万；时间序列读取全量 770 万条）运行，所有报告数字均可由 `run_all_analysis.py` 复现。仓库亦内置生成器可产出 50,000 条 schema 一致的**合成示例数据**用于无数据演示。

**English**: This project presents a comprehensive data analysis framework for traffic accident research in the United States. Using a schema-compatible synthetic sample of 50,000 records (mirroring the real US Accidents dataset, 2016–2023, ≈3.5M records), the framework implements multiple analytical modules including exploratory data analysis, predictive modeling, spatial clustering, time series forecasting, and natural language processing. The project demonstrates the application of modern data science techniques to uncover patterns and insights from large-scale traffic accident data, and the methodology transfers directly to the real dataset.

**关键词 | Keywords**: 交通安全 | Traffic Safety; 数据分析 | Data Analysis; 机器学习 | Machine Learning; 空间分析 | Spatial Analysis; 时间序列 | Time Series; NLP

---

## 1. 引言 | Introduction

### 1.1 研究背景 | Background

**中文**：交通事故是全球范围内的重要公共安全问题。在美国，2021年发生了超过600万起警方记录的交通事故，造成大量人员伤亡和财产损失。数据驱动的方法日益被认为是理解事故模式和制定有效预防策略的关键。

**English**: Traffic accidents remain a critical public safety issue globally. In the United States, over 6 million police-reported crashes occurred in 2021, resulting in significant loss of life and property damage. Data-driven approaches are increasingly recognized as essential for understanding accident patterns and developing effective prevention strategies.

### 1.2 研究目标 | Objectives

**中文**：

1. 开发一套全面的交通事故数据分析框架
2. 揭示事故的时间、地理和环境模式
3. 构建事故严重程度预测模型
4. 识别事故的空间聚类和热点区域
5. 提供交互式可视化工具用于数据探索

**English**:

1. Develop a comprehensive analytical framework for traffic accident data
2. Uncover temporal, geographic, and environmental patterns of accidents
3. Build predictive models for accident severity classification
4. Identify spatial clusters and hotspots of accidents
5. Provide interactive visualization tools for data exploration

### 1.3 主要贡献 | Contributions

| 中文 | English |
|------|---------|
| **模块化架构**：五个独立分析模块，标准化接口 | **Modular Architecture**: Five independent analysis modules with standardized interfaces |
| **机器学习集成**：XGBoost、Random Forest、Logistic Regression | **Machine Learning Integration**: XGBoost, Random Forest, and Logistic Regression |
| **空间分析**：K-Means和DBSCAN聚类算法 | **Spatial Analytics**: K-Means and DBSCAN clustering algorithms |
| **时间序列预测**：移动平均和ARIMA预测模型 | **Time Series Forecasting**: Moving average and ARIMA models |
| **自然语言处理**：事故描述文本挖掘 | **Natural Language Processing**: Text mining of accident descriptions |
| **交互式仪表板**：基于Streamlit的可视化工具 | **Interactive Dashboard**: Streamlit-based visualization tool |

---

## 2. 相关工作 | Related Work

### 2.1 交通事故分析 | Traffic Accident Analysis

**中文**：先前的研究从多个角度探索了交通事故，包括：时间模式（通勤高峰效应和季节性变化）、地理分析（空间聚类和热点识别）、天气影响（天气条件与事故严重程度的相关性）、机器学习（事故严重程度预测模型）。

**English**: Previous studies have explored various aspects of traffic accidents, including: temporal patterns (rush hour effects and seasonal variations), geographic analysis (spatial clustering and hotspot identification), weather impacts (correlation between weather conditions and accident severity), and machine learning (predictive models for accident severity).

### 2.2 数据集 | Dataset

**中文**："US Accidents"数据集（Moosavi等，2019）已被广泛应用于交通安全研究，提供事故位置、时间戳、天气状况和道路设施等详细信息。

**English**: The "US Accidents" dataset (Moosavi et al., 2019) has been widely used in traffic safety research, providing detailed information on accident locations, timestamps, weather conditions, and road infrastructure.

---

## 3. 研究方法 | Methodology

### 3.1 系统架构 | System Architecture

**中文**：项目采用模块化架构，包含四个层次：

**English**: The project follows a modular architecture with four layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 | Data Layer                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  US_Accidents_March23.csv（真实：约 770 万条；演示：5万条合成样例可选） │  │
│  │  (real: ~7.7M records; optional 50k synthetic demo)        │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   工具层 | Utility Layer                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  数据加载 | 预处理 | 内存优化 | 输出管理 | 依赖处理         │  │
│  │  Data Loading | Preprocessing | Memory Optimization     │  │
│  │  Output Management | Dependency Handling                │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   分析层 | Analysis Layer                        │
│  ┌──────────┐ ┌──────────────┐ ┌───────────────┐ ┌───────────┐ │
│  │  EDA     │ │  预测建模     │ │  空间聚类      │ │  时间序列  │ │
│  │  探索分析 │ │  Predictive  │ │  Spatial      │ │  Time     │ │
│  └──────────┘ └──────────────┘ └───────────────┘ └───────────┘ │
│  ┌──────────┐                                                   │
│  │  NLP     │                                                   │
│  │  文本分析 │                                                   │
│  └──────────┘                                                   │
└──────────────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   可视化层 | Visualization Layer                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Streamlit仪表板 | Plotly图表 | 静态报告                   │  │
│  │  Streamlit Dashboard | Plotly Charts | Static Reports   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块说明 | Modules Description

#### 3.2.1 探索性数据分析 | Exploratory Data Analysis (EDA)

| 中文 | English |
|------|---------|
| **用途**：全面的数据质量评估和模式识别 | **Purpose**: Comprehensive data quality assessment and pattern identification |
| 数据类型和缺失值分析 | Data type and missing value analysis |
| 时间分布（小时/日/周/月） | Temporal distribution (hourly/daily/weekly/monthly) |
| 地理分布（州/城市/县） | Geographic distribution (state/city/county) |
| 天气条件分析 | Weather condition analysis |
| 道路设施影响评估 | Road infrastructure impact assessment |
| 相关性分析 | Correlation analysis |

**文件 | File**: `analysis/accident_analysis.py`

#### 3.2.2 预测建模 | Predictive Modeling

| 中文 | English |
|------|---------|
| **用途**：基于环境因素预测事故严重程度 | **Purpose**: Predict accident severity based on environmental factors |
| 特征工程（20+衍生特征） | Feature engineering (20+ derived features) |
| 三种模型对比：XGBoost、Random Forest、Logistic Regression | Model comparison: XGBoost, Random Forest, Logistic Regression |
| 评估指标：准确率、精确率、召回率、F1、混淆矩阵 | Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix |
| 特征重要性分析 | Feature importance analysis |

**文件 | File**: `analysis/predictive_modeling.py`

#### 3.2.3 空间聚类 | Spatial Clustering

| 中文 | English |
|------|---------|
| **用途**：识别事故热点和空间模式 | **Purpose**: Identify accident hotspots and spatial patterns |
| K-Means聚类（肘部法则选最优K） | K-Means clustering with elbow method |
| DBSCAN密度聚类（任意形状检测） | DBSCAN density clustering |
| 噪声点识别 | Noise point identification |
| 热点排名和严重程度分析 | Hotspot ranking and severity analysis |

**文件 | File**: `analysis/spatial_clustering.py`

#### 3.2.4 时间序列预测 | Time Series Forecasting

| 中文 | English |
|------|---------|
| **用途**：预测未来事故趋势 | **Purpose**: Predict future accident trends |
| 趋势和季节性分解 | Trend and seasonal decomposition |
| 移动平均预测 | Moving average prediction |
| ARIMA模型（需statsmodels） | ARIMA model (requires statsmodels) |
| 回测评估（MAPE、MAE） | Backtesting with MAPE and MAE metrics |

**文件 | File**: `analysis/time_series_forecasting.py`

#### 3.2.5 自然语言处理 | Natural Language Processing

| 中文 | English |
|------|---------|
| **用途**：从事故描述文本中提取信息 | **Purpose**: Extract insights from accident description text |
| 词频分析 | Word frequency analysis |
| 停用词过滤和二元词组分析 | Stopword filtering and Bigram analysis |
| 事故类型识别（18种类别） | Accident type identification (18 categories) |
| 道路信息提取 | Road information extraction |
| 严重程度关键词对比 | Severity-based keyword comparison |
| 词云生成 | Word cloud generation |

**文件 | File**: `analysis/nlp_analysis.py`

#### 3.2.6 交互式仪表板 | Interactive Dashboard

| 中文 | English |
|------|---------|
| **用途**：可视化数据探索和展示 | **Purpose**: Visual data exploration and presentation |
| 六个功能模块，标签页导航 | Six functional modules with tab-based navigation |
| 实时筛选（州、严重程度、年份、采样量） | Real-time filtering |
| 地理散点图 | Geographic scatter plot |
| 交互式图表（悬停查看详情） | Interactive charts with hover tooltips |

**文件 | File**: `visualization/dashboard.py`

### 3.3 数据预处理流程 | Data Preprocessing Pipeline

| 步骤 | 中文 | English |
|------|------|---------|
| 1 | 加载：分块读取CSV文件 | Loading: Read CSV file with optimized chunking |
| 2 | 类型转换：转换时间戳和分类字段 | Type Conversion: Convert timestamps and categorical fields |
| 3 | 特征工程：提取时间特征并计算持续时间 | Feature Engineering: Extract temporal features and calculate duration |
| 4 | 缺失值处理：插补或排除 | Missing Value Handling: Imputation or exclusion |
| 5 | 内存优化：数值类型降级 | Memory Optimization: Downcast numerical types |
| 6 | 采样：分层随机抽样 | Sampling: Stratified sampling |

### 3.4 评估指标 | Evaluation Metrics

| 模块 | 中文 | English |
|------|------|---------|
| 预测建模 | 准确率、精确率、召回率、F1（加权+宏平均）、混淆矩阵 | Accuracy, Precision, Recall, F1 (weighted + macro), Confusion Matrix |
| 时间序列 | MAPE、MAE、RMSE | MAPE, MAE, RMSE |
| 空间聚类 | 轮廓系数、Calinski-Harabasz指数 | Silhouette Score, Calinski-Harabasz Index |

---

## 4. 实现 | Implementation

### 4.1 技术栈 | Technical Stack

| 类别 | 中文 | 工具 | English |
|------|------|------|---------|
| 编程语言 | Python 3.9+ | | Python 3.9+ |
| 数据处理 | 数据操作 | Pandas, NumPy | Data Manipulation |
| 机器学习 | 机器学习 | Scikit-learn, XGBoost | Machine Learning |
| 统计分析 | 统计分析 | SciPy, Statsmodels | Statistical Analysis |
| 可视化 | 可视化 | Matplotlib, Seaborn, Plotly | Visualization |
| 交互式仪表板 | 交互式仪表板 | Streamlit | Interactive Dashboard |
| 自然语言处理 | 文本分析 | NLTK, WordCloud | NLP |

### 4.2 项目结构 | Project Structure

```
US_Accidents_Analysis/
├── analysis/                    # 分析模块 | Analysis modules
│   ├── __init__.py
│   ├── accident_analysis.py     # 探索性数据分析 | Exploratory Data Analysis
│   ├── predictive_modeling.py   # 预测建模 | Predictive Modeling
│   ├── spatial_clustering.py    # 空间聚类 | Spatial Clustering
│   ├── time_series_forecasting.py # 时间序列预测 | Time Series Forecasting
│   └── nlp_analysis.py          # NLP文本分析 | NLP Text Analysis
├── utils/                       # 工具模块 | Utility functions
│   ├── __init__.py
│   └── tools.py                 # 通用工具函数 | Data loading, preprocessing
├── visualization/               # 可视化模块 | Visualization modules
│   ├── __init__.py
│   └── dashboard.py             # Streamlit仪表板 | Streamlit Dashboard
├── data/                        # 数据目录 | Data directory
│   └── US_Accidents_March23.csv # 原始数据集 | Raw dataset
├── output/                      # 输出目录 | Output directory (generated)
├── docs/                        # 文档 | Documentation
│   ├── CODE_WIKI.md             # 代码文档 | Code documentation
│   ├── ANALYSIS_REPORT.md       # 分析报告（纯文字）| Text analysis report
│   ├── ANALYSIS_REPORT.html     # 图文报告（自动生成）| Illustrated HTML report
│   └── figures/                 # 报告图表（自动生成）| Report figures (auto-generated)
├── run_all_analysis.py          # 批量执行入口 | Entry point for batch execution
├── requirements.txt             # 依赖清单 | Dependencies
├── README.md                    # 本文件 | This document
├── LICENSE                      # MIT许可证 | MIT License
└── .gitignore                   # Git忽略规则 | Git ignore rules
```

### 4.3 安装 | Installation

```bash
# 克隆仓库 | Clone repository
git clone https://github.com/Liuwenke-hub/US-Accidents-Analysis.git
cd US-Accidents-Analysis

# 创建虚拟环境 | Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖 | Install dependencies
pip install -r requirements.txt
```

### 4.4 数据准备 | Data Preparation

**方式一（推荐，无需下载）：使用合成示例数据**
项目内置数据生成器，可生成一份与真实 schema 一致的示例数据集，便于在没有 Kaggle 原始数据（约 1.5GB）时端到端运行全部分析：

```bash
python utils/generate_sample_data.py --n 50000
```

生成文件位于 `data/US_Accidents_March23.csv`。**注意：该数据为合成数据（synthetic），用于演示，并非真实事故记录；报告中的分析结果均基于所使用的数据运行得出。**

**方式二：使用 Kaggle 真实数据集**
从 Kaggle 下载数据集并放入 `data/` 目录：
- 数据集地址：https://www.kaggle.com/sobhanmoosavi/us-accidents
- 文件名：`US_Accidents_March23.csv`

**English**:
*Option A (recommended, no download):* use the built-in synthetic data generator to create a schema-compatible sample, then run the full pipeline:
```bash
python utils/generate_sample_data.py --n 50000
```
*Note: the generated data is synthetic and for demonstration only; all reported results are produced by running the pipeline on the data actually used.*
*Option B:* Download the real Kaggle dataset and place `US_Accidents_March23.csv` under `data/`.

### 4.5 运行 | Execution

#### 全量分析 | Full Analysis

```bash
# 真实数据代表性大样本（推荐；时间序列模块自动读取全部 770 万条）
python run_all_analysis.py --sample 1500000
```

#### 单独模块 | Individual Modules

```bash
# 探索性数据分析 | Exploratory Data Analysis
python -m analysis.accident_analysis --sample 1500000

# 预测建模（含四分类 + 二分类）| Predictive Modeling
python -m analysis.predictive_modeling --sample 1500000

# 空间聚类（DBSCAN 默认 36.6km 城市级）| Spatial Clustering
python -m analysis.spatial_clustering --sample 1500000 --method both --eps 36.6

# 时间序列预测（全量 770 万条）| Time Series Forecasting
python -m analysis.time_series_forecasting --sample 0 --freq D

# NLP分析 | NLP Analysis
python -m analysis.nlp_analysis --sample 1500000

# 交互式仪表板 | Interactive Dashboard
streamlit run visualization/dashboard.py
```

### 4.6 命令行参数 | Command Line Arguments

| 模块 | 参数 | 类型 | 默认值 | 中文说明 | English |
|------|------|------|--------|----------|---------|
| 全部 | `--sample` | int | 50000 | 采样数量（0=全量） | Sample size (0 = full) |
| 预测 | `--model` | str | all | 模型类型 | Model type |
| 空间 | `--method` | str | both | 聚类方法 | Clustering method |
| 空间 | `--k` | int | 15 | K-Means聚类数 | K-Means clusters |
| 空间 | `--eps` | float | 36.6 | DBSCAN半径(km)，城市级热点 | DBSCAN radius (city-level) |
| 时间序列 | `--freq` | str | D | 频率(D/W/M) | Frequency |
| 时间序列 | `--horizon` | int | 30 | 预测范围 | Prediction horizon |
| NLP | `--wordcloud` | flag | - | 生成词云 | Generate word cloud |

---

### 4.7 运行测试 | Running Tests

项目附带单元测试，覆盖数据加载、基础预处理、共享常量与评估函数：

```bash
# 使用已装好依赖的虚拟环境
.venv/Scripts/python.exe -m unittest discover -s tests -t . -v
```

测试位于 `tests/`：
- `test_utils.py`：数据加载、时间/布尔预处理、共享常量（`FACILITY_COLS` / `ENGLISH_STOPWORDS` / `SEVERITY_DESC`），以及 `accident_analysis` 复用 `utils` 的去重断言
- `test_predictive.py`：基线模型、评估指标、特征工程

**持续集成（CI）**：仓库已配置 GitHub Actions（`.github/workflows/test.yml`）。每次推送到主分支或提交 Pull Request 时，GitHub 会自动开一台 Linux 临时机，安装依赖并运行上述单元测试，结果以 ✅ / ❌ 状态显示在提交记录上。无需本地手动触发。

---

## 5. 研究结果 | Results

> 以下结果均来自**真实 US Accidents (March23) 数据集**的运行（EDA/预测/空间约 150 万条抽样，NLP 50 万条，时间序列全量 770 万条）。详细数字与图表见 `docs/ANALYSIS_REPORT.md` 与 `docs/ANALYSIS_REPORT.html`。

### 5.1 探索性数据分析 | Exploratory Data Analysis

- **时间**：事故在 **8 时与 17 时附近呈双峰**（通勤特征）；白天 79.3% / 夜间 20.7%
- **地理**：CA 居首约 **22.5%**，其次 FL 11.3%、TX 7.5%、SC 5.0%、NY 4.5%；多州分散
- **天气**：Fair/多云类占绝对多数，恶劣天气占比较小
- **严重程度**：等级 2（中等）79.6%、等级 3（严重）16.9%、等级 1 0.9%、等级 4 2.7%
- **周×小时交叉**：工作日呈清晰双峰（早 8 时约 **8.6%**、晚 17 时约 **7.8%**），与通勤高峰吻合；周末无早高峰，事故集中在 **12–17 时午间**（13–14 时约 **6.5%**），呈单峰"休闲出行"形态
- **特征相关性**：数值特征间相关性整体较弱，最强对为湿度–能见度（r=−0.39）与温度–湿度（r=−0.33），最大 |r| 仅 **0.39**，远低于共线阈值，各特征可独立用于建模

> 以下图表由 `visualization/build_report.py` 基于真实数据（采样 150 万条）生成，与上方文字一一对应。

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/severity_dist.png" alt="图1: 事故严重程度分布">
  <br>
  <em>图 1：事故严重程度分布——以 2 级（中等）为主。</em>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/state_top10.png" alt="图2: 各州事故量 Top10">
  <br>
  <em>图 2：各州事故量 Top10——CA 居首。</em>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/hourly.png" alt="图3: 按小时分布">
  <br>
  <em>图 3：按小时分布——8 时与 17 时双峰。</em>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/weather.png" alt="图4: 天气条件分布">
  <br>
  <em>图 4：天气条件分布——Fair/多云占绝对多数。</em>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/heatmap_wh.png" alt="图5: 周×小时热力图">
  <br>
  <em>图 5：周×小时热力图——工作日早晚双峰、周末午间单峰清晰可见。</em>
</p>

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/correlation.png" alt="图6: 数值特征相关性矩阵">
  <br>
  <em>图 6：数值特征相关性——整体弱相关，无共线性风险。</em>
</p>

### 5.2 预测建模 | Predictive Modeling

**四分类（含基线）| 4-class (with baseline)**:

| 模型 | 准确率 | 加权 F1 | 宏平均 F1 |
|-------|----------|-----------|--------|
| Baseline（最多数类） | **62.85%** | 0.4851 | 0.1930 |
| XGBoost | 80.97% | 0.8005 | 0.5101 |
| Random Forest | 80.48% | **0.8056** | **0.5946** |
| Logistic Regression | 54.67% | 0.5972 | 0.3943 |

**结论**：与合成数据不同，真实数据上四分类模型明显超越基线（~80% vs 63%），说明环境特征含可学习信号；但宏平均 F1 仍仅 0.5–0.6，1/4 级少数事故预测力有限。

**二分类（Severity≥3 = 严重）| Binary**: Random Forest **F1=0.848**（准确率 84.6%），XGBoost 0.837，Logistic Regression 0.772，基线 0.517——显著优于四分类，验证了目标重定义（新增 `Is_Night`/`Is_Weekend`/`Holiday`/`Description_Length`/`Street_Length` 等特征）的有效性。

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/model_compare.png" alt="图7: 预测模型对比">
  <br>
  <em>图 7：预测模型对比（四分类准确率 / F1）。</em>
</p>

### 5.3 空间聚类 | Spatial Clustering

- **K-Means（K=15）**：识别出 **15 个**以大城市为核心的空间簇，各簇平均严重程度差异极小（位置与严重程度关联弱）
- **DBSCAN（eps=36.6km, city-level）**：识别出 **57 个**城市级热点聚类，噪声仅 **1.99%**（早期 1km 半径在州级数据上过细，0 聚类）

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/kmeans_map.png" alt="图8: K-Means 空间聚类">
  <br>
  <em>图 8：K-Means 空间聚类（K=15，以大城市为核心）。</em>
</p>

### 5.4 时间序列预测 | Time Series Forecasting

- 基于**全部 770 万条**构建月度序列（87 个月），呈**强季节性**（12 月峰值约 10.8 万起）+ **COVID-19 扰动**（2020 下凹）
- 朴素 7 期移动平均回测 **MAPE 44.6%**，ARIMA(1,1,1) 系数不显著（p>0.89）→ 预测需季节性（SARIMA）/事件特征

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/timeseries.png" alt="图9: 月度时间序列趋势">
  <br>
  <em>图 9：月度时间序列趋势——强季节性（12 月峰值）+ COVID-19 扰动（2020 下凹）。</em>
</p>

### 5.5 NLP分析 | NLP Analysis

- **高频词**：accident、due、blocked、lane、exit、northbound/southbound/eastbound 等（方向词大量出现）
- 事故类型与道路类型经正则识别，结果见 `output/word_frequency.csv`

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Liuwenke-hub/US-Accidents-Analysis@main/docs/figures/topwords.png" alt="图10: 事故描述高频词">
  <br>
  <em>图 10：事故描述高频词——方向词（northbound/southbound 等）大量出现。</em>
</p>

---

## 6. 讨论 | Discussion

### 6.1 关键发现 | Key Insights

**中文**（以下均基于真实 770 万条数据的代表性大样本运行结果）：

1. **通勤双峰显著**：事故在 8 时与 17 时附近呈清晰双峰，与早晚通勤高峰吻合，提示交通流量是事故时间分布的主要驱动。
2. **真实数据存在可学习信号**：四分类模型准确率 ~80% 明显超越 62.9% 基线；二分类（≥3）F1 达 0.85——目标重定义（"是否严重"）是本项目的核心建模决策。
3. **空间尺度决定结论**：K-Means（K=15）稳定分出大城市聚类；DBSCAN 放大到 36.6km 后识别出 57 个城市级热点（噪声 1.99%），而 1km 在州级数据上过细（0 聚类）；各聚类平均严重程度差异极小，空间位置与严重程度关联弱。
4. **时序需季节性模型**：月度事故量强季节性 + COVID 扰动，朴素移动平均 MAPE 达 44.6%，ARIMA 系数不显著——预测需 SARIMA 或外部事件特征。

**English** (all findings below are from representative large-sample runs on the real 7.7M-record dataset):

1. **Clear bimodal rush-hour pattern**: accidents peak near 8 AM and 5 PM, consistent with commuting peaks.
2. **Real data carries learnable signal**: 4-class models reach ~80% accuracy vs. the 62.9% baseline; binary (≥3) F1 hits 0.85 — target redefinition ("severe or not") is the key modeling decision.
3. **Spatial scale drives the conclusion**: K-Means (K=15) separates metro clusters; DBSCAN at 36.6km finds 57 urban hotspots (1.99% noise), while 1km is too fine (0 clusters) at state scale.
4. **Time series needs seasonality**: monthly volume is strongly seasonal with a COVID dip; naive MA gives MAPE 44.6% and ARIMA coefficients are insignificant — SARIMA or event features are required.

### 6.2 局限性 | Limitations

**中文**：
1. **数据偏差**：示例数据为合成数据，州分布经过设计；若使用真实数据集，需注意各州覆盖不均带来的推广性限制
2. **缺失数据**：部分天气属性（如降水量、风寒温度）在真实数据集中缺失率较高
3. **因果性**：相关分析无法建立因果关系
4. **时间范围**：示例数据覆盖 2016-2023 年，后续可扩展到更新或更早的数据

**English**:
1. **Data Bias**: The demo uses synthetic data with a designed state distribution; with the real dataset, note the generalizability limits from uneven state coverage
2. **Missing Data**: Some weather attributes (e.g., precipitation, wind chill) have high missing rates in the real dataset
3. **Causality**: Correlational analysis cannot establish causal relationships
4. **Temporal Scope**: The demo data spans 2016-2023 and can be extended to newer or earlier periods

### 6.3 未来工作 | Future Work

**中文**：
1. **多州分析**：扩展到全美各州数据
2. **深度学习**：实现神经网络预测模型
3. **实时预测**：开发流式分析进行事故预测
4. **因果推断**：应用因果机器学习技术
5. **GIS集成**：整合地理信息系统进行高级空间分析

**English**:
1. **Multi-state Analysis**: Expand to include data from all US states
2. **Deep Learning**: Implement neural network models for prediction
3. **Real-time Prediction**: Develop streaming analytics for accident forecasting
4. **Causal Inference**: Apply causal machine learning techniques
5. **GIS Integration**: Incorporate geographic information systems for advanced spatial analysis

---

## 7. 应用场景 | Applications

### 7.1 交通安全管理 | Traffic Safety Management

| 中文 | English |
|------|---------|
| 热点识别：优先进行道路维护和安全改善 | Hotspot Identification: Prioritize road maintenance and safety improvements |
| 高峰管理：优化高峰时段交通信号配时 | Rush Hour Management: Optimize traffic signal timing during peak periods |
| 天气预警：开展针对恶劣天气的安全宣传 | Weather Alerts: Develop targeted safety campaigns |

### 7.2 城市规划 | Urban Planning

| 中文 | English |
|------|---------|
| 基础设施投资：指导交通信号灯和停车标志的设置 | Infrastructure Investment: Guide placement of traffic signals and stop signs |
| 高速公路设计：为事故多发路段的改善提供依据 | Highway Design: Inform improvements to accident-prone segments |
| 公共交通：识别可通过公共交通缓解拥堵的区域 | Public Transit: Identify areas where public transit could reduce congestion |

### 7.3 政策制定 | Policy Making

| 中文 | English |
|------|---------|
| 安全法规：支持基于证据的政策决策 | Safety Regulations: Support evidence-based policy decisions |
| 资源配置：优化交通安全资源的分配 | Resource Allocation: Optimize distribution of traffic safety resources |
| 公众教育：开展有针对性的教育宣传 | Public Awareness: Develop targeted educational campaigns |

---

## 8. 结论 | Conclusion

**中文**：本项目构建了一个全面的交通事故数据分析框架。通过整合探索性分析、预测建模、空间聚类、时间序列预测和NLP，该框架提供了理解交通事故模式的整体方法。模块化架构允许灵活部署和扩展，适用于学术研究和交通安全管理的实际应用。

**English**: This project presents a comprehensive data analysis framework for traffic accident research. By integrating exploratory analysis, predictive modeling, spatial clustering, time series forecasting, and NLP, the framework provides a holistic approach to understanding traffic accident patterns. The modular architecture allows for flexible deployment and extension, making it suitable for both academic research and practical applications in traffic safety management.

---

## 参考文献 | References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

2. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining*, 226-231.

3. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.

4. Moosavi, S., Hosseini, S. M., & Sharma, A. (2019). US Accidents: A Countrywide Traffic Accident Dataset. *Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, 363-370.

5. National Highway Traffic Safety Administration. (2022). *Traffic Safety Facts 2021*. Retrieved from https://www.nhtsa.gov/sites/nhtsa.gov/files/2022-09/TSF_2021.pdf

---

## 附录 | Appendix

### A. 依赖清单 | Requirements

完整且已锁定的依赖以仓库根目录 **`requirements.txt`** 为准（已验证在当前环境可完整安装并端到端运行 `run_all_analysis.py`）。核心依赖如下（精确版本，避免与环境漂移）：

```
pandas==3.0.3
numpy==2.5.1
scikit-learn==1.9.0
xgboost==3.3.0
scipy==1.18.0
statsmodels==0.14.6
matplotlib==3.11.1
seaborn==0.13.2
plotly==6.9.0
nltk==3.10.0
wordcloud==1.9.6
streamlit==1.59.2
```

> 安装方式见第 4.3 节：`pip install -r requirements.txt`。本附录不再单独维护版本号，避免与 `requirements.txt` 不一致。

### B. 许可证 | License

**中文**：本项目采用MIT许可证 - 详见LICENSE文件

**English**: This project is licensed under the MIT License - see the LICENSE file for details

### C. 联系方式 | Contact

- 作者 | Author: Liuwenke
- GitHub: https://github.com/Liuwenke-hub
- 仓库 | Repository: https://github.com/Liuwenke-hub/US-Accidents-Analysis

---

*版本 3.0 | Version 3.0（基于真实 770 万条数据重跑，新增二分类 / DBSCAN 36.6km / 新特征 / 周×小时与相关性双图）*
*2026年7月21日 | July 21, 2026*
