# 美国交通事故数据分析报告 | US Traffic Accidents Data Analysis Report

> **数据说明（务必先读）**
> 本报告的全部数字均来自**在真实 US Accidents (March23) 数据集上运行本项目分析 pipeline 的实际结果**，可完整复现：
> ```bash
> # 将真实 US_Accidents_March23.csv（约 770 万条）放入 data/ 目录
> python run_all_analysis.py --sample 1500000       # 运行全部分析模块（时间序列自动吃全量）
> python visualization/build_report.py              # 生成图文版 ANALYSIS_REPORT.html
> ```
> 原始数据集约 **770 万条**记录（2016-01 ~ 2023-03，49 个州，47 个字段），可从 Kaggle 获取。受本机内存限制（约 5GB 空闲），各模块采用代表性大样本运行：EDA / 预测建模 / 空间聚类约 150 万条，NLP 文本分析 50 万条，**时间序列模块读取全部 770 万条**（仅取 3 列，内存可控）。下文所有"发现"均指"在该真实数据上的发现"，数字与 `docs/ANALYSIS_REPORT.html` 严格一致。

---

## 摘要 | Abstract

**中文**：本报告基于真实 **US Accidents (March23)** 数据集（约 770 万条、49 个州、2016–2023 年），使用探索性分析、预测建模、空间聚类、时间序列预测与 NLP 五个模块进行系统分析。核心发现：
(1) 事故在时间上呈现 8 时与 17 时附近的双峰（通勤特征），地理上分散于多州多市（CA 居首但仅约 22.5%）；
(2) **在真实数据上，事故严重程度是可被环境特征一定程度预测的**——4 分类模型准确率达约 80%，明显超越 62.9% 的多数类基线；将目标重定义为"是否严重（等级≥3）"的二分类后，F1 进一步提升至 **0.85**；
(3) K-Means 稳定识别出 15 个以大城市为中心的空间簇，DBSCAN 半径放大到约 36km 后识别出 57 个城市级热点（噪声仅 1.99%）；
(4) 月度事故量具强季节性与 COVID-19 扰动，朴素移动平均回测 MAPE 达 **44.6%**，线性 ARIMA 系数不显著，说明预测需季节性/事件特征。

**English**: This report analyzes the real **US Accidents (March23)** dataset (~7.7M records, 49 states, 2016–2023) with five modules: EDA, predictive modeling, spatial clustering, time-series forecasting, and NLP. Key findings: (1) a bimodal hourly pattern (8 AM / 5 PM commuting peaks) and geographically dispersed accidents (CA leads at ~22.5%); (2) on real data, accident severity **is partially predictable** — 4-class models reach ~80% accuracy, clearly above the 62.9% majority-class baseline, and a binary "severe (≥3)" target lifts F1 to **0.85**; (3) K-Means recovers 15 city-level clusters, and DBSCAN at ~36 km radius finds 57 urban hotspots (only 1.99% noise); (4) monthly volume is strongly seasonal with a COVID-19 dip — naive MA backtest MAPE is **44.6%** and linear ARIMA coefficients are insignificant, indicating seasonal/event features are needed.

**关键词 | Keywords**: 交通事故分析 | Traffic Accident Analysis; 机器学习 | Machine Learning; 空间聚类 | Spatial Clustering; 时间序列 | Time Series; 基线对照 | Baseline Comparison; 二分类 | Binary Classification

---

## 1. 引言 | Introduction

### 1.1 研究背景 | Background

交通事故是全球重要的公共安全问题。理解其时间、地理与环境影响因素，对交通安全管理具有参考价值。

### 1.2 研究目标 | Research Objectives

1. 揭示事故在小时 / 星期 / 月份层面的时间模式
2. 识别事故高发的地理区域与道路段
3. 评估天气等环境因素对事故发生与严重程度的影响
4. 构建并**严谨评估**事故严重程度预测模型（含基线对照，并补充二分类目标）
5. 从事故描述文本中提取事故类型与关键信息

### 1.3 数据来源 | Data Source

真实数据集为 "US Accidents (March 2023)"（Moosavi 等，2019），覆盖 2016 年 1 月至 2023 年 3 月、全美约 770 万条记录、47 个属性字段，可从 Kaggle 获取。本分析在本地对其运行完整 pipeline（受内存限制采用代表性大样本，详见上方"数据说明"），所有结论均基于真实数据。

---

## 2. 研究方法 | Methodology

### 2.1 分析流程 | Pipeline

数据加载 → 预处理（时间特征、持续时间、布尔字段、新特征）→ 分模块分析（EDA / 预测 / 空间 / 时序 / NLP）→ 输出 `output/` 与交互式仪表板。

### 2.2 关键方法学约定 | Methodological Conventions

为避免常见的"看起来漂亮但站不住"的结论，本报告遵循以下约定：

1. **模型评估必须对照基线（Baseline）**：预测建模以 `DummyClassifier(strategy='most_frequent')`（始终预测训练集最多数类）为参照。任何模型的指标只有**明显高于基线**才有意义。
2. **同时报告宏平均（Macro）与加权（Weighted）指标**：仅看加权 F1 会掩盖少数类的糟糕表现；宏平均 F1 能暴露模型对 1/4 级事故的预测能力。
3. **特征工程与目标重定义**：在真实数据上，将"严重事故（Severity≥3）"作为二分类目标，比原始 4 分类更易学习、预测力更强（详见 3.3）。新增特征包括 `Is_Night`、`Is_Weekend`、`Holiday`、`Description_Length`、`Street_Length`。
4. **分类变量关联用"卡方 + 效应量（Cramér's V）"**：大样本下卡方极易显著，"显著"本身无信息量；真正要看效应量。

---

## 3. 研究结果 | Results

### 3.1 数据概览 | Overview

| 指标 | 数值 |
|------|------|
| 原始数据集 | 约 7,728,394 条（全量） |
| EDA 分析样本 | 1,500,000 条（代表性抽样） |
| 字段数 | 47 |
| 时间范围 | 2016-01-14 ~ 2023-03-31 |
| 覆盖州数 | 49 |
| 覆盖城市数 | 11,414 |

**严重程度分布**（EDA 样本）：等级 2（中等）79.6%、等级 3（严重）16.9%、等级 4（极严重）2.7%、等级 1（轻微）0.9%。——**多数类（等级 2）约占 79.6%**。注意：预测建模因使用更多特征而丢弃缺失行，其子集内多数类比例降为约 62.9%（见 3.3 基线），两段分布差异源于特征完整度筛选，非数据错误。

### 3.2 探索性分析 | Exploratory Analysis

#### 3.2.1 时间模式 | Temporal

- **小时分布呈双峰**：峰值出现在 **8 时**与 **17 时**附近，凌晨 3–4 时最低，符合通勤高峰特征。
- **昼夜分布**：白天 79.3%，夜间 20.7%。
- **周×小时热力图**（见图文报告图 2b）显示工作日早晚高峰与周末午间次峰清晰可见。

#### 3.2.2 地理分布 | Geographic

| 排名 | 州 | 数量（EDA 样本） | 占比 |
|------|------|------|------|
| 1 | CA | 337,706 | 22.5% |
| 2 | FL | 169,965 | 11.3% |
| 3 | TX | 112,845 | 7.5% |
| 4 | SC | 74,716 | 5.0% |
| 5 | NY | 66,900 | 4.5% |

CA 占比约 **22.5%**，为多州中最高，但整体仍呈多州分散格局（无单一州占绝对多数）。

#### 3.2.3 天气条件 | Weather

| 天气状况 | 数量（EDA 样本） |
|----------|------|
| Fair（晴） | 496,833 |
| Mostly Cloudy（多云） | 196,921 |
| Cloudy（阴） | 158,652 |
| Clear（晴） | 157,159 |
| Partly Cloudy | 135,796 |

晴/多云类天气占绝对多数，雨雪等恶劣天气占比较小。

#### 3.2.4 数值特征相关性 | Correlation

数值特征间的 Pearson 相关系数普遍较弱（|r| < 0.3），说明各特征携带相对独立的信息，适合直接用于建模（详见图文报告图 4b 相关性热力图）。

### 3.3 预测建模 | Predictive Modeling

**任务**：基于天气、时间、地理、道路设施等特征预测事故严重程度。

#### 3.3.1 四分类（Severity 1–4）

| 模型 | 准确率 | 加权 F1 | 宏平均 F1 |
|------|--------|---------|-----------|
| **Baseline（最多数类）** | **62.85%** | 0.4851 | 0.1930 |
| XGBoost | 80.97% | 0.8005 | 0.5101 |
| Random Forest（class_weight=balanced） | 80.48% | **0.8056** | **0.5946** |
| Logistic Regression（class_weight=balanced） | 54.67% | 0.5972 | 0.3943 |

**结论**：与早期合成数据（模型未能超越基线）不同，**在真实数据上四分类模型明显超越基线**——XGBoost / Random Forest 准确率达约 80%（比 62.9% 基线高约 18 个百分点），说明天气、时间、设施等特征在真实数据中含有可学习的严重程度信号。但**宏平均 F1 仍仅 0.51–0.59**，表明 1/4 级这类少数事故预测力有限；若要进一步提升，方向应为构造更有信息量的特征（同址历史事故、路网密度、时段交互）。

#### 3.3.2 二分类（Severity≥3 = 严重）| Binary Classification

将问题重定义为"是否严重事故"（Severity ≥ 3 为正例）——这是 Kaggle 竞赛与多数同行采用的框架，因为 4 分类中 1/4 级样本极少导致信号弱，换成二分类后预测力通常显著提升。

| 模型 | 准确率 | 精确率 | 召回率 | F1 |
|------|--------|--------|--------|-----|
| **Baseline（最多数类）** | 65.41% | 0.4279 | 0.6541 | 0.5173 |
| Random Forest Binary | **84.59%** | 0.8535 | 0.8459 | **0.8479** |
| XGBoost Binary | 83.72% | 0.8368 | 0.8372 | 0.8370 |
| Logistic Regression Binary | 76.72% | 0.7912 | 0.7672 | 0.7722 |

**结论**：二分类最佳为 Random Forest（F1=0.848，准确率 84.6%），相比四分类基线（F1 0.193）与四分类最佳（F1 0.595）均有**质的提升**，验证了"目标重定义"这一决策的有效性。特征重要性中 `Description_Length`（描述长度）居首，印证了新增文本长度特征的价值。

### 3.4 空间聚类 | Spatial Clustering

#### K-Means（K=15）

K-Means 将事故稳定聚合成 **15 个**以大城市为核心的空间簇，各簇平均严重程度差异极小，说明**空间位置本身与严重程度关联很弱**（与预测建模结论一致：地理特征对严重程度信号有限）。

#### DBSCAN（eps=36.6km, city-level）

将半径从早期 1km 放大到约 **36.6km**（≈0.33° 经纬度，城市级热点尺度）后，成功识别出 **57 个城市级热点聚类**，噪声点仅占 **1.99%**。此前的 1km 半径在州级数据上过细（0 聚类），放大后结果与 K-Means 互相印证：事故围绕大城市呈聚集，且热点尺度在城市级而非 1km 点级。

### 3.5 时间序列预测 | Time Series Forecasting

基于**全部 770 万条**记录（有效时间数据约 698 万条）构建月度事故量时间序列（2016–2023，共 87 个月），取最后 30 期作为测试集回测：

| 指标 | 数值 |
|------|------|
| MAPE（朴素 7 期移动平均） | **44.61%** |
| MAE（平均绝对误差） | 43,640 起/月 |
| RMSE（均方根误差） | 49,498 起/月 |
| 月度均值 | 约 80,290 起 |

- 序列呈现**强季节性**（12 月峰值约 10.8 万起、夏季偏低）与 **COVID-19 扰动**（2020 年明显下凹）。
- ARIMA(1,1,1) 的自回归/移动平均系数均不显著（p>0.89），说明线性时序模型对季节性峰值与突发公共卫生事件捕捉力弱；朴素非季节性方法在此显著失效（MAPE 44.6%），需引入季节性（SARIMA）或外部事件特征才能有效预测。

### 3.6 NLP 文本分析 | NLP

- **高频词**：accident、due、blocked、lane、exit、ave、northbound、southbound、right、eastbound——方向词（north/south/east/westbound）大量出现，符合交通事故描述特征。
- 事故类型与道路类型经正则识别后按频次排序，结果见 `output/word_frequency.csv` 与图文报告图 8。

> 说明：NLP 计数是对真实事故描述文本的模式匹配，用于演示文本挖掘流程。

---

## 4. 讨论 | Discussion

1. **真实数据存在可学习信号**：与合成数据不同，真实 US Accidents 的环境特征对严重程度有可观预测力（4 分类 80% vs 基线 63%；二分类 F1 0.85）。这纠正了早期"预测能力有限"的初步印象——该印象源于合成数据信号弱。
2. **目标定义决定上限**：将 4 分类重定义为"是否严重（≥3）"二分类后，F1 从 0.59 跃升至 0.85，是本项目最有价值的建模决策。
3. **空间尺度匹配**：DBSCAN 半径需与问题尺度匹配——1km 在州级数据上过细（0 聚类），36.6km 城市级尺度才能发现热点（57 簇）。
4. **时序预测需季节性**：月度事故量强季节性 + COVID 扰动使朴素方法 MAPE 达 44.6%，ARIMA 系数不显著；真正的难点在"季节性峰值与异常事件（极端天气、大型活动、疫情）下的尖峰"，需要 SARIMA 或外部事件数据。

---

## 5. 结论与建议 | Conclusions

### 5.1 主要发现 | Key Findings

1. 事故在时间上呈 8 时 / 17 时双峰（通勤特征），地理上多州多市分散（CA 居首约 22.5%）。
2. 真实数据上环境特征对严重程度**有可学习信号**：4 分类模型 ~80% 准确率明显超越 62.9% 基线；二分类（≥3）F1 达 0.85。
3. K-Means 识别 15 个城市级空间簇；DBSCAN（36.6km）识别 57 个城市级热点（噪声 1.99%）。
4. 月度事故量具强季节性与 COVID 扰动，朴素移动平均 MAPE ≈ 44.6%，线性模型系数不显著——预测需季节性/事件特征。
5. 正确的评估框架（基线 + 宏平均 + 目标重定义）是获得可靠结论的关键。

### 5.2 建议 | Recommendations

1. **特征工程优先**：构造信息量更高的特征（同址历史事故、路网密度、时段×路网交互）。
2. **采用二分类目标**（已完成验证）："是否严重（等级≥3）"F1 达 0.85，远优于 4 分类。
3. **空间热点升级**：用 KDE 或更大半径 DBSCAN 识别城市级热点，服务于勤务部署。
4. **引入外部数据**：天气预警、节假日、大型活动、疫情等事件特征可显著提升时序尖峰预测（SARIMA / 事件回归）。

### 5.3 局限性 | Limitations

1. **内存受限的代表性抽样**：EDA / 预测 / 空间约 150 万条、NLP 50 万条（时间序列为全量），结论在大样本下稳健，但非逐行全量。
2. **特征信号仍有限**：4 分类宏平均 F1 仅 0.5–0.6，1/4 级少数事故预测力有限。
3. **因果性**：本研究识别相关性，不建立因果。
4. **NLP 依赖文本模板**：文本挖掘为流程演示，事故类型占比受描述文本质量影响。

### 5.4 未来工作 | Future Work

1. 在更高内存环境下尝试更大样本（或分块）复算。
2. 引入特征工程与替代建模目标（如事故时长、是否致命）。
3. 采用 SARIMA / Prophet 处理季节性，融合外部事件数据。
4. 用 KDE 做连续密度热点可视化。

---

## 参考文献 | References

1. Moosavi, S., Hosseini, S. M., & Sharma, A. (2019). US Accidents: A Countrywide Traffic Accident Dataset. *ACM SIGSPATIAL*.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
3. Ester, M., et al. (1996). A Density-Based Algorithm for Discovering Clusters (DBSCAN). *KDD*.
4. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.
5. National Highway Traffic Safety Administration. (2022). *Traffic Safety Facts 2021*.

---

## 附录 | Appendix

### A. 复现步骤 | Reproduction

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 将真实 US_Accidents_March23.csv 放入 data/
python run_all_analysis.py --sample 1500000
python visualization/build_report.py                # 生成图文版 HTML
streamlit run visualization/dashboard.py            # 可选：交互式仪表板
```

### B. 输出文件 | Outputs (`output/`)

- `model_comparison.csv`：4 分类各模型评估指标（含基线）
- `model_comparison_binary.csv`：二分类各模型评估指标（含基线）
- `kmeans_clusters.csv` / `dbscan_clusters.csv`：聚类结果
- `time_series_D.csv` / `ma_forecast_D.csv`：时间序列与预测
- `word_frequency.csv`：NLP 词频

> **图文版报告**：运行 `python visualization/build_report.py` 可自动生成带 10 张图表（含周×小时热力图、数值特征相关性热力图）的图文 HTML 报告 `docs/ANALYSIS_REPORT.html`（图表存于 `docs/figures/`）。其数字与本文档严格一致，适合作为作品集展示。

### C. 版本 | Versions

pandas 3.0.3 · numpy 2.5.1 · scikit-learn 1.9.0 · xgboost 3.3.0 · statsmodels 0.14.6 · streamlit 1.59.2（见 `requirements.txt`，均已验证可运行）

---

*文档版本: 3.0 | Document Version: 3.0*
*更新日期: 2026年7月21日 | Last Updated: July 21, 2026*
*作者: Liuwenke | Author: Liuwenke*
