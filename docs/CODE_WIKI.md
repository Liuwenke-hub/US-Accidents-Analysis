# US Accidents March 2023 - Code Wiki

## 1. 项目概述

### 1.1 项目简介

本项目是一个**美国交通事故数据集**（US Accidents March 2023 Edition），包含了2016年至2023年3月期间美国全境的交通事故记录。该数据集可用于交通事故分析、预测建模、地理空间分析等数据科学研究。

### 1.2 项目基本信息

| 属性 | 值 |
|------|-----|
| 项目名称 | US_Accidents_March23 |
| 数据版本 | March 2023 |
| 数据格式 | CSV (Comma-Separated Values) |
| 数据文件 | `US_Accidents_March23.csv` |
| 文件大小 | ~3.06 GB |
| 记录数量 | 7,728,394 条 |
| 字段数量 | 47 个 |
| 开发环境 | Python 3.14 |
| IDE | PyCharm / IntelliJ IDEA |

### 1.3 项目目录结构

```
US_Accidents_March23/
├── .idea/                          # IntelliJ IDEA 项目配置
│   ├── inspectionProfiles/
│   │   └── profiles_settings.xml
│   ├── .gitignore
│   ├── US_Accidents_March23.iml    # 模块配置文件
│   └── modules.xml
├── output/                         # 分析输出目录（自动生成）
├── US_Accidents_March23.csv        # 主数据文件
├── CODE_WIKI.md                    # 项目文档（本文件）
├── requirements.txt                # 依赖库清单
│
│── 基础分析模块
├── accident_analysis.py            # 基础探索性数据分析（EDA）
│
│── 进阶分析模块
├── utils.py                        # 通用工具函数
├── predictive_modeling.py          # 预测建模（XGBoost/RF/LR）
├── spatial_clustering.py           # 空间聚类（KMeans/DBSCAN）
├── time_series_forecasting.py      # 时间序列预测（ARIMA/MA）
├── nlp_analysis.py                 # NLP文本分析（词频/词云/分类）
│
│── 可视化模块
├── dashboard.py                    # Streamlit交互式仪表板
│
└── run_all_analysis.py             # 综合分析主入口（一键运行）
```

---

## 2. 数据架构

### 2.1 数据字段分类

数据集包含47个字段，可分为以下几大类：

#### 2.1.1 标识与来源类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `ID` | string | 事故唯一标识符，格式为 `A-<数字>` |
| `Source` | string | 数据来源（Source1/Source2/Source3） |

#### 2.1.2 严重程度类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Severity` | int | 事故严重程度，取值1-4（1=最轻，4=最严重） |

#### 2.1.3 时间类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Start_Time` | string | 事故开始时间（格式：YYYY-MM-DD HH:MM:SS） |
| `End_Time` | string | 事故结束时间 |
| `Weather_Timestamp` | string | 天气数据采集时间戳 |

#### 2.1.4 地理位置类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Start_Lat` | float | 事故起始点纬度 |
| `Start_Lng` | float | 事故起始点经度 |
| `End_Lat` | float | 事故结束点纬度（可能为空） |
| `End_Lng` | float | 事故结束点经度（可能为空） |
| `Distance(mi)` | float | 受事故影响的道路距离（英里） |
| `Street` | string | 街道名称 |
| `City` | string | 城市 |
| `County` | string | 县 |
| `State` | string | 州（缩写） |
| `Zipcode` | string | 邮政编码 |
| `Country` | string | 国家（US） |
| `Timezone` | string | 时区（如 US/Eastern） |
| `Airport_Code` | string | 附近机场代码（天气数据源） |

#### 2.1.5 描述类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Description` | string | 事故描述文本 |

#### 2.1.6 天气类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Temperature(F)` | float | 温度（华氏度） |
| `Wind_Chill(F)` | float | 风寒温度（华氏度） |
| `Humidity(%)` | float | 湿度（百分比） |
| `Pressure(in)` | float | 气压（英寸汞柱） |
| `Visibility(mi)` | float | 能见度（英里） |
| `Wind_Direction` | string | 风向 |
| `Wind_Speed(mph)` | float | 风速（英里/小时） |
| `Precipitation(in)` | float | 降水量（英寸） |
| `Weather_Condition` | string | 天气状况描述 |

#### 2.1.7 道路设施类（布尔型）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Amenity` | bool | 附近是否有便利设施 |
| `Bump` | bool | 附近是否有减速带 |
| `Crossing` | bool | 附近是否有路口/人行横道 |
| `Give_Way` | bool | 附近是否有让行标志 |
| `Junction` | bool | 附近是否有交叉口 |
| `No_Exit` | bool | 附近是否有死胡同 |
| `Railway` | bool | 附近是否有铁路 |
| `Roundabout` | bool | 附近是否有环岛 |
| `Station` | bool | 附近是否有车站 |
| `Stop` | bool | 附近是否有停车标志 |
| `Traffic_Calming` | bool | 附近是否有交通减速设施 |
| `Traffic_Signal` | bool | 附近是否有交通信号灯 |
| `Turning_Loop` | bool | 附近是否有转弯环路 |

#### 2.1.8 日出日落类

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `Sunrise_Sunset` | string | 日出/日落状态（Day/Night） |
| `Civil_Twilight` | string | 民用晨昏状态 |
| `Nautical_Twilight` | string | 航海晨昏状态 |
| `Astronomical_Twilight` | string | 天文晨昏状态 |

### 2.2 数据字段详细说明

#### 2.2.1 Severity（严重程度）

严重程度分为4个等级：
- **1**：轻微事故，对交通影响最小
- **2**：中等事故，有一定交通影响
- **3**：严重事故，造成较大交通延误
- **4**：极严重事故，可能造成道路封闭

#### 2.2.2 Source（数据来源）

数据来自多个数据源：
- **Source1**：来源1（具体来源需参考原始数据集说明）
- **Source2**：来源2
- **Source3**：来源3

#### 2.2.3 日出日落字段说明

四个层级的日光状态：
- **Sunrise_Sunset**：基本的白天/黑夜判断
- **Civil_Twilight**：民用晨光/昏影（太阳在地平线以下6度内）
- **Nautical_Twilight**：航海晨光/昏影（太阳在地平线以下12度内）
- **Astronomical_Twilight**：天文晨光/昏影（太阳在地平线以下18度内）

---

## 3. 数据质量分析

### 3.1 缺失值情况

基于1000条样本数据的初步分析：

| 字段 | 缺失数量 | 缺失率 |
|------|----------|--------|
| `End_Lat` | 全部 | ~100% |
| `End_Lng` | 全部 | ~100% |
| `Precipitation(in)` | 较高 | ~81% |
| `Wind_Chill(F)` | 较高 | ~54% |
| `Wind_Speed(mph)` | 中等 | ~5% |
| `Visibility(mi)` | 低 | ~0.3% |
| 其他天气字段 | 低 | <0.3% |

> **注意**：`End_Lat` 和 `End_Lng` 在样本中全部为空，可能是该版本数据集中这些字段未填充。

### 3.2 数据质量建议

1. **天气数据缺失**：部分天气字段存在缺失，分析时需进行缺失值处理
2. **End坐标缺失**：事故终点坐标大量缺失，仅使用起点坐标进行空间分析
3. **降水量数据缺失率高**：`Precipitation(in)` 字段缺失率高，使用时需谨慎
4. **时间字段格式**：时间字段为字符串格式，使用前需转换为datetime类型

---

## 4. 环境与依赖

### 4.1 开发环境

- **Python 版本**：3.14.x
- **主要IDE**：PyCharm / IntelliJ IDEA（Python插件）
- **项目类型**：Python Module

### 4.2 已安装的关键库

| 库名 | 版本 | 用途 |
|------|------|------|
| `pandas` | 3.0.3 | 数据分析与处理 |
| `numpy` | 2.4.6 | 数值计算 |

### 4.3 推荐安装的扩展库

根据项目特点，建议安装以下库：

```bash
# 数据可视化
pip install matplotlib seaborn plotly

# 地理空间分析
pip install geopandas folium

# 机器学习
pip install scikit-learn xgboost

# 统计分析
pip install scipy statsmodels
```

---

## 5. 数据使用指南

### 5.1 快速开始

#### 5.1.1 读取数据

```python
import pandas as pd

# 全量读取（内存需求较大，约需8-12GB内存）
df = pd.read_csv('US_Accidents_March23.csv')

# 分块读取（适合内存有限的情况）
chunk_size = 100000
chunks = []
for chunk in pd.read_csv('US_Accidents_March23.csv', chunksize=chunk_size):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# 仅读取指定列（减少内存占用）
cols = ['ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'State', 'Weather_Condition']
df = pd.read_csv('US_Accidents_March23.csv', usecols=cols)

# 读取前N行进行探索
df_sample = pd.read_csv('US_Accidents_March23.csv', nrows=10000)
```

#### 5.1.2 基本数据探索

```python
# 数据形状
print(f"记录数: {df.shape[0]}, 字段数: {df.shape[1]}")

# 数据类型
print(df.dtypes)

# 基本统计
print(df.describe())

# 缺失值统计
print(df.isnull().sum())

# 各字段唯一值数量
print(df.nunique())
```

### 5.2 常见分析场景

#### 5.2.1 时间序列分析

```python
# 转换时间字段
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek

# 按年统计事故数量
accidents_by_year = df.groupby('Year').size()

# 按小时统计事故数量
accidents_by_hour = df.groupby('Hour').size()
```

#### 5.2.2 地理空间分析

```python
# 按州统计事故数量
accidents_by_state = df.groupby('State').size().sort_values(ascending=False)

# 按城市统计（前10名）
top_cities = df.groupby('City').size().sort_values(ascending=False).head(10)

# 按县统计
accidents_by_county = df.groupby('County').size().sort_values(ascending=False)
```

#### 5.2.3 严重程度分析

```python
# 严重程度分布
severity_dist = df['Severity'].value_counts().sort_index()

# 不同严重程度的平均持续时间
df['Duration'] = (pd.to_datetime(df['End_Time']) - pd.to_datetime(df['Start_Time'])).dt.total_seconds() / 3600
severity_duration = df.groupby('Severity')['Duration'].mean()
```

#### 5.2.4 天气与事故关系

```python
# 天气状况分布
weather_dist = df['Weather_Condition'].value_counts().head(20)

# 不同天气下的严重程度分布
weather_severity = df.groupby('Weather_Condition')['Severity'].value_counts().unstack()
```

#### 5.2.5 道路设施与事故关系

```python
# 计算各设施附近的事故比例
facility_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

facility_stats = {}
for col in facility_cols:
    facility_stats[col] = df[col].mean()
```

### 5.3 内存优化技巧

由于数据集较大（770万条记录），建议采取以下优化措施：

```python
# 1. 使用分类数据类型减少内存
df['Severity'] = df['Severity'].astype('category')
df['State'] = df['State'].astype('category')
df['Weather_Condition'] = df['Weather_Condition'].astype('category')

# 2. 布尔字段转换
bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
             'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 
             'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
for col in bool_cols:
    df[col] = df[col].astype('bool')

# 3. 数值类型优化
df['Start_Lat'] = df['Start_Lat'].astype('float32')
df['Start_Lng'] = df['Start_Lng'].astype('float32')

# 查看内存使用
print(df.memory_usage(deep=True).sum() / 1024 / 1024, 'MB')
```

---

## 6. 典型应用场景

### 6.1 事故预测建模

使用机器学习模型预测事故严重程度或发生概率：
- **特征**：时间、地点、天气、道路设施等
- **目标**：Severity（事故严重程度）
- **模型**：XGBoost、Random Forest、Logistic Regression等

### 6.2 热点区域分析

识别事故高发区域：
- 使用地理空间聚类（如DBSCAN）
- 热力图可视化
- 按行政区域统计

### 6.3 时间模式挖掘

分析事故发生的时间规律：
- 日/周/月/年变化趋势
- 高峰期分析
- 节假日效应

### 6.4 天气影响研究

研究天气条件对交通事故的影响：
- 不同天气类型的事故率
- 极端天气事件与事故的关系
- 能见度、降水等因素的影响

### 6.5 道路安全评估

评估不同道路设施对安全的影响：
- 交通信号灯、停止标志的作用
- 交叉口、环岛的安全性
- 铁路道口风险分析

---

## 7. 项目配置

### 7.1 IDE配置

项目使用 IntelliJ IDEA / PyCharm 进行开发，主要配置文件：

- [US_Accidents_March23.iml](file:///d:/Users/39819/Desktop/US_Accidents_March23/.idea/US_Accidents_March23.iml) - 模块配置
- [modules.xml](file:///d:/Users/39819/Desktop/US_Accidents_March23/.idea/modules.xml) - 模块管理
- [profiles_settings.xml](file:///d:/Users/39819/Desktop/US_Accidents_March23/.idea/inspectionProfiles/profiles_settings.xml) - 代码检查配置

### 7.2 Git忽略配置

[.gitignore](file:///d:/Users/39819/Desktop/US_Accidents_March23/.idea/.gitignore) 文件中忽略以下内容：
- `/shelf/` - 代码搁置
- `/workspace.xml` - 工作区配置
- `/httpRequests/` - HTTP客户端请求
- `/queries/` - 查询文件
- `/dataSources/` - 数据源配置
- `/dataSources.local.xml` - 本地数据源

---

## 8. 注意事项与限制

### 8.1 数据使用限制

1. **数据来源**：本数据集为第三方收集，数据准确性需自行验证
2. **时间范围**：数据覆盖2016年至2023年3月，不包含最新数据
3. **地理范围**：仅限美国本土地区
4. **缺失数据**：部分字段存在缺失，分析时需妥善处理

### 8.2 性能注意事项

1. **内存需求**：全量加载约需8-12GB内存，建议使用分块读取
2. **处理速度**：770万条记录的复杂计算可能需要较长时间
3. **存储优化**：建议转换为Parquet或Feather格式以提高读写速度

### 8.3 隐私与合规

1. 数据中不包含个人身份信息
2. 使用时请遵守相关数据使用协议
3. 研究成果发表时请引用原始数据源

---

## 9. 进阶分析建议

### 9.1 数据预处理建议

1. **时间特征工程**：提取年、月、日、时、星期、节假日等特征
2. **地理特征工程**：城市/州编码、人口密度匹配、经济指标匹配
3. **天气特征编码**：天气状况分类、天气严重程度打分
4. **特征选择**：使用相关性分析、特征重要性等方法筛选特征

### 9.2 建模方向建议

1. **分类问题**：预测事故严重程度
2. **回归问题**：预测事故持续时间
3. **聚类分析**：事故模式发现
4. **异常检测**：异常事故检测
5. **时空预测**：事故热点预测

### 9.3 可视化方向

1. **地理热力图**：事故空间分布
2. **时间序列图**：事故趋势变化
3. **相关性热图**：特征间关系
4. **交互式仪表板**：多维数据探索

---

## 10. 分析模块详解

### 10.1 基础探索性分析 - accident_analysis.py

**功能概述**：全面的探索性数据分析（EDA），涵盖13个分析维度。

**使用方法**：
```bash
# 默认采样5万条
python accident_analysis.py

# 自定义采样量
python accident_analysis.py --sample 100000

# 全量分析（需较大内存）
python accident_analysis.py --sample 0
```

**分析内容**：
| 序号 | 模块 | 说明 |
|------|------|------|
| 1 | 基础信息分析 | 数据形状、内存占用、字段类型 |
| 2 | 缺失值分析 | 各字段缺失数量与缺失率 |
| 3 | 重复值分析 | 完全重复与ID重复检查 |
| 4 | 数值型统计 | count/mean/std/min/分位数 |
| 5 | 分类型统计 | 类别分布与Top类别 |
| 6 | 数据预处理 | 时间转换、特征提取 |
| 7 | 时间序列分析 | 年/月/日/小时/星期分布 |
| 8 | 地理空间分析 | 州/城市/县分布 |
| 9 | 严重程度分析 | 等级分布、持续时间对比 |
| 10 | 天气分析 | 温度/能见度/风速/湿度分布 |
| 11 | 道路设施分析 | 13种设施的影响对比 |
| 12 | 相关性分析 | 数值字段相关系数 |
| 13 | 分析总结 | 主要发现与建议 |

---

### 10.2 预测建模分析 - predictive_modeling.py

**功能概述**：构建机器学习模型预测事故严重程度，对比多种算法性能。

**使用方法**：
```bash
# 全部模型对比
python predictive_modeling.py --sample 50000 --model all

# 仅XGBoost
python predictive_modeling.py --sample 50000 --model xgboost

# 仅随机森林
python predictive_modeling.py --sample 50000 --model rf

# 仅逻辑回归
python predictive_modeling.py --sample 50000 --model lr
```

**特征工程**：
- **时间特征**：小时、星期、月份、是否周末
- **地理特征**：州编码
- **天气特征**：温度、湿度、气压、能见度、风速、降水量
- **道路设施**：13个布尔特征
- **其他**：距离、日出日落

**支持的模型**：
| 模型 | 依赖库 | 说明 |
|------|--------|------|
| XGBoost | xgboost | 梯度提升树，性能优秀 |
| Random Forest | scikit-learn | 随机森林，稳健性好 |
| Logistic Regression | scikit-learn | 基线模型，可解释性强 |

**输出内容**：
- 各模型准确率、精确率、召回率、F1分数
- 混淆矩阵
- 特征重要性排名
- 模型对比汇总表

---

### 10.3 空间聚类分析 - spatial_clustering.py

**功能概述**：识别事故热点区域，使用聚类算法发现地理聚集模式。

**使用方法**：
```bash
# 两种聚类方法都运行
python spatial_clustering.py --sample 50000 --method both

# 仅K-Means
python spatial_clustering.py --sample 50000 --method kmeans --k 15

# 仅DBSCAN
python spatial_clustering.py --sample 50000 --method dbscan --eps 1.0 --min-samples 50

# 肘部法则选K
python spatial_clustering.py --sample 50000 --elbow
```

**聚类算法**：
| 算法 | 参数 | 说明 |
|------|------|------|
| K-Means | k (聚类数) | 快速，适合发现大致分区 |
| DBSCAN | eps (半径km), min_samples | 密度聚类，自动发现任意形状聚类和噪声 |

**分析内容**：
- 聚类数量与大小分布
- Top 10事故热点聚类
- 各聚类平均严重程度
- 严重程度最高的热点
- 按州/城市/县的区域统计

---

### 10.4 时间序列预测 - time_series_forecasting.py

**功能概述**：分析事故时间趋势，构建预测模型预测未来事故数量。

**使用方法**：
```bash
# 日粒度预测
python time_series_forecasting.py --sample 0 --freq D --horizon 30

# 周粒度预测
python time_series_forecasting.py --sample 0 --freq W --horizon 12

# 月粒度预测
python time_series_forecasting.py --sample 0 --freq M --horizon 6

# 跳过回测评估
python time_series_forecasting.py --no-eval
```

**分析内容**：
- 趋势分析（移动平均）
- 星期模式、月度模式
- 高峰时段识别
- 时间序列分解（趋势+季节+残差）
- 移动平均预测
- ARIMA预测（需statsmodels）
- 回测评估（MAPE、MAE）

---

### 10.5 NLP文本分析 - nlp_analysis.py

**功能概述**：对事故描述文本进行深度挖掘，提取关键词和事故类型。

**使用方法**：
```bash
# 基础分析
python nlp_analysis.py --sample 50000

# 生成词云图片
python nlp_analysis.py --sample 50000 --wordcloud

# 显示Top N高频词
python nlp_analysis.py --sample 50000 --top-n 100
```

**分析内容**：
| 模块 | 说明 |
|------|------|
| 文本基本统计 | 长度分布、词数统计 |
| 词频分析 | Top高频词、停用词过滤 |
| 二元词组 | Bigram分析 |
| 事故类型识别 | 18种事故类型自动分类 |
| 道路信息提取 | 道路类型、高速路编号 |
| 方向分析 | 东/南/西/北向事故统计 |
| 严重程度关键词 | 不同等级的关键词差异 |
| 词云生成 | 可视化词云（需wordcloud库） |

**自动识别的事故类型**：
追尾事故、侧面碰撞、正面碰撞、翻车事故、撞车事故、多车事故、单车事故、摩托车事故、卡车事故、巴士事故、行人事故、自行车事故、打滑事故、失控事故、起火事故、坠崖事故、水淹事故、动物事故

---

### 10.6 交互式仪表板 - dashboard.py

**功能概述**：基于Streamlit的交互式可视化仪表板，支持实时筛选和多维度探索。

**启动方式**：
```bash
# 启动仪表板
streamlit run dashboard.py

# 指定端口
streamlit run dashboard.py --server.port 8501
```

**功能模块**（6个Tab页签）：
| Tab | 功能 |
|-----|------|
| ⏰ 时间分析 | 小时/星期/月度分布、工作日vs周末、持续时间分布 |
| 🗺️ 地理分析 | 各州排名、城市排名、地理散点地图、县排名 |
| 🌤️ 天气分析 | 天气分布、温度/能见度/风速/湿度直方图、日出日落饼图 |
| 🚦 道路设施 | 各设施占比、有/无设施严重程度对比 |
| 📈 严重程度 | 严重程度分布、持续时间对比、天气/设施与严重程度关系 |
| 📝 文本分析 | 高频词柱状图、事故类型识别、样本文本 |

**交互功能**：
- 侧边栏调整采样数量
- 按州、严重程度、年份筛选
- 所有图表支持缩放、悬停查看详情

---

### 10.7 综合分析入口 - run_all_analysis.py

**功能概述**：一键运行所有分析模块，自动汇总结果。

**使用方法**：
```bash
# 默认5万采样，全部模块
python run_all_analysis.py

# 自定义采样量
python run_all_analysis.py --sample 100000

# 全量分析
python run_all_analysis.py --full

# 只运行指定模块
python run_all_analysis.py --modules basic,predictive,nlp
```

**可选模块**：
- `basic` - 基础探索性分析
- `predictive` - 预测建模分析
- `spatial` - 空间聚类分析
- `timeseries` - 时间序列预测
- `nlp` - NLP文本分析

**输出内容**：
- 各模块运行状态与耗时
- 总耗时统计
- 输出文件列表
- 下一步建议

---

### 10.8 通用工具模块 - utils.py

**功能概述**：所有分析模块共享的工具函数库。

**主要函数**：
| 函数 | 说明 |
|------|------|
| `print_section(title)` | 打印分节标题 |
| `print_subsection(title)` | 打印子节标题 |
| `load_data(csv_path, sample_size, usecols)` | 加载数据（支持采样） |
| `preprocess_basic(df)` | 基础预处理（时间转换、特征提取） |
| `memory_optimize(df)` | 内存优化 |
| `ensure_output_dir(dir_name)` | 确保输出目录存在 |
| `safe_import(module_name)` | 安全导入（缺失时提示） |

**常量定义**：
- `FACILITY_COLS` - 13个道路设施字段名
- `WEATHER_NUMERIC_COLS` - 天气数值字段名
- `SEVERITY_DESC` - 严重程度等级描述

---

## 11. 附录

### 11.1 常见问题

**Q: 数据文件太大，无法全部加载到内存怎么办？**
A: 可以使用 `chunksize` 参数分块读取，或只读取需要的列，或使用Dask等分布式计算框架。所有分析脚本都支持 `--sample` 参数进行采样分析。

**Q: End_Lat和End_Lng为什么都是空值？**
A: 该版本数据集中可能未填充事故终点坐标。可以使用 `Distance(mi)` 字段估算影响范围。

**Q: 数据来源Source1/2/3分别是什么？**
A: 具体来源需要参考数据集的原始发布说明。不同来源可能有不同的数据采集方式和覆盖范围。

**Q: 某些模块运行时提示缺少依赖库怎么办？**
A: 运行 `pip install -r requirements.txt` 安装所有依赖。各模块已做容错处理，缺少依赖时会优雅降级而不会崩溃。

**Q: Streamlit仪表板如何启动？**
A: 先安装streamlit（`pip install streamlit`），然后运行 `streamlit run dashboard.py`，浏览器会自动打开。

### 11.2 参考资源

- Pandas官方文档：https://pandas.pydata.org/docs/
- Scikit-learn官方文档：https://scikit-learn.org/stable/
- XGBoost文档：https://xgboost.readthedocs.io/
- Streamlit文档：https://docs.streamlit.io/
- Plotly文档：https://plotly.com/python/
- 地理空间分析教程：GeoPandas、Folium官方文档
- 美国交通部公开数据：https://www.transportation.gov/data

---

*文档版本：2.0*  
*生成日期：2026-07-09*  
*数据版本：US_Accidents_March23*  
*分析模块：6个核心模块 + 1个主入口 + 1个工具库*
