"""
基础探索性分析模块
==================
功能：
    1. 数据加载与基础信息分析
    2. 数据清洗与预处理
    3. 探索性数据分析（EDA）
    4. 时间序列分析
    5. 地理空间分析
    6. 天气与事故关系分析
    7. 道路设施影响分析
    8. 严重程度分析

使用方法：
    python -m analysis.accident_analysis --sample 100000
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from utils.tools import (print_section, get_data_path, safe_import, load_data,
                         preprocess_basic, engineer_features, FACILITY_COLS, SEVERITY_DESC)


def analyze_basic_info(df):
    """基础信息分析"""
    print_section("一、基础信息分析")

    print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    print("\n--- 数据类型 ---")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 个字段")


def analyze_missing_values(df):
    """缺失值分析"""
    print_section("二、缺失值分析")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失率(%)': missing_pct
    }).sort_values('缺失数量', ascending=False)

    print("\n缺失值统计（按缺失数量降序）:")
    print(missing_df[missing_df['缺失数量'] > 0])

    print(f"\n完全无缺失的字段: {(missing == 0).sum()} 个")


def analyze_duplicates(df):
    """重复值分析"""
    print_section("三、重复值分析")

    dup_count = df.duplicated().sum()
    print(f"\n完全重复的记录数: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")

    if 'ID' in df.columns:
        id_dup = df['ID'].duplicated().sum()
        print(f"ID重复的记录数: {id_dup:,}")


def analyze_numerical(df):
    """数值型字段统计"""
    print_section("四、数值型字段统计")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n数值型字段共 {len(numeric_cols)} 个")

    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats['缺失数'] = df[numeric_cols].isnull().sum()
        stats['缺失率(%)'] = (stats['缺失数'] / len(df) * 100).round(2)
        print("\n数值字段统计摘要:")
        print(stats.round(3))


def analyze_categorical(df):
    """分类型字段统计"""
    print_section("五、分类型字段统计")

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    print(f"\n分类型字段共 {len(cat_cols)} 个")

    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique <= 20:
            print(f"\n--- {col} (共 {nunique} 个类别) ---")
            value_counts = df[col].value_counts()
            pct = (value_counts / len(df) * 100).round(2)
            result = pd.DataFrame({'数量': value_counts, '占比(%)': pct})
            print(result)
        else:
            top10 = df[col].value_counts().head(10)
            print(f"\n--- {col} (共 {nunique} 个类别，显示Top 10) ---")
            pct = (top10 / len(df) * 100).round(2)
            result = pd.DataFrame({'数量': top10, '占比(%)': pct})
            print(result)


def preprocess_data(df):
    """数据预处理（复用 utils.tools.preprocess_basic，避免重复实现）"""
    print_section("六、数据预处理")
    return preprocess_basic(df)


def analyze_time_series(df):
    """时间序列分析"""
    print_section("七、时间序列分析")

    if 'Start_Time' not in df.columns or 'Year' not in df.columns:
        print("  缺少时间字段，跳过时间序列分析")
        return

    # 按年统计
    print("\n--- 按年统计事故数量 ---")
    yearly = df.groupby('Year').size().sort_index()
    yearly_pct = (yearly / yearly.sum() * 100).round(2)
    print(pd.DataFrame({'事故数': yearly, '占比(%)': yearly_pct}))

    # 按月统计
    print("\n--- 按月统计事故数量 ---")
    monthly = df.groupby('Month').size()
    monthly_pct = (monthly / monthly.sum() * 100).round(2)
    print(pd.DataFrame({'事故数': monthly, '占比(%)': monthly_pct}))

    # 按小时统计
    print("\n--- 按小时统计事故数量 ---")
    hourly = df.groupby('Hour').size()
    hourly_pct = (hourly / hourly.sum() * 100).round(2)
    print(pd.DataFrame({'事故数': hourly, '占比(%)': hourly_pct}))

    # 按星期统计
    print("\n--- 按星期统计事故数量 ---")
    weekday = df.groupby('DayOfWeek_Name').size()
    weekday_pct = (weekday / weekday.sum() * 100).round(2)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = weekday.reindex([d for d in weekday_order if d in weekday.index])
    print(pd.DataFrame({'事故数': weekday, '占比(%)': weekday_pct}))

    # 工作日 vs 周末
    print("\n--- 工作日 vs 周末 ---")
    weekend = df.groupby('Is_Weekend').size()
    weekend_pct = (weekend / weekend.sum() * 100).round(2)
    print(pd.DataFrame({'事故数': weekend, '占比(%)': weekend_pct}))

    # 时间范围
    if 'Start_Time' in df.columns:
        print(f"\n数据时间范围: {df['Start_Time'].min()} ~ {df['Start_Time'].max()}")


def analyze_geography(df):
    """地理空间分析"""
    print_section("八、地理空间分析")

    # 按州统计
    if 'State' in df.columns:
        print("\n--- 按州统计（Top 20） ---")
        by_state = df.groupby('State').size().sort_values(ascending=False).head(20)
        by_state_pct = (by_state / len(df) * 100).round(2)
        print(pd.DataFrame({'事故数': by_state, '占比(%)': by_state_pct}))

        print(f"\n覆盖州数: {df['State'].nunique()} 个")

    # 按城市统计
    if 'City' in df.columns:
        print("\n--- 按城市统计（Top 20） ---")
        by_city = df.groupby('City').size().sort_values(ascending=False).head(20)
        by_city_pct = (by_city / len(df) * 100).round(2)
        print(pd.DataFrame({'事故数': by_city, '占比(%)': by_city_pct}))

    # 按县统计
    if 'County' in df.columns:
        print("\n--- 按县统计（Top 20） ---")
        by_county = df.groupby('County').size().sort_values(ascending=False).head(20)
        by_county_pct = (by_county / len(df) * 100).round(2)
        print(pd.DataFrame({'事故数': by_county, '占比(%)': by_county_pct}))

    # 地理坐标范围
    if 'Start_Lat' in df.columns and 'Start_Lng' in df.columns:
        print(f"\n纬度范围: {df['Start_Lat'].min():.4f} ~ {df['Start_Lat'].max():.4f}")
        print(f"经度范围: {df['Start_Lng'].min():.4f} ~ {df['Start_Lng'].max():.4f}")


def analyze_severity(df):
    """严重程度分析"""
    print_section("九、严重程度分析")

    if 'Severity' not in df.columns:
        print("  缺少 Severity 字段，跳过严重程度分析")
        return

    # 严重程度分布
    print("\n--- 严重程度分布 ---")
    severity = df['Severity'].value_counts().sort_index()
    severity_pct = (severity / len(df) * 100).round(2)
    print(pd.DataFrame({'事故数': severity, '占比(%)': severity_pct}))

    # 严重程度描述（复用 utils.tools.SEVERITY_DESC）
    for level, desc in SEVERITY_DESC.items():
        if level in severity.index:
            print(f"  等级 {level} ({desc}): {severity[level]:,} 起 ({severity_pct[level]}%)")

    # 不同严重程度的持续时间
    if 'Duration_hours' in df.columns:
        print("\n--- 不同严重程度的平均持续时间 ---")
        duration_by_severity = df.groupby('Severity')['Duration_hours'].agg(['mean', 'median', 'count'])
        print(duration_by_severity.round(2))

    # 不同州的平均严重程度
    if 'State' in df.columns:
        print("\n--- 平均严重程度最高的州（Top 10，至少100起事故） ---")
        state_severity = df.groupby('State')['Severity'].agg(['mean', 'count'])
        state_severity = state_severity[state_severity['count'] >= 100]
        state_severity = state_severity.sort_values('mean', ascending=False).head(10)
        print(state_severity.round(3))

    # 不同天气下的严重程度
    if 'Weather_Condition' in df.columns:
        print("\n--- 平均严重程度最高的天气（Top 10，至少50起事故） ---")
        weather_severity = df.groupby('Weather_Condition')['Severity'].agg(['mean', 'count'])
        weather_severity = weather_severity[weather_severity['count'] >= 50]
        weather_severity = weather_severity.sort_values('mean', ascending=False).head(10)
        print(weather_severity.round(3))


def analyze_weather(df):
    """天气与事故关系分析"""
    print_section("十、天气与事故关系分析")

    # 天气状况分布
    if 'Weather_Condition' in df.columns:
        print("\n--- 天气状况分布（Top 20） ---")
        weather = df['Weather_Condition'].value_counts().head(20)
        weather_pct = (weather / len(df) * 100).round(2)
        print(pd.DataFrame({'事故数': weather, '占比(%)': weather_pct}))

    # 温度分布
    if 'Temperature(F)' in df.columns:
        print("\n--- 温度分布（华氏度） ---")
        temp = df['Temperature(F)'].dropna()
        temp_bins = [0, 32, 50, 70, 90, 120]
        temp_labels = ['<32 (冰点以下)', '32-50 (寒冷)', '50-70 (凉爽)', '70-90 (温暖)', '>90 (炎热)']
        temp_cut = pd.cut(temp, bins=temp_bins, labels=temp_labels)
        temp_dist = temp_cut.value_counts().sort_index()
        temp_pct = (temp_dist / len(temp) * 100).round(2)
        print(pd.DataFrame({'事故数': temp_dist, '占比(%)': temp_pct}))

    # 能见度分布
    if 'Visibility(mi)' in df.columns:
        print("\n--- 能见度分布（英里） ---")
        vis = df['Visibility(mi)'].dropna()
        vis_bins = [0, 0.5, 2, 5, 10, 100]
        vis_labels = ['<0.5 (极差)', '0.5-2 (差)', '2-5 (一般)', '5-10 (良好)', '>10 (优秀)']
        vis_cut = pd.cut(vis, bins=vis_bins, labels=vis_labels)
        vis_dist = vis_cut.value_counts().sort_index()
        vis_pct = (vis_dist / len(vis) * 100).round(2)
        print(pd.DataFrame({'事故数': vis_dist, '占比(%)': vis_pct}))

    # 风速分布
    if 'Wind_Speed(mph)' in df.columns:
        print("\n--- 风速分布（mph） ---")
        wind = df['Wind_Speed(mph)'].dropna()
        wind_bins = [0, 5, 15, 25, 40, 100]
        wind_labels = ['<5 (无风)', '5-15 (微风)', '15-25 (轻风)', '25-40 (强风)', '>40 (大风)']
        wind_cut = pd.cut(wind, bins=wind_bins, labels=wind_labels)
        wind_dist = wind_cut.value_counts().sort_index()
        wind_pct = (wind_dist / len(wind) * 100).round(2)
        print(pd.DataFrame({'事故数': wind_dist, '占比(%)': wind_pct}))

    # 湿度分布
    if 'Humidity(%)' in df.columns:
        print("\n--- 湿度分布（%） ---")
        hum = df['Humidity(%)'].dropna()
        hum_bins = [0, 30, 60, 80, 100]
        hum_labels = ['<30 (干燥)', '30-60 (舒适)', '60-80 (潮湿)', '>80 (很潮湿)']
        hum_cut = pd.cut(hum, bins=hum_bins, labels=hum_labels)
        hum_dist = hum_cut.value_counts().sort_index()
        hum_pct = (hum_dist / len(hum) * 100).round(2)
        print(pd.DataFrame({'事故数': hum_dist, '占比(%)': hum_pct}))

    # 日出日落与事故
    if 'Sunrise_Sunset' in df.columns:
        print("\n--- 日出日落与事故 ---")
        day_night = df['Sunrise_Sunset'].value_counts()
        day_night_pct = (day_night / len(df) * 100).round(2)
        print(pd.DataFrame({'事故数': day_night, '占比(%)': day_night_pct}))


def analyze_road_facilities(df):
    """道路设施影响分析"""
    print_section("十一、道路设施影响分析")

    available_facilities = [c for c in FACILITY_COLS if c in df.columns]
    if not available_facilities:
        print("  缺少道路设施字段，跳过分析")
        return

    print(f"\n共 {len(available_facilities)} 个道路设施字段")

    # 各设施附近事故比例
    print("\n--- 各设施附近的事故比例 ---")
    facility_stats = []
    for col in available_facilities:
        true_count = df[col].sum()
        true_pct = true_count / len(df) * 100
        facility_stats.append({
            '设施': col,
            '事故数': true_count,
            '占比(%)': round(true_pct, 2)
        })

    facility_df = pd.DataFrame(facility_stats).sort_values('占比(%)', ascending=False)
    print(facility_df.to_string(index=False))

    # 设施与严重程度的关系
    if 'Severity' in df.columns:
        print("\n--- 有/无设施的平均严重程度对比 ---")
        severity_compare = []
        for col in available_facilities:
            has_facility = df[df[col]]['Severity'].mean()
            no_facility = df[~df[col]]['Severity'].mean()
            severity_compare.append({
                '设施': col,
                '有设施_均值': round(has_facility, 3),
                '无设施_均值': round(no_facility, 3),
                '差异': round(has_facility - no_facility, 3)
            })

        compare_df = pd.DataFrame(severity_compare).sort_values('差异', ascending=False)
        print(compare_df.to_string(index=False))


def analyze_correlation(df):
    """相关性分析"""
    print_section("十二、数值型字段相关性分析")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print("  数值型字段不足，跳过相关性分析")
        return

    print(f"\n参与相关性分析的字段: {len(numeric_cols)} 个")

    # 计算相关系数矩阵
    corr = df[numeric_cols].corr()

    # 显示与 Severity 的相关性（如果有）
    if 'Severity' in corr.columns:
        print("\n--- 与 Severity（严重程度）的相关性 ---")
        severity_corr = corr['Severity'].drop('Severity').sort_values(ascending=False)
        print(severity_corr.round(4))

    # 显示高相关性对（|r| > 0.5）
    print("\n--- 高相关性字段对（|r| > 0.5） ---")
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr.append({
                    '字段1': corr.columns[i],
                    '字段2': corr.columns[j],
                    '相关系数': round(r, 4)
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('相关系数', ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("  没有发现 |r| > 0.5 的高相关性字段对")


def analyze_associations(df):
    """
    分类变量关联分析（卡方检验 + Cramér's V 效应量）

    重要说明：在大样本（数万条）下，卡方检验几乎必然 p < 0.05，
    因此「显著」本身没有信息量，真正要看的是效应量（Cramér's V）。
    本函数同时报告两者，并明确给出效应量解读。
    """
    print_section("十三、分类变量关联分析（卡方 + 效应量）")

    scipy_stats = safe_import('scipy.stats', 'pip install scipy')
    if scipy_stats is None:
        return

    from scipy.stats import chi2_contingency

    def cramers_v(confusion):
        chi2 = chi2_contingency(confusion)[0]
        n = confusion.values.sum()
        r, k = confusion.shape
        return float(np.sqrt((chi2 / n) / (min(r, k) - 1)))

    candidates = []
    if 'Severity' in df.columns and 'Is_Weekend' in df.columns:
        candidates.append(('严重程度 × 是否周末',
                            pd.crosstab(df['Severity'], df['Is_Weekend'])))
    if 'Severity' in df.columns and 'State' in df.columns:
        top_states = df['State'].value_counts().head(5).index
        state_grouped = df['State'].where(df['State'].isin(top_states), other='其他州')
        candidates.append(('严重程度 × 主要州（前5 + 其他）',
                           pd.crosstab(df['Severity'], state_grouped)))

    if not candidates:
        print("  缺少可用的分类字段，跳过关联分析")
        return

    for name, ct in candidates:
        chi2, p, dof, _ = chi2_contingency(ct)
        v = cramers_v(ct)
        n = int(ct.values.sum())
        strength = ('弱（几乎无实际意义）' if v < 0.1
                    else '中等' if v < 0.3
                    else '较强' if v < 0.5
                    else '很强')
        print(f"\n--- {name} ---")
        print(f"  样本量 N = {n:,}")
        print(f"  χ² = {chi2:.2f}, 自由度 = {dof}, p = {p:.3e}")
        print(f"  Cramér's V（效应量） = {v:.4f}  →  {strength}")
        print(f"  提示: 样本量越大，p 值越小；本结论以效应量为准。")


def generate_summary(df, sample_size):
    """生成分析总结"""
    print_section("十三、分析总结")

    print(f"""
数据集概览:
  - 数据文件: US_Accidents_March23.csv
  - 分析样本: {sample_size if sample_size else '全量'} 条
  - 数据时间范围: {df['Start_Time'].min().date() if 'Start_Time' in df.columns else 'N/A'} ~ 
                   {df['Start_Time'].max().date() if 'Start_Time' in df.columns else 'N/A'}
  - 覆盖州数: {df['State'].nunique() if 'State' in df.columns else 'N/A'}
  - 覆盖城市数: {df['City'].nunique() if 'City' in df.columns else 'N/A'}

主要发现:
  1. 事故严重程度以 2 级（中等）为主
  2. 事故时间分布有明显的高峰时段
  3. 不同州的事故数量差异显著
  4. 天气条件与事故发生有一定关联
  5. 道路设施附近的事故特征不同

建议进一步分析方向:
  1. 构建事故严重程度预测模型
  2. 进行地理空间聚类，识别事故热点
  3. 分析节假日、特殊天气事件的影响
  4. 结合人口、经济数据进行深度分析
  5. 构建交互式可视化仪表板
""")


def main():
    parser = argparse.ArgumentParser(description='US Accidents 基础探索性分析工具')
    parser.add_argument('--sample', type=int, default=50000,
                        help='采样分析的记录数（默认50000，0表示全量）')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV文件路径（默认从data/目录读取）')
    args = parser.parse_args()

    # 确定CSV路径
    csv_path = args.csv if args.csv else get_data_path()

    if not os.path.exists(csv_path):
        print(f"错误: 找不到数据文件 {csv_path}")
        print("请将 US_Accidents_March23.csv 放入 data/ 目录或项目根目录")
        sys.exit(1)

    # 确定样本大小
    sample_size = args.sample if args.sample > 0 else None

    print("=" * 80)
    print("  US Accidents March 2023 基础探索性分析")
    print("=" * 80)
    print(f"\n数据文件: {csv_path}")
    print(f"分析模式: {'全量分析' if sample_size is None else f'采样分析 ({sample_size:,} 条)'}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据
    df = load_data(csv_path, sample_size=sample_size)

    # 内存保护：EDA 全量上限（避免 770 万 × 47 列 OOM）
    EDA_MAX_ROWS = 2_000_000
    if sample_size is None and len(df) > EDA_MAX_ROWS:
        print(f"\n  [内存保护] 全量 {len(df):,} 条超过 EDA 上限 {EDA_MAX_ROWS:,}，随机降采样")
        df = df.sample(n=EDA_MAX_ROWS, random_state=42).reset_index(drop=True)

    # 2. 基础信息分析
    analyze_basic_info(df)

    # 3. 缺失值分析
    analyze_missing_values(df)

    # 4. 重复值分析
    analyze_duplicates(df)

    # 5. 数值型字段统计
    analyze_numerical(df)

    # 6. 分类型字段统计
    analyze_categorical(df)

    # 7. 数据预处理
    df_clean = preprocess_data(df)

    # 7.5 高级特征工程（Is_Night / Is_Holiday / 文本长度等）
    df_clean = engineer_features(df_clean)

    # 8. 时间序列分析
    analyze_time_series(df_clean)

    # 9. 地理空间分析
    analyze_geography(df_clean)

    # 10. 严重程度分析
    analyze_severity(df_clean)

    # 11. 天气分析
    analyze_weather(df_clean)

    # 12. 道路设施分析
    analyze_road_facilities(df_clean)

    # 13. 相关性分析
    analyze_correlation(df_clean)

    # 13b. 分类变量关联分析
    analyze_associations(df_clean)

    # 14. 总结
    generate_summary(df_clean, sample_size)

    print("\n" + "=" * 80)
    print(f"  分析完成！结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
