"""
交互式仪表板模块
===============
基于Streamlit的交互式数据探索工具

启动方式：
    streamlit run visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.tools import load_data, preprocess_basic, FACILITY_COLS, ENGLISH_STOPWORDS


def load_data_for_dashboard(sample_size):
    """加载数据用于仪表板"""
    with st.spinner('加载数据中...'):
        usecols = ['Start_Time', 'End_Time', 'Severity', 'State', 'City', 'County',
                   'Start_Lat', 'Start_Lng', 'Distance(mi)', 'Temperature(F)',
                   'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                   'Precipitation(in)', 'Weather_Condition', 'Sunrise_Sunset',
                   'Description'] + FACILITY_COLS

        df = load_data(sample_size=sample_size, usecols=usecols)
        df = preprocess_basic(df)

        st.success(f'数据加载完成！共 {len(df):,} 条记录')

    return df


def dashboard_overview(df):
    """数据概览"""
    st.subheader('📊 数据概览')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('总事故数', f'{len(df):,}')
    with col2:
        st.metric('覆盖州数', df['State'].nunique())
    with col3:
        st.metric('覆盖城市数', df['City'].nunique())
    with col4:
        st.metric('平均严重程度', f'{df["Severity"].mean():.2f}')

    st.subheader('📅 数据时间范围')
    st.write(f"从 {df['Start_Time'].min().date()} 到 {df['Start_Time'].max().date()}")

    st.subheader('📈 严重程度分布')
    severity_dist = df['Severity'].value_counts().sort_index()
    fig = px.bar(severity_dist, x=severity_dist.index, y=severity_dist.values,
                 labels={'x': '严重程度', 'y': '事故数'},
                 title='严重程度分布',
                 color=severity_dist.index,
                 color_continuous_scale='viridis')
    st.plotly_chart(fig)


def dashboard_time_analysis(df):
    """时间分析"""
    st.subheader('⏰ 时间分析')

    # 按小时分布
    st.markdown('### 按小时分布')
    hourly = df.groupby('Hour').size()
    fig = px.line(hourly, x=hourly.index, y=hourly.values,
                  labels={'x': '小时', 'y': '事故数'},
                  title='事故按小时分布')
    st.plotly_chart(fig)

    # 按星期分布
    st.markdown('### 按星期分布')
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = df.groupby('DayOfWeek_Name').size().reindex(weekday_order)
    fig = px.bar(weekday, x=weekday.index, y=weekday.values,
                 labels={'x': '星期', 'y': '事故数'},
                 title='事故按星期分布')
    st.plotly_chart(fig)

    # 按月份分布
    st.markdown('### 按月份分布')
    monthly = df.groupby('Month').size()
    fig = px.bar(monthly, x=monthly.index, y=monthly.values,
                 labels={'x': '月份', 'y': '事故数'},
                 title='事故按月份分布')
    st.plotly_chart(fig)

    # 工作日 vs 周末
    st.markdown('### 工作日 vs 周末')
    weekend = df.groupby('Is_Weekend').size()
    fig = px.pie(weekend, values=weekend.values, names=['工作日', '周末'],
                 title='工作日与周末事故比例')
    st.plotly_chart(fig)

    # 持续时间分布
    st.markdown('### 事故持续时间分布')
    duration = df['Duration_hours'].dropna()
    duration = duration[duration < 24]
    fig = px.histogram(duration, nbins=50,
                       labels={'value': '持续时间(小时)', 'count': '事故数'},
                       title='事故持续时间分布（小于24小时）')
    st.plotly_chart(fig)


def dashboard_geo_analysis(df):
    """地理分析"""
    st.subheader('🗺️ 地理分析')

    # 按州统计
    st.markdown('### 各州事故排名（Top 10）')
    by_state = df.groupby('State').size().sort_values(ascending=False).head(10)
    fig = px.bar(by_state, x=by_state.index, y=by_state.values,
                 labels={'x': '州', 'y': '事故数'},
                 title='各州事故排名')
    st.plotly_chart(fig)

    # 按城市统计
    st.markdown('### 各城市事故排名（Top 10）')
    by_city = df.groupby('City').size().sort_values(ascending=False).head(10)
    fig = px.bar(by_city, x=by_city.index, y=by_city.values,
                 labels={'x': '城市', 'y': '事故数'},
                 title='各城市事故排名')
    st.plotly_chart(fig)

    # 地理散点图
    st.markdown('### 地理分布散点图')
    df_map = df.dropna(subset=['Start_Lat', 'Start_Lng']).sample(min(10000, len(df)))
    fig = px.scatter_mapbox(df_map, lat='Start_Lat', lon='Start_Lng',
                            color='Severity',
                            size='Distance(mi)',
                            zoom=3,
                            mapbox_style='carto-positron',
                            title='事故地理分布')
    st.plotly_chart(fig)


def dashboard_weather_analysis(df):
    """天气分析"""
    st.subheader('🌤️ 天气分析')

    # 天气状况分布
    st.markdown('### 天气状况分布（Top 15）')
    weather = df['Weather_Condition'].value_counts().head(15)
    fig = px.bar(weather, x=weather.index, y=weather.values,
                 labels={'x': '天气状况', 'y': '事故数'},
                 title='天气状况分布')
    st.plotly_chart(fig)

    # 温度分布
    st.markdown('### 温度分布')
    temp = df['Temperature(F)'].dropna()
    fig = px.histogram(temp, nbins=30,
                       labels={'value': '温度(F)', 'count': '事故数'},
                       title='温度分布')
    st.plotly_chart(fig)

    # 能见度分布
    st.markdown('### 能见度分布')
    vis = df['Visibility(mi)'].dropna()
    fig = px.histogram(vis, nbins=20,
                       labels={'value': '能见度(英里)', 'count': '事故数'},
                       title='能见度分布')
    st.plotly_chart(fig)

    # 风速分布
    st.markdown('### 风速分布')
    wind = df['Wind_Speed(mph)'].dropna()
    wind = wind[wind < 50]
    fig = px.histogram(wind, nbins=20,
                       labels={'value': '风速(mph)', 'count': '事故数'},
                       title='风速分布')
    st.plotly_chart(fig)

    # 湿度分布
    st.markdown('### 湿度分布')
    hum = df['Humidity(%)'].dropna()
    fig = px.histogram(hum, nbins=20,
                       labels={'value': '湿度(%)', 'count': '事故数'},
                       title='湿度分布')
    st.plotly_chart(fig)

    # 日出日落
    st.markdown('### 日出日落与事故')
    day_night = df['Sunrise_Sunset'].value_counts()
    fig = px.pie(day_night, values=day_night.values, names=day_night.index,
                 title='日出日落事故比例')
    st.plotly_chart(fig)


def dashboard_facilities_analysis(df):
    """道路设施分析"""
    st.subheader('🚦 道路设施分析')

    # 各设施事故比例
    st.markdown('### 各设施附近事故比例')
    facility_stats = []
    for col in FACILITY_COLS:
        if col in df.columns:
            true_count = df[col].sum()
            true_pct = true_count / len(df) * 100
            facility_stats.append({'设施': col, '比例(%)': true_pct})

    facility_df = pd.DataFrame(facility_stats).sort_values('比例(%)', ascending=False)
    fig = px.bar(facility_df, x='设施', y='比例(%)',
                 title='各设施附近事故比例')
    st.plotly_chart(fig)

    # 设施与严重程度关系
    if 'Severity' in df.columns:
        st.markdown('### 有/无设施的平均严重程度对比')
        severity_compare = []
        for col in FACILITY_COLS:
            if col in df.columns:
                has_facility = df[df[col]]['Severity'].mean()
                no_facility = df[~df[col]]['Severity'].mean()
                severity_compare.append({
                    '设施': col,
                    '有设施': has_facility,
                    '无设施': no_facility
                })

        compare_df = pd.DataFrame(severity_compare)
        fig = px.bar(compare_df, x='设施', y=['有设施', '无设施'],
                     barmode='group',
                     title='有/无设施的平均严重程度对比')
        st.plotly_chart(fig)


def dashboard_severity_analysis(df):
    """严重程度分析"""
    st.subheader('📈 严重程度分析')

    # 严重程度分布
    st.markdown('### 严重程度分布')
    severity = df['Severity'].value_counts().sort_index()
    fig = px.pie(severity, values=severity.values, names=severity.index,
                 title='严重程度分布')
    st.plotly_chart(fig)

    # 不同严重程度的持续时间
    if 'Duration_hours' in df.columns:
        st.markdown('### 不同严重程度的持续时间')
        duration_df = df.dropna(subset=['Duration_hours'])
        duration_df = duration_df[duration_df['Duration_hours'] < 24]
        fig = px.box(duration_df, x='Severity', y='Duration_hours',
                     labels={'x': '严重程度', 'y': '持续时间(小时)'},
                     title='不同严重程度的持续时间分布')
        st.plotly_chart(fig)

    # 不同天气下的严重程度
    if 'Weather_Condition' in df.columns:
        st.markdown('### 不同天气下的平均严重程度（Top 10）')
        weather_severity = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(weather_severity, x=weather_severity.index, y=weather_severity.values,
                     labels={'x': '天气', 'y': '平均严重程度'},
                     title='不同天气下的平均严重程度')
        st.plotly_chart(fig)


def dashboard_text_analysis(df):
    """文本分析"""
    st.subheader('📝 文本分析')

    if 'Description' not in df.columns:
        st.warning('缺少 Description 字段')
        return

    desc = df['Description'].dropna()
    if len(desc) == 0:
        st.warning('没有可用的描述文本')
        return

    # 高频词
    st.markdown('### 高频词（Top 20）')
    import re
    all_words = []
    stop_words = ENGLISH_STOPWORDS

    for text in desc.sample(min(10000, len(desc))):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)

    word_counts = pd.Series(all_words).value_counts().head(20)
    fig = px.bar(word_counts, x=word_counts.index, y=word_counts.values,
                 labels={'x': '词', 'y': '频率'},
                 title='高频词分布')
    st.plotly_chart(fig)

    # 事故类型识别
    st.markdown('### 事故类型识别')
    accident_patterns = {
        '追尾事故': r'rear[ -]end|rearend|tailgate',
        '侧面碰撞': r'side[ -]swipe|side[ -]impact|t[ -]bone',
        '正面碰撞': r'head[ -]on|headon',
        '翻车事故': r'roll[ -]over|overturned',
        '撞车事故': r'collision|crashed into',
        '多车事故': r'multi[ -]vehicle|multiple vehicle',
        '单车事故': r'single[ -]vehicle',
        '摩托车事故': r'motorcycle',
        '卡车事故': r'truck|semi',
        '行人事故': r'pedestrian',
        '自行车事故': r'bicycle',
        '打滑事故': r'skidded|slipped',
        '失控事故': r'lost control|out of control',
        '起火事故': r'fire',
        '动物事故': r'animal|deer'
    }

    type_counts = {}
    for accident_type, pattern in accident_patterns.items():
        count = desc.str.contains(pattern, case=False, na=False).sum()
        type_counts[accident_type] = count

    type_df = pd.DataFrame({'事故类型': list(type_counts.keys()), '数量': list(type_counts.values())})
    type_df = type_df.sort_values('数量', ascending=False)
    fig = px.bar(type_df, x='事故类型', y='数量',
                 title='事故类型识别结果')
    st.plotly_chart(fig)

    # 样本文本
    st.markdown('### 样本文本')
    sample_descs = desc.sample(min(5, len(desc))).tolist()
    for i, text in enumerate(sample_descs, 1):
        st.write(f'{i}. {text}')


def main():
    """主函数"""
    st.title('🚗 US Accidents 交互式数据分析仪表板')

    st.sidebar.header('数据筛选')

    sample_size = st.sidebar.slider('采样数量', 1000, 100000, 50000, step=10000)

    df = load_data_for_dashboard(sample_size)

    state_filter = st.sidebar.multiselect('选择州', sorted(df['State'].unique()) if 'State' in df.columns else [])

    severity_filter = st.sidebar.multiselect('选择严重程度', sorted(df['Severity'].unique()) if 'Severity' in df.columns else [])

    year_filter = st.sidebar.multiselect('选择年份', sorted(df['Year'].unique()) if 'Year' in df.columns else [])

    if state_filter:
        df = df[df['State'].isin(state_filter)]
    if severity_filter:
        df = df[df['Severity'].isin(severity_filter)]
    if year_filter:
        df = df[df['Year'].isin(year_filter)]

    tabs = st.tabs(['📊 概览', '⏰ 时间分析', '🗺️ 地理分析', '🌤️ 天气分析', '🚦 道路设施', '📈 严重程度', '📝 文本分析'])

    with tabs[0]:
        dashboard_overview(df)
    with tabs[1]:
        dashboard_time_analysis(df)
    with tabs[2]:
        dashboard_geo_analysis(df)
    with tabs[3]:
        dashboard_weather_analysis(df)
    with tabs[4]:
        dashboard_facilities_analysis(df)
    with tabs[5]:
        dashboard_severity_analysis(df)
    with tabs[6]:
        dashboard_text_analysis(df)


if __name__ == '__main__':
    main()
