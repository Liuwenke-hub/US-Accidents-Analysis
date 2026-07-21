"""
通用工具模块
===========
提供数据加载、预处理、可视化等通用功能
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """打印子节标题"""
    print(f"\n--- {title} ---")


def get_data_path(csv_filename='US_Accidents_March23.csv'):
    """获取数据文件路径"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    csv_path = os.path.join(data_dir, csv_filename)
    if os.path.exists(csv_path):
        return csv_path
    csv_path = os.path.join(root_dir, csv_filename)
    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(csv_filename):
        return os.path.abspath(csv_filename)
    return csv_filename


def load_data(csv_path=None, sample_size=None, usecols=None):
    """
    加载数据

    Args:
        csv_path: CSV文件路径
        sample_size: 采样数量，None表示全量
        usecols: 指定列，None表示全部

    Returns:
        pd.DataFrame
    """
    if csv_path is None:
        csv_path = get_data_path()

    if not os.path.exists(csv_path):
        print(f"错误: 找不到数据文件 {csv_path}")
        print("请将 US_Accidents_March23.csv 放入 data/ 目录或项目根目录")
        sys.exit(1)

    print(f"正在加载数据: {os.path.basename(csv_path)}")
    if sample_size:
        print(f"采样数量: {sample_size:,} 条")

    if usecols:
        if sample_size:
            df = pd.read_csv(csv_path, nrows=sample_size, usecols=usecols)
        else:
            df = pd.read_csv(csv_path, usecols=usecols)
    else:
        if sample_size:
            df = pd.read_csv(csv_path, nrows=sample_size)
        else:
            df = pd.read_csv(csv_path)

    print(f"加载完成: {len(df):,} 条记录, {df.shape[1]} 个字段")
    return df


def preprocess_basic(df):
    """
    基础预处理（时间转换、特征提取）

    Args:
        df: 原始DataFrame

    Returns:
        预处理后的DataFrame
    """
    df = df.copy()

    # 时间转换
    if 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    if 'End_Time' in df.columns:
        df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    if 'Weather_Timestamp' in df.columns:
        df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')

    # 持续时间
    if 'Start_Time' in df.columns and 'End_Time' in df.columns:
        df['Duration_hours'] = (
            (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600
        )

    # 时间特征
    if 'Start_Time' in df.columns:
        df['Year'] = df['Start_Time'].dt.year
        df['Month'] = df['Start_Time'].dt.month
        df['Day'] = df['Start_Time'].dt.day
        df['Hour'] = df['Start_Time'].dt.hour
        df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
        df['DayOfWeek_Name'] = df['Start_Time'].dt.day_name()
        df['Is_Weekend'] = df['DayOfWeek'] >= 5

    # 布尔字段
    bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df


def engineer_features(df):
    """
    高级特征工程（同行常用特征）

    在 preprocess_basic 之后调用。新增：
      - Is_Night        : 是否夜间（Civil_Twilight 判断）
      - Is_Holiday      : 是否美国联邦假日
      - Description_Length : 描述文本长度
      - Street_Length   : 路名字符串长度

    Args:
        df: 已通过 preprocess_basic 的 DataFrame
    Returns:
        增强后的 DataFrame
    """
    df = df.copy()

    # 1. Is_Night: 基于 Civil_Twilight（最可靠）
    if 'Civil_Twilight' in df.columns:
        night_values = {'Night', 'Night'}
        df['Is_Night'] = df['Civil_Twilight'].isin(night_values)
    elif 'Sunrise_Sunset' in df.columns:
        # 回退：用 Sunrise_Sunset
        df['Is_Night'] = (df['Sunrise_Sunset'] == 'Night')
    else:
        # 最终回退：用 Hour 粗略判断（6-18 为白天）
        if 'Hour' in df.columns:
            df['Is_Night'] = (df['Hour'] < 6) | (df['Hour'] >= 18)
        else:
            df['Is_Night'] = False

    # 2. Is_Holiday: 美国联邦主要节假日（固定日期 + 动态）
    _US_HOLIDAYS = {
        "01-01",   # New Year's Day
        "01-15",   # Martin Luther King Jr. Day（近似，实际每年不同）
        "02-19",   # Presidents' Day（近似）
        "05-27",   # Memorial Day（近似）
        "07-04",   # Independence Day
        "09-02",   # Labor Day（近似）
        "10-14",   # Columbus Day（近似）
        "11-11",   # Veterans Day
        "11-28",   # Thanksgiving（近似，实际每年第4个周四）
        "12-25",   # Christmas
    }
    if 'Start_Time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Start_Time']):
        month_day = df['Start_Time'].dt.strftime('%m-%d')
        df['Is_Holiday'] = month_day.isin(_US_HOLIDAYS)
    else:
        df['Is_Holiday'] = False

    # 3. Description_Length
    if 'Description' in df.columns:
        df['Description_Length'] = df['Description'].fillna('').astype(str).str.len()
    else:
        df['Description_Length'] = 0

    # 4. Street_Length
    if 'Street' in df.columns:
        df['Street_Length'] = df['Street'].fillna('').astype(str).str.len()
    else:
        df['Street_Length'] = 0

    print(f"  高级特征已提取: Is_Night, Is_Holiday, Description_Length, Street_Length")
    return df


def memory_optimize(df):
    """内存优化"""
    df = df.copy()

    # 数值类型优化
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # bool 类型优化
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype('uint8')

    return df


def ensure_output_dir(dir_name='output'):
    """确保输出目录存在"""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_import(module_name, install_hint=None):
    """
    安全导入模块，缺失时给出提示

    Args:
        module_name: 模块名
        install_hint: 安装提示

    Returns:
        模块对象或None
    """
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"\n⚠️  缺少依赖库: {module_name}")
        if install_hint:
            print(f"   安装命令: {install_hint}")
        else:
            print(f"   安装命令: pip install {module_name}")
        return None


FACILITY_COLS = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

WEATHER_NUMERIC_COLS = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                        'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                        'Precipitation(in)']

SEVERITY_DESC = {
    1: '轻微事故',
    2: '中等事故',
    3: '严重事故',
    4: '极严重事故'
}

# 英文停用词（单一来源，供 nlp_analysis 与 dashboard 共用，避免重复定义）
ENGLISH_STOPWORDS = set([
    'the', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its',
    'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'our', 'their',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'but', 'if', 'or', 'because',
    'until', 'while', 'about', 'against',
])

