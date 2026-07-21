"""
时间序列预测模块
===============
功能：
    1. 时间序列探索与可视化
    2. 趋势分析与季节性分解
    3. 移动平均预测
    4. ARIMA预测
    5. 回测评估

使用方法：
    python -m analysis.time_series_forecasting --sample 0 --freq D --horizon 30
"""

import argparse
import os
import numpy as np
import pandas as pd

from utils.tools import (print_section, print_subsection, load_data, preprocess_basic,
                         ensure_output_dir, safe_import)


def prepare_time_series(df, freq='D'):
    """准备时间序列数据"""
    print_section("准备时间序列数据")

    if 'Start_Time' not in df.columns:
        print("错误: 缺少 Start_Time 字段")
        return None

    df_time = df.dropna(subset=['Start_Time']).copy()
    df_time = df_time.sort_values('Start_Time')

    print(f"\n原始数据: {len(df):,} 条")
    print(f"有效时间数据: {len(df_time):,} 条")

    freq_map = {'D': 'M', 'W': 'W', 'M': 'M'}
    actual_freq = freq_map.get(freq, freq)

    ts = df_time.groupby(df_time['Start_Time'].dt.to_period(actual_freq))['ID'].count()
    ts.index = ts.index.to_timestamp()

    print(f"\n时间序列频率: {freq}")
    print(f"时间序列长度: {len(ts)}")
    print(f"时间范围: {ts.index.min()} ~ {ts.index.max()}")
    print(f"平均每日/周/月事故数: {ts.mean():.1f}")
    print(f"最大每日/周/月事故数: {ts.max():.0f}")
    print(f"最小每日/周/月事故数: {ts.min():.0f}")

    return ts


def explore_time_patterns(ts):
    """探索时间模式"""
    print_section("时间模式探索")

    if len(ts) < 7:
        print("  数据量不足，跳过时间模式分析")
        return

    print("\n--- 星期模式分析 ---")
    weekday_counts = ts.groupby(ts.index.dayofweek).mean()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_df = pd.DataFrame({'星期': weekday_names, '平均事故数': weekday_counts.values})
    print(weekday_df.round(1))

    print("\n--- 月度模式分析 ---")
    if len(ts) >= 12:
        month_counts = ts.groupby(ts.index.month).mean()
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        month_df = pd.DataFrame({'月份': month_names, '平均事故数': month_counts.values})
        print(month_df.round(1))

    print("\n--- 趋势分析（移动平均） ---")
    if len(ts) >= 30:
        ma_7 = ts.rolling(window=7).mean()
        ma_30 = ts.rolling(window=30).mean()
        print(f"7日移动平均（近期）:")
        print(ma_7.tail(7).round(1))
        print(f"\n30日移动平均（近期）:")
        print(ma_30.tail(5).round(1))

    print("\n--- 高峰时段识别 ---")
    print(f"平均事故数: {ts.mean():.1f}")
    high_days = ts[ts > ts.mean() * 1.5]
    if len(high_days) > 0:
        print(f"高于平均1.5倍的日期数: {len(high_days)} ({len(high_days)/len(ts)*100:.2f}%)")
        print("Top 10高峰日期:")
        print(high_days.sort_values(ascending=False).head(10).round(1))


def decompose_time_series(ts):
    """时间序列分解"""
    print_section("时间序列分解")

    statsmodels = safe_import('statsmodels', 'pip install statsmodels')
    if statsmodels is None:
        return

    from statsmodels.tsa.seasonal import seasonal_decompose

    if len(ts) < 365:
        print("  数据量不足一年，跳过分解分析")
        return

    print("\n正在进行时间序列分解...")
    try:
        result = seasonal_decompose(ts, model='additive', period=365)
        print(f"\n趋势分量（近期）:")
        print(result.trend.tail(5).round(1))
        print(f"\n季节分量（典型月份）:")
        seasonal_df = result.seasonal.groupby(result.seasonal.index.month).mean()
        print(seasonal_df.round(2))
        print(f"\n残差分量统计:")
        print(result.resid.describe().round(2))
    except Exception as e:
        print(f"  分解失败: {e}")


def moving_average_forecast(ts, horizon=30):
    """移动平均预测"""
    print_section(f"移动平均预测（预测{horizon}步）")

    window_size = min(7, len(ts) // 2)
    ma = ts.rolling(window=window_size).mean()

    last_ma = ma.iloc[-1]
    forecast = pd.Series([last_ma] * horizon, index=pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon))

    print(f"\n使用{window_size}期移动平均")
    print(f"最后一期移动平均值: {last_ma:.1f}")
    print(f"\n预测结果（前10期）:")
    print(forecast.head(10).round(1))

    return forecast


def arima_forecast(ts, horizon=30):
    """ARIMA预测"""
    print_section(f"ARIMA预测（预测{horizon}步）")

    statsmodels = safe_import('statsmodels', 'pip install statsmodels')
    if statsmodels is None:
        return None

    from statsmodels.tsa.arima.model import ARIMA

    print("\n正在训练ARIMA模型...")
    try:
        model = ARIMA(ts, order=(1, 1, 1))
        result = model.fit()

        forecast = result.get_forecast(steps=horizon)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        print(f"\nARIMA模型参数:")
        print(result.summary().tables[1])

        print(f"\n预测结果（前10期）:")
        print(forecast_mean.head(10).round(1))

        return forecast_mean, forecast_ci
    except Exception as e:
        print(f"  ARIMA训练失败: {e}")
        return None


def evaluate_forecast(ts, horizon=30, method='ma'):
    """回测评估"""
    print_section("回测评估")

    if len(ts) < horizon * 2:
        print("  数据量不足，跳过回测评估")
        return

    train = ts[:-horizon]
    test = ts[-horizon:]

    print(f"\n训练集: {len(train)} 期")
    print(f"测试集: {len(test)} 期")

    if method == 'ma':
        window_size = min(7, len(train) // 2)
        ma = train.rolling(window=window_size).mean()
        last_ma = ma.iloc[-1]
        predictions = pd.Series([last_ma] * horizon, index=test.index)
    elif method == 'arima':
        statsmodels = safe_import('statsmodels', 'pip install statsmodels')
        if statsmodels is None:
            return
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(1, 1, 1))
        result = model.fit()
        forecast = result.get_forecast(steps=horizon)
        predictions = forecast.predicted_mean
    else:
        print(f"  未知方法: {method}")
        return

    mape = np.mean(np.abs((test - predictions) / test)) * 100
    mae = np.mean(np.abs(test - predictions))
    rmse = np.sqrt(np.mean((test - predictions) ** 2))

    print(f"\n回测指标:")
    print(f"  MAPE（平均绝对百分比误差）: {mape:.2f}%")
    print(f"  MAE（平均绝对误差）: {mae:.2f}")
    print(f"  RMSE（均方根误差）: {rmse:.2f}")

    print(f"\n测试集统计:")
    print(f"  均值: {test.mean():.2f}")
    print(f"  标准差: {test.std():.2f}")

    return {'mape': mape, 'mae': mae, 'rmse': rmse}


def main():
    parser = argparse.ArgumentParser(description='事故时间序列预测')
    parser.add_argument('--sample', type=int, default=0,
                        help='采样数量（默认0表示全量）')
    parser.add_argument('--freq', type=str, default='D',
                        choices=['D', 'W', 'M'],
                        help='频率：D(日)/W(周)/M(月)')
    parser.add_argument('--horizon', type=int, default=30,
                        help='预测步数（默认30）')
    parser.add_argument('--no-eval', action='store_true',
                        help='跳过回测评估')
    args = parser.parse_args()

    sample_size = args.sample if args.sample > 0 else None

    print_section("US Accidents 时间序列预测分析")
    print(f"\n采样数量: {'全量' if sample_size is None else f'{sample_size:,} 条'}")
    print(f"频率: {args.freq}")
    print(f"预测步数: {args.horizon}")

    usecols = ['Start_Time', 'ID', 'Severity']
    df = load_data(sample_size=sample_size, usecols=usecols)
    df = preprocess_basic(df)

    ts = prepare_time_series(df, freq=args.freq)
    if ts is None:
        return

    explore_time_patterns(ts)

    decompose_time_series(ts)

    ma_forecast = moving_average_forecast(ts, horizon=args.horizon)

    arima_result = arima_forecast(ts, horizon=args.horizon)

    if not args.no_eval:
        evaluate_forecast(ts, horizon=min(args.horizon, len(ts) // 2), method='ma')

    output_dir = ensure_output_dir('output')
    ts.to_csv(os.path.join(output_dir, f'time_series_{args.freq}.csv'), encoding='utf-8-sig')
    if ma_forecast is not None:
        ma_forecast.to_csv(os.path.join(output_dir, f'ma_forecast_{args.freq}.csv'), encoding='utf-8-sig')
    print(f"\n时间序列数据已保存到: {output_dir}")

    print_section("时间序列预测分析完成")


if __name__ == '__main__':
    main()
