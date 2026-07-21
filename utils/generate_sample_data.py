"""
示例数据生成器 | Synthetic Sample Data Generator
================================================

生成一份与真实 US Accidents (March 2023) schema 一致的**合成数据集**，
用于在没有 Kaggle 原始数据（约 1.5GB）时，让整套分析 pipeline 可端到端运行。

⚠️ 重要说明
-----------
1. 本数据是「合成数据」（synthetic），用于演示与教学，**并非真实事故记录**。
2. 分布经过设计以贴近真实规律：
   - 事故多集中在早晚通勤高峰（约 7-9 点、16-18 点）
   - 晴天占比最高，恶劣天气占比较小
   - 各州分散（加利福尼亚州并非 98%，约为真实量级）
   - 严重程度以 2 级（中等）为主，3 级次之
3. 报告中的所有数字均来自「在本数据上运行分析结果」，完全可复现。
4. 如需真实分析，请从 Kaggle 下载数据集放入 data/ 并重新运行：
   https://www.kaggle.com/sobhanmoosavi/us-accidents

用法：
    python utils/generate_sample_data.py --n 50000
    python utils/generate_sample_data.py --n 50000 --out data/US_Accidents_March23.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

# ------------------------- 城市 / 州坐标库 -------------------------
# (城市, 县, 州, 纬度, 经度, 相对权重)
CITY_DB = [
    ("Sacramento", "Sacramento", "CA", 38.58, -121.49, 2.0),
    ("San Jose", "Santa Clara", "CA", 37.34, -121.89, 2.0),
    ("Los Angeles", "Los Angeles", "CA", 34.05, -118.24, 4.0),
    ("San Francisco", "San Francisco", "CA", 37.77, -122.42, 2.5),
    ("Oakland", "Alameda", "CA", 37.80, -122.27, 1.5),
    ("San Diego", "San Diego", "CA", 32.72, -117.16, 2.0),
    ("Houston", "Harris", "TX", 29.76, -95.37, 3.0),
    ("Dallas", "Dallas", "TX", 32.78, -96.80, 2.5),
    ("Austin", "Travis", "TX", 30.27, -97.74, 1.5),
    ("Miami", "Miami-Dade", "FL", 25.76, -80.19, 2.0),
    ("Orlando", "Orange", "FL", 28.54, -81.38, 1.5),
    ("Tampa", "Hillsborough", "FL", 27.95, -82.46, 1.2),
    ("New York", "New York", "NY", 40.71, -74.00, 5.0),
    ("Buffalo", "Erie", "NY", 42.89, -78.88, 1.0),
    ("Philadelphia", "Philadelphia", "PA", 39.95, -75.17, 2.5),
    ("Pittsburgh", "Allegheny", "PA", 40.44, -79.99, 1.3),
    ("Columbus", "Franklin", "OH", 39.96, -82.99, 1.5),
    ("Cleveland", "Cuyahoga", "OH", 41.50, -81.69, 1.3),
    ("Chicago", "Cook", "IL", 41.88, -87.63, 4.0),
    ("Charlotte", "Mecklenburg", "NC", 35.23, -80.84, 1.8),
    ("Raleigh", "Wake", "NC", 35.78, -78.64, 1.0),
    ("Detroit", "Wayne", "MI", 42.33, -83.05, 1.8),
    ("Atlanta", "Fulton", "GA", 33.75, -84.39, 2.5),
    ("Seattle", "King", "WA", 47.61, -122.33, 2.0),
    ("Phoenix", "Maricopa", "AZ", 33.45, -112.07, 2.2),
    ("Boston", "Suffolk", "MA", 42.36, -71.06, 2.0),
]

# ------------------------- 天气 / 道路词典 -------------------------
WEATHER_CONDITIONS = [
    ("Clear", 0.40), ("Overcast", 0.13), ("Partly Cloudy", 0.11),
    ("Mostly Cloudy", 0.10), ("Scattered Clouds", 0.08),
    ("Light Rain", 0.06), ("Rain", 0.04), ("Heavy Rain", 0.02),
    ("Fog", 0.03), ("Haze", 0.02), ("Light Snow", 0.01),
]
ROADS = ["I-80", "I-5", "I-10", "I-95", "I-35", "I-75", "I-90",
         "US-101", "US-1", "SR-99", "I-280"]
DIRECTIONS = ["Westbound", "Eastbound", "Northbound", "Southbound"]
STREETS = ["Main St", "Broadway", "5th Ave", "Market St",
           "University Ave", "Park Blvd", "Washington St", "Lake Rd"]
WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
TWILIGHT = ["Day", "Night"]
SIDES = ["R", "L"]

DESC_TEMPLATES = [
    "Accident on {road} {dir} near {city}. Vehicle collision reported, traffic blocked, delays expected.",
    "Rear-end accident at {street} and {road} in {city}. Minor damage, slowdown.",
    "Multi-vehicle collision on {road} {dir} in {city}. Injuries reported, lanes closed.",
    "Single-vehicle crash on {road} {dir} near {city}. Car struck barrier, fire reported.",
    "Pedestrian accident on {street} in {city}. Emergency responders on scene.",
    "Truck accident on {road} {dir} in {city}. Interstate closed, long delays.",
    "Vehicle lost control on {road} {dir} near {city}. Rollover, serious injuries.",
    "Collision on {street} in {city}. Fender bender, minimal delay.",
]

# 设施出现概率（参考真实数据集量级）
FACILITY_PROBS = {
    "Amenity": 0.01, "Bump": 0.02, "Crossing": 0.08, "Give_Way": 0.02,
    "Junction": 0.13, "No_Exit": 0.02, "Railway": 0.02, "Roundabout": 0.01,
    "Station": 0.02, "Stop": 0.04, "Traffic_Calming": 0.02,
    "Traffic_Signal": 0.10, "Turning_Loop": 0.005,
}


def generate(n, seed=42):
    rng = np.random.default_rng(seed)
    cities = np.array([c[0] for c in CITY_DB])
    counties = np.array([c[1] for c in CITY_DB])
    states = np.array([c[2] for c in CITY_DB])
    lats = np.array([c[3] for c in CITY_DB], dtype=float)
    lngs = np.array([c[4] for c in CITY_DB], dtype=float)
    weights = np.array([c[5] for c in CITY_DB], dtype=float)
    weights = weights / weights.sum()

    # 每行分配一个城市
    idx = rng.choice(len(CITY_DB), size=n, p=weights)
    city = cities[idx]
    county = counties[idx]
    state = states[idx]
    base_lat = lats[idx] + rng.normal(0, 0.15, n)
    base_lng = lngs[idx] + rng.normal(0, 0.15, n)

    # 时间：2016-02-01 ~ 2023-03-31，小时偏向通勤高峰
    start = pd.Timestamp("2016-02-01")
    end = pd.Timestamp("2023-03-31")
    span_days = (end - start).days
    day_offset = rng.integers(0, span_days + 1, n)
    base_dates = pd.to_datetime(start) + pd.to_timedelta(day_offset, unit="D")

    # 小时权重：双峰（早 7-9、晚 16-18）
    hour_probs = np.array([
        0.010, 0.008, 0.006, 0.005, 0.006, 0.012, 0.030, 0.055,
        0.060, 0.045, 0.038, 0.040, 0.045, 0.042, 0.045, 0.050,
        0.058, 0.060, 0.052, 0.040, 0.032, 0.025, 0.018, 0.012,
    ])
    hour_probs = hour_probs / hour_probs.sum()
    hour = rng.choice(24, size=n, p=hour_probs)
    minute = rng.integers(0, 60, n)
    second = rng.integers(0, 60, n)
    start_time = (base_dates
                  + pd.to_timedelta(hour, unit="h")
                  + pd.to_timedelta(minute, unit="m")
                  + pd.to_timedelta(second, unit="s"))

    month = start_time.month.to_numpy()
    # 严重程度分布：2 级为主，3 级次之（多数类约 58%）
    sev_probs = np.array([0.03, 0.58, 0.37, 0.02])
    severity = rng.choice([1, 2, 3, 4], size=n, p=sev_probs)

    # 持续时间（分钟）：随严重程度略增
    dur_base = np.array([30, 45, 70, 95])[severity - 1]
    duration_min = np.clip(dur_base + rng.normal(0, 15, n), 5, 600)
    end_time = start_time + pd.to_timedelta(duration_min, unit="m")

    # 天气
    wc_names = [w[0] for w in WEATHER_CONDITIONS]
    wc_probs = np.array([w[1] for w in WEATHER_CONDITIONS])
    wc_probs = wc_probs / wc_probs.sum()
    weather = rng.choice(wc_names, size=n, p=wc_probs)

    is_rain = np.isin(weather, ["Light Rain", "Rain", "Heavy Rain"])
    is_snow = weather == "Light Snow"
    is_fog = weather == "Fog"

    # 温度：随季节（夏高冬低）与纬度（北冷南暖）
    temp_season = 62 + 18 * np.sin((month - 4) / 12 * 2 * np.pi)
    temperature = temp_season - (base_lat - 35) * 0.7 + rng.normal(0, 8, n)
    temperature = np.clip(temperature, -15, 120)

    humidity = np.where(is_rain | is_snow,
                        rng.integers(75, 100, n),
                        rng.integers(25, 85, n)).astype(float)
    pressure = rng.normal(29.95, 0.4, n)
    pressure = np.clip(pressure, 28.5, 31.0)

    visibility = np.where(is_fog, rng.uniform(0.1, 1.5, n),
                 np.where(is_rain, rng.uniform(2, 7, n),
                          rng.uniform(7, 10, n)))
    wind_speed = np.clip(rng.gamma(2.0, 4.0, n), 0, 45)
    wind_dir = rng.choice(WIND_DIRS, size=n)
    precip = np.where(is_rain, rng.uniform(0.02, 0.6, n),
             np.where(is_snow, rng.uniform(0.01, 0.3, n), 0.0))

    # 风寒：偏冷时计算，否则缺失（贴近真实高缺失率）
    wind_chill = np.where(temperature < 50,
                          temperature - rng.uniform(0, 8, n),
                          np.nan)

    sunrise_sunset = np.where((hour >= 6) & (hour < 19), "Day", "Night")

    # 道路设施布尔（确保输出全部 13 个设施列，与 FACILITY_COLS 一致）
    facility_data = {}
    for col, p in FACILITY_PROBS.items():
        facility_data[col] = (rng.random(n) < p).astype(int)

    # 描述文本
    road = rng.choice(ROADS, size=n)
    direction = rng.choice(DIRECTIONS, size=n)
    street = rng.choice(STREETS, size=n)
    tmpl = rng.integers(0, len(DESC_TEMPLATES), n)
    description = np.array([
        DESC_TEMPLATES[tmpl[i]].format(road=road[i], dir=direction[i],
                                        city=city[i], street=street[i])
        for i in range(n)
    ])

    # 组装
    df = pd.DataFrame({
        "ID": [f"A-{i:07d}" for i in range(1, n + 1)],
        "Source": "Source1",
        "Severity": severity,
        "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "End_Time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Start_Lat": np.round(base_lat, 6),
        "Start_Lng": np.round(base_lng, 6),
        "End_Lat": np.round(base_lat + rng.normal(0, 0.01, n), 6),
        "End_Lng": np.round(base_lng + rng.normal(0, 0.01, n), 6),
        "Distance(mi)": np.round(np.clip(rng.gamma(1.5, 0.8, n), 0.01, 8), 3),
        "Description": description,
        "Number": rng.integers(1, 500, n),
        "Street": street,
        "Side": rng.choice(SIDES, size=n),
        "City": city,
        "County": county,
        "State": state,
        "Zipcode": rng.integers(10000, 99999, n),
        "Country": "US",
        "Timezone": "US/" + state,
        "Airport_Code": state + rng.choice(["001", "002", "003"], size=n),
        "Weather_Timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature(F)": np.round(temperature, 1),
        "Wind_Chill(F)": np.round(wind_chill, 1),
        "Humidity(%)": humidity,
        "Pressure(in)": np.round(pressure, 2),
        "Visibility(mi)": np.round(visibility, 2),
        "Wind_Direction": wind_dir,
        "Wind_Speed(mph)": np.round(wind_speed, 1),
        "Precipitation(in)": np.round(precip, 3),
        "Weather_Condition": weather,
        "Sunrise_Sunset": sunrise_sunset,
        "Civil_Twilight": rng.choice(TWILIGHT, size=n),
        "Nautical_Twilight": rng.choice(TWILIGHT, size=n),
        "Astronomical_Twilight": rng.choice(TWILIGHT, size=n),
    })
    for col, vals in facility_data.items():
        df[col] = vals

    return df


def main():
    parser = argparse.ArgumentParser(description="生成 US Accidents 示例数据集")
    parser.add_argument("--n", type=int, default=50000, help="生成记录数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", type=str,
                        default=None, help="输出 CSV 路径（默认 data/US_Accidents_March23.csv）")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(root, "data", "US_Accidents_March23.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"正在生成 {args.n:,} 条示例数据（seed={args.seed}）...")
    df = generate(args.n, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"已保存: {out_path}")
    print(f"  行数: {len(df):,}，列数: {df.shape[1]}")
    print(f"  州分布 Top5:\n{df['State'].value_counts().head(5).to_string()}")
    print(f"  严重程度分布:\n{df['Severity'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
