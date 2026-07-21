"""
空间聚类分析模块
===============
功能：
    1. K-Means聚类（基于经纬度）
    2. DBSCAN密度聚类（发现热点区域）
    3. 肘部法则选择最优聚类数
    4. 聚类结果可视化
    5. 热点区域分析

使用方法：
    python -m analysis.spatial_clustering --sample 50000 --method both
"""

import argparse
import os
import numpy as np
import pandas as pd

from utils.tools import (print_section, print_subsection, load_data, preprocess_basic,
                         ensure_output_dir, safe_import)


def prepare_spatial_data(df):
    """准备空间数据"""
    print_section("准备空间数据")

    required_cols = ['Start_Lat', 'Start_Lng']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"错误: 缺少必要字段: {missing_cols}")
        return None

    df_spatial = df.dropna(subset=['Start_Lat', 'Start_Lng']).copy()
    print(f"\n有效空间数据: {len(df_spatial):,} 条")
    print(f"原始数据: {len(df):,} 条")
    print(f"过滤缺失: {len(df) - len(df_spatial):,} 条")

    coords = df_spatial[['Start_Lat', 'Start_Lng']].values
    return df_spatial, coords


def elbow_method(coords, max_k=20):
    """肘部法则选择最优K"""
    print_section("肘部法则选择最优K")

    sklearn_cluster = safe_import('sklearn.cluster', 'pip install scikit-learn')
    if sklearn_cluster is None:
        return None

    from sklearn.cluster import KMeans

    inertias = []
    k_values = range(2, max_k + 1)

    print(f"\n正在计算 K=2 到 K={max_k} 的惯性值...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}: 惯性值 = {kmeans.inertia_:.2e}")

    # 找到肘部点（二阶导数最大的点）
    deltas = np.diff(inertias)
    delta_deltas = np.diff(deltas)
    elbow_k = k_values[np.argmin(delta_deltas) + 1]

    print(f"\n推荐聚类数 K = {elbow_k}")

    return k_values, inertias, elbow_k


def kmeans_clustering(df_spatial, coords, k=15):
    """K-Means聚类"""
    print_section(f"K-Means聚类 (K={k})")

    sklearn_cluster = safe_import('sklearn.cluster', 'pip install scikit-learn')
    if sklearn_cluster is None:
        return None

    from sklearn.cluster import KMeans

    print("\n正在执行K-Means聚类...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
    labels = kmeans.fit_predict(coords)

    df_spatial['cluster_kmeans'] = labels

    print(f"\n聚类完成！")
    print(f"聚类数量: {k}")

    cluster_stats = df_spatial.groupby('cluster_kmeans').agg({
        'Start_Lat': ['mean', 'count'],
        'Start_Lng': ['mean'],
        'Severity': ['mean']
    })
    cluster_stats.columns = ['avg_lat', 'count', 'avg_lng', 'avg_severity']
    cluster_stats = cluster_stats.sort_values('count', ascending=False)

    print("\n--- 聚类统计（按事故数量排序） ---")
    print(cluster_stats.round(4))

    print("\n--- Top 10事故热点聚类 ---")
    top_clusters = cluster_stats.head(10)
    for i, (cluster, row) in enumerate(top_clusters.iterrows(), 1):
        print(f"  {i:2d}. 聚类{cluster}: {row['count']:,}条事故, "
              f"平均严重程度: {row['avg_severity']:.3f}, "
              f"中心: ({row['avg_lat']:.4f}, {row['avg_lng']:.4f})")

    return df_spatial, cluster_stats, kmeans


def dbscan_clustering(df_spatial, coords, eps_km=36.6, min_samples=50):
    """DBSCAN密度聚类（默认 eps=36.6km，参考同行城市级热点半径）"""
    print_section(f"DBSCAN密度聚类 (eps={eps_km}km, min_samples={min_samples})")

    sklearn_cluster = safe_import('sklearn.cluster', 'pip install scikit-learn')
    if sklearn_cluster is None:
        return None

    from sklearn.cluster import DBSCAN

    # 大数据量时抽样（DBSCAN O(n²)，7M 点不可行）
    n_total = len(coords)
    sample_coords = coords
    sample_df = df_spatial
    sampled = False
    if n_total > 200000:
        np.random.seed(42)
        idx = np.random.choice(n_total, size=min(200000, n_total), replace=False)
        sample_coords = coords[idx]
        sample_df = df_spatial.iloc[idx].copy()
        sampled = True
        print(f"  ⚠️ 数据量 {n_total:,} 超过 50万，随机抽样 {len(sample_coords):,} 条用于 DBSCAN")

    eps_deg = eps_km / 111.0  # 经纬度近似: 1° ≈ 111km
    print(f"\n正在执行DBSCAN聚类...")
    print(f"  eps(度): {eps_deg:.6f}")
    print(f"  样本数: {len(sample_coords):,}")

    dbscan = DBSCAN(eps=eps_deg, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(sample_coords)

    if sampled:
        # 将标签映射回全量数据（用最近邻）
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(sample_coords)
        _, indices = nn.kneighbors(coords)
        full_labels = labels[indices.flatten()]
        df_spatial['cluster_dbscan'] = full_labels
    else:
        df_spatial['cluster_dbscan'] = labels

    n_clusters = len(set(df_spatial['cluster_dbscan'])) - (1 if -1 in df_spatial['cluster_dbscan'].values else 0)
    n_noise = (df_spatial['cluster_dbscan'] == -1).sum()

    print(f"\n聚类完成！")
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise:,} ({n_noise/len(df_spatial)*100:.2f}%)")

    if n_clusters > 0:
        cluster_stats = df_spatial[df_spatial['cluster_dbscan'] != -1].groupby('cluster_dbscan').agg({
            'Start_Lat': ['mean', 'count'],
            'Start_Lng': ['mean'],
            'Severity': ['mean']
        })
        cluster_stats.columns = ['avg_lat', 'count', 'avg_lng', 'avg_severity']
        cluster_stats = cluster_stats.sort_values('count', ascending=False)

        print("\n--- 聚类统计（按事故数量排序） ---")
        print(cluster_stats.head(15).round(4))

        print("\n--- Top 10事故热点聚类 ---")
        top_clusters = cluster_stats.head(10)
        for i, (cluster, row) in enumerate(top_clusters.iterrows(), 1):
            print(f"  {i:2d}. 聚类{cluster}: {row['count']:,}条事故, "
                  f"平均严重程度: {row['avg_severity']:.3f}, "
                  f"中心: ({row['avg_lat']:.4f}, {row['avg_lng']:.4f})")

        return df_spatial, cluster_stats, dbscan
    else:
        print("  未发现任何聚类（可尝试增大 eps 或减小 min_samples）")
        return df_spatial, None, None


def kde_hotspot_analysis(df_spatial, output_dir=None):
    """
    KDE 核密度估计热力图：识别事故空间热点区域
    比固定半径 DBSCAN 更灵活，能展示连续的密度分布。
    """
    print_section("KDE 核密度估计（空间热点）")

    kde_lib = safe_import('scipy.stats', 'pip install scipy')
    if kde_lib is None:
        return None

    from scipy.stats import gaussian_kde

    coords = df_spatial[['Start_Lng', 'Start_Lat']].dropna().values.T  # (2, N)

    # 大数据抽样
    if coords.shape[1] > 200000:
        np.random.seed(42)
        idx = np.random.choice(coords.shape[1], size=200000, replace=False)
        coords_sampled = coords[:, idx]
        print(f"  KDE 抽样: {coords.shape[1]:,} → {coords_sampled.shape[1]:,} 点")
    else:
        coords_sampled = coords

    try:
        kde = gaussian_kde(coords_sampled, bw_method=0.15)  # 带宽调参
        print(f"  KDE 带宽(bw): {kde.factor:.3f}")

        # 在规则网格上评估密度（覆盖美国大陆范围）
        lng_range = (-130, -65)
        lat_range = (24, 50)
        grid_lng = np.linspace(lng_range[0], lng_range[1], 200)
        grid_lat = np.linspace(lat_range[0], lat_range[1], 150)
        lng_grid, lat_grid = np.meshgrid(grid_lng, grid_lat)
        grid_pts = np.vstack([lng_grid.ravel(), lat_grid.ravel()])

        density = kde(grid_pts).reshape(lng_grid.shape)

        # 找到最高密度点（最热的热点）
        flat_idx = np.unravel_index(density.argmax(), density.shape)
        peak_lng = grid_lng[flat_idx[1]]
        peak_lat = grid_lat[flat_idx[0]]
        peak_val = density.max()

        # 密度统计
        high_density_mask = density > np.percentile(density, 95)
        n_high_cells = high_density_mask.sum()
        avg_density = density.mean()

        print(f"\n  热点中心（KDE 峰值）: ({peak_lng:.2f}, {peak_lat:.2f}), 密度={peak_val:.4e}")
        print(f"  平均密度: {avg_density:.4e}")
        print(f"  高密度区域（>P95）: 占网格 {n_high_cells}/{density.size} "
              f"({n_high_cells/density.size*100:.1f}%)")

        # 保存 KDE 结果供可视化使用
        result = {
            'density': density,
            'grid_lng': grid_lng,
            'grid_lat': grid_lat,
            'peak': (peak_lng, peak_lat, peak_val),
            'coords_used': coords_sampled
        }

        import pickle
        if output_dir is None:
            output_dir = ensure_output_dir('output')
        kde_path = os.path.join(output_dir, 'kde_result.pkl')
        with open(kde_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  KDE 结果已保存: {kde_path}")

        return result

    except Exception as e:
        print(f"  KDE 计算失败: {e}")
        return None


def analyze_hotspots(df_spatial, cluster_col='cluster_kmeans', top_n=10):
    """热点区域分析"""
    print_section("热点区域深度分析")

    if cluster_col not in df_spatial.columns:
        print(f"  缺少聚类字段: {cluster_col}")
        return

    cluster_stats = df_spatial.groupby(cluster_col).agg({
        'Start_Lat': ['mean', 'count'],
        'Start_Lng': ['mean'],
        'Severity': ['mean', 'max', 'min']
    })
    cluster_stats.columns = ['avg_lat', 'count', 'avg_lng', 'avg_severity', 'max_severity', 'min_severity']
    cluster_stats = cluster_stats.sort_values('count', ascending=False)

    print(f"\n--- 平均严重程度最高的热点聚类（Top {top_n}） ---")
    severity_top = cluster_stats.sort_values('avg_severity', ascending=False).head(top_n)
    print(severity_top[['count', 'avg_severity', 'max_severity']].round(3))

    if 'State' in df_spatial.columns:
        print(f"\n--- 各聚类的主要州分布 ---")
        for cluster in cluster_stats.head(top_n).index:
            cluster_data = df_spatial[df_spatial[cluster_col] == cluster]
            top_states = cluster_data['State'].value_counts().head(3)
            print(f"  聚类{cluster}: {top_states.to_dict()}")


def main():
    parser = argparse.ArgumentParser(description='事故空间聚类分析')
    parser.add_argument('--sample', type=int, default=50000,
                        help='采样数量（默认50000，0表示全量）')
    parser.add_argument('--method', type=str, default='both',
                        choices=['kmeans', 'dbscan', 'both'],
                        help='聚类方法（默认两种都运行）')
    parser.add_argument('--k', type=int, default=15,
                        help='K-Means聚类数（默认15）')
    parser.add_argument('--eps', type=float, default=36.6,
                        help='DBSCAN半径(km)（默认36.6，城市级热点）')
    parser.add_argument('--min-samples', type=int, default=50,
                        help='DBSCAN最小样本数（默认50）')
    parser.add_argument('--elbow', action='store_true',
                        help='使用肘部法则选择最优K')
    args = parser.parse_args()

    sample_size = args.sample if args.sample > 0 else None

    print_section("US Accidents 空间聚类分析")
    print(f"\n采样数量: {sample_size:,} 条" if sample_size else "\n全量分析")
    print(f"聚类方法: {args.method}")

    usecols = ['Start_Lat', 'Start_Lng', 'Severity', 'State', 'City', 'County']
    df = load_data(sample_size=sample_size, usecols=usecols)
    df = preprocess_basic(df)

    result = prepare_spatial_data(df)
    if result is None:
        return
    df_spatial, coords = result

    # 内存 + 耗时保护：K-Means 输入上限（避免全量 770 万点时过久）
    SPATIAL_MAX_ROWS = 2_000_000
    if len(df_spatial) > SPATIAL_MAX_ROWS:
        print(f"\n  [内存保护] 空间数据 {len(df_spatial):,} 条超过上限 {SPATIAL_MAX_ROWS:,}，随机降采样")
        df_spatial = df_spatial.sample(n=SPATIAL_MAX_ROWS, random_state=42).reset_index(drop=True)
        coords = df_spatial[['Start_Lat', 'Start_Lng']].values

    if args.elbow:
        elbow_result = elbow_method(coords, max_k=20)
        if elbow_result:
            k_values, inertias, elbow_k = elbow_result
            args.k = elbow_k
            print(f"\n使用肘部法则推荐的K值: {args.k}")

    if args.method in ['kmeans', 'both']:
        kmeans_result = kmeans_clustering(df_spatial, coords, k=args.k)
        if kmeans_result:
            df_spatial, kmeans_stats, kmeans_model = kmeans_result
            analyze_hotspots(df_spatial, 'cluster_kmeans')

    if args.method in ['dbscan', 'both']:
        dbscan_result = dbscan_clustering(df_spatial, coords, eps_km=args.eps, min_samples=args.min_samples)
        if dbscan_result:
            df_spatial, dbscan_stats, dbscan_model = dbscan_result
            analyze_hotspots(df_spatial, 'cluster_dbscan')

    # KDE 核密度热力图（比固定半径 DBSCAN 更直观的热点展示）
    kde_hotspot_analysis(df_spatial)

    output_dir = ensure_output_dir('output')
    if 'cluster_kmeans' in df_spatial.columns:
        df_spatial[['Start_Lat', 'Start_Lng', 'Severity', 'cluster_kmeans']].to_csv(
            os.path.join(output_dir, 'kmeans_clusters.csv'), index=False, encoding='utf-8-sig')
    if 'cluster_dbscan' in df_spatial.columns:
        df_spatial[['Start_Lat', 'Start_Lng', 'Severity', 'cluster_dbscan']].to_csv(
            os.path.join(output_dir, 'dbscan_clusters.csv'), index=False, encoding='utf-8-sig')
    print(f"\n聚类结果已保存到: {output_dir}")

    print_section("空间聚类分析完成")


if __name__ == '__main__':
    main()
