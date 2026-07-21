"""
预测建模模块
=============
功能：
    1. 特征工程
    2. 数据划分与预处理
    3. XGBoost 分类模型
    4. Random Forest 分类模型
    5. 模型评估与对比
    6. 特征重要性分析
    7. 混淆矩阵与分类报告

使用方法：
    python -m analysis.predictive_modeling --sample 100000
    python -m analysis.predictive_modeling --sample 50000 --model xgboost
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from utils.tools import (print_section, print_subsection, load_data, preprocess_basic,
                         engineer_features, ensure_output_dir, safe_import, FACILITY_COLS)


def feature_engineering(df):
    """
    特征工程

    Args:
        df: 预处理后的DataFrame

    Returns:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名列表
    """
    print_section("特征工程")

    df = df.copy()
    features = []

    # 1. 时间特征
    print("\n[1/6] 提取时间特征...")
    time_features = ['Hour', 'DayOfWeek', 'Month', 'Is_Weekend']
    for f in time_features:
        if f in df.columns:
            features.append(f)

    # 2. 地理特征（州编码）
    print("[2/6] 编码地理特征...")
    if 'State' in df.columns:
        df['State_encoded'] = df['State'].astype('category').cat.codes
        features.append('State_encoded')

    # 3. 天气数值特征
    print("[3/6] 添加天气数值特征...")
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    for col in weather_cols:
        if col in df.columns:
            features.append(col)
            df[col] = df[col].fillna(df[col].median())

    # 4. 道路设施特征（布尔转int）
    print("[4/6] 添加道路设施特征...")
    for col in FACILITY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
            features.append(col)

    # 5. 距离特征
    print("[5/6] 添加距离特征...")
    if 'Distance(mi)' in df.columns:
        features.append('Distance(mi)')

    # 6. 日出日落特征
    print("[6/6] 编码日出日落特征...")
    if 'Sunrise_Sunset' in df.columns:
        df['Sunrise_Sunset_encoded'] = (df['Sunrise_Sunset'] == 'Day').astype(int)
        features.append('Sunrise_Sunset_encoded')

    # 7. 高级特征（Is_Night / Is_Holiday / 文本长度等）
    print("[7/7] 添加高级特征...")
    df = engineer_features(df)  # 会添加 Is_Night, Is_Holiday, Description_Length, Street_Length
    for extra_col in ['Is_Night', 'Is_Holiday', 'Description_Length', 'Street_Length']:
        if extra_col in df.columns:
            features.append(extra_col)

    # 目标变量
    target = 'Severity'
    if target not in df.columns:
        print(f"错误: 找不到目标变量 {target}")
        sys.exit(1)

    df_clean = df[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target] - 1

    print(f"\n特征数量: {len(features)}")
    print(f"样本数量: {len(X):,}")
    print(f"类别数量: {y.nunique()}")
    print(f"\n特征列表:")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")

    return X, y, features


def train_test_split(X, y, test_size=0.2, random_state=42):
    """训练集测试集划分"""
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)
    except ImportError:
        np.random.seed(random_state)
        n = len(X)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_xgboost(X_train, y_train, n_classes):
    """训练XGBoost模型"""
    print_subsection("训练 XGBoost 模型")

    xgb = safe_import('xgboost', 'pip install xgboost')
    if xgb is None:
        return None

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )

    print("  正在训练 XGBoost...")
    model.fit(X_train, y_train)
    print("  训练完成！")

    return model


def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    print_subsection("训练 Random Forest 模型")

    sklearn_ensemble = safe_import('sklearn.ensemble', 'pip install scikit-learn')
    if sklearn_ensemble is None:
        return None

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("  正在训练 Random Forest...")
    model.fit(X_train, y_train)
    print("  训练完成！")

    return model


def train_logistic_regression(X_train, y_train):
    """训练逻辑回归模型（基线）"""
    print_subsection("训练 Logistic Regression 模型")

    sklearn_lr = safe_import('sklearn.linear_model', 'pip install scikit-learn')
    if sklearn_lr is None:
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    print("  正在训练 Logistic Regression...")
    model.fit(X_train_scaled, y_train)
    print("  训练完成！")

    return model, scaler


def train_baseline(X_train, y_train):
    """训练基线模型（始终预测最多数类），用于衡量其他模型的实际增益"""
    print_subsection("训练 Baseline（最多数类）模型")

    sklearn_dummy = safe_import('sklearn.dummy', 'pip install scikit-learn')
    if sklearn_dummy is None:
        return None

    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy='most_frequent')
    print("  正在训练 Baseline...")
    model.fit(X_train, y_train)
    print("  训练完成！（该模型仅输出训练集中占比最高的类别）")

    return model


def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    评估模型

    Returns:
        dict: 评估指标（含逐类与宏平均）
    """
    print_subsection(f"评估 {model_name}")

    X_test_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = np.array(model.predict(X_test_eval))
    y_test_arr = np.array(y_test)

    accuracy = np.mean(y_pred == y_test_arr)

    classes = np.unique(np.concatenate([y_test_arr, y_pred]))
    precisions, recalls, f1s, supports = [], [], [], []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_test_arr == c))
        fp = np.sum((y_pred == c) & (y_test_arr != c))
        fn = np.sum((y_pred != c) & (y_test_arr == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(np.sum(y_test_arr == c))

    total_weight = sum(supports)
    if total_weight > 0:
        precision = sum(p * w for p, w in zip(precisions, supports)) / total_weight
        recall = sum(r * w for r, w in zip(recalls, supports)) / total_weight
        f1 = sum(f * w for f, w in zip(f1s, supports)) / total_weight
    else:
        precision = recall = f1 = 0.0

    # 宏平均（各类等权，避免加权指标掩盖少数类）
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    print(f"\n  准确率 (Accuracy):   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  加权精确率/召回率/F1:  {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"  宏平均精确率/召回率/F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")

    # 逐类明细
    print(f"\n  逐类指标 (Class | Precision | Recall | F1 | Support):")
    for i, c in enumerate(classes):
        print(f"    等级{c+1}:  {precisions[i]:.4f}  {recalls[i]:.4f}  {f1s[i]:.4f}  {supports[i]:,}")

    print(f"\n  混淆矩阵:")
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_test_arr, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] += 1

    cm_df = pd.DataFrame(
        cm,
        index=[f'实际_{c+1}' for c in classes],
        columns=[f'预测_{c+1}' for c in classes]
    )
    print(cm_df)

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'y_pred': y_pred,
        'y_test': y_test
    }


def analyze_feature_importance(model, X_test, y_test, feature_names, model_name,
                                scaler=None, top_n=20):
    """
    分析特征重要性

    优先使用置换重要性（Permutation Importance）：随机打乱某列后观察
    准确率下降幅度，不受特征量纲影响，比"原始系数绝对值"更可靠。
    当无法计算时回退到模型内重要性。
    """
    print_subsection(f"{model_name} 特征重要性 Top {top_n}")

    try:
        importances = None
        method = None

        sklearn_inspection = safe_import('sklearn.inspection', 'pip install scikit-learn')
        if sklearn_inspection is not None and X_test is not None:
            from sklearn.inspection import permutation_importance

            # LR 在训练时做了标准化，置换重要性需经同一 scaler 再预测
            class _ScaledPredictor:
                def __init__(self, m, s):
                    self.m, self.s = m, s

                def fit(self, X, y=None):
                    return self

                def predict(self, X):
                    if self.s is not None:
                        X = self.s.transform(X)
                    return self.m.predict(X)

            wrapped = _ScaledPredictor(model, scaler)
            try:
                perm = permutation_importance(
                    wrapped, X_test, np.array(y_test),
                    n_repeats=5, random_state=42, scoring='accuracy', n_jobs=-1
                )
                importances = perm.importances_mean
                method = '置换重要性（准确率下降量，5 次重复）'
            except Exception as e:
                print(f"  置换重要性计算失败（{e}），回退到模型内重要性")

        if importances is None:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                method = '模型内重要性（Gini/impurity）'
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).mean(axis=0)
                method = '标准化逻辑回归系数绝对值（特征已标准化）'
            else:
                print("  该模型不支持特征重要性分析")
                return None

        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        top = fi_df.head(top_n)
        print(f"\n  方法: {method}")
        print(f"  特征重要性 Top {top_n}:")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            bar = '█' * int(row['importance'] / top['importance'].max() * 30)
            print(f"  {i:2d}. {row['feature']:<25s} {row['importance']:.4f} {bar}")

        return fi_df

    except Exception as e:
        print(f"  特征重要性分析失败: {e}")
        return None


def compare_models(results):
    """模型对比"""
    print_section("模型对比")

    if not results:
        print("  没有可对比的模型结果")
        return

    comparison = []
    for r in results:
        comparison.append({
            '模型': r['model'],
            '准确率': f"{r['accuracy']*100:.2f}%",
            '加权F1': f"{r['f1']:.4f}",
            '宏平均F1': f"{r.get('macro_f1', float('nan')):.4f}"
        })

    comparison_df = pd.DataFrame(comparison)
    print(f"\n{comparison_df.to_string(index=False)}")

    real_models = [r for r in results if not r.get('is_baseline')]
    if real_models:
        best = max(real_models, key=lambda x: x['f1'])
        print(f"\n🏆 最佳模型（按加权F1，不含基线）: {best['model']} (F1={best['f1']:.4f})")

    baseline = next((r for r in results if r.get('is_baseline')), None)
    if baseline:
        print(f"\n📌 基线参考: 仅猜最多数类的准确率 = {baseline['accuracy']*100:.2f}%")
        print("   其他模型的准确率需明显高于该值才有实际意义。")


# ==================== 二分类（Severity≥3 vs <3）====================

def run_binary_classification(df, test_size=0.2):
    """
    二分类预测：将 Severity 重编码为二值目标
      - 1 = 严重事故（Severity ≥ 3）
      - 0 = 非严重事故（Severity ≤ 2）

    这是 Kaggle 竞赛和多数同行采用的目标定义，因为：
      ① 4 分类中 1/4 级样本极少，信号弱，"无模型超越基线"
      ② 换成"是否严重"的二元问题后，预测力显著提升
    """
    print_section("二分类预测（Severity ≥ 3 vs < 3）")

    df = df.copy()
    if 'Severity' not in df.columns:
        print("  缺少 Severity 列，跳过二分类")
        return []

    # 创建二分类目标
    y_binary = (df['Severity'] >= 3).astype(int)  # 1=严重(3,4), 0=非严重(1,2)

    pos_rate = y_binary.mean() * 100
    neg_rate = (1 - y_binary.mean()) * 100
    print(f"\n  二分类目标分布：")
    print(f"    0 = 非严重(1-2级): {neg_rate:.1f}% ({(y_binary==0).sum():,}条)")
    print(f"    1 = 严重(3-4级):   {pos_rate:.1f}% ({(y_binary==1).sum():,}条)")

    # 特征工程（复用已有的 feature_engineering，但用二分类目标）
    X, _, feature_names = feature_engineering(df)
    X = X.loc[y_binary.index]  # 对齐索引

    # 数据划分
    from sklearn.model_selection import train_test_split as sk_split
    X_train, X_test, y_train_bin, y_test_bin = sk_split(
        X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
    )
    print(f"\n  训练集: {len(X_train):,} | 测试集: {len(X_test):,}")

    binary_results = []

    # 二分类基线（最多数类）
    baseline = train_baseline(X_train, y_train_bin)
    if baseline:
        res = evaluate_model(baseline, X_test, y_test_bin,
                            'Baseline_Binary (most_frequent)')
        res['is_baseline'] = True
        binary_results.append(res)

    # Random Forest
    rf = train_random_forest(X_train, y_train_bin)
    if rf:
        res = evaluate_model(rf, X_test, y_test_bin, 'Random Forest Binary')
        binary_results.append(res)
        analyze_feature_importance(rf, X_test, y_test_bin, feature_names,
                                   'RF Binary', top_n=15)

    # XGBoost
    xgb = train_xgboost(X_train, y_train_bin, n_classes=2)
    if xgb:
        res = evaluate_model(xgb, X_test, y_test_bin, 'XGBoost Binary')
        binary_results.append(res)
        analyze_feature_importance(xgb, X_test, y_test_bin, feature_names,
                                   'XGBoost Binary', top_n=15)

    # Logistic Regression
    lr_tuple = train_logistic_regression(X_train, y_train_bin)
    if lr_tuple:
        lr_model, lr_scaler = lr_tuple
        res = evaluate_model(lr_model, X_test, y_test_bin,
                            'Logistic Regression Binary', scaler=lr_scaler)
        binary_results.append(res)
        analyze_feature_importance(lr_model, X_test, y_test_bin, feature_names,
                                   'LR Binary', scaler=lr_scaler, top_n=15)

    # 对比输出
    if len(binary_results) > 1:
        print_subsection("二分类模型对比")
        comp = []
        for r in binary_results:
            comp.append({
                '模型': r['model'],
                '是否基线': '是' if r.get('is_baseline') else '否',
                '准确率': f"{r['accuracy']*100:.2f}%",
                '精确率': f"{r['precision']:.4f}",
                '召回率': f"{r['recall']:.4f}",
                'F1': f"{r['f1']:.4f}",
                '宏平均F1': f"{r.get('macro_f1', float('nan')):.4f}"
            })
        comp_df = pd.DataFrame(comp)
        print(f"\n{comp_df.to_string(index=False)}")

        # 找最佳非基线模型
        real_models = [r for r in binary_results if not r.get('is_baseline')]
        if real_models:
            best = max(real_models, key=lambda x: x['f1'])
            print(f"\n  🏆 二分类最佳: {best['model']} (F1={best['f1']:.4f}, Acc={best['accuracy']*100:.2f}%)")

    # 保存
    output_dir = ensure_output_dir('output')
    if binary_results:
        out_df = pd.DataFrame([{
            '模型': r['model'],
            '是否基线': '是' if r.get('is_baseline') else '否',
            '准确率': r['accuracy'],
            '精确率': r['precision'],
            '召回率': r['recall'],
            'F1': r['f1'],
            '宏平均F1': r.get('macro_f1', float('nan'))
        } for r in binary_results])
        out_path = os.path.join(output_dir, 'model_comparison_binary.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n  二分类结果已保存: {out_path}")

    return binary_results


def main():
    parser = argparse.ArgumentParser(description='事故严重程度预测建模')
    parser.add_argument('--sample', type=int, default=50000,
                        help='采样数量（默认50000，0表示全量）')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'xgboost', 'rf', 'lr'],
                        help='要训练的模型（默认全部）')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='测试集比例（默认0.2）')
    args = parser.parse_args()

    sample_size = args.sample if args.sample > 0 else None

    print_section("US Accidents 预测建模分析")
    print(f"\n采样数量: {sample_size:,} 条" if sample_size else "\n全量分析")
    print(f"模型选择: {args.model}")

    usecols = ['Severity', 'Start_Time', 'State', 'Distance(mi)',
               'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
               'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
               'Sunrise_Sunset', 'Civil_Twilight',
               'Description', 'Street'] + FACILITY_COLS

    df = load_data(sample_size=sample_size, usecols=usecols)
    df = preprocess_basic(df)
    df = engineer_features(df)  # 提取 Is_Night / Is_Holiday / 长度特征

    # 内存保护：训练数据上限（避免全量 770 万条时 OOM / 训练过久）
    MAX_TRAIN_ROWS = 1_500_000
    if len(df) > MAX_TRAIN_ROWS:
        print(f"\n  [内存保护] 数据 {len(df):,} 条超过训练上限 {MAX_TRAIN_ROWS:,}，随机降采样")
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    X, y, feature_names = feature_engineering(df)

    print_section("数据划分")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size
    )
    print(f"\n训练集: {len(X_train):,} 条 ({(1-args.test_size)*100:.0f}%)")
    print(f"测试集: {len(X_test):,} 条 ({args.test_size*100:.0f}%)")
    print(f"\n训练集类别分布:")
    train_dist = y_train.value_counts().sort_index()
    for label, count in train_dist.items():
        print(f"  严重程度 {label+1}: {count:,} ({count/len(y_train)*100:.2f}%)")

    results = []

    # 基线模型：始终预测训练集中最多的类别，用于衡量其他模型的真实增益
    baseline_model = train_baseline(X_train, y_train)
    if baseline_model is not None:
        baseline_result = evaluate_model(baseline_model, X_test, y_test, 'Baseline (最多数类)')
        if baseline_result:
            baseline_result['is_baseline'] = True
            results.append(baseline_result)

    if args.model in ['all', 'xgboost']:
        xgb_model = train_xgboost(X_train, y_train, n_classes=y.nunique())
        if xgb_model is not None:
            xgb_result = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
            if xgb_result:
                results.append(xgb_result)
            analyze_feature_importance(xgb_model, X_test, y_test, feature_names, 'XGBoost')

    if args.model in ['all', 'rf']:
        rf_model = train_random_forest(X_train, y_train)
        if rf_model is not None:
            rf_result = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
            if rf_result:
                results.append(rf_result)
            analyze_feature_importance(rf_model, X_test, y_test, feature_names, 'Random Forest')

    if args.model in ['all', 'lr']:
        lr_result_tuple = train_logistic_regression(X_train, y_train)
        if lr_result_tuple is not None:
            lr_model, lr_scaler = lr_result_tuple
            lr_result = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression',
                                       scaler=lr_scaler)
            if lr_result:
                results.append(lr_result)
            analyze_feature_importance(lr_model, X_test, y_test, feature_names,
                                       'Logistic Regression', scaler=lr_scaler)

    if len(results) > 1:
        compare_models(results)

    output_dir = ensure_output_dir('output')
    if results:
        comparison_df = pd.DataFrame([{
            '模型': r['model'],
            '是否基线': '是' if r.get('is_baseline') else '否',
            '准确率': r['accuracy'],
            '加权精确率': r['precision'],
            '加权召回率': r['recall'],
            '加权F1': r['f1'],
            '宏平均F1': r.get('macro_f1', float('nan'))
        } for r in results])
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'),
                             index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {os.path.join(output_dir, 'model_comparison.csv')}")

    # 二分类预测（Severity ≥ 3 vs <3）——同行常用目标定义
    run_binary_classification(df, test_size=args.test_size)

    print_section("建模分析完成")


if __name__ == '__main__':
    main()
