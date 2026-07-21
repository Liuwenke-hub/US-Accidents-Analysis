"""
单元测试：predictive_modeling 的评估/基线/特征工程
=================================================
运行方式（项目根目录，使用已装好依赖的 .venv）：
    .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.tools import preprocess_basic
from analysis.predictive_modeling import (train_baseline, evaluate_model,
                                          feature_engineering)


class TestTrainBaseline(unittest.TestCase):
    """基线模型应始终预测训练集中的多数类。"""

    def test_predicts_majority_class(self):
        X = pd.DataFrame({'Hour': [8, 17, 9, 18, 8],
                          'Temperature(F)': [60, 70, 65, 72, 61]})
        y = pd.Series([1, 1, 1, 0, 2])  # 多数类 = 1（对应 Severity 2）
        model = train_baseline(X, y)
        self.assertIsNotNone(model)
        preds = model.predict(X)
        self.assertTrue(np.all(preds == 1))


class TestEvaluateModel(unittest.TestCase):
    """evaluate_model 应返回准确率/加权F1/宏平均F1 等必要字段。"""

    def test_returns_required_keys(self):
        X = pd.DataFrame({'Hour': [8, 17, 9, 18],
                          'Temperature(F)': [60, 70, 65, 72]})
        y = pd.Series([1, 1, 0, 2])
        model = train_baseline(X, y)
        result = evaluate_model(model, X, y, 'Baseline')
        for k in ['accuracy', 'precision', 'recall', 'f1',
                  'macro_f1', 'macro_precision', 'macro_recall']:
            self.assertIn(k, result)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)


class TestFeatureEngineering(unittest.TestCase):
    """特征工程应产出特征矩阵、目标变量与特征名列表。"""

    def test_produces_features_and_target(self):
        df = pd.DataFrame({
            'Start_Time': ['2019-05-01 08:30:00', '2020-12-15 17:45:00',
                           '2019-06-01 12:00:00'],
            'End_Time': ['2019-05-01 09:30:00', '2020-12-15 18:30:00',
                         '2019-06-01 12:40:00'],
            'Severity': [2, 3, 2],
            'State': ['CA', 'NY', 'TX'],
            'Temperature(F)': [60.0, 70.0, 65.0],
            'Humidity(%)': [50.0, 55.0, 52.0],
            'Amenity': [0, 1, 0],
            'Junction': [1, 0, 1],
            'Distance(mi)': [0.5, 1.2, 0.3],
            'Sunrise_Sunset': ['Day', 'Day', 'Night'],
        })
        df = preprocess_basic(df)  # 补全 Hour / Is_Weekend 等时间特征
        X, y, feats = feature_engineering(df)

        self.assertIn('Hour', feats)
        self.assertIn('State_encoded', feats)
        self.assertIn('Amenity', feats)
        self.assertEqual(len(X), 3)
        self.assertTrue(set(y.unique()).issubset({0, 1, 2, 3}))


if __name__ == '__main__':
    unittest.main()
