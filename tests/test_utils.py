"""
单元测试：utils.tools 与 accident_analysis 去重验证
=================================================
运行方式（项目根目录，使用已装好依赖的 .venv）：
    .venv/Scripts/python.exe -m unittest discover -s tests -t . -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path，确保能 import utils / analysis 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.tools import (load_data, preprocess_basic, FACILITY_COLS,
                         ENGLISH_STOPWORDS, SEVERITY_DESC)


class TestPreprocessBasic(unittest.TestCase):
    """验证基础预处理：时间转换、持续时间、时间特征、布尔化。"""

    def test_time_and_bool_transforms(self):
        df = pd.DataFrame({
            'Start_Time': ['2019-05-01 08:30:00', '2020-12-15 17:45:00'],
            'End_Time': ['2019-05-01 09:00:00', '2020-12-15 18:30:00'],
            'Severity': [2, 3],
            'Amenity': [0, 1],
            'Junction': [1, 0],
        })
        out = preprocess_basic(df)

        # 时间字段转换为 datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out['Start_Time']))
        # 持续时间 = (End - Start) 小时
        self.assertAlmostEqual(out['Duration_hours'].iloc[0], 0.5, places=4)
        # 时间特征提取
        self.assertEqual(out['Hour'].iloc[0], 8)
        self.assertEqual(out['Hour'].iloc[1], 17)
        self.assertFalse(out['Is_Weekend'].iloc[0])  # 2019-05-01 周三
        self.assertFalse(out['Is_Weekend'].iloc[1])  # 2020-12-15 周二
        # 布尔字段化
        self.assertIsInstance(out['Amenity'].iloc[0], (bool, np.bool_))
        self.assertTrue(out['Junction'].iloc[0])


class TestLoadData(unittest.TestCase):
    """验证数据加载。"""

    def test_load_small_csv(self):
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                          delete=False, newline='')
        pd.DataFrame({'Severity': [1, 2, 2], 'State': ['CA', 'NY', 'TX']}).to_csv(
            tmp.name, index=False)
        tmp.close()
        try:
            df = load_data(tmp.name, sample_size=2)
            self.assertEqual(len(df), 2)
            self.assertIn('Severity', df.columns)
        finally:
            os.remove(tmp.name)


class TestSharedConstants(unittest.TestCase):
    """验证去重后的共享常量。"""

    def test_facility_cols(self):
        self.assertEqual(len(FACILITY_COLS), 13)
        self.assertIn('Traffic_Signal', FACILITY_COLS)
        self.assertIn('Turning_Loop', FACILITY_COLS)

    def test_english_stopwords(self):
        self.assertIn('the', ENGLISH_STOPWORDS)
        self.assertIn('and', ENGLISH_STOPWORDS)
        self.assertGreater(len(ENGLISH_STOPWORDS), 50)

    def test_severity_desc(self):
        self.assertEqual(SEVERITY_DESC[2], '中等事故')


class TestAccidentAnalysisDedup(unittest.TestCase):
    """P3 去重验证：accident_analysis 应复用 utils.tools，而非自带副本。"""

    def test_load_data_reuses_utils(self):
        import analysis.accident_analysis as aa
        from utils import tools as ut
        self.assertIs(aa.load_data, ut.load_data)


if __name__ == '__main__':
    unittest.main()
