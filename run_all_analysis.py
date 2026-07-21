"""
综合分析主入口
==============
一键运行所有分析模块

使用方法：
    python run_all_analysis.py
    python run_all_analysis.py --sample 100000
    python run_all_analysis.py --modules basic,predictive,nlp
"""

import argparse
import subprocess
import sys
import time


def run_module(module_name, sample_size):
    """运行单个分析模块"""
    print(f"\n{'='*80}")
    print(f"  运行 {module_name} 模块")
    print(f"{'='*80}")

    start_time = time.time()

    module_map = {
        'basic': ('analysis.accident_analysis', ['--sample', str(sample_size)]),
        'predictive': ('analysis.predictive_modeling', ['--sample', str(sample_size), '--model', 'all']),
        'spatial': ('analysis.spatial_clustering', ['--sample', str(sample_size), '--method', 'both']),
        'timeseries': ('analysis.time_series_forecasting', ['--sample', '0', '--freq', 'D', '--horizon', '30']),
        'nlp': ('analysis.nlp_analysis', ['--sample', str(sample_size)])
    }

    if module_name not in module_map:
        print(f"  未知模块: {module_name}")
        return False

    module_path, args = module_map[module_name]

    try:
        result = subprocess.run(
            [sys.executable, '-m', module_path] + args,
            capture_output=True,
            text=True,
            timeout=3600  # 全量数据（770万条）需要更长超时
        )

        if result.stdout:
            print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.stderr:
            print(f"stderr:\n{result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr}")

        elapsed = time.time() - start_time
        print(f"\n  模块 {module_name} 运行完成，耗时: {elapsed:.2f} 秒")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"  模块 {module_name} 运行超时")
        return False
    except Exception as e:
        print(f"  模块 {module_name} 运行失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='一键运行所有分析模块')
    parser.add_argument('--sample', type=int, default=50000,
                        help='采样数量（默认50000）')
    parser.add_argument('--full', action='store_true',
                        help='全量分析（忽略--sample参数）')
    parser.add_argument('--modules', type=str, default='all',
                        help='指定运行的模块，逗号分隔：basic,predictive,spatial,timeseries,nlp')
    args = parser.parse_args()

    sample_size = 0 if args.full else args.sample

    print("=" * 80)
    print("  US Accidents 综合分析工具")
    print("=" * 80)
    print(f"\n采样数量: {'全量' if sample_size == 0 else f'{sample_size:,} 条'}")

    if args.modules == 'all':
        modules = ['basic', 'predictive', 'spatial', 'timeseries', 'nlp']
    else:
        modules = [m.strip() for m in args.modules.split(',')]

    print(f"运行模块: {', '.join(modules)}")

    start_total = time.time()
    results = {}

    for module in modules:
        results[module] = run_module(module, sample_size)

    end_total = time.time()
    total_elapsed = end_total - start_total

    print("\n" + "=" * 80)
    print("  综合分析完成")
    print("=" * 80)

    print("\n运行结果汇总:")
    for module, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {module}: {status}")

    print(f"\n总耗时: {total_elapsed:.2f} 秒")

    print("\n下一步建议:")
    print("  1. 查看 output/ 目录下的分析结果")
    print("  2. 启动 Streamlit 仪表板进行交互式探索")
    print("     streamlit run visualization/dashboard.py")
    print("  3. 根据分析结果调整参数进行深度分析")


if __name__ == '__main__':
    main()
