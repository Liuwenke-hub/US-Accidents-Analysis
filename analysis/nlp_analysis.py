"""
NLP文本分析模块
===============
功能：
    1. 文本基本统计
    2. 词频分析（停用词过滤）
    3. 二元词组分析（Bigram）
    4. 事故类型自动识别
    5. 道路信息提取
    6. 词云生成

使用方法：
    python -m analysis.nlp_analysis --sample 50000
    python -m analysis.nlp_analysis --sample 50000 --wordcloud
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

from utils.tools import (print_section, print_subsection, load_data, preprocess_basic,
                         ensure_output_dir, safe_import, ENGLISH_STOPWORDS)


def analyze_text_basic(df):
    """文本基本统计"""
    print_section("文本基本统计")

    if 'Description' not in df.columns:
        print("  缺少 Description 字段")
        return None

    desc = df['Description'].dropna()
    print(f"\n有效描述文本: {len(desc):,} 条")
    print(f"缺失描述: {len(df) - len(desc):,} 条 ({(len(df)-len(desc))/len(df)*100:.2f}%)")

    if len(desc) == 0:
        return None

    lengths = desc.str.len()
    word_counts = desc.str.split().str.len()

    print(f"\n描述长度统计:")
    print(f"  最短: {lengths.min()} 字符")
    print(f"  最长: {lengths.max()} 字符")
    print(f"  平均: {lengths.mean():.1f} 字符")
    print(f"  中位数: {lengths.median()} 字符")

    print(f"\n词数统计:")
    print(f"  最少: {word_counts.min()} 词")
    print(f"  最多: {word_counts.max()} 词")
    print(f"  平均: {word_counts.mean():.1f} 词")
    print(f"  中位数: {word_counts.median()} 词")

    return desc


def analyze_word_frequency(desc, top_n=50):
    """词频分析"""
    print_section(f"词频分析（Top {top_n}）")

    nltk = safe_import('nltk', 'pip install nltk')
    stop_words = ENGLISH_STOPWORDS  # 单一来源，避免重复定义
    if nltk is not None:
        try:
            from nltk.corpus import stopwords
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        except Exception:
            pass  # 拉取失败则回退到内置 ENGLISH_STOPWORDS

    all_words = []
    for text in desc:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)

    word_counts = pd.Series(all_words).value_counts()

    print(f"\n总词数: {len(all_words):,}")
    print(f"唯一词数: {len(word_counts):,}")

    print(f"\n--- Top {top_n} 高频词 ---")
    top_words = word_counts.head(top_n)
    for i, (word, count) in enumerate(top_words.items(), 1):
        bar = '█' * int(count / top_words.max() * 30)
        print(f"  {i:2d}. {word:<15s} {count:>6,} {bar}")

    return word_counts


def analyze_bigrams(desc, top_n=30):
    """Bigram分析"""
    print_section(f"二元词组分析（Top {top_n}）")

    bigrams = []
    for text in desc:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))

    bigram_counts = pd.Series(bigrams).value_counts()

    print(f"\n总二元词组数: {len(bigrams):,}")
    print(f"唯一二元词组数: {len(bigram_counts):,}")

    print(f"\n--- Top {top_n} 二元词组 ---")
    top_bigrams = bigram_counts.head(top_n)
    for i, ((w1, w2), count) in enumerate(top_bigrams.items(), 1):
        print(f"  {i:2d}. {w1} {w2:<15s} {count:>6,}")

    return bigram_counts


def identify_accident_types(desc):
    """事故类型自动识别"""
    print_section("事故类型自动识别")

    accident_patterns = {
        '追尾事故': r'rear[ -]end|rearend|tailgate|hit from behind',
        '侧面碰撞': r'side[ -]swipe|side[ -]impact|t[ -]bone|broadside',
        '正面碰撞': r'head[ -]on|headon|frontal collision',
        '翻车事故': r'roll[ -]over|rolled over|overturned',
        '撞车事故': r'collision|crashed into|hit a',
        '多车事故': r'multi[ -]vehicle|multiple vehicle|chain reaction',
        '单车事故': r'single[ -]vehicle|single car',
        '摩托车事故': r'motorcycle|bike crash',
        '卡车事故': r'truck|semi|18[ -]wheeler',
        '巴士事故': r'bus|school bus',
        '行人事故': r'pedestrian|person walking',
        '自行车事故': r'bicycle|bike rider',
        '打滑事故': r'skidded|slipped on',
        '失控事故': r'lost control|out of control',
        '起火事故': r'fire|caught fire',
        '坠崖事故': r'off the road|cliff|embankment',
        '水淹事故': r'flood|water on road',
        '动物事故': r'animal|deer|wildlife'
    }

    type_counts = {}
    for accident_type, pattern in accident_patterns.items():
        count = desc.str.contains(pattern, case=False, na=False).sum()
        type_counts[accident_type] = count

    type_df = pd.DataFrame({'事故类型': list(type_counts.keys()), '数量': list(type_counts.values())})
    type_df = type_df.sort_values('数量', ascending=False)

    print(f"\n--- 事故类型识别结果 ---")
    print(type_df.to_string(index=False))

    return type_df


def extract_road_info(desc):
    """提取道路信息"""
    print_section("道路信息提取")

    road_patterns = {
        '高速公路': r'highway|freeway|interstate|i[ -]\d+',
        '国道': r'us[ -]\d+|route\s+\d+',
        '州道': r'state[ -]\d+|sr[ -]\d+',
        '县道': r'county[ -]\d+',
        '城市道路': r'city street|local road',
        '桥梁': r'bridge|overpass',
        '隧道': r'tunnel|underpass',
        '环岛': r'roundabout|circle',
        '路口': r'intersection|crossroads',
        '坡道': r'onramp|offramp|exit ramp'
    }

    road_counts = {}
    for road_type, pattern in road_patterns.items():
        count = desc.str.contains(pattern, case=False, na=False).sum()
        road_counts[road_type] = count

    road_df = pd.DataFrame({'道路类型': list(road_counts.keys()), '数量': list(road_counts.values())})
    road_df = road_df.sort_values('数量', ascending=False)

    print(f"\n--- 道路类型分布 ---")
    print(road_df.to_string(index=False))

    return road_df


def analyze_direction(desc):
    """方向分析"""
    print_section("方向分析")

    directions = ['eastbound', 'westbound', 'northbound', 'southbound']
    dir_names = ['东向', '西向', '北向', '南向']

    dir_counts = {}
    for dir_en, dir_cn in zip(directions, dir_names):
        count = desc.str.contains(dir_en, case=False, na=False).sum()
        dir_counts[dir_cn] = count

    dir_df = pd.DataFrame({'方向': list(dir_counts.keys()), '数量': list(dir_counts.values())})
    dir_df = dir_df.sort_values('数量', ascending=False)

    print(f"\n--- 事故方向分布 ---")
    print(dir_df.to_string(index=False))

    return dir_df


def analyze_severity_keywords(df, desc):
    """严重程度关键词分析"""
    print_section("严重程度关键词分析")

    if 'Severity' not in df.columns:
        print("  缺少 Severity 字段")
        return None

    severity_words = {}
    for severity in sorted(df['Severity'].unique()):
        severity_desc = desc[df['Severity'] == severity]
        if len(severity_desc) == 0:
            continue

        words = []
        for text in severity_desc:
            words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        word_counts = pd.Series(words).value_counts().head(20)
        severity_words[severity] = word_counts

    print("\n--- 各严重程度高频词对比（Top 10） ---")
    for severity in sorted(severity_words.keys()):
        print(f"\n严重程度 {severity}:")
        top_words = severity_words[severity].head(10)
        for word, count in top_words.items():
            print(f"  {word:<15s} {count:>5,}")

    return severity_words


def generate_wordcloud(word_counts, output_dir):
    """生成词云"""
    print_section("生成词云")

    wordcloud = safe_import('wordcloud', 'pip install wordcloud')
    if wordcloud is None:
        return

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    print("\n正在生成词云...")
    wc = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=200
    )

    wc.generate_from_frequencies(word_counts.to_dict())

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    output_path = os.path.join(output_dir, 'wordcloud.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"词云已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='事故描述NLP分析')
    parser.add_argument('--sample', type=int, default=50000,
                        help='采样数量（默认50000）')
    parser.add_argument('--wordcloud', action='store_true',
                        help='生成词云图片')
    parser.add_argument('--top-n', type=int, default=50,
                        help='显示Top N高频词')
    args = parser.parse_args()

    sample_size = args.sample if args.sample > 0 else None

    print_section("US Accidents NLP文本分析")
    print(f"\n采样数量: {sample_size:,} 条" if sample_size else "\n全量分析")

    usecols = ['Description', 'Severity', 'Start_Time']
    df = load_data(sample_size=sample_size, usecols=usecols)
    df = preprocess_basic(df)

    # 内存 + 耗时保护：NLP 文本分析上限（Python 循环逐条处理长文本，超过后收益递减且极慢）
    NLP_MAX_ROWS = 500_000
    if len(df) > NLP_MAX_ROWS:
        print(f"\n  [内存保护] NLP 文本分析上限 {NLP_MAX_ROWS:,} 条，随机降采样")
        df = df.sample(n=NLP_MAX_ROWS, random_state=42).reset_index(drop=True)

    desc = analyze_text_basic(df)
    if desc is None:
        return

    word_counts = analyze_word_frequency(desc, top_n=args.top_n)

    analyze_bigrams(desc, top_n=30)

    identify_accident_types(desc)

    extract_road_info(desc)

    analyze_direction(desc)

    analyze_severity_keywords(df, desc)

    output_dir = ensure_output_dir('output')

    if word_counts is not None:
        word_counts.head(100).to_csv(
            os.path.join(output_dir, 'word_frequency.csv'),
            header=['count'], encoding='utf-8-sig'
        )

    if args.wordcloud and word_counts is not None:
        generate_wordcloud(word_counts, output_dir)

    print(f"\n分析结果已保存到: {output_dir}")

    print_section("NLP文本分析完成")


if __name__ == '__main__':
    main()
