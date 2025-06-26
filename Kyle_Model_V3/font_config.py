#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体配置工具
自动检测并配置matplotlib的中文字体支持
"""

import matplotlib
import matplotlib.font_manager as fm
import warnings

def setup_chinese_font():
    """
    自动检测并设置中文字体
    
    Returns:
        str: 使用的字体名称，如果没有找到中文字体则返回None
    """
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体优先级列表（按推荐程度排序）
    chinese_fonts = [
        'SimHei',           # Windows 黑体
        'Heiti TC',         # macOS 黑体繁体
        'STHeiti',          # macOS 华文黑体
        'PingFang SC',      # macOS 苹方简体
        'Microsoft YaHei',  # Windows 微软雅黑
        'Arial Unicode MS', # 通用Unicode字体
        'Hiragino Sans GB', # macOS 冬青黑体
        'WenQuanYi Micro Hei', # Linux 文泉驿微米黑
        'Noto Sans CJK SC', # Google Noto字体
        'Source Han Sans SC' # Adobe 思源黑体
    ]
    
    # 查找第一个可用的中文字体
    font_to_use = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_to_use = font
            break
    
    # 配置matplotlib字体
    if font_to_use:
        matplotlib.rcParams['font.sans-serif'] = [font_to_use, 'DejaVu Sans', 'sans-serif']
        print(f"✅ 使用中文字体: {font_to_use}")
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
        warnings.warn("⚠️  未找到中文字体，图表中的中文可能显示为方框")
        print("💡 建议安装中文字体以获得更好的显示效果")
    
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    return font_to_use

def list_available_chinese_fonts():
    """
    列出系统中所有可用的中文字体
    
    Returns:
        list: 中文字体名称列表
    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 包含中文相关关键词的字体
    chinese_keywords = ['chinese', 'hei', 'song', 'kai', 'fang', 'ming', 'yuan', 'han', 'noto', 'pingfang']
    
    chinese_fonts = []
    for font in available_fonts:
        if any(keyword in font.lower() for keyword in chinese_keywords):
            if font not in chinese_fonts:  # 去重
                chinese_fonts.append(font)
    
    return sorted(chinese_fonts)

def test_chinese_display():
    """
    测试中文字体显示效果
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置字体
    font_used = setup_chinese_font()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 绘制图表
    ax.plot(x, y, label='正弦波')
    ax.set_title('中文字体测试 - Kyle模型可视化')
    ax.set_xlabel('轮次 t')
    ax.set_ylabel('价格 $p_t$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存测试图片
    test_file = 'chinese_font_test.png'
    plt.savefig(test_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 测试图表已保存为: {test_file}")
    if font_used:
        print(f"✅ 中文显示正常，使用字体: {font_used}")
    else:
        print("⚠️  可能存在中文显示问题")
    
    return test_file

if __name__ == "__main__":
    print("=== 中文字体配置工具 ===")
    print("\n1. 检测可用中文字体:")
    fonts = list_available_chinese_fonts()
    for font in fonts:
        print(f"  - {font}")
    
    print("\n2. 配置matplotlib字体:")
    setup_chinese_font()
    
    print("\n3. 测试中文显示:")
    test_chinese_display()