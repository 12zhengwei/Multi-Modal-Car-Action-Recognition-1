# PowerPoint Presentation Generator

## 概述 (Overview)

本目录包含用于生成"金陵购车节政企通补贴平台合作方案"PowerPoint演示文稿的脚本。

This directory contains a script to generate the "Jinling Car Purchase Festival Government-Enterprise Platform Cooperation Plan" PowerPoint presentation.

## 文件 (Files)

- `generate_presentation.py` - Python脚本，用于生成完整的PowerPoint演示文稿
- `金陵购车节政企通补贴平台合作方案.pptx` - 生成的PowerPoint文件（11页）

## 依赖 (Dependencies)

需要安装以下Python库：

```bash
pip install python-pptx
```

此库的依赖项包括：
- Pillow >= 3.3.2
- XlsxWriter >= 0.5.7
- lxml >= 3.1.0
- typing-extensions >= 4.9.0

## 使用方法 (Usage)

### 生成演示文稿 (Generate Presentation)

```bash
python generate_presentation.py
```

这将在当前目录下生成 `金陵购车节政企通补贴平台合作方案.pptx` 文件。

### 自定义输出路径 (Custom Output Path)

如果需要自定义输出路径，可以修改脚本中的 `main()` 函数：

```python
def main():
    generator = SubsidyPlatformPresentation()
    output_file = generator.generate('custom_path/presentation.pptx')
```

## 演示文稿内容 (Presentation Contents)

生成的PowerPoint包含11页幻灯片：

1. **封面页** - 项目标题、副标题、合作方Logo和日期
2. **项目背景与机遇分析** - 政府目标、项目概述、银行角色、机遇洞察
3. **目标客群精准定位** - 核心目标人群和两大客群画像
4. **成功案例借鉴** - 建行、邮储银行等案例启示
5. **总体方案设计** - 一站式"政企通"平台总体架构
6. **核心流程设计** - 5分钟极速申请的用户流程
7. **C端核心** - 补贴申请小程序设计
8. **B端核心** - 高效智能的审核平台设计
9. **业务增长策略** - 精准引流信贷业务和提升用户活跃度
10. **项目价值与总结** - 对政府、市民、银行的价值
11. **封底页** - 感谢页面和联系信息

## 设计特点 (Design Features)

### 配色方案 (Color Scheme)
- 招商银行红色 (CMB Red): RGB(204, 0, 0)
- 招商银行金色 (CMB Gold): RGB(255, 215, 0)
- 深灰色 (Dark Gray): RGB(64, 64, 64)
- 浅灰色 (Light Gray): RGB(192, 192, 192)

### 字体大小 (Font Sizes)
- 主标题：36-48pt
- 副标题：28-34pt
- 正文标题：20-22pt
- 正文内容：13-16pt

### 页面布局 (Layout)
- 幻灯片尺寸：10英寸 × 7.5英寸
- 内容区域：适当的边距和间距
- 层次结构：清晰的标题、要点和子要点

## 代码结构 (Code Structure)

`SubsidyPlatformPresentation` 类包含以下方法：

- `__init__()` - 初始化演示文稿和配色方案
- `add_title_slide()` - 添加封面页
- `add_background_slide()` - 添加项目背景页
- `add_target_customers_slide()` - 添加目标客群页
- `add_case_studies_slide()` - 添加案例借鉴页
- `add_overall_solution_slide()` - 添加总体方案页
- `add_user_flow_slide()` - 添加用户流程页
- `add_c_end_design_slide()` - 添加C端设计页
- `add_b_end_design_slide()` - 添加B端设计页
- `add_growth_strategy_slide()` - 添加增长策略页
- `add_value_summary_slide()` - 添加价值总结页
- `add_thank_you_slide()` - 添加感谢页
- `generate()` - 生成完整的演示文稿

## 验证演示文稿 (Verify Presentation)

要验证生成的PowerPoint文件，可以运行以下命令：

```python
from pptx import Presentation

prs = Presentation('金陵购车节政企通补贴平台合作方案.pptx')
print(f"Total slides: {len(prs.slides)}")

for i, slide in enumerate(prs.slides, 1):
    print(f"\nSlide {i}:")
    for shape in slide.shapes:
        if hasattr(shape, "text"):
            print(f"  {shape.text[:50]}...")
```

## 扩展和自定义 (Extension and Customization)

要修改或扩展演示文稿：

1. **修改内容** - 编辑各个 `add_*_slide()` 方法中的文本
2. **调整样式** - 修改 `__init__()` 中的配色方案或字体大小
3. **添加新页** - 创建新的 `add_custom_slide()` 方法并在 `generate()` 中调用
4. **插入图片** - 使用 `slide.shapes.add_picture()` 方法

示例：添加图片

```python
from pptx.util import Inches

img_path = 'path/to/image.png'
left = Inches(1)
top = Inches(2)
slide.shapes.add_picture(img_path, left, top, height=Inches(3))
```

## 技术要求 (Technical Requirements)

- Python 3.8+
- python-pptx 1.0.0+
- 支持中文字符编码

## 许可 (License)

本脚本作为项目的一部分，遵循项目的MIT许可证。

## 联系方式 (Contact)

如有问题或建议，请通过GitHub Issues联系。
