# 法律文本清洗系统

这是一个基于Web的法律文本清洗系统，可以对法律文档进行格式清洗和语义净化处理。

## 功能特点

- 支持上传TXT、DOC、DOCX格式的文件
- 自动进行格式清洗（去除噪声、标准化格式）
- 智能语义净化（去除冗余信息、保留关键内容）
- 实时显示处理结果
- 支持下载处理后的文件
- **新增：** 基于深度学习的NLP高级功能（文本摘要、关键词提取、情感分析）

## 目录结构

```
web_cleaner/
├── app.py              # Flask应用主程序
├── requirements.txt    # Python依赖包列表
├── README.md          # 项目说明文档
├── templates/         # HTML模板目录
│   └── index.html     # 主页面模板
├── js/               # JavaScript文件目录
└── uploads/          # 文件上传临时目录
```

## 安装说明

1. 确保已安装Python 3.7或更高版本
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. **NLP功能说明**：首次启动时，系统会自动下载所需的预训练模型，这可能需要一些时间，具体取决于网络速度。

## 使用方法

1. 启动Web服务器：
   ```bash
   python app.py
   ```

2. 在浏览器中访问：
   ```
   http://localhost:5001
   ```

3. 在网页界面上传需要处理的文件
4. 等待处理完成
5. 查看处理结果并下载

## 系统要求

- Python 3.7+
- 现代浏览器（Chrome、Firefox、Safari等）
- 足够的磁盘空间用于临时文件处理
- 至少4GB内存（使用NLP功能时建议8GB或以上）
- **GPU支持**：NLP功能可以在CPU上运行，但如果有NVIDIA GPU和CUDA环境，处理速度会大幅提升

## 注意事项

- 上传文件大小限制为16MB
- 处理大文件可能需要较长时间
- 建议定期清理uploads目录中的临时文件
- 首次使用NLP功能时会下载模型（约1-2GB），请确保网络连接良好

## 文件格式支持说明

系统支持以下文件格式的处理：

### TXT文件
- 直接进行处理，不需要额外转换
- 支持UTF-8编码的文本文件

### DOCX文件
- 使用python-docx库提取文本内容
- 支持Word 2007及以上版本的文档
- 可以处理文档中的段落文本

### DOC文件
- 尝试使用多种方法提取文本：
  1. 首先尝试使用antiword工具（如果安装）
  2. 其次尝试使用textract库
  3. 如果以上方法都失败，将给出提示
- 建议将DOC文件转换为DOCX或TXT格式以获得更好的处理效果

**注意**：DOC格式文件（Word 97-2003）的处理可能不如DOCX格式完美，因为DOC是二进制格式，提取文本可能会有一定的限制。

## 处理模式说明

系统提供三种处理模式，可以根据需要选择：

### 完整处理（默认）
- 同时进行格式清洗和语义净化
- 先清理格式噪声，再去除冗余内容
- 适合需要全面处理的法律文本

### 仅格式清洗
- 只进行格式噪声清理
- 保留全部文本内容，不进行语义筛选
- 适合格式混乱但内容需要完整保留的文本

### 仅语义净化
- 直接进行语义层面的冗余内容筛选
- 不处理格式噪声，只关注内容精简
- 适合格式已经规范但内容冗长的文本

可以在上传界面中选择相应的处理模式，系统会根据选择进行不同的处理。

### 语义净化强度调节

当选择"完整处理"或"仅语义净化"模式时，系统提供了一个TF-IDF阈值滑块，允许用户调整语义净化的强度：

- **低阈值（0.1-0.3）**：保留较多内容，仅过滤极低价值信息
- **中等阈值（0.35-0.6）**：平衡过滤，去除大部分冗余内容同时保留核心信息
- **高阈值（0.65-0.9）**：严格过滤，仅保留极高价值信息

TF-IDF（词频-逆文档频率）是一种评估词语对文档重要性的统计方法。阈值越高，系统对文本的过滤越严格，保留的内容越少。默认值设置为0.35，通常能够在信息保留和冗余过滤之间取得较好平衡。

## NLP高级功能说明

系统集成了基于Hugging Face Transformers的三种NLP功能，可以对处理后的文本进行更深入的分析：

### 文本摘要
- 使用神经网络模型自动生成文本摘要
- 适合快速了解长文档内容
- 使用Helsinki-NLP中文翻译模型优化处理

### 关键词提取
- 自动识别并提取文本中的关键词和实体
- 使用CKIP Lab中文命名实体识别模型
- 可用于文档标签生成或关键信息提取

### 情感分析
- 分析文本情感倾向（积极/消极）
- 提供情感得分百分比
- 使用优化的中文RoBERTa模型

你可以在处理文件时选择使用所有功能，或仅选择其中一项功能。NLP模型首次加载时需要下载，这可能需要一些时间，请耐心等待。

## GitHub部署说明

### 克隆仓库
```bash
git clone https://github.com/你的用户名/legal-text-cleaner.git
cd legal-text-cleaner
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
```bash
python app.py
```

## 贡献指南

1. Fork 本仓库
2. 创建新的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情