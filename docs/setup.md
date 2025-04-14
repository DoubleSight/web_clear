# 安装与设置指南

本文档提供安装和设置法律文本清洗系统的详细步骤，特别是新增的NLP功能。

## 基本安装

1. 克隆或下载代码库

2. 创建并激活虚拟环境（推荐）：
   ```bash
   # 使用venv
   python -m venv venv
   
   # 在Windows上激活
   venv\Scripts\activate
   
   # 在Linux/Mac上激活
   source venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   
   注意：这将安装所有必要的包，包括用于NLP功能的`transformers`、`torch`等。第一次安装可能需要几分钟。

## NLP模型设置

系统在第一次启动时会自动下载以下预训练模型：

1. 文本摘要模型 (Helsinki-NLP/opus-mt-zh-en)
2. 关键词提取模型 (ckiplab/bert-base-chinese-ner)
3. 情感分析模型 (uer/roberta-base-finetuned-jd-binary-chinese)

请确保：
- 网络连接良好
- 有足够的磁盘空间（约2GB）
- 有足够的内存（建议8GB+）

## GPU加速设置（可选但推荐）

如果您的系统有NVIDIA GPU，可以设置GPU加速以显著提高NLP处理速度：

1. 确保已安装NVIDIA驱动程序
2. 安装与您的CUDA版本兼容的PyTorch：
   ```bash
   # 例如，对于CUDA 11.7
   pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. 验证GPU设置：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应返回True
   print(torch.cuda.device_count())  # 应返回可用GPU数量
   ```

## 系统配置

您可以在`app.py`文件中修改以下配置：

1. 端口设置（默认为5001）：
   ```python
   app.run(debug=True, port=5001)
   ```

2. 修改上传文件大小限制（默认为16MB）：
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
   ```

3. 自定义NLP模型（高级用户）：
   如果您想使用其他预训练模型，可以修改`initialize_nlp_models`函数中的模型加载部分。

## 故障排除

1. **模型下载失败**：
   - 确保网络连接稳定
   - 手动下载模型并放置在默认缓存目录（通常是`~/.cache/huggingface/transformers/`）

2. **内存不足错误**：
   - 减小处理的文件大小
   - 关闭其他内存密集型应用
   - 考虑增加系统内存或使用带有更多RAM的服务器

3. **CUDA相关错误**：
   - 确保PyTorch版本与CUDA版本兼容
   - 更新NVIDIA驱动程序
   - 在环境变量中设置`CUDA_VISIBLE_DEVICES`以指定使用哪个GPU

## 部署建议

对于生产环境，建议使用以下设置：

1. 使用Gunicorn或uWSGI替代Flask内置服务器：
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5001 app:app
   ```

2. 使用Nginx作为反向代理
3. 设置适当的超时时间，因为NLP处理可能需要较长时间
4. 考虑使用Docker容器化应用程序，以简化依赖管理 