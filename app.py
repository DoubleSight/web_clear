from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
import re
from werkzeug.utils import secure_filename
import time
import docx
import tempfile
import threading
import shutil
import json
from urllib.parse import unquote
import logging
import chardet
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import random
import datetime

# 导入NLP模块
import NLP

# 导入三级预处理相关模块
from text_cleaner import LegalTextCleaner
from text_purifier import TextPurifier
try:
    from text_restructurer import TextRestructurer, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: text_restructurer模块未能加载，可能缺少依赖库")

# 配置常量
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB上传限制
PORT = 5001

# 初始化Flask应用
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量存储最后处理的结果
last_processed_result = {}

# 检查允许的文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 读取文本文件内容
def read_text_file(file_or_path):
    """
    读取文本文件内容 - 兼容文件对象和文件路径
    
    Args:
        file_or_path: 文件对象或文件路径
        
    Returns:
        str: 文件内容
    """
    try:
        # 如果是文件对象
        if hasattr(file_or_path, 'read'):
            content = file_or_path.read()
            # 尝试解码如果是二进制数据
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return content
        # 如果是文件路径
        elif isinstance(file_or_path, str) and os.path.exists(file_or_path):
            with open(file_or_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            raise ValueError("不支持的文件类型: {}".format(type(file_or_path)))
    except Exception as e:
        print("读取文件错误: {}".format(e))
        return ""

# 读取Word文档内容
def read_docx_file(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# 清理法律文本 - 整合三个清洗模块的功能
def clean_legal_text(text, processing_mode='format'):
    """
    清理文本内容 - 实现完整的三级预处理
    
    Args:
        text (str): 要清理的文本
        processing_mode (str): 处理模式，可以是 'format'、'semantic'、'structure' 或 'full'
    
    Returns:
        dict: 包含清理结果的字典
    """
    result = {'original_text': text}
    
    # 获取处理参数
    tfidf_threshold = float(request.form.get('tfidf_threshold', 0.35))
    
    # 1. 格式清洗阶段 - 适用于所有模式
    cleaner = LegalTextCleaner()
    formatted_text, clean_details = cleaner.clean_text(text, stats=True)
    result['formatted_text'] = formatted_text
    result['format_stats'] = {
        'noise_chars_removed': clean_details['stats']['noise_chars_removed'],
        'noise_symbols_count': clean_details['stats']['noise_symbols_count']
    }
    
    # 如果只进行格式清洗则返回
    if processing_mode == 'format':
        result['cleaned_text'] = formatted_text
        return result
        
    # 2. 语义净化阶段 - 用于semantic和full模式
    if processing_mode in ['semantic', 'structure', 'full']:
        purifier = TextPurifier(tfidf_threshold=tfidf_threshold)
        purified_text, purify_stats = purifier.purify_text(formatted_text)
        result['purified_text'] = purified_text
        result['semantic_stats'] = {
            'tfidf_threshold': tfidf_threshold,
            'reduction_percentage': purify_stats['reduction_percentage']
        }
        
        # 保存硬过滤器统计信息
        if purify_stats['hard_filter_stats']:
            result['semantic_stats'].update({
                'legal_citations_removed': purify_stats['hard_filter_stats']['legal_citations_removed'],
                'procedural_phrases_removed': purify_stats['hard_filter_stats']['procedural_phrases_removed'],
                'filler_words_removed': purify_stats['hard_filter_stats']['filler_words_removed']
            })
        
        # 保存软过滤器统计信息
        if purify_stats['soft_filter_stats']:
            result['semantic_stats'].update({
                'important_sentences': purify_stats['soft_filter_stats'].get('important_sentences', 0),
                'sentences_removed': purify_stats['soft_filter_stats'].get('sentences_removed', 0)
            })
            
        # 如果只进行语义净化则返回
        if processing_mode == 'semantic':
            result['cleaned_text'] = purified_text
            return result
    
    # 3. 结构重组阶段 - 用于structure和full模式
    if processing_mode in ['structure', 'full'] and TRANSFORMERS_AVAILABLE:
        try:
            # 创建结构重组器
            restructurer = TextRestructurer(bert_threshold=0.7, use_gpu=False)
            
            # 获取输入文本 - 使用语义净化后的文本
            input_text = purified_text if 'purified_text' in result else formatted_text
            
            # 执行结构重组处理
            restructure_result = restructurer.restructure_text(input_text)
            
            # 保存结构化数据
            result['structured_data'] = restructure_result['structured_data']
            result['marked_text'] = restructure_result['marked_text']
            result['action_chains'] = restructure_result['action_chains']
            result['xml_output'] = restructure_result['xml_output']
            
            # 设置最终清理结果为标记后的文本
            result['cleaned_text'] = restructure_result['marked_text']
            
        except Exception as e:
            # 结构重组失败，退回到语义净化结果
            print("结构重组处理失败: {}".format(e))
            result['structure_error'] = str(e)
            result['cleaned_text'] = purified_text if 'purified_text' in result else formatted_text
    else:
        # 没有使用结构重组，或者不可用
        if processing_mode in ['structure', 'full'] and not TRANSFORMERS_AVAILABLE:
            result['structure_error'] = "结构重组功能不可用，缺少所需的transformers库"
        
        # 使用语义净化或格式清洗的结果作为最终输出
        result['cleaned_text'] = purified_text if 'purified_text' in result else formatted_text
    
    return result

# 路由：首页
@app.route('/', methods=['GET', 'POST'])
def index():
    """主页路由"""
    global last_processed_result
    
    # 检查模型状态
    model_status = NLP.get_model_status()
    
    # 如果模型未加载且未在加载中，尝试初始化模型
    if not model_status['models_loaded'] and not model_status['loading_in_progress'] and model_status['loading_error'] is None:
        NLP.initialize_nlp_models()
        model_status = NLP.get_model_status()
    
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('index.html', 
                                   error="未选择任何文件", 
                                   model_status=model_status)
        
        file = request.files['file']
        
        # 如果用户未选择文件，浏览器也会提交一个空文件
        if file.filename == '':
            return render_template('index.html', 
                                   error="未选择任何文件", 
                                   model_status=model_status)
        
        if file and allowed_file(file.filename):
            try:
                # 保存上传的文件
                filename = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filename)
                
                # 读取文档内容
                if filename.lower().endswith(('.doc', '.docx')):
                    text = read_docx_file(filename)
                else:  # 文本文件
                    text = read_text_file(filename)
                
                # 获取处理模式
                processing_mode = request.form.get('processing_mode', 'format')
                
                # 处理文本
                result = clean_legal_text(text, processing_mode)
                
                # 获取NLP处理选项
                nlp_options = {
                    'summarize': 'nlp_summarize' in request.form,
                    'keywords': 'nlp_keywords' in request.form,
                    'sentiment': 'nlp_sentiment' in request.form
                }
                
                # 如果启用了NLP选项，处理文本
                if any(nlp_options.values()):
                    # 使用已格式化的文本进行NLP处理
                    nlp_results = NLP.perform_nlp_analysis(result.get('formatted_text', text), 
                                                         'all' if all(nlp_options.values()) else
                                                         'summarize' if nlp_options['summarize'] else
                                                         'keywords' if nlp_options['keywords'] else
                                                         'sentiment')
                    result.update(nlp_results)
                
                # 存储结果
                last_processed_result = result
                
                # 渲染结果页面
                return render_template('index.html', 
                                      result=result, 
                                      filename=file.filename, 
                                      processing_mode=processing_mode,
                                      nlp_options=nlp_options,
                                      model_status=model_status)
            except Exception as e:
                return render_template('index.html',
                                      error="处理文件时出错: {}".format(str(e)),
                                      model_status=model_status)
    
    # GET请求，显示空表单
    return render_template('index.html', model_status=model_status)

# 路由：检查模型状态
@app.route('/check_model_status', methods=['GET'])
def check_model_status():
    """获取NLP模型状态"""
    return jsonify(NLP.get_model_status())

# 路由：文件上传和处理
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件类型，请上传 {} 格式的文件'.format(", ".join(ALLOWED_EXTENSIONS))}), 400
    
    try:
        # 读取文件内容
        if file.filename.endswith('.txt'):
            content = read_text_file(file)
        elif file.filename.endswith(('.docx', '.doc')):
            content = read_docx_file(file)
        else:
            return jsonify({'error': '不支持的文件类型'}), 400
        
        # 获取处理模式
        process_mode = request.form.get('process_mode', 'full')
        
        # 获取TF-IDF阈值
        tfidf_threshold = float(request.form.get('tfidf_threshold', 0.1))
        
        # 处理文本 - 使用clean_legal_text获取结果
        result = clean_legal_text(content, process_mode)
        
        # 准备响应数据 - 使用cleaned_text作为返回的内容
        response_data = {
            'content': result.get('cleaned_text', content)
        }
        
        # 检查是否需要NLP分析
        use_nlp = request.form.get('use_nlp') == 'true'
        
        if use_nlp:
            # 检查模型是否已加载
            model_status = NLP.get_model_status()
            if not model_status['models_loaded']:
                # 如果模型未加载，启动加载
                if not model_status['loading_in_progress']:
                    threading.Thread(target=NLP.initialize_nlp_models).start()
                response_data['nlp_warning'] = '模型尚未加载完成，NLP分析无法执行。请等待模型加载完成后再试。'
            else:
                # 执行NLP分析
                nlp_process_type = request.form.get('nlp_process_type', 'all')
                nlp_results = NLP.perform_nlp_analysis(response_data['content'], nlp_process_type)
                
                # 检查是否有错误
                if 'error' in nlp_results:
                    response_data['nlp_warning'] = nlp_results['error']
                else:
                    response_data['nlp_results'] = nlp_results
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print("处理文件时出错: {}\n{}".format(str(e), traceback_str))
        return jsonify({'error': '处理文件时出错: {}'.format(str(e))}), 500

# 路由：重置模型状态
@app.route('/reset_model_state')
def reset_model_state():
    """重置NLP模型状态的API端点"""
    return jsonify(NLP.reset_model_state())

# 路由：初始化模型
@app.route('/initialize_models')
def initialize_models():
    """初始化NLP模型的API端点"""
    NLP.initialize_nlp_models()
    return jsonify({"status": "开始初始化模型"})

# 路由：下载处理后的文件
@app.route('/download/<filename>')
def download_file(filename):
    """下载处理后的文件"""
    global last_processed_result
    
    if not last_processed_result:
        return redirect(url_for('index'))
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp:
        # 确定要保存的文本内容
        if 'cleaned_text' in last_processed_result:
            content = last_processed_result['cleaned_text']
        elif 'formatted_text' in last_processed_result:
            content = last_processed_result['formatted_text']
        else:
            content = last_processed_result.get('original_text', '')
        
        # 添加NLP结果
        if 'summary' in last_processed_result:
            content = "【文本摘要】\n{}\n\n【正文内容】\n{}".format(last_processed_result['summary'], content)
        
        if 'keywords' in last_processed_result and isinstance(last_processed_result['keywords'], list):
            keywords_str = '，'.join(last_processed_result['keywords'])
            content = "【关键词】\n{}\n\n{}".format(keywords_str, content)
        
        if 'sentiment' in last_processed_result:
            sentiment = last_processed_result['sentiment']
            sentiment_label = sentiment['label']
            sentiment_score = sentiment['score']
            
            # 增强情感分析结果的描述
            sentiment_description = ""
            if sentiment_label == '积极':
                if sentiment_score > 0.8:
                    sentiment_description = "文本表达了明显的积极情绪"
                elif sentiment_score > 0.6:
                    sentiment_description = "文本倾向于表达积极情绪"
                else:
                    sentiment_description = "文本包含轻微的积极情绪"
            else:
                if sentiment_score > 0.8:
                    sentiment_description = "文本表达了明显的消极情绪"
                elif sentiment_score > 0.6:
                    sentiment_description = "文本倾向于表达消极情绪"
                else:
                    sentiment_description = "文本包含轻微的消极情绪"
            
            sentiment_text = "情感倾向: {} (置信度: {:.2f})\n情感分析: {}".format(sentiment_label, sentiment_score, sentiment_description)
            content = "【情感分析】\n{}\n\n{}".format(sentiment_text, content)
        
        temp.write(content)
        temp_path = temp.name
    
    # 生成下载文件名
    base_name = os.path.splitext(filename)[0]
    download_name = "{}_processed.txt".format(base_name)
    
    # 发送文件
    return send_file(temp_path, as_attachment=True, 
                     download_name=download_name,
                     mimetype='text/plain')

# 路由：查看NLP日志
@app.route('/logs')
def view_logs():
    """查看NLP日志的路由"""
    try:
        log_file = os.path.join(NLP.MODELS_CACHE_DIR, "nlp_log.txt")
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()
                # 只显示最后100行
                logs = logs[-100:] if len(logs) > 100 else logs
                return render_template('logs.html', logs=logs)
        else:
            return render_template('logs.html', error="日志文件不存在")
    except Exception as e:
        return render_template('logs.html', error="读取日志出错: {}".format(str(e)))

# 主函数
if __name__ == '__main__':
    # 启动应用，准备好模型
    print("法律文本清洗系统启动中，访问地址: http://localhost:{}".format(PORT))
    # 预加载NLP模型
    NLP.initialize_nlp_models()
    app.run(debug=True, port=PORT) 