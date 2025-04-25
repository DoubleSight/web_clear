#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
猥亵儿童犯罪行为模式识别模块 - 主应用模块
基于Flask的web应用，提供案例文本的智能处理与行为模式识别功能
"""

import os
import time
import re
import json
import uuid
import shutil
import logging
import traceback
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import docx
import tempfile
import threading
import chardet
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import random
import torch
import transformers

# 确保pythonScript目录在sys.path中，这样可以正确导入模块
parent_dir = os.path.abspath(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 直接导入模块（不使用相对导入）
from pythonScript.text_cleaner import LegalTextCleaner
from pythonScript.text_purifier import TextPurifier
try:
    from pythonScript.text_restructurer import TextRestructurer, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TextRestructurer = None # 定义一个None以便后续检查
    print("警告: text_restructurer模块未能加载，结构重组和标签注入功能不可用。")

# 导入NLP模块 - 先导入模块本身
import pythonScript.NLP as NLP
# 从模块中导入需要的函数和变量
from pythonScript.NLP import (
    USE_LIGHTWEIGHT_MODE,
    NLP_REQUEST_COUNT, 
    NLP_REQUEST_THRESHOLD,
    get_model_status,
    initialize_nlp_models,
    reset_model_state,
    perform_nlp_analysis
)

# 手动设置NLP_INITIALIZED变量（如果导入失败）
try:
    from pythonScript.NLP import NLP_INITIALIZED
except ImportError:
    NLP_INITIALIZED = False
    print("警告: 无法导入NLP_INITIALIZED，使用默认值False")

# >>> 新增：导入特征提取模块
from pythonScript.feature_extractor import FeatureExtractor, CrimeFeatures

# 添加NLP模型的延迟加载控制
USE_NLP = False  # 默认不在启动时加载NLP模型

# 配置常量
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB上传限制
PORT = 5001

# 定义项目根目录及相关文件夹路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'templates')
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')
# 确保上传文件夹也在项目根目录
UPLOAD_FOLDER_PATH = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER)

# 初始化Flask应用 - 明确指定模板和静态文件夹路径
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH # 使用计算出的完整路径
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # 使用配置中的路径

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
    """首页路由 - 显示主界面"""
    try:
        # 判断是否需要展示首页状态信息
        show_stats = request.args.get('stats', 'false').lower() == 'true'
        
        # 如果请求了统计信息，获取NLP模型状态
        model_status = {}
        if show_stats:
            model_status = get_model_status()
        
        # 使用增强版的JS文件
        js_file = 'enhanced_app.js'
        css_file = 'custom.css'
        
        return render_template(
            'index.html', 
            app_version="2.1", 
            model_status=model_status,
            js_file=js_file,
            css_file=css_file
        )
    except Exception as e:
        app.logger.error(f"渲染首页时出错: {str(e)}")
        return render_template('index.html', error=str(e))

# 路由：检查模型状态
@app.route('/check_model_status', methods=['GET'])
def check_model_status():
    """获取NLP模型状态"""
    status = get_model_status()
    
    # 将NLP_INITIALIZED状态添加到返回结果中
    status['is_initialized'] = NLP_INITIALIZED
    
    return jsonify(status)

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
            model_status = get_model_status()
            if not model_status['models_loaded']:
                # 如果模型未加载，启动加载
                if not model_status['loading_in_progress']:
                    threading.Thread(target=initialize_nlp_models).start()
                response_data['nlp_warning'] = '模型尚未加载完成，NLP分析无法执行。请等待模型加载完成后再试。'
            else:
                # 执行NLP分析
                nlp_process_type = request.form.get('nlp_process_type', 'all')
                nlp_results = perform_nlp_analysis(response_data['content'], nlp_process_type)
                
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
    return jsonify(reset_model_state())

# 路由：初始化模型
@app.route('/initialize_models')
def initialize_models():
    """初始化NLP模型的API端点"""
    # 调用初始化函数
    is_initialized = initialize_nlp_models()
    
    # 获取当前状态
    status = get_model_status()
    
    # 添加初始化状态
    status["is_initialized"] = is_initialized
    
    return jsonify({
        "status": "开始初始化模型",
        "is_initialized": is_initialized,
        "model_status": status
    })

# 路由：下载处理后的文件
@app.route('/download')
def download_file():
    global last_processed_result
    if not last_processed_result or 'final_text' not in last_processed_result:
        return "没有可下载的处理结果。", 404
    
    try:
        # 创建临时文件来保存结果
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix='.txt') as temp_file:
            temp_file.write(last_processed_result['final_text'])
            temp_filename = temp_file.name
        
        # 发送文件给用户
        return send_file(temp_filename, 
                         as_attachment=True, 
                         download_name='crime_pattern_analysis.txt')
    except Exception as e:
        print(f"下载文件时出错: {e}")
        return "生成下载文件时出错。", 500
    finally:
        # 确保临时文件被删除
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError as e:
                print(f"删除临时文件时出错 '{temp_filename}': {e}")

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

# 配置日志记录
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler = logging.FileHandler('app.log', encoding='utf-8')
    log_handler.setFormatter(log_formatter)
    app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO)
    # 将根日志记录器也配置一下，以便捕获其他模块的日志
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(logging.INFO)

# 处理文本处理请求
@app.route('/process_text', methods=['POST'])
def process_text():
    """处理上传的文本或文件，并执行选定的预处理步骤"""
    try:
        # 获取处理参数
        processing_level = request.form.get('processing_level', 'format')
        tfidf_threshold = float(request.form.get('tfidf_threshold', 0.35))
        
        # 获取NLP选项
        nlp_summarize = request.form.get('nlp_summarize', 'false').lower() == 'true'
        nlp_keywords = request.form.get('nlp_keywords', 'false').lower() == 'true'
        nlp_sentiment = request.form.get('nlp_sentiment', 'false').lower() == 'true'
        
        # 获取输入文本
        text = ''
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text']
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # 添加安全检查，防止文件名没有扩展名
                try:
                    # 首先检查文件名是否包含点号
                    if '.' in filename:
                        ext = filename.rsplit('.', 1)[1].lower()
                        
                        if ext == 'txt':
                            text = read_text_file(file)
                        elif ext in ['doc', 'docx']:
                            text = read_docx_file(file)
                        else:
                            # 不支持的扩展名作为文本文件处理
                            text = read_text_file(file)
                            app.logger.warning(f"不支持的文件扩展名 {ext}，尝试作为文本文件处理")
                    else:
                        # 如果文件没有扩展名，尝试作为文本文件读取
                        text = read_text_file(file)
                        app.logger.warning(f"文件 {filename} 没有扩展名，尝试作为文本文件处理")
                except Exception as e:
                    # 处理各种可能的异常
                    app.logger.error(f"处理文件 {filename} 时出错: {str(e)}")
                    text = read_text_file(file)  # 作为最后的尝试
        
        if not text:
            return jsonify({'error': '未提供有效的文本或文件'}), 400
            
        # 添加日志
        app.logger.info(f"接收到处理请求，处理深度: {processing_level}，文本长度: {len(text)}字符")
        
        # 创建结果字典
        result = {'original_text': text}
        
        # 1. 格式清洗
        cleaner = LegalTextCleaner()
        formatted_text, clean_details = cleaner.clean_text(text, stats=True)
        result['formatted_text'] = formatted_text
        result['format_stats'] = {
            'noise_removed': clean_details['stats'].get('noise_chars_removed', 0),
            'symbols_detected': clean_details['stats'].get('noise_symbols_count', 0)
        }
        
        current_text = formatted_text
        app.logger.info(f"格式清洗完成，清除了 {result['format_stats']['noise_removed']} 个噪声字符")
        
        # 2. 语义净化
        if processing_level in ['semantic', 'structure', 'feature', 'full']:
            purifier = TextPurifier(tfidf_threshold=tfidf_threshold)
            purified_text, purify_stats = purifier.purify_text(current_text)
            result['purified_text'] = purified_text
            result['semantic_stats'] = {
                'tfidf_threshold': tfidf_threshold,
                'reduction_rate': purify_stats.get('reduction_percentage', 0),
                'hard_filtered': sum([
                    purify_stats.get('hard_filter_stats', {}).get('legal_citations_removed', 0),
                    purify_stats.get('hard_filter_stats', {}).get('procedural_phrases_removed', 0),
                    purify_stats.get('hard_filter_stats', {}).get('filler_words_removed', 0)
                ]),
                'soft_filtered': purify_stats.get('soft_filter_stats', {}).get('sentences_removed', 0)
            }
            
            current_text = purified_text
            app.logger.info(f"语义净化完成，减少了 {result['semantic_stats']['reduction_rate']}% 的文本内容")
        
        # 3. 结构重组
        if processing_level in ['structure', 'feature', 'full'] and TRANSFORMERS_AVAILABLE:
            try:
                restructurer = TextRestructurer(bert_threshold=0.7, use_gpu=False)
                restructure_result = restructurer.restructure_text(current_text)
                
                # 使用前端期望的字段名
                result['structured_text'] = restructure_result['marked_text']
                result['structure_stats'] = {
                    'bert_score': 0.7,
                    'tags_added': len(re.findall(r'\[(\w+)[=]?[^\]]*\]', restructure_result['marked_text']))
                }
                
                current_text = restructure_result['marked_text']
                app.logger.info(f"结构重组完成，添加了 {result['structure_stats']['tags_added']} 个标签")
            except Exception as e:
                app.logger.error(f"结构重组失败: {str(e)}")
                result['structure_error'] = str(e)
        
        # 4. 特征提取
        if processing_level in ['feature', 'full']:
            try:
                # 使用特征提取器处理文本
                feature_extractor = FeatureExtractor()
                extracted_features = feature_extractor.extract_features(current_text)
                
                # 转换为可序列化的结构
                features_dict = {
                    'categories': {}
                }
                
                # 添加分类特征
                for category, items in extracted_features.categories.items():
                    features_dict['categories'][category] = items
                
                # 添加特征向量
                features_dict['vector'] = {
                    name: value for name, value in zip(
                        CrimeFeatures.get_feature_names(), 
                        extracted_features.to_feature_vector()
                    )
                }
                
                result['features'] = features_dict
                app.logger.info("特征提取完成")
            except Exception as e:
                app.logger.error(f"特征提取失败: {str(e)}")
                result['feature_error'] = str(e)
        
        # 5. NLP分析
        if nlp_summarize or nlp_keywords or nlp_sentiment:
            try:
                nlp_options = {
                    'summarize': nlp_summarize,
                    'keywords': nlp_keywords, 
                    'sentiment': nlp_sentiment
                }
                
                # 调用NLP模块进行分析
                nlp_results = perform_nlp_analysis(current_text, 'all', nlp_options)
                if 'error' in nlp_results:
                    # 如果NLP处理出错，但不影响其他功能，记录错误并尝试使用轻量级结果
                    app.logger.warning(f"NLP处理出错: {nlp_results['error']}")
                    result['nlp_error'] = nlp_results['error']
                else:
                    # 将NLP结果字段直接添加到结果中
                    if 'summary' in nlp_results:
                        result['summary'] = nlp_results['summary']
                    if 'keywords' in nlp_results:
                        result['keywords'] = nlp_results['keywords']
                    if 'sentiment' in nlp_results:
                        result['sentiment'] = nlp_results['sentiment']
                    
                    app.logger.info("NLP分析完成")
            except Exception as e:
                app.logger.error(f"NLP处理异常: {str(e)}")
                result['nlp_error'] = str(e)
        
        # 保存处理结果供后续下载
        global last_processed_result
        last_processed_result = result
        
        app.logger.info("文本处理完成，准备返回结果")
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"处理文本时发生错误: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理文本时发生错误: {str(e)}'}), 500

# 主函数
if __name__ == '__main__':
    setup_logging()
    logging.info("Flask 应用启动...")
    # 启动应用，准备好模型
    print("法律文本清洗系统启动中，访问地址: http://localhost:{}".format(PORT))
    # 配置为不在启动时预加载NLP模型
    if USE_NLP:
        print("正在预加载NLP模型...")
        initialize_nlp_models()
    else:
        print("NLP模型将按需加载，系统启动更快速...")
    app.run(host='0.0.0.0', port=PORT, debug=True) 