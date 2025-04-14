#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律文本清洗系统 - NLP模块
负责NLP模型的加载、缓存和处理功能
"""

import os
import time
import json
import threading
import traceback
import re
from pathlib import Path
import logging
from typing import Dict, Optional, Union
from transformers import pipeline

# 创建模型缓存目录
MODELS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
MODELS_INFO_FILE = os.path.join(MODELS_CACHE_DIR, 'models_info.json')
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# NLP模型
summarizer = None
keyword_extractor = None
sentiment_analyzer = None

# 模型加载状态
models_loading = False
models_loaded = False
loading_error = None

# 详细日志记录
LOG_FILE = os.path.join(MODELS_CACHE_DIR, "nlp_log.txt")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def log_message(message):
    """记录日志消息到文件"""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = "[{}] {}\n".format(timestamp, message)
        print(log_entry.strip())
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print("写入日志出错: {}".format(e))

def save_model_info():
    """保存模型加载状态到JSON文件"""
    model_info = {
        'models_loaded': models_loaded,
        'loading_in_progress': models_loading,
        'loading_error': loading_error,
        'last_update': time.time()
    }
    
    try:
        with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        log_message("模型状态已保存: {}".format(model_info))
    except Exception as e:
        log_message("保存模型状态出错: {}".format(e))

def load_model_info():
    """从JSON文件加载模型状态"""
    global models_loaded, models_loading, loading_error
    
    if os.path.exists(MODELS_INFO_FILE):
        try:
            with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                
            models_loaded = model_info.get('models_loaded', False)
            models_loading = model_info.get('loading_in_progress', False)
            loading_error = model_info.get('loading_error', None)
            
            # 如果上次更新时间超过1小时且仍在加载中，重置状态防止卡死
            last_update = model_info.get('last_update', 0)
            if time.time() - last_update > 3600 and models_loading:
                log_message("检测到加载状态超时，重置加载标志")
                models_loading = False
                loading_error = "模型加载超时，状态已重置"
                save_model_info()
                
            log_message("已从文件加载模型状态: loaded={}, loading={}".format(models_loaded, models_loading))
        except Exception as e:
            log_message("加载模型状态文件出错: {}".format(e))
            # 重置状态
            models_loaded = False
            models_loading = False
            loading_error = str(e)
            save_model_info()
    else:
        log_message("模型状态文件不存在，将创建新文件")
        save_model_info()

def initialize_nlp_models():
    """初始化NLP模型，让pipeline处理下载和缓存"""
    global summarizer, keyword_extractor, sentiment_analyzer
    global models_loading, models_loaded, loading_error
    
    # 如果模型已加载，直接返回
    if models_loaded and (summarizer is not None) and (keyword_extractor is not None) and (sentiment_analyzer is not None):
        log_message("模型已加载，跳过初始化")
        return True
    
    # 如果正在加载中，直接返回
    if models_loading:
        log_message("模型加载中，跳过重复初始化")
        return False
    
    models_loading = True
    loading_error = None
    save_model_info() # Save state indicating loading has started
    
    # 后台线程加载模型
    def load_models_thread():
        global summarizer, keyword_extractor, sentiment_analyzer
        global models_loading, models_loaded, loading_error
        
        try:
            start_time = time.time()
            log_message("开始加载NLP模型...")

            # 加载文本摘要模型
            log_message("加载文本摘要模型 (facebook/bart-large-cnn)...")
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", cache_dir=MODELS_CACHE_DIR)
            log_message("文本摘要模型加载成功")
            
            # 加载关键词提取模型 (使用feature-extraction)
            log_message("加载关键词提取模型 (bert-base-chinese)...")
            # 注意: feature-extraction本身不是为关键词提取设计的最佳方案，但遵循原代码逻辑
            keyword_extractor = pipeline("feature-extraction", model="bert-base-chinese", cache_dir=MODELS_CACHE_DIR)
            log_message("关键词提取模型加载成功")
            
            # 加载情感分析模型
            log_message("加载情感分析模型 (uer/roberta-base-finetuned-jd-binary-chinese)...")
            sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese", cache_dir=MODELS_CACHE_DIR)
            log_message("情感分析模型加载成功")
            
            # 更新模型状态
            models_loaded = True
            loading_error = None
            end_time = time.time()
            log_message("所有NLP模型加载完成! 总耗时: {:.2f} 秒".format(end_time - start_time))

        except Exception as e:
            # 记录加载错误
            error_trace = traceback.format_exc()
            loading_error = "{}\\nTraceback:\n{}".format(str(e), error_trace)
            log_message("模型加载失败: {}".format(loading_error))
            
            # 重置模型变量
            summarizer = None
            keyword_extractor = None
            sentiment_analyzer = None
            models_loaded = False
        finally:
            # 无论是否成功，都结束加载状态
            models_loading = False
            save_model_info() # Save final state (loaded or error)
    
    # 启动后台线程加载模型
    thread = threading.Thread(target=load_models_thread)
    thread.daemon = True
    thread.start()
    log_message("已启动后台线程加载模型")
    
    return False # Return False because loading happens in background

def get_model_status():
    """获取模型加载状态"""
    # Reload status from file in case another process updated it?
    # Or rely on global variables updated by the loading thread.
    # Let's rely on global vars for now for simplicity.
    status = {
        'models_loaded': models_loaded,
        'loading_in_progress': models_loading,
        'loading_error': loading_error,
        'cache_dir': MODELS_CACHE_DIR
    }
    
    # 可以在这里加一个检查，如果标记为loaded但模型对象为None，说明可能出错了
    if models_loaded and (summarizer is None or keyword_extractor is None or sentiment_analyzer is None):
        status['models_loaded'] = False
        status['loading_error'] = status.get('loading_error', "") + "模型状态不一致 (标记为已加载但对象为None)，建议重置或重启。"
        log_message(status['loading_error'])
        # Optionally trigger reset or re-init?
    
    return status

def perform_nlp_analysis(text, nlp_process_type='all'):
    """
    执行NLP分析
    
    Args:
        text (str): 要分析的文本
        nlp_process_type (str): 分析类型，可选值为 'all', 'summarize', 'keywords', 'sentiment'
        
    Returns:
        dict: 分析结果
    """
    results = {}
    
    # 确保文本不为空
    if not text or len(text.strip()) == 0:
        log_message("NLP分析输入文本为空")
        return {
            "error": "输入文本为空，无法进行NLP分析"
        }
    
    # 检查模型是否已加载且可用
    current_status = get_model_status()
    if not current_status['models_loaded']:
        log_message("尝试执行NLP分析，但模型尚未加载")
        if current_status['loading_in_progress']:
            return {"error": "NLP模型仍在加载中，请稍后再试"}
        elif current_status['loading_error']:
             return {"error": "NLP模型加载失败，无法执行分析: {}".format(current_status['loading_error'])}
        else:
             # 模型未加载，未在加载中，也没有错误 -> 尝试触发加载
             log_message("模型未加载，尝试初始化")
             initialize_nlp_models() # Trigger background load
             return {"error": "NLP模型尚未初始化，已开始在后台加载，请稍后重试"}

    # 检查所需模型对象是否存在
    if (nlp_process_type in ['all', 'summarize'] and summarizer is None) or \
       (nlp_process_type in ['all', 'keywords'] and keyword_extractor is None) or \
       (nlp_process_type in ['all', 'sentiment'] and sentiment_analyzer is None):
        log_message("模型标记为已加载，但所需分析器对象为None，可能存在问题")
        return {"error": "NLP模型状态异常，请尝试重置模型或重启应用"}

    
    try:
        log_message("执行NLP分析，类型: {}, 文本长度: {}".format(nlp_process_type, len(text)))
        
        # 文本摘要
        if nlp_process_type in ['all', 'summarize']:
            try:
                # BART 模型对输入长度有限制，过长可能导致错误或效果不佳
                # 截断输入文本以适应模型限制，但保留足够上下文
                # 注意：max_length 是输出长度限制，输入长度限制通常更大但依赖模型配置
                # 经验值，例如 1024 或 4096 tokens，这里简单截断字符
                input_text_summary = text[:2048] # Limit input size for summarizer
                log_message("生成文本摘要...")
                start_summary = time.time()
                summary = summarizer(input_text_summary, max_length=150, min_length=30, do_sample=False)
                results['summary'] = summary[0]['summary_text']
                log_message("摘要生成成功，耗时: {:.2f}s, 长度: {}".format(time.time() - start_summary, len(results['summary'])))
            except Exception as e:
                error_trace = traceback.format_exc()
                log_message("摘要生成失败: {}\n{}".format(e, error_trace))
                results['summary_error'] = str(e)
        
        # 关键词提取 (保持原逻辑，但效果可能有限)
        if nlp_process_type in ['all', 'keywords']:
            try:
                # 特征提取模型通常也有长度限制 (e.g., 512 tokens for BERT)
                input_text_keywords = text[:500] # Limit input size
                # 使用与加载模型匹配的分词器进行分词，替代简单的正则
                log_message("提取关键词 (基于词频，使用BERT分词器)...")
                start_keywords = time.time()
                
                # 从 pipeline 获取分词器
                tokenizer = keyword_extractor.tokenizer
                # 使用分词器进行分词
                tokens = tokenizer.tokenize(input_text_keywords)
                
                # 基于分词结果进行词频统计
                word_freq = {}
                for token in tokens:
                    # 过滤掉特殊标记、过短的词、纯数字等
                    if token not in tokenizer.all_special_tokens and len(token) > 1 and not token.isdigit():
                        # 移除可能的子词标记 (对中文BERT可能不是##，但作为通用处理)
                        cleaned_token = token.replace('##', '') 
                        if cleaned_token:
                            word_freq[cleaned_token] = word_freq.get(cleaned_token, 0) + 1
                
                # 选择前N个最频繁的词
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                results['keywords'] = [word for word, _ in keywords]
                
                log_message("关键词提取成功 (基于词频，使用BERT分词器)，耗时: {:.2f}s, 数量: {}".format(time.time() - start_keywords, len(results['keywords'])))
            except Exception as e:
                error_trace = traceback.format_exc()
                log_message("关键词提取失败: {}\n{}".format(e, error_trace))
                results['keywords_error'] = str(e)
        
        # 情感分析
        if nlp_process_type in ['all', 'sentiment']:
            try:
                # 情感分析模型通常也有长度限制 (e.g., 512 tokens)
                input_text_sentiment = text[:510] # Limit input size slightly below 512 tokens
                log_message("执行情感分析...")
                start_sentiment = time.time()
                sentiment = sentiment_analyzer(input_text_sentiment)
                # Map label for consistency if needed
                label_map = {'positive': '积极', 'negative': '消极'}
                results['sentiment'] = {
                    'label': label_map.get(sentiment[0]['label'].lower(), sentiment[0]['label']), # Handle potential case variations
                    'score': sentiment[0]['score']
                }
                log_message("情感分析成功，耗时: {:.2f}s, 结果: {}".format(time.time() - start_sentiment, results['sentiment']))
            except Exception as e:
                error_trace = traceback.format_exc()
                log_message("情感分析失败: {}\n{}".format(e, error_trace))
                results['sentiment_error'] = str(e)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = "NLP分析过程中意外出错: {}\n{}".format(str(e), error_trace)
        log_message(error_msg)
        results['error'] = error_msg # General error if something outside specific blocks failed
    
    return results

def reset_model_state():
    """重置模型状态，适用于模型加载异常后的恢复"""
    global models_loaded, models_loading, loading_error
    global summarizer, keyword_extractor, sentiment_analyzer
    
    log_message("接收到重置模型状态请求")
    
    # 重置模型变量
    summarizer = None
    keyword_extractor = None
    sentiment_analyzer = None
    
    # 重置状态标志
    models_loaded = False
    models_loading = False
    loading_error = None
    
    # 保存状态
    save_model_info()
    log_message("模型状态已重置")
    
    return {"status": "已重置模型状态"}

# 程序启动时尝试加载模型 (if run as main script)
if __name__ == "__main__":
    log_message("NLP模块作为主脚本启动，开始预加载模型...")
    
    # 加载上次状态
    load_model_info()
    
    # 尝试初始化模型 (将在后台进行)
    initialize_nlp_models()
    
    # 可以在这里添加一个等待循环，仅用于测试目的
    print("模型正在后台加载... 主线程可以继续执行其他任务或等待。")
    # Example wait loop (optional, app.py handles the check)
    # while models_loading:
    #     print("模型加载中，请稍候...")
    #     time.sleep(5)
    # 
    # if models_loaded:
    #     log_message("模型加载完成")
    # else:
    #     log_message("模型加载遇到问题: {}".format(loading_error))
    #     print("请检查日志文件 {} 以获取详细信息".format(LOG_FILE))
    print("NLP模块主脚本执行完毕 (模型可能仍在后台加载)")
