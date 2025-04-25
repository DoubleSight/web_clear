#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
法律文本结构重组模块 - 基于BERT的语义修复和要素标记

实现文本预处理.md中描述的"结构重组层"，包括：
1. 语义修复：修复语句中断，使语义连贯
2. 要素标记：检测并标记文本中的时间、地点、人物、行为、方法等关键要素
"""

import os
import re
import json
import jieba
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Union
import logging

# 尝试导入transformers，如果没有安装则给出友好提示
try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers库未安装，语义修复功能将被禁用")
    print("可以通过pip安装: pip install transformers torch")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextRestructurer:
    """
    法律文本结构重组工具，用于对已经格式清洗和语义净化过的文本进行更高层次的结构化处理
    实现"文本预处理.md"中描述的"结构重组层"功能
    """
    
    def __init__(self, bert_threshold=0.7, use_gpu=False):
        """
        初始化结构重组器
        
        Args:
            bert_threshold (float): BERT评分阈值，默认为0.7
            use_gpu (bool): 是否使用GPU加速，默认为False
        """
        self.bert_threshold = bert_threshold
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # 要素标记规则
        self.element_patterns = {
            'time': [
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{4}-\d{1,2}-\d{1,2})',
                r'(\d{2}:\d{2}(:\d{2})?)',
                r'(\d{1,2}时\d{1,2}分(\d{1,2}秒)?)'
            ],
            'location': [
                r'(在|于)?([\u4e00-\u9fa5]{2,}?(?:省|市|县|区|镇|村|路|街|号|楼|室))', 
                r'地点[:：]?\s*([\u4e00-\u9fa5]+[路街道区县市省][\u4e00-\u9fa5\d]+)',
                r'地址[:：]?\s*([\u4e00-\u9fa5]+[路街道区县市省][\u4e00-\u9fa5\d]+)'
            ],
            'person': [
                r'(被告人|被告|原告|证人)?[\u4e00-\u9fa5]{1,2}(?:[某甲乙丙丁]|[A-Z])',
                r'[\u4e00-\u9fa5]{2,3}[（(][职业岗位年龄身份:：].*?[)）]'
            ],
            'action': [
                r'((?:尾随|搂抱|抚摸|触碰|猥亵|侵犯|攻击|威胁|敲诈|勒索|持有|贩卖|运输|制造|走私|贿赂|伪造)(?:[了过着])?)',
                r'((?:盗窃|抢劫|抢夺|诈骗|敲诈|勒索|绑架|非法拘禁|强奸|猥亵)(?:[了过着])?)',
                r'((?:[驾驶行驶][\u4e00-\u9fa5]*?车辆|酒后驾车|无证驾驶|肇事逃逸))'
            ],
            'tool': [
                r'(?:持有|使用)([^，。；,.;]{2,}?(?:刀|枪|棍|棒|武器|工具))',
                r'(?:用|使用|持有)([^，。；,.;]{2,}?(?:刀|枪|棍|棒|武器|工具))'
            ],
            'method': [
                r'(?:通过|采用|利用|以|冒充|假装|伪装)([^，。；,.;]{2,})手段',
                r'(?:冒充|假装|伪装)([^，。；,.;]{2,}?身份|人员)',
                r'(?:假装|谎称)([^，。；,.;]{2,}?理由|借口)'
            ]
        }
        
        # 法律领域特定词汇表
        self.legal_terms = set([
            "无证驾驶", "酒后驾驶", "醉酒驾驶", "盗窃", "抢劫", "抢夺", "诈骗", "敲诈", "勒索",
            "绑架", "非法拘禁", "故意伤害", "故意杀人", "过失致人死亡", "猥亵", "强制猥亵", 
            "性骚扰", "聚众斗殴", "寻衅滋事", "非法持有", "贩卖毒品", "吸毒", "制毒", "运输毒品",
            "贿赂", "受贿", "行贿", "挪用公款", "职务侵占", "贪污", "伪造"
        ])
        
        # 初始化BERT模型
        self.initialize_bert_model()
        
    def initialize_bert_model(self):
        """初始化BERT模型用于语义修复"""
        self.bert_tokenizer = None
        self.bert_model = None
        self.fill_mask_pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # 使用中文BERT预训练模型
                logger.info("正在加载BERT模型用于语义修复...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                
                # 使用掩码语言模型进行填充缺失内容
                self.bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
                self.bert_model.to(self.device)
                self.bert_model.eval()
                
                # 创建fill_mask流水线
                self.fill_mask_pipeline = pipeline(
                    "fill-mask", 
                    model=self.bert_model, 
                    tokenizer=self.bert_tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                
                logger.info("BERT模型加载成功")
            except Exception as e:
                logger.error(f"加载BERT模型时出错: {e}")
                self.bert_tokenizer = None
                self.bert_model = None
    
    def fix_broken_sentences(self, text):
        """
        修复中断的句子，使语义连贯
        
        Args:
            text (str): 包含中断句子的文本
            
        Returns:
            str: 修复后的文本
        """
        if not TRANSFORMERS_AVAILABLE or self.fill_mask_pipeline is None:
            logger.warning("BERT模型未加载，跳过语义修复")
            return text
            
        # 查找需要修复的句子模式：[缺失]
        broken_pattern = r'(\w+)\[缺失\](\w+)'
        matches = re.finditer(broken_pattern, text)
        
        fixed_text = text
        for match in matches:
            broken_sentence = match.group(0)
            context_before = match.group(1)
            context_after = match.group(2)
            
            # 创建用于BERT的masked文本
            masked_text = f"{context_before} [MASK] [MASK] [MASK] {context_after}"
            
            # 使用BERT预测缺失部分
            try:
                # 获取最可能的完成方式
                predictions = self.fill_mask_pipeline(masked_text, top_k=3)
                
                # 获取评分最高的预测
                best_prediction = None
                best_score = 0
                
                if predictions is not None:
                    for prediction_set in predictions:
                        # 可能会有多个[MASK]的情况，所以结果可能是列表的列表
                        if isinstance(prediction_set, list):
                            for p in prediction_set:
                                if isinstance(p, dict) and 'score' in p and p['score'] > best_score:
                                    best_score = p['score']
                                    if 'sequence' in p:
                                        best_prediction = p['sequence']
                        elif isinstance(prediction_set, dict):
                            if 'score' in prediction_set and prediction_set['score'] > best_score:
                                best_score = prediction_set['score']
                                if 'sequence' in prediction_set:
                                    best_prediction = prediction_set['sequence']
                
                # 检查预测分数是否超过阈值
                if best_prediction is not None and best_score >= self.bert_threshold:
                    # 提取填充的文本部分
                    filled_text = best_prediction
                    
                    # 替换原文本中的缺失部分
                    fixed_text = fixed_text.replace(broken_sentence, filled_text)
                    logger.info(f"修复句子: '{broken_sentence}' -> '{filled_text}' (得分: {best_score:.4f})")
                else:
                    # 分数低于阈值，保留原始的[缺失]标记
                    logger.info(f"句子 '{broken_sentence}' 修复置信度不足 ({best_score:.4f})")
            except Exception as e:
                logger.error(f"预测缺失内容时出错: {e}")
        
        return fixed_text

    def mark_text_elements(self, text):
        """
        标记文本中的重要元素（时间、地点、人物、行为等）
        改进：一次性查找所有匹配项，然后统一标记，避免重复标记。
        
        Args:
            text (str): 要处理的文本
            
        Returns:
            tuple: (标记后的文本, 提取到的元素字典)
        """
        marked_text = text
        extracted_elements = {
            'time': [],
            'location': [],
            'person': [],
            'action': [],
            'tool': [],
            'method': []
        }
        all_matches_to_apply = [] # Store (start, end, element_type, element_text)

        # 1. 查找所有类型的匹配项及其原始位置
        for element_type, patterns in self.element_patterns.items():
            for pattern in patterns:
                try:
                    # 在原始文本上查找，避免干扰
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        start, end = match.span()
                        # 尝试获取捕获组，否则获取整个匹配
                        element_text = match.group(1) if match.lastindex and match.lastindex > 0 else match.group(0)
                        # 过滤掉空或过短的匹配（可选）
                        if element_text and len(element_text.strip()) > 0:
                            all_matches_to_apply.append((start, end, element_type, element_text))
                except re.error as e:
                    logger.error(f"Regex error for {element_type} pattern '{pattern}': {e}")

        # 2. 排序并去重/处理重叠（按起始位置排序，如果起始位置相同，选择较长的匹配）
        # 基本的重叠处理：如果一个匹配完全包含在另一个匹配内，保留较长的。
        # 更复杂的重叠（部分重叠）可能需要更精细的逻辑，这里先处理包含关系。
        if not all_matches_to_apply:
            return text, extracted_elements

        all_matches_to_apply.sort(key=lambda x: (x[0], -(x[1] - x[0]))) # Sort by start, then by length descending

        final_matches = []
        last_end = -1
        for i, (start, end, etype, etext) in enumerate(all_matches_to_apply):
            # 简单的去包含和精确重叠处理
            # 如果当前匹配的结束位置 <= 上一个接受的匹配的结束位置，说明被包含或重叠，跳过
            if start >= last_end: 
                final_matches.append((start, end, etype, etext))
                last_end = end
            # （可选）更复杂的重叠判断逻辑可以在这里添加

        # 3. 从后往前插入标记，避免偏移量计算复杂化
        marked_text_builder = list(text) # 使用列表方便插入
        offset = 0 # 相对原始文本的偏移量 - 在从后往前插入时不需要
        processed_indices = set() # 防止对同一文本区域重复标记

        for start, end, element_type, element_text in reversed(final_matches):
             # 检查是否已经处理过这个范围（或部分重叠）- 简单检查
             is_processed = False
             for i in range(start, end):
                 if i in processed_indices:
                     is_processed = True
                     break
             if is_processed:
                 logger.debug(f"Skipping overlapping/processed tag for {element_type} at {start}-{end}")
                 continue
            
             element_tag = f"[{element_type.upper()}]"
             element_end_tag = f"[/{element_type.upper()}]"
             
             # 执行插入
             marked_text_builder.insert(end, element_end_tag)
             marked_text_builder.insert(start, element_tag)
             
             # 标记已处理的原始索引
             for i in range(start, end):
                  processed_indices.add(i)

             # 保存提取到的元素 (去重)
             element_text_clean = element_text.strip()
             if element_text_clean not in extracted_elements[element_type]:
                 extracted_elements[element_type].append(element_text_clean)
        
        marked_text = "".join(marked_text_builder)
        return marked_text, extracted_elements
    
    def detect_action_chains(self, text, extracted_elements=None):
        """
        检测并标记文本中的行为链
        
        Args:
            text (str): 要处理的文本
            extracted_elements (dict, optional): 已提取的元素字典
            
        Returns:
            tuple: (标记了行为链的文本, 提取到的行为链列表)
        """
        if extracted_elements is None or 'action' not in extracted_elements:
            # 如果没有提取过元素，先提取
            _, extracted_elements = self.mark_text_elements(text)
        
        # 没有检测到行为，无法构建行为链
        if not extracted_elements['action']:
            return text, []
        
        actions = extracted_elements['action']
        
        # 在文本中寻找行为之间的关系词（如：后、接着、随后、然后、之后等）
        chain_patterns = [
            r'({})[^，。；,.;]*?(?:后|接着|随后|然后|之后|续)[^，。；,.;]*?({})',
            r'({})[^，。；,.;]*?(?:先|首先|开始|起初)[^，。；,.;]*?(?:(?:接着|继而|然后|之后|随即|紧接着)[^，。；,.;]*?)??({})'
        ]
        
        action_chains = []
        marked_text = text
        
        # 遍历所有可能的行为对组合
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i != j:  # 避免自己跟自己组合
                    for pattern_template in chain_patterns:
                        pattern = pattern_template.format(re.escape(action1), re.escape(action2))
                        matches = re.finditer(pattern, text)
                        
                        for match in matches:
                            chain_text = match.group(0)
                            
                            # 提取并构建行为链
                            action_chain = f"{action1}→{action2}"
                            
                            if action_chain not in action_chains:
                                action_chains.append(action_chain)
                                
                                # 在文本中标记行为链
                                start_pos = match.start()
                                end_pos = match.end()
                                chain_tag = f"[ACTION_CHAIN]{action_chain}[/ACTION_CHAIN]"
                                
                                # 将原始文本中的匹配部分替换为带标记的版本
                                marked_text = marked_text[:start_pos] + chain_tag + marked_text[end_pos:]
        
        return marked_text, action_chains
    
    def restructure_text(self, text):
        """
        对文本进行完整的结构重组处理
        
        Args:
            text (str): 要处理的文本
            
        Returns:
            dict: 包含处理结果的字典
        """
        result = {
            'original_text': text,
            'fixed_text': '',
            'marked_text': '',
            'structured_data': {},
            'action_chains': [],
            'xml_output': ''
        }
        
        try:
            # 1. 修复中断的句子
            fixed_text = self.fix_broken_sentences(text)
            result['fixed_text'] = fixed_text
            
            # 2. 标记并提取文本元素
            marked_text, extracted_elements = self.mark_text_elements(fixed_text)
            result['marked_text'] = marked_text
            result['structured_data'] = extracted_elements
            
            # 3. 检测行为链
            _, action_chains = self.detect_action_chains(fixed_text, extracted_elements)
            result['action_chains'] = action_chains
            
            # 4. 生成XML结构化输出
            xml_output = self.generate_xml_output(extracted_elements, action_chains)
            result['xml_output'] = xml_output
            
            return result
        
        except Exception as e:
            logger.error(f"结构重组过程中出错: {e}")
            result['error'] = str(e)
            return result
    
    def generate_xml_output(self, elements, action_chains=None):
        """
        根据提取到的元素生成XML结构化输出
        
        Args:
            elements (dict): 提取到的元素字典
            action_chains (list, optional): 行为链列表
            
        Returns:
            str: XML格式的输出
        """
        xml_parts = ['<案件>']
        
        # 添加时间元素
        for time_element in elements.get('time', []):
            # 尝试标准化时间格式
            try:
                standardized_time = self.standardize_time(time_element)
                xml_parts.append(f'  <TIME>{standardized_time}</TIME>')
            except:
                xml_parts.append(f'  <TIME>{time_element}</TIME>')
        
        # 添加地点元素
        for location in elements.get('location', []):
            xml_parts.append(f'  <LOC>{location}</LOC>')
        
        # 添加人物元素
        for person in elements.get('person', []):
            xml_parts.append(f'  <PER>{person}</PER>')
        
        # 添加行为元素
        for action in elements.get('action', []):
            xml_parts.append(f'  <ACTION>{action}</ACTION>')
        
        # 添加工具元素
        for tool in elements.get('tool', []):
            xml_parts.append(f'  <TOOL>{tool}</TOOL>')
        
        # 添加方法元素
        for method in elements.get('method', []):
            xml_parts.append(f'  <METHOD>{method}</METHOD>')
        
        # 添加行为链
        if action_chains:
            for chain in action_chains:
                xml_parts.append(f'  <ACTION_CHAIN>{chain}</ACTION_CHAIN>')
        
        xml_parts.append('</案件>')
        
        return '\n'.join(xml_parts)
    
    def standardize_time(self, time_str):
        """
        将各种时间格式标准化为ISO8601格式
        
        Args:
            time_str (str): 原始时间字符串
            
        Returns:
            str: 标准化后的时间字符串
        """
        time_str = time_str.strip()
        
        # 处理"2023年05月07日"格式
        year_month_day_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', time_str)
        if year_month_day_match:
            year, month, day = year_month_day_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        # 处理"19时38分00秒"格式
        hour_min_sec_match = re.search(r'(\d{1,2})时(\d{1,2})分(?:(\d{1,2})秒)?', time_str)
        if hour_min_sec_match:
            groups = hour_min_sec_match.groups()
            hour, minute = groups[0], groups[1]
            second = groups[2] if groups[2] else "00"
            return f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        
        # 处理其他格式或返回原始字符串
        return time_str

def process_file(input_file, output_file, bert_threshold=0.7, use_gpu=False, verbose=False):
    """
    处理文件的便捷函数
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        bert_threshold (float): BERT置信度阈值
        use_gpu (bool): 是否使用GPU
        verbose (bool): 是否显示详细信息
        
    Returns:
        bool: 处理是否成功
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        restructurer = TextRestructurer(bert_threshold=bert_threshold, use_gpu=use_gpu)
        result = restructurer.restructure_text(text)
        
        # 保存处理结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"文本结构重组完成，已保存到: {output_file}")
            print(f"提取到的结构化信息:")
            for element_type, elements in result['structured_data'].items():
                if elements:
                    print(f"- {element_type}: {', '.join(elements)}")
            if result['action_chains']:
                print(f"- 行为链: {', '.join(result['action_chains'])}")
            print("\nXML输出:")
            print(result['xml_output'])
        
        return True
    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='法律文本结构重组工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.7, help='BERT置信度阈值，默认为0.7')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    
    args = parser.parse_args()
    
    process_file(
        args.input,
        args.output,
        bert_threshold=args.threshold,
        use_gpu=args.use_gpu,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 