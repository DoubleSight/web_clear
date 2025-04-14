#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本处理主程序，实现完整的三级预处理流程：
1. 格式清洗 - 排除肉眼可见的格式干扰
2. 语义净化 - 基于TF-IDF的双层级过滤机制，提取核心信息
3. 结构重组 - 修复语义中断，标记要素，构建行为链
"""

import os
import argparse
import time
import json
from text_cleaner import LegalTextCleaner
from text_purifier import TextPurifier
from text_restructurer import TextRestructurer

def process_text(input_file, output_file, intermediate_file=None, structured_file=None, 
                clean_only=False, purify_only=False, restructure_only=False, 
                verbose=False, tfidf_threshold=0.35, bert_threshold=0.7, use_gpu=False):
    """
    处理文本的完整流程：格式清洗 -> 语义净化 -> 结构重组
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 最终输出文件路径
        intermediate_file (str, optional): 格式清洗后的中间文件路径
        structured_file (str, optional): 结构化数据输出文件路径
        clean_only (bool): 仅执行格式清洗步骤
        purify_only (bool): 仅执行语义净化步骤
        restructure_only (bool): 仅执行结构重组步骤
        verbose (bool): 是否显示详细处理信息
        tfidf_threshold (float): 语义净化的TF-IDF阈值，值越大过滤越严格
        bert_threshold (float): 结构重组的BERT评分阈值
        use_gpu (bool): 是否使用GPU加速BERT模型
    """
    try:
        start_time = time.time()
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if verbose:
            print(f"读取文件 {input_file}，大小: {len(text)} 字符")
        
        # 处理结果
        final_result = {
            'original_text': text,
            'cleaned_text': None,
            'purified_text': None,
            'structured_data': None,
            'processing_steps': []
        }
        
        # 1. 格式清洗阶段
        if not purify_only and not restructure_only:
            cleaner = LegalTextCleaner()
            if verbose:
                print("\n===== 格式清洗阶段 =====")
                print("应用格式清洗规则...")
            
            cleaned_text, clean_details = cleaner.clean_text(text, stats=True)
            final_result['cleaned_text'] = cleaned_text
            
            # 添加处理步骤信息
            final_result['processing_steps'].append({
                'step': 'format_cleaning',
                'noise_chars_removed': clean_details['stats']['noise_chars_removed'],
                'noise_symbols_count': clean_details['stats']['noise_symbols_count'],
                'rules_applied': [rule['rule_name'] for rule in clean_details['rules_applied'] if rule['changes_made']]
            })
            
            # 输出清洗统计信息
            if verbose:
                stats = clean_details['stats']
                print(f"格式清洗完成: {stats['noise_chars_removed']} 个干扰字符被移除")
                print(f"清洗后文本大小: {len(cleaned_text)} 字符")
            
            # 保存中间结果
            if intermediate_file:
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                if verbose:
                    print(f"格式清洗结果已保存到中间文件: {intermediate_file}")
        else:
            # 直接使用输入文本
            cleaned_text = text
            final_result['cleaned_text'] = cleaned_text
            if verbose:
                print("跳过格式清洗阶段")
        
        # 2. 语义净化阶段
        if not clean_only and not restructure_only:
            if verbose:
                print("\n===== 语义净化阶段 =====")
                print(f"应用语义净化过滤（TF-IDF阈值: {tfidf_threshold}）...")
            
            purifier = TextPurifier(tfidf_threshold=tfidf_threshold)
            purified_text, purify_stats = purifier.purify_text(cleaned_text)
            final_result['purified_text'] = purified_text
            
            # 添加处理步骤信息
            purify_step_info = {
                'step': 'semantic_purification',
                'tfidf_threshold': tfidf_threshold,
                'original_length': purify_stats['original_length'],
                'purified_length': purify_stats['purified_length'],
                'reduction_percentage': purify_stats['reduction_percentage']
            }
            
            if purify_stats['hard_filter_stats']:
                purify_step_info.update({
                    'legal_citations_removed': purify_stats['hard_filter_stats']['legal_citations_removed'],
                    'procedural_phrases_removed': purify_stats['hard_filter_stats']['procedural_phrases_removed'],
                    'filler_words_removed': purify_stats['hard_filter_stats']['filler_words_removed']
                })
            
            if purify_stats['soft_filter_stats']:
                purify_step_info.update({
                    'important_sentences': purify_stats['soft_filter_stats'].get('important_sentences', 0),
                    'sentences_removed': purify_stats['soft_filter_stats'].get('sentences_removed', 0)
                })
            
            final_result['processing_steps'].append(purify_step_info)
            
            # 输出净化统计信息
            if verbose:
                print(f"语义净化完成:")
                if purify_stats['hard_filter_stats']:
                    print(f"- 移除法定条款引导语: {purify_stats['hard_filter_stats']['legal_citations_removed']} 处")
                    print(f"- 移除程序性固定表述: {purify_stats['hard_filter_stats']['procedural_phrases_removed']} 处")
                    print(f"- 移除高频口语填充词: {purify_stats['hard_filter_stats']['filler_words_removed']} 处")
                
                if purify_stats['soft_filter_stats']:
                    print(f"- 保留重要句子: {purify_stats['soft_filter_stats']['important_sentences']} 句")
                    print(f"- 移除冗余句子: {purify_stats['soft_filter_stats']['sentences_removed']} 句")
                
                print(f"净化后文本大小: {purify_stats['purified_length']} 字符")
                print(f"总缩减比例: {purify_stats['reduction_percentage']:.2f}%")
        else:
            # 非语义净化模式
            if clean_only:
                purified_text = cleaned_text
                final_result['purified_text'] = purified_text
            elif restructure_only:
                purified_text = text
                final_result['purified_text'] = purified_text
                
            if verbose:
                print("跳过语义净化阶段")
        
        # 3. 结构重组阶段
        if not clean_only and not purify_only:
            if verbose:
                print("\n===== 结构重组阶段 =====")
                print(f"应用结构重组（BERT阈值: {bert_threshold}）...")
            
            restructurer = TextRestructurer(bert_threshold=bert_threshold, use_gpu=use_gpu)
            restructure_result = restructurer.restructure_text(purified_text)
            
            # 保存结构化数据
            final_result['structured_data'] = {
                'marked_text': restructure_result['marked_text'],
                'elements': restructure_result['structured_data'],
                'action_chains': restructure_result['action_chains'],
                'xml_output': restructure_result['xml_output']
            }
            
            # 添加处理步骤信息
            restructure_step_info = {
                'step': 'structure_reorganization',
                'bert_threshold': bert_threshold,
                'elements_extracted': {}
            }
            
            for element_type, elements in restructure_result['structured_data'].items():
                restructure_step_info['elements_extracted'][element_type] = len(elements)
            
            restructure_step_info['action_chains_detected'] = len(restructure_result['action_chains'])
            final_result['processing_steps'].append(restructure_step_info)
            
            # 输出结构重组统计信息
            if verbose:
                print("结构重组完成:")
                for element_type, elements in restructure_result['structured_data'].items():
                    if elements:
                        print(f"- 提取到的{element_type}: {len(elements)} 项")
                if restructure_result['action_chains']:
                    print(f"- 检测到的行为链: {len(restructure_result['action_chains'])} 个")
                print("\nXML结构化输出示例:")
                print(restructure_result['xml_output'][:500] + '...' if len(restructure_result['xml_output']) > 500 else restructure_result['xml_output'])
                
            # 保存结构化数据
            if structured_file:
                with open(structured_file, 'w', encoding='utf-8') as f:
                    json.dump(restructure_result, f, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"结构化数据已保存到: {structured_file}")
        else:
            if verbose:
                print("跳过结构重组阶段")
        
        # 确定最终输出的文本内容
        output_text = None
        if clean_only:
            output_text = cleaned_text
        elif purify_only:
            output_text = purified_text
        elif restructure_only and 'structured_data' in final_result and final_result['structured_data']:
            output_text = final_result['structured_data']['xml_output']
        else:
            # 完整处理流程，输出标记后的文本
            if 'structured_data' in final_result and final_result['structured_data']:
                output_text = final_result['structured_data']['marked_text']
            else:
                output_text = purified_text
        
        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 添加处理时间信息
        final_result['processing_time'] = processing_time
        
        if verbose:
            print(f"\n处理完成，用时: {processing_time:.2f} 秒")
            print(f"最终结果已保存到: {output_file}")
        else:
            print(f"处理完成，结果已保存到: {output_file}")
        
        return final_result, True
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None, False

def main():
    parser = argparse.ArgumentParser(description='法律文本处理工具 - 格式清洗、语义净化和结构重组的完整流程')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--intermediate', type=str, help='格式清洗后的中间文件路径')
    parser.add_argument('--structured', type=str, help='结构化数据输出文件路径')
    parser.add_argument('--clean-only', action='store_true', help='仅执行格式清洗步骤')
    parser.add_argument('--purify-only', action='store_true', help='仅执行语义净化步骤')
    parser.add_argument('--restructure-only', action='store_true', help='仅执行结构重组步骤')
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    parser.add_argument('--tfidf-threshold', type=float, default=0.35, help='语义净化的TF-IDF阈值，值越大过滤越严格')
    parser.add_argument('--bert-threshold', type=float, default=0.7, help='结构重组的BERT评分阈值')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速BERT模型')
    
    args = parser.parse_args()
    
    # 处理模式冲突检查
    if sum([args.clean_only, args.purify_only, args.restructure_only]) > 1:
        print("错误: --clean-only, --purify-only, --restructure-only 不能同时使用多个")
        return
    
    # 执行处理流程
    process_text(
        args.input, 
        args.output, 
        args.intermediate,
        args.structured,
        args.clean_only, 
        args.purify_only,
        args.restructure_only,
        args.verbose, 
        args.tfidf_threshold,
        args.bert_threshold,
        args.use_gpu
    )

if __name__ == "__main__":
    main() 