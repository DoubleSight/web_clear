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
import logging
from pythonScript.text_cleaner import LegalTextCleaner
from pythonScript.text_purifier import TextPurifier
from pythonScript.text_restructurer import TextRestructurer
from pythonScript.feature_extractor import FeatureExtractor, FeatureExtractionEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_text(input_file, output_file, intermediate_file=None, structured_file=None, features_file=None,
                clean_only=False, purify_only=False, restructure_only=False, extract_only=False,
                verbose=False, tfidf_threshold=0.35, bert_threshold=0.7, use_gpu=False, use_xml=True,
                evaluate_features=False, ground_truth_file=None, use_nlp=True):
    """
    处理文本的完整流程：格式清洗 -> 语义净化 -> 结构重组 -> 特征提取
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 最终输出文件路径
        intermediate_file (str, optional): 格式清洗后的中间文件路径
        structured_file (str, optional): 结构化数据输出文件路径
        features_file (str, optional): 特征提取结果输出文件路径
        clean_only (bool): 仅执行格式清洗步骤
        purify_only (bool): 仅执行语义净化步骤
        restructure_only (bool): 仅执行结构重组步骤
        extract_only (bool): 仅执行特征提取步骤
        verbose (bool): 是否显示详细处理信息
        tfidf_threshold (float): 语义净化的TF-IDF阈值，值越大过滤越严格
        bert_threshold (float): 结构重组的BERT评分阈值
        use_gpu (bool): 是否使用GPU加速BERT模型
        use_xml (bool): 是否使用XML中间格式进行文本处理
        evaluate_features (bool): 是否评估特征提取质量
        ground_truth_file (str): 人工标注的标准答案文件
        use_nlp (bool): 是否使用NLP增强功能
    """
    try:
        start_time = time.time()
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if verbose:
            print(f"读取文件 {input_file}，大小: {len(text)} 字符")
        
        # 创建结果目录
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if verbose:
                print(f"创建输出目录: {output_dir}")
        
        # 处理结果
        final_result = {
            'original_text': text,
            'cleaned_text': None,
            'purified_text': None,
            'structured_data': None,
            'features': None,
            'evaluation': None,
            'processing_steps': []
        }

        # 1. 格式清洗阶段
        if not purify_only and not restructure_only and not extract_only:
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
            if verbose and not extract_only:
                print("跳过格式清洗阶段")
        
        # 2. 语义净化阶段
        if not clean_only and not restructure_only and not extract_only:
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
                purified_text = cleaned_text
                final_result['purified_text'] = purified_text
            elif extract_only:
                purified_text = cleaned_text 
                final_result['purified_text'] = purified_text
                
            if verbose and not extract_only:
                print("跳过语义净化阶段")
        
        # 3. 结构重组阶段
        if not clean_only and not purify_only:
            if verbose and not extract_only:
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
            if verbose and not extract_only:
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
                if verbose and not extract_only:
                    print(f"结构化数据已保存到: {structured_file}")
        else:
            # 对于extract_only模式，需要生成或加载XML
            if extract_only and use_xml:
                # 尝试加载已有的XML文件
                if structured_file and os.path.exists(structured_file):
                    try:
                        with open(structured_file, 'r', encoding='utf-8') as f:
                            restructure_result = json.load(f)
                            final_result['structured_data'] = {
                                'marked_text': restructure_result.get('marked_text', ''),
                                'elements': restructure_result.get('structured_data', {}),
                                'action_chains': restructure_result.get('action_chains', []),
                                'xml_output': restructure_result.get('xml_output', '')
                            }
                            if verbose:
                                print(f"已加载结构化数据文件: {structured_file}")
                    except Exception as e:
                        logger.error(f"加载结构化数据文件时出错: {e}")
                        # 如果加载失败，自动生成XML
                        restructurer = TextRestructurer(bert_threshold=bert_threshold, use_gpu=use_gpu)
                        restructure_result = restructurer.restructure_text(purified_text)
                        final_result['structured_data'] = {
                            'marked_text': restructure_result['marked_text'],
                            'elements': restructure_result['structured_data'],
                            'action_chains': restructure_result['action_chains'],
                            'xml_output': restructure_result['xml_output']
                        }
                        if verbose:
                            print("加载结构化数据失败，已自动生成XML数据")
                else:
                    # 自动生成XML
                    restructurer = TextRestructurer(bert_threshold=bert_threshold, use_gpu=use_gpu)
                    restructure_result = restructurer.restructure_text(purified_text)
                    final_result['structured_data'] = {
                        'marked_text': restructure_result['marked_text'],
                        'elements': restructure_result['structured_data'],
                        'action_chains': restructure_result['action_chains'],
                        'xml_output': restructure_result['xml_output']
                    }
                    if verbose:
                        print("已自动生成XML数据用于特征提取")
                        
                    # 保存结构化数据
                    if structured_file:
                        with open(structured_file, 'w', encoding='utf-8') as f:
                            json.dump(restructure_result, f, ensure_ascii=False, indent=2)
                        if verbose:
                            print(f"结构化数据已保存到: {structured_file}")
                
        # 4. 特征提取阶段
        if not clean_only and not purify_only and not restructure_only:
            if verbose:
                print("\n===== 特征提取阶段 =====")
                if use_xml:
                    print("从XML中提取案件特征...")
                else:
                    print("从文本中提取案件特征...")
                if use_nlp:
                    print("NLP增强功能已启用")
            
            extractor = FeatureExtractor(use_nlp=use_nlp)
            
            # 根据模式选择提取源
            if use_xml and 'structured_data' in final_result and final_result['structured_data']:
                # 使用XML输出进行特征提取
                xml_output = final_result['structured_data']['xml_output']
                features = extractor.extract_features(xml_output, is_xml=True)
            else:
                # 直接从文本提取
                features = extractor.extract_features(purified_text, is_xml=False)
            
            # 保存特征数据
            final_result['features'] = {
                'extracted_features': vars(features),
                'feature_vector': features.to_feature_vector(),
                'feature_names': features.get_feature_names()
            }
            
            # 添加处理步骤信息
            feature_step_info = {
                'step': 'feature_extraction',
                'features_count': len([f for f in vars(features) if f not in ['CATEGORY_MAPS'] and vars(features)[f] is not None]),
                'vector_dimension': len(features.to_feature_vector()),
                'use_nlp': use_nlp,
                'use_xml': use_xml
            }
            final_result['processing_steps'].append(feature_step_info)
            
            # 输出特征提取统计信息
            if verbose:
                print("特征提取完成:")
                print(f"- 提取到的特征数量: {feature_step_info['features_count']}")
                print(f"- 特征向量维度: {feature_step_info['vector_dimension']}")
                print("\n特征摘要:")
                # 输出主要特征
                feature_categories = {
                    "被害人信息": ['victim_age', 'victim_gender', 'victim_injury_severity', 'victim_relationship_with_perpetrator', 
                              'victim_physical_disability', 'victim_intellectual_disability', 'victim_multiple_victims_involved'],
                    "犯罪人信息": ['perpetrator_age', 'perpetrator_gender', 'perpetrator_occupation_category', 
                              'perpetrator_has_prior_record', 'perpetrator_motive', 'perpetrator_name'],
                    "犯罪行为过程": ['crime_time_of_day', 'crime_location_category', 'crime_duration', 'crime_watched_pornography'],
                    "犯罪行为方式": ['crime_violated_body_part_category', 'crime_perpetrator_exposed_genitals', 
                               'crime_forced_interaction', 'crime_spread_obscene_info_online', 'crime_recorded_on_site',
                               'crime_control_method', 'crime_repeated_offenses', 'crime_multiple_perpetrators']
                }
                
                for category, feat_list in feature_categories.items():
                    print(f"\n{category}:")
                    for feat in feat_list:
                        value = getattr(features, feat, None)
                        if value is not None:
                            print(f"- {feat}: {value}")
            
            # 保存特征数据
            if features_file:
                with open(features_file, 'w', encoding='utf-8') as f:
                    json.dump(final_result['features'], f, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"特征数据已保存到: {features_file}")
                    
            # 5. 特征评估阶段
            if evaluate_features:
                if verbose:
                    print("\n===== 特征评估阶段 =====")
                
                ground_truth = None
                if ground_truth_file and os.path.exists(ground_truth_file):
                    try:
                        with open(ground_truth_file, 'r', encoding='utf-8') as f:
                            ground_truth = json.load(f)
                        if verbose:
                            print(f"已加载标准答案文件: {ground_truth_file}")
                    except Exception as e:
                        logger.error(f"加载标准答案文件时出错: {e}")
                
                if ground_truth:
                    evaluator = FeatureExtractionEvaluator(ground_truth)
                    eval_results = evaluator.evaluate(features)
                    
                    final_result['evaluation'] = eval_results
                    
                    if verbose:
                        formatted_results = evaluator.format_results()
                        print(formatted_results)
                        
                        # 保存评估结果
                        eval_file = os.path.splitext(features_file)[0] + '.eval.json' if features_file else None
                        if eval_file:
                            with open(eval_file, 'w', encoding='utf-8') as f:
                                json.dump(eval_results, f, ensure_ascii=False, indent=2)
                            print(f"评估结果已保存到: {eval_file}")
                else:
                    if verbose:
                        print("未提供标准答案或加载失败，无法评估特征提取质量")
        else:
            if verbose:
                print("跳过特征提取阶段")
        
        # 确定最终输出的文本内容
        output_text = None
        if clean_only:
            output_text = cleaned_text
        elif purify_only:
            output_text = purified_text
        elif restructure_only and 'structured_data' in final_result and final_result['structured_data']:
            output_text = final_result['structured_data']['xml_output']
        elif extract_only and 'features' in final_result and final_result['features']:
            # 以JSON格式输出特征
            output_text = json.dumps(final_result['features'], ensure_ascii=False, indent=2)
        else:
            # 完整处理流程，输出标记后的文本
            if 'structured_data' in final_result and final_result['structured_data']:
                output_text = final_result['structured_data']['marked_text']
            else:
                output_text = purified_text
        
        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            # 如果结果太大，可以选择性地不保存某些字段
            if len(text) > 100000:  # 大文件
                result_to_save = {k: v for k, v in final_result.items() if k != 'original_text'}
                result_to_save['original_text_length'] = len(text)
                json.dump(result_to_save, f, ensure_ascii=False, indent=2)
            else:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
        
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
        logger.error(f"处理文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"处理文件时出错: {e}")
        return None, False

def main():
    parser = argparse.ArgumentParser(description='法律文本处理工具 - 格式清洗、语义净化、结构重组和特征提取的完整流程')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--intermediate', type=str, help='格式清洗后的中间文件路径')
    parser.add_argument('--structured', type=str, help='结构化数据输出文件路径')
    parser.add_argument('--features', type=str, help='特征提取结果输出文件路径')
    parser.add_argument('--clean-only', action='store_true', help='仅执行格式清洗步骤')
    parser.add_argument('--purify-only', action='store_true', help='仅执行语义净化步骤')
    parser.add_argument('--restructure-only', action='store_true', help='仅执行结构重组步骤')
    parser.add_argument('--extract-only', action='store_true', help='仅执行特征提取步骤')
    parser.add_argument('--verbose', action='store_true', help='显示详细处理信息')
    parser.add_argument('--tfidf-threshold', type=float, default=0.35, help='语义净化的TF-IDF阈值，值越大过滤越严格')
    parser.add_argument('--bert-threshold', type=float, default=0.7, help='结构重组的BERT评分阈值')
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU加速BERT模型')
    parser.add_argument('--no-xml', dest='use_xml', action='store_false', help='不使用XML中间格式')
    parser.add_argument('--no-nlp', dest='use_nlp', action='store_false', help='不使用NLP增强功能')
    parser.add_argument('--evaluate', action='store_true', help='评估特征提取质量')
    parser.add_argument('--ground-truth', type=str, help='人工标注的标准答案文件')
    parser.set_defaults(use_xml=True, use_nlp=True)
    
    args = parser.parse_args()
    
    # 处理模式冲突检查
    if sum([args.clean_only, args.purify_only, args.restructure_only, args.extract_only]) > 1:
        print("错误: --clean-only, --purify-only, --restructure-only, --extract-only 不能同时使用多个")
        return
    
    # 执行处理流程
    process_text(
        args.input, 
        args.output, 
        args.intermediate,
        args.structured,
        args.features,
        args.clean_only, 
        args.purify_only,
        args.restructure_only,
        args.extract_only,
        args.verbose, 
        args.tfidf_threshold,
        args.bert_threshold,
        args.use_gpu,
        args.use_xml,
        args.evaluate,
        args.ground_truth,
        args.use_nlp
    )

if __name__ == "__main__":
    main() 