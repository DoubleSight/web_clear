"""
法律文本语义净化模块 - 基于TF-IDF的双层级过滤机制

实现文本预处理.md中描述的"双层级过滤机制"，包括：
1. 硬过滤层：筛除明确无信息价值的词汇
2. 软过滤层：基于TF-IDF动态评估文本重要性
3. 结构化内容过滤层：移除不包含结构化信息模式的句子
"""

import re
import math
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPurifier:
    """
    法律文本语义净化工具，用于对已经格式清洗过的文本进行进一步的语义层面净化
    实现文本预处理.md中描述的三层过滤机制
    """
    
    def __init__(self, tfidf_threshold=0.35):
        """
        初始化语义净化器
        
        Args:
            tfidf_threshold (float): TF-IDF阈值，默认为0.35
        """
        # 设置TF-IDF阈值
        self.tfidf_threshold = tfidf_threshold
        
        # 加载停用词
        self.stopwords = self.load_stopwords()
        
        # 初始化TF-IDF向量化器 - 优化参数配置
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=jieba.cut,
            stop_words=self.stopwords,
            min_df=2,  # 词语至少出现2次才考虑
            max_df=0.95,  # 排除在95%以上文档中都出现的词
            norm='l2',
            use_idf=True,
            smooth_idf=True  # 使用平滑的IDF权重
        )
        
        # 硬过滤层配置 - 无信息价值词汇词典
        self.hard_filter_patterns = {
            # 1. 法定条款引导语
            'legal_citations': [
                r'根据.*?法.*?第\d+条', 
                r'依照.*?法.*?第\d+条',
                r'按照.*?法.*?规定',
                r'依据.*?条例.*?第\d+条',
                r'遵照.*?法.*?第\d+条',
                r'《中华人民共和国.*?法》.*?第\d+条',
                r'依法.*?审理'
            ],
            # 2. 程序性固定表述
            'procedural_phrases': [
                r'以上笔录属实',
                r'捺指印确认',
                r'经过核对无误',
                r'询问人.*?记录人',
                r'我保证所述属实',
                r'签名：.*?日期：',
                r'笔录时间.*?至.*?止',
                r'以上情况如有不实愿承担.*?责任',
                r'该笔录我已(?:经)?过阅读[，,]?与.*?相符',
                r'(?:阅后签名|阅读无误)',
                r'现在开始讯问',
                r'讯问(?:到此)?结束'
            ],
            # 3. 高频口语填充词
            'filler_words': [
                r'\b然后\b',
                r'\b之后\b',
                r'\b接着\b',
                r'\b那个\b',
                r'\b就是说\b',
                r'\b这个\b',
                r'\b所以说\b',
                r'\b我觉得\b',
                r'\b听说\b',
                r'\b好像\b',
                r'\b大概\b',
                r'\b应该\b',
                r'\b可能\b',
                r'\b其实\b',
                r'\b总之\b'
            ]
        }
        
        # 新增：方言词汇与犯罪隐语词典
        self.dialect_and_slang = {
            # 方言词汇(部分示例)
            'dialect': {
                '靓仔': '年轻男性', # 粤语
                '幺爸': '父亲',    # 西南方言
                '伢仔': '小孩子',  # 湖南方言
                '阿婆': '老奶奶',  # 江浙方言
                '嘎拉哈': '小孩子', # 东北方言
                '瓜子': '小孩子',  # 闽南方言
                '巴巴': '父亲',    # 安徽方言
                '砸实': '确实',    # 陕西方言
                '扎扎':  '小孩子'  # 重庆方言
            },
            # 犯罪隐语(部分示例)
            'slang': {
                '游戏': '猥亵行为',
                '金鱼缸': '色情场所',
                '搓麻将': '猥亵行为',
                '喂糖': '诱骗儿童',
                '帮忙找东西': '诱骗儿童',
                '嘿休': '性行为',
                '打幌子': '诱骗行为',
                '抱抱': '身体接触',
                '摸摸': '猥亵行为',
                '看图片': '观看色情内容'
            }
        }

        # 结构化内容过滤层使用的模式 (参考自 TextRestructurer)
        self.element_patterns_for_filtering = {
            'time': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{1,2}-\d{1,2}',
                r'\d{2}:\d{2}(:\d{2})?',
                r'\d{1,2}时\d{1,2}分(?:\d{1,2}秒)?', # 使用非捕获组
                r'(?:上午|下午|晚上|凌晨)\d{1,2}[点时](?:\d{1,2}分)?'
            ],
            'location': [
                # 稍微简化，避免过于复杂的捕获组，只检查模式存在
                r'(?:在|于)?(?:[\u4e00-\u9fa5]{2,}?(?:省|市|县|区|镇|村|路|街|号|楼|室))', 
                r'(?:地点|地址)[:：]?\\s*(?:[\u4e00-\u9fa5]+(?:[路街道区县市省]|小区|大厦|广场|中心|医院|学校|银行|公司|工厂|酒店|商场|市场|公园|广场|车站|机场|码头)[\u4e00-\u9fa5\\d]*)'
            ],
            'person': [
                 # 简化以避免不必要的捕获
                r'(?:被告人|被告|原告|证人|嫌疑人|被害人|当事人|委托人|代理人|辩护人|鉴定人|翻译人员|书记员|法官|检察官|警官|律师)?[\u4e00-\u9fa5]{1,4}(?:[某甲乙丙丁]|[A-Z])(?:\\s*\\(.*?[^)]*?\\))?', # 姓名+身份/备注
                r'[\u4e00-\u9fa5]{2,4}[（(][^）)]*?(?:职业|岗位|年龄|身份|性别|住址|户籍|联系方式)[:：]?[^）)]*?[)）]' # 带括号描述信息
            ],
            'action': [
                # 保留核心动词，增加猥亵儿童案件的特定行为描述
                r'(?:尾随|搂抱|抚摸|触碰|猥亵|侵犯|攻击|威胁|敲诈|勒索|持有|贩卖|运输|制造|走私|贿赂|伪造|强奸|盗窃|抢劫|抢夺|诈骗|绑架|非法拘禁|故意伤害|故意杀人|过失致人死亡|聚众斗殴|寻衅滋事|吸毒|制毒|运输毒品|受贿|行贿|挪用公款|职务侵占|贪污|驾驶|行驶|酒后驾车|无证驾驶|肇事逃逸|脱光|摸胸|摸下体|亲吻|摸臀|拍摄裸照|传播裸照|拍摄视频|收发淫秽信息)',
            ],
            'tool': [
                r'(?:持有|使用|用|携带|藏匿|发现|查获|搜出|缴获)[^，。；,.;]{0,10}?(?:刀|枪|棍|棒|锤|斧|针|管|毒品|现金|手机|电脑|银行卡|证件|文件|车辆|工具|设备|凶器|武器|弹药|爆炸物|麻醉剂|仿真枪|管制刀具|毒品|赃款|赃物|玩具|糖果|钱财|礼物)'
            ]
        }
        
        # 合并所有用于过滤的模式到一个大的正则表达式
        all_patterns = []
        for patterns in self.element_patterns_for_filtering.values():
            all_patterns.extend(patterns)
        # 使用 | 连接所有模式，编译以提高效率
        self.combined_element_regex = re.compile('|'.join(all_patterns))
    
    def load_stopwords(self):
        """加载停用词表，如果文件不存在则使用内置的基本停用词"""
        stopwords_path = 'chinese_stopwords.txt' # 假设文件在同目录下
        try:
            # 尝试多种编码读取
            encodings_to_try = ['utf-8', 'gbk', 'utf-8-sig']
            for enc in encodings_to_try:
                try:
                    with open(stopwords_path, 'r', encoding=enc) as f:
                        words = [line.strip() for line in f if line.strip()]
                        logger.info(f"成功从 {stopwords_path} 加载 {len(words)} 个停用词 (编码: {enc})")
                        return words
                except UnicodeDecodeError:
                    continue
                except FileNotFoundError:
                    logger.warning(f"停用词文件 {stopwords_path} 未找到。")
                    break # 文件不存在，无需尝试其他编码
            # 如果所有编码尝试失败或文件不存在
            logger.warning("将使用内置的基本停用词列表。")
            # 返回更丰富的基本停用词集
            return ['的', '了', '和', '与', '这', '那', '是', '在', '我', '你', '他', '她', '它', '们', 
                    '就', '都', '也', '还', '被', '对', '从', '向', '以', '其', '因', '为', '之', 
                    '于', '但', '所', '并', '或', '则', '却', '如', '若', '即', '当', '既', '虽', '则', 
                    '等', '等等', '啊', '哦', '呀', '呢', '吧', '吗', '呵', '嗯', '哎', '哼', '嘿', '嗨']

        except Exception as e:
            logger.error(f"加载停用词时发生错误: {e}")
            return ['的', '了', '和', '与', '这', '那', '是', '在', '我', '你', '他', '她', '它', '们']
    
    def apply_hard_filters(self, text):
        """
        应用硬过滤层，移除无信息价值词汇
        
        Args:
            text (str): 待处理的文本
            
        Returns:
            tuple: (处理后的文本, 统计信息)
        """
        result_text = text
        filter_stats = {
            'legal_citations_removed': 0,
            'procedural_phrases_removed': 0,
            'filler_words_removed': 0
        }
        
        # 应用法定条款引导语过滤
        for pattern in self.hard_filter_patterns['legal_citations']:
            try:
                matches = re.findall(pattern, result_text)
                filter_stats['legal_citations_removed'] += len(matches)
                result_text = re.sub(pattern, '', result_text)
            except Exception as e:
                logger.warning(f"应用硬过滤模式 '{pattern}' 时出错: {e}")
        
        # 应用程序性固定表述过滤
        for pattern in self.hard_filter_patterns['procedural_phrases']:
            try:
                matches = re.findall(pattern, result_text)
                filter_stats['procedural_phrases_removed'] += len(matches)
                result_text = re.sub(pattern, '', result_text)
            except Exception as e:
                logger.warning(f"应用硬过滤模式 '{pattern}' 时出错: {e}")

        # 应用高频口语填充词过滤
        for pattern in self.hard_filter_patterns['filler_words']:
            try:
                matches = re.findall(pattern, result_text)
                filter_stats['filler_words_removed'] += len(matches)
                result_text = re.sub(pattern, '', result_text)
            except Exception as e:
                logger.warning(f"应用硬过滤模式 '{pattern}' 时出错: {e}")
        
        # 方言词汇与犯罪隐语的标准化处理
        # 不直接删除，而是替换为标准表述，以保留语义信息
        for dialect_word, standard_word in self.dialect_and_slang['dialect'].items():
            result_text = re.sub(rf'\b{dialect_word}\b', f"{dialect_word}({standard_word})", result_text)
            
        for slang_word, standard_word in self.dialect_and_slang['slang'].items():
            result_text = re.sub(rf'\b{slang_word}\b', f"{slang_word}({standard_word})", result_text)
                
        # 清理可能产生的多余空格
        result_text = re.sub(r'\s{2,}', ' ', result_text).strip()
        result_text = re.sub(r'\n{3,}', '\n\n', result_text) # 清理多余空行
        
        return result_text, filter_stats
    
    def apply_soft_filters(self, text):
        """
        应用软过滤层，基于TF-IDF值动态评估句子信息重要性
        TF-IDF值小于阈值的句子被视为冗余信息
        
        Args:
            text (str): 待处理的文本
            
        Returns:
            tuple: (处理后的文本, 统计信息)
        """
        # 分割文本为句子，使用更健壮的分句方式
        sentences = self._split_text_to_sentences(text)
        
        if len(sentences) <= 1:
            logger.info("文本只有一个句子或为空，跳过软过滤")
            return text, {'sentences_removed': 0, 'important_sentences': len(sentences)}
        
        # 计算TF-IDF
        try:
            # 使用jieba分词预处理句子
            processed_sentences = [' '.join(jieba.cut(sentence)) for sentence in sentences]
            
            # 计算TF-IDF矩阵
            tfidf_matrix = self.vectorizer.fit_transform(processed_sentences)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # 计算每个句子的平均TF-IDF得分 - 改进评分逻辑
            sentence_scores = []
            
            # 确定包含犯罪要素的句子，这些句子将获得额外加权
            crime_element_sentences = []
            for i, sentence in enumerate(sentences):
                if any(re.search(pattern, sentence) for pattern_list in self.element_patterns_for_filtering.values() for pattern in pattern_list):
                    crime_element_sentences.append(i)
            
            # 对每个句子计算得分
            for i, sentence in enumerate(sentences):
                sentence_vector = tfidf_matrix[i].toarray()[0]
                
                # 计算该句子中的词语TF-IDF得分
                word_scores = []
                words_in_sentence = list(jieba.cut(sentence))
                
                for word in words_in_sentence:
                    try:
                        # 查找词在特征名中的索引
                        idx = np.where(feature_names == word)[0]
                        if len(idx) > 0:
                            score = sentence_vector[idx[0]]
                            if not np.isnan(score):
                                word_scores.append(score)
                    except ValueError: # 如果词不在特征名中，则忽略
                        pass
                
                # 计算句子得分 - 取最高的几个词得分而不是平均，以避免无意义词的稀释效应
                # 同时如果句子较短，减少考虑的词数
                word_scores.sort(reverse=True)
                top_k = min(3, len(word_scores))  # 只取top 3个词，或者所有词如果少于3个
                
                if top_k > 0:
                    avg_score = sum(word_scores[:top_k]) / top_k
                else:
                    avg_score = 0
                
                # 包含犯罪要素的句子获得额外加权
                if i in crime_element_sentences:
                    avg_score *= 1.25  # 提高25%权重
                
                # 句子长度有适度加权 - 避免过短句子被过滤
                length_factor = min(1.0, max(0.8, len(sentence) / 50))  # 50字以上不加权，短于50字适度降权但不超过20%
                avg_score *= length_factor
                
                sentence_scores.append((sentence, avg_score))
            
            # 筛选重要句子
            important_sentences = []
            for sentence, score in sentence_scores:
                if score >= self.tfidf_threshold:
                    important_sentences.append(sentence)
                else:
                    # 额外检查：如果句子包含明显的关键信息指标，即使TF-IDF低也保留
                    if re.search(r'(?:时间|日期|地点|地址|姓名|年龄|受伤|自残|强迫|威胁|诱骗)', sentence):
                        logger.info(f"低分句保留(包含关键信息指标): {sentence[:30]}...")
                        important_sentences.append(sentence)
                    # 如果句子比较短但得分接近阈值，也保留
                    elif len(sentence) < 20 and score >= self.tfidf_threshold * 0.8:
                        logger.info(f"低分句保留(短句接近阈值): {sentence}")
                        important_sentences.append(sentence)
            
            stats = {
                'sentences_removed': len(sentences) - len(important_sentences),
                'important_sentences': len(important_sentences),
                'important_percentage': (len(important_sentences) / max(1, len(sentences))) * 100
            }
            
            # 输出处理日志
            logger.info(f"软过滤完成: 保留 {stats['important_sentences']} 句, 移除 {stats['sentences_removed']} 句")
            if stats['sentences_removed'] > 0:
                logger.info(f"移除率: {stats['sentences_removed']/len(sentences)*100:.1f}%")
            
            # 保持句子原始顺序，确保上下文语义连贯
            ordered_important_sentences = [s for s in sentences if s in important_sentences]
            result_text = self._join_sentences(ordered_important_sentences)
            
            return result_text, stats
        
        except Exception as e:
            logger.error(f"软过滤层应用失败: {e}")
            # 出错时返回原始文本
            return text, {'sentences_removed': 0, 'important_sentences': len(sentences), 'error': str(e)}

    def _split_text_to_sentences(self, text):
        """
        改进的文本分句功能，处理复杂的中文标点和特殊情况
        
        Args:
            text (str): 待分割的文本
            
        Returns:
            list: 分割后的句子列表
        """
        if not text:
            return []
            
        # 复杂的中文分句规则 - 处理中英文标点
        # 特殊处理引号内的逗号、顿号等非终止标点
        # 1. 先标记引号内的内容
        temp_text = text
        quote_contents = []
        quote_pattern = re.compile(r'["'"](.*?)["'"]')
        
        def replace_with_placeholder(match):
            quote_contents.append(match.group(1))
            return f'QUOTE_PLACEHOLDER_{len(quote_contents)-1}'
            
        temp_text = quote_pattern.sub(replace_with_placeholder, temp_text)
        
        # 2. 分句处理 - 考虑多种分句标点
        sentence_terminators = r'([。！？!?;；])'
        sent_parts = re.split(sentence_terminators, temp_text)
        
        # 3. 重新组装句子
        raw_sentences = []
        for i in range(0, len(sent_parts)-1, 2):
            if i+1 < len(sent_parts):
                raw_sentences.append(sent_parts[i] + sent_parts[i+1])
            else:
                raw_sentences.append(sent_parts[i])
        
        # 如果最后一部分不是终止符结尾
        if len(sent_parts) % 2 == 1:
            last_part = sent_parts[-1].strip()
            if last_part:
                raw_sentences.append(last_part)
        
        # 4. 恢复引号内容
        sentences = []
        for sent in raw_sentences:
            for i in range(len(quote_contents)):
                sent = sent.replace(f'QUOTE_PLACEHOLDER_{i}', f'"{quote_contents[i]}"')
            sentences.append(sent)
        
        # 5. 清洁处理
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
        
    def _join_sentences(self, sentences):
        """
        智能连接句子，保持原始的标点和格式
        
        Args:
            sentences (list): 句子列表
            
        Returns:
            str: 连接后的文本
        """
        if not sentences:
            return ""
            
        # 如果句子已经包含了标点，则直接拼接
        result = ""
        for sent in sentences:
            # 检查句子结尾是否有标点
            if sent and not re.search(r'[。！？!?;；]$', sent):
                result += sent + "。"  # 添加句号作为默认标点
            else:
                result += sent
                
            # 在每个句子后添加空格以提高可读性
            if not sent.endswith('\n'):
                result += ' '
                
        return result.strip()
        
    def _filter_unstructured_content(self, text):
        """
        第三层过滤：移除不包含已知结构化信息模式的句子。
        
        Args:
            text (str): 经过硬过滤和/或软过滤的文本。
            
        Returns:
            tuple: (处理后的文本, 统计信息)
        """
        if not text:
            return "", {'sentences_removed_unstructured': 0, 'sentences_kept_structured': 0}

        # 使用改进的分句方法
        sentences = self._split_text_to_sentences(text)
        
        if not sentences:
             return "", {'sentences_removed_unstructured': 0, 'sentences_kept_structured': 0}

        kept_sentences = []
        for sentence in sentences:
            # 检查句子是否包含任何结构化信息的模式
            if self.combined_element_regex.search(sentence):
                kept_sentences.append(sentence)
            # 增加例外规则：如果句子包含特定的法律术语，也保留
            elif re.search(r'(被告人|原告|被告|证人|法庭|公诉人|辩护人|庭审|案由|案号|诉讼|审理|一审|二审|终审|判决|裁定)', sentence):
                kept_sentences.append(sentence)
        
        stats = {
            'sentences_removed_unstructured': len(sentences) - len(kept_sentences),
            'sentences_kept_structured': len(kept_sentences),
            'structured_kept_percentage': (len(kept_sentences) / max(1, len(sentences))) * 100
        }
        logger.info(f"结构化内容过滤完成: 保留 {stats['sentences_kept_structured']} 句, 移除 {stats['sentences_removed_unstructured']} 句")
        
        # 保持句子原始顺序
        ordered_kept_sentences = [s for s in sentences if s in kept_sentences]
        result_text = self._join_sentences(ordered_kept_sentences)
        
        return result_text, stats

    def get_term_weights(self, text):
        """
        获取文本中词语的TF-IDF权重
        
        Args:
            text (str): 待分析文本
            
        Returns:
            list: 词语及其权重列表 [(word, weight), ...]
        """
        if not text:
            return []
            
        # 分词
        words = list(jieba.cut(text))
        if not words:
            return []
            
        text_processed = ' '.join(words)
        
        try:
            # 计算TF-IDF
            # 注意：fit_transform会拟合数据并转换，如果之前已拟合，应使用transform
            # 为了独立性，这里每次都重新fit_transform
            tfidf_matrix = self.vectorizer.fit_transform([text_processed])
            feature_names = self.vectorizer.get_feature_names_out()
            term_idf = self.vectorizer.idf_
            
            # 计算词语频率
            word_freq = Counter(words)
            total_words = sum(word_freq.values())
            
            # 获取每个词的TF-IDF权重
            term_weights = []
            term_tfidf_map = {} # 用于存储计算出的TF-IDF值

            # 先构建一个词到索引的映射
            feature_indices = {name: i for i, name in enumerate(feature_names)}
            
            # 填充term_tfidf_map
            for word in word_freq.keys():
                 if word in feature_indices:
                     idx = feature_indices[word]
                     # TF-IDF 通常直接从 TfidfVectorizer 的输出矩阵中获取更准确
                     # 但如果需要手动计算:
                     tf = word_freq[word] / total_words
                     idf = term_idf[idx]
                     tfidf_score = tf * idf
                     term_tfidf_map[word] = tfidf_score

            # 转换成列表并排序
            term_weights = sorted(term_tfidf_map.items(), key=lambda item: item[1], reverse=True)
            return term_weights
        except Exception as e:
            logger.error(f"计算词语权重时出错: {e}")
            return []
    
    def purify_text(self, text, apply_hard_filter=True, apply_soft_filter=True, apply_structure_filter=True, tfidf_threshold=None):
        """
        对文本进行语义净化处理 (三层过滤)
        
        Args:
            text (str): 待净化的文本（应已经过格式清洗）
            apply_hard_filter (bool): 是否应用硬过滤层
            apply_soft_filter (bool): 是否应用软过滤层
            apply_structure_filter (bool): 是否应用基于结构化内容的过滤层
            tfidf_threshold (float, optional): 指定的TF-IDF阈值，如果提供则覆盖默认值
            
        Returns:
            tuple: (净化后的文本, 净化过程的详细信息)
        """
        # 如果指定了新的阈值，则临时设置
        original_threshold = self.tfidf_threshold
        if tfidf_threshold is not None:
            self.tfidf_threshold = tfidf_threshold
        
        original_text_length = len(text)
        current_text = text
        stats = {
            'original_length': original_text_length,
            'hard_filter_stats': None,
            'soft_filter_stats': None,
            'structure_filter_stats': None,
            'tfidf_threshold': self.tfidf_threshold,
            'purified_length': 0,
            'reduction_percentage': 0.0
        }
        
        # 保存每个阶段的文本，用于调试和比较
        intermediate_results = {'original': text}
        
        # 应用硬过滤层
        if apply_hard_filter:
            current_text, hard_filter_stats = self.apply_hard_filters(current_text)
            stats['hard_filter_stats'] = hard_filter_stats
            intermediate_results['hard_filtered'] = current_text
            logger.info(f"硬过滤后文本长度: {len(current_text)}")
        
        # 应用软过滤层
        if apply_soft_filter:
            current_text, soft_filter_stats = self.apply_soft_filters(current_text)
            stats['soft_filter_stats'] = soft_filter_stats
            intermediate_results['soft_filtered'] = current_text
            logger.info(f"软过滤后文本长度: {len(current_text)}")

        # 应用结构化内容过滤层
        if apply_structure_filter:
            # 如果经过软过滤后的文本减少率超过50%，跳过结构化过滤以避免过度过滤
            if apply_soft_filter and stats['soft_filter_stats'] and \
               stats['soft_filter_stats'].get('important_percentage', 100) < 50:
                logger.warning("软过滤移除率超过50%，跳过结构化过滤以避免过度过滤")
                stats['structure_filter_stats'] = {
                    'sentences_removed_unstructured': 0,
                    'sentences_kept_structured': 0,
                    'structured_kept_percentage': 100,
                    'skipped': True
                }
            else:
                current_text, structure_filter_stats = self._filter_unstructured_content(current_text)
                stats['structure_filter_stats'] = structure_filter_stats
                intermediate_results['structure_filtered'] = current_text
                logger.info(f"结构化内容过滤后文本长度: {len(current_text)}")
            
        # 计算最终处理结果
        final_text = current_text
        stats['purified_length'] = len(final_text)
        if original_text_length > 0:
            stats['reduction_percentage'] = ((original_text_length - stats['purified_length']) / original_text_length) * 100
            
            # 检查是否过度过滤
            if stats['reduction_percentage'] > 70:
                logger.warning(f"过滤后文本减少了 {stats['reduction_percentage']:.1f}%, 可能过度过滤，检查阈值配置")
                
                # 如果太过激进，尝试仅使用硬过滤
                if apply_soft_filter and stats['purified_length'] < original_text_length * 0.3:
                    logger.warning("尝试仅使用硬过滤的结果...")
                    alternative_text = intermediate_results.get('hard_filtered', text)
                    stats['alternative_text_available'] = True
                    stats['alternative_length'] = len(alternative_text)
                    stats['alternative_reduction'] = ((original_text_length - len(alternative_text)) / original_text_length) * 100
                    
                    # 判断是否使用备选文本
                    if stats['alternative_reduction'] < 50:
                        final_text = alternative_text
                        stats['used_alternative'] = True
                        stats['purified_length'] = len(final_text)
                        stats['reduction_percentage'] = stats['alternative_reduction']
                        logger.info(f"使用备选文本 (仅硬过滤)，减少了 {stats['reduction_percentage']:.1f}%")
                
        else:
             stats['reduction_percentage'] = 0.0

        # 恢复原始阈值
        if tfidf_threshold is not None:
            self.tfidf_threshold = original_threshold
            
        return final_text, stats

def process_file(input_file, output_file, tfidf_threshold=0.35, apply_hard_filter=True, apply_soft_filter=True, apply_structure_filter=True):
    """
    处理文件的便捷函数
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        tfidf_threshold (float): TF-IDF阈值
        apply_hard_filter (bool): 是否应用硬过滤层
        apply_soft_filter (bool): 是否应用软过滤层
        apply_structure_filter (bool): 是否应用结构化内容过滤层
        
    Returns:
        bool: 处理是否成功
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        purifier = TextPurifier(tfidf_threshold=tfidf_threshold)
        purified_text, stats = purifier.purify_text(
            text, 
            apply_hard_filter=apply_hard_filter, 
            apply_soft_filter=apply_soft_filter,
            apply_structure_filter=apply_structure_filter # 应用新过滤器
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(purified_text)
        
        print(f"文本语义净化完成，已保存到: {output_file}")
        print(f"原始长度: {stats['original_length']} 字符")
        print(f"净化后长度: {stats['purified_length']} 字符")
        print(f"总缩减比例: {stats['reduction_percentage']:.2f}%")
        
        if stats['hard_filter_stats']:
            print("\n硬过滤层统计:")
            print(f"- 移除法定条款引导语: {stats['hard_filter_stats']['legal_citations_removed']} 处")
            print(f"- 移除程序性固定表述: {stats['hard_filter_stats']['procedural_phrases_removed']} 处")
            print(f"- 移除高频口语填充词: {stats['hard_filter_stats']['filler_words_removed']} 处")
        
        if stats['soft_filter_stats']:
            print("\n软过滤层 (TF-IDF) 统计:")
            print(f"- 保留重要句子: {stats['soft_filter_stats'].get('important_sentences', 'N/A')} 句")
            print(f"- 移除冗余句子: {stats['soft_filter_stats'].get('sentences_removed', 'N/A')} 句")
            if 'important_percentage' in stats['soft_filter_stats']:
                print(f"- 重要句子占比: {stats['soft_filter_stats']['important_percentage']:.2f}%")

        if stats['structure_filter_stats']: # 显示新过滤器的统计
            print("\n结构化内容过滤层统计:")
            print(f"- 保留含结构化模式句子: {stats['structure_filter_stats']['sentences_kept_structured']} 句")
            print(f"- 移除不含结构化模式句子: {stats['structure_filter_stats']['sentences_removed_unstructured']} 句")
            if 'structured_kept_percentage' in stats['structure_filter_stats']:
                print(f"- 含结构化模式句子占比: {stats['structure_filter_stats']['structured_kept_percentage']:.2f}%")

        return True
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='法律文本语义净化工具 (三层过滤)')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.35, help='TF-IDF阈值，默认为0.35')
    parser.add_argument('--no-hard-filter', action='store_true', help='不应用硬过滤层')
    parser.add_argument('--no-soft-filter', action='store_true', help='不应用软过滤层 (TF-IDF)')
    parser.add_argument('--no-structure-filter', action='store_true', help='不应用结构化内容过滤层') # 新增选项
    
    args = parser.parse_args()
    
    process_file(
        args.input, 
        args.output,
        tfidf_threshold=args.threshold,
        apply_hard_filter=not args.no_hard_filter,
        apply_soft_filter=not args.no_soft_filter,
        apply_structure_filter=not args.no_structure_filter # 控制新过滤器
    )
