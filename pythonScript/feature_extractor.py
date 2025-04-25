import re
import os
import json
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any, Tuple, Union, Set

# 日志记录器 (后续可以配置)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入NLP相关库
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize
    
    # 加载SpaCy模型
    try:
        nlp = spacy.load("zh_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("SpaCy中文模型加载成功")
    except:
        SPACY_AVAILABLE = False
        logger.warning("SpaCy中文模型未安装，高级语义分析将受限")
    
    # 初始化NLTK
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
    except LookupError:
        try:
            nltk.download('punkt')
            NLTK_AVAILABLE = True
        except:
            NLTK_AVAILABLE = False
            logger.warning("NLTK数据未安装，部分文本分析功能将受限")
            
    NLP_LIBRARIES_AVAILABLE = SPACY_AVAILABLE or NLTK_AVAILABLE
except ImportError:
    # 创建伪对象，避免导入错误
    class DummyToken:
        def __init__(self):
            self.text = ""
            self.pos_ = ""
            self.dep_ = ""
            self.children = []
            self.i = 0
            
    class DummySent:
        def __init__(self, text=""):
            self.text = text
            
        def __iter__(self):
            return iter([])
            
    class DummyDoc:
        def __init__(self):
            self.ents = []
            self.sents = []
            
        def __len__(self):
            return 0
            
        def __getitem__(self, key):
            if isinstance(key, slice):
                return []
            return DummyToken()
            
        def __iter__(self):
            return iter([])
            
    class DummyNLP:
        def __call__(self, *args, **kwargs):
            return DummyDoc()
    
    spacy = None
    nltk = None
    nlp = DummyNLP()
    SPACY_AVAILABLE = False
    NLTK_AVAILABLE = False
    NLP_LIBRARIES_AVAILABLE = False
    logger.warning("NLP相关库未安装，高级语义分析将被禁用。考虑安装: pip install spacy nltk")
    logger.warning("安装SpaCy中文模型: python -m spacy download zh_core_web_sm")

class CrimeFeatures:
    """
    存储从法律文本中提取的猥亵儿童犯罪特征。
    结构基于 "文本预处理.md" 中的表 16。
    同时包含将其转换为数值特征向量的方法。
    """
    # 定义类别特征的固定顺序，用于独热编码
    CATEGORY_MAPS = {
        'victim_gender': ['男', '女', '未知'],
        'victim_relationship_with_perpetrator': ['无依赖', '有依赖'],
        'victim_injury_severity': ['轻度', '中度', '重度', '极端'],
        'perpetrator_gender': ['男', '女'],
        'perpetrator_occupation_category': ['无业', '低接触', '中接触', '高接触'],
        'perpetrator_has_prior_record': ['无', '有', '惯犯'],
        'perpetrator_has_hidden_prior_record': ['无', '有', '惯犯'], # 注意：文档中是二分，但表19示例是三分？暂按三分处理
        'perpetrator_motive': ['病理驱动', '权利支配', '经济交易', '报复社会', '替代满足', '机会主义'],
        'crime_location_category': ['常规场所', '公共场所', '公共场所隐蔽空间', '特殊场所', '网络虚拟场所', '隐匿场所'],
        'crime_violated_body_part_category': ['生殖器', '明显性象征部位', '普通部位'],
        'crime_control_method': ['物质利诱', '暴力殴打', '药物迷幻', '工具捆绑', '精神威胁', '精神洗脑']
        # 注意：crime_duration, crime_time_of_day 未在此处向量化，需要特定处理或进一步分类
    }

    def __init__(self):
        # --- 被害人信息 ---
        self.victim_age: Optional[int] = None
        self.victim_gender: Optional[str] = None # 例如: "男", "女", "未知"
        self.victim_physical_disability: Optional[bool] = None # 是否肢体残缺
        self.victim_intellectual_disability: Optional[bool] = None # 是否智力残疾
        self.victim_relationship_with_perpetrator: Optional[str] = None # 例如: "无依赖", "有依赖"
        self.victim_multiple_victims_involved: Optional[bool] = None # 是否一次多人受害 (同案犯侵害多个?)
        self.victim_injury_severity: Optional[str] = None # 例如: "轻度", "中度", "重度", "极端"

        # --- 犯罪人信息 ---
        self.perpetrator_age: Optional[int] = None
        self.perpetrator_gender: Optional[str] = None # 例如: "男", "女"
        self.perpetrator_occupation_category: Optional[str] = None # 例如: "无业", "低接触", "中接触", "高接触"
        self.perpetrator_has_prior_record: Optional[str] = None # 例如: "无", "有", "惯犯"
        self.perpetrator_has_hidden_prior_record: Optional[str] = None # 例如: "无", "有", "惯犯"
        self.perpetrator_motive: Optional[str] = None # 例如: "病理驱动", "权利支配", "经济交易", "报复社会", "替代满足", "机会主义"

        # --- 犯罪行为过程 ---
        self.crime_time_of_day: Optional[str] = None # 例如: "白天", "夜晚", 具体时间段
        self.crime_location_category: Optional[str] = None # 例如: "常规场所", "公共场所", "公共场所隐蔽空间", "特殊场所", "网络虚拟场所", "隐匿场所"
        self.crime_duration: Optional[str] = None # 例如: "短暂(<1分钟)", "几分钟", "较长(>10分钟)" 或具体时长
        self.crime_watched_pornography: Optional[bool] = None # 是否观看色情视频

        # --- 犯罪行为方式 ---
        self.crime_violated_body_part_category: Optional[str] = None # 例如: "生殖器", "明显性象征部位", "普通部位"
        self.crime_perpetrator_exposed_genitals: Optional[bool] = None # 有否暴露自身生殖器
        self.crime_forced_interaction: Optional[bool] = None # 有否强迫性互动
        self.crime_spread_obscene_info_online: Optional[bool] = None # 有否网络传播淫秽信息
        self.crime_recorded_on_site: Optional[bool] = None # 有否现场拍摄
        self.crime_control_method: Optional[str] = None # 例如: "物质利诱", "暴力殴打", "药物迷幻", "工具捆绑", "精神威胁", "精神洗脑"
        self.crime_repeated_offenses: Optional[bool] = None # 是否多次作案 (针对同一被害人?)
        self.crime_multiple_perpetrators: Optional[bool] = None # 是否多人作案
        self.crime_multiple_perpetrators_and_repeated: Optional[bool] = None # 是否多人多次作案

        # --- 其他可能从标签提取的信息 ---
        self.perpetrator_name: Optional[str] = None # 犯罪人姓名
        self.tool_used: Optional[str] = None # 使用的工具

    def __repr__(self):
        features_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and v is not None and k not in ['CATEGORY_MAPS']}
        return f"CrimeFeatures({features_dict})"

    def set_feature(self, key: str, value: Any, overwrite: bool = True):
        """安全地设置特征值，进行类型检查和转换（如果可能）。"""
        if hasattr(self, key):
            if overwrite or getattr(self, key) is None:
                try:
                    # 获取当前值以检查类型
                    current_value = getattr(self, key)
                    
                    converted_value = value
                    if value is not None:
                        # 年龄特征处理 - 整数类型
                        if key in ['victim_age', 'perpetrator_age']:
                            try:
                                converted_value = int(value)
                            except (ValueError, TypeError):
                                logger.warning(f"无法将 '{value}' 转换为整数，特征 '{key}' 设为None")
                                converted_value = None
                        
                        # 布尔特征处理
                        elif key in [
                            'victim_physical_disability', 'victim_intellectual_disability',
                            'victim_multiple_victims_involved', 'crime_watched_pornography',
                            'crime_perpetrator_exposed_genitals', 'crime_forced_interaction',
                            'crime_spread_obscene_info_online', 'crime_recorded_on_site',
                            'crime_repeated_offenses', 'crime_multiple_perpetrators',
                            'crime_multiple_perpetrators_and_repeated'
                        ]:
                            str_val = str(value).lower().strip()
                            if str_val in ['true', '1', 'yes', '是', '有', '属实']:
                                converted_value = True
                            elif str_val in ['false', '0', 'no', '否', '无', '未']:
                                converted_value = False
                            else:
                                logger.warning(f"无法将 '{value}' 明确转换为布尔值，特征 '{key}' 设为None")
                                converted_value = None
                        
                        # 类别特征处理
                        elif key in self.CATEGORY_MAPS:
                            str_val = str(value)
                            if str_val not in self.CATEGORY_MAPS[key]:
                                logger.warning(f"特征 '{key}' 的值 '{value}' 不在预定义的类别 {self.CATEGORY_MAPS[key]} 中，将使用原始值。后续向量化可能忽略此值。")
                            converted_value = str_val
                        
                        # 字符串特征处理
                        elif isinstance(current_value, str) or current_value is None:
                            converted_value = str(value)

                    # 设置值
                    setattr(self, key, converted_value)
                except Exception as e:
                    logger.error(f"设置特征 '{key}' 时发生意外错误: {e}")
        else:
            logger.warning(f"尝试设置不存在的特征: {key}")

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """返回数值特征向量的列名列表。"""
        feature_names = []
        # 按照 to_feature_vector 的顺序添加
        # 数值特征
        feature_names.append('victim_age')
        feature_names.append('perpetrator_age')
        # 布尔特征
        boolean_features = [
            'victim_physical_disability', 'victim_intellectual_disability',
            'victim_multiple_victims_involved', 'crime_watched_pornography',
            'crime_perpetrator_exposed_genitals', 'crime_forced_interaction',
            'crime_spread_obscene_info_online', 'crime_recorded_on_site',
            'crime_repeated_offenses', 'crime_multiple_perpetrators',
            'crime_multiple_perpetrators_and_repeated'
        ]
        feature_names.extend(boolean_features)
        # 类别特征 (独热编码)
        for feature_key, categories in cls.CATEGORY_MAPS.items():
            for category in categories:
                feature_names.append(f'{feature_key}_{category}')
        return feature_names

    @property
    def categories(self) -> List[str]:
        """返回数值特征向量的列名列表。这是get_feature_names的属性版本，保持向后兼容性。"""
        return self.get_feature_names()
    
    def to_feature_vector(self) -> List[float]:
        """将特征转换为数值特征向量。
        返回一个与get_feature_names()顺序一致的浮点数值列表。"""
        vector = []

        # 1. 数值特征 (例如年龄)，缺失值用 -1 填充
        vector.append(float(self.victim_age) if self.victim_age is not None else -1.0)
        vector.append(float(self.perpetrator_age) if self.perpetrator_age is not None else -1.0)

        # 2. 布尔特征 (True -> 1.0, False/None -> 0.0)
        boolean_features = [
            self.victim_physical_disability, self.victim_intellectual_disability,
            self.victim_multiple_victims_involved, self.crime_watched_pornography,
            self.crime_perpetrator_exposed_genitals, self.crime_forced_interaction,
            self.crime_spread_obscene_info_online, self.crime_recorded_on_site,
            self.crime_repeated_offenses, self.crime_multiple_perpetrators,
            self.crime_multiple_perpetrators_and_repeated
        ]
        vector.extend([1.0 if feature else 0.0 for feature in boolean_features])

        # 3. 类别特征 (独热编码)
        for feature_key, categories in self.CATEGORY_MAPS.items():
            one_hot_vector = [0.0] * len(categories)
            current_value = getattr(self, feature_key, None)
            if current_value is not None and current_value in categories:
                try:
                    index = categories.index(current_value)
                    one_hot_vector[index] = 1.0
                except ValueError:
                    logger.warning(f"特征 '{feature_key}' 的值 '{current_value}' 在映射时未找到，独热编码将全为0。")
            vector.extend(one_hot_vector)

        return vector

class FeatureExtractor:
    """
    负责从预处理后的文本中提取犯罪特征。
    支持从普通文本或XML格式的输入中提取特征。
    增强版支持NLP分析和高级语义理解。
    """
    def __init__(self, use_nlp=True):
        self.regex_patterns: Dict[str, Any] = {}
        self.keywords: Dict[str, Dict[str, List[str]]] = {}
        self.dialect_slang: Dict[str, str] = {}
        self.use_nlp = use_nlp and NLP_LIBRARIES_AVAILABLE
        self._load_resources()
        logger.info(f"特征提取器初始化完成。NLP支持: {'启用' if self.use_nlp else '禁用'}")

    def _load_resources(self):
        """加载关键词库、正则表达式等资源，支持从外部文件加载扩展词典。"""
        # 1. 用于显式标签提取的正则表达式
        self.regex_patterns['tags'] = {
            'victim_age': r'\[VICTIM AGE=(\d+)\]',
            'perpetrator_age': r'\[PERP AGE=(\d+)\]',
            'victim_injury_severity': r'\[SEVERITY\](.*?)\[/SEVERITY\]',
            'victim_relationship_with_perpetrator': r'\[RELATIONSHIP\](.*?)\[/RELATIONSHIP\]',
            'crime_control_method': r'\[CONTROL\](.*?)\[/CONTROL\]',
            'crime_time_of_day': r'\[TIME\](.*?)\[/TIME\]',
            'perpetrator_name': r'\[PER\](.*?)\[/PER\]',
            'tool_used': r'\[TOOL(?:-.*?)?\](.*?)\[/TOOL\]',
            'perpetrator_motive': r'\[MOTIVE\](.*?)\[/MOTIVE\]',
            'crime_location_category': r'\[LOCATION\](.*?)\[/LOCATION\]',
        }
        
        # 2. XML标签到特征映射
        self.xml_tag_mapping = {
            'TIME': {'feature': 'crime_time_of_day', 'process': None},
            'LOC': {'feature': 'crime_location_category', 'process': self._process_location},
            'PER': {'feature': 'perpetrator_name', 'process': None},
            'ACTION': {'feature': None, 'process': self._process_action},
            'TOOL': {'feature': 'tool_used', 'process': None},
            'METHOD': {'feature': 'crime_control_method', 'process': self._process_method_type},
            'SEVERITY': {'feature': 'victim_injury_severity', 'process': None},
            'RELATIONSHIP': {'feature': 'victim_relationship_with_perpetrator', 'process': self._process_relationship},
            'MOTIVE': {'feature': 'perpetrator_motive', 'process': self._process_motive_type},
            'VICTIM_AGE': {'feature': 'victim_age', 'process': int},
            'PERP_AGE': {'feature': 'perpetrator_age', 'process': int},
            'ACTION_CHAIN': {'feature': None, 'process': self._process_action_chain},
            'BODY_PART': {'feature': 'crime_violated_body_part_category', 'process': self._process_body_part}
        }

        # 3. 用于关键词/规则提取的模式和关键词
        self.regex_patterns['keywords'] = {
            'prior_record_explicit': r'(曾因|有|涉嫌)(.{1,15}?(犯罪|违法|被判|前科|记录))'
        }

        # 4. 基础关键词库
        self.keywords = {
            # --- 布尔型特征 ---
            'victim_physical_disability': {'True': ['残疾', '残障', '肢体不便', '截肢', '残缺']},
            'victim_intellectual_disability': {'True': ['智力障碍', '智障', '智力低下', '精神发育迟滞', '唐氏综合征', '弱智']},
            'crime_watched_pornography': {'True': ['观看色情', '浏览黄片', '看AV', '播放淫秽视频', '观看淫秽物品']},
            'crime_perpetrator_exposed_genitals': {'True': ['暴露生殖器', '脱裤子', '露阴', '裸露下体']},
            'crime_forced_interaction': {'True': ['强迫互动', '强迫触摸', '强迫观看', '逼迫', '强行']},
            'crime_spread_obscene_info_online': {'True': ['网络传播淫秽', '网上散布', '发送裸照', '传播不雅视频', 'QQ传播', '微信传播']},
            'crime_recorded_on_site': {'True': ['现场拍摄', '录像', '拍照', '录制视频', '手机拍摄']},
            'crime_repeated_offenses': {'True': ['多次作案', '反复猥亵', '长期侵害', '不止一次', '多次实施']},
            'crime_multiple_perpetrators': {'True': ['多人作案', '团伙', '伙同', '共同实施', '轮流', '一起上']},

            # --- 简单类别特征 ---
            'perpetrator_gender': {
                '男': ['男性', '男子', '男孩'],
                '女': ['女性', '女子', '女孩']
            },
            'victim_gender': {
                '男': ['男性', '男童', '男孩'],
                '女': ['女性', '女童', '女孩']
            },
            'perpetrator_has_prior_record': {
                '无': ['无前科', '无犯罪记录', '初犯']
            },
            'perpetrator_has_hidden_prior_record': {
                '有': ['多次处理', '多次被抓']
            },

            # --- 复杂类别特征 (基础词典) ---
            'perpetrator_occupation_category': {
                '高接触': ['教师', '老师', '辅导员', '教练', '医生', '护士', '保姆', '宿管', '福利院', '志愿者', '幼儿教师', '小学老师']
            },
            'perpetrator_motive': {
                '病理驱动': ['恋童癖', '性冲动无法控制', '特殊癖好', '心理变态', '性瘾']
            },
            'crime_location_category': {
                '常规场所': ['家中', '卧室', '住所', '宿舍', '自己家', '被害人家', '客厅', '床上']
            },
            'crime_control_method': {
                '物质利诱': ['糖果', '玩具', '零食', '给钱', '红包', '礼物', '游戏装备', '利诱', '诱骗', '好处']
            },
            'crime_violated_body_part_category': {
                 '生殖器':['生殖器', '阴部', '下体', '私处', '阴茎', '阴道', '肛门']
            }
        }
        
        # 5. 尝试加载外部词典文件
        try:
            ext_dict_path = os.path.join(os.path.dirname(__file__), 'data', 'feature_dictionaries.json')
            if os.path.exists(ext_dict_path):
                with open(ext_dict_path, 'r', encoding='utf-8') as f:
                    ext_dict = json.load(f)
                    # 合并词典
                    for category, terms in ext_dict.items():
                        if category == 'dialect_slang':
                            # 特殊处理方言隐语词典
                            self.dialect_slang.update(terms)
                            continue
                            
                        if category in self.keywords:
                            for key, values in terms.items():
                                if key in self.keywords[category]:
                                    self.keywords[category][key].extend(values)
                                else:
                                    self.keywords[category][key] = values
                        else:
                            self.keywords[category] = terms
                logger.info(f"已加载外部词典: {ext_dict_path}")
                logger.info(f"方言隐语词典: {len(self.dialect_slang)} 项")
        except Exception as e:
            logger.error(f"加载外部词典失败: {e}")
    
    def _process_method_type(self, method_text: str) -> Tuple[str, str]:
        """细分控制手段类型"""
        method_mapping = {
            '威胁': '精神威胁',
            '恐吓': '精神威胁',
            '胁迫': '精神威胁',
            '强制': '暴力殴打',
            '打': '暴力殴打',
            '掐': '暴力殴打',
            '给钱': '物质利诱',
            '零花钱': '物质利诱',
            '礼物': '物质利诱',
            '洗脑': '精神洗脑',
            '操控': '精神洗脑',
            '欺骗': '精神洗脑',
            '奖励': '物质利诱',
            '药物': '药物迷幻',
            '迷药': '药物迷幻',
            '安眠药': '药物迷幻',
            '绳索': '工具捆绑',
            '绑': '工具捆绑',
            '捆': '工具捆绑'
        }
        
        for key, value in method_mapping.items():
            if key in method_text:
                return 'crime_control_method', value
                
        return 'crime_control_method', ''  # 返回空字符串而非None
    
    def _process_motive_type(self, motive_text: str) -> Tuple[str, str]:
        """处理动机类型"""
        motive_mapping = {
            '控制': '权利支配',
            '支配': '权利支配',
            '权力': '权利支配',
            '地位': '权利支配',
            '冲动': '病理驱动',
            '欲望': '病理驱动',
            '癖好': '病理驱动',
            '变态': '病理驱动',
            '金钱': '经济交易',
            '利益': '经济交易',
            '报酬': '经济交易',
            '交易': '经济交易',
            '报复': '报复社会',
            '泄愤': '报复社会',
            '怨恨': '报复社会',
            '孤独': '替代满足',
            '缺失': '替代满足',
            '寂寞': '替代满足',
            '偶然': '机会主义',
            '临时': '机会主义',
            '突发': '机会主义',
            '一时': '机会主义'
        }
        
        for key, value in motive_mapping.items():
            if key in motive_text:
                return 'perpetrator_motive', value
                
        return 'perpetrator_motive', ''  # 返回空字符串而非None
    
    def _process_relationship(self, rel_text: str) -> Tuple[str, str]:
        """处理关系类型"""
        dependency_indicators = {
            '有依赖': ['老师', '教练', '班主任', '家长', '亲戚', '监护人', '照顾', '负责', '管理', '师生'],
            '无依赖': ['陌生', '路人', '初次', '不认识', '偶然相遇', '无关系']
        }
        
        for rel_type, indicators in dependency_indicators.items():
            if any(ind in rel_text for ind in indicators):
                return 'victim_relationship_with_perpetrator', rel_type
                
        return 'victim_relationship_with_perpetrator', ''  # 返回空字符串而非None
    
    def _process_body_part(self, body_part_text: str) -> Tuple[str, str]:
        """处理身体部位类型"""
        body_part_mapping = {
            '生殖器': ['生殖器', '阴部', '下体', '私处', '阴茎', '阴道', '肛门', '隐私部位', '性器官'],
            '明显性象征部位': ['胸部', '臀部', '大腿', '腰部', '乳房', '胸', '臀', '屁股'],
            '普通部位': ['手', '脚', '腿', '肩', '背', '脖子', '脸', '头']
        }
        
        for category, keywords in body_part_mapping.items():
            if any(keyword in body_part_text for keyword in keywords):
                return 'crime_violated_body_part_category', category
                
        return 'crime_violated_body_part_category', ''  # 返回空字符串而非None
    
    def _process_location(self, location_text):
        """处理地点信息，判断地点类别"""
        if any(keyword in location_text for keyword in ['家中', '卧室', '住所', '宿舍']):
            return 'crime_location_category', '常规场所'
        elif any(keyword in location_text for keyword in ['学校', '医院', '商场', '公园']):
            return 'crime_location_category', '公共场所'
        elif any(keyword in location_text for keyword in ['厕所', '角落', '僻静']):
            return 'crime_location_category', '公共场所隐蔽空间'
        elif any(keyword in location_text for keyword in ['网络', '线上', '虚拟']):
            return 'crime_location_category', '网络虚拟场所'
        else:
            return 'crime_location_category', None  # 需要进一步语义判断
        
    def _process_action(self, action_text):
        """处理行为信息，提取多种特征"""
        features_to_set = []
        
        # 检查是否触及敏感部位
        if any(keyword in action_text for keyword in ['触碰下体', '摸胸', '抚摸阴部', '触碰生殖器']):
            features_to_set.append(('crime_violated_body_part_category', '生殖器'))
        
        # 检查是否暴露生殖器
        if any(keyword in action_text for keyword in ['脱裤子', '露阴', '裸露下体']):
            features_to_set.append(('crime_perpetrator_exposed_genitals', True))
            
        # 检查是否强迫互动
        if any(keyword in action_text for keyword in ['强迫', '逼迫', '强行']):
            features_to_set.append(('crime_forced_interaction', True))
            
        # 检查是否现场拍摄
        if any(keyword in action_text for keyword in ['拍摄', '录像', '拍照', '录制视频']):
            features_to_set.append(('crime_recorded_on_site', True))
            
        # 检查是否多次作案
        if any(keyword in action_text for keyword in ['多次', '反复', '长期']):
            features_to_set.append(('crime_repeated_offenses', True))
            
        return features_to_set
    
    def _process_action_chain(self, chain_text):
        """处理行为链信息"""
        # 行为链通常用于确定犯罪过程，可以提取更多信息
        features_to_set = []
        
        # 例如，如果行为链表明多人作案
        if '→' in chain_text and any(keyword in chain_text for keyword in ['伙同', '一起', '共同']):
            features_to_set.append(('crime_multiple_perpetrators', True))
            
        # 如果行为链表明先后实施了多种行为
        if '→' in chain_text and any(keyword in chain_text for keyword in ['多次', '反复']):
            features_to_set.append(('crime_repeated_offenses', True))
            
        return features_to_set

    def extract_features(self, text: str, is_xml: bool = False) -> CrimeFeatures:
        """
        从给定的文本中提取所有定义的犯罪特征。
        支持普通文本和XML格式输入，并可选使用NLP增强分析。
        
        Args:
            text (str): 要处理的文本内容
            is_xml (bool): 是否为XML格式
            
        Returns:
            CrimeFeatures: 提取到的特征对象
        """
        logger.info(f"开始从文本中提取特征 (文本长度: {len(text)}, 格式: {'XML' if is_xml else '普通文本'})...")
        features = CrimeFeatures()

        if is_xml:
            self._extract_from_xml(text, features)
        else:
            # 按顺序应用提取逻辑
            self._extract_explicit_tags(text, features)
            self._extract_keyword_features(text, features)
            self._extract_semantic_features(text, features)
        
        # 应用高级NLP分析
        if self.use_nlp:
            logger.info("执行NLP增强分析...")
            self._analyze_relationship_network(text, features)
            self._analyze_temporal_causal_chains(text, features)
        
        # 后处理：填补逻辑关联的特征
        self._post_process_features(features)

        logger.info("特征提取完成。")
        return features
    
    def _post_process_features(self, features: CrimeFeatures) -> None:
        """后处理：基于已提取的特征推断其他相关特征"""
        # 例如，如果检测到多人作案且多次作案，设置组合特征
        if features.crime_multiple_perpetrators and features.crime_repeated_offenses:
            features.set_feature('crime_multiple_perpetrators_and_repeated', True, overwrite=False)
            logger.debug("  [后处理] 组合特征: 多人多次作案")
        
        # 如果提取到特定身体部位信息但未设置violated_body_part_category
        if features.crime_violated_body_part_category is None:
            # 尝试从其他特征推断
            if features.crime_perpetrator_exposed_genitals:
                features.set_feature('crime_violated_body_part_category', '生殖器', overwrite=False)
                logger.debug("  [后处理] 从'暴露生殖器'推断触及部位为'生殖器'")
    
    def _extract_from_xml(self, xml_text: str, features: CrimeFeatures):
        """从XML格式的文本中提取特征"""
        logger.info("从XML中提取特征...")
        
        try:
            # 处理可能不规范的XML
            if not xml_text.startswith('<?xml'):
                xml_text = f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_text}'
                
            # 解析XML
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                # 如果解析失败，尝试修复常见问题后重试
                xml_text = xml_text.replace('&', '&amp;')
                try:
                    root = ET.fromstring(xml_text)
                except ET.ParseError as e:
                    logger.error(f"解析XML时出错: {e}")
                    # 如果仍然无法解析，退回到正则表达式提取
                    logger.warning("XML解析失败，退回到正则表达式提取模式")
                    self._extract_explicit_tags(xml_text, features)
                    return
            
            # 遍历所有标签，根据映射提取特征
            extracted_count = 0
            for element in root.findall('.//*'):
                tag = element.tag
                text = element.text.strip() if element.text else ""
                
                if not text:
                    continue
                    
                if tag in self.xml_tag_mapping:
                    mapping = self.xml_tag_mapping[tag]
                    feature_name = mapping['feature']
                    processor = mapping['process']
                    
                    if processor:
                        # 如果有特殊处理函数
                        result = processor(text)
                        if isinstance(result, tuple):
                            # 单个特征
                            feat_name, value = result
                            if value is not None:
                                features.set_feature(feat_name, value, overwrite=False)
                                extracted_count += 1
                                logger.debug(f"  [XML] 从标签 <{tag}> 提取特征 '{feat_name}': '{value}'")
                        elif isinstance(result, list):
                            # 多个特征
                            for feat_name, value in result:
                                if value is not None:
                                    features.set_feature(feat_name, value, overwrite=False)
                                    extracted_count += 1
                                    logger.debug(f"  [XML] 从标签 <{tag}> 提取特征 '{feat_name}': '{value}'")
                    elif feature_name:
                        # 直接映射到特征
                        features.set_feature(feature_name, text, overwrite=False)
                        extracted_count += 1
                        logger.debug(f"  [XML] 从标签 <{tag}> 提取特征 '{feature_name}': '{text}'")
            
            logger.info(f"XML特征提取完成，共提取 {extracted_count} 个特征。")
            
            # 额外应用关键词提取，以补充XML未提供的特征
            plain_text = ' '.join(ET.tostring(root, encoding='unicode', method='text').split())
            self._extract_keyword_features(plain_text, features)
            self._extract_semantic_features(plain_text, features)
            
        except Exception as e:
            logger.error(f"XML特征提取过程中出错: {e}")
            # 出错时退回到标准提取
            logger.warning("XML处理出错，退回到标准文本提取")
            self._extract_explicit_tags(xml_text, features)

    # --- 私有辅助提取方法 --- #
    def _extract_explicit_tags(self, text: str, features: CrimeFeatures):
        """使用正则表达式提取显式标记的特征。"""
        logger.info("执行显式标签提取...")
        extracted_count = 0
        tag_patterns = self.regex_patterns.get('tags', {})
        if not tag_patterns:
            logger.warning("没有找到用于标签提取的正则表达式模式。")
            return

        for feature_name, pattern in tag_patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches[0].strip() # 默认取第一个
                    if value:
                        features.set_feature(feature_name, value, overwrite=True)
                        logger.debug(f"  [标签] 找到 '{feature_name}': '{value}'")
                        extracted_count += 1
            except re.error as e:
                logger.error(f"正则表达式错误，特征 '{feature_name}', 模式 '{pattern}': {e}")
            except Exception as e:
                 logger.error(f"提取标签 '{feature_name}' 时发生意外错误: {e}")
        logger.info(f"显式标签提取完成，共找到 {extracted_count} 个有效值。")

    def _extract_keyword_features(self, text: str, features: CrimeFeatures):
        """使用关键词库和规则提取特征（主要处理布尔和简单类别）。"""
        logger.info("执行关键词和规则提取...")
        keyword_extracted_count = 0
        regex_extracted_count = 0

        # A. 处理关键词匹配 (主要处理布尔型和简单类别，复杂类别由 semantic 处理)
        semantic_features = [
            'perpetrator_occupation_category', 
            'perpetrator_motive', 
            'crime_location_category', 
            'crime_control_method',
            'crime_violated_body_part_category'
        ]
        for feature_name, categories in self.keywords.items():
            if feature_name in semantic_features:
                continue # 跳过由 semantic 方法处理的特征

            sorted_categories = sorted(categories.keys(), key=lambda x: (
                 -1 if feature_name == 'perpetrator_has_prior_record' and x.lower() == '惯犯' else
                 0 if feature_name == 'perpetrator_has_prior_record' and x.lower() == '有' else
                 1 if feature_name == 'perpetrator_has_prior_record' and x.lower() == '无' else
                 2 # 其他默认优先级
            ))

            found_category = False
            for category_value in sorted_categories:
                keywords_list = categories[category_value]
                for keyword in keywords_list:
                    if keyword in text: # 简单包含检查
                        if getattr(features, feature_name, None) is None: # 仅在未设置时设置
                            features.set_feature(feature_name, category_value, overwrite=False)
                            logger.debug(f"  [关键词] 找到 '{feature_name}' -> '{category_value}' (关键词: '{keyword}')")
                            keyword_extracted_count += 1
                            found_category = True
                            break # 跳出关键词列表循环
                if found_category:
                    break # 跳出类别循环

        # B. 处理正则表达式匹配 (例如，显性前科)
        keyword_regex_patterns = self.regex_patterns.get('keywords', {})
        if 'prior_record_explicit' in keyword_regex_patterns:
            pattern = keyword_regex_patterns['prior_record_explicit']
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    current_value = getattr(features, 'perpetrator_has_prior_record', None)
                    if current_value not in ['无', '惯犯']:
                         features.set_feature('perpetrator_has_prior_record', '有', overwrite=False)
                         logger.debug(f"  [正则] 找到显性前科表述，设置 perpetrator_has_prior_record = '有'")
                         regex_extracted_count += 1
            except re.error as e:
                 logger.error(f"前科正则表达式错误: {e}")
            except Exception as e:
                 logger.error(f"处理前科正则时发生意外错误: {e}")

        # C. 处理需要组合判断的特征
        if features.crime_multiple_perpetrators and features.crime_repeated_offenses:
             features.set_feature('crime_multiple_perpetrators_and_repeated', True, overwrite=False)
             logger.debug(f"  [组合] 检测到多人作案且多次作案。")

        logger.info(f"关键词提取完成，找到 {keyword_extracted_count} 个简单特征值。正则规则提取完成，找到 {regex_extracted_count} 个。")

    def _extract_semantic_features(self, text: str, features: CrimeFeatures):
        """使用NLP和语义理解提取更高级的特征"""
        logger.info("执行语义特征提取...")
        
        # 复杂类别特征 (使用关键词+上下文判断)
        complex_features = ['perpetrator_occupation_category', 'perpetrator_motive',
                            'crime_location_category', 'crime_control_method',
                            'crime_violated_body_part_category']
        
        for feature_name in complex_features:
            if getattr(features, feature_name, None) is not None:
                continue  # 已有值，跳过
            
            feature_keywords = self.keywords.get(feature_name, {})
            for category, keywords in feature_keywords.items():
                found_category = False
                for keyword in keywords:
                    if keyword in text and not self._check_negation(keyword, text):
                        features.set_feature(feature_name, category, overwrite=False)
                        logger.debug(f"  [语义] 找到 '{feature_name}' -> '{category}' (关键词: '{keyword}')")
                        found_category = True
                        break
                if found_category:
                    break
        
        # NLP增强的语义特征提取
        if SPACY_AVAILABLE and self.use_nlp:
            try:
                doc = nlp(text)
                
                # 尝试推断性别信息
                if features.victim_gender is None and hasattr(doc, 'ents'):
                    for ent in doc.ents:
                        if ent.label_ == 'PERSON' and ('被害' in text[:ent.start_char+100] or '受害' in text[:ent.start_char+100]):
                            # 分析人名，尝试推断性别
                            if any(word in text[ent.start_char-50:ent.end_char+50] for word in ['男孩', '男童', '男性']):
                                features.set_feature('victim_gender', '男', overwrite=False)
                                logger.debug(f"  [NLP] 从实体 '{ent.text}' 上下文推断被害人性别为男")
                            elif any(word in text[ent.start_char-50:ent.end_char+50] for word in ['女孩', '女童', '女性']):
                                features.set_feature('victim_gender', '女', overwrite=False)
                                logger.debug(f"  [NLP] 从实体 '{ent.text}' 上下文推断被害人性别为女")
                
                # 尝试提取年龄信息
                if features.victim_age is None:
                    age_pattern = r'(\d{1,2})(?:岁|周岁)(?:[的]?(?:男|女)(?:孩|童))?'
                    for match in re.finditer(age_pattern, text):
                        age = int(match.group(1))
                        if 0 < age < 18:  # 合理的未成年人年龄范围
                            features.set_feature('victim_age', age, overwrite=False)
                            logger.debug(f"  [NLP] 从文本提取被害人年龄: {age}岁")
                            break
            except Exception as e:
                logger.warning(f"语义特征提取中NLP处理过程中出错: {e}")

    def _check_negation(self, token, doc):
        """检查是否存在否定词"""
        # 简单检查附近3个词内是否有否定词
        if not hasattr(token, 'i'):
            return False
            
        try:
            for i in range(max(0, token.i - 3), min(token.i + 3, len(doc))):
                element = doc[i]
                # 检查element的类型，确保可以安全访问text属性
                if isinstance(element, (list, tuple)):
                    continue
                if hasattr(element, 'text') and element.text in ['不', '没有', '无', '并非', '并未', '否认']:
                    return True
                    
            # 高级检查 - 使用依存关系分析
            if SPACY_AVAILABLE and hasattr(token, 'dep_') and token.dep_ == 'neg':
                return True
                
            return False
        except Exception as e:
            logger.warning(f"检查否定词时发生错误: {e}")
            return False

    def _analyze_relationship_network(self, text: str, features: CrimeFeatures) -> None:
        """分析被害人和犯罪人之间的关系网络"""
        if not SPACY_AVAILABLE or not self.use_nlp:
            return
            
        # 依赖关系指示词
        dependency_indicators = {
            '有依赖': ['老师', '教练', '班主任', '家长', '亲戚', '监护人', '照顾', '负责', '管理', '师生'],
            '无依赖': ['陌生', '路人', '初次', '不认识', '偶然相遇', '无关系']
        }
        
        try:
            doc = nlp(text)
            if not hasattr(doc, 'sents') or not doc.sents:
                # 如果doc没有sents属性或为空，使用整个文本
                sentences = [text]
            else:
                sentences = [sent.text for sent in doc.sents]
            
            for sent_text in sentences:
                # 检查是否同时包含被害人和犯罪人信息
                has_victim = any(word in sent_text for word in ['被害人', '受害人', '孩子', '儿童', '未成年'])
                has_perpetrator = any(word in sent_text for word in ['犯罪人', '嫌疑人', '被告', '犯罪嫌疑人'])
                
                if has_victim and has_perpetrator:
                    # 检查依赖关系
                    for rel_type, indicators in dependency_indicators.items():
                        if any(ind in sent_text for ind in indicators):
                            if not self._check_negation(sent_text, doc):
                                features.set_feature('victim_relationship_with_perpetrator', rel_type, overwrite=False)
                                logger.debug(f"  [关系网络] 从句子 '{sent_text}' 中提取出关系: '{rel_type}'")
                                return
        except Exception as e:
            logger.warning(f"关系网络分析过程中出错: {e}")

    def _analyze_temporal_causal_chains(self, text: str, features: CrimeFeatures) -> None:
        """分析文本中的时间顺序和因果关系"""
        if not SPACY_AVAILABLE or not self.use_nlp:
            return
            
        # 时间顺序标记词
        temporal_markers = ['首先', '然后', '接着', '之后', '最后', '随即', '随后', '紧接着']
        # 因果关系标记词
        causal_markers = ['因为', '所以', '导致', '致使', '引起', '使得', '由于']
        
        try:
            # 提取行为序列
            doc = nlp(text)
            actions = []
            
            # 获取句子列表
            if not hasattr(doc, 'sents') or not doc.sents:
                # 如果doc没有sents属性或为空，简单按句号分割
                sentences = [s for s in text.split('。') if s.strip()]
            else:
                sentences = list(doc.sents)
            
            # 寻找动作序列
            for i, sent in enumerate(sentences):
                if isinstance(sent, str):
                    sent_text = sent
                    is_str = True
                else:
                    sent_text = sent.text
                    is_str = False
                    
                # 检查是否包含时间标记词
                has_temporal = any(marker in sent_text for marker in temporal_markers)
                # 检查是否包含因果标记词
                has_causal = any(marker in sent_text for marker in causal_markers)
                
                # 简单提取可能的动词
                main_verb = None
                if SPACY_AVAILABLE and self.use_nlp and not is_str:
                    for token in sent:
                        # 检查token是否为有效对象并且具有必要的属性
                        if (not isinstance(token, str) and hasattr(token, 'pos_') and 
                            token.pos_ == 'VERB' and hasattr(token, 'dep_') and 
                            token.dep_ in ['ROOT', 'nsubj', 'dobj']):
                            main_verb = token.text
                            break
                else:
                    # 简单的动词提取逻辑，找出可能的动词
                    for word in sent_text.split():
                        if len(word) > 1 and word[-1] in ['了', '着', '过']:
                            main_verb = word
                            break
                    
                if main_verb:
                    actions.append((i, main_verb, has_temporal, has_causal))
            
            # 分析是否存在多次作案行为
            if len(actions) >= 3 and sum(1 for _, _, has_temp, _ in actions if has_temp) >= 2:
                features.set_feature('crime_repeated_offenses', True, overwrite=False)
                logger.debug(f"  [时序分析] 检测到多个时序标记，推断为多次作案")
            
            # 分析是否存在预谋
            premeditated = False
            for i, (_, verb, _, has_causal) in enumerate(actions):
                if has_causal and i < len(actions) - 1:
                    # 检查前因后果关系
                    premeditated = True
                    break
                
            if premeditated:
                # 预谋通常与机会主义相反
                current_motive = features.perpetrator_motive
                if current_motive is None or current_motive == '机会主义':
                    # 如果发现预谋但当前动机是机会主义或未设置，则修正
                    probable_motives = ['病理驱动', '权利支配', '经济交易', '报复社会']
                    
                    # 尝试从已有文本中找更可能的动机
                    for motive in probable_motives:
                        keywords = self.keywords.get('perpetrator_motive', {}).get(motive, [])
                        if any(keyword in text for keyword in keywords):
                            features.set_feature('perpetrator_motive', motive, overwrite=True)
                            logger.debug(f"  [因果分析] 检测到预谋行为，修正动机从'机会主义'到'{motive}'")
                            break
        except Exception as e:
            logger.warning(f"时间顺序和因果关系分析过程中出错: {e}")

if __name__ == '__main__':
    # 示例用法 (用于测试)
    sample_text = """
    [TIME]2023年5月10日15:00[/TIME]，[PER]张三[/PER]在其家中卧室对邻居8岁男童进行猥亵。
    张三(男，[PERP AGE=45]，职业：小区保安)有猥亵儿童前科记录，属于惯犯。他曾因猥亵罪被判刑。此次作案动机为临时起意，偶发冲动。
    据了解，张三多次作案，并且是单独行动。被害人是一名男童，无智力障碍，但存在肢体不便的情况。
    张三在作案时强迫男孩触摸其下体，并用手机拍摄。
    使用了言语威胁手段，如不准说出去，否则就告诉他爸妈。受害者受到[SEVERITY]中度[/SEVERITY]伤害。
    关系：[RELATIONSHIP]有依赖[/RELATIONSHIP](因为是邻居)。 地点：[LOCATION]常规场所[/LOCATION]。
    """

    extractor = FeatureExtractor()
    extracted_features = extractor.extract_features(sample_text)

    print("\n--- 提取的特征对象 --- ")
    print(extracted_features)

    # 生成并打印特征向量及其名称
    feature_vector = extracted_features.to_feature_vector()
    feature_names = CrimeFeatures.get_feature_names()

    print("\n--- 生成的特征向量 --- ")
    print(feature_vector)
    print(f"向量长度: {len(feature_vector)}")

    print("\n--- 特征向量列名 --- ")
    print(feature_names)
    print(f"列名数量: {len(feature_names)}")

    # 验证长度是否一致
    if len(feature_vector) == len(feature_names):
        print("\n向量长度与列名数量一致。")
    else:
        print("\n警告：向量长度与列名数量不一致！")

    # 可以选择性地打印向量和名称的对应关系
    # print("\n--- 向量与列名对应 ---")
    # for name, val in zip(feature_names, feature_vector):
    #     print(f"{name}: {val}")