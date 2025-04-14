import re
import argparse

class LegalTextCleaner:
    """
    法律文本格式清理工具，专用于处理法律笔录、判决书等文档中的各类格式噪声
    """
    
    def __init__(self):
        # 定义清理规则 - 基于文本清洗.md表11中的规则
        self.rules = [
            {
                'id': 1,
                'name': '删除案号标识符',
                'regex': r'\[\d{4}\]\w+字第\d+号',
                'replace': '',
                'description': '删除形如[2023]刑终字第078号的案号标识'
            },
            {
                'id': 2,
                'name': '提取标准化时间戳',
                'regex': r'(\d{2}时\d{2}分\d{2}秒)\W*[提询记]',
                'replace': r'（保留：\1）',
                'description': '保留时间戳信息但删除提讯记等无关文本'
            },
            {
                'id': 3,
                'name': '合并连续空格',
                'regex': r'[\u3000\x20]{2,}',
                'replace': ' ',
                'description': '将连续的空格（包括全角空格）合并为单个空格'
            },
            {
                'id': 4,
                'name': '清理冗余标点',
                'regex': r'([,。!?])\1+',
                'replace': r'\1',
                'description': '将连续重复的标点符号合并为单个'
            },
            {
                'id': 5,
                'name': '删除噪声符号',
                'regex': r'[●◆■▲]+',
                'replace': '',
                'description': '删除文本中的特殊符号标记'
            },
            {
                'id': 6,
                'name': '身份证号脱敏',
                'regex': r'(\d{4})\d{10}(\d{4})',
                'replace': r'\1**********\2',
                'description': '对18位身份证号进行脱敏处理，仅保留前4位和后4位'
            },
            {
                'id': 7,
                'name': '手机号中间四位脱敏',
                'regex': r'(\d{3})\d{4}(\d{4})',
                'replace': r'\1****\2',
                'description': '对11位手机号进行脱敏处理，中间4位用星号替代'
            },
            # 扩展规则 - 根据文档中提到的格式噪声特性添加
            {
                'id': 8,
                'name': '标准化证人匿名符',
                'regex': r'▲+证人匿名',
                'replace': '[证人匿名]',
                'description': '统一证人匿名标记格式'
            },
            {
                'id': 9,
                'name': '清理文本截断标记',
                'regex': r'\.{3,}|…{2,}',
                'replace': '...',
                'description': '统一文本截断符号'
            },
            {
                'id': 10,
                'name': '整理引号格式',
                'regex': r'[""](.+?)[""]',
                'replace': r'"\1"',
                'description': '统一中英文引号为英文双引号'
            }
        ]
    
    def format_clean(self, text):
        """
        对文本进行格式清洗，处理常见的格式问题，如标点符号、空格、段落等
        
        Args:
            text (str): 待清洗的文本
            
        Returns:
            tuple: (清洗后的文本, 统计信息)
        """
        stats = {
            "noise_chars_removed": 0,
            "format_corrections": 0,
            "redundant_symbols": 0,
            "normalized_punctuation": 0,
            "paragraphs_merged": 0
        }
        
        # 去除控制字符
        original_length = len(text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        stats["noise_chars_removed"] += original_length - len(text)
        
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        stats["format_corrections"] += 1
        
        # 去除重复的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        stats["redundant_symbols"] += 1
        
        # 规范化标点符号
        text = re.sub(r'[．。]+', '。', text)
        text = re.sub(r'[,，]+', '，', text)
        text = re.sub(r'[;；]+', '；', text)
        text = re.sub(r'[?？]+', '？', text)
        text = re.sub(r'[!！]+', '！', text)
        stats["normalized_punctuation"] += 5
        
        # 处理段落之间的标点
        try:
            text = re.sub(r'([。！？\?])[\"\'\)\]]([^\"\')\]])', r'\1\n\2', text)
            text = re.sub(r'([。！？\?])[\"\'\)\]]([^\"\'\)\]\n])', r'\1\n\2', text)
            stats["paragraphs_merged"] += 2
        except Exception as e:
            print(f"段落处理出错: {e}")
        
        # 整理空格
        text = re.sub(r' {2,}', ' ', text)
        stats["redundant_symbols"] += 1
        
        return text, stats
    
    def clean_text(self, text, rule_ids=None, stats=False):
        """
        根据指定的规则清理文本
        
        Args:
            text (str): 待清理的文本
            rule_ids (list, optional): 要应用的规则ID列表，如果为None则应用所有规则
            stats (bool): 是否返回统计信息
            
        Returns:
            str: 清理后的文本
            dict: (可选) 应用的规则及其效果的详细信息
        """
        original_text = text
        results = []
        stats_info = {
            'total_chars': len(text),
            'noise_chars_removed': 0,
            'noise_symbols_count': 0
        }
        
        # 先进行格式清洗
        text, format_stats = self.format_clean(text)
        stats_info['noise_chars_removed'] += format_stats["noise_chars_removed"]
        
        # 统计特殊符号数量
        noise_pattern = r'[●◆■▲]'
        stats_info['noise_symbols_count'] = len(re.findall(noise_pattern, text))
        
        # 如果没有指定规则ID，应用所有规则
        if rule_ids is None:
            rule_ids = [rule['id'] for rule in self.rules]
        
        # 按规则ID排序
        applicable_rules = [rule for rule in self.rules if rule['id'] in rule_ids]
        applicable_rules.sort(key=lambda x: rule_ids.index(x['id']))
        
        # 应用每条规则
        for rule in applicable_rules:
            before_text = text
            text = re.sub(rule['regex'], rule['replace'], text)
            
            # 计算删除的字符数
            if stats:
                chars_removed = len(before_text) - len(text)
                stats_info['noise_chars_removed'] += chars_removed
            
            # 记录此规则的应用效果
            results.append({
                'rule_id': rule['id'],
                'rule_name': rule['name'],
                'text_before': before_text,
                'text_after': text,
                'changes_made': before_text != text,
                'chars_removed': len(before_text) - len(text) if stats else None
            })
        
        if stats:
            return text, {
                'original_text': original_text,
                'cleaned_text': text,
                'rules_applied': results,
                'stats': stats_info
            }
        else:
            return text, {
                'original_text': original_text,
                'cleaned_text': text,
                'rules_applied': results
            }
    
    def get_rule_by_id(self, rule_id):
        """获取指定ID的规则"""
        for rule in self.rules:
            if rule['id'] == rule_id:
                return rule
        return None
    
    def list_rules(self):
        """列出所有可用的规则"""
        return self.rules

def main():
    parser = argparse.ArgumentParser(description='法律文本格式清理工具')
    parser.add_argument('--text', type=str, help='要清理的文本')
    parser.add_argument('--file', type=str, help='要清理的文本文件路径')
    parser.add_argument('--rules', type=str, help='要应用的规则ID，用逗号分隔，例如"1,3,5"')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--list-rules', action='store_true', help='列出所有可用的规则')
    parser.add_argument('--verbose', action='store_true', help='显示详细的处理信息')
    parser.add_argument('--stats', action='store_true', help='显示文本清理统计信息')
    
    args = parser.parse_args()
    cleaner = LegalTextCleaner()
    
    # 列出所有规则
    if args.list_rules:
        print("可用的清理规则：")
        for rule in cleaner.list_rules():
            print(f"规则 {rule['id']}: {rule['name']} - {rule['description']}")
        return
    
    # 获取要清理的文本
    '''with open("proposal.txt", "r", encoding="utf-8") as f:
        text_to_clean = f.read()'''
    text_to_clean = "用户▲▲▲（身份证440311198801011234）于2023-05-01拨打13912345678，称在'金鱼缸'场所遭遇问题。地址：北京朝阳区北苑路！"
    # 处理规则ID
    rule_ids = None
    if args.rules:
        try:
            rule_ids = [int(r) for r in args.rules.split(',')]
        except ValueError:
            print("规则ID必须是整数")
            return
    
    # 清理文本
    cleaned_text, details = cleaner.clean_text(text_to_clean, rule_ids, stats=args.stats)
    
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"清理后的文本已保存到: {args.output}")
        except Exception as e:
            print(f"写入文件失败: {e}")
    else:
        print("清理后的文本:")
        print(cleaned_text)
    
    # 显示统计信息
    if args.stats:
        stats = details['stats']
        print("\n文本清理统计:")
        print(f"原始文本字符数: {stats['total_chars']}")
        print(f"清理后文本字符数: {len(cleaned_text)}")
        print(f"移除干扰字符数: {stats['noise_chars_removed']}")
        print(f"原文本中特殊符号数量: {stats['noise_symbols_count']}")
        print(f"处理效率: 移除了 {stats['noise_chars_removed']/stats['total_chars']*100:.2f}% 的干扰内容")
    
    # 显示详细信息
    if args.verbose:
        print("\n详细处理信息:")
        for result in details['rules_applied']:
            rule = cleaner.get_rule_by_id(result['rule_id'])
            print(f"\n应用规则 {result['rule_id']}: {result['rule_name']}")
            print(f"正则表达式: {rule['regex']}")
            print(f"替换为: {rule['replace']}")
            if result['changes_made']:
                print(f"变更: 是 (移除了 {result.get('chars_removed', 'N/A')} 个字符)")
            else:
                print("变更: 否")

if __name__ == "__main__":
    main()