#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
行为模式聚类模块
根据预定义的簇中心和特征权重，对新的特征向量进行分类。
"""

import numpy as np
from typing import List, Dict, Tuple

class BehaviorClusterer:
    def __init__(self):
        # 特征权重 (基于 表 20)
        # 注意：需要与 feature_extractor.py 中 CrimeFeatures.get_feature_names() 的顺序和名称完全对应
        self.feature_weights = {
            # --- 核心特征 (0.7-1.0) ---
            '与犯罪人关系': 1.0,  # 假设向量中 '无依赖'=0, '有依赖'=1
            '职业': 0.7,        # 需要确认向量化方式 (e.g., one-hot)
            '有否前科': 0.9,    # 假设向量中 '无'=0, '有'=1, '惯犯'=2 (或 one-hot)
            '有否隐形前科': 0.8, # 假设向量中 '无'=0, '有'=1, '惯犯'=2 (或 one-hot)
            '犯罪动机': 0.9,    # 需要确认向量化方式 (one-hot)
            '案发地点': 0.7,    # 需要确认向量化方式 (one-hot)
            '控制手段': 0.8,    # 需要确认向量化方式 (one-hot)
            # --- 重要特征 (0.4-0.6) ---
            '被害人年龄': 0.6,    # 数值
            '是否多次作案': 0.5, # 假设 '否'=0, '是'=1
            # '是否多人被害' (表20有，但表23/24中心值未明确列出，暂不加入或权重为0?) - 假设不加入
            '侵犯部位': 0.4,    # 需要确认向量化方式 (one-hot)
            '有否强迫性互动': 0.4, # 假设 '否'=0, '是'=1
            '是否网络传播淫秽信息': 0.4, # 假设 '否'=0, '是'=1
            # --- 中性特征 (0-0.3) ---
            '是否肢体残缺': 0.2, # 假设 '否'=0, '是'=1
            '是否智力残疾': 0.3, # 假设 '否'=0, '是'=1
            '是否一次多人受害': 0.1, # 假设 '否'=0, '是'=1
            '犯罪人性别': 0.1,    # 假设 '女'=0, '男'=1
            '是否观看色情视频': 0.3, # 假设 '否'=0, '是'=1
            '有否暴露自身生殖器': 0.1, # 假设 '否'=0, '是'=1
            '是否现场拍摄': 0.1,  # 假设 '否'=0, '是'=1
            '案发时间': 0.2,     # 需要确认向量化方式 (e.g., 小时数, 或分类?)
            '持续时长': 0.1,     # 需要确认向量化方式 (e.g., 分钟数, 或分类?)
            # --- 表23/24 中提到但表20未列出的特征 ---
            '是否多人作案': 0.0, # 表23/24有数据, 表20没有, 暂设权重0 (或参照'是否多次作案'?) -> 设为0.5? 谨慎起见设为0
            # '受伤害程度' 在表16但不在表20/23/24，不加入
            # '被害人性别' 在表16但不在表20/23/24，不加入
            # '犯罪人年龄' 在表16但不在表20/23/24，不加入
        }
        
        # 预定义的簇中心 (基于 表 23, 24)
        # !! 极度重要：这里的向量维度和顺序必须与 feature_vector 完全一致 !!
        # !! 这里的数值是示例，需要根据 feature_extractor.py 的具体实现来填充 !!
        # !! 特别是 one-hot 编码的特征，需要展开 !!
        # 假设特征顺序如下 (需要 feature_extractor.py 确认):
        # [关系(依赖=1), 多次作案(是=1), 多人作案(是=1), 
        #  动机(病理,权利,经济,报复,替代,机会), 
        #  地点(常规,公共,公隐,特殊,网络,隐匿), 
        #  控制(物质,暴力,药物,工具,精神威胁,精神洗脑), 
        #  智残(是=1), 互动(是=1), 
        #  ... (其他特征按 feature_weights 顺序添加)]
        # 以下中心值是根据表23/24直接转录，向量长度和具体值需要精确匹配！
        self.centroids = {
            '关系操控型': np.array([
                0.94, 0.91, 0.19,  # 关系, 多次, 多人
                0.03, 0.43, 0.28, 0.11, 0.13, 0.02, # 动机
                0.54, 0.09, 0.06, 0.07, 0.21, 0.03, # 地点
                0.11, 0.10, 0.03, 0.08, 0.37, 0.31, # 控制
                0.04, 0.67 # 智残, 互动
                # ... 其他特征的中心值需要补充或设为平均值/0 ...
            ]),
            '冲动宣泄型': np.array([
                0.11, 0.01, 0.01,  # 关系, 多次, 多人
                0.04, 0.05, 0.03, 0.31, 0.34, 0.23, # 动机
                0.11, 0.34, 0.31, 0.06, 0.08, 0.10, # 地点
                0.37, 0.13, 0.31, 0.07, 0.05, 0.07, # 控制
                0.03, 0.17 # 智残, 互动
                # ... 其他特征的中心值需要补充或设为平均值/0 ...
            ]),
            '情境诱导型': np.array([
                0.17, 0.39, 0.23,  # 关系, 多次, 多人
                0.08, 0.09, 0.32, 0.03, 0.41, 0.07, # 动机
                0.12, 0.13, 0.04, 0.39, 0.28, 0.04, # 地点
                0.04, 0.06, 0.03, 0.32, 0.10, 0.45, # 控制
                0.09, 0.87 # 智残, 互动
                # ... 其他特征的中心值需要补充或设为平均值/0 ...
            ]),
            '病理驱动型': np.array([
                0.08, 0.87, 0.01,  # 关系, 多次, 多人
                0.32, 0.11, 0.09, 0.10, 0.04, 0.34, # 动机
                0.07, 0.32, 0.28, 0.09, 0.04, 0.20, # 地点
                0.06, 0.31, 0.02, 0.29, 0.06, 0.26, # 控制
                0.21, 0.11 # 智残, 互动
                # ... 其他特征的中心值需要补充或设为平均值/0 ...
            ])
        }
        
        # 行为模式名称
        self.pattern_names = list(self.centroids.keys())

    def _get_weights_vector(self, feature_names: List[str]) -> np.ndarray:
        """根据输入的特征名称列表，生成对应的权重向量"""
        weights = []
        for name in feature_names:
            # 处理 one-hot 编码的特征名称 (e.g., "职业_无业", "职业_低接触")
            base_name = name.split('_')[0]
            weights.append(self.feature_weights.get(base_name, 0.0)) # 如果原始特征名在权重字典中，使用其权重；否则为0
        return np.array(weights)

    def _weighted_euclidean_distance(self, point1: np.ndarray, point2: np.ndarray, weights: np.ndarray) -> float:
        """计算加权欧氏距离"""
        # 确保权重向量长度与点向量长度一致
        if len(weights) != len(point1) or len(weights) != len(point2):
             raise ValueError(f"Weight length ({len(weights)}) must match point lengths ({len(point1)}, {len(point2)})")
        
        # 处理可能的 NaN 或 Inf 值，替换为0
        point1 = np.nan_to_num(point1)
        point2 = np.nan_to_num(point2)
        weights = np.nan_to_num(weights)

        # 检查权重是否全为0
        if np.all(weights == 0):
            print("Warning: All feature weights are zero. Distance will be zero.")
            # 可以选择返回 0 或计算无权距离
            # return np.sqrt(np.sum((point1 - point2)**2)) # 无权距离
            return 0.0 # 如果权重全0，距离为0

        # 计算加权距离
        # distance_sq = np.sum(weights * (point1 - point2)**2) # 标准公式
        # 尝试处理权重为0的情况，避免除以0的警告，虽然理论上权重为0的特征不影响距离
        diff_sq = (point1 - point2)**2
        weighted_diff_sq = np.where(weights > 0, weights * diff_sq, 0) # 只计算权重>0的部分
        distance_sq = np.sum(weighted_diff_sq)

        # 防止负数开方 (理论上平方和不会为负，但以防万一)
        if distance_sq < 0:
             print(f"Warning: Negative squared distance ({distance_sq}). Clamping to zero.")
             distance_sq = 0

        return np.sqrt(distance_sq)

    def classify(self, feature_vector: List[float], feature_names: List[str]) -> Tuple[str, float, Dict[str, float]]:
        """
        将输入的特征向量分配给最近的预定义簇中心。

        Args:
            feature_vector (List[float]): 输入的特征向量。
            feature_names (List[str]): 特征向量中每个元素对应的名称列表。

        Returns:
            Tuple[str, float, Dict[str, float]]:
                - 最佳匹配的行为模式名称。
                - 与该模式簇中心的加权欧氏距离 (越小越典型)。
                - 所有模式及其对应距离的字典。
        """
        if not feature_vector or not feature_names:
            raise ValueError("Feature vector and names cannot be empty.")
            
        input_vector = np.array(feature_vector)
        
        # 确保输入向量长度与特征名称列表长度一致
        if len(input_vector) != len(feature_names):
             raise ValueError(f"Feature vector length ({len(input_vector)}) must match feature names length ({len(feature_names)})")

        # 获取与当前特征向量匹配的权重向量
        weights_vector = self._get_weights_vector(feature_names)
        
        distances = {}
        min_distance = float('inf')
        best_match = "未知"

        for name, centroid in self.centroids.items():
            # 确保质心向量与输入向量长度匹配 (!! 非常重要 !!)
            # 如果不匹配，可能需要填充质心或截断输入/权重
            if len(centroid) != len(input_vector):
                # 尝试调整质心长度 (简单策略：用0填充缺失值) - 这非常不理想！
                # 更好的方法是确保 feature_extractor 输出的向量维度固定且与质心一致
                print(f"Warning: Centroid '{name}' length ({len(centroid)}) differs from input vector length ({len(input_vector)}). Trying to pad centroid with zeros.")
                
                # 创建一个与输入向量等长的新质心，用0填充
                adjusted_centroid = np.zeros_like(input_vector)
                common_length = min(len(centroid), len(input_vector))
                adjusted_centroid[:common_length] = centroid[:common_length]
                
                # 或者抛出错误，强制要求维度一致
                # raise ValueError(f"Centroid '{name}' length ({len(centroid)}) must match input vector length ({len(input_vector)})")
                
                current_centroid = adjusted_centroid
            else:
                current_centroid = centroid

            try:
                distance = self._weighted_euclidean_distance(input_vector, current_centroid, weights_vector)
                distances[name] = distance
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            except ValueError as e:
                print(f"Error calculating distance for centroid '{name}': {e}")
                distances[name] = float('inf') # 标记为无效距离

        # 对距离进行排序，找到第二近的作为参考？ (可选)
        # sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        
        # 强度/典型性：目前直接使用最小距离。值越小越典型。
        # 可以考虑归一化，例如除以所有距离的平均值或最大值，但需要更多样本才有意义。
        strength_score = min_distance 

        return best_match, strength_score, distances

# 示例用法 (需要有 feature_extractor.py 生成的向量和名称)
if __name__ == '__main__':
    # 假设 feature_extractor 返回了以下内容 (!!! 仅为示例，需要真实数据 !!!)
    # 必须确保这里的名称、顺序、向量值与 feature_extractor.py 的输出完全一致
    example_feature_names = [
        '与犯罪人关系', '是否多次作案', '是否多人作案', 
        '犯罪动机_病理驱动型', '犯罪动机_权利支配型', '犯罪动机_经济交易型', '犯罪动机_报复社会型', '犯罪动机_替代满足型', '犯罪动机_机会主义型',
        '案发地点_常规场所', '案发地点_公共场所', '案发地点_公共场所隐蔽空间', '案发地点_特殊场所', '案发地点_网络虚拟场所', '案发地点_隐匿场所',
        '控制手段_物质利诱', '控制手段_暴力殴打', '控制手段_药物迷幻', '控制手段_工具捆绑', '控制手段_精神威胁', '控制手段_精神洗脑',
        '是否智力残疾', '有否强迫性互动'
        # ... 添加 feature_weights 中定义的所有其他特征名称 ...
    ]
    # 示例向量 (与上述名称对应, 随机生成, 无实际意义)
    example_vector = np.random.rand(len(example_feature_names)).tolist() 
    # 模拟一个更接近"关系操控型"的向量
    example_vector_rel = [
        1, 1, 0, # 关系=依赖, 多次=是, 多人=否
        0, 1, 0, 0, 0, 0, # 动机=权利
        1, 0, 0, 0, 0, 0, # 地点=常规
        0, 0, 0, 0, 1, 0, # 控制=精神威胁
        0, 1 # 智残=否, 互动=是
        # ... 其他特征值 ...
    ]
    # 填充剩余值以匹配名称列表长度
    example_vector_rel.extend([0.0] * (len(example_feature_names) - len(example_vector_rel)))


    clusterer = BehaviorClusterer()
    
    try:
        # 修正质心长度以匹配示例名称列表 (!! 这只是为了示例能运行 !!)
        target_len = len(example_feature_names)
        for name, centroid in clusterer.centroids.items():
             if len(centroid) != target_len:
                 print(f"Adjusting centroid '{name}' length for example.")
                 adjusted_centroid = np.zeros(target_len)
                 common_length = min(len(centroid), target_len)
                 adjusted_centroid[:common_length] = centroid[:common_length]
                 clusterer.centroids[name] = adjusted_centroid

        print("--- Classifying Random Vector ---")
        pattern, strength, all_distances = clusterer.classify(example_vector, example_feature_names)
        print(f"  模式: {pattern}")
        print(f"  强度 (距离): {strength:.4f}")
        print(f"  各模式距离: {all_distances}")

        print("\n--- Classifying Relationship-Control Vector ---")
        pattern_rel, strength_rel, all_distances_rel = clusterer.classify(example_vector_rel, example_feature_names)
        print(f"  模式: {pattern_rel}")
        print(f"  强度 (距离): {strength_rel:.4f}")
        print(f"  各模式距离: {all_distances_rel}")

    except ValueError as e:
        print(f"Error during classification: {e}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() 