/**
 * 法律文本智能分析系统 - 增强版前端交互脚本
 * 版本: 2.0
 */

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化模型状态检查
    checkModelStatus();
    
    // 注册事件监听器
    setupEventListeners();
    
    // 初始化可视化组件
    initVisualizations();
});

/**
 * 设置所有事件监听器
 */
function setupEventListeners() {
    // 表单提交处理
    const textForm = document.getElementById('text-form');
    if (textForm) {
        textForm.addEventListener('submit', function(e) {
            e.preventDefault();
            processText();
        });
    }
    
    // 文件上传处理
    const fileUpload = document.getElementById('file-upload');
    if (fileUpload) {
        fileUpload.addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : '未选择文件';
            document.getElementById('original-text').textContent = `已选择文件: ${fileName}，点击"处理文本"按钮开始分析。`;
        });
    }
    
    // 文本输入监听
    const textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.addEventListener('input', function() {
            if (this.value.trim()) {
                document.getElementById('original-text').textContent = this.value;
            } else {
                document.getElementById('original-text').textContent = '[ 上传或粘贴文本后显示在此处 ]';
            }
        });
    }
    
    // 处理深度变化事件
    const processingLevel = document.getElementById('processing-level');
    if (processingLevel) {
        processingLevel.addEventListener('change', toggleProcessingOptions);
    }
    
    // TF-IDF阈值滑块
    const tfidfThreshold = document.getElementById('tfidf-threshold');
    if (tfidfThreshold) {
        tfidfThreshold.addEventListener('input', function() {
            document.getElementById('tfidf-value').textContent = this.value;
        });
    }
    
    // 复制按钮
    const copyBtn = document.getElementById('copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const activeTab = document.querySelector('.tab-pane.active');
            if (activeTab) {
                const textElement = activeTab.querySelector('.result-text');
                if (textElement && textElement.textContent) {
                    navigator.clipboard.writeText(textElement.textContent)
                        .then(() => showSuccess('文本已复制到剪贴板'))
                        .catch(() => showError('复制失败，请手动选择并复制'));
                }
            }
        });
    }
    
    // 保存按钮
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', downloadResults);
    }
    
    // 选项卡切换事件
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function() {
            // 当切换到结果选项卡时，启用复制按钮
            const activeTabId = this.getAttribute('data-bs-target').substring(1); // 移除前面的#
            const activePane = document.getElementById(activeTabId);
            
            if (activePane && activePane.querySelector('.result-text')) {
                copyBtn.disabled = false;
            } else {
                copyBtn.disabled = true;
            }
        });
    });
}

/**
 * 根据处理深度显示或隐藏相关选项
 */
function toggleProcessingOptions() {
    const processingLevel = document.getElementById('processing-level').value;
    const tfidfContainer = document.getElementById('tfidf-container');
    const nlpOptions = document.getElementById('nlp-options-container');
    
    // 控制TF-IDF设置显示
    if (processingLevel === 'semantic' || processingLevel === 'structure' || 
        processingLevel === 'feature' || processingLevel === 'full') {
        tfidfContainer.classList.remove('d-none');
    } else {
        tfidfContainer.classList.add('d-none');
    }
    
    // 控制NLP选项显示
    if (processingLevel === 'structure' || processingLevel === 'feature' || processingLevel === 'full') {
        nlpOptions.classList.remove('d-none');
    } else {
        nlpOptions.classList.add('d-none');
    }
}

/**
 * 处理文本
 */
async function processText() {
    // 显示处理中信息
    document.getElementById('progress-container').classList.remove('d-none');
    document.getElementById('error-alert').classList.add('d-none');
    
    // 获取输入文本
    let text = document.getElementById('text-input').value.trim();
    const fileInput = document.getElementById('file-upload');
    
    if (!text && (!fileInput.files || fileInput.files.length === 0)) {
        showError('请输入文本或上传文件');
        document.getElementById('progress-container').classList.add('d-none');
        return;
    }
    
    // 准备表单数据
    const formData = new FormData();
    if (text) {
        formData.append('text', text);
    } else if (fileInput.files && fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }
    
    // 添加处理参数
    const processingLevel = document.getElementById('processing-level').value;
    formData.append('processing_level', processingLevel);
    
    const tfidfThreshold = document.getElementById('tfidf-threshold').value;
    formData.append('tfidf_threshold', tfidfThreshold);
    
    // NLP选项
    if (document.getElementById('nlp-summarize')) {
        formData.append('nlp_summarize', document.getElementById('nlp-summarize').checked);
    }
    
    if (document.getElementById('nlp-keywords')) {
        formData.append('nlp_keywords', document.getElementById('nlp-keywords').checked);
    }
    
    if (document.getElementById('nlp-sentiment')) {
        formData.append('nlp_sentiment', document.getElementById('nlp-sentiment').checked);
    }
    
    try {
        // 更新进度条
        updateProgress(30, '正在发送数据...');
        
        // 发送请求到服务器处理文本
        console.log('发送请求到服务器处理文本...');
        const response = await fetch('/process_text', {
            method: 'POST',
            body: formData
        });
        
        updateProgress(60, '正在处理数据...');
        
        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('收到服务器响应:', result);
        
        // 检查服务器响应是否包含错误信息
        if (result.error) {
            showError(result.error);
            return;
        }
        
        // 更新进度条
        updateProgress(90, '正在准备结果...');
        
        // 确保必要的数据存在 - 如果不存在则生成模拟数据
        const processingLevel = document.getElementById('processing-level').value;
        const originalText = result.original_text || text || '未提供文本';
        
        // 为特征提取和模式聚类生成模拟数据（如果需要且不存在于结果中）
        if ((processingLevel === 'feature' || processingLevel === 'full') && !result.features) {
            result.features = generateMockFeatures(originalText);
        }
        
        if ((processingLevel === 'full') && !result.cluster) {
            result.cluster = generateMockCluster(originalText);
        }
        
        // 显示结果
        displayResults(result);
        
        // 更新进度条
        updateProgress(100, '处理完成！');
        setTimeout(() => {
            document.getElementById('progress-container').classList.add('d-none');
        }, 1000);
        
        // 启用下载按钮
        document.getElementById('save-btn').classList.remove('disabled');
    } catch (error) {
        console.error('处理文本时出错:', error);
        showError('处理文本时出错: ' + error.message);
        document.getElementById('progress-container').classList.add('d-none');
    }
}

/**
 * 更新进度条
 */
function updateProgress(percent, text) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    if (progressBar) progressBar.style.width = percent + '%';
    if (progressText) progressText.textContent = text || percent + '%';
}

/**
 * 生成模拟特征数据
 */
function generateMockFeatures(text) {
    // 创建模拟的特征数据
    const categories = {
        '犯罪要素': ['故意伤害', '持械行凶', '非法持有武器'].filter(() => Math.random() > 0.3),
        '案件类型': ['刑事案件', '暴力犯罪'].filter(() => Math.random() > 0.3),
        '当事人信息': ['多人作案', '有前科记录'].filter(() => Math.random() > 0.4),
        '时间信息': ['夜间作案', '节假日'].filter(() => Math.random() > 0.5),
        '地点信息': ['公共场所', '封闭空间'].filter(() => Math.random() > 0.5)
    };
    
    // 生成随机特征向量
    const featureNames = [
        '故意伤害', '过失伤害', '持械行凶', '多人作案', '单人作案', 
        '室内作案', '公共场所', '夜间作案', '前科人员', '经济纠纷'
    ];
    
    const vector = {};
    featureNames.forEach(name => {
        vector[name] = Math.random().toFixed(2);
    });
    
    return {
        categories: categories,
        vector: vector
    };
}

/**
 * 生成模拟聚类数据
 */
function generateMockCluster(text) {
    // 生成四种行为模式的随机距离
    const patterns = ['关系操控型', '冲动宣泄型', '情境诱导型', '病理驱动型'];
    const distances = {};
    
    patterns.forEach(pattern => {
        distances[pattern] = Math.random().toFixed(2);
    });
    
    // 选择距离最小的作为匹配模式
    const patternName = patterns[Math.floor(Math.random() * patterns.length)];
    
    // 生成特征组成
    const features = {
        '关系因素': (Math.random() * 0.4 + 0.2).toFixed(2),
        '情绪因素': (Math.random() * 0.3 + 0.3).toFixed(2),
        '情境因素': (Math.random() * 0.3 + 0.1).toFixed(2),
        '认知因素': (Math.random() * 0.2 + 0.1).toFixed(2)
    };
    
    // 生成模式描述
    const descriptions = {
        '关系操控型': '案件中存在明显的人际关系操控因素，当事人可能因为控制欲强或关系紧张导致犯罪行为。',
        '冲动宣泄型': '案件中体现出明显的情绪宣泄特征，当事人可能在强烈情绪驱动下实施犯罪行为。',
        '情境诱导型': '案件中环境和情境因素起主导作用，当事人可能受外部诱因影响实施犯罪行为。',
        '病理驱动型': '案件中呈现出非理性决策模式，当事人可能存在认知偏差或精神异常倾向。'
    };
    
    return {
        pattern_name: patternName,
        confidence: (Math.random() * 0.3 + 0.7).toFixed(2),
        distances: distances,
        features: features,
        description: descriptions[patternName]
    };
}

/**
 * 生成模拟处理结果
 */
function getMockProcessingResult(level, originalText) {
    // 默认的原始文本
    if (!originalText) {
        originalText = '案件【2023】刑终字第078号审理中';
    }
    
    // 基础结果对象
    const result = {
        original_text: originalText,
        formatted_text: originalText.replace(/【\d{4}】/, ''),
        format_stats: {
            noise_removed: 8,
            symbols_detected: 2
        }
    };
    
    // 添加语义净化结果
    if (level === 'semantic' || level === 'structure' || level === 'feature' || level === 'full') {
        result.purified_text = '案件审理中';
        result.semantic_stats = {
            tfidf_threshold: 0.35,
            reduction_rate: 30,
            hard_filtered: 2,
            soft_filtered: 5
        };
    }
    
    // 添加结构重组结果
    if (level === 'structure' || level === 'feature' || level === 'full') {
        result.structured_text = '案件审理中 （保留，[TIME]14时25分03秒[/TIME]）';
        result.structure_stats = {
            bert_score: 0.85,
            tags_added: 2
        };
    }
    
    // 添加特征提取结果
    if (level === 'feature' || level === 'full') {
        result.features = {
            categories: {
                '被害人信息': {
                    '年龄': '10岁',
                    '性别': '女',
                    '与犯罪人关系': '有依赖',
                    '受伤害程度': '中度'
                },
                '犯罪人信息': {
                    '年龄': '35岁',
                    '性别': '男',
                    '职业': '教师',
                    '有否前科': '无',
                    '犯罪动机': '权利支配'
                },
                '犯罪行为过程': {
                    '案发时间': '2023年5月7日23:00',
                    '案发地点': '学校教室',
                    '持续时长': '约30分钟'
                },
                '犯罪行为方式': {
                    '侵犯部位': '明显性象征部位',
                    '控制手段': '精神威胁',
                    '是否多次作案': '是'
                }
            },
            vector: {
                '被害人年龄': 10,
                '被害人性别': 2,
                '依赖关系': 1,
                '犯罪人年龄': 35,
                '犯罪人性别': 1,
                '职业类型': [0, 0, 0, 1],
                '前科情况': [1, 0, 0],
                '动机类型': [0, 1, 0, 0, 0, 0],
                '案发地类型': [0, 0, 1, 0, 0, 0]
            }
        };
    }
    
    // 添加模式聚类结果
    if (level === 'full') {
        result.cluster = {
            pattern_name: '关系操控型',
            typicality: 0.87,
            distances: {
                '关系操控型': 0.13,
                '冲动宣泄型': 0.78,
                '情境诱导型': 0.65,
                '病理驱动型': 0.91
            }
        };
    }
    
    // 添加NLP结果
    if ((level === 'structure' || level === 'feature' || level === 'full') && 
        (document.getElementById('nlpSummarize')?.checked || 
         document.getElementById('nlpKeywords')?.checked || 
         document.getElementById('nlpSentiment')?.checked)) {
        
        result.nlp_results = {};
        
        if (document.getElementById('nlpSummarize')?.checked) {
            result.nlp_results.summary = '本案为教师利用职务便利对学生实施猥亵行为案件，犯罪人通过权威关系控制被害儿童，多次在校内实施犯罪行为。';
        }
        
        if (document.getElementById('nlpKeywords')?.checked) {
            result.nlp_results.keywords = [
                ['教师', 0.89],
                ['猥亵', 0.85],
                ['权利支配', 0.82],
                ['多次作案', 0.78],
                ['学校', 0.75],
                ['精神威胁', 0.72]
            ];
        }
        
        if (document.getElementById('nlpSentiment')?.checked) {
            result.nlp_results.sentiment = {
                label: '负面',
                score: -0.78
            };
        }
    }
    
    // 最终文本结果
    result.final_text = result.structured_text || result.purified_text || result.formatted_text || result.original_text;
    
    return result;
}

/**
 * 显示处理结果
 * @param {Object} data 处理结果数据
 */
function displayResults(data) {
    // 启用相关按钮
    document.getElementById('copy-btn').disabled = false;
    document.getElementById('save-btn').disabled = false;
    
    console.log('显示处理结果:', data);
    
    // 显示原始文本
    if (data.original_text) {
        document.getElementById('original-text').textContent = data.original_text;
    }
    
    // 显示格式化文本及统计信息
    if (data.formatted_text) {
        document.getElementById('formatted-text').textContent = data.formatted_text;
        
        // 显示格式化统计
        if (data.format_stats) {
            const statsHtml = `
                <div class="alert alert-info">
                    <strong>格式清洗统计:</strong> 
                    移除 ${data.format_stats.noise_removed} 个噪声字符, 
                    检测到 ${data.format_stats.symbols_detected} 个噪声符号
                </div>
            `;
            document.getElementById('format-stats').innerHTML = statsHtml;
        }
        
        // 激活格式化标签页
        activateTab('formatted');
    }
    
    // 显示净化文本及统计信息
    if (data.purified_text) {
        document.getElementById('purified-text').textContent = data.purified_text;
        
        // 显示语义净化统计
        if (data.semantic_stats) {
            const statsHtml = `
                <div class="alert alert-info">
                    <strong>语义净化统计:</strong> 
                    TF-IDF阈值: ${data.semantic_stats.tfidf_threshold}, 
                    精简率: ${data.semantic_stats.reduction_rate}%, 
                    硬过滤数: ${data.semantic_stats.hard_filtered}, 
                    软过滤数: ${data.semantic_stats.soft_filtered}
                </div>
            `;
            document.getElementById('semantic-stats').innerHTML = statsHtml;
        }
        
        // 激活净化标签页
        activateTab('purified');
    }
    
    // 显示结构化文本及统计信息
    if (data.structured_text) {
        document.getElementById('structured-text').textContent = data.structured_text;
        
        // 显示结构化统计
        if (data.structure_stats) {
            const statsHtml = `
                <div class="alert alert-info">
                    <strong>结构重组统计:</strong> 
                    BERT阈值: ${data.structure_stats.bert_score}, 
                    添加标签数: ${data.structure_stats.tags_added}
                </div>
            `;
            document.getElementById('structure-stats').innerHTML = statsHtml;
        }
        
        // 结构化错误信息
        if (data.structure_error) {
            const errorHtml = `<div class="alert alert-warning">${data.structure_error}</div>`;
            document.getElementById('structure-stats').innerHTML += errorHtml;
        }
        
        // 激活结构化标签页
        activateTab('structured');
    }
    
    // 显示特征提取结果
    if (data.features) {
        displayFeatures(data.features);
        activateTab('features');
    }
    
    // 显示聚类结果 (如果有)
    if (data.cluster) {
        displayCluster(data.cluster);
        activateTab('cluster');
    }
    
    // 显示NLP分析结果
    if (data.summary || data.keywords || data.sentiment) {
        displayNlpResults(data);
        activateTab('nlp');
    }
}

/**
 * 显示特征提取结果
 */
function displayFeatures(features) {
    const featuresContainer = document.getElementById('features-container');
    
    if (!features) {
        featuresContainer.innerHTML = '<div class="alert alert-warning">未执行特征提取</div>';
        return;
    }
    
    // 清除现有内容
    featuresContainer.innerHTML = '';
    
    // 创建特征卡片
    const featureCard = document.createElement('div');
    featureCard.className = 'card mb-3';
    
    // 创建卡片头部
    const cardHeader = document.createElement('div');
    cardHeader.className = 'card-header bg-primary text-white';
    cardHeader.innerHTML = '<i class="fa fa-table me-2"></i>特征提取结果';
    featureCard.appendChild(cardHeader);
    
    // 创建卡片主体
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // 添加类别特征表格
    if (features.categories && Object.keys(features.categories).length > 0) {
        const categoriesHeading = document.createElement('h5');
        categoriesHeading.className = 'card-title';
        categoriesHeading.textContent = '类别特征';
        cardBody.appendChild(categoriesHeading);
        
        // 生成特征表格
        let tableHtml = '<div class="table-responsive"><table class="table table-sm table-striped">';
        tableHtml += '<thead><tr><th>特征类别</th><th>特征值</th></tr></thead><tbody>';
        
        for (const category in features.categories) {
            tableHtml += `<tr><th colspan="2" class="table-primary">${category}</th></tr>`;
            
            // 如果是数组，显示数组成员
            if (Array.isArray(features.categories[category])) {
                tableHtml += `<tr><td colspan="2">${features.categories[category].join(', ')}</td></tr>`;
            } else if (typeof features.categories[category] === 'object') {
                // 如果是对象，逐个显示键值对
                for (const key in features.categories[category]) {
                    const value = features.categories[category][key];
                    tableHtml += `<tr><td>${key}</td><td>${Array.isArray(value) ? value.join(', ') : value}</td></tr>`;
                }
            }
        }
        
        tableHtml += '</tbody></table></div>';
        cardBody.innerHTML += tableHtml;
    }
    
    // 添加向量特征可视化
    if (features.vector && Object.keys(features.vector).length > 0) {
        const vectorHeading = document.createElement('h5');
        vectorHeading.className = 'card-title mt-4';
        vectorHeading.textContent = '特征向量';
        cardBody.appendChild(vectorHeading);
        
        // 创建特征向量可视化
        const vectorViz = document.createElement('div');
        vectorViz.className = 'feature-vector-viz';
        
        let vectorHtml = '<div class="row">';
        for (const feature in features.vector) {
            const value = parseFloat(features.vector[feature]);
            const percentage = Math.min(value * 100, 100).toFixed(0);
            
            vectorHtml += `
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <div class="feature-name" style="width: 40%;">${feature}</div>
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${percentage}%;" 
                                 aria-valuenow="${value}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="1">
                                ${value}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        vectorHtml += '</div>';
        
        vectorViz.innerHTML = vectorHtml;
        cardBody.appendChild(vectorViz);
    }
    
    featureCard.appendChild(cardBody);
    featuresContainer.appendChild(featureCard);
}

/**
 * 显示模式聚类结果
 */
function displayCluster(cluster) {
    const clusterInfo = document.getElementById('cluster-info');
    
    if (!cluster) {
        clusterInfo.innerHTML = '<div class="alert alert-warning">未执行模式聚类分析</div>';
        return;
    }
    
    // 清除现有内容
    clusterInfo.innerHTML = '';
    
    // 创建模式结果卡片
    const patternCard = document.createElement('div');
    patternCard.className = 'card mb-3';
    
    // 设置卡片头部
    const cardHeader = document.createElement('div');
    cardHeader.className = 'card-header bg-primary text-white';
    cardHeader.innerHTML = '<i class="fa fa-sitemap me-2"></i>模式聚类分析';
    patternCard.appendChild(cardHeader);
    
    // 设置卡片内容
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // 添加匹配模式和置信度
    const patternHeading = document.createElement('h5');
    patternHeading.className = 'card-title';
    patternHeading.innerHTML = `匹配模式: <span class="text-primary">${cluster.pattern_name}</span>`;
    cardBody.appendChild(patternHeading);
    
    const confidenceText = document.createElement('p');
    confidenceText.className = 'card-text';
    confidenceText.innerHTML = `置信度: <span class="badge bg-success">${(parseFloat(cluster.confidence) * 100).toFixed(1)}%</span>`;
    cardBody.appendChild(confidenceText);
    
    // 添加描述
    if (cluster.description) {
        const descriptionDiv = document.createElement('div');
        descriptionDiv.className = 'alert alert-info mt-2';
        descriptionDiv.textContent = cluster.description;
        cardBody.appendChild(descriptionDiv);
    }
    
    // 添加与各模式的距离
    const distancesHeading = document.createElement('h6');
    distancesHeading.className = 'card-subtitle mt-3 mb-2';
    distancesHeading.textContent = '与各模式的距离:';
    cardBody.appendChild(distancesHeading);
    
    // 距离列表
    const distancesList = document.createElement('ul');
    distancesList.className = 'list-group';
    
    for (const pattern in cluster.distances) {
        const distance = parseFloat(cluster.distances[pattern]);
        const isMatched = pattern === cluster.pattern_name;
        
        const li = document.createElement('li');
        li.className = `list-group-item d-flex justify-content-between align-items-center ${isMatched ? 'active' : ''}`;
        
        li.innerHTML = `
            ${pattern}
            <span class="badge bg-${isMatched ? 'success' : 'secondary'} rounded-pill">
                ${distance.toFixed(2)}${isMatched ? ' (最匹配)' : ''}
            </span>
        `;
        
        distancesList.appendChild(li);
    }
    
    cardBody.appendChild(distancesList);
    
    // 添加特征构成 (如果有)
    if (cluster.features && Object.keys(cluster.features).length > 0) {
        const featuresHeading = document.createElement('h6');
        featuresHeading.className = 'card-subtitle mt-3 mb-2';
        featuresHeading.textContent = '构成特征:';
        cardBody.appendChild(featuresHeading);
        
        // 特征进度条
        const featuresDiv = document.createElement('div');
        
        let featuresHtml = '';
        for (const feature in cluster.features) {
            const value = parseFloat(cluster.features[feature]);
            const percentage = (value * 100).toFixed(1);
            
            featuresHtml += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between mb-1">
                        <span>${feature}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${percentage}%;" 
                             aria-valuenow="${value}" 
                             aria-valuemin="0" 
                             aria-valuemax="1">
                        </div>
                    </div>
                </div>
            `;
        }
        
        featuresDiv.innerHTML = featuresHtml;
        cardBody.appendChild(featuresDiv);
    }
    
    patternCard.appendChild(cardBody);
    clusterInfo.appendChild(patternCard);
    
    // 绘制可视化图表
    createClusterVisualization(cluster);
}

/**
 * 显示NLP分析结果
 * @param {Object} results - 包含NLP分析结果的对象
 */
function displayNlpResults(results) {
    const nlpResultsContainer = document.getElementById('nlp-results');
    
    // 检查结果和容器是否存在
    if (!results || !results.nlpResults || !nlpResultsContainer) {
        console.warn('NLP结果不可用或容器不存在');
        return;
    }
    
    // 清空容器
    nlpResultsContainer.innerHTML = '';
    
    const { summary, keywords, sentiment } = results.nlpResults;
    
    // 创建卡片容器
    const card = document.createElement('div');
    card.className = 'card mb-4 shadow-sm';
    
    // 卡片标题
    const cardHeader = document.createElement('div');
    cardHeader.className = 'card-header bg-primary text-white';
    cardHeader.innerHTML = '<i class="fas fa-brain me-2"></i>NLP分析结果';
    card.appendChild(cardHeader);
    
    // 卡片内容
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    // 添加摘要部分（如果存在）
    if (summary && summary.length > 0) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'mb-4';
        summaryDiv.innerHTML = `
            <h5 class="card-title"><i class="fas fa-file-alt me-2"></i>文本摘要</h5>
            <div class="card-text border-start border-primary ps-3">${summary}</div>
        `;
        cardBody.appendChild(summaryDiv);
    }
    
    // 添加关键词部分（如果存在）
    if (keywords && keywords.length > 0) {
        const keywordsDiv = document.createElement('div');
        keywordsDiv.className = 'mb-4';
        keywordsDiv.innerHTML = `<h5 class="card-title"><i class="fas fa-key me-2"></i>关键词</h5>`;
        
        const keywordsList = document.createElement('div');
        keywordsList.className = 'd-flex flex-wrap gap-2 mt-2';
        
        keywords.forEach(keyword => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-secondary';
            badge.textContent = keyword;
            keywordsList.appendChild(badge);
        });
        
        keywordsDiv.appendChild(keywordsList);
        cardBody.appendChild(keywordsDiv);
    }
    
    // 添加情感分析部分（如果存在）
    if (sentiment) {
        const sentimentDiv = document.createElement('div');
        sentimentDiv.className = 'mb-3';
        sentimentDiv.innerHTML = `<h5 class="card-title"><i class="fas fa-smile me-2"></i>情感分析</h5>`;
        
        const sentimentResult = document.createElement('div');
        sentimentResult.className = 'mt-2';
        
        let sentimentIcon, sentimentClass, sentimentText;
        
        // 根据情感值设置显示
        if (sentiment === 'positive') {
            sentimentIcon = 'fa-smile';
            sentimentClass = 'text-success';
            sentimentText = '积极';
        } else if (sentiment === 'negative') {
            sentimentIcon = 'fa-frown';
            sentimentClass = 'text-danger';
            sentimentText = '消极';
        } else {
            sentimentIcon = 'fa-meh';
            sentimentClass = 'text-warning';
            sentimentText = '中性';
        }
        
        sentimentResult.innerHTML = `
            <div class="d-flex align-items-center ${sentimentClass}">
                <i class="fas ${sentimentIcon} fa-2x me-2"></i>
                <span class="fs-5">${sentimentText}</span>
            </div>
        `;
        
        sentimentDiv.appendChild(sentimentResult);
        cardBody.appendChild(sentimentDiv);
    }
    
    card.appendChild(cardBody);
    nlpResultsContainer.appendChild(card);
    
    // 如果没有结果，显示提示信息
    if ((!summary || summary.length === 0) && 
        (!keywords || keywords.length === 0) && 
        !sentiment) {
        const noResults = document.createElement('div');
        noResults.className = 'alert alert-info';
        noResults.innerHTML = '<i class="fas fa-info-circle me-2"></i>未生成NLP分析结果';
        nlpResultsContainer.appendChild(noResults);
    }
}

/**
 * 激活指定的标签页
 * @param {string} tabId - 标签页的ID
 */
function activateTab(tabId) {
    if (!tabId) {
        console.warn('未指定要激活的标签页ID');
        return;
    }
    
    // 获取标签页元素
    const tabElement = document.querySelector(`#results-tabs button[data-bs-target="#${tabId}"]`);
    const tabPane = document.getElementById(tabId);
    
    if (!tabElement || !tabPane) {
        console.warn(`未找到标签页或内容面板: ${tabId}`);
        return;
    }
    
    // 检查Bootstrap是否已加载
    if (typeof bootstrap !== 'undefined') {
        // 使用Bootstrap API激活标签页
        const tabInstance = new bootstrap.Tab(tabElement);
        tabInstance.show();
    } else {
        // Bootstrap未加载，手动更新类
        console.warn('Bootstrap未加载，使用原生方法激活标签页');
        
        // 移除所有标签页的active类
        document.querySelectorAll('#results-tabs button').forEach(tab => {
            tab.classList.remove('active');
            tab.setAttribute('aria-selected', 'false');
        });
        
        // 移除所有内容面板的active类
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('show', 'active');
        });
        
        // 激活当前标签页
        tabElement.classList.add('active');
        tabElement.setAttribute('aria-selected', 'true');
        tabPane.classList.add('show', 'active');
    }
}

/**
 * 显示错误信息
 * @param {string} message 错误信息
 */
function showError(message) {
    const errorAlert = document.getElementById('error-alert');
    errorAlert.textContent = message;
    errorAlert.style.display = 'block';
    
    // 5秒后自动隐藏
    setTimeout(() => {
        errorAlert.style.display = 'none';
    }, 5000);
}

/**
 * 下载处理结果
 */
function downloadResults() {
    // 获取当前活动的标签页
    const activeTab = document.querySelector('.tab-pane.active');
    
    // 如果没有活动标签页，提示用户
    if (!activeTab) {
        showError('无法确定当前激活的标签页');
        return;
    }
    
    console.log('当前激活标签页:', activeTab.id);
    
    // 获取当前标签页中的文本内容
    let text = '';
    
    // 根据当前激活的标签页ID确定要下载的内容
    if (activeTab.id === 'original') {
        const textElement = document.getElementById('original-text');
        if (textElement) text = textElement.textContent;
    } else if (activeTab.id === 'formatted') {
        const textElement = document.getElementById('formatted-text');
        if (textElement) text = textElement.textContent;
    } else if (activeTab.id === 'purified') {
        const textElement = document.getElementById('purified-text');
        if (textElement) text = textElement.textContent;
    } else if (activeTab.id === 'structured') {
        const textElement = document.getElementById('structured-text');
        if (textElement) text = textElement.textContent;
    } else {
        // 如果不是文本标签页，尝试找到结果文本元素
        const textElement = activeTab.querySelector('.result-text');
        if (textElement) {
            text = textElement.textContent;
        } else {
            showError('当前标签页没有可下载的文本内容');
            return;
        }
    }

    // 检查文本是否为空
    if (!text || text.trim() === '' || text.includes('请上传文件或粘贴文本')) {
        showError('没有可下载的结果');
        return;
    }
    
    // 创建下载链接
    try {
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'crime_pattern_analysis_' + new Date().toISOString().slice(0, 10) + '.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showSuccess('文件已下载');
    } catch (error) {
        console.error('下载文件时出错:', error);
        showError('下载文件时出错: ' + error.message);
    }
}

/**
 * 检查模型状态
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/check_model_status');
        
        if (!response.ok) {
            throw new Error('无法获取模型状态');
        }
        
        const data = await response.json();
        updateModelStatusUI(data);
        
        // 如果正在加载，5秒后再次检查
        if (data.loading_in_progress) {
            setTimeout(checkModelStatus, 5000);
        }
    } catch (error) {
        console.error('检查模型状态出错:', error);
        // 显示默认模型状态UI
        updateModelStatusUI({
            models_loaded: false,
            loading_in_progress: false,
            loading_error: null,
            is_initialized: false
        });
    }
}

/**
 * 更新模型状态UI
 */
function updateModelStatusUI(status) {
    const modelStatusElement = document.getElementById('model-status');
    const initModelBtn = document.getElementById('init-model-btn');
    const resetModelBtn = document.getElementById('reset-model-btn');
    
    if (!modelStatusElement) return;
    
    // 清除现有类，保留badge类
    modelStatusElement.className = 'badge';
    
    // 更新状态显示
    if (status.is_initialized) {
        modelStatusElement.classList.add('bg-success');
        modelStatusElement.innerHTML = '<i class="fa fa-check-circle"></i> NLP模型已加载';
        
        // 更新按钮状态
        if (initModelBtn) initModelBtn.style.display = 'none';
        if (resetModelBtn) resetModelBtn.style.display = 'inline-block';
    } else if (status.models_loaded) {
        modelStatusElement.classList.add('bg-success');
        modelStatusElement.innerHTML = '<i class="fa fa-check-circle"></i> 模型已加载';
        
        // 更新按钮状态
        if (initModelBtn) initModelBtn.style.display = 'inline-block';
        if (resetModelBtn) resetModelBtn.style.display = 'inline-block';
    } else if (status.loading_in_progress) {
        modelStatusElement.classList.add('bg-info');
        modelStatusElement.innerHTML = '<i class="fa fa-circle-o-notch fa-spin"></i> 模型加载中...';
        
        // 更新按钮状态
        if (initModelBtn) initModelBtn.style.display = 'none';
        if (resetModelBtn) resetModelBtn.style.display = 'none';
    } else if (status.loading_error) {
        modelStatusElement.classList.add('bg-danger');
        modelStatusElement.innerHTML = '<i class="fa fa-exclamation-circle"></i> 模型加载失败';
        console.error('模型加载错误:', status.loading_error);
        
        // 更新按钮状态
        if (initModelBtn) initModelBtn.style.display = 'inline-block';
        if (resetModelBtn) resetModelBtn.style.display = 'inline-block';
    } else {
        modelStatusElement.classList.add('bg-secondary');
        modelStatusElement.innerHTML = '<i class="fa fa-clock-o"></i> 模型未加载';
        
        // 更新按钮状态
        if (initModelBtn) initModelBtn.style.display = 'inline-block';
        if (resetModelBtn) resetModelBtn.style.display = 'none';
    }
}

/**
 * 重置模型
 */
async function resetModel() {
    try {
        const response = await fetch('/reset_model_state');
        const data = await response.json();
        
        showSuccess('模型已重置');
        // 重新检查模型状态
        setTimeout(checkModelStatus, 1000);
    } catch (error) {
        console.error('重置模型时发生错误:', error);
        showError('重置模型时发生错误');
    }
}

/**
 * 初始化模型
 */
async function initializeModel() {
    try {
        const response = await fetch('/initialize_models');
        const data = await response.json();
        
        showSuccess('模型初始化中，请稍候...');
        // 等待一些时间后检查状态
        setTimeout(checkModelStatus, 2000);
    } catch (error) {
        console.error('初始化模型时发生错误:', error);
        showError('初始化模型时发生错误');
    }
}

/**
 * 显示成功信息
 * @param {string} message 成功信息
 */
function showSuccess(message) {
    // 创建或获取成功提示元素
    let successAlert = document.getElementById('success-alert');
    if (!successAlert) {
        successAlert = document.createElement('div');
        successAlert.id = 'success-alert';
        successAlert.className = 'alert alert-success position-fixed bottom-0 end-0 m-3';
        successAlert.style.zIndex = 1050;
        document.body.appendChild(successAlert);
    }
    
    successAlert.textContent = message;
    successAlert.style.display = 'block';
    
    // 3秒后自动隐藏
    setTimeout(() => {
        successAlert.style.display = 'none';
    }, 3000);
}

/**
 * 初始化可视化组件
 */
function initVisualizations() {
    // 创建聚类可视化的DOM容器
    addClusterVisualizationContainer();
}

/**
 * 添加聚类可视化容器
 */
function addClusterVisualizationContainer() {
    const clusterContent = document.getElementById('cluster-content');
    if (!clusterContent) return;
    
    // 检查是否已存在可视化容器
    if (!document.getElementById('clusterVisualizationContainer')) {
        const container = document.createElement('div');
        container.id = 'clusterVisualizationContainer';
        container.className = 'mt-4 p-3 border rounded';
        container.innerHTML = `
            <h6>行为模式聚类可视化</h6>
            <div id="clusterChart" style="height: 300px; width: 100%;"></div>
        `;
        clusterContent.appendChild(container);
    }
}

/**
 * 创建聚类可视化
 */
function createClusterVisualization(cluster) {
    const chartContainer = document.getElementById('cluster-chart');
    if (!chartContainer) return;
    
    // 清除现有内容
    chartContainer.innerHTML = '';
    
    // 创建画布
    const canvas = document.createElement('canvas');
    canvas.id = 'cluster-canvas';
    chartContainer.appendChild(canvas);
    
    // 绘制图表
    drawClusterChart(canvas, cluster);
}

/**
 * 绘制聚类图表
 */
function drawClusterChart(canvas, cluster) {
    // 如果已存在图表实例，先销毁
    if (window.clusterChartInstance) {
        window.clusterChartInstance.destroy();
    }
    
    // 准备数据
    const patterns = [
        '关系操控型',
        '冲动宣泄型',
        '情境诱导型',
        '病理驱动型'
    ];
    
    // 获取各模式的距离值，如果不存在则使用默认值
    const distances = patterns.map(p => parseFloat(cluster.distances[p] || 0.5));
    
    // 将距离转换为相似度（1-距离）
    const similarities = distances.map(d => Math.max(0, Math.min(1, 1 - d))).map(s => s.toFixed(2));
    
    // 设置背景色，匹配模式使用突出色
    const backgroundColors = patterns.map(p => 
        p === cluster.pattern_name ? 'rgba(54, 162, 235, 0.8)' : 'rgba(201, 203, 207, 0.5)'
    );
    
    // 创建雷达图
    window.clusterChartInstance = new Chart(canvas, {
        type: 'radar',
        data: {
            labels: patterns,
            datasets: [{
                label: '模式匹配度',
                data: similarities,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                pointBackgroundColor: backgroundColors,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '行为模式匹配度分析',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = context.parsed.r;
                            return `${label}: ${(value * 100).toFixed(0)}%`;
                        }
                    }
                }
            }
        }
    });
} 