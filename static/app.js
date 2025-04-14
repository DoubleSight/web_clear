/**
 * 法律文本清洗系统 - 前端交互脚本
 * 版本: 1.0
 */

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化TF-IDF设置显示
    toggleTfIdfSettings();
    
    // 注册事件监听器
    setupEventListeners();
    
    // 启动模型状态检查
    startModelStatusCheck();
});

/**
 * 设置所有事件监听器
 */
function setupEventListeners() {
    // 处理模式变化时触发TF-IDF设置显示
    document.querySelectorAll('input[name="processMode"]').forEach(radio => {
        radio.addEventListener('change', toggleTfIdfSettings);
    });
    
    // 控制NLP选项的显示
    const nlpCheckbox = document.getElementById('useNlp');
    if (nlpCheckbox) {
        nlpCheckbox.addEventListener('change', function() {
            const nlpOptions = document.getElementById('nlpOptions');
            if (this.checked) {
                nlpOptions.style.display = 'block';
            } else {
                nlpOptions.style.display = 'none';
            }
        });
    }
    
    // 更新滑块值显示
    const tfIdfSlider = document.getElementById('tfIdfThreshold');
    if (tfIdfSlider) {
        tfIdfSlider.addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = this.value;
        });
    }
    
    // 表单提交处理
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }
}

/**
 * 启动模型状态检查
 */
function startModelStatusCheck() {
    // 立即检查一次状态
    checkModelStatus();
    
    // 每5秒检查一次状态
    setInterval(checkModelStatus, 5000);
}

/**
 * 检查模型加载状态
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/check_model_status');
        const data = await response.json();
        
        updateModelStatusUI(data);
    } catch (error) {
        console.error('检查模型状态出错:', error);
    }
}

/**
 * 更新模型状态UI
 * @param {Object} status - 模型状态信息
 */
function updateModelStatusUI(status) {
    const nlpCheckbox = document.getElementById('useNlp');
    const nlpStatusLabel = document.getElementById('nlpStatusLabel');
    const modelStatusText = document.getElementById('modelStatusText');
    const modelLoadingBar = document.getElementById('modelLoadingBar');
    const modelStatusCard = document.querySelector('.model-status');
    
    // 更新模型加载状态标签
    if (!nlpCheckbox || !nlpStatusLabel || !modelStatusText || !modelLoadingBar || !modelStatusCard) return;
    
    // 移除所有状态类
    modelStatusCard.classList.remove('loaded', 'loading', 'error', 'not-loaded');
    
    if (status.models_loaded) {
        // 模型已加载成功
        nlpCheckbox.disabled = false;
        nlpStatusLabel.innerHTML = '<span class="badge bg-success">模型已加载</span>';
        modelStatusText.innerHTML = '<span class="text-success"><i class="bi bi-check-circle-fill me-1"></i>所有模型已加载完成</span>';
        modelLoadingBar.style.width = '100%';
        modelLoadingBar.className = 'progress-bar bg-success';
        modelStatusCard.classList.add('loaded');
    } else if (status.loading_in_progress) {
        // 模型正在加载中
        nlpCheckbox.disabled = true;
        nlpStatusLabel.innerHTML = '<span class="badge bg-warning model-loading-indicator">正在加载模型...</span>';
        modelStatusText.innerHTML = '<span class="text-info model-loading-indicator"><i class="bi bi-arrow-repeat me-1"></i>正在下载和加载模型，请耐心等待...</span>';
        modelLoadingBar.style.width = '60%';
        modelLoadingBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-info';
        modelStatusCard.classList.add('loading');
    } else if (status.loading_error) {
        // 模型加载出错
        nlpCheckbox.disabled = true;
        nlpStatusLabel.innerHTML = `<span class="badge bg-danger">模型加载失败</span>`;
        modelStatusText.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle-fill me-1"></i>模型加载失败: ${status.loading_error}</span>`;
        modelLoadingBar.style.width = '100%';
        modelLoadingBar.className = 'progress-bar bg-danger';
        modelStatusCard.classList.add('error');
    } else {
        // 模型尚未开始加载
        nlpCheckbox.disabled = true;
        nlpStatusLabel.innerHTML = '<span class="badge bg-secondary">模型未加载</span>';
        modelStatusText.innerHTML = '<span class="text-secondary"><i class="bi bi-clock me-1"></i>模型尚未加载，将在系统后台自动下载</span>';
        modelLoadingBar.style.width = '0%';
        modelLoadingBar.className = 'progress-bar bg-secondary';
        modelStatusCard.classList.add('not-loaded');
    }
}

/**
 * 控制TF-IDF设置区域的显示
 */
function toggleTfIdfSettings() {
    const processMode = document.querySelector('input[name="processMode"]:checked').value;
    const tfIdfSettings = document.getElementById('tfIdfSettings');
    
    // 只有在语义净化或完整处理模式下才显示TF-IDF设置
    if (processMode === 'semantic' || processMode === 'full') {
        tfIdfSettings.style.display = 'block';
    } else {
        tfIdfSettings.style.display = 'none';
    }
}

/**
 * 处理表单提交
 * @param {Event} e - 表单提交事件
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('file');
    const loading = document.querySelector('.loading');
    const resultArea = document.querySelector('.result-area');
    const errorMessage = document.querySelector('.error-message');
    const resultContent = document.getElementById('resultContent');
    const nlpResults = document.querySelector('.nlp-results');
    
    // 验证文件是否选择
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('请选择要上传的文件');
        return;
    }
    
    // 重置显示状态
    loading.style.display = 'block';
    resultArea.style.display = 'none';
    nlpResults.style.display = 'none';
    errorMessage.textContent = '';
    errorMessage.style.display = 'none';
    
    // 准备表单数据
    const formData = prepareFormData(fileInput.files[0]);
    
    try {
        // 发送请求
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // 处理成功的响应
            displayResults(data);
            
            // 显示NLP警告（如果有）
            if (data.nlp_warning) {
                showWarning(data.nlp_warning);
            }
        } else {
            // 处理错误的响应
            showError(data.error || '处理失败');
        }
    } catch (error) {
        showError('上传过程中发生错误');
        console.error('上传错误:', error);
    } finally {
        loading.style.display = 'none';
    }
}

/**
 * 准备表单数据
 * @param {File} file - 上传的文件
 * @returns {FormData} - 准备好的表单数据
 */
function prepareFormData(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // 获取处理模式
    const processMode = document.querySelector('input[name="processMode"]:checked').value;
    formData.append('process_mode', processMode);
    
    // 获取TF-IDF阈值
    const tfIdfThreshold = document.getElementById('tfIdfThreshold').value;
    formData.append('tfidf_threshold', tfIdfThreshold);
    
    // 获取NLP设置
    const useNlp = document.getElementById('useNlp').checked;
    formData.append('use_nlp', useNlp);
    
    if (useNlp) {
        const nlpProcessType = document.querySelector('input[name="nlpProcessType"]:checked').value;
        formData.append('nlp_process_type', nlpProcessType);
    }
    
    return formData;
}

/**
 * 显示处理结果
 * @param {Object} data - 服务器返回的数据
 */
function displayResults(data) {
    const resultArea = document.querySelector('.result-area');
    const resultContent = document.getElementById('resultContent');
    
    // 显示处理后的文本
    resultContent.textContent = data.content;
    
    // 更新处理模式信息
    updateProcessInfo();
    
    // 显示NLP结果（如果有）
    if (data.nlp_results) {
        displayNlpResults(data.nlp_results);
    }
    
    // 显示结果区域并滚动到可见位置
    resultArea.style.display = 'block';
    window.scrollTo({ top: resultArea.offsetTop - 20, behavior: 'smooth' });
}

/**
 * 更新处理模式信息显示
 */
function updateProcessInfo() {
    const processMode = document.querySelector('input[name="processMode"]:checked').value;
    const tfIdfThreshold = document.getElementById('tfIdfThreshold').value;
    const processInfo = document.getElementById('processInfo');
    
    if (processMode === 'full') {
        processInfo.textContent = `完整处理（格式清洗 + 语义净化，TF-IDF阈值: ${tfIdfThreshold}）`;
    } else if (processMode === 'format') {
        processInfo.textContent = '仅格式清洗';
    } else if (processMode === 'semantic') {
        processInfo.textContent = `仅语义净化（TF-IDF阈值: ${tfIdfThreshold}）`;
    }
}

/**
 * 显示NLP分析结果
 * @param {Object} results - NLP分析结果
 */
function displayNlpResults(results) {
    const nlpResults = document.querySelector('.nlp-results');
    const summaryArea = document.getElementById('summaryArea');
    const keywordsArea = document.getElementById('keywordsArea');
    const sentimentArea = document.getElementById('sentimentArea');
    
    // 重置所有区域
    summaryArea.style.display = 'none';
    keywordsArea.style.display = 'none';
    sentimentArea.style.display = 'none';
    
    // 显示摘要
    if (results.summary) {
        document.getElementById('summaryContent').textContent = results.summary;
        summaryArea.style.display = 'block';
    }
    
    // 显示关键词
    if (results.keywords && results.keywords.length > 0) {
        displayKeywords(results.keywords);
    }
    
    // 显示情感分析
    if (results.sentiment) {
        displaySentiment(results.sentiment);
    }
    
    // 显示NLP结果区域
    nlpResults.style.display = 'block';
}

/**
 * 显示关键词
 * @param {Array} keywords - 关键词列表
 */
function displayKeywords(keywords) {
    const keywordsContent = document.getElementById('keywordsContent');
    const keywordsArea = document.getElementById('keywordsArea');
    
    keywordsContent.innerHTML = '';
    
    keywords.forEach(keyword => {
        const badge = document.createElement('span');
        badge.className = 'keyword-badge';
        badge.innerHTML = `<i class="bi bi-hash me-1"></i>${keyword}`;
        keywordsContent.appendChild(badge);
    });
    
    keywordsArea.style.display = 'block';
}

/**
 * 显示情感分析结果
 * @param {Object} sentiment - 情感分析结果
 */
function displaySentiment(sentiment) {
    const { label, score } = sentiment;
    const sentimentLabel = document.getElementById('sentimentLabel');
    const sentimentBar = document.getElementById('sentimentBar');
    const sentimentScore = document.getElementById('sentimentScore');
    const sentimentArea = document.getElementById('sentimentArea');
    const sentimentEmoji = document.getElementById('sentimentEmoji');
    const sentimentNeedle = document.getElementById('sentimentNeedle');
    const sentimentExplanation = document.getElementById('sentimentExplanation');
    
    // 移除旧的类
    sentimentLabel.classList.remove('positive', 'negative');
    sentimentScore.classList.remove('positive', 'negative', 'bg-success', 'bg-danger', 'bg-warning');
    sentimentEmoji.classList.remove('positive', 'negative', 'neutral');
    
    // 设置情感标签和图标
    const isPositive = label === '积极';
    
    // 更新表情图标
    if (isPositive) {
        const emojiType = score > 0.8 ? 'bi-emoji-laughing' : 'bi-emoji-smile';
        sentimentEmoji.innerHTML = `<i class="bi ${emojiType}"></i>`;
        sentimentEmoji.classList.add('positive');
        sentimentLabel.innerHTML = '积极';
        sentimentLabel.classList.add('positive');
        sentimentScore.classList.add('positive');
        sentimentBar.className = 'progress-bar bg-success';
    } else {
        const emojiType = score > 0.8 ? 'bi-emoji-angry' : 'bi-emoji-frown';
        sentimentEmoji.innerHTML = `<i class="bi ${emojiType}"></i>`;
        sentimentEmoji.classList.add('negative');
        sentimentLabel.innerHTML = '消极';
        sentimentLabel.classList.add('negative');
        sentimentScore.classList.add('negative');
        sentimentBar.className = 'progress-bar bg-danger';
    }
    
    // 设置得分和进度条
    const scorePercentage = Math.round(score * 100);
    sentimentBar.style.width = `${scorePercentage}%`;
    sentimentScore.textContent = `${scorePercentage}%`;
    
    // 设置仪表盘指针
    // 将情感值从[0,1]映射到[-90,90]度 (消极→积极)
    const needleDegree = isPositive 
        ? score * 90  // 积极: 0 到 90度
        : -score * 90; // 消极: 0 到 -90度
    
    sentimentNeedle.style.transform = `rotate(${needleDegree}deg) translateX(-50%)`;
    
    // 设置解释文本
    if (isPositive) {
        if (score > 0.8) {
            sentimentExplanation.textContent = '文本表达了明显的积极情绪';
        } else if (score > 0.6) {
            sentimentExplanation.textContent = '文本倾向于表达积极情绪';
        } else {
            sentimentExplanation.textContent = '文本包含轻微的积极情绪';
        }
    } else {
        if (score > 0.8) {
            sentimentExplanation.textContent = '文本表达了明显的消极情绪';
        } else if (score > 0.6) {
            sentimentExplanation.textContent = '文本倾向于表达消极情绪';
        } else {
            sentimentExplanation.textContent = '文本包含轻微的消极情绪';
        }
    }
    
    // 使用动画显示情感区域
    sentimentArea.style.display = 'block';
    sentimentArea.style.opacity = '0';
    sentimentArea.style.transform = 'translateY(20px)';
    
    // 触发重排以使动画生效
    void sentimentArea.offsetWidth;
    
    // 应用过渡动画
    sentimentArea.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    sentimentArea.style.opacity = '1';
    sentimentArea.style.transform = 'translateY(0)';
}

/**
 * 显示错误信息
 * @param {string} message - 错误信息
 */
function showError(message) {
    const errorMessage = document.querySelector('.error-message');
    errorMessage.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i>${message}`;
    errorMessage.className = 'error-message alert alert-danger';
    errorMessage.style.display = 'block';
}

/**
 * 显示警告信息
 * @param {string} message - 警告信息
 */
function showWarning(message) {
    const errorMessage = document.querySelector('.error-message');
    errorMessage.innerHTML = `<i class="bi bi-info-circle-fill me-2"></i>${message}`;
    errorMessage.className = 'error-message alert alert-warning';
    errorMessage.style.display = 'block';
}

/**
 * 下载处理结果
 */
function downloadResult() {
    const content = document.getElementById('resultContent').textContent;
    if (!content.trim()) {
        showError('没有可下载的内容');
        return;
    }
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cleaned_text.txt';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

$(document).ready(function() {
    // 初始化TF-IDF滑块值显示
    $('#tfidfThreshold').on('input', function() {
        $('#tfidfValue').text($(this).val());
    });
    
    // 处理表单提交
    $('#uploadForm').submit(function(e) {
        e.preventDefault();
        
        // 显示加载中的提示
        $('#resultContainer').hide();
        $('#nlpResultsContainer').hide();
        $('#structuredOutputContainer').hide();
        $('#statsContainer').hide();
        $('#loadingIndicator').show();
        
        // 获取表单数据
        var formData = new FormData(this);
        formData.append('processingMode', $('#processingMode').val());
        formData.append('tfidfThreshold', $('#tfidfThreshold').val());
        formData.append('exportFormat', $('#exportFormat').val());
        
        // 发送处理请求
        $.ajax({
            url: '/process',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                // 隐藏加载指示器
                $('#loadingIndicator').hide();
                
                // 更新结果区域
                $('#cleanedText').text(response.cleaned_text);
                $('#resultContainer').show();
                
                // 如果有NLP结果且处理模式为full，显示NLP结果
                if (response.nlp_results && $('#processingMode').val() === 'full') {
                    updateNLPResults(response.nlp_results);
                    $('#nlpResultsContainer').show();
                }
                
                // 如果有结构化分析结果
                if (response.structured_results && 
                   ($('#processingMode').val() === 'structure' || $('#processingMode').val() === 'full')) {
                    updateStructuredResults(response.structured_results);
                    $('#structuredOutputContainer').show();
                }
                
                // 如果有统计信息
                if (response.statistics) {
                    updateStatistics(response.statistics);
                    $('#statsContainer').show();
                }
            },
            error: function(xhr, status, error) {
                $('#loadingIndicator').hide();
                var errorMessage = xhr.responseJSON && xhr.responseJSON.error 
                    ? xhr.responseJSON.error : '处理过程中发生错误，请重试。';
                alert('错误: ' + errorMessage);
            }
        });
    });
    
    // 更新NLP结果区域（保留现有功能）
    function updateNLPResults(nlpResults) {
        // 更新摘要
        if (nlpResults.summary) {
            $('#summaryResult').text(nlpResults.summary);
            $('#summarySection').show();
        } else {
            $('#summarySection').hide();
        }
        
        // 更新关键词
        if (nlpResults.keywords && nlpResults.keywords.length > 0) {
            $('#keywordsResult').empty();
            nlpResults.keywords.forEach(function(keyword) {
                $('#keywordsResult').append('<span class="badge badge-info mr-2 mb-2">' + keyword + '</span>');
            });
            $('#keywordsSection').show();
        } else {
            $('#keywordsSection').hide();
        }
        
        // 更新情感分析
        if (nlpResults.sentiment) {
            var sentimentClass = nlpResults.sentiment.score > 0 ? 'text-success' : 
                                (nlpResults.sentiment.score < 0 ? 'text-danger' : 'text-secondary');
            $('#sentimentResult').html('<span class="' + sentimentClass + '">' + 
                                     nlpResults.sentiment.label + ' (' + 
                                     (nlpResults.sentiment.score * 100).toFixed(1) + '%)</span>');
            $('#sentimentSection').show();
        } else {
            $('#sentimentSection').hide();
        }
    }
    
    // 新增：更新结构化结果区域
    function updateStructuredResults(structuredResults) {
        // 更新标记文本
        if (structuredResults.marked_text) {
            $('#markedTextContent').html(structuredResults.marked_text);
        }
        
        // 更新元素列表
        updateElementsList('timeElements', structuredResults.elements?.time || []);
        updateElementsList('locationElements', structuredResults.elements?.location || []);
        updateElementsList('personElements', structuredResults.elements?.person || []);
        updateElementsList('actionElements', structuredResults.elements?.action || []);
        updateElementsList('toolElements', structuredResults.elements?.tool || []);
        
        // 更新行动链
        if (structuredResults.action_chains && structuredResults.action_chains.length > 0) {
            var chainsHtml = '';
            structuredResults.action_chains.forEach(function(chain, index) {
                chainsHtml += '<div class="card mb-3">';
                chainsHtml += '<div class="card-header">行动链 #' + (index + 1) + '</div>';
                chainsHtml += '<div class="card-body p-0">';
                chainsHtml += '<ul class="list-group list-group-flush">';
                
                chain.forEach(function(step) {
                    chainsHtml += '<li class="list-group-item">';
                    chainsHtml += '<div class="d-flex align-items-center">';
                    chainsHtml += '<div class="mr-3"><i class="fas fa-arrow-right text-primary"></i></div>';
                    chainsHtml += '<div>' + step + '</div>';
                    chainsHtml += '</div></li>';
                });
                
                chainsHtml += '</ul></div></div>';
            });
            $('#actionChains').html(chainsHtml);
        }
        
        // 更新XML输出
        if (structuredResults.xml) {
            $('#xmlOutputContent').text(structuredResults.xml);
        }
    }
    
    // 辅助函数：更新元素列表
    function updateElementsList(elementId, items) {
        var $container = $('#' + elementId);
        $container.empty();
        
        if (items.length === 0) {
            $container.append('<li class="list-group-item text-muted">无数据</li>');
            return;
        }
        
        items.forEach(function(item) {
            var frequency = item.frequency || 1;
            var confidenceClass = '';
            
            if (item.confidence) {
                if (item.confidence >= 0.8) confidenceClass = 'border-success';
                else if (item.confidence >= 0.5) confidenceClass = 'border-warning';
                else confidenceClass = 'border-danger';
            }
            
            var html = '<li class="list-group-item d-flex justify-content-between align-items-center ' + confidenceClass + '">';
            html += item.text || item;
            
            if (frequency > 1) {
                html += '<span class="badge badge-primary badge-pill">' + frequency + '</span>';
            }
            
            html += '</li>';
            $container.append(html);
        });
    }
    
    // 新增：更新统计信息
    function updateStatistics(statistics) {
        // 更新格式清洗统计
        updateStatList('formatStats', statistics.format || {});
        
        // 更新语义净化统计
        updateStatList('semanticStats', statistics.semantic || {});
        
        // 更新结构重组统计
        updateStatList('structureStats', statistics.structure || {});
    }
    
    // 辅助函数：更新统计列表
    function updateStatList(listId, stats) {
        var $list = $('#' + listId);
        $list.empty();
        
        if (Object.keys(stats).length === 0) {
            $list.append('<li class="list-group-item text-muted">无数据</li>');
            return;
        }
        
        for (var key in stats) {
            var value = stats[key];
            var label = getStatLabel(key);
            
            var html = '<li class="list-group-item d-flex justify-content-between align-items-center">';
            html += '<span>' + label + '</span>';
            html += '<span class="badge badge-primary badge-pill">' + value + '</span>';
            html += '</li>';
            
            $list.append(html);
        }
    }
    
    // 辅助函数：获取统计标签
    function getStatLabel(key) {
        var labels = {
            'noise_chars_removed': '移除噪声字符数',
            'format_corrections': '格式修正次数',
            'redundant_symbols': '冗余符号数',
            'normalized_punctuation': '标点规范化数',
            'paragraphs_merged': '段落合并数',
            'removed_sentences': '移除句子数',
            'citation_count': '法律引用数',
            'procedural_phrases': '程序性短语数',
            'noise_ratio': '噪声比例',
            'tf_idf_filtered': 'TF-IDF过滤句子数',
            'elements_identified': '识别元素数',
            'sections_structured': '结构化章节数',
            'xml_elements': 'XML元素数',
            'action_chains': '行动链数',
            'confidence_score': '置信度评分'
        };
        
        return labels[key] || key;
    }
    
    // 监听处理模式变化
    $('#processingMode').change(function() {
        var mode = $(this).val();
        
        // 根据选择的模式显示/隐藏相关选项
        if (mode === 'format') {
            $('#tfidfThreshold').closest('.form-group').parent().hide();
        } else {
            $('#tfidfThreshold').closest('.form-group').parent().show();
        }
    });
    
    // 初始触发一次处理模式变化事件
    $('#processingMode').trigger('change');
}); 