# AI Security Paper Search Assistant

这是一个基于RAG (Retrieval-Augmented Generation) 的论文搜索助手，专注于AI安全领域的论文检索和问答。系统使用通义千问大模型作为基础，并结合了本地知识库来提供更准确的回答。

## 功能特点

- 支持PDF论文的自动导入和解析
- 基于语义的相似度搜索
- 支持流式输出
- 提供OpenAI兼容的API接口
- 包含完整的API调试工具

## 系统架构

系统主要包含以下组件：

1. **RAG系统** (`rag.py`)
   - PDF文档处理
   - 文本分块
   - 向量化和检索
   - 上下文组装

2. **API服务** (`api.py`)
   - OpenAI兼容的接口
   - 流式响应支持
   - 错误处理
   - 系统提示词管理

3. **调试工具** (`debug_api.py`)
   - API健康检查
   - 端点测试
   - 响应验证
   - 错误诊断

## 快速开始

### 环境要求

- Python 3.10+
- PyPDF2
- FAISS
- Flask
- OpenAI Python SDK

### API密钥获取

在开始使用前，请先获取API密钥：
1. 访问 https://cloud.siliconflow.cn/account/ak
2. 注册/登录账号
3. 获取免费的API密钥（新用户可获得2000万tokens）

### 安装

```bash
# 克隆仓库
git clone [repository-url]
cd paper-search-assistant

# 安装依赖
pip install -r requirements.txt
```

### 使用方法

1. **启动服务器**
```bash
python api.py
```

2. **添加论文**
- 将PDF论文放入 `security_papers` 目录
- 服务器启动时会自动处理新添加的论文

3. **测试API**
```bash
python debug_api.py
```

## API 接口

### 1. 聊天补全 `/v1/chat/completions`

```python
# 示例请求
curl -X POST "http://localhost:11435/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dummy" \
     -d '{
       "model": "Qwen/Qwen2.5-7B-Instruct",
       "messages": [{"role": "user", "content": "介绍AI模型的安全威胁"}],
       "stream": false
     }'
```

### 2. 获取标签 `/api/tags`

```python
# 示例请求
curl "http://localhost:11435/api/tags"
```

### 3. 获取版本信息 `/api/version`

```python
# 示例请求
curl "http://localhost:11435/api/version"
```

### 4. 获取模型列表 `/v1/models`

```python
# 示例请求
curl "http://localhost:11435/v1/models" \
     -H "Authorization: Bearer dummy"
```

## 调试指南

系统提供了全面的调试工具，可以测试所有API端点：

```bash
python debug_api.py
```

调试工具会检查：
- 服务器状态
- API可用性
- 标签系统
- 模型列表
- 普通聊天补全
- 流式聊天补全

## 配置说明

主要配置项在 `api.py` 中：

```python
# API配置
base_url = "http://localhost:11435/v1"
api_key = "dummy"  # 本地测试用

# RAG系统配置
papers_dir = "security_papers"  # PDF文件目录
threshold = 0.4  # 相似度阈值
topk = 5  # 检索结果数量
```

## 错误处理

系统实现了完整的错误处理机制：
- API错误返回标准的HTTP状态码
- 详细的错误信息记录
- 自动重试机制
- 超时保护

## 注意事项

1. 本系统设计用于本地部署，不建议直接暴露到公网
2. API密钥在本地测试时使用"dummy"
3. 确保PDF文件格式规范，便于系统解析
4. 建议定期清理和更新文档库

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。在提交代码前，请：
1. 运行完整的调试测试
2. 确保代码符合PEP 8规范
3. 更新相关文档

## 许可证

[添加许可证信息] 