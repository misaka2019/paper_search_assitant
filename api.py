from flask import Flask, request, jsonify, Response
from openai import OpenAI
import json
import os

from typing import List
import logging
import time
from rag import SecurityRAGSystem
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
rag_system = None
client = None

def init_services():
    """初始化RAG系统和OpenAI客户端"""
    global rag_system, client
    api_key = "your_api_key"
    if not api_key:
        raise ValueError("API key is required")
    # 初始化RAG系统
    rag_system = SecurityRAGSystem(api_key)
    papers_dir = "security_papers"
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)
        logger.info(f"Created directory: {papers_dir}")
    else:
        for file in os.listdir(papers_dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(papers_dir, file)
                logger.info(f"Processing: {file}")
                rag_system.add_documents(pdf_path)
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url='http://localhost:11435/v1',  # 使用本地服务器
        api_key="dummy"  # 本地测试使用dummy token
    )
    logger.info("Services initialized successfully")
def format_context(results: List[dict]) -> str:
    """格式化检索结果"""
    context = ""
    for result in results:
        context += f"[Paper: {result['title']}\n"
        context += f"File: {result['file_name']}, Page: {result['page']}\n"
        context += f"Content: {result['text']}\n\n"
    return context

SYSTEM_PROMPT = """你是一个AI安全专家，请基于以下论文内容回答问题：

{context}

请在回答时：
1. 引用具体的论文内容作为支持
2. 如果涉及多个安全问题，请分点说明
3. 如果有相关的解决方案，请一并说明
4. 如果论文中没有直接相关的内容，可以基于专业知识进行补充说明

用户问题：{query}"""

@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completion():
    try:
        data = request.json
        messages = data['messages']
        stream = data.get('stream', False)
        
        # 获取最后一条用户消息
        last_message = messages[-1].get('content', '')
        
        # 使用RAG系统检索相关内容
        results = rag_system.retrieval(last_message, threshold=0.4, topk=5)
        context = format_context(results)
        
        # 构建系统提示词
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context, query=last_message)
        }
        
        # 在消息列表开头插入系统提示词
        messages = [system_message] + messages
        logger.info(f"Messages: {messages}")

        if stream:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=messages,
                stream=True
            )
            
            def generate():
                for chunk in response:
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # 模拟响应
            mock_response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "这是一个模拟的响应，用于测试API是否正常工作。实际部署时请删除这个模拟响应。"
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            return jsonify(mock_response)
            
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }), 500

@app.route('/api/tags', methods=['GET'])
def get_tags():
    """获取标签列表"""
    return jsonify({
        "tags": ["ai-security", "chinese", "english"]
    })

@app.route('/api/version', methods=['GET'])
def get_version():
    """获取API版本信息"""
    return jsonify({
        "version": "1.0.0",
        "build_date": "2024-03-27",
        "model_version": "Qwen2.5-72B-Instruct"
    })

@app.route('/v1/models', methods=['GET'])
def list_models():
    """获取可用模型列表，兼容 OpenAI API"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Qwen",
                "permission": [],
                "root": "Qwen/Qwen2.5-7B-Instruct",
                "parent": None
            }
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": {
            "message": "Not found",
            "type": "invalid_request_error",
            "code": 404
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": {
            "message": "Internal server error",
            "type": "server_error",
            "code": 500
        }
    }), 500

if __name__ == '__main__':
    init_services()
    app.run(host='0.0.0.0', port=11435)