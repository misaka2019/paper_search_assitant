import requests
import json
import logging
from openai import OpenAI
import os
import httpx
import time

# 禁用代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalAPIDebugger:
    def __init__(self):
        self.base_url = "http://localhost:11435"
        self.api_key = "dummy"  # 本地测试使用dummy作为API key
        
        # 创建不使用代理的httpx客户端
        transport = httpx.HTTPTransport(proxy=None)
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            http_client=httpx.Client(transport=transport, timeout=10.0)  # 设置10秒超时
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def test_server_status(self):
        """测试服务器是否在运行"""
        try:
            response = requests.get(f"{self.base_url}/api/version", proxies={'http': None, 'https': None}, timeout=5)
            logger.info(f"Server status code: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Server version info: {response.json()}")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.error("Connection failed: Server might not be running")
            return False
        except Exception as e:
            logger.error(f"Server status test failed: {str(e)}")
            return False

    def test_available_tags(self):
        """测试获取可用标签"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", proxies={'http': None, 'https': None}, timeout=5)
            logger.info(f"Tags API status code: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Available tags: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Tags API test failed: {str(e)}")
            return False

    def test_chat_completion(self, stream=False):
        """测试聊天补全功能"""
        try:
            messages = [{"role": "user", "content": "介绍AI模型的安全威胁"}]
            
            if stream:
                # 使用requests直接测试流式响应
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "messages": messages,
                        "stream": True
                    },
                    stream=True,
                    proxies={'http': None, 'https': None},
                    timeout=5
                )
                
                if response.status_code != 200:
                    logger.error(f"Stream request failed with status code: {response.status_code}")
                    return False
                
                logger.info("Testing streaming response:")
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content')
                                if content:
                                    logger.info(f"Chunk content: {content}")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse chunk: {data}")
                return True
            else:
                response = self.client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=messages,
                    stream=False
                )
                logger.info("Testing normal response:")
                logger.info(f"Response: {response.choices[0].message.content}")
                return True
        except Exception as e:
            logger.error(f"Chat completion test failed: {str(e)}")
            return False

    def test_models_endpoint(self):
        """测试获取可用模型列表"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", headers=self.headers, proxies={'http': None, 'https': None}, timeout=5)
            logger.info(f"Models API status code: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Available models: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Models API test failed: {str(e)}")
            return False

def run_all_tests():
    """运行所有测试"""
    debugger = LocalAPIDebugger()
    
    logger.info("=== Starting Local API Debug Tests ===")
    
    # 测试服务器状态
    logger.info("\n1. Testing Server Status")
    server_success = debugger.test_server_status()
    logger.info(f"Server status test {'passed' if server_success else 'failed'}")
    
    if not server_success:
        logger.error("Server is not running. Please start the server first.")
        return
    
    # 测试获取标签
    logger.info("\n2. Testing Tags API")
    tags_success = debugger.test_available_tags()
    logger.info(f"Tags API test {'passed' if tags_success else 'failed'}")
    
    # 测试获取模型列表
    logger.info("\n3. Testing Models API")
    models_success = debugger.test_models_endpoint()
    logger.info(f"Models API test {'passed' if models_success else 'failed'}")
    
    # 测试普通聊天补全
    logger.info("\n4. Testing Normal Chat Completion")
    chat_success = debugger.test_chat_completion(stream=False)
    logger.info(f"Normal chat completion test {'passed' if chat_success else 'failed'}")
    
    # 测试流式聊天补全
    logger.info("\n5. Testing Streaming Chat Completion")
    stream_success = debugger.test_chat_completion(stream=True)
    logger.info(f"Streaming chat completion test {'passed' if stream_success else 'failed'}")
    
    # 总结测试结果
    logger.info("\n=== Test Summary ===")
    all_tests = [
        ("Server Status", server_success),
        ("Tags API", tags_success),
        ("Models API", models_success),
        ("Normal Chat Completion", chat_success),
        ("Streaming Chat Completion", stream_success)
    ]
    
    for test_name, success in all_tests:
        logger.info(f"{test_name}: {'✓' if success else '✗'}")

if __name__ == "__main__":
    run_all_tests() 