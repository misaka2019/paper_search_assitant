import PyPDF2
import faiss
import numpy as np
import requests
from typing import List
import os
import logging
import time
import re
logger = logging.getLogger(__name__)

class SecurityRAGSystem:
    def __init__(self, api_key: str="your_api_key"):
        self.dimension = 1024
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
        
        self.embed_url = "https://api.siliconflow.cn/v1/embeddings"
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def encode_text(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        # 每批处理的文本数量
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            payload = {
                "model": "BAAI/bge-large-en-v1.5",
                "input": batch_texts,
                "encoding_format": "float"
            }
            
            try:
                response = requests.post(
                    self.embed_url, 
                    json=payload, 
                    headers=self.headers
                )
                response.raise_for_status()
                embeddings = response.json()['data']
                all_embeddings.extend([emb['embedding'] for emb in embeddings])
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                return None
        
        return np.array(all_embeddings) if all_embeddings else None

    def read_pdf(self, pdf_path: str) -> List[dict]:
        """读取PDF文件并按段落切分文本
        
        Args:
            pdf_path: PDF文件路径
        Returns:
            包含文本段落和元数据的字典列表
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        chunks_with_metadata = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                logger.info(f"Successfully opened PDF: {pdf_path} with {len(reader.pages)} pages")
                
                # 提取标题（从第一页）
                first_page = reader.pages[0].extract_text()
                title = self._extract_title(first_page)
                logger.info(f"Extracted title: {title}")
                
                # 处理每一页
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    logger.debug(f"Processing page {page_num}, raw text length: {len(text)}")
                    
                    # 清理文本
                    text = self._clean_text(text)
                    
                    # 按段落分割
                    paragraphs = self._split_into_paragraphs(text)
                    logger.debug(f"Page {page_num}: Split into {len(paragraphs)} paragraphs")
                    
                    # 处理每个段落
                    valid_paragraphs = 0
                    for para in paragraphs:
                        # 过滤无效段落
                        if self._is_valid_paragraph(para):
                            valid_paragraphs += 1
                            chunks_with_metadata.append({
                                'text': para
                            })
                    logger.debug(f"Page {page_num}: Found {valid_paragraphs} valid paragraphs")
                    
            logger.info(f"Extracted {len(chunks_with_metadata)} valid paragraphs from {pdf_path}")
            return chunks_with_metadata
                    
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return []

    def _extract_title(self, first_page_text: str) -> str:
        """从第一页提取论文标题"""
        lines = first_page_text.split('\n')
        for line in lines[:3]:  # 通常标题在前三行
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # 标题长度的合理范围
                return line
        return "Unknown Title"

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 替换多余的空白字符
        text = ' '.join(text.split())
        # 替换特殊破折号
        text = text.replace('–', '-').replace('—', '-')
        # 规范化空格和换行
        text = text.replace('\r', '\n')
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 首先按多个换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 清理每个段落
        cleaned_paragraphs = []
        for para in paragraphs:
            # 清理段落中的多余空白
            para = ' '.join(para.split())
            if para:  # 只保留非空段落
                # 如果段落太长，按句号分割
                if len(para) > 1000:
                    sentences = re.split(r'(?<=[.!?。！？])\s+', para)
                    current_chunk = []
                    current_length = 0
                    for sentence in sentences:
                        if current_length + len(sentence) > 1000:
                            if current_chunk:
                                cleaned_paragraphs.append(' '.join(current_chunk))
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                    if current_chunk:
                        cleaned_paragraphs.append(' '.join(current_chunk))
                else:
                    cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs

    def _is_valid_paragraph(self, para: str) -> bool:
        """检查段落是否有效"""
        # 移除空白字符
        para = para.strip()
        
        # 检查长度（限制最大长度以避免API限制）
        if len(para) < 50 or len(para) > 1000:
            return False
        
        # 检查是否包含足够的单词
        words = para.split()
        if len(words) < 10:
            return False
        
        # 检查是否是参考文献
        if para.lower().startswith(('references', 'bibliography', '参考文献')):
            return False
        
        # 检查是否是图表标题
        if re.match(r'^(figure|table|fig\.|图|表)\s*\d+', para.lower()):
            return False
        
        return True

    def _is_paragraph_end(self, text: str) -> bool:
        """检查文本是否是段落的结束"""
        # 检查是否以句号等标点符号结尾
        if any(text.endswith(end) for end in ['.', '。', '!', '?', '！', '？']):
            return True
        
        # 检查是否是小节标题（通常以数字开头）
        if re.match(r'^\d+\..*$', text):
            return True
        
        return False

    def add_documents(self, pdf_path: str):
        chunks_with_metadata = self.read_pdf(pdf_path)
        if not chunks_with_metadata:
            logger.warning(f"No valid text extracted from {pdf_path}")
            return
            
        texts = [chunk['text'] for chunk in chunks_with_metadata]
        vectors = self.encode_text(texts)
        
        if vectors is not None:
            self.texts.extend(texts)
            self.metadata.extend(chunks_with_metadata)
            self.index.add(vectors)
            logger.info(f"Added {len(chunks_with_metadata)} chunks from {pdf_path}")

    def retrieval(self, query: str, threshold: float = 0.8, topk: int = 5) -> List[dict]:
        query_vector = self.encode_text(query)
        if query_vector is None:
            return []
            
        distances, indices = self.index.search(query_vector, topk)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance < threshold:
                results.append({
                    'text': self.texts[idx],
                    'distance': float(distance)
                })
        logger.info(f"Retrieved {len(results)} results for query '{query}'")
        return results
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    pdf_path = "security_papers/1504.00680v2.pdf"
    rag = SecurityRAGSystem()
    rag.add_documents(pdf_path)
    results = rag.retrieval("What are the main security threats in online communities?")
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']:.3f}")
        print(f"Text: {result['text'][:200]}...")