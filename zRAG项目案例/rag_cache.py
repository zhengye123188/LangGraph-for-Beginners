import hashlib
import json
import os

# ======================
# 1. 核心 MD5 工具函数
# ======================
def get_md5(text: str) -> str:
    """计算文本/内容的 MD5 值（作为唯一指纹）"""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

# ======================
# 2. RAG 专用去重管理器
# 功能：自动跳过已经处理过的文档/切片
# ======================
class RAGDuplicateChecker:
    def __init__(self, cache_file="rag_processed_cache.json"):
        self.cache_file = cache_file
        self.processed_md5 = self._load_cache()  # 加载已处理的 MD5 列表

    def _load_cache(self):
        """从文件加载已处理的 MD5 记录"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()

    def _save_cache(self):
        """保存 MD5 记录到文件"""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_md5), f, ensure_ascii=False, indent=2)

    def is_processed(self, content: str) -> bool:
        """检查内容是否已经处理过"""
        md5 = get_md5(content)
        return md5 in self.processed_md5

    def mark_processed(self, content: str):
        """标记内容为已处理"""
        md5 = get_md5(content)
        self.processed_md5.add(md5)
        self._save_cache()