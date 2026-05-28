"""
InfoSeek → MRAG 多选题转换器

功能:
- 将 InfoSeek 开放式问题转换为 MRAG 格式（四选一）
- 使用 LLM 生成答案和干扰项
- SQLite 缓存避免重复 API 调用
- 支持批量处理、错误恢复、进度追踪

使用示例:
    converter = InfoSeekConverter(cache_db="/tmp/infoseek_cache.db")
    
    # 方式 1: 单条转换
    result = converter.convert_single(
        question="What place inflows lake?",
        image_id="oven_05494604"
    )
    print(result)  # {"answer": "...", "distractors": ["...", "...", "..."], ...}
    
    # 方式 2: 批量转换
    converter.batch_convert(
        input_jsonl="data/infoseek/Entity/infoseek_test.jsonl",
        output_jsonl="data/converted_test.jsonl",
        max_samples=100,
        batch_size=10
    )
"""

import json
import sqlite3
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """转换结果数据类"""
    data_id: str
    image_id: str
    question: str
    answer: str
    distractors: List[str]  # [B, C, D] 三个干扰项
    confidence: float
    llm_model: str
    is_cached: bool
    conversion_time: float
    
    def to_mrag_format(self) -> Dict:
        """转换为 MRAG-Bench 格式"""
        return {
            "data_id": self.data_id,
            "image_id": self.image_id,
            "question": self.question,
            "options": {
                "A": self.answer,
                "B": self.distractors[0],
                "C": self.distractors[1],
                "D": self.distractors[2],
            },
            "correct": "A",
            "confidence": self.confidence,
            "metadata": {
                "llm_model": self.llm_model,
                "is_cached": self.is_cached,
                "conversion_time": self.conversion_time,
            }
        }
    
    def to_dict(self) -> Dict:
        return asdict(self)


class InfoSeekConverter:
    """InfoSeek 开放式问题 → MRAG 多选题转换器"""
    
    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        cache_db: str = "/tmp/infoseek_cache.db",
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        """
        初始化转换器
        
        Args:
            llm_model: 使用的 LLM 模型（'gpt-3.5-turbo', 'gpt-4', 或 'local'）
            cache_db: SQLite 缓存数据库路径
            api_key: OpenAI API key（可选，会从环境变量读取）
            cache_enabled: 是否启用缓存
        """
        self.llm_model = llm_model
        self.cache_db = cache_db
        self.cache_enabled = cache_enabled
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        
        # 初始化缓存
        if cache_enabled:
            self._init_cache()
    
    def _init_cache(self):
        """初始化 SQLite 缓存数据库"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversion_cache (
                    question_hash TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    result JSON NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"缓存数据库初始化: {self.cache_db}")
        except Exception as e:
            logger.error(f"缓存初始化失败: {e}")
            raise
    
    def _get_cache(self, question: str) -> Optional[Dict]:
        """从缓存读取转换结果"""
        if not self.cache_enabled:
            return None
        
        try:
            q_hash = hashlib.md5(question.encode()).hexdigest()
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT result FROM conversion_cache WHERE question_hash = ? AND model = ?",
                (q_hash, self.llm_model)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                logger.debug(f"缓存命中: {question[:50]}...")
                return json.loads(row[0])
            return None
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
            return None
    
    def _set_cache(self, question: str, result: Dict):
        """存储转换结果到缓存"""
        if not self.cache_enabled:
            return
        
        try:
            q_hash = hashlib.md5(question.encode()).hexdigest()
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO conversion_cache 
                (question_hash, question, result, model) 
                VALUES (?, ?, ?, ?)
                """,
                (q_hash, question, json.dumps(result), self.llm_model)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """
        调用 LLM 生成答案和干扰项
        
        Args:
            question: InfoSeek 问题
            
        Returns:
            {"answer": "...", "distractors": ["...", "...", "..."], "confidence": 0.0-1.0}
        """
        if self.llm_model.startswith("gpt"):
            return self._call_openai_api(question)
        elif self.llm_model == "local":
            return self._call_local_model(question)
        else:
            raise ValueError(f"不支持的模型: {self.llm_model}")
    
    def _call_openai_api(self, question: str) -> Dict[str, Any]:
        """调用 OpenAI API"""
        try:
            import openai
            
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY 未设置")
            
            openai.api_key = self.api_key
            
            prompt = f"""你是一个多选题生成专家。给定一个问题，你需要：
1. 给出一个正确答案
2. 生成 3 个合理的干扰项（错误但看起来合理的答案）
3. 评估你的信心度（0-1，其中 1 表示完全确定）

问题: {question}

请返回 JSON 格式（必须是有效的 JSON，不要包含其他文本）:
{{
    "answer": "正确答案",
    "distractors": ["干扰项1", "干扰项2", "干扰项3"],
    "confidence": 0.85,
    "reasoning": "简要说明"
}}
"""
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates multiple-choice questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=500,
                timeout=30
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            # 如果返回的是 markdown 代码块，先提取内容
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # 验证必要字段
            required_fields = ["answer", "distractors", "confidence"]
            if not all(f in result for f in required_fields):
                raise ValueError(f"缺少必要字段: {set(required_fields) - set(result.keys())}")
            
            if len(result["distractors"]) != 3:
                raise ValueError(f"干扰项数量错误: 期望 3，得到 {len(result['distractors'])}")
            
            if not (0 <= result["confidence"] <= 1):
                logger.warning(f"信心度范围错误: {result['confidence']}, 强制设为 0.5")
                result["confidence"] = 0.5
            
            return result
        
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n内容: {content}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            raise
    
    def _call_local_model(self, question: str) -> Dict[str, Any]:
        """
        调用本地模型（示例实现）
        实际部署时可替换为 Ollama/LLaMA 等
        """
        logger.warning("本地模型调用为示例实现，返回随机结果")
        # 这是一个占位符实现
        # 实际可使用 ollama, transformers 等库
        return {
            "answer": "sample answer",
            "distractors": ["distractor 1", "distractor 2", "distractor 3"],
            "confidence": 0.5,
            "reasoning": "Using local model (placeholder)"
        }
    
    def convert_single(
        self,
        question: str,
        data_id: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> ConversionResult:
        """
        转换单个问题
        
        Args:
            question: InfoSeek 问题
            data_id: 数据 ID
            image_id: 图像 ID
            
        Returns:
            ConversionResult 对象
        """
        data_id = data_id or f"converted_{hash(question)}"
        image_id = image_id or "unknown"
        
        start_time = time.time()
        
        # 检查缓存
        cached_result = self._get_cache(question)
        is_cached = cached_result is not None
        
        if is_cached:
            llm_result = cached_result
        else:
            logger.info(f"调用 LLM 转换: {question[:60]}...")
            llm_result = self._call_llm(question)
            self._set_cache(question, llm_result)
        
        elapsed = time.time() - start_time
        
        return ConversionResult(
            data_id=data_id,
            image_id=image_id,
            question=question,
            answer=llm_result["answer"],
            distractors=llm_result["distractors"],
            confidence=llm_result.get("confidence", 0.5),
            llm_model=self.llm_model,
            is_cached=is_cached,
            conversion_time=elapsed
        )
    
    def batch_convert(
        self,
        input_jsonl: str,
        output_jsonl: str,
        max_samples: Optional[int] = None,
        batch_size: int = 10,
        skip_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        批量转换 InfoSeek 问题
        
        Args:
            input_jsonl: 输入 JSONL 文件路径
            output_jsonl: 输出 JSONL 文件路径
            max_samples: 最多转换样本数（None=全部）
            batch_size: 批处理大小（用于进度汇报）
            skip_errors: 是否跳过错误样本继续处理
            
        Returns:
            统计信息字典
        """
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "cached": 0,
            "avg_confidence": 0.0,
            "errors": []
        }
        
        input_path = Path(input_jsonl)
        output_path = Path(output_jsonl)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        confidences = []
        
        try:
            with open(input_path) as fin, open(output_path, 'w') as fout:
                for idx, line in enumerate(fin):
                    if max_samples and idx >= max_samples:
                        break
                    
                    try:
                        record = json.loads(line)
                        stats["total"] += 1
                        
                        # 转换
                        result = self.convert_single(
                            question=record["question"],
                            data_id=record.get("data_id"),
                            image_id=record.get("image_id")
                        )
                        
                        # 写出
                        mrag_format = result.to_mrag_format()
                        fout.write(json.dumps(mrag_format, ensure_ascii=False) + "\n")
                        
                        stats["success"] += 1
                        if result.is_cached:
                            stats["cached"] += 1
                        confidences.append(result.confidence)
                        
                        # 进度输出
                        if (idx + 1) % batch_size == 0:
                            avg_conf = sum(confidences) / len(confidences)
                            logger.info(
                                f"进度: {idx+1}/{max_samples or '?'} | "
                                f"成功: {stats['success']} | "
                                f"缓存命中: {stats['cached']} | "
                                f"平均信心度: {avg_conf:.2f}"
                            )
                    
                    except Exception as e:
                        stats["failed"] += 1
                        stats["errors"].append({
                            "idx": idx,
                            "record": record,
                            "error": str(e)
                        })
                        logger.error(f"第 {idx} 行转换失败: {e}")
                        
                        if not skip_errors:
                            raise
        
        except Exception as e:
            logger.error(f"批量转换中止: {e}")
            raise
        
        # 最终统计
        if confidences:
            stats["avg_confidence"] = sum(confidences) / len(confidences)
        
        logger.info(
            f"\n批量转换完成:\n"
            f"  总计: {stats['total']}\n"
            f"  成功: {stats['success']}\n"
            f"  失败: {stats['failed']}\n"
            f"  缓存命中: {stats['cached']}\n"
            f"  平均信心度: {stats['avg_confidence']:.2f}\n"
            f"  输出: {output_jsonl}"
        )
        
        return stats


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="InfoSeek 开放式问题转换为 MRAG 多选题")
    parser.add_argument("input", help="输入 JSONL 文件")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 文件")
    parser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="LLM 模型")
    parser.add_argument("-c", "--cache", default="/tmp/infoseek_cache.db", help="缓存数据库路径")
    parser.add_argument("-n", "--max-samples", type=int, help="最多转换样本数")
    parser.add_argument("-b", "--batch-size", type=int, default=10, help="批处理大小")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--skip-errors", action="store_true", default=True, help="跳过错误继续处理")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = InfoSeekConverter(
        llm_model=args.model,
        cache_db=args.cache,
        cache_enabled=not args.no_cache,
    )
    
    # 批量转换
    stats = converter.batch_convert(
        input_jsonl=args.input,
        output_jsonl=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        skip_errors=args.skip_errors,
    )
    
    # 输出统计
    print("\n" + "="*60)
    print("转换统计")
    print("="*60)
    for key, value in stats.items():
        if key != "errors":
            print(f"{key:20s}: {value}")
