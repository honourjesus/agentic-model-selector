"""
Benchmarking Agent - Tests model performance with real inference
"""

import torch
import time
import asyncio
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    pipeline,
    AutoModelForCausalLM
)
from typing import List, Dict, Any, Optional

from src.models.schemas import (
    TaskType, UserRequirements, BenchmarkResult
)


class BenchmarkingAgent:
    """
    Benchmarks models with real inference tests to measure latency and memory usage.
    
    This agent loads each model, runs warmup inferences, then measures
    performance over multiple iterations.
    """
    
    def __init__(self, sample_data: Dict[str, Any] = None):
        self.sample_data = sample_data or self._get_default_samples()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")
    
    async def benchmark_models(self, 
                              model_ids: List[str], 
                              task_type: TaskType,
                              requirements: UserRequirements,
                              max_models: int = 3) -> List[BenchmarkResult]:
        """
        Benchmark top models with quick inference tests.
        
        Args:
            model_ids: List of model IDs to benchmark
            task_type: Type of ML task
            requirements: User requirements
            max_models: Maximum number of models to benchmark
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        print(f"   Benchmarking up to {max_models} models...")
        
        for i, model_id in enumerate(model_ids[:max_models]):
            print(f"    Testing {i+1}/{min(len(model_ids), max_models)}: {model_id}")
            
            try:
                result = await self._benchmark_single_model(
                    model_id, task_type, requirements
                )
                results.append(result)
                
                if not result.error:
                    print(f"       Latency: {result.latency_ms:.2f}ms, "
                          f"Memory: {result.memory_usage_mb:.2f}MB")
                else:
                    print(f"       Error: {result.error}")
                    
            except Exception as e:
                print(f"       Failed: {e}")
                # FIXED: Added task_type to the error response
                results.append(BenchmarkResult(
                    model_id=model_id,
                    task_type=task_type,  # This was missing!
                    latency_ms=0,
                    memory_usage_mb=0,
                    error=str(e)
                ))
            
            # Small delay between models
            await asyncio.sleep(0.5)
        
        return results
    
    async def _benchmark_single_model(self, 
                                model_id: str,
                                task_type: TaskType,
                                requirements: UserRequirements) -> BenchmarkResult:
        """Benchmark a single model"""
        
        model = None
        tokenizer = None
        nlp_pipeline = None
        
        # Load model and tokenizer
        try:
            print(f"      Loading model...")
            
            if task_type == TaskType.TRANSLATION:
                # For translation models, we need to use pipeline with specific task format
                try:
                    # Try the standard translation pipeline first
                    nlp_pipeline = pipeline(
                        "translation",
                        model=model_id,
                        device=self.device
                    )
                except Exception as e:
                    # If that fails, try with specific language pair format
                    if requirements.translation_reqs:
                        src = requirements.translation_reqs.source_language.value
                        tgt = requirements.translation_reqs.target_language.value
                        task_name = f"translation_{src}_to_{tgt}"
                        try:
                            nlp_pipeline = pipeline(
                                task_name,
                                model=model_id,
                                device=self.device
                            )
                        except:
                            # If both fail, try loading as a general seq2seq model
                            from transformers import AutoModelForSeq2SeqLM
                            tokenizer = AutoTokenizer.from_pretrained(model_id)
                            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                
            elif task_type in [TaskType.TEXT_CLASSIFICATION, TaskType.NAMED_ENTITY_RECOGNITION]:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                
            elif task_type == TaskType.TEXT_GENERATION:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                
            else:
                # Use pipeline for other tasks
                nlp_pipeline = pipeline(
                    task_type.value,
                    model=model_id,
                    device=self.device
                )
                
        except Exception as e:
            return BenchmarkResult(
                model_id=model_id,
                task_type=task_type,
                latency_ms=0,
                memory_usage_mb=0,
                error=f"Failed to load model: {str(e)}"
            )
        
        # Move model to device
        if model:
            model.to(self.device)
            model.eval()
        
        # Get appropriate sample data
        sample = self._get_task_sample(task_type)
        
        # Run warmup (first inference is always slower)
        try:
            await self._run_warmup(model_id, task_type, sample, model, tokenizer, nlp_pipeline, requirements)
        except Exception as e:
            print(f"      Warmup warning: {e}")
        
        # Benchmark inference
        latencies = []
        memory_usage = []
        
        for i in range(5):  # Run 5 iterations for stable measurement
            # Reset memory stats if using CUDA
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            
            # Run inference
            try:
                with torch.no_grad():
                    if task_type == TaskType.TRANSLATION and nlp_pipeline:
                        # For translation pipeline
                        result = nlp_pipeline(sample["text"], max_length=128)
                        
                    elif task_type == TaskType.TRANSLATION and model and tokenizer:
                        # For seq2seq model
                        inputs = tokenizer(
                            sample["text"], 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=128
                        ).to(self.device)
                        outputs = model.generate(**inputs, max_new_tokens=50)
                        
                    elif task_type == TaskType.TEXT_CLASSIFICATION and model and tokenizer:
                        inputs = tokenizer(
                            sample["text"], 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=128
                        ).to(self.device)
                        outputs = model(**inputs)
                        
                    elif task_type == TaskType.TEXT_GENERATION and model and tokenizer:
                        inputs = tokenizer(
                            sample["text"], 
                            return_tensors="pt", 
                            truncation=True
                        ).to(self.device)
                        outputs = model.generate(**inputs, max_new_tokens=20)
                        
                    elif nlp_pipeline:
                        result = nlp_pipeline(sample["text"])
                        
            except Exception as e:
                return BenchmarkResult(
                    model_id=model_id,
                    task_type=task_type,
                    latency_ms=0,
                    memory_usage_mb=0,
                    error=f"Inference failed: {str(e)}"
                )
            
            end_time = time.perf_counter()
            
            # Measure memory
            if self.device.type == "cuda":
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - start_memory) / (1024 ** 2)  # Convert to MB
                memory_usage.append(memory_used)
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Small delay between runs
            await asyncio.sleep(0.1)
        
        # Clean up
        if model:
            del model
        if tokenizer:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate statistics
        avg_latency = float(np.mean(latencies))
        avg_memory = float(np.mean(memory_usage)) if memory_usage else 0
        
        return BenchmarkResult(
            model_id=model_id,
            task_type=task_type,
            latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            throughput=1000 / avg_latency if avg_latency > 0 else 0
        )
    
    async def _run_warmup(self, model_id: str, task_type: TaskType, sample: Dict,
                     model=None, tokenizer=None, nlp_pipeline=None, requirements=None):
        """Run warmup inference to initialize model"""
        try:
            with torch.no_grad():
                if task_type == TaskType.TRANSLATION and nlp_pipeline:
                    nlp_pipeline(sample["text"], max_length=50)
                    
                elif task_type == TaskType.TRANSLATION and model and tokenizer:
                    inputs = tokenizer(
                        sample["text"], 
                        return_tensors="pt", 
                        truncation=True
                    ).to(self.device)
                    model.generate(**inputs, max_new_tokens=20)
                    
                elif task_type == TaskType.TEXT_CLASSIFICATION and model and tokenizer:
                    inputs = tokenizer(
                        sample["text"], 
                        return_tensors="pt", 
                        truncation=True
                    ).to(self.device)
                    model(**inputs)
                    
                elif task_type == TaskType.TEXT_GENERATION and model and tokenizer:
                    inputs = tokenizer(
                        sample["text"], 
                        return_tensors="pt", 
                        truncation=True
                    ).to(self.device)
                    model.generate(**inputs, max_new_tokens=10)
                    
                elif nlp_pipeline:
                    nlp_pipeline(sample["text"])
                    
        except Exception as e:
            raise e
    
    def _get_task_sample(self, task_type: TaskType) -> Dict[str, Any]:
        """Get sample data for benchmarking"""
        samples = {
            TaskType.TEXT_CLASSIFICATION: {
                "text": "This is a sample text for classification benchmarking."
            },
            TaskType.TEXT_GENERATION: {
                "text": "Once upon a time in a land far away",
            },
            TaskType.SUMMARIZATION: {
                "text": """Artificial intelligence is transforming industries across the globe. 
                From healthcare to finance, AI systems are being deployed to solve complex problems. 
                Machine learning algorithms can now diagnose diseases, predict market trends, 
                and even create art. The rapid advancement of AI technology brings both opportunities 
                and challenges that society must address."""
            },
            TaskType.QUESTION_ANSWERING: {
                "context": "The Eiffel Tower is located in Paris, France.",
                "question": "Where is the Eiffel Tower?"
            },
            TaskType.TRANSLATION: {
                "text": "Hello, how are you today?"
            },
            TaskType.TEXT_TO_SPEECH: {
                "text": "Hello, this is a test of the text to speech system."
            },
            TaskType.SPEECH_TO_TEXT: {
                "text": "This is a sample audio transcription test."
            },
            TaskType.OCR: {
                "text": "Sample text from an image."
            }
        }
        
        return samples.get(task_type, {"text": "Sample text for benchmarking."})
    
    def _get_default_samples(self) -> Dict[str, Any]:
        """Get default sample data for various tasks"""
        return {
            "text_classification": [
                {"text": "I love this product, it's amazing!", "label": "positive"},
                {"text": "This is the worst experience ever.", "label": "negative"}
            ],
            "summarization": [
                {"text": "Long article about AI advancements..."}
            ],
            "translation": [
                {"text": "Hello world", "source_lang": "en", "target_lang": "fr"}
            ]
        }