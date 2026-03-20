"""
Research Agent - Searches HuggingFace for relevant models
"""

import asyncio
import aiohttp
import re
from datetime import datetime
from typing import List, Optional
from huggingface_hub import HfApi

from src.models.schemas import UserRequirements, ModelMetadata, TaskType


class ResearchAgent:
    """
    Searches HuggingFace Hub for models matching user requirements.
    
    This agent queries the HuggingFace API, fetches model details,
    and creates ModelMetadata objects for candidate models.
    """
    
    # Pipeline tags for different tasks
    PIPELINE_TAGS = {
        # Translation
        TaskType.TRANSLATION: "translation",
        
        # Speech
        TaskType.TEXT_TO_SPEECH: "text-to-speech",
        TaskType.SPEECH_TO_TEXT: "automatic-speech-recognition",
        
        # LLMs
        TaskType.TEXT_GENERATION: "text-generation",
        TaskType.CHAT: "text-generation",
        TaskType.INSTRUCTION_FOLLOWING: "text-generation",
        TaskType.CODE_GENERATION: "text-generation",
        TaskType.QUESTION_ANSWERING: "question-answering",
        TaskType.SUMMARIZATION: "summarization",
        
        # OCR
        TaskType.OCR: "image-to-text",
        TaskType.DOCUMENT_UNDERSTANDING: "document-question-answering",
        
        #Legacy
        TaskType.TEXT_CLASSIFICATION: "text-classification",
        TaskType.NAMED_ENTITY_RECOGNITION: "token-classification",
        TaskType.IMAGE_CLASSIFICATION: "image-classification",
        TaskType.OBJECT_DETECTION: "object-detection",
        TaskType.ZERO_SHOT_CLASSIFICATION: "zero-shot-classification"
    }
    
    def __init__(self):
        self.api = HfApi()
        
    async def search_models(self, requirements: UserRequirements, top_k: int = 20) -> List[ModelMetadata]:
        """
        Search for models matching the requirements.
        
        Args:
            requirements: UserRequirements object
            top_k: Maximum number of models to return
            
        Returns:
            List of ModelMetadata objects
        """
        task = requirements.task_type
        pipeline_tag = self.PIPELINE_TAGS.get(task)
        
        print(f"   Searching for {task.value} models (pipeline: {pipeline_tag})...")
        
        try:
            models = []
            
            # Method 1: Search by pipeline_tag filter (current API)
            if pipeline_tag:
                try:
                    print(f"    Method 1: Using pipeline_tag filter...")
                    models = list(self.api.list_models(
                        filter=f"pipeline_tag:{pipeline_tag}",
                        sort="downloads",
                        limit=top_k * 2
                    ))
                    print(f"    Found {len(models)} models")
                except Exception as e:
                    print(f"    Method 1 failed: {e}")
                    models = []
            
            # Method 1.5: For translation, try specific translation tags
            if not models and task == TaskType.TRANSLATION:
                try:
                    print(f"    Method 1.5: Trying specific translation tags...")
                    # Try common translation model patterns
                    search_terms = ["translation", "mbart", "nllb", "m2m", "opus"]
                    for term in search_terms:
                        try:
                            term_models = list(self.api.list_models(
                                search=term,
                                sort="downloads",
                                limit=top_k
                            ))
                            models.extend(term_models)
                            print(f"      Found {len(term_models)} models with '{term}'")
                        except:
                            continue
                    # Remove duplicates
                    unique_ids = set()
                    unique_models = []
                    for m in models:
                        if m.modelId not in unique_ids:
                            unique_ids.add(m.modelId)
                            unique_models.append(m)
                    models = unique_models
                    print(f"    Total unique models after search: {len(models)}")
                except Exception as e:
                    print(f"    Method 1.5 failed: {e}")

            # Method 2: Try without filter (get popular models)
            if not models:
                try:
                    print(f"    Method 2: Getting popular models...")
                    models = list(self.api.list_models(
                        sort="downloads",
                        limit=top_k * 2
                    ))
                    print(f"    Found {len(models)} models")
                except Exception as e:
                    print(f"    Method 2 failed: {e}")
                    models = []
            
            # Method 3: Try with search parameter
            if not models and pipeline_tag:
                try:
                    print(f"    Method 3: Using search parameter...")
                    models = list(self.api.list_models(
                        search=pipeline_tag,
                        sort="downloads",
                        limit=top_k * 2
                    ))
                    print(f"    Found {len(models)} models")
                except Exception as e:
                    print(f"    Method 3 failed: {e}")
                    models = []
            
            print(f"  Total candidate models: {len(models)}")
            
            if not models:
                print("   No models found from HuggingFace API")
                return []
            
            # Fetch detailed metadata
            model_details = []
            for i, model in enumerate(models[:top_k]):
                try:
                    print(f"    Processing {i+1}/{min(len(models), top_k)}: {model.modelId}")
                    
                    metadata = await self._fetch_model_details(model, requirements)
                    if metadata:
                        model_details.append(metadata)
                        print(f"      Added to candidates")
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"      Error: {e}")
                    continue
            
            print(f"   Found {len(model_details)} suitable models")
            return model_details
            
        except Exception as e:
            print(f"   Error searching models: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _fetch_model_details(self, model, requirements: UserRequirements) -> Optional[ModelMetadata]:
        """Fetch detailed information for a specific model"""
        try:
            # Get model info from HuggingFace
            model_info = self.api.model_info(model.modelId)
            
            # Fetch model card content (README.md)
            model_card = await self._fetch_model_card(model.modelId)
            
            # Extract performance metrics from model card
            performance_metrics = self._extract_performance_metrics(model_card)
            
            # Estimate model size
            model_size = self._estimate_model_size(model_info)
            
            # Filter by size constraint if specified
            if requirements.max_model_size_gb and model_size:
                if model_size > requirements.max_model_size_gb:
                    return None
            
            # Extract language tags
            languages = ["en"]  # Default
            if hasattr(model_info, 'tags'):
                for tag in model_info.tags:
                    if tag.startswith("language:"):
                        languages = [tag.replace("language:", "")]
                        break
            
            # Get license from card data
            license_info = "unknown"
            if hasattr(model_info, 'cardData') and model_info.cardData:
                license_info = model_info.cardData.get("license", "unknown")
            
            # Get tags
            tags = getattr(model_info, 'tags', [])
            
            # Get pipeline tag
            pipeline_tag = getattr(model_info, 'pipeline_tag', None)
            
            # Get LLM-specific info
            llm_info = self._extract_llm_info(model_info, model_card)
            
            # Get OCR-specific info
            ocr_info = self._extract_ocr_info(model_info, model_card)
            
            # Determine task type from pipeline tag
            task_type = self._map_pipeline_to_task(pipeline_tag) or requirements.task_type
            
            # Extract source and target languages for translation
            source_languages = self._extract_source_languages(model_info, model_card)
            target_languages = self._extract_target_languages(model_info, model_card)
            
            # Extract voice count for TTS
            voice_count = self._extract_voice_count(model_card)
            
            # Extract sample rate for audio models
            sample_rate = self._extract_sample_rate(model_card)
            
            # Extract WER for STT
            wer_score = self._extract_wer(model_card)
            
            return ModelMetadata(
                model_id=model.modelId,
                task_type=task_type,
                downloads=getattr(model_info, 'downloads', 0) or 0,
                likes=getattr(model_info, 'likes', 0) or 0,
                last_modified=getattr(model_info, 'lastModified', datetime.now()),
                license=license_info,
                model_size=model_size,
                languages=languages,
                tags=tags,
                pipeline_tag=pipeline_tag,
                base_model=getattr(model_info, 'base_model', None),
                finetuned_from=getattr(model_info, 'finetuned_from', None),
                performance_metrics=performance_metrics,
                hardware_requirements=self._extract_hardware_info(model_card),
                model_card_content=model_card,
                # Additional fields
                source_languages=source_languages,
                target_languages=target_languages,
                voice_count=voice_count,
                sample_rate=sample_rate,
                wer_score=wer_score,
                context_length=llm_info.get("context_length"),
                has_chat_template=llm_info.get("has_chat_template", False),
                supports_function_calling=llm_info.get("supports_function_calling", False),
                supports_code=llm_info.get("supports_code", False),
                supports_instruction=llm_info.get("supports_instruction", False),
                quantization_supported=llm_info.get("quantization_supported", []),
                supports_handwriting=ocr_info.get("supports_handwriting", False),
                supports_layout=ocr_info.get("supports_layout", False),
                supports_tables=ocr_info.get("supports_tables", False),
                supports_formulas=ocr_info.get("supports_formulas", False),
                supported_image_formats=ocr_info.get("supported_image_formats", ["jpg", "png"])
            )
            
        except Exception as e:
            print(f"      Error fetching details: {e}")
            return None
    
    async def _fetch_model_card(self, model_id: str) -> str:
        """Fetch model card content from HuggingFace"""
        try:
            card_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(card_url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
                    return ""
        except Exception:
            return ""
    
    def _extract_performance_metrics(self, model_card: str) -> dict:
        """Extract performance metrics from model card text"""
        metrics = {}
        
        # Common metrics to look for
        metric_patterns = {
            "accuracy": r"accuracy[\s:]*([\d\.]+)%?",
            "f1": r"f1[\s:]*([\d\.]+)",
            "bleu": r"bleu[\s:]*([\d\.]+)",
            "rouge": r"rouge[\s:]*([\d\.]+)",
            "perplexity": r"perplexity[\s:]*([\d\.]+)",
            "precision": r"precision[\s:]*([\d\.]+)",
            "recall": r"recall[\s:]*([\d\.]+)"
        }
        
        for metric, pattern in metric_patterns.items():
            matches = re.findall(pattern, model_card.lower())
            if matches:
                try:
                    metrics[metric] = float(matches[0])
                except ValueError:
                    pass
        
        return metrics
    
    def _estimate_model_size(self, model_info) -> Optional[float]:
        """Estimate model size in GB"""
        try:
            # Try to get from config
            if hasattr(model_info, 'config') and model_info.config:
                param_size = model_info.config.get("num_parameters", 0)
                if param_size:
                    # Rough estimate: 4 bytes per parameter (float32)
                    size_gb = (param_size * 4) / (1024 ** 3)
                    return size_gb
            
            # Alternative: look for safetensors files
            if hasattr(model_info, 'siblings'):
                total_size = 0
                for sibling in model_info.siblings:
                    if hasattr(sibling, 'rfilename') and sibling.rfilename.endswith(('.safetensors', '.bin')):
                        if hasattr(sibling, 'size'):
                            total_size += sibling.size
                if total_size > 0:
                    return total_size / (1024 ** 3)  # Convert to GB
        except Exception:
            pass
        
        # Default size based on model name patterns
        model_id = model_info.modelId.lower() if hasattr(model_info, 'modelId') else ""
        
        if any(x in model_id for x in ['tiny', 'mini', 'albert']):
            return 0.05  # 50MB
        elif any(x in model_id for x in ['small', 'distilbert']):
            return 0.2   # 200MB
        elif any(x in model_id for x in ['base', 'bert-base']):
            return 0.5   # 500MB
        elif any(x in model_id for x in ['large', 'bert-large']):
            return 1.5   # 1.5GB
        elif any(x in model_id for x in ['xl', 'gpt2-xl']):
            return 3.0   # 3GB
        
        return None
    
    def _extract_hardware_info(self, model_card: str) -> dict:
        """Extract hardware requirements from model card"""
        hardware = {
            "cpu_compatible": True,  # Assume CPU compatible by default
            "gpu_required": False,
            "tpu_compatible": False,
            "min_ram_gb": 4  # Default assumption
        }
        
        card_lower = model_card.lower()
        
        if "gpu" in card_lower and "no gpu" not in card_lower:
            hardware["gpu_required"] = True
        if "tpu" in card_lower:
            hardware["tpu_compatible"] = True
        
        # Look for RAM requirements
        ram_match = re.search(r'(\d+)\s*gb?\s*ram', card_lower)
        if ram_match:
            hardware["min_ram_gb"] = int(ram_match.group(1))
        
        return hardware
    
    def _extract_source_languages(self, model_info, model_card: str) -> List[str]:
        """Extract source languages for translation models"""
        languages = []
        if hasattr(model_info, 'cardData') and model_info.cardData:
            src_langs = model_info.cardData.get("src_lang", [])
            if src_langs:
                if isinstance(src_langs, str):
                    languages = [src_langs]
                elif isinstance(src_langs, list):
                    languages = src_langs
        return languages or ["en"]
    
    def _extract_target_languages(self, model_info, model_card: str) -> List[str]:
        """Extract target languages for translation models"""
        languages = []
        if hasattr(model_info, 'cardData') and model_info.cardData:
            tgt_langs = model_info.cardData.get("tgt_lang", [])
            if tgt_langs:
                if isinstance(tgt_langs, str):
                    languages = [tgt_langs]
                elif isinstance(tgt_langs, list):
                    languages = tgt_langs
        return languages or ["en"]
    
    def _extract_voice_count(self, model_card: str) -> int:
        """Extract number of voices for TTS models"""
        patterns = [
            r'(\d+)\s+voices?',
            r'voices?:?\s*(\d+)',
            r'multi-voice.*?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_card.lower())
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        return 0
    
    def _extract_sample_rate(self, model_card: str) -> Optional[int]:
        """Extract sample rate for audio models"""
        patterns = [
            r'(\d+)\s*[kK]?[hH][zZ]',
            r'sample rate:?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_card.lower())
            if match:
                try:
                    rate = int(match.group(1))
                    if 'k' in match.group(0).lower():
                        rate *= 1000
                    return rate
                except:
                    pass
        return None
    
    def _extract_wer(self, model_card: str) -> Optional[float]:
        """Extract Word Error Rate for STT models"""
        patterns = [
            r'wer:?\s*([\d.]+)%?',
            r'word error rate:?\s*([\d.]+)%?',
            r'wer[\s=]+([\d.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_card.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        return None
    
    def _extract_llm_info(self, model_info, model_card: str) -> dict:
        """Extract LLM-specific information"""
        info = {
            "context_length": 2048,  # default
            "has_chat_template": False,
            "supports_function_calling": False,
            "supports_code": False,
            "supports_instruction": False,
            "quantization_supported": []
        }
        
        card_lower = model_card.lower()
        
        # Check context length
        context_patterns = [
            r'context length:?\s*(\d+)[kK]?',
            r'max[ _]?length:?\s*(\d+)[kK]?',
            r'(\d+)[kK]\s*(?:context|tokens)'
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, card_lower)
            if match:
                val = int(match.group(1))
                if 'k' in match.group(0).lower():
                    info["context_length"] = val * 1024
                else:
                    info["context_length"] = val
                break
        
        # Check for chat template
        info["has_chat_template"] = "chat template" in card_lower or "conversation" in card_lower
        
        # Check for function calling
        info["supports_function_calling"] = any(x in card_lower for x in 
            ["function calling", "tools", "function call", "tool use"])
        
        # Check for code generation
        info["supports_code"] = any(x in card_lower for x in 
            ["code generation", "programming", "python", "javascript"])
        
        # Check for instruction following
        info["supports_instruction"] = "instruction" in card_lower
        
        # Check quantization support
        if "4bit" in card_lower or "4-bit" in card_lower:
            info["quantization_supported"].append("4bit")
        if "8bit" in card_lower or "8-bit" in card_lower:
            info["quantization_supported"].append("8bit")
        
        return info
    
    def _extract_ocr_info(self, model_info, model_card: str) -> dict:
        """Extract OCR-specific information"""
        info = {
            "supports_handwriting": False,
            "supports_layout": False,
            "supports_tables": False,
            "supports_formulas": False,
            "supported_image_formats": ["jpg", "png"]  # default
        }
        
        card_lower = model_card.lower()
        
        # Check for handwriting
        info["supports_handwriting"] = "handwriting" in card_lower or "handwritten" in card_lower
        
        # Check for layout analysis
        info["supports_layout"] = any(x in card_lower for x in 
            ["layout", "paragraph", "document structure"])
        
        # Check for table extraction
        info["supports_tables"] = any(x in card_lower for x in 
            ["table", "spreadsheet", "tabular"])
        
        # Check for formula recognition
        info["supports_formulas"] = any(x in card_lower for x in 
            ["formula", "equation", "math", "latex"])
        
        # Check image formats
        formats = []
        for fmt in ["jpg", "jpeg", "png", "tiff", "bmp", "pdf"]:
            if fmt in card_lower:
                formats.append(fmt)
        if formats:
            info["supported_image_formats"] = formats
        
        return info
    
    def _map_pipeline_to_task(self, pipeline_tag: Optional[str]) -> Optional[TaskType]:
        """Map HuggingFace pipeline tag to our TaskType"""
        mapping = {
            "translation": TaskType.TRANSLATION,
            "text-to-speech": TaskType.TEXT_TO_SPEECH,
            "automatic-speech-recognition": TaskType.SPEECH_TO_TEXT,
            "text-generation": TaskType.TEXT_GENERATION,
            "question-answering": TaskType.QUESTION_ANSWERING,
            "summarization": TaskType.SUMMARIZATION,
            "image-to-text": TaskType.OCR,

            "document-question-answering": TaskType.DOCUMENT_UNDERSTANDING,
            "text-classification": TaskType.TEXT_CLASSIFICATION,
            "token-classification": TaskType.NAMED_ENTITY_RECOGNITION,
            "image-classification": TaskType.IMAGE_CLASSIFICATION,
            "object-detection": TaskType.OBJECT_DETECTION,
            "zero-shot-classification": TaskType.ZERO_SHOT_CLASSIFICATION
        }
        return mapping.get(pipeline_tag) if pipeline_tag else None