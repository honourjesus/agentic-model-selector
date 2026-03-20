"""
Evaluation Agent - Scores speech and translation models
"""

import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional

from src.models.schemas import (
    ModelMetadata, UserRequirements, HardwareConstraint, 
    ModelScore, TaskType
)


class EvaluationAgent:
    """
    Evaluates and scores translation, TTS, and STT models.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'downloads': 0.20,
            'recency': 0.15,
            'license': 0.10,
            'size': 0.15,
            'performance': 0.25,
            'language_match': 0.15
        }
        
    def score_models(self, 
                    models: List[ModelMetadata], 
                    requirements: UserRequirements) -> List[ModelScore]:
        """Score models based on multiple criteria"""
        scored_models = []
        
        for model in models:
            scores = {}
            
            # Common scores
            scores['downloads'] = self._score_downloads(model.downloads)
            scores['recency'] = self._score_recency(model.last_modified)
            scores['license'] = self._score_license(model.license)
            scores['size'] = self._score_size(model.model_size, requirements)
            scores['performance'] = self._score_performance(model, requirements)
            scores['language_match'] = self._score_language_match(model, requirements)
            
            # Task-specific scores
            if model.task_type in [TaskType.TEXT_GENERATION, TaskType.CHAT, 
                                  TaskType.INSTRUCTION_FOLLOWING, TaskType.CODE_GENERATION,
                                  TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION]:
                scores['llm_capabilities'] = self._score_llm_capabilities(model, requirements)
            elif model.task_type in [TaskType.OCR, TaskType.DOCUMENT_UNDERSTANDING]:
                scores['ocr_capabilities'] = self._score_ocr_capabilities(model, requirements)
            
            # Calculate weighted total
            total_score = 0.0
            weight_sum = 0.0
            
            all_weights = {
                'downloads': 0.15,
                'recency': 0.10,
                'license': 0.05,
                'size': 0.10,
                'performance': 0.20,
                'language_match': 0.15,
                'llm_capabilities': 0.25,
                'ocr_capabilities': 0.25
            }
            
            for metric, score in scores.items():
                if metric in all_weights:
                    total_score += score * all_weights[metric]
                    weight_sum += all_weights[metric]
            
            if weight_sum > 0:
                total_score /= weight_sum
            
            # Apply hardware penalty
            hardware_penalty = self._check_hardware_constraints(model, requirements)
            total_score *= hardware_penalty
            
            scored_models.append(ModelScore(
                model_id=model.model_id,
                task_type=model.task_type,
                total_score=float(total_score),
                component_scores=scores,
                metadata=model
            ))
        
        return sorted(scored_models, key=lambda x: x.total_score, reverse=True)
    
    def _score_downloads(self, downloads: int) -> float:
        """Score based on downloads (log scale)"""
        if downloads <= 0:
            return 0.0
        log_downloads = np.log10(downloads + 1)
        return min(log_downloads / 6.0, 1.0)
    
    def _score_recency(self, last_modified) -> float:
        """Score based on recency - FIXED timezone issue"""
        if not last_modified:
            return 0.5
        
        try:
            # Make last_modified timezone-naive for comparison
            if hasattr(last_modified, 'tzinfo') and last_modified.tzinfo is not None:
                # Convert to timezone-naive by removing timezone info
                last_modified = last_modified.replace(tzinfo=None)
            
            # Get current time as timezone-naive
            now = datetime.now()
            
            # Calculate days difference
            days_since_update = (now - last_modified).days
            
            if days_since_update < 30:
                return 1.0
            elif days_since_update < 90:
                return 0.8
            elif days_since_update < 180:
                return 0.6
            elif days_since_update < 365:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            print(f"      Warning: Error calculating recency: {e}")
            return 0.5
    
    def _score_license(self, license: str) -> float:
        """Score based on license"""
        license_lower = license.lower()
        open_licenses = ['mit', 'apache', 'bsd', 'cc', 'gpl', 'lgpl']
        
        if any(open_license in license_lower for open_license in open_licenses):
            return 0.9
        elif 'commercial' in license_lower:
            return 0.5
        else:
            return 0.7
    
    def _score_size(self, model_size: Optional[float], requirements: UserRequirements) -> float:
        """Score based on size - smaller is better"""
        if not model_size:
            return 0.5
        
        if requirements.max_model_size_gb:
            if model_size > requirements.max_model_size_gb:
                return 0.0
            size_ratio = 1.0 - (model_size / requirements.max_model_size_gb)
            return 0.5 + (size_ratio * 0.5)
        
        # No constraint - smaller is better
        if model_size < 0.5:
            return 1.0
        elif model_size < 1.0:
            return 0.9
        elif model_size < 2.0:
            return 0.7
        elif model_size < 5.0:
            return 0.5
        else:
            return 0.3
    
    def _score_performance(self, model: ModelMetadata, requirements: UserRequirements) -> float:
        """Score based on performance metrics"""
        metrics = model.performance_metrics
        
        if not metrics:
            return 0.5
        
        if model.task_type == TaskType.TRANSLATION:
            # Prefer BLEU scores
            if 'bleu' in metrics:
                return min(metrics['bleu'] / 50, 1.0)  # BLEU up to 50
            return 0.5
        
        elif model.task_type == TaskType.SPEECH_TO_TEXT:
            # Prefer low WER
            if 'wer' in metrics:
                return max(0, 1.0 - (metrics['wer'] / 100))
            return 0.5
        
        elif model.task_type == TaskType.TEXT_TO_SPEECH:
            # Prefer more voices and higher sample rate
            score = 0.5
            if model.voice_count > 0:
                score += min(model.voice_count / 10, 0.3)
            if model.sample_rate and model.sample_rate >= 16000:
                score += 0.2
            return min(score, 1.0)
        
        return 0.5
    
    def _score_language_match(self, model: ModelMetadata, requirements: UserRequirements) -> float:
        """Score based on language support"""
        if requirements.task_type == TaskType.TRANSLATION and requirements.translation_reqs:
            req = requirements.translation_reqs
            source = req.source_language.value
            target = req.target_language.value
            
            score = 0.5
            if model.source_languages and source in model.source_languages:
                score += 0.25
            if model.target_languages and target in model.target_languages:
                score += 0.25
            return score
        
        elif requirements.task_type == TaskType.TEXT_TO_SPEECH and requirements.tts_reqs:
            req = requirements.tts_reqs
            lang = req.language.value
            
            if model.languages and lang in model.languages:
                return 1.0
            return 0.5
        
        elif requirements.task_type == TaskType.SPEECH_TO_TEXT and requirements.stt_reqs:
            req = requirements.stt_reqs
            lang = req.language.value
            
            if model.languages and lang in model.languages:
                return 1.0
            return 0.5
        
        return 0.5
    
    def _score_llm_capabilities(self, model: ModelMetadata, requirements: UserRequirements) -> float:
        """Score LLM based on capabilities"""
        if not requirements.llm_reqs:
            return 0.5
        
        req = requirements.llm_reqs
        score = 0.5
        
        # Check context length
        if model.context_length and req.context_length:
            if model.context_length >= req.context_length:
                score += 0.2
            else:
                score -= 0.1
        
        # Check chat template
        if req.wants_chat_template and model.has_chat_template:
            score += 0.2
        
        # Check function calling
        if req.wants_function_calling and model.supports_function_calling:
            score += 0.2
        
        # Check code generation
        if req.wants_code_generation and model.supports_code:
            score += 0.2
        
        # Check instruction following
        if req.wants_instruction_following and model.supports_instruction:
            score += 0.2
        
        # Check quantization support
        if req.quantization and req.quantization in model.quantization_supported:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_ocr_capabilities(self, model: ModelMetadata, requirements: UserRequirements) -> float:
        """Score OCR model based on capabilities"""
        if not requirements.ocr_reqs:
            return 0.5
        
        req = requirements.ocr_reqs
        score = 0.5
        
        # Check handwriting support
        if req.handwritten and model.supports_handwriting:
            score += 0.3
        elif req.handwritten and not model.supports_handwriting:
            score -= 0.2
        
        # Check layout analysis
        if req.wants_layout_analysis and model.supports_layout:
            score += 0.2
        
        # Check table extraction
        if req.wants_table_extraction and model.supports_tables:
            score += 0.2
        
        # Check formula recognition
        if req.wants_formula_recognition and model.supports_formulas:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_hardware_constraints(self, model: ModelMetadata, 
                                   requirements: UserRequirements) -> float:
        """Apply penalty if hardware constraints not met"""
        if not requirements.hardware_constraints:
            return 1.0
        
        hardware = model.hardware_requirements
        
        for constraint in requirements.hardware_constraints:
            if constraint == HardwareConstraint.CPU:
                if hardware.get('cpu_compatible', True):
                    return 1.0
            elif constraint in [HardwareConstraint.GPU_4GB, HardwareConstraint.GPU_8GB, 
                               HardwareConstraint.GPU_16GB, HardwareConstraint.GPU_24GB,
                               HardwareConstraint.GPU_40GB, HardwareConstraint.GPU_80GB]:
                if hardware.get('gpu_required', False):
                    return 0.9  # Small penalty for requiring GPU
                return 1.0
        
        return 0.8