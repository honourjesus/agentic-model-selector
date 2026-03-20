"""
Data models and schemas for the HuggingFace Model Selector
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class TaskType(str, Enum):
    """Supported ML tasks"""
    TRANSLATION = "translation"
    TEXT_TO_SPEECH = "text-to-speech"
    SPEECH_TO_TEXT = "automatic-speech-recognition"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    CHAT = "text-generation"  # Same pipeline tag
    INSTRUCTION_FOLLOWING = "text-generation"
    CODE_GENERATION = "text-generation"
    SUMMARIZATION = "summarization"
    OCR = "image-to-text"  # Pipeline tag for OCR
    DOCUMENT_UNDERSTANDING = "document-question-answering"
    # Legacy/Other
    TEXT_CLASSIFICATION = "text-classification"
    NAMED_ENTITY_RECOGNITION = "token-classification"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"


class Language(str, Enum):
    """Common languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


class VoiceType(str, Enum):
    """Voice types for TTS models"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class HardwareConstraint(str, Enum):
    """Hardware constraints for model deployment"""
    CPU = "cpu"
    GPU_4GB = "gpu_4gb"
    GPU_8GB = "gpu_8gb"
    GPU_16GB = "gpu_16gb"
    GPU_24GB = "gpu_24gb"
    GPU_40GB = "gpu_40gb"
    GPU_80GB = "gpu_80gb"
    TPU = "tpu"

class ModelSize(str, Enum):
    """Model size categories for LLMs"""
    TINY = "tiny"        # < 1B parameters
    SMALL = "small"      # 1B-3B parameters
    MEDIUM = "medium"    # 3B-7B parameters
    LARGE = "large"      # 7B-13B parameters
    XLARGE = "xlarge"    # 13B-30B parameters
    XXLARGE = "xxlarge"  # 30B-70B parameters
    MASSIVE = "massive"  # 70B+ parameters

class TranslationRequirements(BaseModel):
    """Requirements for translation models"""
    source_language: Language
    target_language: Language
    domain: Optional[str] = None  # e.g., medical, legal, technical
    quality_preference: str = "balanced"  # "speed", "quality", "balanced"


class TTSRequirements(BaseModel):
    """Requirements for text-to-speech models"""
    language: Language
    voice_type: VoiceType = VoiceType.NEUTRAL
    speaking_rate: float = 1.0  # 0.5 to 2.0
    pitch: float = 1.0  # 0.5 to 2.0
    wants_multiple_voices: bool = False


class STTRequirements(BaseModel):
    """Requirements for speech-to-text models"""
    language: Language
    domain: Optional[str] = None  # e.g., general, medical, telephony
    wants_word_timestamps: bool = False
    wants_diarization: bool = False  # Speaker diarization


class LLMRequirements(BaseModel):
    """Requirements for Large Language Models"""
    model_size: ModelSize = ModelSize.MEDIUM
    context_length: int = 2048  # tokens
    wants_chat_template: bool = False
    wants_function_calling: bool = False
    wants_code_generation: bool = False
    wants_instruction_following: bool = False
    wants_multilingual: bool = False
    quantization: Optional[str] = None  # "4bit", "8bit", None


class OCRRequirements(BaseModel):
    """Requirements for OCR models"""
    languages: List[Language] = Field(default_factory=lambda: [Language.ENGLISH])
    handwritten: bool = False  # Handwriting recognition
    document_type: Optional[str] = None  # "scanned", "photo", "document"
    wants_layout_analysis: bool = False  # Detect paragraphs, tables, etc.
    wants_table_extraction: bool = False
    wants_formula_recognition: bool = False  # Math formulas

class UserRequirements(BaseModel):
    """Combined user requirements"""
    task_type: TaskType
    hardware_constraints: List[HardwareConstraint] = Field(default_factory=lambda: [HardwareConstraint.CPU])
    max_model_size_gb: Optional[float] = None
    
    # Task-specific requirements
    translation_reqs: Optional[TranslationRequirements] = None
    tts_reqs: Optional[TTSRequirements] = None
    stt_reqs: Optional[STTRequirements] = None
    llm_reqs: Optional[LLMRequirements] = None
    ocr_reqs: Optional[OCRRequirements] = None
    

class ModelMetadata(BaseModel):
    """Metadata for a HuggingFace model"""
    model_id: str
    task_type: TaskType
    downloads: int
    likes: int
    last_modified: datetime
    license: str
    model_size: Optional[float] = None  # in GB
    parameter_count: Optional[int] = None  # Number of parameters
    languages: List[str] = Field(default_factory=list)
    framework: str = "pytorch"  # pytorch, tensorflow, jax
    pipeline_tag: Optional[str] = None
    
    # Translation-specific
    source_languages: List[str] = Field(default_factory=list)
    target_languages: List[str] = Field(default_factory=list)
    
    # TTS-specific
    voice_count: int = 0
    sample_rate: Optional[int] = None  # Hz
    
    # STT-specific
    wer_score: Optional[float] = None  # Word Error Rate
    
    # LLM-specific
    context_length: Optional[int] = None
    has_chat_template: bool = False
    supports_function_calling: bool = False
    supports_code: bool = False
    supports_instruction: bool = False
    quantization_supported: List[str] = Field(default_factory=list)
    
    # OCR-specific
    supports_handwriting: bool = False
    supports_layout: bool = False
    supports_tables: bool = False
    supports_formulas: bool = False
    supported_image_formats: List[str] = Field(default_factory=list)
    
    # General
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    hardware_requirements: Dict[str, Any] = Field(default_factory=dict)
    model_card_content: str = ""


class ModelScore(BaseModel):
    """Scored model with component scores"""
    model_id: str
    task_type: TaskType
    total_score: float
    component_scores: Dict[str, float]
    metadata: ModelMetadata


class BenchmarkResult(BaseModel):
    """Benchmark results for a model"""
    model_id: str
    task_type: TaskType
    latency_ms: float
    memory_usage_mb: float
    accuracy: Optional[float] = None #oCR
    throughput: Optional[float] = None
    # Task-specific metrics
    bleu_score: Optional[float] = None  # Translation
    wer_score: Optional[float] = None   # STT
    mos_score: Optional[float] = None   # TTS Mean Opinion Score
    perplexity: Optional[float] = None  # LLMs
    error: Optional[str] = None


class DeploymentType(str, Enum):
    """Supported deployment types"""
    FASTAPI = "fastapi"
    GRADIO = "gradio"
    DOCKER = "docker"


class SelectionResult(BaseModel):
    """Final model selection result"""
    status: str
    selected_model: Optional[str] = None
    task_type: Optional[TaskType] = None
    requirements: Optional[UserRequirements] = None
    all_scores: List[ModelScore] = Field(default_factory=list)
    benchmark_results: List[BenchmarkResult] = Field(default_factory=list)
    deployment_files: Optional[Dict[str, str]] = None
    error: Optional[str] = None