"""
Input Agent - Parses user requirements for all model types
"""

import re
from typing import List, Optional

from src.models.schemas import (
    TaskType, HardwareConstraint, UserRequirements,
    TranslationRequirements, TTSRequirements, STTRequirements,
    LLMRequirements, OCRRequirements,
    Language, VoiceType, ModelSize
)


class InputAgent:
    """
    Analyzes user's task description and extracts structured requirements
    for translation, TTS, STT, LLMs, and OCR.
    """
    
    def __init__(self):
        # Map keywords to task types
        self.task_keywords = {
            # Translation
            TaskType.TRANSLATION: [
                "translate", "translation", "convert language", "english to french",
                "en to es", "german", "spanish", "french"
            ],
            
            # TTS
            TaskType.TEXT_TO_SPEECH: [
                "text to speech", "tts", "speech synthesis", "voice", 
                "speak", "audio from text", "read aloud"
            ],
            
            # STT
            TaskType.SPEECH_TO_TEXT: [
                "speech to text", "stt", "transcribe", "audio to text",
                "asr", "automatic speech recognition", "voice to text"
            ],
            
            # LLMs
            TaskType.TEXT_GENERATION: [
                "generate", "write", "completion", "continue", "story",
                "llm", "large language model", "gpt", "llama"
            ],
            TaskType.CHAT: [
                "chat", "conversation", "dialogue", "chatbot", "assistant"
            ],
            TaskType.INSTRUCTION_FOLLOWING: [
                "instruction", "follow instruction", "task", "command"
            ],
            TaskType.CODE_GENERATION: [
                "code", "programming", "python", "javascript", "function"
            ],
            TaskType.QUESTION_ANSWERING: [
                "question answering", "qa", "answer question", "extract answer"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "abstract", "condense", "tl;dr"
            ],
            
            # OCR
            TaskType.OCR: [
                "ocr", "optical character recognition", "extract text from image",
                "read image", "scan document", "image to text", "text from photo"
            ],
            TaskType.DOCUMENT_UNDERSTANDING: [
                "document understanding", "document qa", "document analysis",
                "form understanding", "invoice parsing"
            ]
        }
        
    def parse_requirements(self, task_description: str) -> UserRequirements:
        """
        Parse user's task description and extract requirements.
        """
        task_type = self._extract_task_type(task_description)
        hardware_constraints = self._extract_hardware_constraints(task_description)
        max_model_size = self._extract_model_size(task_description)
        
        # Parse task-specific requirements
        translation_reqs = None
        tts_reqs = None
        stt_reqs = None
        llm_reqs = None
        ocr_reqs = None
        
        if task_type == TaskType.TRANSLATION:
            translation_reqs = self._parse_translation_requirements(task_description)
        elif task_type == TaskType.TEXT_TO_SPEECH:
            tts_reqs = self._parse_tts_requirements(task_description)
        elif task_type == TaskType.SPEECH_TO_TEXT:
            stt_reqs = self._parse_stt_requirements(task_description)
        elif task_type in [TaskType.TEXT_GENERATION, TaskType.CHAT, 
                          TaskType.INSTRUCTION_FOLLOWING, TaskType.CODE_GENERATION,
                          TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION]:
            llm_reqs = self._parse_llm_requirements(task_description, task_type)
        elif task_type in [TaskType.OCR, TaskType.DOCUMENT_UNDERSTANDING]:
            ocr_reqs = self._parse_ocr_requirements(task_description)
        
        return UserRequirements(
            task_type=task_type,
            hardware_constraints=hardware_constraints,
            max_model_size_gb=max_model_size,
            translation_reqs=translation_reqs,
            tts_reqs=tts_reqs,
            stt_reqs=stt_reqs,
            llm_reqs=llm_reqs,
            ocr_reqs=ocr_reqs
        )
    
    def _extract_task_type(self, description: str) -> TaskType:
        """Identify the task from description"""
        description_lower = description.lower()
        
        for task_type, keywords in self.task_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return task_type
        
        # Default to text generation if unsure
        return TaskType.TEXT_GENERATION
    
    def _parse_translation_requirements(self, description: str) -> TranslationRequirements:
        """Extract translation-specific requirements"""
        description_lower = description.lower()
        
        # Default values
        source_lang = Language.ENGLISH
        target_lang = Language.SPANISH
        
        # Try to extract language pairs
        patterns = [
            r'(?:from\s+)?(\w+)\s+(?:to|in(?:to)?)\s+(\w+)',
            r'(\w+)[\s-]+to[\s-]+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description_lower)
            if match:
                lang1, lang2 = match.groups()
                source_lang = self._map_language(lang1)
                target_lang = self._map_language(lang2)
                break
        
        # Determine domain
        domain = None
        domains = ["medical", "legal", "technical", "financial", "literary"]
        for d in domains:
            if d in description_lower:
                domain = d
                break
        
        # Quality preference
        quality = "balanced"
        if "fast" in description_lower or "quick" in description_lower:
            quality = "speed"
        elif "high quality" in description_lower or "accurate" in description_lower:
            quality = "quality"
        
        return TranslationRequirements(
            source_language=source_lang,
            target_language=target_lang,
            domain=domain,
            quality_preference=quality
        )
    
    def _parse_tts_requirements(self, description: str) -> TTSRequirements:
        """Extract TTS-specific requirements"""
        description_lower = description.lower()
        
        # Extract language
        language = Language.ENGLISH
        for lang in Language:
            if lang.value in description_lower or lang.name.lower() in description_lower:
                language = lang
                break
        
        # Extract voice type
        voice = VoiceType.NEUTRAL
        if "male" in description_lower:
            voice = VoiceType.MALE
        elif "female" in description_lower:
            voice = VoiceType.FEMALE
        
        # Check if multiple voices wanted
        multiple = "multiple voices" in description_lower or "different voices" in description_lower
        
        return TTSRequirements(
            language=language,
            voice_type=voice,
            wants_multiple_voices=multiple
        )
    
    def _parse_stt_requirements(self, description: str) -> STTRequirements:
        """Extract STT-specific requirements"""
        description_lower = description.lower()
        
        # Extract language
        language = Language.ENGLISH
        for lang in Language:
            if lang.value in description_lower or lang.name.lower() in description_lower:
                language = lang
                break
        
        # Extract domain
        domain = None
        if "medical" in description_lower:
            domain = "medical"
        elif "telephone" in description_lower or "call" in description_lower:
            domain = "telephony"
        elif "meeting" in description_lower:
            domain = "meeting"
        
        # Check for advanced features
        timestamps = "timestamp" in description_lower or "word timing" in description_lower
        diarization = "speaker" in description_lower or "who said" in description_lower
        
        return STTRequirements(
            language=language,
            domain=domain,
            wants_word_timestamps=timestamps,
            wants_diarization=diarization
        )
    
    def _parse_llm_requirements(self, description: str, task_type: TaskType) -> LLMRequirements:
        """Extract LLM-specific requirements"""
        description_lower = description.lower()
        
        # Determine model size preference
        model_size = ModelSize.MEDIUM
        if any(x in description_lower for x in ["tiny", "small", "lightweight", "fast"]):
            model_size = ModelSize.SMALL
        elif any(x in description_lower for x in ["large", "powerful", "best quality"]):
            model_size = ModelSize.LARGE
        elif any(x in description_lower for x in ["xlarge", "huge", "massive"]):
            model_size = ModelSize.XXLARGE
        
        # Context length
        context_length = 2048  # default
        context_match = re.search(r'(\d+)[kK]?\s*(context|token)', description_lower)
        if context_match:
            val = context_match.group(1)
            if 'k' in context_match.group(0).lower():
                context_length = int(val) * 1024
            else:
                context_length = int(val)
        
        # Check for specific capabilities
        wants_chat = any(x in description_lower for x in ["chat", "conversation", "dialogue"])
        wants_code = any(x in description_lower for x in ["code", "programming", "python", "javascript"])
        wants_instruction = any(x in description_lower for x in ["instruction", "task", "command"])
        wants_function = any(x in description_lower for x in ["function calling", "tools", "actions"])
        wants_multilingual = any(x in description_lower for x in ["multilingual", "multiple languages"])
        
        # Check for quantization
        quantization = None
        if "4bit" in description_lower or "4-bit" in description_lower:
            quantization = "4bit"
        elif "8bit" in description_lower or "8-bit" in description_lower:
            quantization = "8bit"
        
        return LLMRequirements(
            model_size=model_size,
            context_length=context_length,
            wants_chat_template=wants_chat,
            wants_function_calling=wants_function,
            wants_code_generation=wants_code,
            wants_instruction_following=wants_instruction,
            wants_multilingual=wants_multilingual,
            quantization=quantization
        )
    
    def _parse_ocr_requirements(self, description: str) -> OCRRequirements:
        """Extract OCR-specific requirements"""
        description_lower = description.lower()
        
        # Extract languages
        languages = [Language.ENGLISH]
        for lang in Language:
            if lang.value in description_lower or lang.name.lower() in description_lower:
                languages = [lang]
                break
        
        # Check for handwriting
        handwritten = "handwriting" in description_lower or "handwritten" in description_lower
        
        # Document type
        doc_type = None
        if "scanned" in description_lower:
            doc_type = "scanned"
        elif "photo" in description_lower or "photograph" in description_lower:
            doc_type = "photo"
        elif "document" in description_lower:
            doc_type = "document"
        
        # Check for advanced features
        layout = "layout" in description_lower or "paragraph" in description_lower
        tables = "table" in description_lower or "spreadsheet" in description_lower
        formulas = any(x in description_lower for x in ["formula", "equation", "math"])
        
        return OCRRequirements(
            languages=languages,
            handwritten=handwritten,
            document_type=doc_type,
            wants_layout_analysis=layout,
            wants_table_extraction=tables,
            wants_formula_recognition=formulas
        )
    
    def _map_language(self, lang_text: str) -> Language:
        """Map language name/code to Language enum"""
        lang_map = {
            "en": Language.ENGLISH, "english": Language.ENGLISH,
            "es": Language.SPANISH, "spanish": Language.SPANISH,
            "fr": Language.FRENCH, "french": Language.FRENCH,
            "de": Language.GERMAN, "german": Language.GERMAN,
            "it": Language.ITALIAN, "italian": Language.ITALIAN,
            "pt": Language.PORTUGUESE, "portuguese": Language.PORTUGUESE,
            "nl": Language.DUTCH, "dutch": Language.DUTCH,
            "ru": Language.RUSSIAN, "russian": Language.RUSSIAN,
            "zh": Language.CHINESE, "chinese": Language.CHINESE,
            "ja": Language.JAPANESE, "japanese": Language.JAPANESE,
            "ko": Language.KOREAN, "korean": Language.KOREAN,
            "ar": Language.ARABIC, "arabic": Language.ARABIC,
            "hi": Language.HINDI, "hindi": Language.HINDI
        }
        return lang_map.get(lang_text.lower(), Language.ENGLISH)
    
    def _extract_hardware_constraints(self, description: str) -> List[HardwareConstraint]:
        """Extract hardware constraints"""
        constraints = []
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["cpu", "no gpu", "without gpu"]):
            constraints.append(HardwareConstraint.CPU)
        if any(word in description_lower for word in ["4gb", "4 gb", "small gpu"]):
            constraints.append(HardwareConstraint.GPU_4GB)
        if any(word in description_lower for word in ["8gb", "8 gb", "medium gpu"]):
            constraints.append(HardwareConstraint.GPU_8GB)
        if any(word in description_lower for word in ["16gb", "16 gb"]):
            constraints.append(HardwareConstraint.GPU_16GB)
        if any(word in description_lower for word in ["24gb", "24 gb"]):
            constraints.append(HardwareConstraint.GPU_24GB)
        if any(word in description_lower for word in ["40gb", "40 gb", "a100"]):
            constraints.append(HardwareConstraint.GPU_40GB)
        if any(word in description_lower for word in ["80gb", "80 gb"]):
            constraints.append(HardwareConstraint.GPU_80GB)
        if "tpu" in description_lower:
            constraints.append(HardwareConstraint.TPU)
        
        return constraints if constraints else [HardwareConstraint.CPU]
    
    def _extract_model_size(self, description: str) -> Optional[float]:
        """Extract maximum model size constraint"""
        size_pattern = r'(\d+(?:\.\d+)?)\s*(gb|mb)'
        match = re.search(size_pattern, description.lower())
        
        if match:
            size = float(match.group(1))
            unit = match.group(2).lower()
            
            if unit == 'gb':
                return size
            elif unit == 'mb':
                return size / 1024
        
        return None