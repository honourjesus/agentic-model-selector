"""
Main orchestrator for the HuggingFace Model Selector
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
from typing import Optional

from src.agents.input_agent import InputAgent
from src.agents.research_agent import ResearchAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.benchmarking_agent import BenchmarkingAgent
from src.agents.deployment_agent import DeploymentAgent
from src.models.schemas import DeploymentType, SelectionResult, TaskType


class HuggingFaceModelSelector:
    """
    Main orchestrator that coordinates all agents.
    
    This class:
    1. Takes a task description
    2. Parses requirements
    3. Searches for models
    4. Evaluates and scores them
    5. Benchmarks top models
    6. Generates deployment code
    """
    
    def __init__(self):
        self.input_agent = InputAgent()
        self.research_agent = ResearchAgent()
        self.evaluation_agent = EvaluationAgent()
        self.benchmarking_agent = BenchmarkingAgent()
        self.deployment_agent = DeploymentAgent()
    
    async def select_and_deploy(self,
                               task_description: str,
                               deployment_type: DeploymentType = DeploymentType.FASTAPI,
                               benchmark: bool = True,
                               top_k: int = 5) -> SelectionResult:
        """
        Complete pipeline: select best model and generate deployment code.
        
        Args:
            task_description: Natural language task (e.g., "translate english to french")
            deployment_type: Type of deployment to generate
            benchmark: Whether to run performance benchmarks
            top_k: Number of top models to consider
            
        Returns:
            SelectionResult with all details
        """
        try:
            print("\n" + "=" * 60)
            print(" HuggingFace Model Selector")
            print("=" * 60)
            
            # Step 1: Parse requirements
            print("\n Step 1: Analyzing requirements...")
            requirements = self.input_agent.parse_requirements(task_description)
            print(f"  Task: {requirements.task_type.value}")
            
            # Display task-specific requirements
            if requirements.task_type == TaskType.TRANSLATION and requirements.translation_reqs:
                req = requirements.translation_reqs
                print(f"  Translation: {req.source_language.value} → {req.target_language.value}")
                if req.domain:
                    print(f"  Domain: {req.domain}")
            
            elif requirements.task_type == TaskType.TEXT_TO_SPEECH and requirements.tts_reqs:
                req = requirements.tts_reqs
                print(f"  TTS Language: {req.language.value}")
                print(f"  Voice: {req.voice_type.value}")
            
            elif requirements.task_type == TaskType.SPEECH_TO_TEXT and requirements.stt_reqs:
                req = requirements.stt_reqs
                print(f"  STT Language: {req.language.value}")
                if req.domain:
                    print(f"  Domain: {req.domain}")
            
            elif requirements.task_type in [TaskType.TEXT_GENERATION, TaskType.CHAT, 
                                           TaskType.INSTRUCTION_FOLLOWING, TaskType.CODE_GENERATION,
                                           TaskType.QUESTION_ANSWERING, TaskType.SUMMARIZATION] and requirements.llm_reqs:
                req = requirements.llm_reqs
                print(f"  LLM Size: {req.model_size.value}")
                print(f"  Context Length: {req.context_length}")
            
            elif requirements.task_type in [TaskType.OCR, TaskType.DOCUMENT_UNDERSTANDING] and requirements.ocr_reqs:
                req = requirements.ocr_reqs
                langs = [lang.value for lang in req.languages]
                print(f"  OCR Languages: {langs}")
                if req.handwritten:
                    print(f"  Handwriting: Yes")
            
            print(f"  Hardware: {[c.value for c in requirements.hardware_constraints]}")
            
            # Step 2: Search for models
            print("\n Step 2: Searching HuggingFace...")
            models = await self.research_agent.search_models(requirements, top_k=top_k*2)
            
            if not models:
                return SelectionResult(
                    status="error",
                    error="No models found matching your requirements"
                )
            
            # Step 3: Score models
            print("\n Step 3: Evaluating models...")
            scored_models = self.evaluation_agent.score_models(models, requirements)
            
            # Display top models
            print("\n Top Models:")
            for i, scored in enumerate(scored_models[:5]):
                print(f"  {i+1}. {scored.model_id} (Score: {scored.total_score:.3f})")
                # Show top 3 component scores
                top_metrics = sorted(scored.component_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                for metric, score in top_metrics:
                    print(f"     - {metric}: {score:.2f}")
            
            # Step 4: Select best model
            best_model = scored_models[0]
            print(f"\n Step 4: Selected model: {best_model.model_id}")
            
            # Step 5: Benchmark (optional)
            benchmark_results = []
            if benchmark:
                print("\n Step 5: Running benchmarks...")
                benchmark_results = await self.benchmarking_agent.benchmark_models(
                    [m.model_id for m in scored_models[:3]],
                    requirements.task_type,
                    requirements
                )
                
                if benchmark_results and not benchmark_results[0].error:
                    print(f"\n Benchmark Results for {best_model.model_id}:")
                    print(f"  Latency: {benchmark_results[0].latency_ms:.2f} ms")
                    print(f"  Memory: {benchmark_results[0].memory_usage_mb:.2f} MB")
                    if benchmark_results[0].throughput:
                        print(f"  Throughput: {benchmark_results[0].throughput:.2f} samples/sec")
            
            # Step 6: Generate deployment code
            print("\n Step 6: Generating deployment code...")
            deployment_files = self.deployment_agent.generate_deployment(
                model_id=best_model.model_id,
                task_type=requirements.task_type,
                deployment_type=deployment_type,
                benchmark_results=benchmark_results[0] if benchmark_results else None,
                requirements=requirements
            )
            
            # Save files
            output_folder = self.deployment_agent.save_deployment_files(deployment_files)
            print(f"\n Deployment files saved to: {output_folder}")
            
            return SelectionResult(
                status="success",
                selected_model=best_model.model_id,
                task_type=requirements.task_type,
                requirements=requirements,
                all_scores=scored_models,
                benchmark_results=benchmark_results,
                deployment_files=deployment_files
            )
            
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()
            return SelectionResult(
                status="error",
                error=str(e)
            )


    

        