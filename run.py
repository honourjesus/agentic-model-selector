#!/usr/bin/env python
"""
Entry point for the HuggingFace Model Selector
Run this file to start the model selection process
"""

import asyncio
import argparse
import sys
from src.main import HuggingFaceModelSelector
from src.models.schemas import DeploymentType


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Selector - Find and deploy the best ML models"
    )
    
    parser.add_argument(
        "task",
        type=str,
        help="Task description (e.g., 'sentiment analysis', 'text summarization')"
    )
    
    parser.add_argument(
        "--deploy",
        type=str,
        choices=["fastapi", "gradio", "docker"],
        default="fastapi",
        help="Deployment type to generate"
    )
    
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip performance benchmarking"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top models to consider"
    )
    
    args = parser.parse_args()
    
    # Map deployment type
    deploy_map = {
        "fastapi": DeploymentType.FASTAPI,
        "gradio": DeploymentType.GRADIO,
        "docker": DeploymentType.DOCKER
    }
    
    print("\n" + "=" * 60)
    print(" HuggingFace Model Selector")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"Deployment: {args.deploy}")
    print(f"Benchmark: {'No' if args.no_benchmark else 'Yes'}")
    
    # Run the selector
    async def run():
        selector = HuggingFaceModelSelector()
        result = await selector.select_and_deploy(
            task_description=args.task,
            deployment_type=deploy_map[args.deploy],
            benchmark=not args.no_benchmark,
            top_k=args.top_k
        )
        
        if result.status == "success":
            print("\n" + "=" * 60)
            print(" Selection Complete!")
            print(f"Selected Model: {result.selected_model}")
            print("\nNext steps:")
            print(f"1. cd into the deployment folder")
            print(f"2. pip install -r requirements.txt")
            print(f"3. python app.py")
            print("=" * 60)
        else:
            print("\n Selection failed:", result.error)
            sys.exit(1)
    
    # Run async function
    asyncio.run(run())


if __name__ == "__main__":
    main()