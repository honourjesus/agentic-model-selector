import gradio as gr
import asyncio
from src.main import HuggingFaceModelSelector
from src.models.schemas import DeploymentType

selector = HuggingFaceModelSelector()

async def find_model(task, deployment, top_k):
    result = await selector.select_and_deploy(
        task_description=task,
        deployment_type=DeploymentType.FASTAPI if deployment == "fastapi" else DeploymentType.GRADIO,
        benchmark=True,
        top_k=int(top_k)
    )
    if result.status == "success":
        scores = "\n".join([
            f"{i+1}. {s.model_id} (score: {s.total_score:.3f})"
            for i, s in enumerate(result.all_scores[:5])
        ])
        return (
            f"Best model: {result.selected_model}",
            scores,
            f"Deployment saved to: deployments/{result.selected_model.replace('/', '_')}"
        )
    else:
        return f"Error: {result.error}", "", ""

def run(task, deployment, top_k):
    return asyncio.run(find_model(task, deployment, top_k))

with gr.Blocks(title="Agentic Model Selector") as demo:
    gr.Markdown("# Agentic HuggingFace Model Selector")
    gr.Markdown("Describe your task and I will find the best model for you.")
    task_input = gr.Textbox(label="Describe your task", placeholder="e.g. sentiment analysis...", lines=2)
    with gr.Row():
        deployment_type = gr.Dropdown(choices=["fastapi", "gradio"], value="fastapi", label="Deployment type")
        top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of models to consider")
    submit_btn = gr.Button("Find Best Model", variant="primary")
    best_model = gr.Textbox(label="Selected Model")
    rankings = gr.Textbox(label="Model Rankings", lines=6)
    deployment_path = gr.Textbox(label="Deployment Files")
    submit_btn.click(fn=run, inputs=[task_input, deployment_type, top_k], outputs=[best_model, rankings, deployment_path])

demo.launch(server_name="0.0.0.0", server_port=7860)
