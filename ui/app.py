import gradio as gr
import requests
from PIL import Image
import io
import base64
import numpy as np

# API endpoints
API_BASE = "http://inference:8000"
CLASSIFY_ENDPOINT = f"{API_BASE}/predict"
CAPTION_ENDPOINT = f"{API_BASE}/caption"

# Custom CSS for better UI
custom_css = """
    .gradio-container {
        max-width: 800px !important;
    }
    .output-image {
        max-width: 100%;
        border-radius: 8px;
        margin: 10px 0;
    }
    .caption-box {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .prediction-box {
        background: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
"""

def process_image(image_path):
    """Helper function to process image for display"""
    if image_path is None:
        return None
    return Image.open(image_path)

def classify_image(image_path):
    """Classify an image using the inference API"""
    if image_path is None:
        return "Please upload an image first"
    
    try:
        # Open the image file
        with open(image_path, 'rb') as f:
            files = {"file": ("image.png", f, "image/png")}
            response = requests.post(CLASSIFY_ENDPOINT, files=files)
        
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if not predictions:
                return "No predictions returned from the API"
                
            # Format predictions as HTML for better display
            html_output = "<div class='predictions-container'>"
            for i, pred in enumerate(predictions, 1):
                score = float(pred.get('score', 0))
                score_display = f"{score:.2f}%"
                label = pred.get('label', 'Unknown').replace('_', ' ').title()
                html_output += f"""
                <div class='prediction-box'>
                    <strong>{i}. {label}</strong>
                    <div style='width: 100%; background: #e0e0e0; border-radius: 5px; margin: 5px 0;'>
                        <div style='width: {min(100, score)}%; background: #2196F3; color: white; text-align: center; padding: 2px 0; border-radius: 5px;'>
                            {score_display}
                        </div>
                    </div>
                </div>
                """
            html_output += "</div>"
            return html_output
        else:
            return f"<div class='error'>API error: {response.text}</div>"
            
    except Exception as e:
        return f"<div class='error'>Error: {str(e)}</div>"

def generate_captions(image_path):
    """Generate captions for an image using the BLIP model"""
    if image_path is None:
        return "Please upload an image first", None
    
    try:
        # Open the image file
        with open(image_path, 'rb') as f:
            files = {"file": ("image.png", f, "image/png")}
            response = requests.post(CAPTION_ENDPOINT, files=files)
        
        if response.status_code == 200:
            captions = response.json()
            simple = captions.get('simple_caption', 'No caption generated')
            detailed = captions.get('detailed_caption', 'No detailed caption generated')
            
            # Format captions as HTML for better display
            html_output = f"""
            <div class='caption-box'>
                <h3>Simple Caption:</h3>
                <p>{simple}</p>
                <h3>Detailed Description:</h3>
                <p>{detailed}</p>
            </div>
            """
            return html_output, image_path
        else:
            return f"<div class='error'>API error: {response.text}</div>", None
            
    except Exception as e:
        return f"<div class='error'>Error: {str(e)}</div>", None

# Create the tabbed interface
with gr.Blocks(css=custom_css, title="Image Analysis Tool") as demo:
    gr.Markdown("# Image Analysis Tool")
    gr.Markdown("Upload an image to classify it or generate a detailed description.")
    
    with gr.Tabs():
        with gr.TabItem("Image Classification"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="filepath", label="Upload Image")
                with gr.Column():
                    classify_btn = gr.Button("Classify Image")
                    classification_output = gr.HTML()
            
            classify_btn.click(
                fn=classify_image,
                inputs=[image_input],
                outputs=[classification_output]
            )
            
        with gr.TabItem("Image Captioning"):
            with gr.Row():
                with gr.Column():
                    caption_image_input = gr.Image(type="filepath", label="Upload Image")
                    caption_btn = gr.Button("Generate Caption")
                with gr.Column():
                    caption_output = gr.HTML()
                    image_output = gr.Image(visible=False)
            
            caption_btn.click(
                fn=generate_captions,
                inputs=[caption_image_input],
                outputs=[caption_output, image_output]
            )
            
    # Removed examples to prevent download issues

# Run the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
