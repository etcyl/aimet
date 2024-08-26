from flask import Flask, request, render_template, Markup
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import torch
import torchvision.transforms as transforms
from torchvision import models
import jax
import jax.numpy as jnp
import openai
import os

app = Flask(__name__)

# Load OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the PyTorch model (ResNet18 for demonstration)
def initialize_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()
    return model

model = initialize_model()

def convert_params_to_jax(params):
    # Convert PyTorch parameters to JAX-compatible arrays
    return {name: jnp.array(param.cpu().detach().numpy()) for name, param in params.items()}

def prune_jax(params):
    def prune_fn(x):
        return jnp.where(jnp.abs(x) < 0.01, 0, x)
    
    pruned_params = jax.tree_util.tree_map(prune_fn, params)
    num_pruned_params = sum((x == 0).sum() for x in pruned_params.values())
    return pruned_params, num_pruned_params

def quantize_jax(params):
    def quantize_fn(x):
        return jax.lax.round(x * 255) / 255

    quantized_params = jax.tree_util.tree_map(quantize_fn, params)
    return quantized_params

def apply_compression_jax(params):
    pruned_params, num_pruned_params = prune_jax(params)
    compressed_params = quantize_jax(pruned_params)
    return compressed_params, num_pruned_params

def process_jax(params, option):
    if option == "prune":
        pruned_params, num_pruned_params = prune_jax(params)
        explainability_output = f"Pruning reduced the model size by removing {num_pruned_params} parameters."
        return pruned_params, explainability_output, "pruning"
    elif option == "quantize":
        quantized_params = quantize_jax(params)
        explainability_output = "Quantization was applied to compress the model by reducing precision."
        return quantized_params, explainability_output, "quantization"
    elif option == "compression":
        compressed_params, num_pruned_params = apply_compression_jax(params)
        explainability_output = f"Compression combined pruning (removed {num_pruned_params} parameters) and quantization for optimal model size reduction."
        return compressed_params, explainability_output, "compression"
    else:
        explainability_output = "No optimization was applied."
        return params, explainability_output, "none"

def get_activation_maps(model, image):
    activation = {}
    def hook_fn(module, input, output):
        activation['value'] = output.detach()

    layer = model.layer4[1].conv2
    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image)

    hook.remove()

    return activation['value']

def visualize_activation_map(activation_map):
    avg_map = torch.mean(activation_map, dim=1).squeeze().cpu()
    plt.imshow(avg_map, cmap='viridis')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def generate_llm_explanation(model_name, framework, technique, explainability_output, llm_model):
    prompt = f"""
    You are tasked with providing an explainability report for a neural network model optimization. The model is a {model_name} using the {framework} framework. The applied technique was {technique}. Here is the summary of changes after optimization:

    {explainability_output}

    Based on this summary, please generate a clear and concise explainability report that highlights the key changes and their impact on the model's performance.
    """

    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are an AI that generates concise and informative explainability reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,  # Adjust if needed
        temperature=0.7
    )

    # Collect the full response in case it’s paginated
    llm_output = response['choices'][0]['message']['content'].strip()
    
    # Check if the response is cut off and handle continuation (for large responses)
    if response['choices'][0].get('finish_reason') == 'length':
        while True:
            continuation_prompt = "Continue the explainability report."
            additional_response = openai.ChatCompletion.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an AI that generates concise and informative explainability reports."},
                    {"role": "user", "content": continuation_prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )
            additional_output = additional_response['choices'][0]['message']['content'].strip()
            llm_output += "\n" + additional_output
            if additional_response['choices'][0].get('finish_reason') != 'length':
                break

    # Replace newlines with HTML line breaks
    formatted_explanation = Markup(llm_output.replace("\n", "<br>"))

    return formatted_explanation

@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    llm_explanation = ""
    selected_llm_model = "gpt-3.5-turbo"  # Default LLM model
    if request.method == 'POST':
        file = request.files['image']
        optimization_option = request.form.get('optimization_option')
        selected_llm_model = request.form.get('llm_model', "gpt-3.5-turbo")  # Capture selected LLM model

        if file:
            img = Image.open(file).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            original_activation_map = get_activation_maps(model, img_tensor)
            original_image = visualize_activation_map(original_activation_map)

            if optimization_option != "none":
                # Convert the model parameters to JAX-compatible format
                params = convert_params_to_jax({name: param for name, param in model.named_parameters()})
                processed_params, explainability_output, technique = process_jax(params, optimization_option)
                
                # Generate LLM-based explainability report
                model_name = "ResNet18"
                framework = "PyTorch (with JAX processing)"
                llm_explanation = generate_llm_explanation(model_name, framework, technique, explainability_output, selected_llm_model)
                
                compressed_image = original_image  # Placeholder for JAX-based visualization (currently unchanged)
            else:
                compressed_image = original_image
                explainability_output = "No JAX-based operations were applied. The displayed activation maps are unmodified."
                llm_explanation = Markup(explainability_output.replace("\n", "<br>"))

            return render_template('index.html', original_image=original_image, compressed_image=compressed_image, explainability_output=llm_explanation, selected_llm_model=selected_llm_model)

    return render_template('index.html', original_image=None, compressed_image=None, explainability_output=None, selected_llm_model=selected_llm_model)

if __name__ == '__main__':
    app.run(debug=True)
