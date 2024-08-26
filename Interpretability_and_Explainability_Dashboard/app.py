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
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Load OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set device for PyTorch (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up JAX to use GPU/TPU if available (handled automatically by JAX)
jax_device = jax.devices()[0]  # Automatically selects the available device

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dictionary to map model names to torchvision model functions
MODEL_OPTIONS = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "VGG16": models.vgg16,
    "DenseNet121": models.densenet121
}

def initialize_model(selected_model):
    model_function = MODEL_OPTIONS.get(selected_model, models.resnet18)  # Default to ResNet18 if model not found
    model = model_function(weights="IMAGENET1K_V1")
    model = model.to(device)  # Send model to the appropriate device
    model.eval()
    return model

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

def update_model_parameters(model, new_params):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in new_params:
                param.copy_(torch.tensor(np.array(new_params[name])))

def get_activation_maps(model, image):
    activation = {}

    def hook_fn(module, input, output):
        activation['value'] = output.detach()

    # Dynamically determine which layer to hook based on the model architecture
    if isinstance(model, models.ResNet):
        layer = model.layer4[1].conv2  # For ResNet, we hook into layer4's conv2
    elif isinstance(model, models.DenseNet):
        layer = model.features[-1]  # For DenseNet, we hook into the last layer in the features
    elif isinstance(model, models.VGG):
        layer = model.features[-1]  # For VGG, we hook into the last layer in the features
    else:
        raise ValueError("Model architecture not supported for activation extraction.")

    # Register the forward hook
    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image.to(device))

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

def calculate_similarity(original, processed):
    original_np = np.array(original)
    processed_np = np.array(processed)

    # Calculate structural similarity (SSIM) with a smaller window size and correct channel axis
    similarity_score, _ = ssim(original_np, processed_np, full=True, multichannel=True, win_size=3, channel_axis=-1)

    # Calculate pixel difference
    pixel_difference = np.sum(np.abs(original_np - processed_np)) / original_np.size

    return similarity_score, pixel_difference


@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    llm_explanation = ""
    selected_llm_model = "gpt-3.5-turbo"  # Default LLM model
    selected_pytorch_model = "ResNet18"  # Default PyTorch model
    similarity_score, pixel_difference = None, None

    if request.method == 'POST':
        file = request.files['image']
        optimization_option = request.form.get('optimization_option')
        selected_llm_model = request.form.get('llm_model', "gpt-3.5-turbo")
        selected_pytorch_model = request.form.get('pytorch_model', "ResNet18")

        model = initialize_model(selected_pytorch_model)

        if file:
            img = Image.open(file).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Generate the original activation map and visualize it
            original_activation_map = get_activation_maps(model, img_tensor)
            original_image = visualize_activation_map(original_activation_map)

            if optimization_option != "none":
                # Convert the model parameters to JAX-compatible format
                params = convert_params_to_jax({name: param for name, param in model.named_parameters()})
                processed_params, explainability_output, technique = process_jax(params, optimization_option)
                
                # Update the PyTorch model with the optimized parameters
                update_model_parameters(model, processed_params)

                # Generate the activation map after optimization and visualize it
                optimized_activation_map = get_activation_maps(model, img_tensor)
                compressed_image = visualize_activation_map(optimized_activation_map)

                # Calculate similarity between the original and optimized activation maps
                similarity_score, pixel_difference = calculate_similarity(
                    Image.open(io.BytesIO(base64.b64decode(original_image))), 
                    Image.open(io.BytesIO(base64.b64decode(compressed_image)))
                )

                # Generate LLM-based explainability report
                framework = "PyTorch (with JAX processing)"
                llm_explanation = generate_llm_explanation(selected_pytorch_model, framework, technique, explainability_output, similarity_score, pixel_difference, selected_llm_model)
            else:
                compressed_image = original_image
                explainability_output = "No JAX-based operations were applied. The displayed activation maps are unmodified."
                llm_explanation = Markup(explainability_output.replace("\n", "<br>"))
                similarity_score, pixel_difference = calculate_similarity(
                    Image.open(io.BytesIO(base64.b64decode(original_image))), 
                    Image.open(io.BytesIO(base64.b64decode(original_image)))
                )

            return render_template(
                'index.html',
                original_image=original_image,
                compressed_image=compressed_image,
                explainability_output=llm_explanation,
                selected_llm_model=selected_llm_model,
                selected_pytorch_model=selected_pytorch_model,
                similarity_score=similarity_score,
                pixel_difference=pixel_difference
            )

    return render_template('index.html', original_image=None, compressed_image=None, explainability_output=None)

def generate_llm_explanation(model_name, framework, technique, explainability_output, similarity_score, pixel_difference, llm_model):
    prompt = f"""
    You are tasked with explaining a neural network model optimization. The model is a {model_name} using {framework}. The applied technique was {technique}. The similarity score (SSIM) is {similarity_score:.2f} and the pixel difference is {pixel_difference:.2f}.

    {explainability_output}

    Generate a concise, clear report summarizing key changes and the impact. Keep the explanation under 10 sentences.
    """

    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are an AI that generates concise and informative explainability reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,  # Increased token limit
        temperature=0.525  # Reduced temperature for more focused output
    )

    # Collect the full response
    llm_output = response['choices'][0]['message']['content'].strip()

    # Replace newlines with HTML line breaks
    formatted_explanation = Markup(llm_output.replace("\n", "<br>"))

    return formatted_explanation

if __name__ == '__main__':
    app.run(debug=True)
