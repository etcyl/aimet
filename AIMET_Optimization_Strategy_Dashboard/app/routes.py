from flask import render_template, request, jsonify
from app import app, socketio
import torch
import torchvision.models as pytorch_models
import tensorflow as tf
import openai
import os
import sys
import datetime

# Check for the OPENAI_API_KEY environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)
else:
    openai.api_key = openai_api_key

# Define available models for each framework
AVAILABLE_MODELS = {
    "pytorch": {
        "resnet18": pytorch_models.resnet18,
        "mobilenet_v2": pytorch_models.mobilenet_v2,
        "vgg16": pytorch_models.vgg16,
        "densenet121": pytorch_models.densenet121,
        "alexnet": pytorch_models.alexnet
    },
    "tensorflow": {
        "ResNet50": tf.keras.applications.ResNet50,
        "MobileNet": tf.keras.applications.MobileNet,
        "VGG19": tf.keras.applications.VGG19,
        "InceptionV3": tf.keras.applications.InceptionV3
    }
}

# Store selected model and its details
current_model = {"framework": "pytorch", "name": "resnet18", "model": pytorch_models.resnet18(pretrained=True)}

# Store the latest LLM suggestions globally
latest_suggestion = ""

# Function to get model parameters
def get_model_parameters(model, framework):
    if framework == "pytorch":
        num_params = sum(p.numel() for p in model.parameters())
        return {
            "num_layers": len(list(model.children())),
            "num_params": num_params
        }
    elif framework == "tensorflow":
        num_params = model.count_params()
        return {
            "num_layers": len(model.layers),
            "num_params": num_params
        }

# Function to get layer-by-layer details
def get_layer_details(model, framework):
    layer_details = []
    if framework == "tensorflow":
        dummy_input = tf.random.normal([1] + list(model.input_shape[1:]))
        model(dummy_input)  # Run a forward pass to populate output shapes
        
        for idx, layer in enumerate(model.layers):
            output_shape = getattr(layer, 'output_shape', "N/A")
            if output_shape == "N/A" and hasattr(layer, 'output'):
                output_shape = layer.output.shape
            
            layer_details.append({
                "layer_number": idx + 1,
                "layer_type": str(layer.__class__.__name__),
                "output_shape": str(output_shape),
                "num_params": layer.count_params()
            })
    elif framework == "pytorch":
        for idx, layer in enumerate(model.children()):
            layer_details.append({
                "layer_number": idx + 1,
                "layer_type": str(layer.__class__.__name__),
                "output_shape": str(list(layer.parameters())[0].shape if len(list(layer.parameters())) > 0 else "N/A"),
                "num_params": sum(p.numel() for p in layer.parameters())
            })
    return layer_details

# Enhanced function to parse ChatGPT output and format it correctly as steps
def format_chatgpt_output(output):
    steps = output.split("Step ")
    
    formatted_steps = []
    for step in steps:
        if step.strip():
            if "```" in step:
                parts = step.split("```")
                formatted_steps.append(f"Step {parts[0].strip()}")
                for idx in range(1, len(parts)):
                    if idx % 2 == 1:
                        formatted_steps.append(f"```{parts[idx]}```")
                    else:
                        formatted_steps.append(parts[idx].strip())
            else:
                formatted_steps.append(f"Step {step.strip()}")
    
    return "\n".join(formatted_steps)

# Updated ChatGPT Integration for Model-Specific AIMET Suggestions
def get_chatgpt_suggestions(framework, model_name, model_parameters):
    global latest_suggestion
    prompt = f"""
    Given the architecture and parameters of the selected model ({model_name}) running on {framework.upper()}, provide a clear guide for optimizing the model using AIMET. The guide should focus on:
    - Evaluating which AIMET features are most relevant to the model (e.g., Layer Sensitivity Analysis, Quantization-Aware Training, Cross-Layer Equalization).
    - The decision-making process for selecting between per-channel or per-tensor quantization, considering the model's structure and layers.
    - Detailed, step-by-step instructions that balance the use of AIMET tools, including compression strategies (like pruning or transfer learning) with theoretical explanations for each step.
    - The guide should also highlight specific considerations relevant to the chosen model and its architecture.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI expert providing recommendations for optimizing machine learning models using AIMET."},
            {"role": "user", "content": prompt}
        ]
    )
    
    raw_response = response['choices'][0]['message']['content']
    formatted_response = format_chatgpt_output(raw_response)

    # Prepare the formatted content as if it were saved in the file
    framework = current_model["framework"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_header = f"### Model: {model_name} ({framework.upper()})\n### Generated: {timestamp}\n\n"
    formatted_output = file_header + formatted_response

    # Save the output to a temporary file
    temp_filepath = f"temp_{framework}_{model_name}_suggestion.txt"
    with open(temp_filepath, "w") as temp_file:
        temp_file.write(formatted_output)

    latest_suggestion = formatted_output

    # Read from the temporary file and return its content
    with open(temp_filepath, "r") as temp_file:
        pretty_output = temp_file.read()

    return pretty_output

@app.route('/')
def index():
    model_name = current_model["name"]
    framework = current_model["framework"]
    model = current_model["model"]
    model_parameters = get_model_parameters(model, framework)
    layer_details = get_layer_details(model, framework)
    return render_template('index.html', model_name=model_name, parameters=model_parameters, available_models=AVAILABLE_MODELS[framework].keys(), layers=layer_details)

@app.route('/update_model', methods=['POST'])
def update_model():
    global current_model
    selected_framework = request.form['framework']
    selected_model_name = request.form['model']
    
    socketio.emit('downloading_model', {"downloading": True})
    
    if selected_framework == "pytorch":
        selected_model = AVAILABLE_MODELS[selected_framework][selected_model_name](pretrained=True)
    elif selected_framework == "tensorflow":
        selected_model = AVAILABLE_MODELS[selected_framework][selected_model_name](weights='imagenet')

    current_model = {"framework": selected_framework, "name": selected_model_name, "model": selected_model}
    
    model_parameters = get_model_parameters(selected_model, selected_framework)
    layer_details = get_layer_details(selected_model, selected_framework)
    
    socketio.emit('downloading_model', {"downloading": False})
    
    return jsonify({"parameters": model_parameters, "layers": layer_details})

@app.route('/get_suggestions')
def get_suggestions():
    framework = current_model["framework"]
    model_name = current_model["name"]
    model = current_model["model"]
    model_parameters = get_model_parameters(model, framework)
    suggestion = get_chatgpt_suggestions(framework, model_name, model_parameters)

    # Wrap the suggestion in <pre> tags for better formatting
    suggestion_formatted = f"<pre>{suggestion}</pre>"

    return jsonify({"suggestion": suggestion_formatted})


@app.route('/save_suggestion', methods=['POST'])
def save_suggestion():
    global latest_suggestion
    save_directory = "llm_suggestion"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    framework = current_model["framework"]
    model_name = current_model["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{framework}_{model_name}_llm_response_gpt4_{timestamp}.txt"
    filepath = os.path.join(save_directory, filename)

    with open(filepath, "w") as f:
        f.write(latest_suggestion)

    return jsonify({"message": f"Suggestions saved to {filepath}"})
