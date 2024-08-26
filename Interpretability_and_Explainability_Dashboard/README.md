<p align="center">
  <img src="example_images/example3_Screenshot 2024-08-26 013110.png" alt="Visualization and explainability dashboard." width="400" />
</p>

# Model Interpretability and Explainability Dashboard

This project demonstrates a Flask-based web application that allows users to visualize model activation maps before and after optimizations like pruning and quantization. The tool supports both PyTorch and JAX, enabling users to see the impact of these optimizations on a computer vision model. Additionally, the dashboard allows users to select from different PyTorch models and choose which LLM model to use for generating explainability reports. The tool also provides metrics for evaluating the similarity between pre- and post-optimization visualizations.

## Features
- **Model Selection and Visualization:** Upload an image and visualize activation maps. Choose between different PyTorch models (e.g., ResNet18, ResNet34, VGG16, DenseNet121) for visualization and optimization.
- **Quantization and Pruning Options:** Apply model quantization and pruning using JAX, allowing comparisons of pre- and post-optimization effects.
- **LM-Generated Explainability Reports:**  Get detailed explainability reports powered by LLMs (e.g., GPT-3.5, GPT-4) to better understand how the applied optimizations impact the model’s behavior.
- **Pixel Similarity and Comparison Metrics:** Evaluate the impact of optimizations using Structural Similarity Index (SSIM) and pixel difference metrics, helping users quantify the changes introduced by optimizations.

## Requirements
* Python 3.8+
* Flask
* JAX
* PyTorch
* OpenAI API access

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create and activate a Python virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install dependencies
```bash
python install -r requirements.txt
```

### 4. Run the Flask app
```bash
flask run
```
The app will be running at http://127.0.0.1:5000.
In the terminal, control click the http web address to view the app in the browser.

## Usage instructions
1. Upload an image using the provided form.
2. Choose from various PyTorch models (ResNet18, ResNet34, VGG16, DenseNet121) for visualization.
3. Select whether to apply pruning, quantization, or both (compression) using JAX.
4. Choose the LLM model (e.g., GPT-3.5, GPT-4) for generating a concise explainability report that considers both the optimization technique and pixel similarity metrics.
3. View the original and optimized activation maps side-by-side, along with the detailed LLM-generated explainability report.
4. Experiment with different settings to observe how model optimization affects interpretability and performance.
