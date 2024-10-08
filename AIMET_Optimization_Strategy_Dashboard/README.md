<p align="center">
  <img src="example_images/example1_Screenshot 2024-08-26 124108.png" alt="Visualization and explainability dashboard." width="400" />
</p>

# LLM AIMET Strategy For Quantization-Aware Training in PyTorch and TensorFlow

This project demonstrates a Flask-based web dashboard that integrates PyTorch and TensorFlow models for LLM guidance on AIMET (AI Model Efficiency Toolkit) quantization-aware training. The dashboard allows users to select models, view detailed layer-by-layer analysis, and get dynamic optimization suggestions using OpenAI's ChatGPT API.

## Features
- **Model Selection:** Choose between PyTorch or TensorFlow frameworks and select models like ResNet, VGG, MobileNet, etc.
- **Layer-by-Layer Analysis:** View detailed information about the selected model, including layer types, output shapes, and the number of parameters.
- **Dynamic Suggestions:** Get real-time optimization recommendations for the selected model using ChatGPT, focusing on AIMET’s capabilities.

## Requirements
* Python 3.8+
* Flask
* PyTorch
* TensorFlow
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

### 4. Set your OpenAI API key
```bash
export OPENAI_API_KEY=<your-api-key>  # On Windows: set OPENAI_API_KEY=<your-api-key>
```

### 5. Run the Flask app
```bash
flask run
```
The app will be running at http://127.0.0.1:5000.
In the terminal, control click the http web address to view the app in the browser.

## Usage instructions
1. Select a framework (PyTorch or TensorFlow) from the dropdown menu.
2. Choose a model from the list of available models.
3. View the detailed layer-by-layer analysis for the selected model.
4. Click on "Get Suggestions" to receive optimization recommendations using ChatGPT, specifically tailored for AIMET integration.
5. When switching frameworks, sometimes the page may not load the new framework model e.g. switching from PyTorch resnet to TensorFlow resnet might not change the display.
If this occurs, select a different model like vgg16, then select the original model (resnet in this case). This should properly update the display.
