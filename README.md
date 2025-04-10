# 🖼️ AI Image Generation
A Python-based application that leverages state-of-the-art AI models to generate images from text prompts or other inputs. Built for researchers, developers, and creators interested in exploring generative models for image synthesis.

## ⚙️ Deploy on RunPod (Recommended: 48GB VRAM)
To use this application with large generative models (e.g., Stable Diffusion XL), it is recommended to use a **GPU with at least 48GB VRAM**. RunPod.io is an excellent choice for scalable, on-demand GPU compute.

## 🐳 Steps to Launch a Pod:
1. Create a RunPod Account:
Sign up at https://www.runpod.io and verify your email.

2. Start a New Pod:
    * Go to "My Pods" → "Create Pod"
    * Select an image such as runpod/pytorch or runpod/python as the base container
    * Choose a GPU instance type with 48 GB VRAM or higher (e.g. A6000 or higher)
    * Enable Docker with SSH access and persistent volume (optional but recommended)
    * Ensure port 7860 is forwarded

3. Clone Your Repository inside the pod:
    ```bash
    git clone https://github.com/jtorregrosa/workshop-ai-image.git
    cd workshop-ai-image
    ```

4. Follow the Setup Instructions below to create a virtual environment, install dependencies, and authenticate.

## 🚀 Features
🔮 Text-to-image generation using pre-trained models

🧪 Easily extensible architecture for different model backends (e.g. Diffusion, GANs)

🔧 Clean Python project structure powered by Poetry

🌐 Hugging Face integration for model loading and authentication

## 🛠️ Setup Instructions (Local or Remote)
1. Prepare a Virtual Environment
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. Install Required Tools
    ```bash
    pip install -U pip setuptools
    pip install poetry
    poetry completions bash >> ~/.bash_completion
    ```

3. Install Dependencies
    ```bash
    poetry install
    ```

4. Authenticate with Hugging Face
    ```bash
    git config --global credential.helper store
    huggingface-cli login
    ```

5. Run the Application
    ```bash
    poetry run python src/workshop-ai-image/app.py
    ```
