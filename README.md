# ContentV: Text to Video Running Locally

<div align="center">
<p align="center">
  <a href="https://contentv.github.io">
    <img
      src="https://img.shields.io/badge/Gallery-Project Page-0A66C2?logo=googlechrome&logoColor=blue"
      alt="Project Page"
    />
  </a>
  <a href='https://arxiv.org/abs/2506.05343'>
    <img
      src="https://img.shields.io/badge/Tech Report-ArXiv-red?logo=arxiv&logoColor=red"
      alt="Tech Report"
    />
  </a>
  <a href="https://huggingface.co/ByteDance/ContentV-8B">
    <img 
        src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="Model"
    />
  </a>
  <a href="https://github.com/bytedance/ContentV">
    <img 
        src="https://img.shields.io/badge/Code-GitHub-orange?logo=github&logoColor=white" 
        alt="Code"
    />
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img
      src="https://img.shields.io/badge/License-Apache 2.0-5865F2?logo=apache&logoColor=purple"
      alt="License"
    />
  </a>
</p>
</div>
# ContentV Video Generator

This repository hosts a Gradio web application for generating high-quality videos from text prompts using the `ByteDance/ContentV-8B` model. It provides a user-friendly interface to control various aspects of video generation, including prompts, frame count, and random seed.

## Features

* **Text-to-Video Generation:** Create videos from detailed text descriptions.

* **Customizable Prompts:** Input positive and negative prompts to guide video content.

* **Adjustable Frame Count:** Control the length of the generated video.

* **Reproducible Results:** Specify a random seed to get consistent outputs.

* **Intuitive Gradio UI:** Easy-to-use web interface.

* **Live Progress Bar:** Monitor the video generation and encoding process directly in the UI.

* **Teal Theme:** A custom aesthetic for the Gradio interface.

## Hardware Requirements

To run this application locally, a high-performance GPU with substantial VRAM is essential. We strongly recommend the following specifications for optimal performance:

* **GPU:** NVIDIA GeForce RTX 5090 or equivalent.

* **VRAM:** A minimum of 32 GB of VRAM.

**Please note:** If your system does not meet these specifications, you may encounter out-of-memory errors, extremely slow processing times, or be unable to run the model at all. Ensure your GPU drivers are up to date.

## Installation and Running the Gradio App

Follow these steps to set up and run the ContentV Video Generator Gradio application on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.11** (or a compatible version). You can download it from [python.org](https://www.python.org/) or use a tool like Anaconda/Miniconda.

* **Git** (for cloning the repository).

### Step 1: Clone the Repository

First, clone this GitHub repository to your local machine:

```
git clone [https://github.com/jjmlovesgit/contentv.git](https://github.com/jjmlovesgit/contentv.git)
cd contentv

```

### Step 2: Set Up Python Environment

It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects.

**Using Conda (Recommended):**

```
conda create -n contentv_env python=3.11
conda activate contentv_env

```

**Using venv (Standard Python Virtual Environment):**

```
python -m venv contentv_env
.\contentv_env\Scripts\activate  # On Windows
source contentv_env/bin/activate # On macOS/Linux

```

### Step 3: Install Dependencies

With your virtual environment activated, install all the required Python packages using the `requirements.txt` file. This file contains exact versions for reproducibility.

```
pip install -r requirements.txt

```

This command will install all the necessary libraries, including `torch` (with CUDA support for your NVIDIA GPU), `diffusers`, `gradio`, and others your application depends on.

### Step 4: Run the Gradio Application

Once all dependencies are installed, you can start the Gradio web application:

```
python contentv_gradio.py

```

### Step 5: Access the App in Your Browser

After running the command, you will see output in your terminal similar to this:

```
Models loaded successfully!
* Running on local URL:  [http://127.0.0.1:7860](http://127.0.0.1:7860)
* To create a public link, set `share=True` in `launch()`.

```

Open your web browser and navigate to the `http://127.0.0.1:7860` (or whatever local URL is displayed) to access the ContentV Video Generator UI.

## How to Use the App

Once the Gradio app is loaded in your browser:

* **Prompt:** Enter a detailed description of the video you want to generate.

* **Negative Prompt:** Enter concepts you want the video to avoid.

* **Number of Frames:** Adjust the slider to control the length of the output video.

* **Seed:** Input an integer value to ensure reproducible video generations. Using the same seed with the same prompts will yield the same video.

* **Submit:** Click the "Submit" button to start the video generation process.

## Understanding the Progress Bar

The application provides real-time feedback through a dynamic progress bar:

* **Frame Generation Phase:** The bar will fill from 0% to 100%, indicating the progress of the core video frame generation (diffusion) process. The message will show `Generating frames... (Step X/50)`.

* **Transition:** Upon completion of frame generation, the bar will instantly display 100% with the message `Frames generated successfully! Starting video encoding....`

* **Video Encoding Phase:** The same progress bar will then switch to an indeterminate spinner and display the message `Encoding video... This may take a few minutes.`. This phase involves compressing the raw frames into an `.mp4` file and saving it to disk. This can be a CPU-intensive process and may take several minutes depending on your hardware.

* **Completion:** Once encoding is complete, the progress indicator will disappear, and the generated video will be displayed in the "Generated Video" output area and automatically saved as `generated_video.mp4` in your project directory.

## Project Structure

* `contentv_gradio.py`: The main script that sets up and runs the Gradio web interface.

* `contentv_pipeline.py`: (Assumed) Your custom ContentV diffusion pipeline logic.

* `contentv_transformer.py`: (Assumed) Contains the SD3Transformer3DModel definition.

* `requirements.txt`: Lists all Python package dependencies and their exact versions.

* `assets/`: (Optional) Directory for images or other static assets.

* `__init__.py`, `.gitignore`, `LICENSE.txt`, `Notice`, `README.md`, `demo.py`: Standard repository files.

## Special Considerations for NPU Users (Optional)

If you are using an Ascend NPU for acceleration, ensure your environment is configured correctly. The script includes conditional imports for `torch_npu` based on the `USE_ASCEND_NPU` environment variable. You might need to set this before running:

```
# Example for Windows (in Command Prompt)
set USE_ASCEND_NPU=1
python contentv_gradio.py

# Example for Linux/macOS (in Terminal)
export USE_ASCEND_NPU=1
python contentv_gradio.py

```

(Please note that `torch_npu` and `transfer_to_npu` require specific NPU drivers and installations beyond standard Python packages.)

## License

\[TODO: Add your project's license here, e.g., MIT, Apache 2.0, etc.\]
