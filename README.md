![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

# AlphaDent AI

AlphaDent AI is an automated dental caries detection and diagnosis assistance system based on YOLOv8 instance segmentation.

## Features

- High Precision Instance Segmentation: Detects and segments dental caries based on G.V. Black classification using YOLOv8 architecture.
- Interactive Web Interface: Provides a user-friendly web UI built with Streamlit, enabling real-time image upload and analysis.
- Advanced Visualization: Utilizes Plotly to render interactive, hoverable segmentation masks directly on the browser without occluding the original clinical photos.
- Dual Model Support: Switch seamlessly between a simplified 4-class model and a detailed 9-class diagnostic model.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/USERNAME/AlphaDent.git
cd AlphaDent
pip install -r requirements.txt
```

## Usage

To start the Streamlit web application locally, run:

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`.

## Deployment

The repository is configured for seamless deployment on Streamlit Cloud. 
Simply connect the repository to your Streamlit Cloud account to host the application online.

## Architecture & Logic

The system utilizes YOLOv8 weights (e.g., `yolov8x_AlphaDent_9_classes_960px.pt`) pre-loaded into the global context for low-latency inference. 
Prediction outputs are intercepted and converted into Plotly scatter plots (polygons) overlaid on the original image, allowing dynamic interaction.
