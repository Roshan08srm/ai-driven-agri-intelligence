# AI Driven Agri Intelligence



## 👥 Project Architecture & Team Roles

This repository hosts the centralized codebase for our 2-person engineering and research project. Because I handled the final architecture integration and repository tracking from my account, it is hosted under my personal profile. The workload was divided cleanly as follows:

* **Roshan Kantipudi (Lead Data Engineer & Performance Specialist)**: Owned the end-to-end data pipeline and model evaluation architecture. Engineered the logical tracking map using in-memory Pandas DataFrames to manage 28,000 crop images across 69 distinct disease classes. Built the runtime lazy-loading batch generators to prevent GPU memory crashes, implemented tensor normalization pipelines, constructed the multi-model benchmarking framework (evaluating VGG16, ResNet50, and InceptionV3), and integrated the Grad-CAM++ explainability activation layer. Co-authored the final research manuscript tailored to IEEE publication standards.
* **Satya (Frontend Dashboard & Environment Specialist)**: Developed the interactive web diagnostic dashboard interface for end-users, managed local execution environment dependencies, and assisted with initial dataset documentation and environment configurations.
  
This project uses a VGG16 model to detect crop diseases and provides fertilizer recommendations based on crop data.

## Features
- **Disease Detection**: Uses deep learning (VGG16) to identify plant diseases from images.
- **Fertilizer Recommendation**: Suggests suitable fertilizers based on soil and crop parameters.
- **Web Interface**: Easy-to-use interface built with Flask/Streamlit.

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the model file:
   - Download `vgg16_final.h5` and place it in the root directory (file is too large for standard GitHub upload).

## Usage
Run the application:
```bash
python app2.py
```

## Data Files
- `Crop_Data.csv`: Historical crop data.
- `disease_metadata.csv`: Metadata for identified diseases.
- `fertilizers.csv`: Fertilizer database.
