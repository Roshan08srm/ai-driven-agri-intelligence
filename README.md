# AI Driven Agri Intelligence

## 👥 Project Architecture & Team Roles

This repository hosts the centralized codebase for our 2-person engineering project. Because I handled the final module consolidation and code tracking from my account, it is hosted under my personal profile. The workload was divided cleanly as follows:

* **Roshan Kantipudi (Lead Backend AI Architect)**: Owned the entire server-side AI engineering. Built the Python text-extraction pipeline using the Google Cloud Vision API, engineered the Y-Axis Line Reconstruction logic, developed the 2-Stage Heuristic Filtering Layer (~95% accuracy), and designed the time-series Proactive Expiry Engine.
* **Satya (Frontend & Database Specialist)**: Developed the client-side user interface layouts using React Native/Expo and managed local relational database schemas.

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
