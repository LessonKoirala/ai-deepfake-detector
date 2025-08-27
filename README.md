AI Deepfake Detector

## 🔍 Overview

This system provides a dual-model approach to digital media authentication:

- **Image Classification**: Detects AI-generated images vs. human-created content

- **Video Analysis**: Identifies deepfake videos through frame-by-frame processing



The pipeline includes workflow support from data preprocessing to model deployment with a RESTful API interface.


## ✨ Features

- **Video Preprocessing**: Converts video datasets into sequential image frames for model training

- **Dual-Model Architecture**: Separate specialised models for image and video analysis

- **RESTful API**: Flask-based endpoints for real-time media analysis

- **Web Interface**: User-friendly frontend for testing and demonstration

- **Model Management**: Organised storage for trained models, training logs, and metadata

- **Reproducibility**: Complete training pipeline with configurable parameters



## 📁 Project Structure



```

.

├── app.py                      # Main Flask application

├── cleanvideo.py               # Video preprocessing script

├── image_train.py              # Image classification model training

├── train_video.py              # Deepfake video model training

├── deepfake_detector_model.keras  # Trained video analysis model

├── image_model_output/         # Trained models and metadata

│   ├── ai_vs_human.keras

│   ├── audio_model.h5

│   ├── fake_real_detector.keras

│   └── training_metadata.json

├── image_logs/                 # Training logs and checkpoints

│   └── 20250818_105521/

│       ├── train/

│       └── validation/

├── processed_image_data/       # Preprocessed image frames

│   ├── ai/

│   └── real/

├── templates/

│   └── index.html             # Web interface

├── uploads/                   # Temporary file storage

├── archive (1)/              # Raw image dataset

│   ├── test/

│   └── train/

└── videodata/                # Raw video dataset

    ├── ai/

    └── real/

```



## 🛠️ Prerequisites



- **Python 3.8+**

- **pip** (Python package manager)

- Recommended: Virtual environment (venv or conda)



## ⚡ Quick Start



### Installation



1. **Clone the Repository**

   ```bash

   git clone https://github.com/your-username/ai-deepfake-detector.git

   cd ai-deepfake-detector

   ```



2. **Install Dependencies**

   ```bash

   pip install Flask Flask-Cors numpy tensorflow Pillow opencv-python

   ```



### Dataset Preparation



1. **Image Dataset**: Place your image classification dataset in `archive (1)/` with `test/` and `train/` subdirectories

2. **Video Dataset**: Organise video files in `videodata/` with `ai/` and `real/` subfolders



### Model Training Pipeline



1. **Preprocess Video Data**

   ```bash

   python cleanvideo.py

   ```

   *Converts videos to image frames in `processed_image_data/`*



2. **Train Image Classification Model**

   ```bash

   python image_train.py

   ```

   *Generates `image_model_output/ai_vs_human.keras` and `class_names.json`*



3. **Train Deepfake Detection Model**

   ```bash

   python train_video.py

   ```

   *Produces `deepfake_detector_model.keras` in project root*



### Deployment



1. **Start the API Server**

   ```bash

   python app.py

   ```



2. **Access the Application**

   - API: `http://0.0.0.0:5001`

   - Web Interface: `http://0.0.0.0:5001/`



## 🌐 API Endpoints



| Endpoint | Method | Description | Parameters |

|----------|--------|-------------|------------|

| `/` | GET | Serves web interface | - |

| `/analyse/image` | POST | Analyses uploaded image | `file`: image file (multipart/form-data) |

| `/analyse/video` | POST | Analyses uploaded video | `file`: video file (multipart/form-data) |

| `/model/status` | GET | Returns model loading status and versions | - |



### Example API Response



```json

{

  "prediction": "AI-generated",

  "confidence": 0.92,

  "model_version": "1.0.0"

}

```



## 🚀 Usage Examples



### Analyse Image via curl

```bash

curl -X POST -F "file=@test_image.jpg" http://localhost:5001/analyze/image

```



### Analyse Video via curl

```bash

curl -X POST -F "file=@test_video.mp4" http://localhost:5001/analyze/video

```



## 📊 Model Architecture



- **Image Model**: Custom CNN architecture trained on diverse image datasets

- **Video Model**: Two-stage fine-tuning process using temporal features from video frames

- **Input Processing**: Standardised preprocessing pipeline for consistent analysis



## 🔧 Configuration



Training parameters and model configurations can be modified in the respective training scripts:

- `image_train.py`: Image model hyperparameters

- `train_video.py`: Video training pipeline settings

- Metadata stored in `training_metadata.json` for reproducibility



1. Fork the repository

2. Create a feature branch (`git checkout -b feature/amazing-feature`)

3. Commit your changes (`git commit -m 'Add amazing feature'`)

4. Push to the branch (`git push origin feature/amazing-feature`)

5. Open a Pull Request
