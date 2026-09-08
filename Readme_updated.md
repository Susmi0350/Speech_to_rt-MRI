
Speech-to-rtMRI: Speech-Driven Articulatory Motion Prediction
An end-to-end deep learning pipeline for mapping speech audio to real-time MRI (rt-MRI) motion representations using Wav2Vec2 and a Transformer-based sequence model.
📌 Overview
Speech-to-rtMRI is a deep learning project developed during my internship to investigate the relationship between speech acoustics and articulatory movements captured using real-time Magnetic Resonance Imaging (rt-MRI).
Speech production involves coordinated movements of the tongue, lips, jaw, and other vocal-tract structures. Real-time MRI provides a non-invasive way to capture these movements over time. This project explores whether these MRI-derived articulatory motion patterns can be predicted from speech audio.
The system takes speech audio as input, extracts meaningful speech representations using Wav2Vec2, and uses a Transformer-based decoder to predict the corresponding temporal MRI motion representation.
Core Pipeline
Speech Audio → Wav2Vec2 → Speech Features → Transformer Decoder → Predicted rt-MRI Motion
🎯 Objectives
The main objectives of the project are to:
Develop an end-to-end speech-to-articulatory-motion prediction pipeline.
Extract meaningful acoustic representations from speech using Wav2Vec2.
Process and prepare real-time MRI sequences for deep learning.
Establish temporal correspondence between speech and MRI data.
Model the relationship between acoustic features and articulatory motion.
Generate predicted MRI motion sequences from speech input.
Provide a foundation for future research in speech production and articulatory modelling.
🧠 Problem Statement
Real-time MRI can provide detailed information about the movement of the vocal tract during speech. However, collecting MRI data is comparatively complex, resource-intensive, and not always practical for large-scale applications.
This project investigates the following research question:
Can articulatory motion captured using real-time MRI be predicted from speech audio using deep learning?
To explore this, paired speech and MRI recordings are processed and used to train a model that learns the relationship between audio features and articulatory motion.
🔄 System Architecture
Input Video
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        Speech Audio          rt-MRI Video
              │                   │
              ▼                   ▼
       Audio Preprocessing   Frame Extraction
              │                   │
              ▼                   ▼
           WAV File         Grayscale Frames
              │                   │
              ▼                   ▼
          Wav2Vec2          Flatten & Normalize
              │                   │
              ▼                   ▼
      Speech Representations   MRI Motion Data
              │                   │
              └─────────┬─────────┘
                        ▼
                Temporal Alignment
                        │
                        ▼
              Transformer Decoder
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
        Positional   Temporal   Subject
         Encoding      Bias     Embedding
             │          │          │
             └──────────┼──────────┘
                        ▼
              Autoregressive Prediction
                        │
                        ▼
              Predicted MRI Motion
                        │
                        ▼
                    .npy Output
🏗️ Methodology
1. Data Preprocessing
The original recordings contain synchronized speech and real-time MRI information.
The preprocessing pipeline separates and prepares these modalities for model training.
Audio Processing
Audio is extracted from the source recordings and converted into a model-compatible format.
Sampling rate: 16 kHz
Channels: Mono
Format: 16-bit PCM WAV
The processed audio is then passed to the Wav2Vec2 processor.
MRI Processing
MRI videos are processed frame by frame.
The preprocessing pipeline:
Reads the MRI video.
Extracts individual frames.
Converts frames to grayscale.
Flattens each frame into a one-dimensional representation.
Normalizes pixel values.
Stores the resulting sequences as NumPy arrays (.npy).
This converts the MRI video data into a numerical temporal representation suitable for deep learning.
🎙️ 2. Speech Representation using Wav2Vec2
The project uses Wav2Vec2 as the speech representation model.
The implementation is based on:
facebook/wav2vec2-base-960h
Wav2Vec2 processes the raw speech waveform and extracts contextualized acoustic representations.
The resulting speech features have a dimensionality of 768, which are subsequently projected into the feature space used by the Transformer model.
Raw Audio
    ↓
Wav2Vec2
    ↓
768-dimensional Speech Features
    ↓
Feature Projection
    ↓
Transformer
This allows the model to work with learned speech representations rather than relying only on manually engineered audio features.
🧲 3. Real-Time MRI Representation
The rt-MRI recordings provide the target articulatory motion information.
Instead of directly treating the MRI video as a conventional image-generation problem, the project converts each MRI frame into a numerical representation.
MRI Video
    ↓
Frame Extraction
    ↓
Grayscale Conversion
    ↓
Flattening
    ↓
Normalization
    ↓
Temporal MRI Representation
The processed representations are stored as .npy files and used as the target sequences during training.
👤 4. Subject-Specific Conditioning
Speech production and vocal-tract movement can vary significantly between individuals.
To account for subject-specific characteristics, the model incorporates a subject representation.
The subject information is transformed into an embedding and provided to the Transformer-based prediction model.
Subject ID
    ↓
One-Hot Representation
    ↓
Linear Embedding
    ↓
Transformer Decoder
A subject-specific template is also generated from MRI data and used as part of the prediction pipeline.
🤖 5. Transformer-Based Motion Prediction
The core prediction component uses a Transformer Decoder to model the temporal relationship between speech representations and MRI motion.
The decoder incorporates:
Speech feature projections
Multi-head attention
Positional encoding
Temporal attention bias
Subject embeddings
Autoregressive sequence generation
The Transformer learns how changes in the acoustic speech representation correspond to changes in the articulatory motion representation.
Prediction Flow
Speech Features
      │
      ▼
Feature Projection
      │
      ▼
Transformer Decoder
      │
      ├── Positional Encoding
      ├── Temporal Bias
      └── Subject Embedding
      │
      ▼
Autoregressive Motion Prediction
      │
      ▼
Predicted MRI Representation
⏱️ 6. Temporal Alignment
Speech audio and MRI sequences operate at different temporal resolutions. Therefore, temporal alignment is an important part of the pipeline.
The implementation processes the sequences using overlapping temporal chunks.
Key parameters include:
Parameter
Value
Audio sampling rate
16,000 Hz
MRI frame rate
~83 FPS
Training chunk size
200 MRI frames
Chunk stride
100 frames
The overlapping-window approach allows longer recordings to be processed in manageable segments while preserving temporal continuity between neighbouring segments.
🗃️ Dataset & Data Loading
A custom PyTorch Dataset and DataLoader are used to manage the paired audio and MRI data.
For each sample, the data pipeline handles:
Speech waveform loading
Wav2Vec2 preprocessing
MRI sequence loading
Subject template loading
Subject representation
Corresponding file identification
The project also supports separate training, validation, and testing data configurations.
🏋️ Training Pipeline
The model is trained using paired speech and MRI motion sequences.
Paired Audio + MRI Data
          ↓
      Preprocessing
          ↓
    Temporal Chunking
          ↓
      Wav2Vec2
          ↓
    Speech Features
          ↓
 Transformer Decoder
          ↓
 Predicted MRI Motion
          ↓
 Compare with Ground Truth
          ↓
       MSE Loss
          ↓
    Backpropagation
          ↓
    Adam Optimizer
          ↓
      Model Update
Training Configuration
Component
Implementation
Deep Learning Framework
PyTorch
Speech Encoder
Wav2Vec2
Sequence Model
Transformer Decoder
Loss Function
Mean Squared Error (MSE)
Optimizer
Adam
Hardware
CUDA / GPU
The training pipeline also includes checks for invalid loss values such as NaN or infinite values.
📊 Loss Function
The model uses Mean Squared Error (MSE) to measure the difference between the predicted MRI motion representation and the corresponding ground-truth representation.
The objective can be expressed as:
MSE = Mean((Prediction − Ground Truth)²)
Minimizing this loss encourages the model to generate motion representations that are closer to the observed MRI sequence.
🔮 Inference
During inference, the trained model receives speech audio and subject-specific information.
The audio is processed through Wav2Vec2, and the resulting representations are passed to the Transformer decoder.
The model then generates the MRI motion sequence autoregressively.
Input Speech
     ↓
Wav2Vec2
     ↓
Speech Features
     ↓
Transformer Decoder
     ↓
Autoregressive Generation
     ↓
Subject-Specific Conditioning
     ↓
Predicted MRI Motion
     ↓
.npy Output
The generated predictions can be further analysed or visualized against the corresponding ground-truth MRI representations.
📂 Repository Structure
Speech_to_rt-MRI/
│
├── main_intern.py
│   └── Main training, validation and inference pipeline
│
├── wav2vec_intern.py
│   └── Wav2Vec2-based speech encoder
│
├── rt_mri_intern.py
│   └── Transformer-based rt-MRI motion prediction model
│
├── data_preprocessing_intern.py
│   └── Audio and MRI preprocessing
│
├── dataloader_intern.py
│   └── PyTorch Dataset and DataLoader
│
├── template.pkl_intern.py
│   └── Subject-specific MRI template generation
│
├── .npy & .wav_intern.py
│   └── NumPy/WAV data preparation utilities
│
└── README.md
🛠️ Technologies & Tools
Technology
Purpose
Python
Core programming language
PyTorch
Deep learning and model training
Hugging Face Transformers
Wav2Vec2 implementation
Wav2Vec2
Speech representation learning
NumPy
Numerical processing and data storage
OpenCV
MRI video and frame processing
Librosa
Audio processing
FFmpeg
Audio extraction
Pickle
Subject template storage
CUDA
GPU acceleration
tqdm
Progress monitoring
📌 Key Features
🎙️ Speech-driven articulatory motion prediction
🧠 Wav2Vec2-based speech feature extraction
🤖 Transformer-based temporal modelling
🔄 Autoregressive sequence generation
🧲 Real-time MRI data processing
👤 Subject-specific conditioning
⏱️ Temporal alignment between speech and MRI
🧩 Overlapping temporal-window processing
🚀 GPU-accelerated training
💾 NumPy-based prediction storage
📤 Input & Output
Input
The system primarily works with:
Speech Audio
+
Corresponding rt-MRI Data
+
Subject Information
Output
The trained model generates:
Predicted MRI Motion Representation
                ↓
           .npy file
These outputs can subsequently be used for comparison with ground-truth sequences and further visualization or analysis.
🌍 Potential Applications
The research direction explored by this project has potential applications in several areas.
🗣️ Speech Production Research
Studying the relationship between acoustic speech signals and vocal-tract movements.
🔬 Phonetics & Linguistics
Investigating articulatory patterns associated with different speech sounds.
🧑‍⚕️ Speech Therapy Research
Potentially supporting future research into articulatory behaviour and speech assessment.
🧠 Speech Impairment Research
Providing a framework for studying differences in speech production and articulatory motion.
🤖 Human–Computer Interaction
Exploring speech-driven representations of human articulatory behaviour.
Note: These are potential research applications. The current implementation is not a clinically validated diagnostic or therapeutic system.
⚠️ Limitations
The current implementation has several limitations:
Performance depends on the availability and quality of paired speech and MRI data.
Accurate synchronization between audio and MRI is important.
Subject-specific differences may affect generalization.
MRI data processing and model inference can be computationally intensive.
The current model predicts a numerical motion representation rather than directly producing a conventional MRI video.
Further quantitative evaluation and visualization are required to assess reconstruction quality comprehensively.
🚀 Future Scope
Future improvements could include:
Improving cross-subject generalization.
Fine-tuning the Wav2Vec2 encoder.
Exploring larger and more advanced Transformer architectures.
Improving audio–MRI temporal synchronization.
Introducing additional evaluation metrics.
Reconstructing predicted representations into interpretable MRI visualizations.
Developing real-time inference.
Improving robustness across speakers and speaking styles.
Supporting multilingual speech.
Building an interactive visualization interface.
Evaluating the approach on larger and more diverse datasets.
Investigating applications in speech assessment and therapy research.
🎓 Internship Learning & Contribution
This project provided practical experience in developing an end-to-end AI pipeline combining speech processing, deep learning, computer vision, sequence modelling, and medical imaging data.
Key areas of contribution
Designed and implemented the speech-to-rtMRI processing pipeline.
Preprocessed synchronized audio and MRI recordings.
Extracted audio from source recordings.
Converted MRI videos into machine-learning-ready numerical representations.
Developed a custom PyTorch data-loading pipeline.
Integrated Wav2Vec2 for speech representation extraction.
Implemented Transformer-based temporal prediction.
Incorporated subject-specific conditioning.
Implemented temporal chunking and alignment.
Developed model training and inference workflows.
Generated and stored predicted MRI motion sequences.
Worked with GPU-based deep learning workflows.
📚 Technical Summary
Category
Details
Project Type
Deep Learning / Multimodal AI
Primary Input
Speech Audio
Target Data
rt-MRI Motion Representation
Speech Model
Wav2Vec2
Speech Feature Dimension
768
Prediction Model
Transformer Decoder
Sequence Generation
Autoregressive
Subject Conditioning
One-Hot + Linear Embedding
Temporal Processing
Overlapping Chunks
Loss Function
MSE
Optimizer
Adam
Audio Sampling Rate
16 kHz
MRI Frame Rate
~83 FPS
Training Chunk Size
200 frames
Chunk Stride
100 frames
Output Format
.npy
Acceleration
CUDA / GPU
🔗 Repository
GitHub:
https://github.com/Susmi0350/Speech_to_rt-MRI
👩‍💻 Author
Susmitha Guntuku
Developed as part of an internship project focused on speech processing, deep learning, and real-time MRI-based articulatory motion modelling.
⚠️ Disclaimer
This project is intended for research and educational purposes. It is not a clinically validated medical device and should not be used for medical diagnosis, treatment, or clinical decision-making.
