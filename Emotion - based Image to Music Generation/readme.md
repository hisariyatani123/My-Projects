# Bridging Paintings and Music - Emotion-Based Music Generation

An AI system that generates music reflecting emotions perceived in paintings, using deep learning models and multimodal integration.

## Project Overview
**Key Features:**
- Converts paintings to emotional text descriptions
- Generates music matching emotional content
- Four model variants with progressive improvements
- Comprehensive evaluation metrics (FAD, CLAP, THD)
- Pre-trained models and dataset available

## Repository Structure
├── data_model/           # Dataset directory
├── gen_audio/            # Generated music samples
├── models/               # Model checkpoints
├── processed_data/       # Pre-processed tensors
├── wandb/                # Training logs
├── calc_fad.py           # FAD metric calculation
├── calc_kl.py            # KL divergence calculation
├── gen.py                # Music generation script
├── main.ipynb            # Main workflow notebook
└── evaluation.ipynb      # Evaluation metrics notebook

## Installation
Clone repository: git clone https://github.com/hisariyatani123/MY-Projects/music-generation.git

## Create virtual environments:

	python -m venv main_env
	python -m venv eval_env

## Install dependencies:

	source main_env/bin/activate
	pip install torch torchvision transformers wandb jupyter

### Evaluation environment
	source eval_env/bin/activate
	pip install librosa pandas numpy matplotlib

## Dataset & Models
Resource	Links

Dataset		link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/EbNnv4y4dRtHnnjaPCQY5v0B0jFbMDy5UWxTiHt7BchAnQ
		link 2:	https://drive.google.com/file/d/1n-uLQskwO5eO3YyNQDZwN5kh0ynj8brI/view?

Model Checkpoints	link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/Eb_Cig6YtZFCsoBttRcfLmwBViqmRK9GTY9ZikgRjCb5vA
			link 2: https://drive.google.com/file/d/1aEo1DzL5lI9Ih8b7EkFGTL4zpC7yqaBa/view?usp=sharing

Generated Audio		link 1: https://qmulprod-my.sharepoint.com/:u:/g/personal/ec23691_qmul_ac_uk/EdAk3Df-pwJOg-6ZOJBNlwMBjjHxnFKlFdn5Vcp5ybZnnw
			link 2: https://drive.google.com/file/d/1qZH7kiMaqvwMkWAtBYqQagV-FCwH_m-a/view?usp=sharing

# In main.ipynb
%run train_mod.py --epochs 40 --batch_size 16 --lr 1e-5

# Generate music from painting
%run gen.py --model models/MG-S_Optimized.pt --image data_model/happy/painting_001.jpg


# In evaluation.ipynb
%run calc_fad.py --reference real_audio/ --generated gen_audio/
%run calc_kl.py --dataset processed_data/ --model models/MG-S_Optimized.pt

## Model Variants
Model		Key Features	
MG-S Emotive	Single emotion labels	
MG-S Narrative	Enhanced image captions	
MG-S Lyrical	LLM-enhanced descriptions	
MG-S Optimized	Architecture improvements	

## Evaluation Metrics
Metric	MG-S Emotive	MG-S Optimized
FAD ↓	7.02		5.54
CLAP ↑	0.075		0.13
THD ↓	1.79		1.75

### For more details on results please check the results.docx file


