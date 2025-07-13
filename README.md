# 🚀 PRODIGY\_GA\_01 — GPT-2 Fine-Tuning Project

This repository contains **Task 1: GPT-2 Fine-Tuning Project** as part of my Generative AI Internship at Prodigy Infotech, where I have fine-tuned the GPT-2 model on custom data.

## 💡 Project Overview

In this task, I have:

- Fine-tuned the GPT-2 model using Hugging Face Transformers and PyTorch.
- Used a custom text dataset (`mydata.txt`) for training.
- Generated text samples using the fine-tuned model.
The goal is to generate more personalized and meaningful text outputs based on specific data.

## 📄 Files

- `finetune_gpt2.py`: Script to fine-tune the GPT-2 model on custom data.
- `generate_text.py`: Script to generate text from the fine-tuned GPT-2 model.
- `mydata.txt`: Custom text dataset used for fine-tuning.
- `.gitignore`: To ignore unnecessary files in version control.
- `cached_lm_GPT2Tokenizer_128_mydata.txt`: Tokenizer cache file.

## ⚙️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Accelerate

## 💬 How to Run

1️⃣ Clone the repository:
```
git clone https://github.com/anushkakashyap31/PRODIGY_GA_01.git
cd PRODIGY_GA_01
```

2️⃣ Install dependencies (inside virtual environment):
```
pip install -r requirements.txt
```

3️⃣ Run fine-tuning:
```
python finetune_gpt2.py
```

4️⃣ Generate text:
```
python generate_text.py
```

## 🎯 Objectives

- Learn the process of fine-tuning large language models.
- Understand text generation using custom-trained models.
- Improve practical skills with Hugging Face Transformers and PyTorch.

## 👩‍💻 Author

- **Anushka Kashyap**

## ⭐ Acknowledgements

- Prodigy Infotech internship guidance
- Hugging Face Transformers library

✨ Feel free to ⭐ star this repository if you find it helpful!
