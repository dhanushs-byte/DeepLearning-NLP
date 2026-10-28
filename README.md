# Week 3 Assessment — Deep Learning + NLP

This repository implements the four tasks in the Week 3 Weekly Assessment PDF:
1. Neural network fundamentals with a NumPy XOR model.
2. PyTorch training pipeline on the Breast Cancer Wisconsin dataset.
3. Classical NLP using TF-IDF and Logistic Regression, with optional GloVe embeddings.
4. DistilBERT fine-tuning and a FastAPI `/predict` endpoint.

## Requirements

- Python 3.10–3.12 recommended. Newer Python releases may not yet be supported by every PyTorch/Transformers dependency.
- VS Code and Git.
- Internet access for installing packages and downloading Hugging Face/GloVe models.
- For Task 4, Google Colab or a GPU is recommended but not mandatory.

## Project layout

```text
Task1_Neural_Networks/task1.py
Task2_PyTorch/task2.py
Task3_Classical_NLP/task3.py
Task4_Transformers_API/train_model.py
Task4_Transformers_API/app.py
Task4_Transformers_API/requirements.txt
```

## Setup in VS Code (Windows PowerShell)

Open this repository folder in VS Code, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy matplotlib torch pandas scikit-learn joblib
pip install -r Task4_Transformers_API/requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 or 3.12 from python.org, reopen VS Code, and select it using **Python: Select Interpreter**. If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` directly.

## Task 1 — Neural Network Fundamentals

```powershell
python Task1_Neural_Networks/task1.py
```

The script prints softmax values, confirms gradients with PyTorch autograd, trains a 2-4-1 NumPy network on XOR for 5,000 epochs, and compares learning rates 0.01, 0.5 and 5. It saves `task1_loss.png` in the current working directory. A high learning rate may diverge; exact predictions depend on initialization.

Hand calculation:
- \(L=(wx+b-t)^2\)
- \(\partial L/\partial w=2(wx+b-t)x=-32\)
- \(\partial L/\partial b=2(wx+b-t)=-16\)

## Task 2 — PyTorch Training Pipeline

```powershell
python Task2_PyTorch/task2.py
```

The script scales the breast cancer dataset using train-only fitting, creates a `CsvDataset`, trains an MLP with BatchNorm and Dropout, uses early stopping, saves the model state dictionary, reloads it, and compares accuracy against Logistic Regression. Generated files are written to `artifacts/`. Accuracy can vary by library version and hardware.

## Task 3 — Classical NLP Baseline and Embeddings

```powershell
python Task3_Classical_NLP/task3.py
```

Creates a reproducible 360-example customer-feedback dataset, cleans text, trains a TF-IDF + Logistic Regression pipeline, reports accuracy, macro F1 and confusion matrix, and prints top class words. It also attempts to download GloVe vectors to demonstrate nearest words, the `king - man + woman` analogy and averaged sentence vectors. GloVe is optional and requires internet; if it fails, the TF-IDF baseline still completes.

Generated metrics and data are saved under `artifacts/`.

## Task 4 — DistilBERT fine-tuning and FastAPI

Run the tokenizer and pipeline demonstrations first (downloads pretrained models on first run):

```powershell
python Task4_Transformers_API/tokenizer_pipelines.py
```

This prints tokens, input IDs and attention masks for two sentences, then runs sentiment analysis and NER on three sentences each.

First run Task 3 to create `artifacts/customer_feedback.csv`. Then from the repository root:

```powershell
cd Task4_Transformers_API
pip install -r requirements.txt
python train_model.py --data ../artifacts/customer_feedback.csv --epochs 2 --max_rows 600
```

If you want to use every row, add `--max_rows 0`. Training downloads DistilBERT and can take time. The model and metrics are saved in `Task4_Transformers_API/saved_model/`. Do not commit the model checkpoint unless your instructor specifically requests it; model files are large.

Start the API from `Task4_Transformers_API` after training:

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`, select `POST /predict`, and test:

```json
{
  "text": "The delivery was excellent and support was helpful"
}
```

Example response shape (the actual label and confidence depend on the trained model):

```json
{
  "text": "The delivery was excellent and support was helpful",
  "label": "positive",
  "confidence": 0.95
}
```

An empty or whitespace-only text returns HTTP 422. The model is loaded once at application startup.

### BERT vs GPT

- BERT-style models use an encoder and bidirectional context to understand text.
- GPT-style models use a decoder and causal attention to generate text token by token.
- DistilBERT is a smaller, faster version of BERT.
- Sentiment/category classification is a discriminative understanding task, so DistilBERT is a suitable fit.
- GPT can also classify text through prompting, but may add latency and inference cost for this task.

## Model comparison — measured results

The following TF-IDF scores were measured by running `Task3_Classical_NLP/task3.py` with its reproducible 360-row synthetic customer-feedback dataset. DistilBERT values are from the Task 4 training run (2 epochs, up to 600 rows). Runtime depends on your computer, Python packages, and hardware, so your locally recorded timing may differ.

| Model | Test accuracy | Test macro F1 | Training time | Inference time/text |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 1.0000 (100%) | 1.0000 (100%) | 0.0137 s | 0.0000167 s (0.0167 ms) |
| Fine-tuned DistilBERT | 1.0000 (100%) | 1.0000 (100%) | 132.14 s | 0.0177 s (17.7 ms) |

### Additional evaluation details

**TF-IDF + Logistic Regression**

- Dataset: 360 synthetic feedback examples; stratified 80/20 split (288 train, 72 test).
- Test confusion matrix, class order `negative`, `neutral`, `positive`:

```text
[[24, 0, 0],
 [ 0, 24, 0],
 [ 0,  0, 24]]
```

- Precision, recall, and F1 were 1.00 for each class on this test split.

**Fine-tuned DistilBERT**

- Fine-tuning: 2 epochs; measured training time 132.14 seconds.
- Final test metrics: accuracy `1.0000`, macro F1 `1.0000`, evaluation loss `0.3054`.
- Measured inference time: approximately `0.0177` seconds per text.
- The saved training log reported validation accuracy/F1 of `0.9773`/`0.9778` after epoch 1 and `1.0000`/`1.0000` after epoch 2.

> **Important limitation:** Both models were evaluated on a small, synthetic dataset built from repeated text templates. The perfect test scores do not establish that either model will achieve 100% accuracy on real customer feedback. Evaluate on a separate, representative real-world dataset before drawing conclusions about generalization.

Task 3 writes `artifacts/tfidf_metrics.csv`. Task 4 writes transformer metrics under `Task4_Transformers_API/saved_model/` when training completes. If your own run reports different values, replace the table values with the values from your local CSV/logs.

## GitHub submission process

1. Create a new **public** GitHub repository named `Week3-DeepLearning-NLP` (or use your trainer's required visibility).
2. In VS Code, open the `Week3-DeepLearning-NLP` folder.
3. Open Terminal → PowerShell and run:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Add Week 3 deep learning and NLP assessment"
git remote add origin https://github.com/YOUR-USERNAME/Week3-DeepLearning-NLP.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If `origin` already exists, check it with `git remote -v`; update it using `git remote set-url origin https://github.com/YOUR-USERNAME/Week3-DeepLearning-NLP.git`.

If GitHub rejects the push because the remote already has a README or other commits, run:

```powershell
git pull --rebase origin main
git push -u origin main
```

If Git says `src refspec main does not match any`, create a commit first. Never upload `.env`, API keys, passwords, virtual environments, or large model checkpoints.

## Submission checklist

- [ ] Task 1 runs and loss plot is captured.
- [ ] Task 2 shows training, validation, reload equality and Logistic Regression comparison.
- [ ] Task 3 reports accuracy, macro F1, confusion matrix and class keywords.
- [ ] Task 4 shows tokenizer/pipeline demonstrations, fine-tuning metrics, comparison table and `/predict` screenshot.
- [ ] README and source files are pushed to GitHub.
- [ ] Actual output screenshots and measured metrics are added if required by the trainer.
