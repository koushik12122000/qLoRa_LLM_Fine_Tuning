# 🚀 QLoRA Fine-Tuning of Gemma LLM for Text-to-SQL Generation

This project demonstrates the implementation of **Parameter-Efficient Fine-Tuning (PEFT)** using **QLoRA (Quantized Low-Rank Adaptation)** to adapt Google's **Gemma Large Language Model** for Text-to-SQL generation.

The objective of this project is to enable a Large Language Model (LLM) to translate natural language questions into syntactically correct SQL queries while significantly reducing GPU memory requirements through **4-bit quantization** and **LoRA adapters**.

Unlike traditional full-model fine-tuning, which requires updating billions of parameters, this implementation leverages **QLoRA**, allowing efficient training on consumer-grade GPUs such as the NVIDIA T4 available in Google Colab.

---

## ✨ Key Highlights

- Fine-tuned Gemma using QLoRA for Text-to-SQL tasks.
- Implemented 4-bit NF4 quantization using BitsAndBytes.
- Utilized LoRA adapters for parameter-efficient training.
- Built an end-to-end Hugging Face fine-tuning pipeline.
- Converted Text-to-SQL datasets into conversational instruction format.
- Trained and evaluated the model on Google Colab GPUs.
- Generated SQL queries directly from natural language prompts.
- Applied GPU memory optimization techniques for resource-constrained environments.

---

## 🎯 Business Use Case

Organizations often require business users to query databases without SQL expertise. This project demonstrates how Large Language Models can bridge that gap by converting human-readable questions into executable SQL statements, improving accessibility to data and accelerating analytics workflows.

---

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │ Text-to-SQL     │
                    │ Dataset          │
                    └────────┬────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Prompt Formatting      │
                │ Conversational Dataset │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │ Gemma Base Model       │
                │ (4-bit Quantized)      │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │ LoRA Adapters          │
                │ PEFT Training          │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │ Fine-Tuned Adapters    │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │ Text ➜ SQL Inference   │
                └────────────────────────┘
```

---

## 🧠 Concepts Demonstrated

- Large Language Models (LLMs)
- Parameter-Efficient Fine-Tuning (PEFT)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- 4-bit Quantization
- NF4 Quantization
- Supervised Fine-Tuning (SFT)
- Prompt Engineering
- GPU Memory Optimization
- Model Inference & Evaluation

---

## 🛠️ Tech Stack

### AI / Machine Learning
- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes
- Accelerate

### Model
- Gemma LLM

### Training Techniques
- QLoRA
- LoRA
- PEFT
- SFT (Supervised Fine-Tuning)
- 4-bit Quantization (NF4)

### Environment
- Google Colab
- NVIDIA T4 GPU

---

## 📂 Project Structure

```text
qLoRa_LLM_Fine_Tuning/
│
├── notebook/
│   └── qlora_finetuning.ipynb
│
├── outputs/
│   └── trained_lora_adapters/
│
├── screenshots/
│   ├── training_results.png
│   └── inference_results.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The model is trained on a **Text-to-SQL dataset** containing:

- Natural language questions
- Database schema information
- SQL query answers
- Query explanations

The dataset is transformed into a conversational instruction-following format suitable for supervised fine-tuning.

### Example

**Input Question**

```text
What is the average release year of songs in the folk genre?
```

**Database Schema**

```text
songs(song_id, title, year, genre)
```

**Expected SQL**

```sql
SELECT AVG(year)
FROM songs
WHERE genre = 'folk';
```

---

## ⚙️ QLoRA Configuration

### Quantization Configuration

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.float16
bnb_4bit_use_double_quant=True
```

### LoRA Configuration

```python
r=16
lora_alpha=32
lora_dropout=0.05
bias="none"
task_type="CAUSAL_LM"
```

---

## 🚀 Training Pipeline

1. Load the Text-to-SQL dataset.
2. Convert dataset samples into instruction-based conversations.
3. Load Gemma model using 4-bit quantization.
4. Configure LoRA adapters for PEFT training.
5. Fine-tune using Hugging Face TRL's SFTTrainer.
6. Save trained LoRA adapter weights.
7. Load adapters for inference and evaluation.

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|--------|
| Model | Gemma |
| Fine-Tuning Method | QLoRA |
| Training Type | Supervised Fine-Tuning |
| Quantization | 4-bit NF4 |
| Precision | FP16 |
| Epochs | 2-3 |
| Max Sequence Length | 512 |
| GPU | NVIDIA T4 |
| Optimizer | AdamW |

---

## 💻 Inference Example

### User Query

```text
What is the average release year of songs in the folk genre?
```

### Generated SQL

```sql
SELECT AVG(song_release_year)
FROM songs
WHERE genre = 'folk';
```

---

## 📈 Results

✅ Successfully fine-tuned Gemma using QLoRA

✅ Reduced GPU memory requirements through 4-bit quantization

✅ Trained on Google Colab T4 GPU

✅ Generated SQL queries from natural language instructions

✅ Demonstrated practical usage of PEFT and LoRA adapters

✅ Implemented end-to-end LLM fine-tuning workflow

---

## 🔍 Challenges & Learnings

During development, several practical fine-tuning challenges were encountered:

- Managing limited GPU memory during training.
- Understanding quantization and LoRA adapter configurations.
- Handling tokenizer and model compatibility issues.
- Optimizing sequence length and training parameters for stable training.
- Evaluating generated SQL output against expected queries.

### Key Learnings

- QLoRA drastically reduces hardware requirements for LLM fine-tuning.
- LoRA adapters enable efficient model adaptation without modifying base weights.
- Proper tokenizer-model alignment is critical for successful inference.
- Quantization enables training and inference on affordable GPU hardware.
- Hugging Face ecosystem simplifies production-ready LLM workflows.

---

## 🔮 Future Enhancements

- Fine-tune larger models such as Gemma 7B and Llama 3.
- Experiment with Flash Attention for faster training.
- Implement automated evaluation metrics.
- Deploy model using FastAPI.
- Containerize deployment using Docker.
- Integrate with real-world SQL databases.
- Build a web interface for natural language querying.

---

## 📸 Screenshots

Add screenshots demonstrating:

- Training progress
- Loss metrics
- GPU utilization
- Inference examples
- Generated SQL outputs

```text
screenshots/
├── training_logs.png
├── gpu_usage.png
├── loss_curve.png
└── inference_demo.png
```

---

## 🎓 Skills Demonstrated

- Generative AI
- LLM Fine-Tuning
- Hugging Face Ecosystem
- QLoRA
- LoRA
- PEFT
- Model Quantization
- Prompt Engineering
- PyTorch
- GPU Optimization
- SQL Generation
- AI Model Evaluation

---

## 🤝 Contributing

Contributions, improvements, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 👨‍💻 Author

**Koushik**

**GenAI Engineer | Agentic AI Engineer | AI/LLM Engineer | Software Engineer**

- GitHub: https://github.com/koushik12122000
- LinkedIn: www.linkedin.com/in/koushik-jakkula

---

## ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.

It helps others discover the project and supports future open-source AI initiatives.
