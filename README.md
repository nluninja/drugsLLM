# DrugsLLM: Fine-tuning Large Language Models for Italian Pharmaceutical Information

**Thesis Project - 2025**

## Abstract

This thesis project demonstrates the application of Parameter-Efficient Fine-Tuning (PEFT) techniques to create a specialized Large Language Model (LLM) capable of answering pharmaceutical questions in Italian. By fine-tuning Meta's Llama-3.1-8B-Instruct model using Low-Rank Adaptation (LoRA), we developed a domain-specific assistant that provides accurate information about medications, including their uses, contraindications, dosages, and side effects.

## Project Overview

### Objective

Develop an Italian-language pharmaceutical chatbot that can:
- Answer questions about drug indications and uses
- Provide information on active ingredients
- Explain contraindications and warnings
- Describe administration methods and dosages
- Discuss side effects and adverse reactions
- Address pregnancy-related medication concerns

### Methodology

The project employs:
- **Base Model**: Meta Llama-3.1-8B-Instruct
- **Fine-tuning Technique**: LoRA (Low-Rank Adaptation)
- **Quantization**: 8-bit precision via BitsAndBytes
- **Training Framework**: HuggingFace Transformers + PEFT

## Repository Structure

```
drugsLLM/
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
├── drugLLM_training.ipynb                 # Training notebook (Google Colab)
├── drugLLM_evaluation.ipynb               # Evaluation notebook
├── datasets/
│   ├── dataset_farmaci_qaCOMPLETO.json    # Complete QA dataset (5,709 samples)
│   ├── dataset_farmaci_qaNEW 1.json       # Additional QA dataset (1,705 samples)
│   └── dataset_bugfarmaciNEW.csv          # Drug database (1,071 drugs)
└── attic/                                 # Standalone Python scripts
    ├── belli_gen_fine.py                  # Fine-tuning script (CLI)
    └── belli_gen_infer.py                 # Inference script (CLI)
```

> **Note**: The notebooks are designed for both research and educational purposes. They can be used as teaching material for courses on LLM fine-tuning and NLP. The `attic/` folder contains standalone Python scripts for command-line execution.

## Dataset

### Format

The training data follows an instruction-question-answer structure in Italian:

```json
{
  "istruzione": "System instruction defining the assistant's role",
  "domanda": "User question about a specific medication",
  "risposta": "Detailed response with pharmaceutical information"
}
```

### Statistics

| Dataset | Samples | Size |
|---------|---------|------|
| Complete QA | 5,709 | 2.6 MB |
| Additional QA | 1,705 | 780 KB |
| Drug Database | 1,071 drugs | 2.4 MB |

### Coverage

The dataset covers comprehensive pharmaceutical information including:
- Drug uses and indications
- Active ingredients (principio attivo)
- Contraindications (controindicazioni)
- Side effects (effetti collaterali)
- Administration methods (somministrazione)
- Dosage information (dosaggio)
- Pregnancy considerations (gravidanza)

## Model Architecture

### Base Model
- **Model**: Meta Llama-3.1-8B-Instruct
- **Parameters**: 8.1 billion
- **Architecture**: Transformer-based causal language model

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 32 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj |
| Trainable Parameters | 83.9M (1.03% of total) |

### Quantization
- 8-bit quantization using BitsAndBytes
- Enables training on consumer-grade GPUs
- Reduces memory footprint significantly

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 (per device) |
| Learning Rate | 2e-4 |
| Warmup Steps | 200 |
| Max Steps | 5,000 |
| Weight Decay | 0.1 |
| Precision | FP16 |
| Context Length | 512 tokens |
| Train/Test Split | 98% / 2% |

### Results

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.2174 |
| Training Duration | ~1.3 hours |
| Samples per Second | 8.32 |
| Total FLOPs | 3.23 × 10¹⁷ |

## Usage

### Environment Setup

```bash
pip install torch transformers datasets trl peft bitsandbytes accelerate
```

### Training

1. Open `drugLLM_training.ipynb` in Google Colab
2. Mount Google Drive for data access
3. **Important**: Set your HuggingFace token securely using Colab Secrets or environment variables:
   ```python
   import os
   from google.colab import userdata
   hf_token = userdata.get('HF_TOKEN')  # Using Colab Secrets (recommended)
   # Or: hf_token = os.environ.get("HF_TOKEN")
   ```
4. Run all cells sequentially

> **Security Note**: Never commit API tokens or secrets to version control. Use environment variables or secret management tools.

### Inference

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoPeftModelForCausalLM.from_pretrained(
    "path/to/checkpoint",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoint")

# Prepare input
instruction = "Sei un esperto di farmacologia..."
question = "A cosa serve il farmaco Abba?"

text = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ],
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt")

# Generate response
outputs = model.generate(
    input_ids=inputs["input_ids"].to("cuda"),
    max_new_tokens=256
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Interactions

**Question**: A cosa serve il farmaco Abba?

**Response**: Il farmaco Abba (Amoxicillina/acido clavulanico) viene utilizzato per il trattamento delle infezioni batteriche delle vie respiratorie, delle vie urinarie, della pelle e dei tessuti molli...

**Question**: Quali sono le controindicazioni del farmaco Abba?

**Response**: Le principali controindicazioni del farmaco Abba includono: ipersensibilità ai principi attivi, alle penicilline o ad altri antibiotici beta-lattamici, anamnesi di ittero o disfunzione epatica...

## Technical Notes

### Chat Template

The model uses Llama-3.1's native chat template with three roles:
- **System**: Defines the assistant's expertise and behavior
- **User**: Contains the pharmaceutical question
- **Assistant**: Provides the detailed response

### Limitations

- Context window limited to 512 tokens
- Italian language only
- Information based on training data cutoff
- Not intended as a substitute for professional medical advice

## Model Availability

The trained model checkpoints are available on Google Drive:
[Model Download](https://drive.google.com/drive/u/1/folders/1uwD4tzcDeDpbFMUkqSXU2AE61HQRVckX)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- HuggingFace account with Llama model access

## Dependencies

```
torch>=2.0.0
transformers>=4.44.0
datasets>=3.0.0
peft>=0.6.0
bitsandbytes>=0.44.0
trl>=0.12.0
accelerate>=0.34.0
```

## Future Work

- Expand dataset with additional pharmaceutical sources
- Implement evaluation metrics (BLEU, ROUGE, medical accuracy)
- Add support for drug-drug interaction queries
- Extend context length for more detailed responses
- Multi-language support (English, French, German)

## Acknowledgments

- Meta AI for the Llama-3.1 model
- HuggingFace for the Transformers and PEFT libraries
- The Italian pharmaceutical data sources

## Disclaimer

This model is developed for educational and research purposes only. The information provided should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical questions.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

**Note**: The fine-tuned model weights are derived from Meta's Llama 3.1 and are subject to the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).

---

*Thesis Project - 2025*
