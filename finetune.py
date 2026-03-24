# finetune.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import torch

# --- Your training data: abstracts from your papers ---
# Just paste them as strings. Even 4 abstracts help the model
# learn your terminology (rank-order coding, EMG, predictive coding...)
abstracts = [
    "Abstract of your Neural Networks paper...",
    "Abstract of your IEEE ASRU paper...",
    "Abstract of your ICANN paper...",
    "Abstract of your ICDL paper...",
]
dataset = Dataset.from_dict({"text": abstracts})

# --- Load base model ---
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,       # 4-bit quantization: reduces VRAM from ~14GB to ~4GB
    device_map="auto",
    torch_dtype=torch.float16,
)

# --- Define LoRA config ---
# r=8: rank of the low-rank matrices (low = fewer params, faster)
# target_modules: which weight matrices to adapt (attention layers)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,            # scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj"],   # query and value attention matrices
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~2M || all params: ~3.8B || trainable: 0.05%

# --- Train ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained("./lora_output/final")
print("Fine-tuned model saved!")