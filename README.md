# Instruction Tuning with and without Context: Behavioral Shifts and Downstream Impact

[EACL 2026] This is the official codebase for [**Instruction Tuning with and without Context: Behavioral Shifts and Downstream Impact**](https://arxiv.org/abs/2506.15480). It includes the training and inference code for reproducing our main experiments.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📦 Data & Checkpoints

### 🔹 Data

We use a filtered version of the [Self-RAG dataset](https://huggingface.co/datasets/selfrag/selfrag_train_data) for training.

### 🔹 Checkpoints

Model checkpoints will be released soon. Stay tuned!

---

## 🚀 Training

### 📝 Text-Only Domain

To train using **LLaMA 3 8B** as the base model:

```bash
torchrun --nnodes 1 --master_port=29100 --nproc_per_node 8 train.py \
  --enable_fsdp --low_cpu_fsdp \
  --training_argument configs/training_configs/llama3_train.json \
  --num_epochs 3 \
  --dataset llava_llama3_selfrag_multi_dataset \
  --dist_checkpoint_folder llama3_basemodel \
  --batch_size_training 128 \
  --micro_batch_size 32 \
  --loss_mask_context context \
  --model_use_peft
```

**Argument descriptions:**

* `--training_argument`: Select config with the base model name. Check files under `configs/training_configs`.
* `--model_name`: Base model to fine-tune.
* `--token_name`: Tokenizer name (Instruct version for chat template compatibility).
* `--dataset`: Training dataset (aligned with your base model). Check the list of datasets under `configs/datasets_dpr.py`.
* `--dist_checkpoint_folder`: Folder to save checkpoints.
* `--loss_mask_context`: Choose from:

  * `no_context` – standard instruction tuning
  * `context` – CINGS (ours)
  * `no_mask` – CINGS without context masking
* `--model_use_peft`: Use LoRA for parameter-efficient fine-tuning (remove to train all parameters).

We have included examples in `run_llama3.sh`

---

### 🖼️ Vision-Language Domain

After training the language model, use the [official LLaVA repo](https://github.com/haotian-liu/LLaVA) for vision-language alignment.

Update the following scripts:

* `scripts/pretrain.sh`
* `scripts/finetune.sh`

Replace `model_name_or_path` with the checkpoint folder from the text-only training step (`dist_checkpoint_folder`).

---

## 🔍 Inference

### 📝 Text-Only Domain

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py \
  --training_argument {training_argument}.json \
  --dataset {dataset} \
  --dist_checkpoint_folder {dist_checkpoint_folder} \
  --val_batch_size 1 \
  --add_docs \
  --model_use_peft
```

Use the same arguments as training. Only `--dataset` should be updated to point to your evaluation dataset (see `configs/datasets.py` for available options).

---

### 🖼️ Vision-Language Domain

We follow the evaluation process from the [official LLaVA repo](https://github.com/haotian-liu/LLaVA). See the [evaluation guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for details.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{lee2026instructiontuningcontextbehavioral,
      title={Instruction Tuning with and without Context: Behavioral Shifts and Downstream Impact}, 
      author={Hyunji Lee and Seunghyun Yoon and Yunjae Won and Hanseok Oh and Geewook Kim and Trung Bui and Franck Dernoncourt and Elias Stengel-Eskin and Mohit Bansal and Minjoon Seo},
      year={2026},
      eprint={2506.15480},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.15480}, 
}
```

---

## 🙏 Acknowledgements

This repository builds on [Meta’s LLaMA Recipes](https://github.com/facebookresearch/llama-recipes), and part of the codes are from [Official LLaVA Repo](https://github.com/haotian-liu/LLaVA). We are grateful to the community and all contributors.

## Point of Contact
For any questions about the implementation or content of the paper, you could contact `hyunjil at cs.unc.edu`
