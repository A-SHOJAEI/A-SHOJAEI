# Alireza Shojaei

AI/ML engineer building production-grade deep learning systems. Focused on novel model architectures, uncertainty quantification, and efficient training strategies.

## Selected Projects

### LLM Inference & Efficient Fine-Tuning
- **[CascadeExit-Research](https://github.com/A-SHOJAEI/CascadeExit-Research)** - Adaptive early-exit speculative decoding for LLM inference. 1.76x speedup on Llama-3.2-3B with SwiGLU exit adapters trained on WikiText-103
- **[MoLE-LoRA](https://github.com/A-SHOJAEI/MoLE-LoRA)** - Mixture of LoRA Experts with BERT-tiny router on Llama-3.2-3B. 73% memory savings, evaluated on MMLU/GSM8K/HellaSwag/ARC
- **[instruction-complexity-aware-lora-routing](https://github.com/A-SHOJAEI/instruction-complexity-aware-lora-routing)** - Mixture-of-LoRA-experts with dynamic instruction routing on Alpaca
- **[adaptive-curriculum-learning-for-domain-transfer-in-llm-evaluation](https://github.com/A-SHOJAEI/adaptive-curriculum-learning-for-domain-transfer-in-llm-evaluation)** - Curriculum learning for LLM domain transfer on MMLU
- **[contrastive-curriculum-mmlu-with-adaptive-difficulty-sampling](https://github.com/A-SHOJAEI/contrastive-curriculum-mmlu-with-adaptive-difficulty-sampling)** - Contrastive learning with adaptive difficulty sampling on MMLU

### NLP & Question Answering
- **[adaptive-retrieval-qa-with-answerability-calibration](https://github.com/A-SHOJAEI/adaptive-retrieval-qa-with-answerability-calibration)** - Retrieval-augmented QA with answerability calibration on SQuAD 2.0
- **[contrastive-qa-verifier-with-adversarial-unanswerable](https://github.com/A-SHOJAEI/contrastive-qa-verifier-with-adversarial-unanswerable)** - Dual-encoder QA verification with adversarial unanswerable detection
- **[hierarchical-contrastive-qa-with-adversarial-unanswerable-detection](https://github.com/A-SHOJAEI/hierarchical-contrastive-qa-with-adversarial-unanswerable-detection)** - Hierarchical span prediction with contrastive learning on SQuAD 2.0
- **[genre-adaptive-nli-summarization-validator](https://github.com/A-SHOJAEI/genre-adaptive-nli-summarization-validator)** - Cross-genre NLI-based summarization validation on CNN/DailyMail + MultiNLI
- **[legal-clause-risk-scorer](https://github.com/A-SHOJAEI/legal-clause-risk-scorer)** - DeBERTa-v3 multi-task contract clause risk assessment on CUAD + LEDGAR

### Multimodal & Generative AI
- **[preference-guided-image-captioning-alignment](https://github.com/A-SHOJAEI/preference-guided-image-captioning-alignment)** - CLIP + GPT-2 with DPO preference alignment trained on 25K COCO images and UltraFeedback

### Computer Vision & Uncertainty
- **[pet-breed-uncertainty-aware-classifier](https://github.com/A-SHOJAEI/pet-breed-uncertainty-aware-classifier)** - EfficientNet-B0 with MC Dropout uncertainty on Oxford-IIIT Pet (7,393 images)
- **[fairness-aware-income-prediction-with-constraint-optimization](https://github.com/A-SHOJAEI/fairness-aware-income-prediction-with-constraint-optimization)** - Fairness-constrained LightGBM with Optuna on UCI Adult Census

### Molecular ML & Graph Networks
- **[spectral-temporal-curriculum-molecular-gap-prediction](https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gap-prediction)** - Spectral graph wavelets with curriculum learning on PCQM4Mv2 (3.7M molecules)
- **[molecular-scaffold-aware-multi-task-toxicity-prediction](https://github.com/A-SHOJAEI/molecular-scaffold-aware-multi-task-toxicity-prediction)** - Scaffold-aware GCN with attention pooling on Tox21 (7,823 molecules)
- **[hierarchical-attention-pooling-for-molecular-scaffold-transfer](https://github.com/A-SHOJAEI/hierarchical-attention-pooling-for-molecular-scaffold-transfer)** - Hierarchical attention for molecular scaffold transfer on MoleculeNet BBBP

### Systems & RL
- **[adaptive-model-serving-optimizer](https://github.com/A-SHOJAEI/adaptive-model-serving-optimizer)** - UCB bandit-based model serving with latency/accuracy/cost optimization
- **[adaptive-traffic-signal-control-via-hierarchical-multi-agent-rl](https://github.com/A-SHOJAEI/adaptive-traffic-signal-control-via-hierarchical-multi-agent-rl)** - Hierarchical multi-agent RL for traffic signal control
- **[temporal-distribution-shift-detector-with-adaptive-ensemble-reweighting](https://github.com/A-SHOJAEI/temporal-distribution-shift-detector-with-adaptive-ensemble-reweighting)** - Bayesian online ensemble reweighting for distribution shift detection

## Tech Stack

**Frameworks**: PyTorch, HuggingFace Transformers, PyTorch Geometric, DGL, scikit-learn, XGBoost, LightGBM

**Techniques**: LoRA/PEFT, GNNs, contrastive learning, curriculum learning, uncertainty quantification, multi-task learning, DPO alignment, early-exit inference

**Infrastructure**: MLflow, Docker, NVIDIA RTX 3090 (dual GPU), AMD Threadripper 3960X
