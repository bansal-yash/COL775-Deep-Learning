# COL775-Deep-Learning
Course assignments of COL775:- Deep Learning course at IIT Delhi under Professor Parag Singla


This repository contains assignments exploring modern deep learning architectures including ResNets, attention mechanisms, contrastive learning, vision-language models, and generative diffusion models.

---

## 📁 Assignments

### Assignment 1: Vision and Translation

#### [Part 1: ResNet Normalization](./Assignment_1_P1_Resnet_Normalisation_Schemes)

* Implemented **ResNet-18 architecture** from scratch in PyTorch without using pre-built torchvision models.
* Compared five normalization schemes: **Batch Normalization**, **Instance Normalization**, **Batch-Instance Normalization**, **Layer Normalization**, and **Group Normalization**.
* Trained models on 100-class ImageNet subset with 680 training and 50 validation images per class.
* Analyzed the impact of different normalization techniques on training stability and generalization performance.
* Implemented **Grad-CAM visualization** to interpret model decisions and understand feature importance across normalization variants.
* Evaluated models using classification accuracy metrics and conducted ablation studies on normalization layer placement.

#### [Part 2: Neural Translation](./Assignment_1_P2_Neural_Machine_Translation)

* Built **seq2seq architecture** with encoder-decoder framework for English-Hindi neural machine translation.
* Implemented **LSTM-based decoder** with attention mechanism to handle variable-length sequences and align source-target tokens.
* Explored cross-lingual transfer learning by fine-tuning English-Hindi model on English-Marathi translation task.
* Evaluated translation quality using **BLEU**, **chrF**, and **TER** (Translation Edit Rate) metrics.
* Experimented with teacher forcing schedules and different attention mechanisms for improved translation fluency.
* Analyzed the effectiveness of transfer learning between closely related languages (Hindi and Marathi) sharing the same script.

---

### [Assignment 2: Multimodal Learning](./Assignment_2_Multimodal_Learning)

#### [Part A: Representation Learning](./Assignment_2_Multimodal_Learning/PartA)

* Implemented **CLIP** (Contrastive Language-Image Pre-training) from scratch with Vision Transformer image encoder and lightweight text transformer.
* Built **DINO** (self-distillation with no labels) for self-supervised visual representation learning through knowledge distillation.
* Trained both models on CLEVR dataset with programmatically generated captions describing object counts, colors, shapes, and materials.
* Designed custom task-specific tokenizer for caption encoding without relying on pre-trained tokenizers.
* Evaluated learned representations through **linear probing** on downstream tasks: object counting, color prediction, and shape classification.
* Conducted image-text retrieval experiments to assess alignment quality in the shared embedding space.

#### [Part B: Vision-Language Model](./Assignment_2_Multimodal_Learning/PartB)

* Integrated pre-trained vision encoder with language model to build a general-purpose **Vision-Language Model (VLM)** inspired by LLaVA.
* Implemented two-stage training pipeline: first training projection layer with frozen vision encoder, then optional joint fine-tuning.
* Aligned visual and textual modalities through learnable projection interface connecting frozen CLIP/DINO encoder to language decoder.
* Enabled visual question answering and image captioning on CLEVR scenes by grounding language generation in learned visual representations.
* Evaluated model on visual reasoning tasks requiring understanding of spatial relationships and object attributes.

#### [Part C: Generative Modeling](./Assignment_2_Multimodal_Learning/PartC)

* Implemented **Variational Autoencoder (VAE)** to learn dense latent representations of CLEVR images.
* Built text-guided **Latent Diffusion Model (LDM)** for generating diverse CLEVR scenes from natural language captions.
* Trained diffusion process in VAE latent space for efficient generation compared to pixel-space diffusion.
* Conditioned image generation on text embeddings to enable fine-grained control over object attributes (count, color, shape, material).
* Evaluated generation quality through visual inspection and diversity metrics across different caption prompts.
* Explored trade-offs between reconstruction quality and latent space regularization in VAE training.

---

Each assignment combines theoretical understanding with hands-on implementation of state-of-the-art architectures, covering computer vision, natural language processing, multimodal learning, and generative modeling.
