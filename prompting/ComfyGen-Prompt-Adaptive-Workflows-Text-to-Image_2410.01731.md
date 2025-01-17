# ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation

Rinon Gal, Adi Haviv, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, Gal Chechik
[2410.01731](https://arxiv.org/abs/2410.01731) [website](https://comfygen-paper.github.io/)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 Method](#3-method)
  - [3.3 ComfyGen-IC](#33-comfygen-ic)
  - [3.4 ComfyGen-FT](#34-comfygen-ft)
  - [3.5 Implementation details](#35-implementation-details)
- [4 Experiments](#4-experiments)
- [5 Analysis](#5-analysis)
- [6 Limitations](#6-limitations)
- [7 Conclusions](#7-conclusions)

## Abstract
**Prompt-Adaptive Workflows for Text-to-Image Generation (ComfyGen)**

**Background:**
- Evolution of text-to-image generation from monolithic models to complex workflows
- Expertise required for effective workflow design due to component availability, interdependence, and prompt dependency

**Problem Statement:**
- Need for automated tailoring of workflows based on user prompts (prompt-adaptive workflow generation)

**Approaches Proposed:**
1. **Tuning-based method**:
   - Learns from user-preference data
2. **Training-free method**:
   - Uses LLM to select existing flows

**Benefits:**
- Improved image quality compared to monolithic models or generic workflows
- Complementary research direction in text-to-image generation field

**Figure 1:**
- Standard text-to-image generation flow: monolithic model transforms prompt into an image (top)
- Proposed approach: LLM synthesizes custom workflows based on user’s prompt (bottom)
- LLM chooses components that better match the prompt for improved quality.

## 1 Introduction

**Text-to-Image Generation: Advanced Workflows**
* Researchers shift to complex workflows combining components for enhancing image quality (Rombach et al., 2022; Ramesh et al., 2021)
* Components include fine-tuned generative models, LLMs, LoRAs, improved decoders, super resolution blocks
* Effective workflows depend on prompt and image content
* Nature photographs may use photorealism models; human images often contain negative prompts or specific super-resolution models
* Building well-designed workflows requires expertise

**Proposed Approach:** Learn to build text-to-image generation workflows conditioned on user prompt using LLMs.

**Components:**
1. Prompt: describes desired image
2. LLM: interprets prompt and matches content with most appropriate blocks
3. Workflow: tailored to specific prompt for improved image quality
4. ComfyUI (comfyanonymous, 2023): stores workflows as JSON files, accesses multiple human-created workflows
5. Training set: 500 diverse prompts, images generated using each workflow, scored by aesthetic and human preference estimators
6. Two approaches for matching flows to novel prompts: ComfyGen-IC and ComfyGen-FT
7. Comparison against baselines: single-model approaches (SDXL model, fine-tunes, DPO optimized version), prompt-independent popular workflows
8. Benefits: outperforms all baselines on human-preference and prompt-alignment benchmarks.

## 2 Related work

**Improving Text-to-Image Generation Quality: Related Work**

**Fine-tuning Pretrained Models:**
- Curated datasets and improved captioning techniques used for fine-tuning (Dai et al., 2023; Betker et al., 2023; Segalis et al., 2023)
- Reward models as an alternative: reinforcement learning or differentiable rewards (Kirstain et al., 2023; Wu et al., 2023b; Xu et al., 2024; Lee et al., 2023; Clark et al., 2024; Prabhudesai et al., 2023; Wallace et al., 2024)
- Exploring diffusion input noise space using reward models (Eyring et al., 2024; Qi et al., 2024)
- Self-guidance or frequency-based feature manipulations for more detailed outputs (Hong et al., 2023; Si et al., 2024; Luo et al., 2024)

**Leveraging Large Language Models:**
- Significant improvements in reasoning abilities and adaptability through fine-tuning methods, zero-shot prompting or in-context learning (Schick et al., 2024; Wang et al., 2024; Surís et al., 2023; Shen et al., 2024; Gupta & Kembhavi, 2023; Wu et al., 2023a)
- LLM agents proposed to equip the model with external tools through API tags, documentation, model descriptions and code samples (Schick et al., 2024; Wang et al., 2024; Surís et al., 2023; Shen et al., 2024; Gupta & Kembhavi, 2023; Wu et al., 2023a)
- Our work focuses on prompt-adaptive pipeline creation and tapping into this under-explored path to improving the quality of downstream generations.

**Pipeline Generation:**
- Compound systems with multiple models used for state-of-the-art results across various domains (AlphaCode Team, 2024; Trinh et al., 2024; Nori et al., 2023; Yuan et al., 2024)
- Crafting such compound systems is a daunting task due to careful component selection and parameter tuning (Khattab et al., 2023; Zhuge et al., 2024)
- Our work tackles the task of pipeline generation for text-to-image models, focusing on designing compound pipelines that depend on user's prompt.

## 3 Method

**ComfyUI Method**

**Goal**: Match input prompt with appropriate text-to-image workflow for improved visual quality and prompt alignment.

**Hypothesis**: Effective workflows depend on specific topics in the prompt.

**Proposed Approach**: Use LLM to reason over prompt, identify topics, and select or synthesize new flow.

**Description of ComfyUI**:
- Open-source software for designing and executing generative pipelines
- Users create pipelines by connecting model blocks
- Simple example pipeline: base model, face restoration block, positive/negative prompt (Figure 2a)
- Complex pipelines include LoRAs, ControlNets, IP-Adapters, etc. (Figure 2b,c)
- Pipelines can be exported to JSON files for automation

**Training Data**:
- Collect approximately 500 human-generated ComfyUI workflows from popular websites
- Filter out video and control image generation flows, highly complex flows, and community-written blocks appearing in fewer than three flows
- Augment data by randomly switching models, LoRAs, samplers, or changing parameters (310 distinct workflows)
- Collect 500 popular prompts from Civitai.com to synthesize images with each flow using ensemble of quality prediction models: LAION Aesthetic Score, ImageReward, HPS v2.1, Pickscore.
- Standardize and sum scores for a single scalar score per prompt-workflow pair (higher scores correlate with better image quality)

### 3.3 ComfyGen-IC

**Approach to Providing Prompt-Dependent Flows using LLMs:**
* Use in-context based solutions that leverage a powerful, closed-source LLM
* First step: provide LLM with list of labels for training prompts (object-categories, scene categories, styles)
	+ Examples: "People", "Wildlife", "Urban", "Nature", "Anime", "Photo-realistic"
* Calculate average quality score of images produced by each flow across all prompts in a label category
* Repeat for all flows and all labels, creating a table of flows and their performance across categories
* Ideally: provide LLM with full JSON description of flows to learn relationships between components and downstream performance
* Alternative approach: **ComfyGen-IC** - classifier capable of parsing new prompts, breaking them down into relevant categories, and selecting best matching flow

### 3.4 ComfyGen-FT
**Approach to Fine-tuning LLM for Predicting High-Quality Workflows:**
* Instead of best-scoring method: fine-tune LLM to predict specific flow that achieved given score for a prompt
* Significant drawbacks of best-scoring method: reduces number of training tokens, sensitive to randomness, no negative examples
* Proposed alternative formulation: task LLM with predicting flow given prompt and associated score
* Increases available data points for training by utilizing all flows instead of just highest scorers
* Reduces impact of random fluctuations by considering a wider range of scores and their associated flows
* Allows learning from negative examples, helping identify ineffective components or combinations
* Inference: provide LLM with prompt and high score, have it predict effective flow for given prompt
* Variation: **ComfyGen-FT**

### 3.5 Implementation details
**Implementation Details:**
* ComfyGen-IC implemented using Claude Sonnet 3.5
* ComfyGen-FT on top of pre-trained Meta Llama models (70B and 3.1B checkpoints)
* Unless otherwise noted, all results in the paper use 70B model with target score of 0.725
* Fine-tune for a single epoch using LoRA rank 16 and learning rate 2e−4
* Additional details provided in supplementary materials.

## 4 Experiments

**Method Showcase**:
- Generates higher quality images across diverse domains and styles
- Prompts available in supplementary material
- Examples shown in Figure 3:
  - Subject-focused images
  - Photo-realistic imagery
  - Artistic or abstract creations

**Baseline Comparison**:
- Two types of alternative approaches:
  - **Fixed, monolithic models**: Pre-trained diffusion model directly conditioned by prompts (SDXL, JuggernautXL, DreamshaperXL, DPO-SDXL)
  - **Generic workflows**: Same workflow for all images regardless of prompt (SSD-1B, Pixart-Σ)
- Evaluated on:
  - GenEval benchmark: Prompt-alignment tasks like single-object generation, counting, and attribute binding
  - User study on CivitAI prompts using human preference scores

**GenEval Results**:
- Tuning-based model outperforms all baselines despite only using human preference scores during training
- In-context approach underperforms due to short, simplistic prompts challenging its ability to match prompts with labels

| Model | Object Detection Scores | Single Subject Object | Counting | Colors | Position | Attribute Binding | Overall |
| ------ | --------------------- | -------------------- | -------- | ------ | --------- | --------------- | ------- |
| SDXL   | 0.98                | 0.39               | 0.85     | 0.15   | 0.23     | 0.23        | 0.55       |
| JuggernautXL | 1.00             | 0.73              | 0.48     | 0.89    | 0.11      | 0.19         | 0.57       |
| DreamShaperXL | 0.99            | 0.78              | 0.45     | 0.81    | 0.17      | 0.24         | 0.57       |
| DPO-SDXL | 1.00             | 0.81              | 0.44     | 0.90    | 0.15      | 0.23        | 0.59       |
| Fixed Flow - Most Popular | 0.95            | 0.38             | 0.77     | 0.06     | 0.12       | 0.12        | 0.42       |
| Fixed Flow - 2nd Most Popular | 1.00          | 0.65             | 0.86     | 0.13     | 0.34       | 0.34        | 0.59       |
| ComfyGen-IC (ours)    | 0.99            | 0.78              | 0.38     | 0.84     | 0.13      | 0.25        | 0.56       |
| ComfyGen-FT (ours)   | 0.99            | 0.82             | 0.50     | 0.90    | 0.13      | 0.29         | 0.61       |

**CivitAI Prompts Evaluation**:
- ComfyGen-FT outperforms all baseline approaches, despite being tuned with human preference scores and not strictly for prompt alignment

## 5 Analysis

**Findings of ComfyGen's Performance Analysis**

**Three aspects examined:**
1. Originality and diversity of generated flows
2. Human-interpretable patterns
3. Effect of using target score in ComfyGen-FT prompts

**Originality and diversity**
- ComfyGen-FT generates novel flows with minimal similarity to training corpus (0.9995 compared to expected 1.0)
- More diverse outputs than ComfyGen-IC, suggesting potential for further data or parameter search

**Analyzing chosen flows**
- Patterns identified in selected models per category: intuitive in some cases but not always clear
- Future work may involve explaining reasoning behind component selections

**Effect of target scores**
- ComfyGen-FT learns to associate target scores with varying quality flows
- Appropriate choice of score leads to comparable performance to ComfyGen-IC
- Predicting best model instead of score leads to diminished performance, highlighting importance of approach.

**Comparative Analysis:**
- Both ComfyGen-FT models (8B and 70B) perform equally well and significantly outperform baseline SDXL model and ComfyGEN-IC in most evaluations.

## 6 Limitations

**Limitations of ComfyGen's Approach**

**Text-to-Image Workflows**:
- Current model is limited to text-to-image workflows
- Cannot address more complex editing or control-based tasks
- Potential resolution: Vision-Language Models (VLMs) could be used in the future

**Generation Speed and Scalability**:
- Generations take an order of 15 seconds per image
- With a set of 500 prompts and 300 flows, requires a month of GPU time to create
- Scaling up would likely require significant computational resources or more efficient ways (e.g., Reinforcement Learning) to explore the flow parameter space

**Drawbacks of Fine-Tuning Approach**:
- Cannot easily generalize to new blocks as they become available
- Requires retraining with new flows that include these blocks

**Drawbacks of In-Context Approach**:
- Can be easily expanded by including new flows in the score table provided to the LLM
- Increases the number of input tokens used, making it more expensive to run
- Eventually saturates the maximum context length

**Future Work**:
- More advanced retrieval-based approaches or use of collaborative agents could potentially address these limitations.

## 7 Conclusions

**Conclusions**

**Introduction**:
- Presented ComfyGen - a set of two approaches for prompt-adaptive workflow generation
- Demonstrated that such prompt-dependent flows can outperform monolithic models or fixed, user created flows in improving image quality

**Future Work**:
- Explore more prompt-dependent workflow creation methods
- Increase originality and expand scope to image-to-image or video tasks
- Potential collaboration with language model on creating such flows, providing feedback through instructions or examples of outputs
- Enable non-expert users to push the boundary of content creation.

