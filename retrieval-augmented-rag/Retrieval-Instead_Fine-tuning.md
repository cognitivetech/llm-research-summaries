# Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning

**Authors**: Pengfei Jin (First author: pjin1@mgh.harvard.edu) and Peng Shu (Co-first author: peng.shu@uga.edu)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Method](#3-method)
  - [3.1 Construction of LoRA-VecDB](#31-construction-of-lora-vecdb)
  - [3.2 Retrieval and Weighted Ensemble](#32-retrieval-and-weighted-ensemble)
- [4 Experiments](#4-experiments)
  - [4.1 Implementation detail](#41-implementation-detail)
  - [4.2 Medical report impression](#42-medical-report-impression)
  - [4.3 Medical Image segmentation](#43-medical-image-segmentation)
  - [4.4 Ablation Study](#44-ablation-study)
- [5 Discussion and Future work](#5-discussion-and-future-work)
- [6 Conclusion](#6-conclusion)

## Abstract

**Foundation Models and Improvements:**
- **Low-Rank Adaptation (LoRA)**: efficient fine-tuning of large models
- **Retrieval-Augmented Generation (RAG)**: uses vectorized databases, improves performance with external information

**Challenges:**
- Extensive training or labeled data required
- Limitations in resource-constrained environments

**Introduction of Retrieval-based Parameter Ensemble (RPE):**
- Creating a vectorized database of LoRAs for efficient retrieval and application
- Minimizes need for extensive training, eliminates requirement for labeled data
- Effective for zero-shot learning

**Advantages:**
- Well-suited for privacy-sensitive domains like healthcare
  - Modifies model parameters without accessing raw data

**Applications and Results:**
- Medical report generation: effective and surpassed supervised fine-tuning methods in certain cases
- Image segmentation: not specified if it also outperformed supervised methods.

## 1 Introduction

**Retrieval-based Parameter Ensemble (RPE) Model**

**Introduction:**
- Foundation models: CLIP, LLaMA, SAM
- Large datasets, minimal adaptation
- Fine-tuning resource-intensive
- LoRA offers solution with minimal fine-tuning and reduced computational overhead
- RAG for hallucination mitigation and zero-shot learning
- Combining strengths of LoRA and RAG: Retrieval-based Parameter Ensemble (RPE) model

**Components:**
1. **LoRA-VecDB**: vectorized database of LoRAs and their representations across tasks
2. New task querying: extract target representation ztrgsuperscript𝑧trgz^{\text{trg}}italic_z start_POSTSUPERSCRIPT trg end_POSTSUPERSCRIPT to find similar LoRAs {δ⁢θiref}
3. Ensemble formation: combine retrieved LoRAs using weighted ensemble methods
4. Model adaptation without extensive fine-tuning
5. Reduced computational costs and privacy concerns
6. Foundation models continue to scale: energy consumption and privacy issues

**Methodology:**
1. Establishing a vectorized database, LoRA-VecDB
2. Construction of the database: comprehensive repository of LoRAs and their corresponding representations across various tasks
3. Accessibility and collaboration
4. Querying for similar adaptors when new tasks arise
5. Calculate appropriate weights wi to form a parameter ensemble
6. Reducing redundancy, preserving data privacy
7. Analysis of relationship between parameter and feature spaces: new weighting strategy enhances model adaptability and accuracy
8. Real-world validation in medical applications

**Related Work:**
- Background for approach
- Fine-tuning methods and their computational costs
- Privacy concerns in foundation models' deployment

**Experiments:**
1. Medical language processing tasks
2. Medical image processing tasks
3. Demonstrating effectiveness of the RPE model

**Discussion and Future Work:**
- Implications of findings
- Suggestions for future research directions

## 2 Related Work

**Related Approaches to RAG**

**RAG (Retrieval Augmented Generation)**
- Integrates external knowledge into large language models (LLMs) by retrieving relevant information for enhanced generation accuracy: Ma et al., [2023](https://arxiv.org/html/2410.09908v1#bib.bib18)
- Recent advancements optimize query prompting, indexing structures, and retrieval mechanisms to address limitations: Ma et al., [2023](https://arxiv.org/html/2410.09908v1#bib.bib18); Peng et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib21); Gao et al., [2022](https://arxiv.org/html/2410.09908v1#bib.bib10)
- Enhances retrieval precision and reduces hallucinations in low-resource domains: Seo et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib25); Parvez et al., [2022](https://arxiv.org/html/2410.09908v1#bib.bib20)

**Parameter Combination Methods**
- **Model Soup**: Simplifies model combination through parameter averaging, achieving state-of-the-art performance without added inference or memory costs: Wortsman et al., [2022](https://arxiv.org/html/2410.09908v1#bib.bib31)
- **Federated Learning (FL)**: Distributed learning with decentralized setup preserves privacy, making it ideal for privacy-sensitive applications: McMahan et al., [2017](https://arxiv.org/html/2410.09908v1#bib.bib19)
- **Mixture of Experts (MoE)**: Dynamic expert selection capabilities improve performance and efficiency in large-scale LLMs: Xue et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib33); Lin et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib15)

**Zero-Shot Learning**
- Machine learning technique for recognizing unseen tasks using shared attributes or semantic relationships: Wang et al., [2019](https://arxiv.org/html/2410.09908v1#bib.bib29); Xian et al., [2017](https://arxiv.org/html/2410.09908v1#bib.bib32); Fu et al., [2018](https://arxiv.org/html/2410.09908v1#bib.bib9)
- Requires a specific task representation zisubscript𝑧𝑖z_{i}italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT to map from known tasks and parameters θisubscript𝜃𝑖\theta_{i}italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUPERSCRIPT to novel tasks Ttrgsuperscript𝑇trgT^{\text{trg}}italic_T start_POSTSUPERSCRIPT trg end_POSTSUPERSCRIPT : DeViSE, GCN-ZL, DGP-ZL
- Our work uses pretrained models for representation zisubscript𝑧𝑖z_{i}italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and replaces traditional neural network approach with retrieval and algorithm-based method to perform the mapping 𝒜𝒜\mathcal{A}caligraphic_A.
- Offers a scalable and efficient alternative to conventional zero-shot learning approaches where acquiring labeled data for all potential classes is impractical.

## 3 Method

**Components of Approach:**
* **Construction of LoRA-VecDB**: vectorized database for storing model adaptations and their corresponding representations
* **Retrieval and Weighted Ensemble Mechanism**
	+ Transforming dataset for new task into query representation: z^{\text{trg}}
	+ Retrieving relevant LoRAs: {z_{i}^{\text{ref}}, δ\theta_{i}^{\text{ref}}}
	+ Computing weights based on similarity between z^{\text{trg}} and {z_{i}^{\text{ref}}} in representation space
	+ Applying these weights to adjust δ\theta_{i}^{\text{trg}} in parameter space.

**LoRA-VecDB**:
- Vectorized database for storing model adaptations and their corresponding representations

**Retrieval and Weighted Ensemble Mechanism:**
1. Transform dataset for new task into query representation: z^{\text{trg}}
2. Retrieve relevant LoRAs (model adaptations): {z_{i}^{\text{ref}}, δ\theta_{i}^{\text{ref}}}
3. Compute weights based on similarity between z^{\text{trg}} and {z_{i}^{\text{ref}}} in representation space
4. Apply weights to adjust LoRAs: {δ\theta_{i}^{\text{trg}}}

### 3.1 Construction of LoRA-VecDB

**LoRA-VecDB: A Central Repository for LoRAs and Representations**

**Introduction:**
- LoRA-VecDB: central repository for LoRAs (δθi) and their corresponding representations (zi)
- Facilitates accessibility and encourages community contributions

**Training LoRAs:**
- Using foundation model F(⋅, θ0)
- Freeze pre-trained weights, introduce trainable low-rank matrices
- Generates a representation zi for each dataset Di

**Representing Datasets:**
- Feature maps from encoder of F as input
- Maintain integrity and originality of model's initial pre-training
- No fine-tuning

**Initial Exploration:**
- Various distribution distance metrics: Chamfer, Nearest Neighbor Distance, Mean Distance
- Did not show significant differences in dataset characteristics

**Representing Features:**
- Averaging all associated data feature maps
- Simplifies computational process and facilitates efficient storage

**Benefits:**
- Structured and efficient way to store and retrieve adaptations
- Supports scalable framework for experimentation and enhancement in model adaptability
- Valuable asset for researchers and practitioners.

### 3.2 Retrieval and Weighted Ensemble

**LoRA-Based Zero-Shot Learning Ensemble**

**Process Overview:**
1. **Transform dataset**: Convert new task data into query representation (ztrg)
2. **Retrieve related LoRAs**: Find most relevant LoRAs and their corresponding weighted adjustments (w, δθ)
3. **Compute weights**: Calculate the new model parameters based on retrieved LoRAs and foundational model
4. **Parameter Ensemble:** Implement different strategies for calculating optimal parameter relationships
5. **Similarity Calculation Strategy**
   - Premised on transfer learning concept: Similar tasks have similar feature representations, benefit from similar param adjustments.

**Steps in Detail:**
1. Transform dataset (ztrg) into query representation using the foundational model F(⋅, θ0).
2. Compute feature representation for Dtrg using EF(xj, θ0) and argsort function to obtain k closest LoRAs based on similarity.
3. Retrieve closest LoRAs (ziref) from the database based on their indices in argsort output.
4. Calculate weights (wi) using algorithm A(zi, ztrg).
5. Compute new model parameters (θtrg) by summing the weighted adjustments (δθi) and foundational model parameters (θ0).
6. Implement different strategies, denoted as 𝒜𝒜\mathcal{A}caligraphic_A, to calculate optimal parameter relationships based on latent space structures.
7. Evaluate performance improvement across varied datasets.

#### Linear Combination of Latent Representations for Zero-shot Learning

**Ensemble Methods for Zero-shot Learning: Retrieval vs. Linear Combination**

**Similarity Calculation (Retrieval)**
- Strategy calculates the similarity between target feature vector and reference feature vectors using squared ℓ2 norm
- Weights assigned using softmax function to normalize inverse distances
- Temperature parameter λ1 controls sharpness of distribution
- Linear relationship assumed between latent representations and parameter adjustments
- Objective: minimize error between target representation and weighted sum of reference representations

**Linear Combination**
- Seeks a linear combination of retrieved LoRAs that best approximates target representation
- Constraint: combined weights equal one to maintain normalized contribution from each LoRA
- Minimizes error between target representation and weighted sum of reference representations
- Regularization introduced for managing influence of each LoRA, especially with sparse or high-dimensional data

**Regularization (Linear Combination)**
- Penalizes weights to encourage simpler solutions that may generalize better
- Sparse solution preferred in presence of many possible solutions
- ℓ1 norm penalty encourages sparsity among weights
- Regularization parameter λ2 balances trade-off between fidelity and sparsity of solution.

**Comparison:**
- Retrieval methods focus on proximity relationships with positive coefficients
- Linear combination can include structural information and potentially negative coefficients.

**Figure 2**: Demonstration of methods, highlighting similarity calculation's proximity relationships and linear combination's structural information.

## 4 Experiments

### 4.1 Implementation detail

**Experimental Validation Approach**
* Two foundational models used: Llama 3.1 8B (Dubey et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib7)) and SAM (Kirillov et al., [2023](https://arxiv.org/html/2410.09908v1#bib.bib13))
* Hardware: 8 H100 80G GPUs for training and fine-tuning

**Llama 3.1 8B Model Evaluation**
* Fine-tune four LoRA models derived from the pre-trained Llama 3.1 8B model using:
  * CT abdomen reports (24,801)
  * CT head reports (63,745)
  * MR image reports (18,157)
  * X-ray image reports (60,000)
* Consistent hyperparameter settings:
  + Training batch size = 8
  + Gradient accumulation steps = 4
  + Optimizer = paged adamw 32bit
  + Learning rate = 5∗10−65superscript1065*10^{-6}5 ∗ 10 start_POSTSUPERSCRIPT - 6 end_POSTSUPERSCRIPT
  + Weight decay = 0.001
  + Maximum gradient normal = 0.3
  + LoRA r = 16, LoRA alpha = 0.05
* Training epochs: CT abdomen (2), CT head (1), MR (3), X-ray reports (1)
* Testing: collecte 200 new reports for each type of medical image

**SAM Model Evaluation**
* Focus on medical image segmentation tasks
* Reproduce and train six individual MA-SAM models, each corresponding to one prostate dataset (Liu et al., [2020](https://arxiv.org/html/2410.09908v1#bib.bib17))
* Consistent hyperparameter settings as the MA-SAM framework
* Target dataset treated iteratively, while remaining datasets serve as reference datasets for zero-shot learning.

### 4.2 Medical report impression

**Medical Report Ensemble Models Evaluation**

**Similarity Calculation and Linear Combination**:
- Form ensemble models for each medical report type using both similarity calculation and linear combination
- Do not use regularization

**Evaluation Metrics**:
- **ROUGE-L**: Lin, 2004
- **BertScore**: Zhang et al., 2019
- **GPT score**: Shi et al., 2024
- Used to evaluate fundamental word matching and semantic level accuracy

**Performance Comparison**:
- **Pre-trained Llama 3.1 8B** vs. **SFT**, **Zero-shot Models**:
  - **AVG (Average)**: Ours (sim), Ours (lin), SFT
  - **CT Abdomen Medical Report Impression Task**:
    - Our models outperform zero-shot pre-trained model
    - Our linear combination model surpasses SFT in most metrics
    - Our similarity-based ensemble demonstrates competitive performance compared to SFT

**Weight Distributions**:
- **Similarity Ensemble vs. Average Ensemble**:
  - Slightly different weight distributions
  - Surpassing average ensemble in all metrics except GPT score
- **Linear Combination Model**:
  - Integrates 80% CT head model, 18% MR model, and knowledge from other reports
  - Contributes to overall performance improvement

### 4.3 Medical Image segmentation

**Experimental Analysis of LoRA Models**

**Datasets**:
- Trained LoRAs on 6 distinct datasets from various manufacturers
- Differences in signal strength and resolution introduced notable shifts in data distribution
- Challenged the performance of a single LoRA model, emphasizing the need for training on similar datasets

**Dataset Comparison and Correlation**:
- Figure 3 illustrates the correlation between dataset similarity and LoRA model accuracy:
  - Testing sets more similar to training sets tend to achieve higher accuracy
  - Substantiates the significant impact of dataset characteristics on model performance

**Linear Combinations for Optimizing Model Performance**:
- Computed the similarity between datasets and adjusted LoRA representations through linear combinations
- With and without regularization to optimize performance for each dataset
- Evaluated effectiveness using the DICE Score metric:
  - Pre-trained SAM model failed without LoRA
  - Regularized linear combinations (Ours (lin+R)) significantly outperformed other methods, comparable to supervised fine-tuning

**Weight Analysis**:
- Table 4 details weight distribution of testing set E compared to reference datasets
- Linear interpolation without regularization results in deviant weights and suboptimal performance
- Regularized linear combinations address distribution shifts and enhance robustness and overall performance.

### 4.4 Ablation Study

**Ablation Studies Evaluating LoRA Efficacy**
* Exploring nearest LoRA vs ensemble methods
	+ Nearest dataset's LoRA: Table [5](https://arxiv.org/html/2410.09908v1#S4.T5 "Table 5 ‣ 4.4.1 Nearest LoRA vs. Ensemble Methods ‣ 4.4 Ablation Study ‣ 4 Experiments ‣ Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning")
		- Solely relying on the most similar training set results in overfitting and variable outcomes
	+ Integrating multiple models provides a more robust performance across diverse datasets
* Using LoRAs derived from multiple training sets to enhance Supervised Fine-Tuning (SFT) performance: Table [6](https://arxiv.org/html/2410.09908v1#S4.T6 "Table 6 ‣ 4.4.2 Whether to Improve SFT ‣ 4.4 Ablation Study ‣ 4 Experiments ‣ Retrieval Instead of Fine-tuning: A Retrieval-based Parameter Ensemble for Zero-shot Learning")
	+ Linear combination of all LoRA variants on dataset C's testing set
		- Negative correlation between testing set of dataset C and training set of dataset A
	+ Enhancing SFT methods with potential for further improvement, marking a promising direction for future research.

## 5 Discussion and Future work

**Experimental Results**
* Our approach yields promising results for enhancing adaptability and efficiency of foundational models in tasks with scarce or unavailable labeled data
* Significant improvements observed using the RPE model

**Limitations and Further Discussion**
* Limited number of LoRAs available: potential improvements to encoder used to derive z𝑧zitalic_z (utilize pre-trained models or specifically train)
* Retrieval of large pools of LoRAs and efficient computation of weights: exploration needed for further compression techniques
* Future work focuses on enhancing scalability and efficiency of retrieval processes in privacy-sensitive or resource-constrained environments.

## 6 Conclusion

**RPE Model for Zero-Shot Learning**

- A new model introduced, capable of zero-shot learning without additional data or training
- Maintains data privacy and shows promise in medical application scenarios
- Reduces redundant computational resource consumption across community groups
- Potential to be a significant future framework

**Acknowledgments**
- Research supported by the National Institutes of Health under Grant R01HL159183 (Quanzheng Li)

