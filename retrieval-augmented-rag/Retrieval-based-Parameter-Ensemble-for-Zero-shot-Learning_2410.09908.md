# Retrieval-based Parameter Ensemble for Zero-shot Learning

Pengfei Jin, Peng Shu
https://arxiv.org/html/2410.09908

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

**Foundation Models and Deep Learning:**
- **Low-Rank Adaptation (LoRA)**: Efficient fine-tuning of large models
- **Retrieval-Augmented Generation (RAG)**: Grounds outputs in external information, improves performance
- Challenges: Extensive training or labeled data required

**Introducing Retrieval-based Parameter Ensemble (RPE):**
- New method for efficient adaptation to new tasks
- Creates a vectorized database of LoRAs
- Enables retrieval and application of model adaptations

**Benefits of RPE:**
- Minimizes need for extensive training
- Eliminates requirement for labeled data
- Effective in zero-shot learning
- Privacy-sensitive: Modifies parameters without accessing raw data

**Applications and Performance:**
- Medical report generation, image segmentation
- Surpassed supervised fine-tuning methods in certain cases
- Enhances computational efficiency and privacy.

## 1 Introduction

**Retrieval-Based Parameter Ensemble (RPE) Model:**

**Background:**
- Foundation models: CLIP, LLaMA, SAM
- Large datasets used for pre-training
- Applications in NLP, computer vision, healthcare
- Fine-tuning resource-intensive and requires extensive computational power and large-scale data

**Low-Rank Adaptation (LoRA):**
- Freezing most model parameters during fine-tuning
- Significantly reduces computational overhead while maintaining high performance
- Susceptible to hallucinations

**Retrieval-Augmented Generation (RAG):**
- Incorporates external retrieval step to ground model outputs in factual data
- Mitigates hallucination and supports zero-shot learning
- Allows models to handle new tasks or unfamiliar categories with minimal labeled examples

**Problem:**
- Fine-tuning delivers superior task-specific performance but requires extensive resources
- RAG mitigates hallucination and supports zero-shot learning but relies on raw data access, posing privacy concerns

**Solution: Retrieval-based Parameter Ensemble (RPE)**
1. Establish a vectorized database, LoRA-VecDB, containing LoRAs and their representations across various tasks.
2. When a new task arises, extract model's representation and query the database for similar LoRAs.
3. Combine retrieved LoRAs using weighted ensemble methods to adapt the model to the new task without extensive fine-tuning.

**Advantages:**
1. Reduces computational costs and redundancy associated with traditional fine-tuning methods.
2. Enhances privacy by avoiding raw data access during adaptation process.
3. Scalable solution for foundation models as they continue to grow in size and complexity.

**Contributions:**
1. Introducing zero-shot learning framework using LoRA retrieval, eliminating labeling or training requirements.
2. Analyzing parameter and feature spaces interaction to enhance model adaptability and accuracy.
3. Validating RPE model effectiveness in medical language and image processing tasks.

## 2 Related Work

**RAG vs Parameter Combination Methods vs Zero-shot Learning**

**RAG**:
- Integrates external knowledge into large language models (LLMs) by retrieving relevant information to enhance generation accuracy
- Recent advancements optimize query prompting, indexing structures, and retrieval mechanisms to address limitations of naive RAG approaches
- Enhances retrieval precision and reduces hallucinations in generated outputs, especially in low-resource domains
- Examples: Seo et al. (2024) generates new training samples with LLMs using retrieved instances; Parvez et al. (2022) expands positive examples through retriever models
- Challenges: Reliance on external data introduces challenges related to privacy and computational constraints

**Parameter Combination Methods**:
- Various methods developed to combine or utilize model parameters to enhance performance, robustness, and generalization
- Examples: Model Soup (2022) averages parameters from different models; Federated Learning (FL) (2017) trains models locally on their own data
- Benefits: Improves performance and efficiency, preserves privacy, and is particularly well-suited for privacy-sensitive applications
- Limitations: Scalability is often limited by the fixed number of available experts

**Zero-shot Learning**:
- Machine learning technique where a model is trained to recognize objects, categories, or concepts not seen during training
- Requires a specific task representation zisubscript𝑧𝑖z_{i}italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT to map θisubscript𝜃𝑖\theta_{i}italic_θ start_POSTSUBSCRIPT italic_i end_POSTSUPERSCRIPT ref end_POSTSUPERSCRIPT from known tasks to novel tasks
- Examples: DeViSE (2013) used a linear mapping from image features; GCN-ZL (2018) utilized Graph Neural Networks; DGP-ZL (2019) introduced Dense Graph Propagation
- Our work leverages pretrained models to obtain representations zisubscript𝑧𝑖z_{i}italic_z start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and uses a retrieval and algorithm-based method to perform the mapping 𝒜𝒜\mathcal{A}caligraphic_A, simplifying the generalization process and improving adaptability to new tasks

## 3 Method

**Components of Approach: LoRA-VecDB and Retrieval & Weighted Ensemble Mechanism**

**LoRA-VecDB:**
- Construction of a vectorized database for storing model adaptations and corresponding representations

**Retrieval & Weighted Ensemble Mechanism:**
1. **Transforming dataset for new task**: ztrg_{superscript𝑧trg}^{trg} ↔ {ziref}subscript𝑧𝑖ref\\{z_{i}^{\text{ref}}\\}
2. **Retrieving relevant LoRAs:**
   - {ziref}subscript𝑧𝑖ref\\{z_{i}^{\text{ref}}}
   - {δ⁢θiref}subscript𝜃𝑖ref\delta\theta_{i}^{\text{ref}}
3. **Computing weights**: wi_w_{i} based on similarity between ztrg and {ziref} in representation space
4. **Applying weights in parameter space**: adjust δ⁢θitrg\delta\theta_{i}^{\text{trg}} based on computed weights.

### 3.1 Construction of LoRA-VecDB

## Key Components
- **LoRA-VecDB**: Central repository cataloging LoRAs and their representations
- **Foundation Model**: Denoted as F(⋅,θ0)
- **Dataset**: Represented as Di
- **LoRA**: Denoted as δθi
- **Representation**: Denoted as zi

## Database Construction Process
- **LoRA Training**:
  - Uses foundation model with frozen pre-trained weights
  - Introduces trainable low-rank matrices
  - Reduces parameter count for adaptation

- **Feature Representation**:
  - Derives zi directly from F's encoder feature map
  - Optional additional encoder for enhanced interpretability
  - Uses EF(xj,θ0) for individual data items

## Feature Map Implementation
- Uses original, non-fine-tuned feature maps
- Maintains integrity of initial pre-training
- Aligns with MoE encoder component strategy

## Dataset Feature Representation
- **Initially Explored Methods**:
  - Chamfer distance
  - Nearest Neighbor Distance
  - Mean Distance

- **Final Approach**:
  - Uses averaged feature maps
  - Formula: zi = 1/|Di| ∑(xj∈Di) EF(xj,θ0)
  - |Di| represents dataset element count

## Benefits and Applications
- Supports scalable experimentation
- Facilitates efficient storage and retrieval
- Encourages community contributions
- Maintains collaborative resource
- Enables adaptation to new datasets and problems

## Technical Considerations
- Raw projection of data features maintained
- Simplified computational process
- Efficient storage in VecDB
- Original data representation integrity preserved
- Supports ongoing research and practical applications

### 3.2 Retrieval and Weighted Ensemble

**Algorithm for Retrieval-based Parameter Ensemble (RPE)**

**Step 1: Foundation Model F(⋅, θ0)**:
- Initial model with parameters θ0
- Performs feature representation on target dataset Dtrg using EF(xj, θ0)

**Step 2: Compute ztrg**:
- Calculate ztrg as the average of feature representations in Dtrg
- Divide by number of samples in Dtrg to obtain normalized vector

**Step 3: Retrieve nearest LoRAs (ziref):**
- For each zi in Dtrg, find k closest matches in {zi} using argsort and distance function d(zi, ztrg)
- Store these indices in a set {w_i} = 𝒜({ziref}, ztrg)

**Step 4: Calculate weights and new parameters (θtrg):**
- Compute weights based on the sum of weights w_i and LoRA adjustments δθiref for each i in [1, k]
- Update original parameters θ0 with new weights to obtain θtrg

**Step 5: Various Strategies for Parameter Interrelationships (Acaligraphic_A):**
- Different strategies to calculate effective parameter interrelationships based on latent space structures
- Transfer learning concept assumes tasks with similar feature representations benefit from similar parameter adjustments.

**Step 6: Hypothesis**:
- Specific correspondences between data representations and optimal parameters allow methods to deduce relationships between δθi and zi.

**Assumptions about connections between representation space and parameter space**:
- Influence the derivation of different Acaligraphic_A strategies
- Tailoring algorithms to better capture and leverage these relationships enhances model performance across varied datasets.

#### Linear Combination Method for Parameter Ensemble in Zero-Shot Learning

**Retrieval-based Parameter Ensemble for Zero-shot Learning: Strategies**

**Similarity Calculation:**
- Method based on linear relationship between latent representations and parameter adjustments
- Find a combination of retrieved LoRAs that best approximates target representation
- Normalize contribution from each LoRA to maintain normalized weight sum
- Calculate inverse distances between target feature vector and reference feature vectors using squared ℓ2 norm: |d²(zi, ztrg)|
- Weights assigned using softmax function: wi = exp⁡(−λ₁d²(zi, ztrg)) / ∑jexp⁡(−λ₁d²(zj, ztrg))
- Temperature parameter (λ₁) controls sharpness of distribution and emphasizes more similar LoRAs

**Linear Combination:**
- Minimize error between target representation and weighted sum of reference representations: wi = arg min⁡∑wi=1∥ztrg−∑wiziref∥²
- Normalization constraint: ∑wi=1 wi=1
- Regularization introduced to manage influence of each LoRA, especially with sparse or high-dimensional data
- Penalizes weights to encourage simpler solutions that generalize better using ℓ1 norm penalty: wi = arg min⁡∑wi=1∥ztrg−∑wiziref∥² + λ₂∥wi∥₁
- Regularization parameter (λ₂) balances trade-off between approximation fidelity and solution sparsity.

## 4 Experiments

### 4.1 Implementation detail

**Experimental Validation Approach**
- Two foundational models: Llama 3.1 8B (Dubey et al., [2024](https://arxiv.org/html/2410.09908v1#bib.bib7)) and SAM (Kirillov et al., [2023](https://arxiv.org/html/2410.09908v1#bib.bib13))
- **Llama 3.1 8B Model**: evaluated on generating medical report impressions using LoRA fine-tuning with:
  - CT abdomen reports: 24,801 reports, 200 new tests for evaluation
  - CT head reports: 63,745 reports, 200 new tests for evaluation
  - MR image reports: 18,157 reports, 200 new tests for evaluation
  - X-ray image reports: 60,000 reports, 200 new tests for evaluation
  - Training hyperparameters: batch size = 8, gradient accumulation steps = 4, optimizer = paged adamw 32bit, learning rate = 5∗10−65superscript1065*10^{-6}5 ∗ 10 start_POSTSUPERSCRIPT - 6 end_POSTSUPERSCRIPT, weight decay = 0.001, maximum gradient normal = 0.3, LoRA r = 16, LoRA alpha = 0.05
  - Training epochs: CT abdomen (2), CT head (1), MR (3), X-ray reports (1)
- **SAM Model**: focused on medical image segmentation tasks using MA-SAM framework with the same hyperparameter settings as in MA-SAM.
  - Six individual MA-SAM models trained for each prostate dataset, with remaining datasets used as reference datasets for zero-shot learning.

### 4.2 Medical report impression

**Performance Comparison of Models on Medical Report Impression Task:**
* **Metrics used for evaluation**: ROUGE-L (Lin, 2004), BertScore (Zhang et al., 2019), GPT score (Shi et al., 2024)
* **Models compared**: Pre-trained Llama 3.1 8B, LoRA Supervised Fine-tuning (SFT), and zero-shot models on CT abdomen medical report impression task.

**Table 1:** Performance comparison of models on CT abdomen medical report impression task:

|                         | Pre-trained | SFT      | Zero-shot | Our method (similarity) | Our method (linear) |
|------------------------|-------------|----------|-----------|-------------------------|---------------------|
| **AVG**                |             |          |          |                       |                   |
| CT abdomen            |              |          |           | 0.34                | 0.80              |
| MR                    |              |          |           | 0.18                | 0.18              |
| X-ray                |              |          |           | N/A                 | N/A                |
| **ROUGE-L**           | 0.1264       | 0.1387    | N/A      | 0.1369             | 0.1374            |
| BertScore (precision) | 0.7779       | 0.7789     | N/A      | 0.781               | 0.7815           |
| BertScore (recall)    | 0.8321       | 0.8355    | N/A      | 0.8348             | 0.835              |
| BertScore (F1)        | 0.8039       | 0.806     | N/A      | 0.8068            | 0.8071            |
| GPT score             | 2.89         | 3.215     | N/A      | 3.095             | **3.285**          |

* **Similarity ensemble** vs **linear combination**:
  * Our similarity-based ensemble model outperforms the average ensemble in all metrics except for GPT score.
  * The linear combination model achieves the best performance on CT abdomen reports across most metrics, surpassing SFT method.
* **Weight distributions**:
  * Similarity ensemble has slightly different weight distribution than the average ensemble.
  * Linear combination model integrates 80% weight from CT head model and 18% from MR model, which is reasonable given their similar patterns.

### 4.3 Medical Image segmentation

**LoRA Model Training and Evaluation**
- **Dataset diversity**: Introduces significant shifts in data distribution, challenging a single LoRA model
- **Necessity of similar datasets**: Enhances task performance for LoRA models
- **Analysis of datasets**: Confirms correlation between dataset similarity and LoRA model accuracy (Figure 3)
  * Left side: Ranking of testing sets' similarity to training sets
  * Right side: Corresponding LoRA model accuracy rankings
- **Linear combination models**: Adjusted for each dataset based on similarity, with and without regularization
  * Effectiveness evaluated using DICE Score metric (Table 3)
  * Pretrained SAM failed without LoRA
- **Findings**: Models employing regularized linear combinations significantly outperformed other methods
  * Comparable to supervised fine-tuning performance
  * Analysis of weights in Table 4 revealed that testing sets with significant distribution shifts benefit from regularization to address performance challenges.

### 4.4 Ablation Study

**LoRA Ablation Studies: Nearest vs. Ensemble Methods**

**Evaluating Efficacy of LoRAs**:
- A series of ablation studies to compare the efficacy of using the nearest LoRA (k=1) compared to an ensemble approach
- Exploring potential benefits of incorporating LoRAs derived from multiple training sets in enhancing performance through Supervised Fine-Tuning (SFT)

**Nearest vs. Ensemble Methods**:
- Table 5: DICE scores for different testing datasets using nearest LoRA
    * Using the most similar dataset directly (k=1) may result in overfitting to that specific dataset
    * Integrating multiple models provides more robust and stable performance across diverse datasets

**Improving SFT**:
- Table 6: Weight distribution of linear combination including Supervised Fine-tuning LoRA applied to testing dataset C
    * This approach is effective in scenarios where there is a shift in data distribution between training and testing datasets
    * Linear combination using all LoRAs on the testing set surpasses the performance of SFT, suggesting potential for further enhancing SFT methods

## 5 Discussion and Future work

**Limitations and Future Work**

**Improving Encoder**:
- Utilize a pre-trained model or specifically train an encoder to optimize weight determination
- Aims to enhance the adaptability and efficiency of the foundational models in tasks with limited labeled data

**Retrieval of Large Pool of LoRAs**:
- Efficiently retrieve and compute weights for large pools of LoRAs
- Explore further compression techniques to reduce memory requirements
- An open direction for future work to enhance scalability and efficiency of retrieval processes

**Improving Robustness and Applicability**:
- Focus on refining these aspects to fully leverage the potential of retrieval-based learning systems
- Particularly beneficial in privacy-sensitive or resource-constrained environments.

## 6 Conclusion

- Presented a **RPE model** for zero-shot learning without additional data or training and maintaining data privacy.
- Shown promising results in medical applications.
- Significantly reduces computational resource consumption for community groups.
- Could become an essential framework in the future due to its potential benefits.
- Research conducted by Quanzheng Li is supported, in part, by the National Institutes of Health under Grant R01HL159183.

