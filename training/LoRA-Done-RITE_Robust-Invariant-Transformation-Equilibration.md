# # LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization

Authors: Jui-Nan Yen (UCLA), Si Si (Google), Zhao Meng (Google), Felix Yu (Google), Sai Surya Duvvuri (UT Austin), Inderjit S. Dhillon (Google), Cho-Jui Hsieh (Google & UCLA), Sanjiv Kumar (Google)

Email addresses: junianyen@cs.ucla.edu, sisidaisy@google.com, mengzhao@google.com, felixyu@google.com, saisurya@cs.utexas.edu, isd@google.com, chohsieh@cs.ucla.edu, sanjivk@google.com

(Source: https://arxiv.org/html/2410.20625v1)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Transformation Invariance for LoRA Optimization](#2-transformation-invariance-for-lora-optimization)
  - [2.1 Definition of Transformation Invariance](#21-definition-of-transformation-invariance)
  - [2.2 Existing Optimizers are not Scalar Scale Invariant](#22-existing-optimizers-are-not-scalar-scale-invariant)
  - [2.3 Benefits of Transformation Invariance](#23-benefits-of-transformation-invariance)
- [3 Our Proposed Optimizer](#3-our-proposed-optimizer)
  - [3.1 Diagonal preconditioning is not enough for transformation invariance](#31-diagonal-preconditioning-is-not-enough-for-transformation-invariance)
  - [3.2 Achieving Transformation Invariance](#32-achieving-transformation-invariance)
  - [3.3 Theoretical Analysis](#33-theoretical-analysis)
- [4 Related Work](#4-related-work)
- [5 Experimental Results](#5-experimental-results)
- [6 Conclusions](#6-conclusions)
- [Appendix A Appendix](#appendix-a-appendix)
  - [A.1 Hyperparameters](#a1-hyperparameters)
  - [A.2 Dataset](#a2-dataset)
  - [A.3 Proof of Theorem 1](#a3-proof-of-theorem1)
  - [A.4 Proof of Theorem 2](#a4-proof-of-theorem2)
  - [A.5 Proof of Theorem 3](#a5-proof-of-theorem3)
  - [A.6 Gemma2 Experiment Result](#a6-gemma2-experiment-result)

## Abstract

**LoRA-RITE: A Novel Adaptive Matrix Preconditioning Method for LoRA Optimization**

**Background:**
- Low-rank adaption (LoRA) is a popular parameter-efficient finetuning method for large language models (LLMs)
- Current LoRA optimizers lack transformation invariance, leading to suboptimal solutions and inefficient learning

**Introducing LoRA-RITE:**
- Novel adaptive matrix preconditioning method for LoRA optimization
- Achieves transformation invariance while being computationally efficient

**Benefits of LoRA-RITE:**
- Theoretical analysis demonstrates the benefit of this method
- Improvements over existing optimizers on various LLM tasks
  - Gemma 2B, 7B, and mT5-XXL models
  - Consistent improvements across multiple benchmarks: Super-Natural Instructions, HellaSwag, ArcChallenge, GSM8K, OpenBookQA

**Results:**
- Replacing Adam with LoRA-RITE during LoRA fine-tuning of Gemma-2B yields:
  - **4.6% accuracy gain** on Super-Natural Instructions
  - **3.5% accuracy gain** across other four LLM benchmarks

## 1 Introduction

**LoRA (Low-Rank Adaptation)**
* Popular parameter-efficient method for fine-tuning Large Language Models (LLMs)
* Freezes pretrained weights, introduces low-rank matrices into each layer: WinR^mxn → W + Z = A_1B_1^⊤ = A_2B_2^⊤
* Reduces memory requirements and mitigates overfitting in limited data settings
* Recent research explores improvements over classic LoRA algorithm

**Challenges with Standard Optimizers for LoRA:**
* Updates not transformation invariant: Z = A_1B_1^⊤ = A_2B_2^⊤, but optimizers may yield different updates based on factorization
* Violates mathematical consistency and leads to inefficiencies during training
* One LoRA factor often dominates optimization process while the other remains nearly fixed
* Recent approaches employ different learning rates for factors partially mitigate this issue

**Motivation for a More Principled Optimizer:**
* Prove diagonal preconditioners cannot achieve transformation invariance
* Use matrix preconditioners instead

**LoRA-RITE (Robust Invariant Transformation Equilibration):**
* Novel optimizer designed specifically for LoRA optimization
* Employs a transformation-invariant preconditioner on the low-rank side
* Achieves transformation invariance without significant overhead
* Maintains transformation invariance when incorporating first and second moments

**Contributions:**
* Propose LoRA-RITE, the first adaptive matrix preconditioning optimizer for LoRA that is transformation-invariant
* Provide a convergence analysis for the method
* Utilizes matrix preconditioners with little overhead compared to first-order optimizers when r (LoRA rank) is much smaller than m and n
* Significantly improves performance across multiple datasets and architectures:
  * GSM8K dataset with Gemma 7B IT model: LoRA-RITE achieves a 55.50% accuracy rate, outperforming Adam (48.37%) and Lamb (50.64%).

## 2 Transformation Invariance for LoRA Optimization

**Transformational Invariance in LoRA Training**
- Explanation of transformational invariance concept within LoRA training
- Demonstration that many existing optimizers fail to meet this property when used with LoRA
- This deficiency results in inefficient learning in practice

### 2.1 Definition of Transformation Invariance

**LoRA Optimization: Transformation Invariance**

**LoRA Factors and Equivalence**:
- LoRA adds a low-rank matrix Z=AB^⊤ to the original weight matrix W
- Many different LoRA factors (A_1,B_1),(A_2,B_2) can represent the same finetuned weight Z
- Different parameterizations can lead to different updates to Z
- This suggests a serious inconsistency and potential for suboptimal updates

**Definition of Transformation Invariance**:
- **Transformation Invariance**: An optimizer's updates (δA_1,δB_1) and (δA_2,δB_2) satisfy A_1(B_1+δB_1)^⊤ = A_2(B_2+δB_2)^⊤
- Ensures the optimizer produces the same update, δZ, to the fine-tuned weights for any equivalent LoRA factorizations

**Condition for Transformation Invariance**:
- δA_1B_1^⊤ = δA_2B_2^⊤, A_1δB_1^⊤ = A_2δB_2^⊤, δA_1δB_1^⊤ = δA_2δB_2^⊤

**Scalar Scale Invariance**:
- Weaker version of transformation invariance
- Requires updates remain equivalent when LoRA factors are scaled up or down by a constant factor

### 2.2 Existing Optimizers are not Scalar Scale Invariant

**Scalar Scale Invariance in LoRA Optimization**

**Gradient Descent**:
- Not scalar scale invariant for LoRA optimization
- First term has a scale difference of 1/s^2
- Second term has a scale difference of s^2
- Third term remains identical
- Gradients can be arbitrary large when s goes to 0 or infinity

**Adaptive Updates (e.g., Adam)**:
- Also not scalar scale invariant for LoRA optimization
- Discrepancy in scale of terms in condition ([4](https://arxiv.org/html/2410.20625v1#S2.E4))
- Failing to satisfy scalar scale invariance

**Existing Optimizers**:
- Gradient Descent, Adam, Adagrad [^4], RMSProp [^25], and Shampoo [^8] are not scalar scale invariant for LoRA optimization.

### 2.3 Benefits of Transformation Invariance

**Importance of Transformation Invariance**
- **Transformation invariance**: ensures equivalent weight updates for same parameters
- Demonstrates more efficient feature learning as network width grows

**Efficient Feature Learning**
- Describes asymptotic training behavior of LoRA
- Requires both δAB^⊤x and AδB^⊤x to be O(1) with respect to network width n

**Conventional Optimizers vs. Transformation-Invariant Optimizer**
- Conventional optimizers do not satisfy efficient feature learning (Figure 1 in [^9])
- Lacking scalar scale invariance leads to one LoRA factor updating improperly while the other remains unchanged

**Theorem 1**
- Transformation-invariant optimizer with same update rule for A and B guarantees efficient feature learning

**Proof of Theorem 1**
- Given in Appendix.

## 3 Our Proposed Optimizer

**Proposed Algorithm**:
- Achieves transformation invariance
- Significantly outperforms previous LoRA optimizers in various tasks

Note: This response is concise and summarizes the main points from the original passage while maintaining its meaning.

### 3.1 Diagonal preconditioning is not enough for transformation invariance

**Diagonal Preconditioning vs Transformation Invariance for LoRA Optimization**

**Existing Optimizers**:
- Diagonal preconditioning methods like Adam and Adagrad assume the preconditioner P is a diagonal matrix
- Matrix preconditioning methods such as Shampoo and CASPR use non-diagonal P

**LoRA Factors AinR^mxr and BinR^nxr**:
- A_2 = A_1 * R
- B_2 = B_1 * R^-⊤, where R is an invertible matrix

**Gradient Calculation**:
- ∇A_2 = ∇ZB_2 = ∇(A_1 * R^-⊤)

**Diagonal Preconditioning**:
- vec(δA)^⊤ = δA and vec(∇A)^⊤ = ∇A
- δA = -∇AP^⊤δA, B^⊤ = -∇AP^⊤B^⊤

**Transformation Invariance**:
- If A_1 and A_2 are equivalent LoRA pairs with preconditioners P_1 and P_2:
  - δA_2 * B_2^⊤ = -∇A_2 * P_2^⊤ * B_2^⊤
  - δA_1 * B_1^⊤ = -∇A_1 * (R^-⊤ * P_2 * R^-1) * B_1^⊤
- For arbitrary R, there might not exist diagonal preconditioners P_1 and P_2 that satisfy this condition, leading to non-diagonal transformation invariance

**Conclusion**:
- Matrix preconditioning is necessary to achieve transformation invariance.

### 3.2 Achieving Transformation Invariance

**LoRA-RITE Algorithm: Transformation Invariance for LoRA Optimization**

**Background:**
- Recognize that LoRA factors A and B can be decomposed into orthogonal bases and magnitudes
- U_A, U_B obtained through polar decomposition
- Gradients depend on both basis and magnitude

**Transformation Invariance:**
- Introduce concept of "unmagnified gradients" (∇̅A, ∇̅B)
- Remain invariant to transformations of LoRA factors
- Used for adaptive matrix preconditioning in proposed optimization method

**Update Rule for A:**
1. δA_t=-∇̅A_t(V̅_A_t+ρ_A_tI)^-1/2
2. Split into two parts:
   a) Adaptive preconditioning mechanism using unmagnified gradients
   b) Magnitude adjustment for different LoRA pairs
3. Satisfies transformation invariance condition ([4](https://arxiv.org/html/2410.20625v1#S2.E4))

**Incorporating Second Moment:**
- Accumulate second moment: V̅_A_t=P_A_tV̅_A _t-1P_A_t^⊤+∇̅A_t^⊤∇̅A_t
- Account for varying basis at each step using projection matrix P_A_t
- Monitor loss of information with d_λ()
- Compensate by adding ρ_A_tI to preconditioner
- Final proposed update rule: S̅_A_t=∇̅A_t(V̅_ A_t+ρ_A_tI)^-1/2

**Incorporating First Moment:**
- Maintain first moment using projection matrix M̅_A_t
- Update rule: M̅_A_t=β_1M̅_A_t-1P _A_t^⊤+(1-β_1)S̅_A_t
- Final proposed update rule: δA_t=-M̅_A_tR_B^-⊤

**LoRA-RITE Algorithm:**
1. Initialize: unmagnified first and second moment (M̅_A_0=0, V̅_A_0=0)
2. For t=1 to T do:
   a. Compute gradient ∇A_t
   b. Compute QR decomposition of LoRA factor B_t
   c. Compute unmagnified gradient ∇̅A_t and P_A_t
   d. Update second moment: V̅_A_t, ρ_A_t, escaped mass
   e. Update preconditioned step: S̅_A_t
   f. Update first moment: M̅_A_t
   g. Update model parameters A_t+1 with δA_t
3. end for

### 3.3 Theoretical Analysis

**Online Convex Optimization Analysis:**
* In online optimization setting, a parameter **θ\_t** is chosen iteratively based on convex decision set **𝒦**.
* After each decision-making step **θ\_t**, a convex loss function **f\_t** is revealed, potentially adversarially.
* Regret accumulated by the algorithm up to step T: **Regret\_T = ∑\_{t=1}^T f\_t(θ\_t) - min\_{θ∈𝒦} ∑\_{t=1}^T f\_t(θ)**.

**Bounding First-Order Term:**
* Bound the first-order term: **∇θ\_t^⊤(θ\_t-θ^*)**.
* Use convexity of **f\_t** to connect it to loss function.
* In LoRA case, loss functions are not convex with respect to **θ**, so we directly bound the first-order term instead.

**Constraints on Frobenius Norm:**
* Assume constrains for fine-tuned weight Z of each layer: **A\_F ≤ D\_A, B\_F ≤ D\_B**.
* Gradient satisfies: **∇Z\_F ≤ G**.

**Convergence Analysis:**
* Analyze convergence in simplified scenario with omitted first moment and summed second moment.
* LoRA-Rite method yields the following result:
	+ Theorem 3: **LoRA-RITE satisfies**: **1/T ∑\_{t=1}^T 1/η∇θ\_t^⊤(θ\_t-θ\_{t+1}) = O(GT^-1/2)**.
		- Suggests convergence to a stable solution or no change in function value.
* Introduce additional assumption: Assumption 1, constraining **Q\_A\_t** and **P\_A\_t**.
* Stronger convergence result (Theorem 4): **1/T ∑\_{t=1}^T∇θ\_t^⊤(θ\_t-θ^*) = O(GD\_AD\_BT^-1/2)**.

**Comparison with One-sided Matrix Adagrad:**
* Regret bound of LoRA-RITE: **O(GD\_AD\_BT^-1/2)** vs. one-sided matrix Adagrad: **O(G(D\_A^2+D\_B^2)T^-1/2)**.
* Advantage in imbalanced cases: **D\_A^2 + D\_B^2 >= 2D\_AD\_B**.
	+ Previous work has shown LoRA factors often exhibit such imbalances.

## 4 Related Work

**Related Optimizers**
* **Adagrad**: Accumulated second moments used for scaling updates in each coordinate (diagonal preconditioner)
* **Adam, RMSProp**: Standard methods for deep neural network training using diagonal preconditioners; lack transformation invariance when applied to LoRA
* **Shampoo**: Higher-order preconditioners with 𝒪(m^2+n^2) additional memory overhead and periodic computation of roots L and R with 𝒪(m^3+n^3) computational cost
* **LARS, Lamb**: Lack transformation invariance but ensure scalar scale invariance

**LoRA Optimization Variants**
* **DyLoRA, IncreLoRA, AdaLoRA**: Dynamically adjusting LoRA rank during training
* **DoRA, DeepLoRA**: Enhancing LoRA performance through addition of scaling matrices
* **LoRA+**: Uses two different learning rates for LoRA weights; leads to an extra hyperparameter to be tuned in practice
* **Riemannian gradient descent**: Only method in literature that satisfies transformation invariance, but does not incorporate momentum and adaptivity (ScaledAdam)

**Table 1: Experimental Results on Super-Natural instruction dataset**
(Results not provided in text)

**Table 2: Experimental Results on LLM benchmarking datasets**
(Results not provided in text)

## 5 Experimental Results

**LoRA Optimizer Evaluation Results**

**Comparison of Optimizers:**
- **Adam**: Most widely used default optimizer for LoRA finetuning
- **LoRA+**: Adam with different learning rates for A and B (B 4x larger than A)
- **ScaledAdam**: Variant of Adam designed for LoRA optimization
- **Shampoo**: Adaptive matrix preconditioning method
- **Lamb**: Variant of Adam that normalizes updates based on parameter norms
- **LoRA-RITE**: Proposed optimizer, transformation invariant

**Evaluation Datasets:**
- SuperNatural Instructions dataset: Collection of diverse NLP tasks
- 4 standard LLM benchmarking datasets: HellaSwag, ArcChallenge, GSM8K, OpenBookQA

**Evaluation Metrics:**
- Exact match accuracy (classification)
- ROUGE-L score (generation)

**Results on SuperNatural Instructions Dataset:**
- LoRA-RITE demonstrates superior performance across both classification and generation tasks
- Improvements over Adam: 2.3% to 4.9% accuracy on classification tasks, significant improvements in global training setting
- Lamb performs well but has significant gap from LoRA-RITE
- Transformation invariance crucial for optimal performance

**Results on Other LLM Benchmarking Datasets:**
- LoRA-RITE achieves best performance on all datasets, Lamb usually second best

**Ablation Study:**
- Consistent performance across different LoRA ranks and model architectures (Gemma 2B, mT5-XXL)
- Successfully applied to encoder-decoder architecture (mT5-XXL)

**Training Speed Comparison:**
- Small overhead of LoRA-RITE compared to first-order methods
- Slightly slower than Adam on Gemma 2B, difference decreases with larger model size
- Shampoo slower due to recomputing preconditioner infrequently

## 6 Conclusions

**New Transformation-Invariant Optimization Algorithm for LoRAs**

Current LoRAs lack transformation invariance, causing disparate updates and hindering feature learning. This often leads to suboptimal solutions. We present a novel algorithm that maintains comparable time/memory overhead as Adam while being transformation-invariant. Our algorithm consistently delivers higher accuracy than existing LoRA optimizers across various datasets and models.

## Appendix A Appendix

### A.1 Hyperparameters

**Hyperparameter Settings (Table 5)**

Based on initial experiments, we set weight decay and dropout probability to zero as they do not improve baseline performance when given a non-zero value. This information can be found in Appendix A.1 of the paper "LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization."

### A.2 Dataset

**Table 6**: Summary Information of LLM Benchmarking Datasets (<https://arxiv.org/html/2410.20625v1#A1.T6>) shows the overview of these datasets. For evaluation purposes, we use the test set, since it is more extensive than the development set, for assessing ArcChallenge.

### A.3 Proof of [Theorem 1](https://arxiv.org/html/2410.20625v1#Thmtheorem1 "Theorem 1. ‣ 2.3 Benefits of Transformation Invariance ‣ 2 Transformation Invariance for LoRA Optimization ‣ LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization")

**Bulleted Notes (Concise Version)**

- Notation: A\_1 = θ(n^a), B\_1 = θ(n^b), ∇Z = θ(n^c), η = θ(n^d), where n is the network width and η represents the learning rate. Z = A\_1B\_1^T, from chain rule we know ∇A = ∇ZB, ∇B = ∇Z^T\*A
- Update Rule: Updates expressed as δA\_1 = θ(n^xa + yb + zc + d), δB\_1 = θ(n^xb + ya + zc + d)
- Scalar Invariance: If the update rule is scalar scale invariant, then for any A\_2 = n^\_δA\_1, B\_2 = n^\_-δB\_1 we have δA\_1B\_1 = δA\_2B\_2, which implies xa - (y+1)δ = 0 => y = x-1
- Equality of Updates: δA\_1B\_1 = θ(n^xa + (x-1)b + sc + d), A\_1δB\_1 = θ(n^xb + (x-1)a + sc + d) => xδ - (x-1)δ = 0 for all δ, thus x = y+1
- Efficient Feature Learning: By selecting a proper learning rate (η=θ(n^d)), AδBx = δAB x = θ(1), where x is the input vector

### A.4 Proof of [Theorem 2](https://arxiv.org/html/2410.20625v1#Thmtheorem2 "Theorem 2. ‣ Incorporating first moment. ‣ 3.2 Achieving Transformation Invariance ‣ 3 Our Proposed Optimizer ‣ LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization")

**Consistency of Matrices in LoRA Optimization**

- In LoRA optimization, matrices `X_AinR^mxr` and `H_AinR^rxr` are defined as consistent if their transposes, `U_BU_B^⊤` and `U_BH_AU_B^⊤`, are identical across all LoRA pairs.
- Given that `U_BU_B^⊤` is the same for all pairs, `(∇̅A)^⊤∇̅AU _B^⊤ = U_BU_B^⊤∇Z^⊤ ∇ZU_BU_B^⊤` implies that `(∇̅A)^⊤∇̅A` is consistent.
- If `bar{V}_{bm{A}}_t-1}` is consistent, then `{{bm{P}}_{bm{A}}_t}}}bar{{bm{V}}}_{bm{A}}_t-1}}{bm{P}}_{bm{A}}_t} }^^T}` is consistent as well. When `bar{{bm{V}}}_{bm{A}}_0}=bf{0}`, it implies that `bar{{bm{V}}}_{bm{A}}_t}` is consistent.
- Using the equation `U_B(V̅_A_t+ρ_A_tI)^ -1/2U_B^⊤=(U_BV̅_A_t U_B^⊤+ρ_A_tU_BU_B^ ⊤)^-1/2`, both `S̅_A_t` and `M̅_A_t` are consistent, thereby completing the proof.

### A.5 Proof of [Theorem 3](https://arxiv.org/html/2410.20625v1#S3.Ex27 "Theorem 3. ‣ 3.3 Theoretical Analysis ‣ 3 Our Proposed Optimizer ‣ LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization")

**Lemma 1 (Online Optimization)**
* For matrix XinR^mxr , HinR^rxr : X_H = Tr(XHX^⊤)^1/2
* Lemmas for online optimization: Lemma 1 (Lemma 5.13 [^10]) and Lemma 2 (Lemma 5.13, 5.14 [^10]).

**Preconditioner X_A_t**
* X_A_t = R_B_t^-1V̅_A _t^-1/2R_B_t^-⊤
* Unmagnified preconditioner: X̅_A_t = V̅_A_t^-1/2
* Lemma 2 applied to A factor: ∑_t=1^Tvec(∇A_t)^⊤ vec(δA_t) ≤ η∑_t=1^T∇A_t_X_A_t ^2 = η∑_t=1^T∇̅A_t_X̅_A _t^2 ≤ 2η*Tr(V̅_A_T^1/2)

**Proof of Theorem 3**
* Bounding the third term: (A_t-A_*)R_B_t^⊤²X̅_A_t^-1 - Q_AX̅_A_t-1^-1Q_A_t ≤ D_A²D_B²X̅_A_t^-1 - P_AX̅_A_t-1^-1P_A_t
* Inequality: X̅_A_t^-1 ≽ P_A_tX̅_A_t-1^-1P_A_t^⊤
* Summing up the second and third term: (2η+1/ημ D_A²D_B²)*Tr(V̅_A_T^1/2)
* Choosing η = (1/√(2))μ^1/2D_AD_B : O(D_AD_BGT^-1/2).

### A.6 Gemma2 Experiment Result

**Gemma2 Experimental Results** \
Conducted experiments on Gemma2-2B and presented results in Table 8 [1]. Proposed method shows a slight advantage over baselines, but performance difference between methods is low due to limited improvement from LoRA finetuning for Gemma2 on these tasks.

\*Reference: [1] A.6 Gemma2 Experiment Result ‣ Appendix A ‣ LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization

