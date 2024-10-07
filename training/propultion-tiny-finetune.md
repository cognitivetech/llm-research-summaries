# Propulsion: Steering LLM with Tiny Fine-Tuning

by Md Kowsher, Nusrat Jahan Prottasha, Prakash Bhat
https://arxiv.org/pdf/2409.10927v2

## Abstract
**Propulsion: A Novel Parameter-Efficient Fine-Tuning Method**

**Background:**
- Large Language Models (LLMs) revolutionized natural language processing (NLP)
- Fine-tuning LLMs for specific tasks is computationally expensive and risks degrading pre-learned features

**Problem Statement:**
- Challenges: high computational overhead, risk of overwriting existing knowledge

**Propulsion Proposed Solution:**
- Inspired by controlled adjustments in physical motion
- Selectively rescales specific dimensions of a pre-trained model
- Guides output predictions towards task objectives without modifying parameters
- Introduces lightweight, trainable Propulsion parameters at the pre-trained layer
- Minimizes number of updated parameters during fine-tuning

**Benefits:**
- Prevents overfitting or overwriting existing knowledge
- Approximates performance of full fine-tuning with fewer trainable parameters (as shown by Neural Tangent Kernel theory)
- Reduces parameter count from 355.3 million to 0.086 million - a 10x reduction compared to standard approaches like LoRA

**Availability:**
- Code available at: https://github.com/Kowsher/Propulsion

## 1 Introduction

**Introduction:**
- Training large language models consumes significant resources and takes up to six months
- Mitigated by fine-tuning pre-trained models like BERT, GPT, RoBERTa instead of training from scratch
- Challenges: large sizes (up to 7 billion parameters), traditional full model fine-tuning is expensive and inefficient

**PEFT Techniques:**
- Adapter layers, prompt tuning, low-rank adaptation, quantization, lightweight fine-tuning propose modifications that require fewer params.
- Propulsion introduces a novel approach for fine-tuning based on small changes to output vectors of neural network layers.

**Propulsion:**
- Leverages observation that small changes in output vectors can lead to substantial shifts in model behavior
- Applies minimal, strategic adjustments or re-scalings to pre-trained dimensions as a "steering" mechanism for the model's responses
- Introduces trainable linear parameters ("Propulsion parameters") to amplify/attenuate specific aspects of the model's behavior and optimize performance on tasks with minimal computational overhead

**Comparison:**
- Compares Propulsion to other PEFT methods in terms of efficiency, performance, and adaptability
- Analyzes Propulsion within the NTK framework to ensure similar performance to full fine-tuning despite reduced computational requirements.

**Benefits:**
- Outperforms current PEFT techniques while requiring fewer trainable parameters (e.g., 12 times fewer than AdaLoRA)
- Minimal computational overhead and maximally retains pre-learned features.

## 2 Propulsion

**Propulsion Concept and Its Practical Benefits**
* Pretrained language model M with N layers: L = {Li, ..., LN}
* Input x ∈ Rs×d: sequence length s, dimension d
* Each layer's output or next following layer's input
* Freeze all parameters in pre-trained weights W ∈ Rdin×dout

**Extracting Output Vi from Pretrained Model**
* Represents i-th token representation of output Vi from Li
* Extract using pre-trained frozen weight W
* Input x to extract output: Vi = Li(x;W) ∈ Rs×d

**Introducing Task-Specific Modifications with Propulsion Matrix Z**
* Performs scalar transformation on each element vj in output projection of Li
* Initializes trainable Propulsion matrix Z ∈ RN×dout
* Each zi = {z1, z2, ..., zdout} ∈ RN×dout performs transformation on corresponding element vj in Vi
* Train by calculating element-wise multiplication vj⊙zi to generate vj′
* Define as: vj' = [v1·z1, v2·z2, ..., vdout·zdout] (Equation 1)

**Steering Output of Each Layer with Propulsion Parameters**
* For all tokens, steer output of Vi by training Z
* Transformed output V'i used as input to next layer Li+1
* Define as: V'i = [v1⊙zi, v2⊙zi, ..., vs⊙zi] (Equation 2)

**Polynomial Scaling for Flexible Adjustment of Model's Responses**
* Incorporate polynomial scaling into Propulsion parameter zi
* Allows for more flexible and dynamic adjustment of model's responses to input features
* Adjust magnitude of propulsion effect by raising zi to the power of k
* Define as: V'i = [v1⊙zik, v2⊙zik, ..., vs⊙zik] (Equation 3)

**General Structure of Propulsion Method in Transformer Block**
* Modifies output of K, Q, V, and MLP matrix through element-wise multiplication with trainable Propulsion parameters
* Illustrated in Figure 2.

## 3 Neural Tangent Kernel (NTK) Analysis

**Neural Tangent Kernel (NTK) Analysis**
- **Introduction**:
  - NTK introduced by Jacot et al. (2018) to characterize how small changes in network parameters affect its output
  - In the NTK regime, where network width becomes very large, training dynamics of neural networks are determined by the NTK, which remains nearly constant during training
- **Propulsion Method Analysis in NTK Regime**:
  - Propulsion method: model output at time step t is based on pre-trained and fixed base matrix θ0, and a propagation matrix zt updated during training
  - Fully fine-tuned model output at time step t (φF) is compared to the propagation method's output (φP) in the NTK regime:
    - **Theorem 1**: under the NTK regime, the NTK for Propulsion fine-tuning approximates the NTK for full fine-tuning with high probability
    - **Formally**, the NTK for Propulsion satisfies: KF(xi, xj) ≈ KP(θ0xi, θ0xj) for inputs x and x' in Rd
    - The error between the NTK for Propulsion and the NTK for full fine-tuning can be bounded using the **Johnson-Lindenstrauss Lemma**
- **Propulsion Method in Transformer Block**:
  - Red cells represent trainable parameters, blue cells represent frozen ones
  - All layers use the same Propulsion matrix but are modified by their corresponding vector zi

## 4 Experiments

**Experiments Evaluating Methods on NLP Tasks:**
- **GLUE benchmark**: Included for evaluation along with question answering, text summarization, common sense reasoning, arithmetic reasoning
- **Training and algorithm details**: Described in Appendix E

**Baselines:**
- Uses popular PEFT methods as comparisons: Adapter (Houlsby et al., 2019), Prompt Tuning (Lester et al., 2021), Prefix-Tuning (Liang & Li, 2021), IA3 (Liu et al., 2022a), Bitfit (Zaken et al., 2021), LoRA (Hu et al., 2021), AdaLoRA (Zhang et al., 2023), MAM Adapter (He et al., 2021), PROPETL (Zeng et al., 2023), LoKr (Edalati et al., 2022), and LoHa (Hyeon-Woo et al., 2021)
- Implementations used for these methods come from Hugging Face (Mangrulkar et al., 2022)
- Experimental setup follows that of Xu et al. (2023) for GLUE benchmark, and Zhang et al. (2023) for other datasets.

### 4.2 Language Model

**Performance Datasets and Benchmarks:**
- **GLUE Benchmark**: Evaluated Propulsion method on CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, and RTE tasks.
- Used SQuAD v1.1, SQuAD v2.0 for question answering tasks
- XSum and CNN/DailyMail datasets for text summarization

**Model Selection & Hyperparameter:**
- **GLUE benchmark**: Selected RoBERTa-base (RoB B) with 125M parameters and RoBERTa-large (RoB L) with 355M parameters from Liu et al.
- Set Propulsion degree to specific values for different tasks: SST-2, QQP, RTE = 15; QNLI, MRPC = 55; other GLUE datasets = 20
- For question answering tasks: Used DeBERTaV3-base with Propagation degree set to 35 for both SQuAD v1.1 and SQuAD v2.0
- For text summarization: Chose the BART-large model with 406M parameters and set Propulsion degrees to 35 (XSum) and 25 (CNN/DailyMail)

**Results:**
- On GLUE dataset, Propulsion exhibits enhancements in accuracy: +2.68%, +3.32%, +3.31% over AdaLoRA, PROPETL Prefix, and (IA)3 respectively. Also, improvements in F1 score: +1.94%, +1.87%, and +8.92%
- For question answering tasks, Propulsion outperforms other PEFT methods on both SQuAD datasets. Beats AdaLoRA by 0.66 in EM and 0.51 in F1 score while being 7.89 times smaller in parameter size.
- Comparing to LoKr with fewer trainable parameters, Propulsion outperforms LoKr by 2.92 in EM and by 2.29 in F1-score.
- On XSum and CNN/DailyMail datasets, Propulsion achieves on-par performance while having significantly smaller parameter sizes compared to LoRA and AdaLoRA.

### 4.3 Large Language Models

**Evaluation Methods**
- Thorough evaluation using thirteen benchmark datasets: BoolQ, PIQA, SIQA, HelSwag, WinoGrande, ARC-easy, ARC-challenge, OBQA, MultiArith, GSM8K, AddSub, SingleEq, SV AMP
- For common sense reasoning: BoolQ, PIQA, SIQA, H.Swag, W.Grande, ARC-easy, ARC-challenge, OBQA
- For arithmetic reasoning tasks: MultiArith, GSM8K, AddSub, SingleEq, SV AMP

**Performance of Propulsion Method on Commonsense and Mathematical Reasoning Tasks**
- Outperforms state-of-the-art PEFT methods on both common sense and mathematical reasoning tasks
- Across all four LLMs tested: competitive performance on all benchmarks while maintaining the highest accuracy on some datasets (e.g., GSM8K)
- Significant improvements in accuracy for specific datasets compared to state-of-the-art PEFT method: SIQA (1.33%), HellaSwag (0.97%), AddSub (0.97%), SV QMP (0.66%)
- Maintains or improves accuracy while having fewer total trainable parameters compared to other baselines.

**Comparison of Propulsion Method with Other Methods in Terms of Space/Time Complexity and Total Trainable Parameters (TTPs)**
- Table 4 shows the space/time complexity, total trainable parameters for Propulsion method and baseline methods for single layer W∈Rd×d
- Definitions: dk, dv, df, lp as dimensions of learned vectors in (IA)3; lp as length of prompt added to input/layers in prompt tuning and prefix-tuning. For LoRA-type methods, r represents rank dimensions.

**Evaluation Methodology**
- Fine-tuned several LLMs: BLOOMz (7B parameters), GPT-J (6B), LLaMA 7B (13B), and GPT-J 6B with Propulsion methods for comparison
- Set Propulsion degree to 15 for both reasoning tasks, dropout rate of 0.1 for hidden layers and attention mechanisms, along with L2 regularization
- Applied different Propulsion parameters to assess effectiveness.

### 4.4 Efficiency Comparison

**Study Evaluation of PEFT Techniques:**
* Convergence comparison using SST-2 dataset: Figure 3 (Left)
	+ Propulsion exhibits faster convergence, achieves higher accuracy (0.9) in fewer iterations (50) compared to other methods
	+ AdaLoRA requires approximately 200 iterations for an accuracy of 0.87
	+ LoRA requires around 75 iterations for comparable accuracy to Propulsion
	+ Other methods like LoKr, (IA)3, and LoHa take more than 150 iterations to reach 0.8 accuracy
* Parameter efficiency: Tables 1, 2, and Figure 3 (Right)
	+ Propulsion demonstrates superior efficiency with faster training time and reduced memory usage due to parameter reduction
* Space/time complexities and total trainable parameters comparison: Table 4
	+ Propulsion outperforms other PEFT methods in terms of parameter count and computational resources
* Memory efficiency: Figure 4, approximately 17.2GB for Propulsion vs. over 26.0GB for other PEFT methods (approx. 1.5 times more memory-efficient)
* Delta weight reparameterization comparison during optimization: Appendix D (Table 6)

### 4.5 Ablation Study Propulsion

**Propulsion Degree Impact on Model Performance:**
* Exploring impact of Propulsion degree on accuracy for SST-2, QNLI, MRPC (Figure 5)
  * Highest accuracy: SST-2 (95% at degree 25), QNLI (94% between 50 and 75 degrees), MRPC (92% around 25 degrees)
* Overfitting after peak accuracy as Propulsion degree increases
* Training dynamics on SST-2 (Figure 5, right): Faster convergence for lower degrees vs longer convergence time for higher degrees.
* By 2000 steps, all degrees converge but lower degrees stabilize faster suggesting better performance.

**Positional Impact of Propulsion:**
* Ablation analysis on various layers: embedding, MLP, Key (K), Query(Q), Value (V), combinations of layers
* Adding Propulsion to attention mechanism (K + V + Q): accuracy 93.72% on SST-2 dataset
* Individual layer performances: Query (91.52%), Value (92.52%), and Query, Value, and Embedding layers outperform Propagation in embedding layer.
* Substantial accuracy improvements for all layers: SST-2 (94.89%), MRPC (90.52%), Key (90.86%), Query (92.79%), and RTE (77.60%) datasets.

## 5 Related Work

**Related Work: Parameter-Efficient Fine-Tuning (PEFT)**
* PEFT essential for NLP due to increasing complexity of Large Language Models (LLMs)
* Enhances performance while reducing computational and memory requirements
* Effective across various NLP tasks, such as NLU, NLG
* Previous research shows PEFT techniques significantly improve LLM performance with low resources

**Prompt Tuning:**
* Add learnable parameters as virtual tokens at model input or within layers
* Recent advances refine methods for NLU and NLG
* Techniques like MixPAVE and E2VPT integrate input and value prompts to boost performance
* Significantly enhances specific NLP tasks: text classification, machine translation, dialogue generation

**Low-Rank Adaptation (LoRA):**
* Memory-efficient fine-tuning technique extensively studied
* Multitask learning potential explored by Renduchintala et al., Sheng et al., Xia et al.
* Practical applications shown by Wang et al. and Dettmers et al.
* Memory usage optimized by LoRA's developers
* ReLoRA proposed by Lialin et al., requiring full-rank warm-up
* Adaptive methods: Zhang et al. dynamically adjust low-rank parameters
* LoKr introduced by Edalati et al., ResLoRA with residual paths developed by Shi et al.
* Low-Rank Hadamard Product (LoHa) presented by Hyeon-Woo et al.
* Orthogonal Finetuning (OFT) and OFT with butterfly factorization (BOFT) introduced by Qiu et al. and Liu et al.

**Our Proposed Approach:**
* Differentiates from previous PEFT approaches: proposes a new concept of adaptive Propulsion that changes the output direction of the model by applying a force to achieve task-specific goals
* Adjust Propulsion parameter during training process to decide how much push is needed to change the direction. (More details related work in Appendix J)

## 6 Conclusion

**Conclusion**
- Fine-tuning extensive language models can be costly in terms of hardware, storage, and financial investment
- Propulsion: a parameter-efficient fine-tuning method that adds trainable Propulsion parameters to each layer while keeping the original frozen
- Goal is to achieve task-specific objectives without modifying the original LLMs' parameters
- Experiments on NLP, question answering, text summarization, common sense reasoning, and mathematical reasoning show Propulsion outperforms existing methods in: accuracy, efficiency, faster convergence, reduced training time, and lower memory usage
- Demonstrates that Propulsion outperforms current PEFT techniques while requiring fewer trainable parameters (e.g., 37x fewer than AdaLoRA)

**Limitations of Propulsion:**
- Offers limited control over the model compared to other methods like LoRA, which allows adjustments through changes in rank
- Ability to steer a model is constrained by the number of dimensions in each layer
- Each parameter works independently without influencing others, making it harder to make coordinated changes across the model
- Success depends on the quality of the pre-trained language model.

## Appendixes
### A Training Dynamics of Propulsion Explained by NTK

**Propulsion Model and Neural Tangent Kernel (NTK) Theorem**
* **Output of Propulsion Model**: Let `φP(x;θθθt)` be the output at time step t, with fixed base matrix `θθθ0` and updated diagonal matrix `Zt`.
* **Output of Fully Fine-Tuned Model**: Let `φF(x;θθθt)` be the output at time step t of the fully fine-tuned model.
* **NTK Regime**: Under large network width, NTK for Propulsion approximates that of full fine-tuning with high probability.
* **Approximation Error**: Can be bounded using Johnson-Lindenstrauss Lemma for any `ε>0` and constant `c`. With high probability:
  * Pr(θθθ0xi)⊤(θθθ0xj)−xi⊤xj ≥ i −4exp (−ε2−ε3)d 4
* **Definition of NTK Kernel**: Inner product of gradients of model outputs with respect to parameters `θθθ`.
* **Kernel Behavior**: Properties of the Propulsion model that satisfy linearization and fixed features.
* **Linearization Property**: Change in model's output approximated by first-order Taylor expansion.
* **Fixed Features Property**: Gradient of model at time t is approximately the same as initialization (θθθ0).
* **Proof**: NTK change expressed as a first-order Taylor expansion; Propulsion model outputs written in terms of pre-trained weights and diagonal matrix Z.

### B Empirical Validation of NTK Approximation

**Empirical Validation of NTK Approximation**
- Presented results comparing NTK matrices for full fine-tuning and Propulsion fine-tuning across four datasets: SST-2, RTE, CoLA, and STSB
- Results show high accuracy approximation of NTK for Propulsion method compared to full fine-tuning for all datasets (Figure 6)
- Two types of NTK matrices computed: KF(x,x′) from fully fine-tuned models and KP(θθθ0x,θθθ0x′) from Propulsion method
- Absolute distance between the two NTK matrices (|KF(x,x′)-KP(θθθ0x,θθθ0x′)|) measures discrepancies
- Heatmaps demonstrate that NTK matrices for Propulsion fine-tuning closely resemble those of full fine-tuning across all datasets
- Minimal differences between the two methods support theoretical findings on Propulsion approximating full fine-tuning in the NTK regime with high probability. Significant for efficient training while maintaining comparable performance.

### C Kernel Behavior in the NTK Regime

**Neural Network Training Dynamics in the Neural Tangent Kernel (NTK) Regime**

**Validation of Kernel Behavior in NTK**:
- As width of neural network tends to infinity, gradient of network output with respect to parameters stabilizes
- Network exhibits linear behavior in parameter space
- Crucial for understanding training dynamics, especially fine-tuning scenarios like Propulsion

**Evaluating Kernel Behavior**:
- Measure stability of gradients using **Jacobian matrix** of network output with respect to initial parameters (θ0) and after training steps (t)
- Compute absolute difference between two Jacobian matrices: ∇**θθθtφ(x)** - ∇**θθθ0φ(x)**

**Heatmap Results**:
- Demonstrate stability of **Jacobian matrices** across SST-2, RTE, CoLA, and STSB datasets
- Initial and trained Jacobian matrices remain relatively close, indicating stable gradients
- Confirms network remains in NTK regime, with output becoming a function of NTK matrix

**Implications for Fine-tuning**:
- Stability of gradients ensures efficient fine-tuning without large deviations from pre-trained parameters

### D Comparison of Delta Weight Reparameterization in PEFT Methods

**Comparison of Delta Weight Reparameterization in PEFT Methods**

**Table 6 Comparison**:
- Various PEFT methods based on delta weight matrix ∆W reparameterization comparison
- Each method uses different strategies for adjusting weight updates during fine-tuning: optimizing parameter efficiency while maintaining performance.

**LoRA (Low-Rank Approximation)**:
- **∆W Reparameterization**: Wdown and Wup matrices, r much smaller than k or d.
- **KronA**:
  - **∆W Reparameterization**: Wdown⊗Wup (Kronecker product)
  - Rank(Wdown⊗Wup) = rank(Wdown)×rank(Wup).

**DyLoRA (Dynamic LoRA)**:
- **∆W Reparameterization**: Wdown,jµjWup,b; Wdown,j,b=Wdown,b and Wup,b=Wup,b (for b in {rmin,..., rmax}).

**AdaLoRA (Adaptive LoRA)**:
- **∆W Reparameterization**: PAQ PP⊤=P⊤P̸=I=QQ⊤=Q⊤Q, Λ=diag(σ1,σ2,...,σr).

**IncreLoRA (Incremental LoRA)**:
- **∆W Reparameterization**: WdownλWup; λ is an arbitrary constant.

**DeltaLoRA**:
- **∆W Reparameterization**: WdownWup; updates at time t+1 = W(t)+ΔW(t).

**LoRAPrune**:
- **∆W Reparameterization**: WdownWup⊙M (element-wise product with mask M); δ=(W⊙WdownWup)∈M, M∈ {0,1}d×r.

**QLoRA (Quantized LoRA)**:
- **∆W Reparameterization**: WupWB16BdqWp; B is a quantization function.

**QA-LoRA (Quantized Adaptive LoRA)**:
- **∆W Reparameterization**: WdownWup; Wdown∈Rd×r, Wup∈Rr×L (quantization group number of W).

**LoFTQ (Few-Shot Transfer Quantization)**:
- **∆W Reparameterization**: SV D(W−Qt); Qt=arg(W−WtWup, WFP4WFP8), G1 is N-bit quantization function.

**Kernel-mix LoRA**:
- **∆Wh Reparameterization**: (BA LoRA BA h)ΛA R ΛR; BA h provides rank-r update in each head, BLoRA shared across all heads.

**LoRA-FA (Frozen Adaptive LoRA)**:
- **∆W Reparameterization**: Wdown is frozen, only Wup updated.

**Propulsion**:
- **∆W Reparameterization**: W⊙Z; Wis frozen, only Z updated.

**Comparison Notes**:
- Each method uses different strategies for adjusting weight updates during fine-tuning to optimize parameter efficiency while maintaining performance.
- LoRA and KronA employ low-rank decompositions, while Propulsion uses element-wise updates with frozen base weights W and only updating task-specific matrix Z.

### E Training

**Propulsion Method Training Algorithm**

**Input**: x - input data
- pre-trained language model M(.) with L layers (all parameters frozen)
- Propulsion parameters Z initialized

**Epochs**:
- During each epoch:
  * Extract output Vi from layer Li
  * Update transformed output V' through multiplication with zi
    - New V' sent to subsequent layer Li+1 as input x
  * Calculate task-specific loss
  * Update Propulsion parameters Z based on loss

**Processing Steps**:
1. Modify outputs at all layers using Propulsion method
2. Freeze model parameters (except for Propulsion)
3. Calculate task-specific loss
4. Update only Propulsion parameters Z

**Objective Functions**:
- For STS-B: Mean Squared Error
- Rest of experiments: Cross-entropy loss (Equation 16)

**Cross-entropy Loss**:
L(y, ˆy) = −1ⁿ Tⁿ ∑t=1ⁿ tlog(ˆyt)
- Total number of data samples: T
- Ground truth: y
- Predicted labels: ˆy

### F More Ablation Study

**Propulsion Parameter Initialization:**
- **Importance of setting Propulsion parameters correctly**: Enhances model accuracy and efficiency
- **Methods for initializing Propulsion parameters**: Tested on SST-2 dataset, best results with initial value set to 1
- **Benefits of initializing to 1**: Ensures output of each layer in initial forward pass remains unchanged, facilitating smoother updates and adjustments to Propulsion parameters

**Propulsion Weights After Training:**
- **Variation in Propulsion weights**: Range between 0.98 and 1.02 after training
- **Initial setting of weights**: Uniformly set to 1, then adjusted for task optimization
- **Significance of small adjustments**: Can significantly impact model's ability to meet goals
- **Contribution to performance**: Propulsion parameter plays an important role in fine-tuning process

**Multi-Propulsion:**
- **Use of multiple Propulsion vectors**: Gains more control over model adjustments through pooling operation
- **Pooling operation**: Dynamically synthesizes influence of vectors into a single output matrix V′ i
- **Processing**: Processed as input for subsequent layer or adjusted based on requirements

**Number of Propulsion Layers:**
- **Evaluation on five prominent NLP benchmarks**: SST2, QNLI, MRPC, MNLI, RTE
- **Performance across varying Propulsion layer counts**: High accuracy maintained across datasets (95% for SST2 and 80% for RTE)

**Pooling Strategies:**
- **Comparison of pooling strategies**: Average, Max, Min, L2
- **Impact on model accuracy**: Figure 11 shows Average Pooling consistently achieves highest accuracy (up to 1.06% better than others), especially on SST-2 and QNLI
- **Performance on other datasets**: Minimal differences between pooling methods, with Average Pooling maintaining a slight edge in most cases.

### G Baseline Methods

**Fine-Tuning Methods for Language Models**

**Full Finetuning (FT):**
- Entails updating all pre-trained weights of a language model with task-specific data
- Enables learning intricate patterns, but requires substantial computational resources and labeled data
- Can result in overfitting when the task-specific dataset is limited or the model is already well suited for the target task

**AdapterS:**
- Involves incorporating task-specific adapter modules into a pretrained model
- Allows parameter-efficient tuning without extensive modifications to the original model's weights
- Adapters are often characterized by their low-rank properties and include a non-linear activation function for task-specific adjustments

**Prompt Tuning:**
- Involves appending trainable prompt tokens to the input of a language model
- Updates only the prompt parameters through gradient descent while leaving pretrained model's parameters frozen
- Success contingent on length and training of prompt tokens

**Prefix-Tuning:**
- An extension of prompt tuning that introduces task-specific vectors into the activations of multi-head attention layers
- Prefixes are optimized independently without modifying original pre-trained parameters
- Achieves fine-tuning efficiency and stability through a parameterized feed-forward network

**(IA)3:**
- Infused Adapter through Inhibiting and Amplifying Inner Activations approach
- Facilitates effective adaptation to mixed-task batches without altering the model's architectural structure
- Preserves efficiency and original form of the model

**Bitfit:**
- Capitalizes on selectively updating only the bias parameters of a model during fine-tuning
- Minimizes memory and computational resources required for full model training

**LoRA: (Low-Rank Adaptation)**
- Fine-tunes a model by making low-rank updates to weight matrices, enabling efficient adaptation with minimal alterations to original parameters
- Effectively combines efficiency of parameter utilization and performance in subsequent tasks

**AdaLoRA:**
- Enhances LoRA's capabilities by adaptively allocating the rank and budget of updates among different weight matrices based on their importance
- Improves both fine-tuning efficiency and task-specific performance

**MAM Adapter:**
- Integrates principles of parallel adapter and prefix-tuning into a cohesive structure for model adaptation improvement
- Refines various aspects of the model outputs by adjusting a combination of parameters across multiple layers

**ProPETL: (Hybrid Fine-Tuning Methods)**
- A set of techniques combining aspects of adapters, prefix-tuning, and LoRA to optimize performance across multiple tasks
- Leverages strengths of each technique while mitigating their weaknesses.

### H Evaluation Metric

**Evaluation Metrics for GLUE Benchmark Suite**

**CoLA Task**:
- Evaluated using Matthews correlation coefficient (MCC) as the metric
- MCC considers true and false positives/negatives, providing a balanced measure even with imbalanced datasets

**MRPC and QQP Tasks**:
- Evaluated using two metrics: accuracy and F1 score
- **Accuracy**: percentage of correctly identified paraphrase pairs
- **F1 Score**: balance between precision and recall, offering a more nuanced view of model's performance in identifying paraphrases

**MNLI Task**:
- Evaluated using the Average Matched Accuracy metric
- Measures the model's accuracy on the matched validation set (in-domain data)
- Reflects the model's ability to generalize across different genres, providing insights into its robustness and versatility

**STS-B Task**:
- Evaluated using Pearson correlation coefficient and Spearman rank correlation coefficient for measuring linear and rank correlations between predicted and actual similarity scores, respectively

**Dataset Details**:
- Various datasets are evaluated in the GLUE benchmark suite
- Includes tasks like arithmetic reasoning (Math), commonsense reasoning (CS), question answering (SQuAD, QNLI), text classification (MNLI), and more.

### I Dataset Description

| Dataset    | Train   | Validation | Test    |
|------------|---------|------------|---------|
| SQuAD v1.1 | 87.6k   | 10.6k      | -       |
| SQuAD v2.0 | 130k    | 11.9k      | -       |
| XSum       | 204k    | 11.3k      | 11.3k   |
| DailyMail  | 287k    | 13.4k      | 11.5k   |
| CoLA       | 8.55k   | 1.04k      | 1.06k   |
| SST2       | 67.3k   | 872        | 1.82k   |
| MRPC       | 3.67k   | 408        | 1.73k   |
| STS-B      | 5.75k   | 1.5k       | 1.38k   |
| QQP        | 364k    | 40.4k      | 391k    |
| MNLI       | 393k    | 9.8k       | 9.8k    |
| QNLI       | 105k    | 5.46k      | 5.46k   |
| RTE        | 2.49k   | 277        | 3k      |

This table represents the data description for various datasets, showing the number of samples in the train, validation, and test sets for each dataset.

### J Details Related Work

**Parameter-Efficient Fine-Tuning Methods for NLP+**

**Background:**
- Increasing complexity of large language models (LLMs)
- Need for methods to improve performance while reducing computational and memory requirements

**Effectiveness of PEFT Techniques:**
- Demonstrated effectiveness on various NLP tasks (Fu et al., 2023; He et al., 2021)
- Several proposed methods for reducing computational demands (Liu et al., 2021b, 2023; Zhang et al., 2023; Hu et al., 2021; Li and Liang, 2021; Zaken et al., 2021)

**Prompt Tuning:**
- Improve natural language understanding and generation tasks
- Adjust learnable parameters (Lester et al., 2021)
- Extensions: real-time transformation (Yang et al., 2023b), multi-level control (Wang et al., 2022), multimodal prompt tuning (MixPrompt, E2VPT, prefix-tuning)

**Memory-Efficient Methods:**
- Low-Rank Adaptation (LoRA) introduced by Hu et al. (2021)
- Extensions: multitask learning (Renduchintala et al., 2023; Sheng et al., 2023; Xia et al., 2024), memory optimization (Dettmers et al., 2024), ReLoRA (Lialin et al., 2023), LoKr (Edalati et al., 2022), ResLoRA (Shi et al., 2024), LoHa (Hyeon-Woo et al., 2021)

**Subspace Learning:**
- Focus on learning within lower-dimensional parameter spaces (Larsen et al., 2021; Gurari et al., 2018)
- Recent advancements: adaptive subspace learning methods (Nunez et al., 2023), integration with neural architecture search (Chen et al., 2022b)

**Optimization Algorithms:**
- Projected gradient descent (PGD) improved by GaLore method (Zhao et al., 2024; Chen and Wainwright, 2015; Chen et al., 2019)
- Recent developments: addressing sparsity and redundancy in neural network gradients (Zhao et al., 2024), memory-efficient optimization techniques (Li et al., 2024)

### K LLM Performance

**LLM Performance Evaluation:**
* Conducted comparison of various LLMs: Bloom, Llama2, Falcon, Mistral, Phi-2
* Evaluated effectiveness of different fine-tuning techniques for each model
* Traditional approaches: Finetuning, Prefix-Tuning, Prompt Tuning, PTuning, LoRA Rank 1, and LoRA Rank 2
* Propulsion methods introduced (All and Attn) consistently outperformed traditional methods across datasets
* Superior efficiency and effectiveness of Propulsion in different classification tasks: Fake News Filipino, Emotion, SST-2, Cola
* Documented performances in Tables 9, 10, 11, 12, and 13
* Propulsion(All) and Propulsion(Attn) require fewer trainable parameters compared to fine-tuning methods.

#### K.2 Token Classification

**Token Classification Comparison of Propulsion and PEFT Methods:**
* Tables 14 through 18 compare results on token classification between Propulsion and other PEFT methods
* Majority of experiments show Propulsion having higher accuracy and F1-scores compared to other PEFT methods
* Accuracy under Propulsion still less than full fine-tuning but higher among PEFT methods
* Mix in performance between two Propulsion applications:
  * Propulsion(Attn) provided better results on four out of five LLMs for Conll103 dataset
  * Propulsion(All) had superior accuracy and F1-scores on WikiAnn dataset
* Regardless of dataset, Propulsion application to any combination of layers showed similar or improved metrics with reduced parameter size.

**Sequence Classification Results for Bloom Model:**
* Table 9 displays sequence classification results for the Bloom model
* Best and second-best results highlighted in bold and underlined respectively
* Propulsion(All) and Propulsion(Attn) outperformed other methods on some datasets, but full fine-tuning remained superior.

**Sequence Classification Results for Llama2 Model:**
* Table 10 presents sequence classification results for the Llama2 model
* As with Bloom model, best and second-best results are highlighted in bold and underline respectively
* Propulsion(All) and Propulsion(Attn) showed competitive performance but could not surpass full fine-tuning.

##### Performance Organized by Model and Task

**Models**:
1. Falcon
2. Mistral
3. Phi-2
4. Bloom (for some tasks)
5. Llama2 (for some tasks)

**Tasks**:
1. Fake News Classification (FilipinoFull dataset)
2. Emotion Classification (EmotionFull dataset)
3. Semantic Similarity (SST2Full dataset)
4. Named Entity Recognition (ColaFull dataset)
5. Token Classification (conll03Full, NCBI disease, and WikiAnn datasets)

**Fine-tuning Techniques**:
1. Full Fine-tuning
2. Prefix-Tuning
3. Prompt Tuning
4. P-Tuning
5. LoRA (Rank 1 and Rank 2)
6. Propulsion (All and Attn)

**Key Findings**

1. Full fine-tuning consistently achieves the highest accuracy and F1-scores across all tasks and models, but it requires using 100% of the parameters.
2. Among the parameter-efficient techniques:
   - LoRA (especially Rank 2) often performs well across different tasks and models.
   - Propulsion methods (All and Attn) frequently achieve competitive results while using very few parameters.
3. Performance varies significantly across tasks and models, with some techniques working better for certain combinations.
4. The Mistral model generally outperforms other models across most tasks.
5. Token Classification tasks (especially for conll03 and WikiAnn datasets) seem to be more challenging, with lower F1-scores compared to other tasks.

**Detailed Performance by Task**
1. **Fake News Classification**
   - Falcon: LoRA Rank 2 (F1: 89.44%)
   - Mistral: Propulsion(All) (F1: 91.96%)
   - Phi-2: Propulsion(All) (F1: 88.73%)
2. **Emotion Classification**
   - Falcon: LoRA Rank 2 (F1: 86.13%)
   - Mistral: Propulsion(Attn) (F1: 84.99%)
   - Phi-2: Propulsion(Attn) (F1: 83.24%)
3. **Semantic Similarity**
   - Falcon: Propulsion(Attn) (F1: 95.18%)
   - Mistral: Propulsion(Attn) (F1: 97.25%)
   - Phi-2: Propulsion(All) (F1: 96.75%)
4. **Named Entity Recognition**
   - Falcon: LoRA Rank 2 / Propulsion(Attn) (F1: 85.33%)
   - Mistral: Propulsion(All) (F1: 86.32%)
   - Phi-2: Propulsion(Attn) (F1: 82.74%)
5. **Token Classification** (Best results for conll03 dataset)
   - Bloom: Propulsion(Attn) (F1: 71.70%)
   - Llama2: Propulsion(All) (F1: 70.93%)
   - Falcon: Propulsion(Attn) (F1: 72.08%)
   - Mistral: Propulsion(All) (F1: 72.80%)
   - Phi-2: Propulsion(Attn) (F1: 71.88%)

#### K.3 Entailment Detection

**Entailment Detection Results**

**Tables Presented**: Tables 19-23 provide results of entailment detection using various models: Bloom, Llama2, Falcon, Mistral, and Phi-2.

**Full Fine-tuning vs. Other Techniques**: Across all three datasets (RTE, MRPC, SNLI), full fine-tuning consistently achieves the highest accuracy and F1-score. Bloom and Mistral models demonstrate remarkable results. In contrast, Propulsion(All) and Propulsion(Attn) techniques yield significantly lower accuracy and F1-scores. This suggests that limiting parameter updates to specific Propulsion methods may not be sufficient for optimal entailment classification performance.

**LoRA Rank 1 and LoRA Rank 2 Models**: Deliver competitive results, particularly in the RTE dataset where they outperform other techniques. Techniques like LoRA Rank strike a balance between model adaptation and computational efficiency. However, Propulsion consistently performs well across datasets, demonstrating its effectiveness as an alternative fine-tuning strategy.

**Computational Efficiency**: Propulsion achieves strong results with minimal increase in the number of parameters, making it a promising approach for entailment classification tasks where computational resources are a concern.

### L Variable Description: Sequence Classification

**Variables and Descriptions**

**Pre-trained language model with frozen parameters (M(.))**:
- Pre-trained language model
- Parameters are frozen

**N (Number of layers)**:
- Number of layers in the model

**Li(x)**: Output of the **i-th layer** given input **x**

**x**: Input representation
- Sequence length of tokens (**s**)
- Dimension of each token (**d**)

**V (Output of layer Li)**:
- Output of layer **Li**

**Z (Trainable Propulsion matrix zi)**:
- Element-wise transformation vector

**Transformed output after Propulsion (Zi)**:
- Multiplied with Propulsion matrix (**z**) to form new output

**Propagation degree for nonlinear transformation (k)**: Degree of nonlinear transformation applied during propagation

**V' (New output after Propagation and Propagation L)**: New output after propulsion and propagation layer **L**

**Cross-entropy loss function (T)**:
- Total number of data samples
- Ground truth labels (**y**)
- Predicted labels (**o**)

**Table 24: Table of Variables and Descriptions**
