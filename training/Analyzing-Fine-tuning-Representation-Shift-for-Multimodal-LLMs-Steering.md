# Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering

source: https://arxiv.org/html/2501.03012v1
by Pegah Khayatan,  Mustafa Shukor,  Jayneel Parekh,  Matthieu Cord

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Notations and Background](#3-notations-and-background)
- [4 Fine-tuning and evolution of concept representations](#4-fine-tuning-and-evolution-of-concept-representations)
  - [4.1 Impact of fine-tuning on learned concepts](#41-impact-of-fine-tuning-on-learned-concepts)
  - [4.2 Recovering concepts in fine-tuned model via shift vectors](#42-recovering-concepts-in-fine-tuned-model-via-shift-vectors)
- [5 Fine-grained multimodal LLM steering](#5-fine-grained-multimodal-llm-steering)
  - [5.1 Multimodal LLMs steering framework.](#51-multimodal-llms-steering-framework)
  - [5.2 Coarse-grained model steering](#52-coarse-grained-model-steering)
  - [5.3 Discovering meaningful and fine-grained steering directions.](#53-discovering-meaningful-and-fine-grained-steering-directions)
  - [5.4 Steering towards a specific target answer](#54-steering-towards-a-specific-target-answer)
  - [5.5 Steering image captions](#55-steering-image-captions)
- [6 Discussion](#6-discussion)
- [Appendix A Fine-tuning and evolution of concept representations](#appendix-a-fine-tuning-and-evolution-of-concept-representations)
  - [A.1 Notations](#a1-notations)
  - [A.2 Implementation details](#a2-implementation-details)
  - [A.3 Concepts change during training](#a3-concepts-change-during-training)
  - [A.4 Concepts recovery and shift vectors](#a4-concepts-recovery-and-shift-vectors)
  - [A.5 Concepts shift consistency and magnitude](#a5-concepts-shift-consistency-and-magnitude)
  - [A.6 Additional experiments with LLaVA model](#a6-additional-experiments-with-llava-model)
- [Appendix B Fine-grained multimodal LLM steering](#appendix-b-fine-grained-multimodal-llm-steering)
  - [B.1 Implementation details](#b1-implementation-details)
  - [B.2 Steering other MLLMs](#b2-steering-other-mllms)
  - [B.3 Discovering meaningful steering directions.](#b3-discovering-meaningful-steering-directions)
  - [B.4 Steering image captions.](#b4-steering-image-captions)
  - [B.5 Ablation study](#b5-ablation-study)
  - [B.6 Linear separability of concepts inside MLLMs.](#b6-linear-separability-of-concepts-inside-mllms)
  - [B.7 Qualitative results](#b7-qualitative-results)

## Abstract

**Multimodal Language Models (LLMs)**
- High proficiency in understanding multimodal inputs
- Extensive research for developing more powerful models
- Less attention on underlying mechanisms

**Explainability Research**:
- Examines models only in final states
- Overlooks dynamic representational shifts during training

**Current Findings**:
- Analyze hidden state representations to reveal how fine-tuning alters model structure
- Map hidden states to interpretable visual and textual concepts
- Trace changes in encoded concepts across modalities as training progresses
- Demonstrate use of shift vectors for capturing concept changes
- Recover fine-tuned concepts by shifting those in original model

**Impact on Model Steering**:
- Adjust multimodal LLMs behaviors without any training:
  - Modify answer types, caption styles, or biasing responses

**Significance**:
- Sheds light on how multimodal representations evolve through fine-tuning
- Offers a new perspective for interpreting model adaptation in multimodal tasks

**Project Code**:
- Available at [https://github.com/mshukor/xl-vlms](https://github.com/mshukor/xl-vlms/tree/main)

## 1 Introduction

**Overview:**
- Analysis of concept representations change during fine-tuning of Multimodal Large Language Models (MLLMs)
- Importance for steering MLLMs without additional training
- Rapid progress in MLLMs and their capabilities in multimodal tasks like image captioning, visual question answering
- Composition of visual encoder, LLM, and intermediary connector
- Pretraining and fine-tuning on multimodal datasets for specialization
- Efficient approaches: diverse instruction-tuning datasets or fine-tuning model parameters (connector) while keeping LLM frozen
- High computational cost and need to understand MLLMs

**Background:**
- Existing research overlooks internal changes during fine-tuning
- Study by [^45] examines multimodal alignment evolution during training
- Focus on exploring semantic representations change in MLLMs for multimodal tasks (Figure 1)

**Findings:**
- Concepts adapt specifically to the fine-tuning task: some becoming more specialized, others transformed completely
- Many concepts encoded in fine-tuned models can be reconstructed from original model using shift vectors in latent space
- Possibility to steer MLLMs outputs without additional training by applying findings (*e.g.,* LLaVA)

**Implications:**
- Summarized: learned concepts adapt specifically during fine-tuning, many can be reconstructed using shift vectors, and applications for steering MLLMs.

## 2 Related Work

**Concept-based Explainability Methods vs Traditional Feature Attribution**
* Concept-based explainability methods: alternative to traditional feature attribution methods
* Capable of extracting key semantic features from model internal representations

**Recent Post-hoc Concept Explainability Approaches**
* Based on the idea of **concept activation vectors (CAV)**
* Represent concepts as vectors in the activation space
* Recent works proposed methods for automatic concept discovery via:
  - Clustering
  - Matrix decomposition
  - Viewed as instances of a dictionary learning problem

**Application to Multimodal Language Models (LLMs)**
* Extensive research on understanding and explaining behavior of LLMs
* Studies focus on identifying multimodal neurons or analyzing modality-specific subnetworks
* Textual explanations for model outputs generated by text-generative nature of LLMs
* Research on limitations, biases, and factors enhancing in-context learning performance
* CoX-LMMs employ dictionary learning to extract multimodal semantic concepts from model representations

**Steering Models with Feature Editing vs Representation/Feature Editing**
* Representation or feature editing methods aim to modify model outputs without altering the model's weights
* Identifying steering vectors in feature space for contrasting concepts: enhancing factuality, reducing hallucinations, inducing sentiment shifts, etc.
* Primarily applied to language models; application to multimodal Language Models (LLMs) yet to be explored.

## 3 Notations and Background

**Model Architecture**
* A generic Multimodal Language Model (MLLM) consists of:
  * Visual encoder f_V
  * Trainable connector C
  * Language model f_LM
* Pretrained on multimodal dataset 𝒮={(x\_i,y\_i)}\_i with images x\_i and associated captions y\_i
* Model generates next text tokens conditioned on image and previous tokens
* Input to f_LM includes:
  * Sequence of visual tokens extracted from image via f_V and connector C(f\_V(x))
  * Linearly embedded textual tokens
* Output token derived by normalizing last layer tokens, applying unembedding layer, and softmax operation
* Model keeps predicting next token until end of sentence to obtain generated response

**Concept-based Explainability**
* Leverage approach introduced in [^37] for analyzing MLLM's internal representations
* Extract residual stream representations after l-th layer, h\_(l)\_p=f\_l(x), from a subset X\_TOI of samples containing Token of Interest (TOI)
* Feature matrix ZinR^DxM is obtained by combining extracted hidden states, and decomposed as Z≈UV to discover encoded concepts
  * UinR^DxK: matrix of K concepts
  * VinR^KxM: coefficients/activations of samples projected onto these concepts
* Each representation can be projected on U to obtain its activation vector v(x) ∈ R^K
* Concepts are grounded in both image and text spaces
  * Top N\_MAS most activating samples for concept u\_k represent its image grounding X\_MAS(u\_k)
  * Text grounding decodes features using the unembedding matrix of language model W\_U, extracting top N\_grounding words with highest logits
* K-Means used to learn concept dictionaries for simplicity and arithmetic manipulation

**Fine-tuning Setup**
* Original model f^a is fine-tuned on samples including target concepts {w\_1,…,w\_m} using Low-Rank Adaptation (LoRA) approach to develop specialized model f^b
* After fine-tuning, extract feature matrices AinR^DxM^a and BinR^DxM^b from f^a and f^b respectively
  * Contain residual stream representations of samples where TOI is present in both ground truth and response for both models
* Matrices A and B are decomposed to extract concepts each encodes: A≈U^aV^a, B≈U^bV^b
  * U^a/b with V^a/b represent the learned concepts {u\_k}^K and their activations in R^DxM
* Concepts are grounded in image modality (X\_k,MAS^a and X\_k,MAS^b) and text modality (T\_words^a and T\_words^b)

**Concept Similarity**
* Text Grounding Overlap between two concepts u and u' is defined as: 100x|T\_words(u)∩T\_words(u')|/|T\_words(u)|.

## 4 Fine-tuning and evolution of concept representations

**Studying Fine-Tuning Process Effects on Concepts Learned by Models**

**Goals:**
- Study how fine-tuning affects concepts learned by models
- Recover fine-tuned model concepts using shift vectors in feature space

**Figure 3 Illustration:**
- Comparison of groundings (concepts) from original and fine-tuned models: f^a (top) and f^b (bottom)
- Concepts extracted from textperson (TOI=text) for places in f^a and f^b
- Notable difference: Stronger association with places in concepts from f^b

**Implementation Details:**
- Focus on Multimodal Language Model (MLLM) architecture described in [Section 3](https://arxiv.org/html/2501.03012v1#S3 "3 Notations and Background ‣ Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering")
- **Image Encoder**: ViT-L/14 CLIP [^40]
- **Transformer-based Connector**: Reduces encoded visual tokens number
- **LLM f_LM**: OPT-6.7B model [^59] with 32 layers
- Controlled setup: Fine-tuning on three subsets of Visual Genome dataset [^24]: places, colors, and sentiments (see App. A for more details)

### 4.1 Impact of fine-tuning on learned concepts

**Fine-Tuning Process:**
* Introduces changes in overall structure of learned concepts
* Observable through concept grounding in both text and image space (Figure 4)

**Matched Concepts:**
* Focus on each concept and its match based on cosine similarity
* Matching function m:i→j^* associates concept vectors u^a_i to closest vector u^b_j in fine-tuned model U^b

**Concept Evolution:**
* Quantify change using T-Overlap between original and matched fine-tuned model concepts (Figure 5)
* Varying rates of change across different concepts and fine-tunings
* Two main behaviors: refined concepts and completely changed concepts

**Refined Concepts:**
* Slightly change to be more specialized towards the fine-tuning task
* Exhibit relatively high T-Overlap(u^a_i,u^b_m(i))

**Changed Concepts:**
* Emerge or disappear in the fine-tuned model
* Likely due to introduction of novel patterns or relationships not present in original model
* Exhibit relatively low T-Overlap(u^a_i,u^b_m(i))

**Impact on Original Model Concepts:**
* Decrease in T-Overlap as fine-tuning iterations increase (Figure 6)
* Indicates deviation from original model concepts.

### 4.2 Recovering concepts in fine-tuned model via shift vectors

**Concept Shift Vectors and Fine-Tuning Concept Recovery**

**Characterizing concept shift in fine-tuning:**
- Associate each original model concept u^a_k with a subset of common samples, A_k
- Compute change in representation from f^a to f^b for each sample in A_k: δ^a→b_m = b_m - a_m
- Aggregate shifts to obtain concept shift vector Δ_k^a→b(u^a_k): Δ_k^a→b(u^a_k) = 1/|A_k|∑δ^a→b_m
- Use α coefficient (default: 1) to control magnitude of shift

**Evaluating fine-tuned concept recovery:**
- Match original and fine-tuned concepts using a bijective mapping m:i→j
- Evaluate similarity between shifted concept u^s_k and matched fine-tuned concept u^b_m(k) using overlap metrics

**Shift magnitude (α) and concepts recovery:**
- Study the effect of different shift magnitudes on recovered concepts
- Report average recovery for K=20 concepts per fine-tuning task at various α values
- α=0 corresponds to original concepts; optimal value is often 1 or close to it

**Correlation between concept shift vectors and recovery:**
- Examine potential correlation between computed concept shift vectors and fine-tuned model concept recovery
- Hypothesize that consistent shifts towards fine-tuned models lead to effective recovery of original concepts
- Observe noticeable correlation between magnitude of concept shift vectors ||Δ_k^a→b|| and cosine similarity between shifted and matched fine-tuned concepts (CR) in Figure 9.

**Steering MLLMs Answers:**
- Overview: Add a steering vector to residual stream of MLLMs to change its response
- Each example illustrates different steering vectors that alter specific original answers to target ones (Figure 10).

## 5 Fine-grained multimodal LLM steering

**Model Steering: An Alternative Method for Guiding Model Outputs**
* **Concept Recovery**: shifting features of fine-tuned models towards desired outcomes without altering weights ([Fig. 10](https://arxiv.org/html/2501.0304v1#S4.F10))
* **Motivation**
	+ Feasibility of recovering target concepts in fine-tuned models through shift vectors (previous sections)
	+ Linear separability of features related to different concepts, becoming more prominent in deeper layers ([Fig. 11](https://arxiv.org/html/2501.0304v1#S5.F11))
	+ Potential cost savings by avoiding fine-tuning processes
* **Multimodal Large Models**: focus on visual question answering and image captioning tasks (additional qualitative results, ablation study in App. B)
* **Linear Representation Hypothesis**: validated for MLLMs following previous studies on LLMs [^38]

### 5.1 Multimodal LLMs steering framework.

**Multimodal Model Steering**

**Approach**:
- Modify model's output to desired output by applying a shift or steering vector to residual stream features
- No change in model parameters

**Evaluation**:
- Within visual question-answering (VQA) framework
- Measure effectiveness of approach: number of generated answers aligning with target output/answer type
- Targeted steering: ensure only specific answer types are influenced

**Implementation Details**:
- Experiments on LLaVA model: CLIP image encoder, two-layer MLP connector, 7B Vicuna-1.5 LLM
- Focus on VQAv2 dataset for evaluation
- Steering vectors derived from training set, evaluated on validation set
- Effectiveness increases in deeper layers (see Figure 11 and Appendix B)
- Apply steering on last layer for VQAv2

**Table 1: Steering MLLMs answers type**:
| Answer Type | Number of Target Answers Increases Significantly After Steering Model with Corresponding Vector |
|---|---|

### 5.2 Coarse-grained model steering

**Model Steering Techniques**
* **Coarse-grained Steering**: Adjust model outputs to align with target samples (changing answers type)
	+ Extract representations B from target samples, A from original samples
	+ Compute coarse steering vector s_c: s_c = (1/N * ∑B_i) - (1/M * ∑A_i)
	+ Apply s_c to all validation set samples
* **Experiments**: Steer model answers towards specific types (yes/no, numbers, other)
	+ Computes steering vector for each target answer type
	+ Number of answer types increases when applying corresponding steering vector, validating efficacy
	+ Figure 12 shows steering directions and their impact on accuracy.

### 5.3 Discovering meaningful and fine-grained steering directions.

**Fine-grained Model Steering: Adjusting Concepts**

**Focus on concept level adjustments:**
- Fine-grained steering focuses on specific concepts U
- Decompose hidden states into set of concepts
- Calculate steering vectors between concepts: s_ij^f = u_j - u_i

**Identifying Meaningful Steering Vectors:**
- Not all computed vectors are effective for steering
- Identify based on impact on guiding model towards specific answers or concepts

**Experiments and Results:**
- [Fig. 12](https://arxiv.org/html/2501.03012v1#S5.F12) illustrates steering vectors between concepts from "yes/no", "number" and "other" answer sets
- Validation: finding vectors corresponding to answering "No", "4" and "Black"
- Significant change in number of original/target answers for some types while accuracy on others remains almost constant.

**Table 2:**
| Answer Type | Original Answers | Target Answers | Change in Counts |
|---|---|---|---|
| Yes (Yes/No) | X | No | Decrease |
| Number | X | 1, 3, ... | Increase |
| Other | X | White, Black, Red... | Increase |

### 5.4 Steering towards a specific target answer

**Steering Model Towards Specific Answers:**
- **Objective**: Steer the model towards a specific answer specified by the user
- **Data Collection**:
  - Collect few hundred samples for each pair of original/target answers (*e.g.*, yes/no)
  - Compute steering vectors as in [Section 5.2](https://arxiv.org/html/2501.03012v1#S5.SS2 "5.2 Coarse-grained model steering ‣ 5 Fine-grained multimodal LLM steering ‣ Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering")
- **Application**: Apply vectors on all samples in the validation set

**Evaluation Metrics:**
- Reported when targeting the last layer
- Effectiveness shown as number of target answers increase
- Accuracy on other answer types only slightly changes ([Table 2](https://arxiv.org/html/2501.03012v1#S5.T2 "In Experiments. ‣ 5.3 Discovering meaningful and fine-grained steering directions. ‣ 5 Fine-grained multimodal LLM steering ‣ Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering"))

**Figure 13:**
- Demonstrates steering of MLLMs captions style: colors (left), places (middle), sentiments (right) (*Refer to caption*)

### 5.5 Steering image captions

**Steering Longer Descriptive Outputs using COCO Dataset:**
- Extending earlier experiments on VQAv2 dataset to longer outputs from COCO image captioning dataset [^29]
- Multiple captions describe an image, focusing on various aspects like main object, surroundings, actions or events [^30]
- Goal: Modifying captions to align with a specific target style (Figure 13)

**Steering MLLMs Captions Style:**
- Table 3 presents results of steering MLLMs towards different captions styles
- Each line corresponds to a distinct steering vector [^31]
- Steering towards a target style increases the number of captions with that style.

## 6 Discussion

**Limitations:**
- Ideally, steering should only affect initially answered "yes" questions
- Tradeoff between steer strength and quality of responses
- Focuses on changing model answers or style
- Extensions to addressing biases, safety concerns, larger models, different architectures

**Conclusion:**
- Understanding recent foundation models' mechanisms is crucial for AI research
- Demonstrated that post-fine-tuning concepts can be recovered from original models
- Steered model behavior by modifying features without additional training
- Changed model answers or focused image captions on different aspects
- Motivates further research into understanding these models and efficient methods for adapting/refining them.

## Appendix A Fine-tuning and evolution of concept representations
### A.1 Notations

**Residual Stream View Details:**
- h_(l+1)^p = h_(l)^p + a_(l)^p + m_(l)^p: representation at residual stream [^12]
- a_(l)^p computed using attention mechanism and h_(l)^1,…,h_(l)^p
- m_(l)^p output of MLP block on h_(l)^p + a_(l)^p

**Bijective Matching:**
- Cosine similarity between U^a and U^b: SinR^KxK
  * S_ij = u^a_i·u^b_j / (u^a_iu^b _j)
- Optimal transport approach for association optimization
  * Transport plan γR^KxK
  * Minimize cost: sum(γ_ij * (1 - S_ij))
  * Constraints: γ1 = 1, γT1 = 1, γ_ij in {0, 1}
  * Indicates matching state of concepts u^a_i and u^b_j.

### A.2 Implementation details

**Multi-task vs Single-task Tuning:**
* Two setups: single-task (mostly covered in main paper) and multi-task (appendix)

**Multi-task Setup:**
- Architecture: CLIP image encoder, two-layer MLP connector, 7B Vicuna-1.5 LLM

**Single-task Setup:**
- Following setup from [^37]
- Fine-tuning with Low-Rank Adaptation (LoRA) [^19]
  * Modifies weight matrices of the model with a low-rank update
  * Best learning rate and LoRA rank for each fine-tuning dataset

**Models Fine-tuned:**
- Three distinct subsets of Visual Genome (VG) dataset: color, sentiment, place
  * Color subset: about 21k samples describing colors
  * Sentiment subset: about 5k samples containing sentiments
  * Place subset: about 27k samples that describe locations or environments
- Curated based on keyword occurrences in [Fig. 15](https://arxiv.org/html/2501.03012v1#A1.F15 "In A.2 Implementation details ‣ Appendix A Fine-tuning and evolution of concept representations ‣ Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering")

**COCO Captioning Dataset:**
- Used for hidden states extraction in quantitative experiments
- Contains captions describing the image generally, often focusing on central object (not specifically used to curate VG subsets)

**Concept Changes during Training:**
- Figure 14 illustrates similarity between original concepts and fine-tuned concepts: top for individual concepts, bottom for average concepts.

### A.3 Concepts change during training

**Fine-Tuning Deviations:**
* Comparison of fine-tuned concepts vs. original ones: cosine similarity and text grounding overlap (T-Overlap) analysis
* Cosine similarity, text overlap decrease throughout training epochs in Fig. 14 [Refer to caption](https://arxiv.org/html/2501.0306v1/x25.png)
* Consistent decreasing trend indicates systematic deviation from original concepts as training progresses
* Per-concept plot reveals varying magnitudes of change, with impact on different concepts:
	+ Concepts 0 and 10 (hot dogs) exhibit smaller drift compared to others
* Fine-tuning process affects each concept differently
* Figure 16: Mean intersection between text grounding of shifted concepts and fine-tuning datasets' keywords. [Refer to caption](https://arxiv.org/html/2501.03012v1/x25.png)
* Alpha=0 corresponds to original model in Fig. 16.

### A.4 Concepts recovery and shift vectors

**A.4.1 Fine-tuning Task and Concepts Specialization**
* Relationship between shift vectors effect and fine-tuning task
* Vary α, report average overlap between:
  * Text grounding of shifted concept u^s_k
  * Set of keywords {w_1,…,w_m} associated with fine-tuning task
* [Figure 16](https://arxiv.org/html/2501.03012v1#A1.F16) shows shift vectors move original clusters closer to task keywords

**A.4.2 Ablation Studies on Design Choices**

**Number of Concepts and Recovery**:
* Varying the number K of concepts
* Report T-Overlap between fine-tuned model concepts and their matches (bijective) in shifted u^s_k and original u^a_k
* [Figure 17](https://arxiv.org/html/2501.03012v1#A1.F17) shows number of concepts does not significantly influence concept recovery

**Concepts Recovery Across Layers**:
* Vary layer from which concepts are extracted
* Report average and maximum T-Overlap:
  * Shifted u^s_k vs original u^a_k concepts
* [Figure 18](https://arxiv.org/html/2501.03012v1#A1.F18) shows gap between T-Overlap of shifted and original concepts is higher in deeper layers, indicating better recovery.

### A.5 Concepts shift consistency and magnitude

**Concept Shift Vector Computation**
* Concept shifts computed from set of individual sample representation shifts {δ_m}_minA_k
* Hypothesis: Consistent individual shifts correspond to higher concept shift vector magnitudes, linked to recovery of concepts
* Quantify consistency through mean cosine similarity between each vector δ_m and concept shift vector Δ_k^a→b(u^a_k)
* Corroborated by plotting variation of concept shift magnitudes with consistency for animal dog concepts [Fig. 19]
* Average individual shifts remain similar across concepts: animal dog (48.06±3.887, 57.6±1.796, 47.9±3.96)
* Higher consistency indicates larger mean shift vector magnitude and aligned shifts that do not cancel each other out.

**Concept Shift Vector Computation: LLaVA Model**
* Additional experiments with LLaVA model [Fig. 21]

### A.6 Additional experiments with LLaVA model

**Experiments with LLaVA Model**
- Extending experiments to LLaVA model following same setup as main experiments
- Reporting results of text grounding recovery evaluation: [Fig. 20](https://arxiv.org/html/2501.03012v1#A1.F20) and relationship between shift metrics and recovery in [Fig. 21](https://arxiv.org/html/2501.03012v1#A1.F21)

**Recovering Fine-Tuned LLaVA Concepts**
- For each fine-tuned concept: text grounding overlap with original and shifted concepts
- Model fine-tuned on: places (top), colors (middle), sentiments (bottom) - [Figure 20](https://arxiv.org/html/2501.03012v1#A1.F20)

**Analyzing Fine-Tuning Representation Shift for Multimodal LLMs**
- Steering: further insights into consistency and generalizability of findings across different models.

## Appendix B Fine-grained multimodal LLM steering
### B.1 Implementation details

**Experiments on LLaVA Model:**
- Conducted on LLaVA model: CLIP image encoder, two-layer MLP connector, 7B Vicuna-1.5 LLM
- Focus on VQAv2 dataset for evaluation
- Provide experiments on COCO captioning with automatic annotation for styles

**VQAv2 Dataset:**
- Visual question-answering corpus with image-question-answer triplets and annotated answer types: yes/no, number, other
- Evaluation metrics: accuracy, CIDEr
- Steering vectors derived from a subset of training set, model performance evaluated on validation set
- Ablation study to determine best layer for steering (last layer for VQAv2, 20th layer for COCO)

**COCO Captioning:**
- Contains images and captions describing them
- No style annotations; automatically annotated based on descriptive keywords in captions
- Evaluation metrics: accuracy, CIDEr

**Steering Vectors:**
- Derived from a subset of training set
- Few hundred examples used as they have minimal impact on final results
- Ablation study to determine which layer to apply steering and select the best one

**Evaluation Metrics:**
- Reported for both VQAv2 and COCO: accuracy, CIDEr
- 5k random samples for VQAv2, 3k random samples for COCO.

### B.2 Steering other MLLMs

Table 4 shows steering results for MLLMs, where answers change significantly from "Yes"/"White", to "No"/"Black". Accuracy remains mostly unchanged, while counts decrease or increase slightly at layers last (LLaVA-1.5), 23 (Qwen2-VL), and 25 (Idefics2).

### B.3 Discovering meaningful steering directions.

**Steering Vectors Selection Metric**

**Selection Process**:
- For each steering vector in a set, apply it to steer the model's behavior
- Measure the change in answers' occurrence between steered model and original model
- Keep top N answers with highest relative occurrence counts for each vector
- Use k-means (k=2) to cluster the top N answers
- Assign each answer to one of the two clusters
- Primary answers: those belonging to the cluster with highest total occurrences, considered target answers for the steering vector
- Calculate difference in relative occurrence between primary and secondary cluster answers
- Select steering directions with highest differences in relative occurrence between clusters as selection score
- Use clustering to accommodate steering multiple concepts at a time

**Steering Directions Towards Single Concept**:
- Illustrate some vectors that steer the model towards very specific answers (e.g., No, Red, 4)

**Steering Directions Towards Multiple Concepts**:
- Some vectors steer the model towards more than one answer (e.g., 3 and 4 or Yellow and Orange)

**Discovering Meanful Steering Directions**:
- Figures show relative increase in number of words counts for different fine-grained steering directions: Figure 25 for single concept and Figure 26 for multiple concepts

### B.4 Steering image captions.

**Concept Extraction from Image Captions**
- **Extract concepts** from a set of image captions
- Compute **steering vectors** between each pair of concepts:
    - Illustrated in [Fig. 25](https://arxiv.org/html/2501.03012v1#A2.F25)

**Analysis of Steering Vectors**
- Some steering vectors are related to **specific concepts**: e.g., "holding", "black"
- Other vectors correspond to **multiple concepts**: e.g., "standing" and "sitting"; "young", "group", "large"

### B.5 Ablation study

**Ablation Study Findings on Steering Design Choices for Multimodal LLMs:**
* **Number of samples**: effective even with few samples (50), robust to increasing number of samples up to 500, data-efficient solution [Fig. 27]
* **Steering layer**: deeper layers more effective [Fig. 28]
* **Steering strength (α)**: increases effect but less targeted [Fig. 29], pushes model towards target answer or type [Figs. 30 & 31], significant impact on diversity when changing answer types [Fig. 32-34], leads to degradation in captioning quality for significant increase beyond 1 [Fig. 35]
* **Which tokens to apply steering to**: not further detailed in provided figure (Fig. 35).

### B.6 Linear separability of concepts inside MLLMs.

A simple linear operation in feature space can steer a model's output. Visualizing PCA projections of concept features extracted from different MLLM layers shows clearer separation of concepts at deeper layers. This validates the linear representation hypothesis for MLLMs and explains why steering is more effective when applied to deeper layers.

### B.7 Qualitative results

**Qualitative Results for Multimodal Language Model Large Scale (MLLM) Answers:**
* Figure 36: Steering MLLMs answers [1]
	+ Each line corresponds to different steering vector changing specific original answer to target one
	+ Top: "white" to "black"
	+ Middle: "1" to "3"
	+ Bottom: "yes" to "no"
* Figure 37: Steering MLLMs answers type [2]
	+ Each line corresponds to different steering vectors changing answers type to target one
	+ Top: yes/no (change to numbers)
	+ Middle: numbers
* Figure 38: Steering MLLMs captions type [3]
	+ Each line corresponds to different steering vectors changing caption style to target one
	+ Top: colors
	+ Middle: places
	+ Bottom: sentiments.

[1] Figure 36 shows the effect of various steering vectors on MLLM answers, resulting in changes from "white" to "black," "1" to "3," and "yes" to "no."
[2] Figure 37 illustrates how different steering vectors alter the answer type in MLLMs, shifting from yes/no responses to numbers.
[3] Figure 38 demonstrates the impact of various steering vectors on the caption style for MLLMs, modifying the content to include more colors, places, or sentiments.

