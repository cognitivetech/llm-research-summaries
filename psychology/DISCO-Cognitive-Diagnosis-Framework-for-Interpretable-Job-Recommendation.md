# DISCO: A Hierarchical Disentangled Cognitive Diagnosis Framework for Interpretable Job Recommendation

by Xiaoshan Yu, Chuan Qin (corresponding author), Qi Zhang, Chen Zhu, Haiping Ma (corresponding author)
https://arxiv.org/pdf/2410.07671v1

## Contents
- [Abstract](#abstract)
- [I introduction](#i-introduction)
- [II related work](#ii-related-work)
  - [II-A Job Recommendation](#ii-a-job-recommendation)
  - [II-B Cognitive Diagnosis](#ii-b-cognitive-diagnosis)
  - [II-C Disentangled Learning](#ii-c-disentangled-learning)
- [III Preliminaries](#iii-preliminaries)
  - [III-A Problem Formulation](#iii-a-problem-formulation)
  - [III-B Base Embedding Model](#iii-b-base-embedding-model)
- [IV methodology](#iv-methodology)
  - [IV-A Hierarchical Representation Disentangling](#iv-a-hierarchical-representation-disentangling)
  - [IV-B Level-Aware Association Modeling](#iv-b-level-aware-association-modeling)
  - [IV-C Interaction Diagnosis Module](#iv-c-interaction-diagnosis-module)
- [V experiments](#v-experiments)
  - [V-A Experiment Setups](#v-a-experiment-setups)
  - [V-B Performance Comparison](#v-b-performance-comparison)
  - [V-C Ablation Study](#v-c-ablation-study)
  - [V-D Parameter Sensitivity Analysis](#v-d-parameter-sensitivity-analysis)
  - [V-E Case Study](#v-e-case-study)
- [VI conclusion](#vi-conclusion)

## Abstract

**Job Recommendation Systems**

**Online Recruitment Platforms**:
- Create unprecedented opportunities for job seekers
- Pose significant challenge of quickly and accurately aligning jobs with skills/preferences

**Job Recommendation Models**:
- Text-matching based methods
- Behavior modeling based methods
- Realized impressive outcomes, but research on explainability is unexplored

**Proposed Framework: DISCO**
- Hierarchical Disentanglement based Cognitive diagnosis framework
- Accommodates underlying representation learning model for effective and interpretable job recommendations

**DISCO Components**:
- **Hierarchical representation disentangling module**: Mines hierarchical skill-related factors in hidden representations of job seekers and jobs
- **Level-aware association modeling**:
  - Inter-level knowledge influence module: Enhances information communication and robust representation learning between levels
  - Level-wise contrastive learning: Improves inter- and intra-level knowledge transfer
- **Interaction diagnosis module**:
  - Incorporates a neural diagnosis function for modeling multi-level recruitment interaction process
  - Introduces cognitive measurement theory

**Experimental Results**:
- Demonstrate the effectiveness and interpretability of DISCO on real-world recruitment recommendation datasets and an educational recommendation dataset

**Availability**:
- Code available at https://github.com/LabyrinthineLeo/DISCO

## I introduction

**Job Recommender Systems:**
* Online platforms like LinkedIn and Glassdoor have revolutionized job seeking process
* Need for accurate and trustworthy recommender systems to suggest positions based on job seeker's preferences and capabilities
* Recent studies focus on text-matching based methods using vast textual data from resumes and job postings for job suggestions
* Interaction behavior based approaches explore users' personalized preferences and intentions through modeling interaction behaviors between job seekers and recruiters (DPGNN)

**Challenges in Traditional Job Recommendation Systems:**
* Lack of reasons for recommendations makes it difficult for users to choose from suggestions
* Job seekers may pursue positions misaligned with their abilities or career aspirations, hindering job search success
* Need for interpretable and explainable job recommendation systems

**Proposed Hierarchical Disentanglement based Cognitive diagnosis framework (DISCO):**
* Re-examines recruitment recommendation process from a hierarchical disentanglement based cognitive diagnosis perspective
* Facilitates evaluation of diverse skill demands of various job positions and current competitive standing of candidates
* Addresses technical challenges in mapping user and job representations to specific skill dimensions, hierarchizing competency levels, and mitigating interaction bias.

**DISCO Framework:**
1. Hierarchical representation disentangling module: extracts and clarifies hierarchical skill-related factors embedded in hidden representations of job seekers and jobs.
2. Level-aware self-attention network: explores intrinsic associations between inter-level skill prototypes.
3. Noise perturbation based level-wise contrastive module: enhances robust representation learning.
4. Interaction diagnosis module with neural diagnosis function: effectively captures multi-level recruitment interaction process and incorporates cognitive measurement theory for explainability.
5. Extensive experiments demonstrate the effectiveness and interpretability of DISCO in job recommendation task on real-world datasets.

## II related work

### II-A Job Recommendation

**Online Job Recommendation:**
* Emergence as pivotal task due to potential accurate matching of job seekers with suitable positions [[18](https://arxiv.org/html/2410.07671v1#bib.bib18), [19](https://arxiv.org/html/2410.07671v1#bib.bib19), [20](https://arxiv.org/html/2410.07671v1#bib.bib20), [21](https://arxiv.org/html/2410.07671v1#bib.bib21)]
* Two main categories: text-matching based methods and interaction behavior based methods

**Text-Matching Based Methods:**
* Focus on matching textual content [[5](https://arxiv.org/html/2410.07671v1#bib.bib5), [4](https://arxiv.org/html/2410.07671v1#bib.bib4), [22](https://arxiv.org/html/2410.07671v1#bib.bib22)]
* Text-matching strategies or text enhancement techniques [[23](https://arxiv.org/html/2410.07671v1#bib.bib23), [24](https://arxiv.org/html/2410.07671v1#bib.bib24)]
* Example: APJFNN [[4](https://arxiv.org/html/2410.07671v1#bib.bib4)] uses RNN for word-level semantic representations and hierarchical ability-aware attention strategy.

**Interaction Behavior Based Methods:**
* Focus on users' personalized preferences and intentions [[9](https://arxiv.org/html/2410.07671v1#bib.bib9), [10](https://arxiv.org/html/2410.07671v1#bib.bib10)]
* Modeling interaction behaviors between job seekers and jobs (recruiters)
* Example: SHPJF [[10](https://arxiv.org/html/2410.07671v1#bib.bib10)] models users' search histories in addition to learning semantic information from text content.

**Limitations:**
* Significant shortfall in interpretable exploration of matching job seekers with job positions.

### II-B Cognitive Diagnosis

**Cognitive Diagnosis (CD)**

**Background:**
- Classical methodology for assessing ability in educational psychology [[25](https://arxiv.org/html/2410.07671v1#bib.bib25), [26](https://arxiv.org/html/2410.07671v1#bib.bib26)]
- Analyzes learning behaviors to portray learners’ proficiency profile [[27](https://arxiv.org/html/2410.07671v1#bib.bib27), 28]

**Traditional Psychometric-Based CD Approaches:**
- Based on psychological theories to depict student knowledge state through latent factors [[25](https://arxiv.org/html/2410.07671v1#bib.bib25), 29]
- Examples:
  * **Deterministic Inputs, Noisy And gate (DINA) model** [[29](https://arxiv.org/html/2410.07671v1#bib.bib29)]
    - Characterizes each student with a binary vector indicating mastery of knowledge concepts
    - Requires all relevant skills for highest positive response probability

**Neural Network (NN)-Based CD Approaches:**
- Propelled by the development of deep learning [[30](https://arxiv.org/html/2410.07671v1#bib.bib30), 31], [32]]
- Model complex interactions among learning elements (students, exercises, and knowledge concepts)

**Examples:**
* **NeuralCD** [[30](https://arxiv.org/html/2410.07671v1#bib.bib30)]
  - Employs multidimensional parameters for detailed depiction of students’ knowledge level and exercise attributes
  - Incorporates Multi-Layer Perceptron (MLP) to model complex interactions between students and exercises
* **RDGT** [[31](https://arxiv.org/html/2410.07671v1#bib.bib31)]
  - Designs a relation-guided dual-side graph transformer model to mine potential associations between learners and exercises

**Gap in Skillful Application:**
- Remains in effectively modeling the job recommendation task using cognitive diagnostics.

### II-C Disentangled Learning

**Disentangled Learning**
- **Purpose**: Identify and disentangle underlying explanatory factors of observed complicated data, enhancing efficiency and interpretability [[33](https://arxiv.org/html/2410.07671v1#bib.bib33), 34(https://arxiv.org/html/2410.07671v1#bib.bib34)]
- Initially in computer vision due to effectiveness [[34](https://arxiv.org/html/2410.07671v1#bib.bib34)]
- Recent approaches for graph-structured data: DGCL, DisenGCN [[35](https://arxiv.org/html/2410.07671v1#bib.bib35), 36(https://arxiv.org/html/2410.07671v1#bib.bib36)]
- **DGCL**: Employs contrastive learning to uncover latent factors within the graph, extracting disentangled representations [[36](https://arxiv.org/html/2410.07671v1#bib.bib36)]
- **DisenGCN**: Proposes a unique neighborhood routing mechanism for disentangling node representation in graph networks, enabling dynamic identification of latent factors [[35](https://arxiv.org/html/2410.07671v1#bib.bib35)]
- Learning disentangled representations of user latent intents from interaction feedback in recommendation domain [[37](https://arxiv.org/html/2410.07671v1#bib.bib37), 38(https://arxiv.org/html/2410.07671v1#bib.bib38), 39(https://arxiv.org/html/2410.07671v1#bib.bib39)]
- **MacridVAE**: Proposes a macro-micro disentangled variational auto-encoder to learn disentangled representations based on user behavior across multiple geometric spaces [[37](https://arxiv.org/html/2410.07671v1#bib.bib37)]

**Learning Disentangled Competency Representations (Cognitive Diagnostic Perspective)**
- Unexplored in this area [[---]]

**Overview Architecture of Proposed DISCO Framework**
- Figure 2: [Link](https://arxiv.org/html/2410.07671v1/extracted/5915675/figures/DISCO_framework.png)

## III Preliminaries

### III-A Problem Formulation

**Job Recommendation System**

**Introduction**:
- Problem definition for job recommendation
- Set of job seekers: **𝒞={c1, c2, …, cN}**
- Set of jobs: **𝒥={j1, j2, …, jM}**
- Set of skills at different granularity levels: **Å=∪l=1LÅl**, where **𝒮L** is the atomic skill level and K=∑l=1L|𝒮l|
- Each job seeker and job associated with textual documents describing resumes and requirements

**Job-Skill Relationship**:
- Q-matrix representing relationship between jobs and skills: **𝒬={qu⁢v}M×K**
  - qu⁢v=1 if job j requires skill sv, 0 otherwise

**Interaction Matrix**:
- Interaction matrix between job seekers and jobs: **ℛ={ru⁢v}N×M**
  - ru⁢v ∈ {0, 1, 2, 3} corresponds to four interaction behaviors:
    - **Browse**: candidate browses job
    - **Click**: candidate clicks on job
    - **Chat**: candidate engages in chat with recruiter about job
    - **Match**: both parties are satisfied and the pair is matched (ru⁢v=3)

**Goals**:
- Predict compatibility between jobs and candidates using interaction records ℛ, relationship matrix 𝒬, and resume/job descriptions
- Top-K job recommendation based on predicted degree of matching between candidates and jobs.

### III-B Base Embedding Model

**DISCO Framework:**
- Goal: model interaction patterns between users and jobs for flexible recommendation models
- Base embedding model (ℳ(𝒞,𝒥,ℛ)): encodes job seekers (𝒞) and jobs (𝒥) into d-dimensional matrices C and J, respectively
- Variable acquisition of embedding representations depends on base models
  - Collaborative information: MF [[40](https://arxiv.org/html/2410.07671v1#bib.bib40)]
  - Higher-order connectivity: NGCF/LightGCN [[41](https://arxiv.org/html/2410.07671v1#bib.bib41), 42]
  - Textual content: DPGNN [[9](https://arxiv.org/html/2410.07671v1#bib.bib9)]
- Example: NGCF using user-item interaction graph for propagation and embedding learning (right part of Figure 2)
  * Refined embeddings after k layers of propagation: cu(k) and jv(k)
  * Nonlinear activation function: σ
  * Element-wise multiply operator: ⊙
  * Sets representing interacted users/items: 𝒩u, 𝒩v
  * Trainable weight matrices for feature transformation: W1, W2
- Output embeddings obtained by concatenating all layers' representations (cu∈C, jv∈J)

## IV methodology

**DISCO Framework Overview**

The DISCO framework is detailed here with its four main components:
- **Hierarchical skill-aware representation disentangling**
- **Level-aware self-attention network**
- **Level-wise contrastive learning**
- **Interaction diagnosis module**, as depicted in Figure 2.

### IV-A Hierarchical Representation Disentangling

**Modeling Latent Skill Factors**
* The main goal is to predict job seeker-job matching based on observed interactions
* Skill factors influence outcome: candidate's skill mastery, job difficulty [[43](https://arxiv.org/html/2410.07671v1#bib.bib43)]
* Prediction objective: y = f(X) = g(E(zc), E(zj))
	+ X = (cu, jv, sjv)
	+ y: learned matching score
	+ zc, zj: skill factors for candidates and jobs, respectively
	+ E(z): encoding function
* Optimization objective: θ∗ = arg min∑i⁡|ℛ|−log⁡p(yi│Xi) = arg max∑i⁡|ℛ|log⁡p(zi│Xi)
	+ Approximation: E(z) ≈ g(E(z)) [[44](https://arxiv.org/html/2410.07671v1#bib.bib44)]
* Constraining approximation error within prediction function g(⋅)

**Hierarchical Skill-Aware Disentangling**
* Primary determinants of interaction outcome: skill proficiency, demand [[18](https://arxiv.org/html/2410.07671v1#bib.bib18)]
* Explicitly disentangle skill factors for understanding and enhancing interpretability
* Skills at different levels of granularity [[43](https://arxiv.org/html/2410.07671v1#bib.bib43)]
* Construct L-layer mappers to project embeddings into hierarchical skill spaces: cu,lh and jv,lh
	+ Wlc, Wlj: trainable matrices
	+ dh: hidden dimension
* Build multi-level encoders for users and jobs to learn ability prototypes (cu,lz) and skill difficulty prototypes (jv,lz) at each skill layer
	+ Encoder architecture: multilayer perceptron network (MLP)
	+ dz = |ℒL|: number of atomic skills.

### IV-B Level-Aware Association Modeling

**Level-Aware Self-Attention Network**
* Enhances learning by exploring inter-level skill relationships
* Incorporates correlated information into enhanced skill representations (c~u,lz}, jv,lz)
* Uses Self-Attention module for processing query, key and value vectors (𝒬c, 𝒦c, 𝒱c; 𝒬j, 𝒦j, 𝒱j)
* Design:
  + c~u,lz, jv,lz ∈ ℝdz are enhanced l-level skill aware representations
  + S⁢e⁢l⁢f⁢A⁢t⁢t⁢(⋅) indicates the Self-Attention module [[45](https://arxiv.org/html/2410.07671v1#bib.bib45)]

**Level-Wise Contrastive Learning**
* Enhances robustness of skill-aware disentangled representations
* Inspired by recent developments in contrastive learning [[44](https://arxiv.org/html/2410.07671v1#bib.bib44)]
* Proposed level-wise contrastive learning loss:
  + Maximizes expectation of L subtasks (pθ⁢(cu′\|cu,zc,l))
  + Formulated for job seeker side only
* Contrastive learning subtask for l-level ability:
  + Defined as Eq. (8) in the text
  + pθ⁢(cu′\|cu,zc,l) denotes candidate ability contrastive learning subtask
  + zc,l is the l-th level latent skill factor of the job seeker
* Goal: Learn optimal L ability prototypes that maximize expectation of L subtasks.
* Implementation:
  + Augment l-level ability representation (c~u,lz+) by adding random noises (Δu,l′), subject to ‖Δ‖2=ϵ and the second constraint for maintaining validity of positive samples.

### IV-C Interaction Diagnosis Module

**Interaction Diagnosis Module for Job Seekers and Jobs**

**Cognitive Diagnosis Theory**:
- Key research focus: measuring a tester's ability level by modeling their ability representations against the difficulty representations of exercises across different knowledge concepts
- In DISCO framework, skill characteristics of job seekers and jobs are disentangled and mapped to skill dimensions, enabling interaction modeling using diagnosis functions

**Neural Diagnosis Function**:
- Seamlessly integrates with non-linear neural network layers
- Capable of modeling high-dimensional interactive elements, enabling acquisition of extensive knowledge and presentation of interpretable information
- Formalized as equation (12): 𝒯⁢(c~u,lz,j~v,lz) = 𝒬jvl⊙(σ⁢(c~u,lz) - σ⁢(j~v,lz))
- Obtains the matching distance in the l-level skill space between cu and jv from a diagnosis perspective

**Hierarchical Diagnosis Prediction**:
- Aggregates hierarchical competency matching distances between candidates and jobs
- Concatenates L-layer matching distance representations into an aggregated interaction vector hu,v
- Uses full connection layers to model high-order interaction features
- Predicts the probabilities of different interaction categories between cu and jv

**Loss Function**:
- Multi-class cross-entropy loss function for predicting job seeker-job interaction categories
- Constructs complete contrastive learning loss as the optimization objective

**Statistics of Experimental Datasets**:
| Statistics        | Technology | Service | Edu-Rec  |
|------------------|:---------:|:-------:|:--------:|
| #Candidates      | 4,726     | 10,022  | 61,567   |
| #Items           | 34,962    | 23,866  | 20,828   |
| #Skills         | 986       | 3,241   | 384      |
| #Interactions     | 616,504   | 866,065 | 2,200,731|
| Avg. interactions per user | 130.45   | 86.41    | 35.74    |

## V experiments

**Experimental Validation of DISCO Framework**

**Datasets**:
- MF (Normal)
- NCF
- AutoInt
- FINAL
- NGCF (Normal)
- LightGCN
- DPGNN

**Performance Metrics**:
- AUC (Area Under the ROC Curve)
- HR@5, HR@10 (Hits at position n)
- NDCG@5, NDCG@10 (Normalized Discounted Cumulative Gain at positions n)

**Base Models and Baselines**:
- Normal: MF, NCF, AutoInt, NGCF, LightGCN, DPGNN
- Underline: NCF, AutoInt, FINAL
- DISCO

**Results**:
- **DISCO** outperforms baselines in most metrics on all datasets
- Significant improvements marked with "*"

**Technology and Service Based Models**:
- MF (Mean-Field)
- NCF (Neural Collaborative Filtering)
- AutoInt (Autoencoder with Attention)
- FINAL (Final version of DISCO)
- NGCF (Network-Based Collaborative Filtering)
- LightGCN (Light Graph Convolutional Network)
- DPGNN (Dynamic Point-wise and Graph Neural Network)

### V-A Experiment Setups

**Dataset Description and Preparation:**
- Dataset provided by an online recruitment platform with four behaviors: Browse, Click, Chat, Match
- Filtered out job seekers with fewer than ten Match interaction logs and jobs with fewer than five records
- No sensitive information, all IDs remapped to ensure they do not correspond to original identifiers
- Selected two subsets based on career clusters: technology and service
- Randomly split data into three parts for training, validation, and testing sets

**Baseline Approaches:**
- Four widely used recommendation methods as base models: MF, NGCF, LightGCN, DPGNN
- Incorporated three interaction modeling methods (NCF, AutoInt, FINAL) into base models to construct complete baselines
- Selected two state-of-the-art methods (SHPJF and ECF) for job recommendation and interpretable recommendation, respectively

**Evaluation Protocols and Implementation Details:**
- Employed three widely used metrics: Area Under the ROC Curve (AUC), Hit Ratio (HR@k), Normalized Discounted Cumulative Gain (NDCG@k)
- Set k to 5 and 10 for evaluation of job recommendation task
- Utilized random sampling of 25 jobs as negative instances for each positive instance
- Implemented all models using Pytorch with Python on a Linux server with eight Nvidia A800 GPUs
- Conducted experiments five times and used average value as final result
- Used t-test to identify significant differences between performances of DISCO and baselines
- Initialized network parameters with Xavier initialization, learning rate searched from {5e-5, 8e-5, 1e-4, 2e-4, 5e-4}
- Set coefficient λ of contrastive loss to 1e-3.

### V-B Performance Comparison

**Performance Comparison: DISCO Framework vs Baselines**

**Observations:**
- **DISCO framework** embedded in four models outperforms all baselines on two recruitment datasets: AUC metric improved by an average of 0.65 and 0.64, HR@5 and NDCG@5 metrics improved on average by 2.96 to 3.62.
- Significant advantages for job recommendation tasks with DISCO framework over baselines in terms of:
  - Greater improvement in recommendation metrics than classification metrics
  - Modeling high-order user-item interactions effective in enhancing performance (FINAL, AutoInt methods)
  - Job recommendations using NGCF and LightGCN models are more effective than other model types due to high connectivity between job seekers and jobs.

**Experimental Results:**
- **Technology dataset**: DISCO outperforms baselines in AUC, HR@5, NDCG@5, HR@10, NDCG@10 (refer to Table II).
- **Edu-Rec dataset**: DISCO holds significant advantages over interaction methods for recommendation tasks on educational data despite the increase in data size (not limited to job recommendation domain).

**Additional Comparisons:**
- DISCO outperforms SHPJF model by 2.25% and 3.20% for HR@5 and NDCG@5, respectively.
- Significant relative improvement of DISCO over interpretable recommendation model ECF: 31.81% and 52.36% for HR@5 and NDCG@5, respectively.

### V-C Ablation Study

**RQ2 Ablation Experiment Results:**
* Conducted to investigate effectiveness of each component in DISCO framework on Technology dataset using NGCF as base model
* Variations: w/o HD, SA, CL, ID

**Findings:**
1. **Impact of Submodules**: All variations perform worse than NGCF-DISCO, highlighting significance of designed submodules.
2. **Hierarchical Skill-Aware Disentangling Module (w/o HD)**: Elimination causes considerable drop in performance, validating importance and effectiveness of hierarchical disentangling idea.
3. **Sensitivity Analysis:**
   - Learning rate: Figure 4 (refer to the caption for details)
   - Coefficient λ: Figure 5 (refer to the caption for details)

### V-D Parameter Sensitivity Analysis

**Parameter Sensitivity Analysis for RQ3:**

* Explores hyper-parameter impacts, mainly focusing on learning rate and weight coefficient λ of contrastive loss
* Experiments conducted on Technology dataset using NGCF and DPGNN as base models
* Set learning rates to {5e-5, 8e-5, 1e-4, 2e-4, 5e-4} and λ values {1e-5, 5e-4, 1e-3, 5e-3, 1e-2}
* Optimal learning rates found to be 8e-5 for NGCF-DISCO and 1e-4 for DPGNN-DISCO (both increase before decreasing)
* Best performance achieved when λ is set to 1e-3 for both models
* An intriguing observation: trends of coefficients impacting performance differ across the three metrics as λ value increases

### V-E Case Study

**Case Study on Interpretability of DISCO Model**

**Purpose**:
- Explore interpretability of DISCO model through a case study
- Analyze job seekers' abilities and difficulty of job skills

**Methodology**:
- Select pair of job seeker and position that achieved matching in the job search process
- Demonstrate interpretable content by outputting hierarchical skill-associated representations from the model

**Findings**:
- **Candidate c's mastery of each skill at second and third levels**: shown as examples, with corresponding proficiency influencing coarse-grained level (e.g., s1)
- **Job requirement values for each skill**: compared to candidate c's proficiency level
- **Compatibility between candidate c's proficiency level and job j's required level**: explains the pair's matching

**Benefits**:
- Output from the model improves interpretability of job recommendations
- Provides deeper understanding of the job search process for both job seekers and recruiters.

## VI conclusion

**DISCO Framework for Job Recommendations:**
* **Components**: hierarchical representation disentangling module, level-aware association modeling (inter-level knowledge influence module & level-wise contrastive learning), interaction diagnosis module with neural diagnosis function.
* **Hierarchical Representation Disentangling Module**: mines skill-related factors in job and job seeker representations.
* **Level-Aware Association Modeling**: enhances communication and robust representation learning, includes inter-level knowledge influence module and level-wise contrastive learning.
* **Interaction Diagnosis Module**: integrates a neural diagnosis function for effective modeling of multi-level recruitment interaction process between job seekers and jobs.
* **Cognitive Measurement Theory**: incorporated in the interaction diagnosis module.
* **Datasets**: two real-world recruitment recommendation datasets, one educational recommendation dataset used for evaluation.
* **Results**: demonstrate effectiveness and interpretability of DISCO framework.

