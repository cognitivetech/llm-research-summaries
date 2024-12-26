# MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders

**Authors**: Cheng Li, May Fung, Qingyun Wang, Chi Han, Manling Li, Jindong Wang

https://arxiv.org/abs/2410.06845v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [Related Work](#related-work)
  - [2.1 Large Language Models for healthcare](#21-large-language-models-for-healthcare)
  - [2.2 Self-play frameworks in Large Language Models](#22-self-play-frameworks-in-large-language-models)
- [3 MentalArena](#3-mentalarena)
  - [3.1 Preliminaries](#31-preliminaries)
  - [3.2 Overview of the Framework](#32-overview-of-the-framework)
  - [3.3 Patient: Symptom Encoder](#33-patient-symptom-encoder)
  - [3.4 Therapist: Symptom Decoder](#34-therapist-symptom-decoder)
  - [3.5 Model Optimizer](#35-model-optimizer)
- [Experiment](#experiment)
  - [4.1 Setup](#41-setup)
  - [4.2 Main Results and Ablation Study](#42-main-results-and-ablation-study)
  - [4.3 Effectiveness Analysis](#43-effectiveness-analysis)
- [Discussion](#discussion)
  - [5.1 Can Symptom Encoder mimic real mental health patient?](#51-can-symptom-encoder-mimic-real-mental-health-patient)
  - [5.2 The validity of generated data](#52-the-validity-of-generated-data)
  - [5.3 Generalization](#53-generalization)
  - [5.4 Fine-tuning vs. forgetting](#54-fine-tuning-vs-forgetting)
  - [5.5 Qualitative analysis](#55-qualitative-analysis)
- [6 Conclusion, Societal Impact and Limitations](#6-conclusion-societal-impact-and-limitations)
  - [Ethics Statement](#ethics-statement)
- [Appendix A Prompts](#appendix-a-prompts)
- [Appendix B Prompt template for baseline](#appendix-b-prompt-template-for-baseline)
- [Appendix C](#appendix-c)
  - [C.1 Introduction](#c1-introduction)
  - [C.2 Benchmarks for generalization](#c2-benchmarks-for-generalization)
  - [C.3 Examples](#c3-examples)
- [Appendix D](#appendix-d)
  - [D.1 Perplexity](#d1-perplexity)
  - [D.2 Diversity Gain](#d2-diversity-gain)
- [Appendix E Training data samples](#appendix-e-training-data-samples)
- [Appendix F](#appendix-f)
  - [F.1 Examples](#f1-examples)
  - [F.2 Introduction on cognitive model](#f2-introduction-on-cognitive-model)
- [Appendix G Detailed experimental results](#appendix-g-detailed-experimental-results)
- [Appendix H Training details](#appendix-h-training-details)
  - [H.1 Setup for GPT-3.5-turbo](#h1-setup-for-gpt-35-turbo)
  - [H.2 Setup for Llama-3-8b](#h2-setup-for-llama-3-8b)
- [Appendix I Case study](#appendix-i-case-study)

## Abstract

**Introduction:**
- Mental health disorders are a serious issue worldwide
- Lack of access to adequate care highlights importance of models for diagnosis and treatment
- Privacy concerns limit accessibility of personalized mental health data

**MentalArena:**
- Self-play framework for training language models
- Generates domain-specific personalized data
- Improves capability for personalized diagnosis, treatment (as therapist) and information seeking (as patient)

**Symptom Encoder:**
- Simulates real mental health patients from cognition and behavior perspectives

**Symptom Decoder:**
- Addresses intent bias during patient-therapist interactions
- Compares diagnosed symptoms with encoded symptoms
- Manages dialogue between patient and therapist based on identified deviations

**Evaluation:**
- Evaluated against 6 benchmarks including biomedicalQA and mental health tasks
- Compared to 6 advanced models
- Significantly outperforms counterparts, including GPT-4, when fine-tuned on GPT-3.5 and Llama-3-8b

**Conclusion:**
- MentalArena inspires future research on personalized care

## 1 Introduction

**Mental Health Disorders:**
* Prevalence: 970 million people worldwide in 2019 (WHO, 2022)
* Most common disorders: anxiety and depression
* Lack of access to adequate care due to under-resourced health systems
* Importance of developing machine learning models for diagnosis and treatment

**Challenges:**
* Existing AI therapist systems use templates and decision trees, not flexible enough for personalized care (Fiske et al., 2019; D’Alfonso, 2020; Grodniewicz and Hohol, 2023; Devarakonda et al., 2019)
* Data collection is challenging due to privacy concerns in the medical domain
* Limited availability of training data as models scale (Hu et al., 2024b; Yang et al., 2024b; Liang et al., 2024; Wu et al., 2024; Wang et al., 2024d)

**Approaches:**
* Self-play training: models play different roles and evolve during interaction with other models (Hu et al., 2024b; Yang et al., 2024b; Liang et al., 2024; Wu et al., 2024; Wang et al., 2024d)
* Collecting data from interactions for training (Figure 1)

**MentalArena:**
* Self-play framework for mental health disorder diagnosis and treatment
* Consists of three modules: Symptom Encoder, Symptom Decoder, and Model Optimizer (Figure 1)
* Mental health patients modeled based on cognitive models and behavioral patterns (Symptom Encoder)
* Diagnosis and treatment interactions simulated to generate personalized dialogues while mitigating intent bias (Symptom Decoder)
* Data collected during each iteration for evolving models through training (Figure 1)

**Evaluation:**
* Experiments conducted on six benchmarks: biomedical QA and mental health detection
* Comparisons with state-of-the-art and mental health models, as well as prompt engineering approaches
* Improvements to base models (20.7% over GPT-3.5-turbo and 6.6% over Llama-3-8b)
* Significant outperformance of fine-tuned model based on GPT-3.5-turbo over GPT-4o (7.7%)

**Self-play Training Dynamics:**
* High correlation between perplexity score and model performance (Marion et al., 2023; Wang et al., 2023)
* Increase in performance if diversity gain exceeds certain thresholds (Bilmes, 2022)

**Generalization:**
* MentalArena's generalization ability proven on MedMCQA and MMLU datasets (Pal et al., 2022; Hendrycks et al., 2020)

**Catastrophic Forgetting:**
* Results on BIG-Bench-Hard show no decrease in performance for fine-tuned models (Suzgun et al., 2022)

## Related Work
### 2.1 Large Language Models for healthcare

**Large Language Models (LLMs) in Healthcare**

**Research on Large Language Models**:
- Explored potential applications in healthcare
    - Jiang et al., [2023](https://arxiv.org/html/2410.06845v1#bib.bib17)
    - Li et al., [2023b](https://arxiv.org/html/2410.06845v1#bib.bib22)
    - Liu et al., [2023](https://arxiv.org/html/2410.06845v1#bib.bib24)
    - Lupetti et al., [2023](https://arxiv.org/html/2410.06845v1#bib.bib27)
    - Nori et al., [2023a](https://arxiv.org/html/2410.06845v1#bib.bib30)
    - Singhal et al., [2023](https://arxiv.org/html/2410.06845v1#bib.bib41)
    - Wu et al., [2023](https://arxiv.org/html/2410.06845v1#bib.bib53)
    - Wang et al., [2024c](https://arxiv.org/html/2410.06845v1#bib.bib49)
    - Li et al., [2023a](https://arxiv.org/html/2410.06845v1#bib.bib21)

**Fine-Tuning LLMs**:
- **Singhal et al.** ([2023](https://arxiv.org/html/2410.06845v1#bib.bib41)) fine-tuned **PaLM-2** for medical applications, achieving 86.5% accuracy on the MedQA dataset
- **Wu et al.** ([2023](https://arxiv.org/html/2410.06845v1#bib.bib53)) fine-tuned **LLaMA** on medical literature, showing strong performance in biomedical QA tasks

**Mental Health Domain**:
- **Two main approaches**:
    - Fine-tuning domain-specific LLMs on existing datasets or social media data (e.g., Mental-LLaMA, Mental-LLM)
    - Enhancing mental health performance through prompt engineering
- **Prompt Engineering**:
    - Yang et al. ([2023](https://arxiv.org/html/2410.06845v1#bib.bib56)) proposed emotion-enhanced prompting strategies for LLMs in explainable mental health analyses
- **Self-Play Training**:
    - MentalArena fine-tunes mental health models through self-play training, with the base model assuming both patient and therapist
    - Allows for more effective model refinement as data is generated dynamically during interactions

### 2.2 Self-play frameworks in Large Language Models

**Self-Play Methods in Language Models (LLMs)**

**Overview:**
- Self-play involves a model interacting with copies of itself, refining performance through feedback loop [Silver et al., 2016][Silver et al., 2017]
- Effective in multiplayer games and environments with multiple roles [Silver et al., 2016][Silver et al., 2017]

**Advantages:**
- More efficient strategy for obtaining feedback without external environment

**Research:**
- Taubenfeld et al. (2024): [Biases in LLM-generated debate simulations](https://arxiv.org/abs/2410.06845v1#bib.bib43)
- Ulmer et al. (2024): [Principle-guided conversations](https://arxiv.org/abs/2410.06845v1#bib.bib45)
- Lu et al. (2024): [Self-simulated dialogues with character profiles](https://arxiv.org/abs/2410.06845v1#bib.bib26)
- Askari et al. (2024): [SOLID framework for intent-aware role-play](https://arxiv.org/abs/2410.06845v1#bib.bib1)

**Challenges:**
- Inadequate data in training corpus prevents accurate simulation of real mental health patients

**Solution:**
- MentalArena introduces Symptom Encoder component to effectively model real mental health patients [No specific reference found, please check the document for more information]

## 3 MentalArena

**Figure 2: Symptom Decoder** - Aims to reduce intent bias between therapists and patients via patient decoding and dynamic conversation control. For the accuracy of diagnostic information from the therapist, the patient simulates their updated health condition following treatment or medication implementation.

### 3.1 Preliminaries

**Mental Health Disorder Diagnosis and Treatment**
- Evaluation of an individual's overall health state, focusing on mental and emotional well-being
- Key symptoms: emotional (anxiety, depression), cognitive (memory problems), behavioral changes (social withdrawal)
- Formal diagnosis via clinical interviews identifying specific disorders like depression, anxiety, or schizophrenia
- Treatment often combines psychotherapy (e.g., cognitive-behavioral therapy), lifestyle changes, and medication
- Medications like antidepressants and mood stabilizers regulate brain chemicals to alleviate symptoms (Prince et al., 2007)

### 3.2 Overview of the Framework

**MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders**

**Introduction**:
- Self-play training paradigm not yet explored in medical domain due to data deficiency and intent bias problem (Schmidgall et al., 2024; Wang et al., 2024a)
- MentalArena framework designed for self-play training of language models to facilitate diagnosis, treatment, and medication of mental health disorders

**MentalArena Modules**:
1. **Symptom Encoder**:
   - Models mental health patient from cognitive models and behavioral patterns
   - Provides information on coping strategy and behavior principles
2. **Symptom Decoder**:
   - Emulates diagnosis and treatment process between a patient and therapist
   - Generates personalized dialogue to mitigate intent bias (Britten et al., 2000; West, 1984)
3. **Model Optimizer**:
   - Fine-tunes the model using paired data from Symptom Encoder and Decoder

**Self-play Training**:
- Objective: Obtain M∗ that can achieve better performance in diagnosis and treatment of patient, and information disclosure (as a patient)
- Initial health information 𝐱 and treatment/medication information 𝐳 are used as input
- Model iteratively plays therapist (Mt) and patient (Pt) roles to generate diagnosis, treatment, and medication data
- Symptom Encoder disentangles initial health information into cognitive and behavioral principles
- Symptom Decoder generates personalized dialogue with key information 𝐳={δ,β,γ} from each round of communication
- Patient's health state evolves as treatment/medication plans are administered, reflected in sequential updates to encoded symptoms S1, S2, …, Sk−1
- Therapist provides optimal diagnosis information 𝐳besτ that is crucial for model optimization
- Model Optimizer fine-tunes the model using paired data (S1, 𝐳besτ), (Sd, γbesτ), and (Sd, βbesτ) over T rounds to obtain the optimal model.

### 3.3 Patient: Symptom Encoder

**Symptom Encoder Module**
- **Models mental health patients from cognitive and behavioral perspectives**: learns **symptoms S0** from aspects of cognition and behavior

**Cognitive Model (based on CBT principles)**
- Designed based on cognitive behavior therapy (CBT) principles
- Addresses maladaptive cognitive structures in various contexts: familial conflicts, relationship challenges, workplace challenges, etc.
- Consists of 8 key components:
    - **Relevant history**
    - **Core beliefs**
    - **Intermediate beliefs**
    - **Coping strategies**
    - **Situational factors**
    - **Automatic thoughts**
    - **Emotions**
    - **Behaviors**
- Examples of cognitive models can be found in [Section F.2](https://arxiv.org/html/2410.06845v1#A6.SS2 "F.2 Introduction on cognitive model ‣ Appendix F Cognitive model and behavior pattern ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")

**Patient Behavior Modeling**
- Uses behavior principles collected by Louie et al. (2024) as a **behavior library**, created by 25 mental health experts
- Examples of behavior patterns are shown in [Section F.1](https://arxiv.org/html/2410.06845v1#A6.SS1 "F.1 Examples ‣ Appendix F Cognitive model and behavior pattern ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")
- Semantically match coping strategies of cognitive models with each **behavior pattern**
- Obtain embeddings for each coping strategy and behavior principle via Bert-base, considering effectiveness and cost
- Compute semantic similarity between coping strategies and behavior patterns
- Select the 5 highest-scoring behavior patterns and prompt GPT-4-turbo to pick one as the final pattern
- Integrate the final behavior pattern into patient via prompt (see [Appendix A](https://arxiv.org/html/2410.06845v1#A1 "Appendix A Prompts ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders"))

### 3.4 Therapist: Symptom Decoder

**Mental Arena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders**

**Intent Bias in Therapist-Patient Interactions**:
- Patient may express one opinion, but therapist may misunderstand due to prior knowledge and lack of experience (Britten et al., [2000](https://arxiv.org/html/2410.06845v1#bib.bib4); West, [1984](https://arxiv.org/html/2410.06845v1#bib.bib51))
- Intention bias can arise in conversations between patients and therapists played by AI models, resulting in inaccurate diagnosis and treatment

**Symptom Decoder**:
- Designed to mitigate intent bias
- Therapist reviews patient's health information and conducts detailed analysis of cognitive and behavioral patterns
- Semantically matches encoded symptom (S0) with diagnosed symptom (Sd)
- Guides subsequent conversations based on differences between S0 and Sd

**Therapist's Role**:
- Decodes cognitive and behavior principles according to conversation history
- Computes semantic similarity score of decoded symptom (Sd) and encoded symptom (S0)
- If score is less than 0.9, therapist receives feedback to guide further inquiries

**Diagnostic and Treatment Plans**:
- Patient reviews diagnostic plans and selects the most appropriate one based on health condition
- Therapist proposes treatment and medication plans in accordance with selected diagnosis
- Encoded symptoms are updated as different plans are administered, reflecting patient's evolving health state
- Therapist provides optimal diagnosis and treatment information (δbesst)

### 3.5 Model Optimizer

**MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders**

**Training Process**:
- Obtain treatment, diagnosis, and medication using Symptom Decoder
- Train M model in a self-play manner to improve its capability for personalized diagnosis and treatment (as therapist) and presenting information (as patient)

**Supervised Fine-tuning**:
- Illustrated in [Figure 8](https://arxiv.org/html/2410.06845v1#A5.F8 "In Appendix E Training data samples ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders"))
- During each iteration, both patient and therapist are powered by the same model M and improved when it is updated

**Flexibility**:
- Framework is flexible to allow different base models for the two roles
- Adopt same base model due to:
  - **Efficiency**: Training one base model is more efficient than training different models
  - **Reducing knowledge gap**: Training one base model can help reduce the knowledge gap between two roles

**Detailed Training Settings**: [Appendix H](https://arxiv.org/html/2410.06845v1#A8 "Appendix H Training details ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")

## Experiment
### 4.1 Setup

**Datasets:**
- Six datasets used: MedQA, MedMCQA, PubMedQA, CASM, Dreaddit, Irf (details in Appendix C)
- Evaluation covers biomedical QA and mental health detection
- Datasets span diagnosis, treatment, medication, general mental health tasks (depression/suicide, stress, interpersonal risk factors), real-world cases

**Baselines:**
- Comparison with state-of-the-art LLMs: GPT-3.5-turbo, GPT-4o, Llama-3-8b
- Comparison with mental health models: MentaLLaMa-13b, Mental-LLM-alpaca, Mental-LLM-t5
- Comparison with prompt engineering methods: MedPrompt, Zero-shot CoT (implemented on GPT-3.5-turbo, GPT-4o, Llama-3-8b)
- Results reported based on accuracy in zero-shot setting for fair comparison.

### 4.2 Main Results and Ablation Study

**Mental Health Models Performance Comparison**

**Key Findings:**
- Our fine-tuned models outperform other open-source and closed-source models in MentalArena:
  * Surpass GPT-3.5-turbo by 20.74% on average
  * Outshine Llama-3-8b by 6.64% on average
  * Surpass baseline GPT-4o's performance (GPT-3.5-turbo and Llama-3-8b)

**Table 2 Results:**
| Model | MedQA | MedMCQA | PubMedQA | CAMS | dreaddit | Irf | AVG |
|--------|-------|---------|----------|------|---------|-----|-----|
| MentaLLaMa-13b | 28.32% | 12.42% | 28.96% | 37.28% | 62.08% | 46.81% | 35.98% |
| Mental-LLM-alpaca | 28.32% | 12.42% | 0.00% | 29.76% | 64.98% | 51.96% | 31.24% |
| Mental-LLM-t5 | 0.00% | 0.32% | 49.09% | 27.04% | 63.29% | 47.70% | 31.24% |
| GPT-4o | 87.86% | 74.20% | 60.06% | 27.68% | 49.03% | 64.65% | 60.58% |
| GPT-4o+MedPrompt | 90.17% | 78.34% | 67.38% | 31.52% | 53.27% | 64.65% | 64.22% |
| Base: GPT-3.5-turbo+Chain-of-thought | 64.16% | 33.76% | 44.68% | 28.96% | 49.03% | 64.65% | 47.54% |
| +MedPrompt | 69.94% | 43.89% | 47.26% | 30.20% | 49.03% | 64.65% | 50.83% |
| +Ours | 74.57% | 91.08% | 97.56% | 32.80% | 49.03% | 64.65% | 68.28% |
| Base: Llama-3-8b+Chain-of-thought | 70.52% | 42.04% | 86.59% | 25.12% | 58.45% | 45.76% | 54.75% |
| +Chain-of-thought | 75.14% | 47.77% | 88.21% | 33.60% | 62.22% | 45.91% | 58.81% |
| +MedPrompt | 76.88% | 49.41% | 89.99% | 35.08% | 61.59% | 48.05% | 60.17% |
| +Ours | 78.03% | 50.32% | 92.68% | 29.60% | 65.46% | 52.25% | 61.39%

**Ablation Study:**
- Each bar represents the performance of a model trained on different settings: Baseline, Diagnosis, Treatment, Medicine, Symptom Encoder, Symptom Decoder.
- The bars in dark blue are higher than others, indicating each part of our data is effective in different models.
- Treatment and medicine data are more effective in biomedical QA tasks than mental health tasks; diagnosis data contributes to all tasks similarly.

### 4.3 Effectiveness Analysis

**Self-Play Training Results Analysis**

**Performance Improvement:**
- [Refer to caption](https://arxiv.org/html/2410.06845v1/x4.png) Figure 4: Effectiveness analysis of self-play training
- Initial models improve iteratively until performance peaks and declines
- GPT-3.5-turbo: performance improves in first two iterations, then declines
- Llama-3-8b: performance increases in first four iterations before weakening after iter_4

**Authenticity and Validity Verification:**
- [Table 4](https://arxiv.org/html/2410.06845v1#A7.T4) presents detailed results for each iteration
- Llama: authenticity = 65.67, validity = 85.49
- Our model: authenticity = 73.35 (improvement), validity = 93.13 (significant increase)
- GPT: authenticity = 63.82, validity = 93.13

**Iteration Analysis:**
- Perplexity score and model performance highly relevant
- Diversity gain indicator for improvement or decline of the model
- Figure 4 shows results of perplexity score, diversity gain, and model performance at each iteration
- Trends in perplexity score and model performance are similar
- Borderline exists for diversity gain: if surpassed, performance improves; below borderline, performance declines.

**Perplexity Score:**
- Computed by sampling 500 generated data at each iteration
- Indicates how well the model understands the training data

**Diversity Gain:**
- Measures novelty of generated data compared to previous iteration's data
- Helps maintain a balance between diversity and performance during training.

## Discussion
### 5.1 Can Symptom Encoder mimic real mental health patient?

**Study Design**
- Generate 50 four-turn conversations between AI-patient (baseline or model) and AI-therapist (GPT-4o, OpenAI [2024](https://arxiv.org/html/2410.06845v1#bib.bib34))
- Assess whether the patient is human or AI-generated after each conversation by the therapist
- Present findings in Table 3 ([Table 3](https://arxiv.org/html/2410.06845v1#S4.T3 "In 4.3 Effectiveness Analysis ‣ 4 Experiment ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders"))
**Results**
- Our models more accurately simulate mental health patients compared to baseline models ([Table 3](https://arxiv.org/html/2410.06845v1#S4.T3 "In 4.3 Effectiveness Analysis ‣ 4 Experiment ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders"))

### 5.2 The validity of generated data

**Verifying Data Validity:**

- **Sample Selection**: Choose 1500 samples from data for fine-tuning GPT and Llama versions.
- **Validation Process**: GPT-4o is prompted with a query, followed by its response. The validity is determined by answering "Yes" or "No" to the question: Is the answer reasonable?
- **Validity Rate Calculation**: Compute the validity rate of these QA pairs and present results in [Table 3](https://arxiv.org/html/2410.06845v1#S4.T3 "In 4.3 Effectiveness Analysis ‣ 4 Experiment ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")
- **Results**: [Table 3](https://arxiv.org/html/2410.06845v1#S4.T3 "In 4.3 Effectiveness Analysis ‣ 4 Experiment ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders") demonstrates that data generated by MentalArena is both valid and reasonable.

### 5.3 Generalization

**Generalization Experiments**
- **Figure 5**: MentalArena models surpass corresponding baseline models for a large margin on all tasks, covering several different diseases (source: arxiv.org)
  - Models generate data for training domain model by simulating cognitive and behavioral patterns of real mental health patients
  - Estimated that 26% of Americans ages 18 and over suffer from a diagnosable mental disorder in a given year
- **Exploring generalization to other illnesses** (source: arxiv.org)

**Forgetting Experiments**
- **Figure 6**: Evaluation on 6 medically relevant subset of MMLU tasks: medical genetics test, college biology test, college medicine test, professional medicine test, clinical knowledge test, high school biology test (source: arxiv.org)
- MedMCQA and MMLU used as benchmarks in this study (sources: Pal et al., 2022; Hendrycks et al., 2020)
- Results show that MentalArena models surpass corresponding baseline models for a large margin on all tasks, proving the generalization ability of the method in medical domain. (source: arxiv.org)

### 5.4 Fine-tuning vs. forgetting

**Potential Dilemma of Fine-Tuning LLM on Specific Tasks:**
- Catastrophic forgetting of original capabilities possible
- Exploring forgetting possibility of MentalArena on BIG-Bench-Hard (BBH)

**BBH Overview:**
- Contains 21 tasks: semantic understanding and logical reasoning
- Sampling 100 instances for each task due to cost savings

**Comparing Fine-Tuned Model:**
- With baseline models GPT-3.5-turbo and Llama-3-8b
- Reporting average performance on those 21 tasks in [Figure 6]
- Detailed results in [Appendix G]

**Results:**
- Our model does not decrease performance in most benchmarks
- Can even improve results, suggesting latent relationships with general benchmarks
- Generated data may benefit other cognitive tasks due to cognitive similarity in humans.

### 5.5 Qualitative analysis

**Case Study Comparison: GPT-3.5-Turbo vs. Our Model**

**Qualitative Analysis:**
- Illustrated in [Figure 7](https://arxiv.org/html/2410.06845v1#S5.F7 "In 5.5 Qualitative analysis ‣ 5 Discussion ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")
- GPT-3.5-Turbo provides incorrect response to medical question ([Figure 7](https://arxiv.org/html/2410.06845v1#S5.F7))
- Discrepancy due to valuable medical knowledge in patient-therapist interactions data
- Additional cases for comparison in [Appendix I](https://arxiv.org/html/2410.06845v1#A9 "Appendix I Case study ‣ MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")

## 6 Conclusion, Societal Impact and Limitations

**MentalArena: Self-play Framework for Diagnosis and Treatment of Mental Health Disorders**

**Introduction:**
- Introduce MentalArena as a self-play framework for training language models in generating domain-specific personalized data
- Enables creation of models capable of functioning as therapists and patients

**Evaluation:**
- Compared MentalArena against six benchmarks, including biomedical QA and mental health tasks
- Evaluated using GPT-3.5-turbo and Llama-3-8b models
- Significantly outperformed counterparts, including GPT-4o on these benchmarks

**Contributions:**
- Offers solutions for personalized care
- Enhances accessibility to tailored treatments while safeguarding patient privacy
- Can help bridge the gap between mental health needs and availability of effective, individualized care

**Limitations:**
1. **Data Authenticity and Validity**:
   - Evaluated using GPT-4o, which may introduce deviations in results due to potential limitations in its performance
2. **Model Optimality**:
   - Llama-3-8b model may not represent the optimal model of MentalArena, as large-scale training was constrained by computational resources
3. **Additional Open-Source Models**:
   - Further implementation on additional open-source models could provide stronger evidence supporting the effectiveness of MentalArena.

### Ethics Statement

**Ethical Considerations for AI Mental Health Applications**

- Prioritize privacy and data protection
- Ensure transparency in AI-generated data usage
- Guidelines: AI should augment human judgment, not replace healthcare professionals
- Prevent bias to promote fairness in diagnosis and treatment
- Aim to avoid exacerbating disparities in mental healthcare

## Appendix A Prompts

**Patient Instructions:**
- You are a mental health patient who has been attending therapy sessions for several weeks
- Engage in a conversation with the therapist as you would during a cognitive behavioral therapy (CBT) session
- Align your responses with the provided background information and cognitive conceptualization diagram
- Use natural language, including hesitations, pauses, and emotional expressions, to enhance realism
- Gradually reveal deeper concerns and core issues, allowing the conversation to flow naturally
- Maintain consistency with your profile throughout the conversation

**Therapist Instructions:**
- As a psychiatric expert, figure out the patient's mental illness diagnosis and severity
- Ask for personal information, symptom details (emotional, cognitive, behavioral, physiological), and relevant history events
- Provide a specific treatment based on the diagnosed illness
- Discuss potential changes in health state after treatment or medication.

**Patient Roleplay:**
[Name], a mental health patient with the following background information:
- History: [history]
- Cognitive Conceptualization Diagram:
  - Intermediate Beliefs: [intermediate belief]
  - Intermediate Beliefs during Depression: [intermediate belief depression]
  - Coping Strategies: [coping strategies]

During therapy sessions, you will simulate this patient while the user plays the role of the therapist. Follow these guidelines:
1. Emulate genuine patient demeanor and responses
2. Gradually reveal deeper concerns and core issues
3. Maintain consistency with your profile throughout the conversation
4. Engage in a dynamic and interactive conversation, responding authentically to prompts.

**Therapist Roleplay:**
- As a psychiatric expert, diagnose the patient's mental illness and severity by asking for relevant information
- Provide a specific treatment based on the diagnosed illness
- Discuss potential changes in health state after treatment or medication.

## Appendix B Prompt template for baseline

Baseline Prompt Templates:
- Zero-shot Input: Question
- Zero-shot CoT Input: Question + "Let's think step by step"
- MedPrompt: Random few-shot + Chain-of-thought + kNN + Ensemble with choice shuffle

## Appendix C
### C.1 Introduction

**Medical Datasets Used for Evaluation:**

**1. MedQA [Jin et al., 2021]**
- Free-form multiple-choice OpenQA dataset for medical problems
- Collected from professional medical board exams in English, simplified Chinese, and traditional Chinese
- Focus on psychosis subset of USMLE (United States Medical Licensing Exam)
- Testset contains 173 samples
- Prompt GPT-4 with "Are the questions related to psychosis? Just answer with Yes or No."

**2. MedMCQA [Pal et al., 2022]**
- Contains real world medical entrance exam questions from Indian universities: AIIMS and NEET-PG
- Get testset by selecting samples whose "subject name" is related to psychosis
- Total of 314 samples for evaluation

**3. PubMedQA [Jin et al., 2019]**
- Contains tests requiring a yes, no, or maybe answer to biomedical research questions using context from PubMed abstracts
- Evaluate LLMs' performance on domain knowledge in zero-shot setting without context
- Testset contains 328 samples

**4. Mental health datasets:**
- CASM [Garg et al., 2022]: Depression/suicide cause detection, has 625 test samples
- Dreaddit [Turcan and McKeown, 2019]: Stress detection, contains 414 samples for testing
- Irf [Garg et al., 2023]: Annotated dataset for interpersonal risk factors of mental disturbance, testset contains 2113 samples.

### C.2 Benchmarks for generalization

* MedMCQA: Biomedical question-answer pairs for multiple illnesses (Ophthalmology, Microbiology, Pediatrics, Anatomy, Medicine, Pathology, Skin, Psychiatry, ENT, Pharmacology) are tagged with "subject name."
* Evaluation conducted on subsets from the "dev" test set for 10 illnesses.
* MMLU: Multitask benchmark suite of 57 datasets across STEM, humanities, and social sciences domains.

### C.3 Examples

**Medication Counseling**:
- Input: An otherwise healthy 26-year-old man comes to the physician for medication counseling after recently being diagnosed with schizophrenia. Risperidone therapy is initiated.
- Options:
    - A: Agranulocytosis
    - B: Shortened QT interval
    - C: Gynecomastia
    - D: Weight loss
- Output: **Gynecomastia**

**Medication-Related Questions**:
- Input: Topical administration of I% Pilocarpine failed to produce pupillary constriction in a patient who had a large, dilated pupil.
- Options:
    - A: Adie's tonic pupils
    - B: Diabetic III nerve palsy
    - C: Pharmacological blockade
    - D: Uncal herniation
- Output: **Pharmacological blockade**

**Medical Queries**:
- Input: Are sleep spindles related to schizotypal personality traits and thalamic glutamine/glutamate in healthy subjects?
- Output: **Yes** (Without explanation)

**Clinical Decision Support**:
- Input: Consider this post: "The most heartbreaking thought I have is my mom’s thoughts when she knows her kid wants to die. She doesnt know how to help. I dont know how to help either tho… Some fuckin weed would be a good start tho lol"
- Question: What causes of depression does this post show?
- Output: This post shows causes of depression related to medication.
    - Reasoning: The post mentions the poster's feelings of wanting to die and their concern for their mother's thoughts. It also mentions that they don't know how to help themselves. The mention of weed as a possible solution suggests that the poster may be seeking some form of self-medication to cope with their mental state.

**Emotional Intelligence**:
- Input: Consider this post: "This is good and bad… I was a STEM major before and probably would have had a challenging and high paid career by now, had my life taken a different course. However, post PTSD, I found myself extremely connected with the arts. I have intense emotional reactions to aesthetic experiences, and I think that’s largely responsible for the fact that I now work in a creative industry. I experience aesthetics in a unique way that I'm not sure resonates with the typical non-affected individual."
- Question: Does the poster suffer from stress?
- Output: No, the poster does not suffer from stress.
    - Reasoning: The poster describes a shift in career choices from STEM to the arts due to their intense emotional reactions to aesthetic experiences. While they mention having PTSD, there is no indication in the post that they are currently experiencing stress or emotional distress related to it. The tone of the post is reflective and positive, focusing on the unique way they experience aesthetics rather than any negative impact on their well-being.

**Medication Side Effects**:
- Input: Consider this post: "[I’m not seeking any medical advice, just want to know if anyone has had similar experiences] I’ve just started taking Lexapro for moderate depression and anxiety. For the first week I was prescribed 5mg, 10mg thereafter. It’s been two days and I have this mild feeling like I’m on ecstasy/MDMA. My body feels light and I have this weird concentration of energy above my neck at the back of my head. Has anyone felt this/experienced this?"
- Output: No, this post does not show any risk of thwarted belongingness.
    - Reasoning: The individual in the post is seeking information and experiences from others who have taken Lexapro. They mention experiencing a mild feeling, but it is not related to feelings of disconnectedness or lack of belongingness. The focus of the post is on the side effects of the medication, rather than interpersonal factors or mental disturbance. Therefore, it does not show any risk of thwarted belongingness.

## Appendix D
### D.1 Perplexity

The perplexity (ppl) of a language model M on a test dataset D is calculated as follows: `ppl(D,M) = exp(-1 \* N \* sum(i=1 to N log P(xi|M))`, where N is the number of tokens in D, xi is the i-th token, and P(xi|M) is the probability of generating token xi given model M. Lower ppl values indicate better performance on test data, while higher ppl values when evaluating training data may provide more valuable information.

### D.2 Diversity Gain

To evaluate the extent of diversity our generated dataset brings to the base dataset, we use Diversity Gain [Bilmes, 2022]. The base dataset is defined as 𝒟b⁢a⁢s⁢e = {xi=(qi,ri,ai)}i=1N with N samples. The generated dataset is 𝒟n⁢e⁢w = {xi=(qi,ri,ai)}i=1M with M samples. Diversity gain (Dn⁢e⁢w) relative to Db⁢a⁢s⁢e is calculated as:

dg⁢a⁢i⁢n = 1/M \* ∑\_{xi∈𝒟n⁢e⁢w} min\_{xj∈𝒟b⁢a⁢s⁢e} (||f(xi) - f(xj)||),

where f is the feature extractor, and we use OpenAI Embedding API text-embedding-ada-002 to extract features.

## Appendix E Training data samples

* **Caption Reference** [Figure 8](https://arxiv.org/html/2410.06845v1#A5.F8): Examples of training data for MentalArena project
* **Caption Reference** [Figure 9](https://arxiv.org/html/2410.06845v1#A5.F9): Training data examples in ablation study setting (Baseline + c)

Both figures demonstrate various instances of training data associated with the MentalArena project, which includes self-play training for language models used to diagnose and treat mental health disorders:

* Figure 8 showcases standard training data examples.
* Figure 9 displays training data examples in an ablation study setting using Baseline + c configuration.

## Appendix F
### F.1 Examples

**Captioned Figures:**
- Figure 10: Example of cognitive model ([Link](https://arxiv.org/html/2410.06845v1#A6.F10))
- Figure 11: Example of behavior pattern ([Link](https://arxiv.org/html/2410.06845v1#A6.F11))

**Concise Description:**
These figures demonstrate a cognitive model and behavior pattern, respectively, which are part of the Symptom Encoder. ([Reference: MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders](https://arxiv.org/html/2410.06845v1))

### F.2 Introduction on cognitive model

**Cognitive Model: Component Description**

**1. Relevant History:**
- Significant past events that influence mental state

**2. Core Beliefs:**
- Deeply ingrained perceptions about self, others, world

**3. Intermediate Beliefs:**
- Underlying rules, attitudes, assumptions from core beliefs
- Shaping thought patterns

**4. Coping Strategies:**
- Techniques used to manage negative emotions

**Cognitive Model: Component Interactions**
1. Situation occurs
2. Automatic thoughts arise (immediate evaluative)
3. Thoughts derive from intermediate beliefs based on core beliefs
4. Emotions result from automatic thoughts
5. Behaviors ensue due to emotions

**Cognitive Model: Framework for Understanding Patients' Cognitive Processes**
- Identifying components and their interactions
- Provides comprehensive understanding of mental health disorders.

## Appendix G Detailed experimental results

**Iteration Results (Table 4)**
- **GPT-3.5-turbo**: MedQA - 64.16, MedMCQA - 33.76, PubMedQA - 44.68, CAMS - 28.96, dreaddit - 49.03, Irf - 64.65, avg - 47.54
- **Iteration 1**: MedQA - 72.83, MedMCQA - 46.18, PubMedQA - 70.12, CAMS - 32.64, dreaddit - 49.03, Irf - 64.65, avg - 55.91
- **Iteration 2 (Best)**: MedQA - 74.57, MedMCQA - 91.08, PubMedQA - 97.56, CAMS - 32.80, dreaddit - 49.03, Irf - 64.65, avg - 68.28
- **Iteration 3**: MedQA - 72.25, MedMCQA - 46.50, PubMedQA - 95.43, CAMS - 31.20, dreaddit - 49.03, Irf - 64.65, avg - 59.84
- **Iteration 4**: MedQA - 70.52, MedMCQA - 50.64, PubMedQA - 92.07, CAMS - 31.68, dreaddit - 49.03, Irf - 64.65, avg - 59.77
- **llama-3-8b**: MedQA - 70.52, MedMCQA - 42.04, PubMedQA - 86.59, CAMS - 25.12, dreaddit - 58.45, Irf - 45.76, avg - 54.75
- **Iteration 1 (Ours)**: MedQA - 76.88, MedMCQA - 48.09, PubMedQA - 89.33, CAMS - 27.20, dreaddit - 46.57, Irf - 57.91, avg - 57.91
- **Iteration 2 (Ours)**: MedQA - 76.88, MedMCQA - 48.41, PubMedQA - 89.63, CAMS - 28.48, dreaddit - 60.39, Irf - 55.67, avg - 58.24
- **Iteration 3 (Ours)**: MedQA - 77.46, MedMCQA - 49.04, PubMedQA - 92.38, CAMS - 28.64, dreaddit - 61.84, Irf - 56.24, avg - 59.27
- **Iteration 4 (Best)**: MedQA - 78.03, MedMCQA - 50.32, PubMedQA - 92.68, CAMS - 29.60, dreaddit - 65.46, Irf - 52.25, avg - 61.39
- **Iteration 5 (Ours)**: MedQA - 77.46, MedMCQA - 48.73, PubMedQA - 91.16, CAMS - 27.36, dreaddit - 65.46, Irf - 44.72, avg - 59.15
- **Iteration 6**: MedQA - 78.03, MedMCQA - 45.86, PubMedQA - 91.77, CAMS - 26.56, dreaddit - 61.11, Irf - 56.57, avg - 58.32

**Forgetting Experiments (Table 5)**
- **GPT-3.5-turbo**: dia (-10.59), cau (4), epi (-14), imp (60), log (-100), mov (-5.33), nav (0), pre (13), que (11.03), rui (11.03), sna (-2.78), spo (8), win (12), dyc (33), gen (30), lin (0), obj (47), ope (0), ten (92), ws (85), wu (97), avg (19.44)
- **Ours (GPT)**: dia (4.36), cau (6), epi (-14), imp (66), log (-100), mov (8), nav (6), pre (26.5), que (18.88), rui (2.56), sna (50), win (8), dyc (1), gen (1), lin (0), obj (56), ope (96), ten (87), ws (83), wu (100), avg (26.49)
- **llama**: dia (-4.61), cau (2), epi (-14), imp (14), log (-98), mov (0), nav (-2), pre (28), que (50.28), rui (-0.11), sna (24), win (8), dyc (1), gen (1), lin (0), obj (0), ope (80), ten (96), ws (83), wu (77), avg (17.93)
- **Ours (llama)**: dia (-0.12), cau (6), epi (-14), imp (28), log (-98), mov (2.67), nav (6), pre (25), que (52.9), rui (1.22), sna (36), win (8), dyc (12), gen (1), lin (0), obj (61), ope (81), ten (95), ws (83), wu (83), avg (21.08)

## Appendix H Training details

**Table 6: Llama-3-8b Fine-Tuning Epoch Numbers**

| iter    | nepochs |
|---------|---|
| 1       | 4      |
| 2       | 5      |
| 3       | 7      |
| 4       | 7      |

### H.1 Setup for GPT-3.5-turbo

**Fine-tuning Setting for GPT-3.5-turbo**:
- Iteration 1: Epoch number = 4
- Iteration 2: Epoch number = 6

### H.2 Setup for Llama-3-8b

**Fine-tuning Llama-3-8b with Lora:**

* lora_alpha: 16
* lora_dropout: 0.1
* r: 64
* bias: none
* task\_type: CAUSAL\_LM

For each iteration, the following settings remain constant:
- er\_device\_train\_batch\_size: 4
- gradient\_accumulation\_steps: 1
- optim: paged\_adamw\_32bit
- learning\_rate: 2e-4
- weight\_decay: 0.001
- fp16: False
- bf16: False
- max\_grad\_norm: 0.3
- max\_steps: -1
- warmup\_ratio: 0.03
- group\_by\_length: True
- lr\_scheduler\_type: constant
- report\_to: tensorboard

Training details for num\_train\_epochs can be found in Appendix H, Table 6 ([Link](https://arxiv.org/html/2410.06845v1#A8.T6 "MentalArena: Self-play Training of Language Models for Diagnosis and Treatment of Mental Health Disorders")).

## Appendix I Case study

**Case Study: MentalArena's Language Models for Mental Health Diagnosis and Treatment**

**Figures**
- **Figure 12**: Case study on Llama-3-8b (1) [Link](https://arxiv.org/html/2410.06845v1/x15.png)
- **Figure 13**: Case study on Llama-3-8b (2) [Link](https://arxiv.org/html/2410.06845v1/x16.png)
- **Figure 14**: Case study on Llama-3-8b (3) [Link](https://arxiv.org/html/2410.06845v1/x17.png)
- **Figure 15**: Case study on GPT-3.5-turbo (1) [Link](https://arxiv.org/html/2410.06845v1/x18.png)
- **Figure 16**: Case study on GPT-3.5-turbo (2) [Link](https://arxiv.org/html/2410.06845v1/x19.png)

**Findings:**
- Our models accurately answer medical questions during patient-therapist interactions, while baseline models provide incorrect responses.
- The discrepancy arises because the data generated during interactions contains valuable medical knowledge that aids in analysis and formulation of answers.

