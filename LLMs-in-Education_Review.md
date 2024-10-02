# LLMs in Education: Novel Perspectives, Challenges, and Opportunities

by Kaushal Kumar Maurya, Ekaterina Kochmar
https://arxiv.org/html/2409.11917v1

## Contents
- [LLMs in Education](#llms-in-education)
  - [Abstract](#abstract)
  - [1 Introduction](#1-introduction)
  - [2 Outline](#2-outline)
  - [3 Recommended Reading List](#3-recommended-reading-list)
  - [4 Target Audience](#4-target-audience)
  - [5 Tutorial Description](#5-tutorial-description)
  - [6 Diversity Considerations](#6-diversity-considerations)
- [Tutorial Reading List](#tutorial-reading-list)
  - [Readability and Simplification](#readability-and-simplification)
  - [Spoken Language Learning and Assessment](#spoken-language-learning-and-assessment)

## LLMs in Education
### Abstract

**Overview:**
- Role of LLMs in education is an increasing area of interest
- Offers new opportunities for teaching, learning, and assessment
- This tutorial provides an overview of educational applications of NLP
  - Impact of recent advances in LLMs on this field

**Discussion Topics:**
1. Key challenges and opportunities presented by LLMs in education
2. Four major educational applications:
   * Reading skills
   * Writing skills
   * Speaking skills
   * Intelligent tutoring systems (ITS)

**Audience:**
- Researchers and practitioners interested in the educational applications of NLP
- The first tutorial to address this timely topic at COLING 2025.

### 1 Introduction

**Large Language Models (LLMs)**
- **GPT-3.5 (ChatGPT)**: remarkable capabilities across various tasks Wei et al. (2022); Minaee et al. (2024)
- Rapid adoption by EdTech companies: Duolingo Naismith et al. (2023), Grammarly Raheja et al. (2023)
- Impact on educational applications research and development: enabling new opportunities in writing assistance, personalization, and interactive teaching and learning, among others

**Ethical Considerations in Integrating LLMs into Educational Settings**
- Paradigm shift in educational applications
- Present novel challenges regarding ethical considerations Bommasani et al. (2021)

**Key Topics Covered**
- Examining the challenges and opportunities presented by LLMs for educational applications through four key tasks:
  - Writing assistance
  - Personalization
  - Interactive teaching and learning

### 2 Outline

- **Impact of LLMs on**:
  - Writing assistance
  - Reading assistance
  - Spoken language learning and assessment
  - Development of Intelligent Tutoring Systems (ITS)

#### 2.2 LLMs for Writing Assistance

**Grammatical Error Correction (GEC) Tutorial**

**Overview:**
- Focuses on GEC for writing assistance
- Automatically detects and corrects errors in text
- Provides pedagogical benefits to L1 and L2 language teachers and learners: instant feedback, personalized learning
- Covers history, popular datasets (Yannakoudakis et al., 2011; Napoles et al., 2017; Náplava et al., 2022), evaluation methods (Bryant et al., 2023), and techniques: rule-based to sequence-to-sequence models

**LLMs for GEC:**
- Thorough overview on using Language Models (LLMs) for GEC
- Evaluates their performance in terms of fluency, coherence, and fixing various error types across popular benchmarks (Fang et al., 2023; Raheja et al., 2024; Katinskaia and Yangarber, 2024)
- Discusses prompting techniques and strategies for GEC, evaluating their effectiveness and limitations (Loem et al., 2023)
- Compares LLMs to supervised GEC approaches, examining strengths and weaknesses (Omelianchuk et al., 2024)
- Discusses using LLMs to evaluate GEC systems (Kobayashi et al., 2024)
- Provides insights into future directions: evaluation, usability, interpretability from a user-centric perspective.

#### 2.3 LLMs for Reading Assistance

- **Readability assessment:** important part of literacy; ability to read and comprehend text plays major role in critical thinking, effective communication
- Over two decades of NLP research on:
  - Readability assessment
  - Text simplification
- Focused on improving accessibility to written content

**Approaches:**
- Various methods explored in NLP research over time
- Challenges involved include maintaining context and preserving meaning while making adjustments

**Domain Specific Issues and Multilingual Approaches:**
- Research addressing domain-specific issues Garimella et al. (2021)
  - Medical terminology, technical jargon
- Advancements in multilingual text simplification Saggion et al. (2022)

**LLMs for Reading Support:**
- Arrival of LLMs led to new advances in readability assessment and text simplification Kew et al. (2023)
- Zero-shot usage, prompt tuning, fine-tuning Lee and Lee (2023); Tran et al. (2024)
  - Improving accuracy and relevance to individual users

**Current Limitations:**
- Limitations of current approaches Štajner (2021)
  - Difficulty in preserving context and meaning when making adjustments
- Focus on user-based evaluation Vajjala and Lučić (2019); Säuberli et al. (2024); Agrawal and Carpuat (2024)
  - Ensuring that simplifications are meaningful to the intended audience

**Future Directions:**
- Increased focus on user-based evaluation
- Supporting more languages Shardlow et al. (2024)
- Exploring new techniques for preserving context and meaning while simplifying text.

#### 2.4 LLMs for Spoken Language Learning and Assessment

**Speaking as a Crucial Language Skill**
- Speaking is a core language skill in language education curricula (Fulcher, 2000)
- Increasing interest in automated spoken language proficiency assessment

**Overview of Automated Spoken Language Assessment and Writing Assessment**
- **History**: overview of approaches to speaking assessment and their counterpart - automatic essay scoring (Burstein, 2002; Rudner et al., 2006; Landauer et al., 2002)
- **Breakthroughs**: first commercial systems for speech assessment (Townshend et al., 1998; Xi et al., 2008) and deep neural network approaches for writing (Alikaniotis et al., 2016) and speaking (Malinin et al., 2017) assessment
- **Applications**: analytic assessment, grammatical error detection (GEC), and correction (GED)

**Focus on LLMs for Spoken Language Assessment and Feedback**
- Use of text-based foundation models like BERT (Devlin et al., 2019) for holistic assessment (Craighead et al., 2020; Raina et al., 2020; Wang et al., 2021) and analytic assessment approaches (Bannò et al., 2024b)
- Use of speech foundation models like wav2vec 2.0 (Baevski et al., 2020), HuBERT (Hsu et al., 2021), and Whisper (Radford et al., 2022) for mispronunciation detection, pronunciation assessment (Kim et al., 2022), and holistic assessment (Bannò and Matassoni, 2023; Bannò et al., 2023)
- Addressing interpretability issues using analytic assessment approaches for writing
- Exploring opportunities for multimodal models like SALMONN (Tang et al., 2024) and Qwen Audio (Chu et al., 2023), as well as text-to-speech models like Bark (Schumacher et al., 2023) and Voicecraft (Peng et al., 2024) for assessment and learning.

#### 2.5 LLMs in Intelligent Tutoring Systems (ITS)

**Overview**:
- Computerized learning environments that provide personalized feedback based on learning progress
- Capable of providing one-on-one tutoring for equitable and effective learning experience
- Leads to substantial learning gains

**Lack of Individualized Tutoring**:
- Less effective learning and dissatisfaction in large classrooms

**Key Principles of Learning Sciences**:
- Goals for ITS development
- Outline: Pre-LLM ITS systems, including those tailored for misconception identification, model-tracing tutors, constraint-based models, Bayesian network models, and systems designed for specific knowledge areas

**LLMs in Intelligent Tutoring Systems Development**:
- Question solving
- Error correction
- Confusion resolution
- Question generation
- Content creation
- Simulating student interactions for teaching assistants and teacher training
- Assisting in creating datasets for fine-tuning LLMs
- Developing prompt-based techniques or modularized prompting for ITS development

**Future Directions**:
1. Development of standardized evaluation benchmarks to assess progress in ITS
2. Collection and creation of large public educational datasets for LLM training and fine-tuning
3. Development of specialized foundational LLMs for educational purposes
4. Investigations into long-term impact on students and teachers
5. Examining ethical considerations, potential biases, and pedagogical value in dialogue-based ITS.

### 3 Recommended Reading List
- Relevant papers cited in this proposal
- Available on the tutorial website

### 4 Target Audience
- Graduate students, researchers, and practitioners attending COLING 2025
- Background in Computational Linguistics (CL), NLP, or Machine Learning (ML)
- Interested in educational applications of NLP and generative AI
- Basic knowledge of educational technologies not required

### 5 Tutorial Description
- Self-contained and accessible to a wide audience
- Focuses on advanced technologies for various educational applications
- Covers recent advances brought by generative AI and LLMs
- Addresses opportunities, challenges, and risks in the field
- First tutorial on this topic at COLING or any other CL conference

### 6 Diversity Considerations
- Tutorial covers a wide range of applications
- Highlights opportunities to reach underrepresented groups
- Addresses fairness and accessibility challenges
- Instructors are diverse in gender, nationality, affiliation, and seniority
- Includes open Q&A sessions for participant engagement and discussion.

## Tutorial Reading List

[COLING 2025 Tutorial Additional Reading List](https://docs.google.com/document/d/1IlJldGFj3Sl6jEDwhwmXV9roo_7u5wXpfrKUCnIhaKY/edit)

### Datasets

1) [The CoNLL-2014 Shared Task on Grammatical Error Correction](https://aclanthology.org/W14-1701)  
2) [The First QALB Shared Task on Automatic Text Correction for Arabic](https://aclanthology.org/W14-3605)  
3) [The Second QALB Shared Task on Automatic Text Correction for Arabic](https://aclanthology.org/W15-3204)  
4) [The BEA-2019 Shared Task on Grammatical Error Correction](https://aclanthology.org/W19-4406)  
5) [Grammar Error Correction in Morphologically Rich Languages: The Case of Russian](https://aclanthology.org/Q19-1001)  
6) [Construction of an Evaluation Corpus for Grammatical Error Correction for Learners of Japanese as a Second Language](https://aclanthology.org/2020.lrec-1.26)

### Evaluation methods and their reliability

1) [Ground Truth for Grammatical Error Correction Metrics](https://aclanthology.org/P15-2097)  
2) [There's No Comparison: Reference-less Evaluation Metrics in Grammatical Error Correction](https://aclanthology.org/D16-1228)  
3) [Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction](https://aclanthology.org/P17-1074)  
4) [Reference-based Metrics can be Replaced with Reference-less Metrics in Evaluating Grammatical Error Correction Systems](https://aclanthology.org/I17-2058)  
5) [Classifying Syntactic Errors in Learner Language](https://aclanthology.org/2020.conll-1.7)  
6) [IMPARA: Impact-Based Metric for {GEC} Using Parallel Data](https://aclanthology.org/2022.coling-1.316)  
7) [Reassessing the Goals of Grammatical Error Correction: Fluency Instead of Grammaticality](https://aclanthology.org/Q16-1013)  
8) [Inherent Biases in Reference-based Evaluation for Grammatical Error Correction](https://aclanthology.org/P18-1059)

### Methods: Statistical and rule-based

1) [Detection of Grammatical Errors Involving Prepositions](https://aclanthology.org/W07-1604)  
2) [The Ups and Downs of Preposition Error Detection in ESL Writing](https://aclanthology.org/C08-1109)  
3) [Grammatical Error Correction with Alternating Structure Optimization](https://aclanthology.org/P11-1092)  
4) [Joint Learning and Inference for Grammatical Error Correction](https://aclanthology.org/D13-1074)  
5) [Generalized Character-Level Spelling Error Correction](https://aclanthology.org/P14-2027)  
6) [Grammatical error correction using hybrid systems and type filtering](https://aclanthology.org/W14-1702)  
7) [The AMU System in the CoNLL-2014 Shared Task: Grammatical Error Correction by Data-Intensive and Feature-Rich Statistical Machine Translation](https://aclanthology.org/W14-1703)  
8) [Phrase-based Machine Translation is State-of-the-Art for Automatic Grammatical Error Correction](https://aclanthology.org/D16-1161)

### Methods: sequence-to-sequence

1) [Grammatical error correction using neural machine translation](https://aclanthology.org/N16-1042)  
2) [Approaching Neural Grammatical Error Correction as a Low-Resource Machine Translation Task](https://aclanthology.org/N18-1055)  
3) [Utilizing Character and Word Embeddings for Text Normalization with Sequence-to-Sequence Models](https://aclanthology.org/D18-1097)  
4) [Neural and FST-based approaches to grammatical error correction](https://aclanthology.org/W19-4424)  
5) [Improving Grammatical Error Correction via Pre-Training a Copy-Augmented Architecture with Unlabeled Data](https://aclanthology.org/N19-1014)  
6) [Neural Grammatical Error Correction Systems with Unsupervised Pre-training on Synthetic Data](https://aclanthology.org/W19-4427)  
7) [Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model](https://aclanthology.org/2020.aacl-main.83)  
8) [Document-level grammatical error correction](https://aclanthology.org/2021.bea-1.8)

### Methods: text-editing neural models

1) [Parallel Iterative Edit Models for Local Sequence Transduction](https://aclanthology.org/D19-1435)  
2) [Encode, Tag, Realize: High-Precision Text Editing](https://aclanthology.org/D19-1510)  
3) [Seq2Edits: Sequence Transduction Using Span-level Edit Operations](https://aclanthology.org/2020.emnlp-main.418)  
4) [FELIX: Flexible Text Editing Through Tagging and Insertion](https://aclanthology.org/2020.findings-emnlp.111)  
5) [Character Transformations for Non-Autoregressive {GEC} Tagging](https://aclanthology.org/2021.wnut-1.46)  
6) [EdiT5: Semi-Autoregressive Text Editing with T5 Warm-Start](https://aclanthology.org/2022.findings-emnlp.156)  
7) [An Extended Sequence Tagging Vocabulary for Grammatical Error Correction](https://aclanthology.org/2023.findings-eacl.119)

### LLMs for GEC

1) [Is ChatGPT a Highly Fluent Grammatical Error Correction System? A Comprehensive Evaluation](https://arxiv.org/pdf/2304.01746)   
2) [Analyzing the Performance of GPT-3.5 and GPT-4 in Grammatical Error Correction](https://arxiv.org/pdf/2303.14342)   
3) [ChatGPT or Grammarly? Evaluating ChatGPT on Grammatical Error Correction Benchmark](https://arxiv.org/pdf/2303.13648)  
4) [Prompting open-source and commercial language models for grammatical error correction of English learner text](https://arxiv.org/pdf/2401.07702)   
5) [MEDIT: Multilingual Text Editing via Instruction Tuning](https://aclanthology.org/2024.naacl-long.56.pdf)   
6) [GPT-3.5 for Grammatical Error Correction](https://aclanthology.org/2024.lrec-main.692.pdf)   
7) [Exploring Effectiveness of GPT-3 in Grammatical Error Correction: A Study on Performance and Controllability in Prompt-Based Methods](https://aclanthology.org/2023.bea-1.18.pdf)   
8) [Pillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of Large Language Models](https://aclanthology.org/2024.bea-1.3.pdf) 

### LLMs as GEC Evaluators / Explainors

1) [Large Language Models Are State-of-the-Art Evaluator for Grammatical Error Correction](https://aclanthology.org/2024.bea-1.6.pdf)   
2) [GMEG-EXP: A Dataset of Human- and LLM-Generated Explanations of Grammatical and Fluency Edits](https://aclanthology.org/2024.lrec-main.688.pdf)   
3) [Controlled Generation with Prompt Insertion for Natural Language Explanations in Grammatical Error Correction](https://aclanthology.org/2024.lrec-main.350v2.pdf) 

### Recent papers

1) [Towards Automated Document Revision: Grammatical Error Correction, Fluency Edits, and Beyond](https://aclanthology.org/2024.bea-1.21.pdf)   
2) [Read, Revise, Repeat: A System Demonstration for Human-in-the-loop Iterative Text Revision](https://aclanthology.org/2022.in2writing-1.14.pdf)   
3) [Improving Iterative Text Revision by Learning Where to Edit from Other Revision Tasks](https://aclanthology.org/2022.emnlp-main.678.pdf)   
4) [Understanding Iterative Revision from Human-Written Text](https://aclanthology.org/2022.acl-long.250.pdf)

### Ethical considerations

1) [Unraveling Downstream Gender Bias from Large Language Models: A Study on AI Educational Writing Assistance](https://aclanthology.org/2023.findings-emnlp.689.pdf) 

### Readability and Simplification

#### Surveys

1) [Computational assessment of text readability: A survey of current and future research](https://www.jbe-platform.com/content/journals/10.1075/itl.165.2.01col)  
2) [Trends, limitations and open challenges in automatic readability assessment research](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.574.pdf)  
3) [A survey of research on text simplification](https://www.jbe-platform.com/content/journals/10.1075/itl.165.2.06sid)  
4) [Data-Driven Sentence Simplification: Survey and Benchmark](https://direct.mit.edu/coli/article/46/1/135/93384/Data-Driven-Sentence-Simplification-Survey-and)

**Shared Tasks**

- [SemEval-2012 Task 1: English Lexical Simplification](https://aclanthology.org/S12-1046/)  
- [SemEval 2016 Task 11: Complex Word Identification](https://aclanthology.org/S16-1085/)  
- [SemEval-2021 Task 1: Lexical Complexity Prediction](https://aclanthology.org/2021.semeval-1.1/)  
- [The BEA 2024 Shared Task on the Multilingual Lexical Simplification Pipeline](https://aclanthology.org/2024.bea-1.51/) 

#### Methods

1) [The Principles of Readability](https://eric.ed.gov/?id=ed490073)  
2) [Do NLP and machine learning improve traditional readability formulas?](https://aclanthology.org/W12-2207.pdf)  
3) [Multiattentive Recurrent Neural Network Architecture for Multilingual Readability Assessment](https://aclanthology.org/Q19-1028/)  
4) [Text readability assessment for second language learners](https://aclanthology.org/W16-0502.pdf)  
5) [Exploring hybrid approaches to readability: experiments on the complementarity between linguistic features and transformers](https://aclanthology.org/2024.findings-eacl.153.pdf)  
6) [Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features](https://aclanthology.org/2021.emnlp-main.834/)  
7) [Automatic induction of rules for text simplification](https://www.sciencedirect.com/science/article/pii/S0950705197000294)  
8) [Learning to simplify sentences with quasi-synchronous grammar and integer programming](https://aclanthology.org/D11-1038.pdf)  
9) [Optimizing statistical machine translation for text simplification](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00107/43364/Optimizing-Statistical-Machine-Translation-for)  
10) [Learning to Paraphrase Sentences to Different Complexity Levels](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00606/118113)  
11) [Elaborative Simplification for German-language Texts](https://aclanthology.org/2024.sigdial-1.3/)  
12) [Supervised and Unsupervised Neural Approaches to Text Readability](https://aclanthology.org/2021.cl-1.6/)   
13) [All Mixed Up? Finding the Optimal Feature Set for General Readability Prediction and Its Application to English and Dutch](https://aclanthology.org/J16-3004/)  

#### LLMs

1) [Prompt-based Learning for Text Readability Assessment](https://aclanthology.org/2023.findings-eacl.135/)  
2) [FPT: Feature Prompt Tuning for Few-shot Readability Assessment](https://aclanthology.org/2024.naacl-long.16/)  
3) [Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts](https://aclanthology.org/2024.bea-1.5/)   
4) [BLESS: Benchmarking Large Language Models on Sentence Simplification](https://aclanthology.org/2023.emnlp-main.821/)  
5) [An LLM-Enhanced Adversarial Editing System for Lexical Simplification](https://aclanthology.org/2024.lrec-main.102/)   
6) [On Simplification of Discharge Summaries in Serbian: Facing the Challenges](https://aclanthology.org/2024.cl4health-1.12/)  
   
#### Evaluation 
1) [Towards grounding computational linguistic approaches to readability: Modeling reader-text interaction for easy and difficult texts](https://aclanthology.org/W16-4105.pdf)  
2) [Are Cohesive Features Relevant for Text Readability Evaluation?](https://aclanthology.org/C16-1094.pdf)   
3) [On understanding the relation between expert annotations of text readability and target reader comprehension](https://aclanthology.org/W19-4437.pdf)   
4) [Linguistic Corpus Annotation for Automatic Text Simplification Evaluation](https://aclanthology.org/2022.emnlp-main.121.pdf)  
5) [The (Un)Suitability of Automatic Evaluation Metrics for Text Simplification](https://direct.mit.edu/coli/article/47/4/861/106930/The-Un-Suitability-of-Automatic-Evaluation-Metrics)   
6) I[nvestigating Text Simplification Evaluation](https://aclanthology.org/2021.findings-acl.77.pdf)   
7) [Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Reading Comprehension](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00653/120649/Do-Text-Simplification-Systems-Preserve-Meaning-A) 

#### Explainability:
1) [Explainable AI in Language Learning: Linking Empirical Evidence and Theoretical Concepts in Proficiency and Readability Modeling of Portuguese](https://aclanthology.org/2024.bea-1.17/)   
2) [“Geen makkie”: Interpretable Classification and Simplification of Dutch Text Complexity](https://aclanthology.org/2023.bea-1.42/) 

#### Broader impact/Ethical issues 
3) [Automatic text simplification for social good: Progress and challenges](https://aclanthology.org/2021.findings-acl.233.pdf)   
4) [When readability meets computational linguistics: a new paradigm in readability](https://www.cairn.info/revue-francaise-de-linguistique-appliquee-2015-2-page-79.htm)   
5) [Problems in Current Text Simplification Research: New Data Can Help](https://aclanthology.org/Q15-1021/)  

### Spoken Language Learning and Assessment

#### Automated Speaking Assessment

1) [The use of DBN-HMMs for mispronunciation detection and diagnosis in {L2 English} to support computer-aided pronunciation training](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/xiaojun_interspeech2012.pdf)  
2) [Improvements to an Automated Content Scoring System for Spoken CALL Responses: the ETS Submission to the Second Spoken CALL Shared Task](http://vikramr.com/pubs/CALL_task_IS2018.pdf)  
3) [Automated Speaking Assessment: Using Language Technologies to Score Spontaneous Speech](https://www.routledge.com/Automated-Speaking-Assessment-Using-Language-Technologies-to-Score-Spontaneous-Speech/Zechner-Evanini/p/book/9781138056879)  
4) [Incorporating uncertainty into deep learning for spoken language assessment](https://aclanthology.org/P17-2008.pdf)  
5) [Automated scoring of spontaneous speech from young learners of English using transformers](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9383553)  
6) [Automatic pronunciation assessment using self-supervised speech representation learning](https://www.isca-archive.org/interspeech_2022/kim22k_interspeech.pdf)  
7) [View-Specific Assessment of L2 Spoken English](https://www.isca-archive.org/interspeech_2022/banno22_interspeech.pdf)  
8) [Proficiency assessment of L2 spoken English using wav2vec 2.0](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10023019)  
9) [Can GPT-4 do L2 analytic assessment?](https://aclanthology.org/2024.bea-1.14.pdf)

#### Spoken GED and GEC

1) [Automatic error detection in the Japanese learners’ English spoken data](https://aclanthology.org/P03-2026.pdf)  
2) [Impact of ASR Performance on Spoken Grammatical Error Detection](https://www.isca-archive.org/interspeech_2019/lu19b_interspeech.pdf)  
3) [Automatic Grammatical Error Detection of Non-Native Spoken Learner English](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683080)  
4) [On Assessing and Developing Spoken ‘Grammatical Error Correction’ Systems](https://aclanthology.org/2022.bea-1.9.pdf)  
5) [Towards End-to-End Spoken Grammatical Error Correction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446782)

#### LLMs for Speaking Assessment and Feedback

1) [Transformer Based End-to-End Mispronunciation Detection and Diagnosis](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/2%20wu21h_interspeech_hccl.pdf)  
2) [Explore wav2vec 2.0 for Mispronunciation Detection](https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.pdf)  
3) [Automatic Assessment of Conversational Speaking Tests](https://www.isca-archive.org/slate_2023/mcknight23_slate.pdf)

#### Validity and reliability

1) [Assessing L2 English speaking using automated scoring technology: examining automaker reliability](https://www.tandfonline.com/doi/pdf/10.1080/0969594X.2021.1979467)

#### LLM in STEM Education and ITS

*People have valued and thought deeply about education for a long time* - John Dewey, 1923

#### Overview: Opportunities and Challenges

1) [Opportunities and Challenges in Neural Dialog Tutoring](https://arxiv.org/pdf/2301.09919)  
2) [Are We There Yet? \- A Systematic Literature Review on Chatbots in Education](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.654924/full)  
3) [A Systematic Literature Review of Intelligent Tutoring Systems With Dialogue in Natural Language](https://oa.upm.es/78939/1/A_Systematic_Literature_Review_of_Intelligent_Tutoring_Systems_With_Dialogue_in_Natural_Language.pdf)  
   
#### Pre-LLM ITS
1) [AutoTutor 3-D Simulations: Analyzing Users' Actions and Learning Trends](https://escholarship.org/content/qt7c99p489/qt7c99p489.pdf)  
2) [Gaze tutor: A gaze-reactive intelligent tutoring system](https://www.sciencedirect.com/science/article/pii/S1071581912000250?casa_token=VmKBp0U6OcMAAAAA:XHbs2ml6Tv-hy-z6lHEbxFdjqEOncfJ-5GT4Y1tmC6NUl8ACzhpjlQPK6ie4JXFc6x5g581WLg)  
3) [Interactive Conceptual Tutoring in Atlas-Andes](https://cdn.aaai.org/Symposia/Fall/2000/FS-00-01/FS00-01-023.pdf)  
4) [Individualizing self-explanation support for ill-defined tasks in constraint-based tutors](https://ir.canterbury.ac.nz/server/api/core/bitstreams/64c8724a-9436-45fa-8fbb-85efe9da583a/content)  
5) [Jacob-An animated instruction agent in virtual reality](https://www.researchgate.net/profile/Anton-Nijholt/publication/2590439_Jacob_-_An_Animated_Instruction_Agent_in_Virtual_Reality/links/5d1209e9299bf1547c7cb394/Jacob-An-Animated-Instruction-Agent-in-Virtual-Reality.pdf)  
6) [Data mining in education](https://wires.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/widm.1075?casa_token=QNGFMpDgGjwAAAAA%3AgMdWKEinB_766fdUlQ1NgE7LIVy6Gpy0VasQqSJLEqJeqirNA4YKAZ2Z2F8kmCPoj5TBoHtSdniptDs)  
   
#### LLM in STEM Education and ITS
1) [Stepwise Verification and Remediation of Student Reasoning Errors with Large Language Model Tutors](https://arxiv.org/pdf/2407.09136)  
2) [Backtracing: Retrieving the Cause of the Query](https://aclanthology.org/2024.findings-eacl.48/)  
3) [MATHDIAL: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems](https://arxiv.org/pdf/2305.14536.pdf)  
4) [Improving Teachers’ Questioning Quality through Automated Feedback: A Mixed-Methods Randomized Controlled Trial in Brick-and-Mortar Classrooms](https://edworkingpapers.com/sites/default/files/ai23-875.pdf)  
5) [Bridging the Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math Mistakes](https://arxiv.org/abs/2310.10648)  
6) [GPTeach: Interactive TA Training with GPT-based Students](https://dl.acm.org/doi/abs/10.1145/3573051.3593393)  
7) [NAISTeacher: A Prompt and Rerank Approach to Generating Teacher Utterances in Educational Dialogues](https://aclanthology.org/2023.bea-1.63.pdf)  
8) [Is ChatGPT a Good Teacher Coach? Measuring Zero-Shot Performance For Scoring and Providing Actionable Insights on Classroom Instruction](https://arxiv.org/abs/2306.03090)  
9) [Demographic predictors of students’ science participation over the age of 16: An Australian case study](https://link.springer.com/article/10.1007/s11165-018-9692-0)  

#### Evaluation of ITS
1)  [The AI Teacher Test: Measuring the Pedagogical Ability of Blender and GPT-3 in Educational Dialogues](https://arxiv.org/pdf/2205.07540.pdf)  
2)  [Measuring Conversational Uptake: A Case Study on Student-Teacher Interactions](https://arxiv.org/pdf/2106.03873.pdf)  
3) [Evaluation Methodologies for Intelligent Tutoring Systems](https://d1wqtxts1xzle7.cloudfront.net/3624370/10.1.1.52.6842-libre.pdf?1390834477=&response-content-disposition=inline%3B+filename%3DEvaluation_methodologies_for_intelligent.pdf&Expires=1721892588&Signature=KYUgkfFfaGBNbKjSKJcIz8lLNGc~Urwx125tpD5BFXzwx9jsiHkuoSeqyXccjpQ4fYD9J~RZoIsetjY-reSZYTDSyWMpTkFlCGdGBJ8E6hMPHsKEZIr1xy2Ex6~cuAqVwmxEXBnzka61eBffPXT13V088bNFDKdWgPSivFTjquxxtde4pA1u6qaLqsOYDmV8M0T7nUdjzLyxj6JLniQ0P2LG~UTnJDjvOdnIzoZRzG57zTSR-ytj180UxmyzeNHz~KGpSHcX0WuZnHg~xNTXt-ZY2Op-5RK5XTZyEKyIlAQiyGKHUHTI~VfDqsex4~Ku5y3Y9Bwzg1edPQbx6HXVwQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

#### Ethical Considerations

1) [What if the devil is my guardian angel: ChatGPT as a case study of using chatbots in education](https://link.springer.com/content/pdf/10.1186/s40561-023-00237-x.pdf)
