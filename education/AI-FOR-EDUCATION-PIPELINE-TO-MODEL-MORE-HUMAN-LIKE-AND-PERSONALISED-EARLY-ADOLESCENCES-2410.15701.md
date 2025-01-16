# STUDENTS RATHER THAN EXPERTS: A NEW AI FOR EDUCATION PIPELINE TO MODEL MORE HUMAN-LIKE AND PERSONALISED EARLY ADOLESCENCES

authors: Yiping Ma, Shiyu Hu, Xuchen Li

paper: https://arxiv.org/pdf/2410.15701

video: https://www.youtube.com/watch?v=fr_zUWaKDF8

code: https://marsgemini.github.io/SOE-LVSA/

## Contents
- [Abstract](#abstract)
- [1 INTRODUCTION](#1-introduction)
- [2 RELATED WORK](#2-related-work)
- [3 SCENE: BASIC CHINESE UNDERSTANDING ABILITY IN EDUCATION](#3-scene-basic-chinese-understanding-ability-in-education)
  - [3.1 BASIC CHINESE UNDERSTANDING ABILITY EVALUATION DATASET](#31-basic-chinese-understanding-ability-evaluation-dataset)
  - [3.2 FOUNDATION MODELS](#32-foundation-models)
- [4 OBJECT: VIRTUAL STUDENT CONSTRUCTION. 4.1 THEORETICAL FRAMEWORK](#4-object-virtual-student-construction-41-theoretical-framework)
  - [4.2 FINE-TUNING DATASET](#42-fine-tuning-dataset)
  - [4.3 FINE-TUNING THE FOUNDATION MODELS](#43-fine-tuning-the-foundation-models)
- [5 EVALUATION: ANALYSIS OF OBJECT CAPABILITIES IN SCENES](#5-evaluation-analysis-of-object-capabilities-in-scenes)
  - [5.1 SUBJECTIVE EVALUATION DATASET](#51-subjective-evaluation-dataset)
  - [5.2 HUMAN TURING TEST](#52-human-turing-test)
  - [5.3 HUMAN-GPT4 COMPARISON VALIDATION](#53-human-gpt4-comparison-validation)
  - [5.4 LARGE-SCALE EVALUATION EXPERIMENT WITH GPT-4](#54-large-scale-evaluation-experiment-with-gpt-4)
  - [5.5 BAD CASE ANALYSIS](#55-bad-case-analysis)
- [6 CONCLUSION](#6-conclusion)
- [A COMPREHENSIVE RELATED WORK](#a-comprehensive-related-work)
  - [A.1 VIRTUAL STUDENT DEVELOPMENT AND RELATED TECHNOLOGIES](#a1-virtual-student-development-and-related-technologies)
  - [A.2 LLMS’ MAIN CAPABILITIES TO SUPPORT AI4EDUCATION](#a2-llms-main-capabilities-to-support-ai4education)
  - [A.3 REPRESENTATIVE AI4EDUCATION APPLICATIONS](#a3-representative-ai4education-applications)
- [B DETAILED INFORMATION FOR THEORY AND TECHNOLOGY USED IN THIS WORK](#b-detailed-information-for-theory-and-technology-used-in-this-work)
  - [B.1 FOUNDATION LLMS](#b1-foundation-llms)
  - [B.2 EDUCATIONAL THEORY](#b2-educational-theory)
  - [B.2.2 OPERATIONAL THEORY: CLASSIFICATION OF TEACHER-STUDENT DIALOGUE DATASETS](#b22-operational-theory-classification-of-teacher-student-dialogue-datasets)
  - [B.2.3 THE UNIQUE VALUE OF MIDDLE SCHOOL STUDENTS AND THE LANGUAGE-BASED SUBJECT IN THIS STUDY](#b23-the-unique-value-of-middle-school-students-and-the-language-based-subject-in-this-study)
- [C DETAILED INFORMATION FOR DATASET CONSTRUCTION PROCESS](#c-detailed-information-for-dataset-construction-process)
  - [C.1 BASIC CHINESE UNDERSTANDING ABILITY EVALUATION DATASET](#c1-basic-chinese-understanding-ability-evaluation-dataset)
  - [C.2 FINE-TUNING DATASET](#c2-fine-tuning-dataset)
  - [C.3 SUBJECTIVE EVALUATION DATASET](#c3-subjective-evaluation-dataset)
- [D DETAILED EXPERIMENT SETTINGS AND RESULTS](#d-detailed-experiment-settings-and-results)
  - [D.1 FINE-TUNING CONFIGURATION](#d1-fine-tuning-configuration)
  - [D.2 FINE-TUNING RESULTS](#d2-fine-tuning-results)
  - [D.3 HUMAN TURING EVALUATION QUESTIONNAIRE](#d3-human-turing-evaluation-questionnaire)
  - [D.4 HUMAN TURING EVALUATION RESULTS](#d4-human-turing-evaluation-results)
  - [D.5 GPT-4 LARGE-SCALE EVALUATION PROMPT](#d5-gpt-4-large-scale-evaluation-prompt)
  - [D.6 BAD CASE EXAMPLES](#d6-bad-case-examples)

## Abstract 
**AI in Education (AI4Education)**
* Large language models (LLMs) used for educational applications
* Focus on simulating teachers to enhance student learning outcomes
* Neglected potential of modeling virtual students
* Challenges: replicating learning difficulties, emotional responses, linguistic uncertainties

**Proposed Framework: SOE (Scene - Object - Evaluation)**
* Systematic construction of LLM-based Virtual Student Agents (LVSA)
* Curated dataset of personalized teacher-student interactions
* Fine-tuning LLMs using LoRA

**Study Objectives:**
1. Generate LVSA framework
2. Integrate human evaluation metrics into GPT-4 assessments
3. Validate LLMs for generating human-like, personalized virtual student agents

**Methodology:**
* Curating dataset of teacher-student interactions
* Fine-tuning LLMs using LoRA
* Multi-dimensional evaluation experiments
1. Develop theoretical framework for LVSA generation
2. Human evaluation of LVSA authenticity
3. Validate LLMs in educational contexts

**Significance:**
* Foundation for future applications in pre-service teacher training and multi-agent simulation environments.

## 1 INTRODUCTION

**Introductions:**
- Recent advancements in LLMs have led to breakthroughs in NLP and knowledge generation, enabling AI-driven educational applications (Ge et al., 2024; Liu et al., 2024b; Hong et al., 2024; Latif et al., 2023; Chen et al., 2024a; Jury et al., 2024).
- LLMs have the potential to be virtual tutors in generating high-quality content and personalized recommendations (Lee et al.; Wang et al., 2024).

**Challenges:**
- Traditional pre-service teacher training faces constraints like limited environments and insufficient practical experience (Loewenberg Ball & Forzani, 2009; Zeichner, 2012; Ward et al., 2018; Sparks, 2011).
- Virtual students offer cost-effective, flexible practice opportunities but often have predictable behaviors that fail to capture the complexity of human emotional responses and cognitive variability (Badiee & Kaufman, 2015).
- Existing models lack realistic virtual student modeling, raising a critical question: "Can LLMs create human-like, personalized virtual student agents for teacher training?"

**Framework:**
- The proposed "Scene - Object - Evaluation" (SOE) framework addresses the challenges of modeling and evaluating LLM-based virtual student agents (LVSA).

**Contributions:**
1. Theoretical framework for LVSA: A comprehensive framework for constructing virtual student agents with scientific rigor and feasibility.
2. Subjective evaluation metrics integration: Incorporating human subjective metrics into GPT-4’s evaluation pipeline, aligning with human assessments of virtual student authenticity.
3. LVSA validation: Extensive multi-dimensional experiments confirming the feasibility of creating human-like, personalized LVSA for educational AI.

**Conclusion:**
The SOE framework provides a robust foundation for applications in pre-service teacher training, multi-agent simulations, and educational AI systems.

## 2 RELATED WORK

**Virtual Student Development and Related Technologies**

**Three Approaches to Virtual Students:**
- **Role-playing**: Realistic but lacks scalability and flexibility due to resource constraints (Kersh, 1963; Colwell, 2013; Frederick et al., 2010; Shapira-Lishchinsky, 2015; Dalgarno et al., 2016).
- **Programmatically pre-set models**: Provide predictable responses but lack adaptability to real-world classroom dynamics (Christensen et al., 2011; Delamarre et al., 2021; Shernoff et al., 2018; Kelleci & Aksoy, 2021).
- **LLM-based virtual students**: Offer greater flexibility in natural language interactions (Achiam et al., 2023; Li et al., 2024c; Zhang et al., 2024; Lee et al.; Markel et al., 2023).

**LLMs' Capabilities for AI4Education:**
- Enhanced natural language processing abilities: Understanding, generation, reasoning, transfer learning, and multimodal processing (Brown, 2020; Achiam et al., 2023; Guo et al., 2023; Li et al., 2024b; Huang et al., 2024; Ahuja et al., 2023; Kafle & Kanan, 2017).
- Models like GPT-4, LLaMA, and PaLM demonstrate remarkable performance (Touvron et al., 2023; Chowdhery et al., 2023; Wang et al., 2019; Clark et al., 2018).
- Struggle to replicate nuanced cognitive and emotional behaviors of real students due to focus on knowledge delivery (Achiam et al., 2023; Li et al., 2024c; Zhang et al., 2024; Lee et al.; Markel et al., 2023).

**Representative AI4Education Applications:**
- Duolingo Max, Khan Academy’s Khanmigo, and TAL’s MathGPT: LLM-based products to enhance student learning (Duolingo, 2023; Khan, 2023; TAL, 2023a; squirrelai, 2023; Chen et al., 2021).
- C-Eval, SciQAG, and Mukea: Benchmark datasets and evaluation tools for LLMs’ effectiveness in education (Huang et al., 2024; Wan et al., 2024; Ding et al., 2022).

**Gap in Teacher Development Support:**
- Most research prioritizes student learning over enhancing teachers' pedagogical skills (Lee et al.; Wang et al., 2024).
- LLM applications focus on automating educational tasks, leaving a gap in teacher development support (García-Méndez et al., 2024; Yue et al., 2024; McNichols et al., 2023; Sahai et al., 2023; Han et al., 2023).

**Study's Novel Approach:**
- Leverages virtual students to enhance teachers’ skills through realistic, interactive environments (for more details, see Appendix A.3).

## 3 SCENE: BASIC CHINESE UNDERSTANDING ABILITY IN EDUCATION

**Study Overview:**
- Explores LLMs' capabilities in simulating realistic student behavior for early adolescents (ages 10-15)
- Critical developmental phase: transition from concrete to abstract thinking, with distinct cognitive and linguistic challenges
- Focus on junior high school Chinese education due to emphasis on expression, emotional experience, and value formation

**Research Questions:**
- Do foundation models possess basic comprehension abilities for middle school language subjects?
- Can LLMs accurately replicate the learning behaviors of early adolescents in real classroom settings?

**Assessment:**
- Evaluate foundation models' Chinese comprehension abilities to determine their potential.

### 3.1 BASIC CHINESE UNDERSTANDING ABILITY EVALUATION DATASET

**Basic Chinese Understanding Ability Evaluation Dataset**

**Data Source**: National Smart Education Platform (developed by the Chinese Ministry of Education)

**Focus Skills**:
- Text comprehension
- Memorization

**Purpose**: Assessing junior high school Chinese language proficiency

**Construction Process**:
1. **Data Preparation**: Organizing Chinese exercises into units and converting them to PDF format for model input
2. **Prompt Design**: Using structured prompts to guide model tasks
3. **Expert Revision**: Junior high school teachers reviewed and refined model outputs for accuracy

**Collected Items**: 613 comprehension items, 438 memorization tasks

**Detailed Information**: Provided in Appendix C.1

### 3.2 FOUNDATION MODELS

**Section 3: Foundation Models and Performance Evaluation**

**Foundation Models**:
- InternVL: integrates visual and linguistic information for cross-modal reasoning
- LLaVa: balances efficiency with strong language understanding for medium-scale tasks
- MiniCPM: optimized for Chinese tasks, excels in comprehension and recitation
- Qwen: performs well in visual-linguistic and text-based contexts, offers flexibility for multimodal applications

**Evaluation of Foundation Models**:
- Performance on junior high school Chinese language tasks (text comprehension and memorization)
- **InternVL**: 
  - Comprehension accuracy: 0.736
  - Memorization accuracy: 0.758
- **LLaVa**: 
  - Comprehension accuracy: 0.491
  - Memorization accuracy: 0.397
- **MiniCPM**: 
  - Comprehension accuracy: 0.664
  - Memorization accuracy: 0.584
- **Qwen**: 
  - Comprehension accuracy: 0.584
  - Memorization accuracy: 0.614

**Overall Performance and Comparison**:
- **InternVL** achieved the highest average accuracy (0.747)
- **MiniCPM** followed with an average accuracy of 0.700
- **Qwen** and **LLaVa** had lower average accuracies: Qwen (0.599), LLaVa (0.444)
- The lower performance of **Qwen** and **LLaVa** may be due to their design focus on multimodal tasks, limiting their effectiveness in Chinese language processing

**Conclusion**:
- **InternVL** and **MiniCPM** are more suitable for junior high school Chinese tasks
- **Qwen** and **LLaVa** may have broader applications in multimodal educational contexts.

## 4 OBJECT: VIRTUAL STUDENT CONSTRUCTION. 4.1 THEORETICAL FRAMEWORK

**Theoretical Framework for Virtual Student Construction**

**Conceptual Theory:**
- Early adolescence (ages 10-15): significant physiological, cognitive, social-emotional, and moral-spiritual changes
- Rapid prefrontal cortex development leads to intense emotions, inconsistent behaviors
- Cognitive transition from concrete to abstract reasoning, influenced by emotions
- Heightened self-awareness and sensitivity to peer feedback impact engagement
- Moral development involves internalizing societal norms, leading to simplified moral judgments

**Operational Theory:**
- Generate dialogues that are operational and predictable for realistic simulations of student performance
- Practical dimensions: question-answer types, personality traits, learning stages, response styles, generation sources

**Question-Answer Types:**
- Simulate cognitive diversity in virtual students

**Personality Traits:**
- Capture individual differences using the Big Five Personality Traits

**Learning Stages:**
- Reflect different cognitive abilities

**Response Styles:**
- Represent individual language differences

**Generation Source:**
- Combines techniques like Few-shot In-Context Learning (Few-shot ICL) and Chain of Thought (CoT) with fine-tuning on classroom data to maintain response authenticity.

**Figure 3:**
- Theoretical framework for virtual student construction, extending from conceptual theory to operational theory.

### 4.2 FINE-TUNING DATASET

**Fine-Tuning Dataset**

**Data Source**:
- Aligned with Basic Chinese Understanding Ability Evaluation Dataset described in Section 3.1
- Incorporates data from:
    - Real classroom video recordings
    - Textbook content
    - Teacher-prepared lesson plans

**Dataset Construction**:
- Stages:
    1. Data preparation
    2. Prompt design
    3. Expert revision
    4. Large-scale dialogue generation based on the Big Five personality traits
    5. Creation of fine-tuning datasets
- Incorporating Few-shot ICL enhanced model's ability to simulate distinct student personalities

**Data Preparation**:
- Selecting representative Chinese texts and transcribing classroom videos for authentic teacher-student dialogues

**Prompt Design**:
- Aligned with real classroom interactions to align with Big Five traits using GPT-4
- Enhanced virtual student personalization

**Expert Revision**:
- Ensured alignment before large-scale dialogue generation
- Resulted in dataset reflecting diverse personality-based interactions

**Fine-Tuning Datasets**:
- 5 distinct instruction fine-tuning datasets created, each corresponding to a specific personality trait

**Dataset Analysis**:
- Word cloud visualizations used to analyze students' expression styles:
    - HE: Dominant words indicate frequent use of self-referential and positive language, strong inclination toward interaction and narration (Figure 5a)
    - HN: Hesancy, anxiety, uncertainty (Figure 5b)
    - LO: Conservative and restrained language, reliance on established knowledge structures (Figure 5c)
    - HA: Cooperation, care, and empathy (Figure 5d)
    - LC: Lack of precision, disorganized expression, vague language (Figure 5e)
- Findings align with existing research showing language style reflects cognitive abilities and personality traits

**Additional Examples**:
(Refer to Appendix C.2.3 for more details)

### 4.3 FINE-TUNING THE FOUNDATION MODELS

**Fine-tuning Foundation Models (4.3)**
- Utilized a high-performance setup with eight A6000 GPUs to fine-tune the model for capturing linguistic traits of students with various personalities.
- SWIFT infrastructure, along with LoRA, was employed for efficient fine-tuning, tailoring hyperparameters to each LVSA type.
- This approach personalizes the model to reflect specific personality traits effectively.
- Detailed examples are available in Appendix D.2.
- Pre- and post-fine-tuning results showed significant improvements in linguistic capabilities, with fine-tuned models producing more aligned responses to targeted language styles and traits, demonstrating enhanced structure and coherence.
- Further evaluation of these improvements is discussed later.

## 5 EVALUATION: ANALYSIS OF OBJECT CAPABILITIES IN SCENES

**Evaluation of LVSA**

* Construct subjective evaluation dataset (Figure 1(e))
	+ Inference dataset creation
	+ Fine-tuned inference
	+ Direct inference
	+ Evaluation data reconstruction
* Evaluation process (Figure 1(f))
	+ Human evaluation (Section 5.2)
	+ Human-GPT-4 comparison evaluation (Section 5.3)
	+ Large-scale GPT-4 evaluation (Section 5.4)

### 5.1 SUBJECTIVE EVALUATION DATASET

**Subjective Evaluation Dataset Construction**
- **Four-step process**: Inference dataset creation, fine-tuned inference, direct inference, evaluation data reconstruction
- **Inference dataset construction**:
  - GPT-4 generates teacher questions based on student personality traits using prompt engineering techniques from the fine-tuning process
  - Includes two fields: system (teaching scenario and student traits), query (teacher questions)
- **Fine-tuned inference**:
  - Student responses generated for five distinct personality types: HE, LO, HN, HA, LC
  - Personalized outputs from fine-tuned models
- **Direct inference**:
  - Assessing baseline capabilities of non-fine-tuned models
  - Highlights improvements in personalization and realism after fine-tuning
- **Subjective evaluation dataset reconstruction**:
  - Total of 12,312 responses generated across four foundation models and five personality types
  - Covering different learning stages and question types before and after fine-tuning
- **Human evaluation**:
  - 80 samples selected: 40 from fine-tuned models, 40 from direct inference, 35 real student responses as control group (total of 115 samples)
    - Five used for fatigue test items
- **GPT-4 evaluation**:
  - Remaining 12,232 responses used for GPT-4's large-scale evaluation
    - 6,116 pre-fine-tuned and 6,116 post-fine-tuned responses

### 5.2 HUMAN TURING TEST

**Human Turing Test**

**Experiment Overview**:
- Aimed to determine if LVSA could emulate real students' language expressions and cognitive abilities
- Human evaluators assessed whether generated dialogues resembled those of real students

**Participants and Procedure**:
- Participants, acting as "judges", used teaching experience to evaluate 120 teacher-student dialogues
- Participants verbalized thought processes during evaluation, providing insights into linguistic features influencing judgments
- Subsequent semi-structured interviews explored evaluation criteria, developing a scientific framework for optimizing future virtual students

**Evaluation Results**:
- **Fleiss's Kappa**: 0.6917, indicating substantial inter-rater agreement and strong consensus among participants
- **Average recognition probability**: LVSA exceeded 0.9 across multiple personality traits, demonstrating close resemblance to real students
- In some cases, virtual students surpassed real students in recognition rates, highlighting LVSA's potential to effectively emulate human-like language in educational contexts

### 5.3 HUMAN-GPT4 COMPARISON VALIDATION

**GPT-4 Comparison Validation**

**Human-GPT4 Evaluation Differences**:
- GPT-4 prompt engineering: Systematically extracted human evaluator interview criteria for Large-Scale Virtual Student Assessments (LVSA) and integrated into GPT-4 as prompts
- Two coders used ATLAS.ti software for a two-level coding process, resulting in 4 primary and 15 secondary codes covering:
  - Emotional integration
  - Cognitive level
  - Psychological state
  - Verbal expression
- High inter-rater reliability (0.876) confirmed the validity of these dimensions
- Coded dimensions were used to design prompts for GPT-4 using a chain-of-thought approach, enabling step-by-step evaluations closely aligned with human judgments
- Structured prompt design improved GPT-4's accuracy and ensured consistency with human evaluators

**GPT-4 Evaluation Ability**:
- GPT-4 achieved an average evaluation score of 0.978, with perfect alignment (score of 1) for certain personality traits
- Overall Fleiss’s Kappa of 0.6806 indicates substantial agreement with human evaluators, supporting GPT-4's reliability in educational evaluation

**Comparison of Evaluation Capabilities**:
- Table comparing GPT-4 and 10 human evaluators' assessment of LVSA agents with different personality traits
- GPT-4 demonstrated effective simulation of nuanced performance, with the highest accuracy for certain traits

**Conclusion**:
- The findings establish a foundation for GPT-4's future application in educational research and practice.

### 5.4 LARGE-SCALE EVALUATION EXPERIMENT WITH GPT-4

**Evaluation of LVSA Performance: GPT-4 Analysis**

**Analysis of Different Personality Types:**
* Pre-fine-tuning scores: HA (29.29%), HN (44.98%), HE (26.86%), LC (15.22%), and non-LC types (36.76%)
* Post-fine-tuning scores: all models showed substantial improvements, with an average score of 72.51%
* Statistically significant improvements for all models as indicated by paired t-tests
* Virtual students with HA demonstrated the most notable improvements, p-values below 0.001
* All personality types except LC showed significant gains post-fine-tuning

**Evaluation of Different Learning Stages:**
* Pre-fine-tuning scores: InternVL (54.53%), LLaVa (15.22%), MiniCPM (15.38%), Qwen (68.15%), and average performance of 54.51%
* Post-fine-tuning scores: significant improvements for all models with an average improvement of 37.28%
* Statistically significant differences as indicated by paired t-tests
* Virtual students effectively adapt to different learning stages, offering comprehensive support for pre-service teacher training

**Evaluation of Different Question Types:**
* Pre-fine-tuning scores: closed questions (54.51%) and open questions (38.00%)
* Post-fine-tuning scores: significant improvements for all question types, average improvement of 39.76%
* Paired t-tests showed statistically significant differences, p-values below 0.05
* Greater impact on open questions due to their higher complexity and lower initial performance.

### 5.5 BAD CASE ANALYSIS

**Limitations of LLMs (Limited Learning Machines) in Role-Playing Contexts**

**Poor Fine-Tuning Performance of LC (Low Conscientiousness) Virtual Students:**
- Sparse distribution of relevant data for low-conscientiousness behaviors in original training set
- Difficulty learning accurate traits due to underrepresentation
- Higher likelihood of "hallucination" - generated responses lacking semantic coherence
- Complication in modeling negative traits associated with antisocial performance

**Inconsistent Fine-Tuning Effects Across Question Types:**
- Improvements seen for closed and open-ended questions, but no statistically significant improvements across both types
- Structured nature of closed-ended questions makes them easier to manage
- Deeper reasoning and creative thinking required for open-ended questions, resulting in greater variability in performance

**Suboptimal Fine-Tuning Performance of LLaVa:**
- Overall performance weaker than others despite improvements in personalization, question types, and learning stages
- Differences between pre-training and fine-tuning data domains, particularly cross-language issues
- Adaptability and generalization to Chinese contexts constrained due to reliance on English pre-training data.

## 6 CONCLUSION

**Conclusion:**
* Introduced "Scenario-Object-Evaluation" (SOE) pipeline for creating realistic virtual student agents using LLMs
* Designed for pre-service teacher training, addressing limitations of current simulation platforms
* Multi-dimensional evaluation assessed personality traits, question types, and learning stages
* Comparative experiments with human evaluators showed similarity between LLM dialogues and real student behavior
* Future work: integrate multimodal tasks (images, videos), improve error generation mechanism, enhance low-conscientiousness simulations.

## A COMPREHENSIVE RELATED WORK
### A.1 VIRTUAL STUDENT DEVELOPMENT AND RELATED TECHNOLOGIES

**Virtual Student Development and Related Technologies**

**Role of Teachers**:
- Central resource for educational development
- Pre-service teachers depend heavily on practical training for professional growth

**Virtual Students as a Tool for Pre-service Teacher Training**:
- Provide cost-effective, interactive, and flexible platform
- Simulate classroom interactions

**Types of Virtual Students**:
1. **Role-playing virtual students**: 
   - Imitated by teachers or adults
   - Offer real-time interaction but rely on participants' ability to accurately simulate student responses
   - Resource-intensive, lacks scalability and automation
2. **Pre-defined virtual students**:
   - Use pre-defined algorithms and scripted rules to simulate student interactions
   - Repeatable and operationally controlled, useful for consistent teaching tasks
   - Rigidity limits their ability to adapt to real-world classroom dynamics
3. **Virtual students powered by Large Language Models (LLMs)**:
   - Leverage advanced natural language processing capabilities to generate dynamic and contextually relevant student responses
   - Introduce greater flexibility and linguistic variability but face challenges in replicating full range of student behaviors
   - Demonstrated strong capabilities in natural language generation and knowledge understanding, but not yet fully mimicking language expression levels and cognitive challenges faced by real students

**Goal of the Study**:
- Explore whether LLMs can be leveraged to create human-like and personalized virtual student agents that better simulate student cognition and emotional responses
- Provide more authentic, dynamic, and adaptive learning interactions for pre-service teacher training.

### A.2 LLMS’ MAIN CAPABILITIES TO SUPPORT AI4EDUCATION

**LLMs' Capabilities for AI Education**

**Language Understanding**:
- LLMs demonstrate exceptional performance in language processing tasks
- Ability to comprehend complex linguistic structures
- High scores on benchmarks like GlUE, SuperGLUE, SQuAD
- Crucial for building virtual students that respond appropriately to teachers' questions

**Language Generation**:
- Models like GPT-4 and T5 demonstrate strong language generation capabilities
- Generate natural language paragraphs, dialogues, and long-form text
- Maintain contextual consistency throughout the generation process

**Language Reasoning**:
- LLMs required to perform complex logical inference and commonsense knowledge reasoning
- Current top-tier models like GPT-4 and PaLM excel in handling more complex reasoning tasks

**Knowledge Transfer**:
- Language models can apply learned knowledge to other domains or language tasks
- Models demonstrate strong cross-linguistic capabilities, handling comprehension and reasoning across various languages

**Multimodal Processing**:
- Models like Flamingo and InternVL combine visual and textual processing capabilities
- Offer greater potential for future development of virtual students in multimodal learning environments.

### A.3 REPRESENTATIVE AI4EDUCATION APPLICATIONS

**Educational Technology Applications of Large Language Models (LLMs)**

**Duolingo Max**:
- Leverages LLMs to enhance language learning through generative dialogue and intelligent feedback
- Helps students improve language skills via interactive exercises

**Khan Academy's Khanmigo**:
- Powered by LLMs
- Offers **intelligent tutoring** and personalized learning support, spanning primary to higher education

**Google Socratic Platform**:
- Assists students in solving problems across multiple disciplines

**Chinese Edtech Companies**:
- TAL Education's MathGPT: Provides step-by-step guidance in mastering core mathematical problem-solving techniques
- Youdao's "ZiYue": Prioritizes a "scenario-first" approach
- iFLYTEK's "Spark Desk": Conducts human-like interactive learning in various fields
- Squirrel AI: Develops a personalized learning system that tailors individual learning paths

**Universities**:
- MIT: Achieves 81% correctness on undergraduate-level math problems using LLMs and pre-trained models
- East China Normal University's EduChat: Offers functionalities such as open Q&A, essay grading, heuristic teaching, and emotional support

**Benchmark Datasets**:
- C-Eval: Evaluates the performance of LLMs across various subjects from primary to higher education
- SciQAG: Measures the model's ability to answer scientific questions
- Mukea: Benchmarks multimodal knowledge extraction and integration

**Conferences and Research**:
- NeurIPS and AAAI have seen a growing number of research papers on LLMs in education
- Topics include supporting individualized learning paths, providing personalized feedback, and education dataset construction

**Future Work**:
- Research is emerging on using LLMs to support teacher skills, particularly in areas like problem-solving, knowledge explanation, and automated grading
- This study aims to introduce a novel research direction: using virtual students to enhance teachers' pedagogical skills

## B DETAILED INFORMATION FOR THEORY AND TECHNOLOGY USED IN THIS WORK
### B.1 FOUNDATION LLMS

**LLMs for Junior High School Chinese Language Tasks**

**Selection of Representative Foundational Models**:
- **InternVL**: Multimodal large language model integrating visual and linguistic processing capabilities, excels at cross-modal reasoning
- **LLaVa**: Lightweight yet powerful LLM based on Mistral architecture, specializes in text generation and language understanding
- **MiniCPM**: Optimized for Chinese language tasks, deep capabilities in understanding and generating Chinese
- **Qwen**: Supports multimodal interactions, excels in visual-linguistic exchanges and pure text-based tasks

**Performance Evaluation**:
- Evaluated for comprehension and recitation abilities, essential for constructing virtual student agents
- Assesses ability to process and understand Chinese text, specifically focusing on comprehension and recitation
- Provides foundation for future expansion into multimodal teaching environments involving both textual and visual processing

**Foundation LLMs Repository URLs**:
- **InternVL2 (8B)**: [Shanghai AI Laboratory](https://huggingface.co/ShanghaiAI/InternVL2-8B)
- **LLaVa (7b-hf)**: [Microsoft](https://huggingface.co/microsoft/LLaMA-7b-hf)
- **MiniCPM (8B)**: [MiniCPM-V](https://huggingface.co/MiniCPM-V)
- **Qwen (vl-chat)**: [Alibaba Cloud](https://huggingface.co/Alibaba-Cloud/Qwen-vl-chat)

### B.2 EDUCATIONAL THEORY

**Educational Theory: Characteristics of Early Adolescent Students**

**Physiological Development**:
- Rapid development of prefrontal cortex (complex cognitive functions) vs. faster development of limbic system (emotional regulation)
- Balance leads to intense emotional responses, limited cognitive regulation, and variability in participation

**Cognitive Development**:
- Transition from concrete operational thinking to abstract logical reasoning
- Gradual ability to handle abstract concepts and logical analysis
- Easily influenced by emotions and external factors
- Repetitive and uncertain language expression, especially with abstract issues

**Social-Emotional Development**:
- Increased self-awareness and concern for social roles
- Frequent emotional fluctuations
- Heightened sensitivity affects classroom behavior (teacher feedback, peer reactions)
- Simulation of emotional dynamics crucial for accurate reflection of student responses

**Moral and Spiritual Development**:
- Beginning to develop own values and face complex moral issues
- Transition from external rule adherence to internalizing societal norms
- Moral judgments often emotionally driven and oversimplified
- Interpret moral situations through personal experiences
- Emerging critical thinking shows students analyzing multiple perspectives, but skills not yet fully mature.

### B.2.2 OPERATIONAL THEORY: CLASSIFICATION OF TEACHER-STUDENT DIALOGUE DATASETS

**Operational Theory: Classification of Teacher-Student Dialogue Datasets**

**Introduction:**
- Importance of operational theory in generating realistic student behavior and language capabilities
- Foundation for generating diverse student behaviors and consistent responses across classroom scenarios

**Dimensions of Classification:**
1. **Question-Answer Type:**
   - Fundamental for generating virtual students
   - Different types require various cognitive processes, influencing student language output and development
   - Open-ended questions encourage creative thinking; closed-ended questions focus on recalling facts
   - Careful design aids in simulating realistic classroom interactions
2. **Personality Traits:**
   - Critical dimension for personalized student modeling
   - Based on the Big Five Personality Traits model
   - Impact on student behavior, language expression, and creation of distinct personalities
   - Enhances personalized expression and exploration of language patterns
3. **Low Openness (LO):**
   * Conservative and pragmatic
   * Low receptivity to new content
   * Lack of initiative in exploration
   * Weaker ability to handle complex issues
   * Characterized by simple, direct responses and difficulty expanding discussions
4. **Low Conscientiousness (LC):**
   * Careless behavior and lack of organization
   * Inconsistent responses, including self-correction
   * Lack of systematic thinking leading to disorganized responses
   - Characterized by simplicity, occasional self-correction, and unreliability in language style
5. **High Extraversion (HE):**
   * Active participation and strong social skills
   * Desire for engagement and expressing themselves confidently
   * Fluent and confident language style with minimal hesitation or pauses
6. **High Agreeableness (HA):**
   * Cooperative and empathetic behavior
   * Thoughtful, patient, and tolerant of others
   * Warm and friendly language expression
7. **High Neuroticism (HN):**
   * Anxiety, nervousness, and emotional fluctuations
   * Hesitant, repetitive backtracking in responses
   - Disjointed speech patterns due to emotional instability
8. **Learning Stages:**
   - Reflects cognitive abilities and language expression development at different stages
   - Accurate modeling of learning stages creates virtual students suitable for various instructional contexts
9. **Answering Style:**
   - Individual differences in students’ language expression
   - High openness leads to more creative, exploratory language; high conscientiousness results in detailed and accurate responses
10. **Generation Source:**
    - Traditional prompt engineering, few-shot ICL, CoT, and LoRA methods used for generating realistic student dialogues
    - Combination of these methods ensures that generated dialogues closely resemble real-life classroom interactions

### B.2.3 THE UNIQUE VALUE OF MIDDLE SCHOOL STUDENTS AND THE LANGUAGE-BASED SUBJECT IN THIS STUDY

**Target Group for Virtual Student Modeling: Early Adolescents (Middle School Students)**

**Characteristics of Early Adolescents:**
- Transitioning from concrete to abstract thinking
- Cognitive development not fully matured
- Weaker logical reasoning skills
- Rely more on intuitive perception and concrete experiences
- Unclear language expression and inconsistent thought processes
- Limited language proficiency
- Heightened uncertainty and repetition in verbal responses

**Value of Studying Early Adolescents:**
- Represents diverse classroom behaviors
- Challenges large language models on complex tasks
- Offers insights into language expression, emotion, and attitudes

**Cognitive Processes Involved in Classroom Interactions:**
1. Focusing on the question
   - Adequate attention and selective information processing
2. Understanding the question's meaning
   - Relating it to existing knowledge
3. Providing a response
   - Knowledge application
   - Clear articulation of thoughts

**Importance of Language-Based Subjects:**
- Emphasizes language expression, emotional experience, and value cultivation
- Aligns with core capabilities of large language models (LLMs)
- Offers insights into linguistic styles, emotional expressions, and responses to complex questions.

## C DETAILED INFORMATION FOR DATASET CONSTRUCTION PROCESS
### C.1 BASIC CHINESE UNDERSTANDING ABILITY EVALUATION DATASET

**Chinese Understanding Ability Evaluation Dataset: Junior High School Chinese Language**

**Data Source:**
- National Smart Education Platform: authorized and standardized educational materials from Ministry of Education for teachers and students nationwide
- Integrates official junior high school Chinese language curriculum, including textbook passages, practice exercises, comprehension assessments, and recitation tasks

**Focus Areas:**
1. Text Comprehension
2. Text Memorization

**Text Comprehension:**
- Evaluate understanding of text structure, sentence organization, inference, vocabulary context, emotional tones
- Assesses ability to process and understand Chinese texts at junior high level

**Text Memorization:**
- Tests model's ability to recall and reproduce texts
- Essential for assessing long-term memory and linguistic retention in Chinese education

**Data Preparation:**
1. Organize exercises by unit, maintaining core knowledge points and skill requirements
2. Convert PDF format for model input
3. Ensure integrity of data structure
4. Clear framework for model processing

**Prompt Design:**
1. Instruction-based learning method with structured prompts
2. Provide clear task instructions for accurate outputs
3. Optimize complex tasks, such as text comprehension and recitation

**Expert Revision:**
1. Team of professional junior high school Chinese teachers reviews generated data
2. Evaluates model's outputs for accuracy, consistency, alignment with educational standards
3. Corrects errors to improve dataset quality
4. Enhances dataset's applicability in educational settings

**Text Comprehension Data Construction:**
1. Collect 613 text comprehension items through expert revision
2. Improves overall dataset quality and scientific rigor
3. Ensure alignment with junior high school Chinese education requirements

**Text Memorization Data Construction:**
1. Create multiple-choice questions for students to assess specific content recall from the text
2. Design questions based on original text without rewriting or modification
3. Evaluate students' ability to remember content from recitation passages.

### C.2 FINE-TUNING DATASET

**Personality Traits: Low Conscientiousness (LC)**

**Teacher's Expectations**:
- Answering teacher's questions during class
- Exhibit traits of carelessness, inconsistency, and lack of systematic thinking
- Provide simple and direct answers that are occasionally self-corrected and unreliable

**Strategy for Answering Questions**:
1. Focus on the question and identify the phase of the lesson (pre-lesson introduction, new lesson instruction, consolidation of new knowledge, classroom practice, or lesson summary)
2. Understand the nature of the question - determine if it is open-ended or closed-ended
3. If closed-ended, give a brief answer based on your knowledge
4. If open-ended, answer according to your personality traits

**Word Cloud Visualization of Big Five Personality Traits: LVSA Fine-Tuning Dataset**
*Figure A10: Top 200 High Frequency Words (LC LVSA)*
- The first 200 words are displayed, based on Chinese answers translated into English

**Data Analysis: The Top 200 High Frequency Words (HA LVSA)**
*The experiment is based on Chinese, with Chinese answers counted for word frequency and translated into English for display.*

**Conclusion**:
- This section provides an analysis of the data from virtual students with low conscientiousness personality traits. The top 200 high frequency words are presented in a word cloud visualization and discussed.

### C.3 SUBJECTIVE EVALUATION DATASET

**Subjective Evaluation Dataset (SE Dataset)**
* **Table A3**: Details of Subjective Evaluation Dataset
	+ Learning Stage:
		- Big Five Type: HA, HE, LC, LO, HN
		- Total: 16 questions per category for each type
	+ Pre-lesson Introduction:
		- Closed Question: 0-3 (HA: 3, HE: 0, LC: 1, LO: 1, HN: 5)
		- Open Question: 1-8 (HA: 2, HE: 3, LC: 1, LO: 4, HN: 1)
	+ New Lesson Instruction:
		- Closed Question: 3-12 (HA: 3, HE: 3, LC: 1, LO: 2, HN: 5)
		- Open Question: 2-8 (HA: 2, HE: 3, LC: 1, LO: 2, HN: 2)
	+ Knowledge Consolidation:
		- Closed Question: 3-11 (HA: 3, HE: 3, LC: 1, LO: 2, HN: 5)
		- Open Question: 5-14 (HA: 3, HE: 2, LC: 2, LO: 2, HN: 8)
	+ Classroom Exercises:
		- Closed Question: 0-3 (HA: 0, HE: 1, LC: 1, LO: 1, HN: 3)
		- Open Question: 5-14 (HA: 3, HE: 2, LC: 2, LO: 2, HN: 8)
	+ Lesson Summary:
		- Closed Question: 0-2 (HA: 0, HE: 0, LC: 0, LO: 1, HN: 2)
		- Open Question: 0-8 (HA: 0, HE: 3, LC: 0, LO: 3, HN: 5)
* **GPT-4-E Pre-lesson Introduction**:
	+ Closed Question: 424-1816 (HA: 336, HE: 280, LC: 392, LO: 384)
	+ Open Question: 104-512 (HA: 144, HE: 96, LC: 72, LO: 96)
* **New Lesson Instruction**:
	+ Closed Question: 296-1528 (HA: 296, HE: 160, LC: 360, LO: 304)
	+ Open Question: 288-1080 (HA: 216, HE: 264, LC: 128, LO: 184)
* **Knowledge Consolidation**:
	+ Closed Question: 192-1064 (HA: 192, HE: 200, LC: 224, LO: 256)
	+ Open Question: 320-1296 (HA: 264, HE: 224, LC: 264, LO: 224)
* **Classroom Exercises**:
	+ Closed Question: 256-1176 (HA: 184, HE: 208, LC: 280, LO: 280)
	+ Open Question: 240-1208 (HA: 272, HE: 280, LC: 192, LO: 192)
* **Lesson Summary**:
	+ Closed Question: 80-520 (HA: 72, HE: 136, LC: 104, LO: 136, HN: 128)
	+ Open Question: 512-432 (HA: 432, HE: 360, LC: 376, LO: 352, HN: 2032)
* Number of Questions per Category:
	+ Pre-lesson Introduction: 100 (closed) and 200 (open) questions
	+ New Lesson Instruction: 200 (closed) and 400 (open) questions
	+ Knowledge Consolidation: 400 (closed) and 800 (open) questions
	+ Classroom Exercises: 300 (closed) and 1200 (open) questions
	+ Lesson Summary: 100 (closed) and 800 (open) questions
* **Distribution of Subjective Evaluation Dataset**: Refer to the chart (Figure A11) for details.

## D DETAILED EXPERIMENT SETTINGS AND RESULTS
### D.1 FINE-TUNING CONFIGURATION

**Model Fine-tuning Process**
* Conducted using high-performance computing cluster with 8 A6000 GPUs for efficient training and stable performance
* SWIFT framework employed: reduces computational cost of fine-tuning large models through integration of Low-Rank Adaptation (LoRA) method
	+ Minimizes parameter updates with low-rank matrices
	+ Personalizes large models without extensive resources
* Hyperparameters fine-tuned for each personality type based on language characteristics and cognitive development
* Table A4 provides hyperparameter settings:
	+ Model: Fine-tuning parameters for High Extraversion (HE), High Neuroticism (HN), Low Openness (LO), High Agreeableness (HA), and Low Conscientiousness (LC) students.

**Model Hyperparameters:**
| Personality | Hyperparameter |
| --- | --- |
| HE, HA | Learning rate: 1.0E-04 |
| HE, HA | Num train epochs: 3 |
| HN, LC | Learning rate: 1.0E-04 |
| HN, LC | Num train epochs: 3 |
| LO | Dropout: 0.2 |
| LO | Learning rate: 5.0E-05 |
| LO | Num train epochs: 2 |
| All | LoRA alpha: 64 for HE, HN, and HA; 32 for LO |

### D.2 FINE-TUNING RESULTS

**Pre-Fine Tuning Student Responses:**

**Ode to the Yellow River**
- HN Student: Gives examples of poet's praise for the heroic spirit of the Yellow River (long, continuous flow, nurtures generations, steadfast and unyielding)
- LLaVa Student: Depicts grandeur of Yellow River as a large body of water with vast hydropower potential
- MiniCPM Student: Speaks for the people and reflects societal suffering
- Qwen Student: Mother river of China, nurturing Chinese civilization

**Memories of Mr. Lu Xun**
- HN Student: Impressions of Mr. Lu Xun as a famous modern writer and thinker who advocated for vernacular language movement
- LLaVa Student: Daily scenes like sleeping, eating, and playing hide-and-seek with Mr. Lu Xun
- MiniCPM Student: Focuses on emotional bond between Xiao Hong and Mr. Lu Xun

**Ode to the Lotus**
- HE Student: Lotus grows in clear water and symbolizes purity, integrity

**The Ballad of Mulan**
- HE Student: Highlights Mulan's filial piety and bravery, symbolizing women's independent spirit

**Importance of the Yellow River to China**
- HA Student: Mother river of China, nurturing Chinese civilization, important water resource, significant transportation route

**Natural Landscapes in Northeast China (Oath of the Land)**
- LLVa Student: Mountains, forests, lakes

**Praise for Lotus (Ode to the Lotus)**
- Zhou Dunyi: Embodies qualities of purity and refusal to succumb to worldly corruption

**Wen Yiduo's Unconventional Acceptance of Zang Kejia**
- Wen Yiduo: Values talent, not just academic achievements

**Post-Fine Tuning Student Responses:**

**Ode to the Yellow River**
- HN Student: Mother river of China, symbolizes Chinese nation
- LLaVa Student: Clear water, significant natural resource and historical significance
- MiniCPM Student: Continuous flow, nurtures generations
- Qwen Student: Important role in cultural transmission and development

**Oath of the Land (Northeast China)**
- Natural landscapes: Large mountains, waters, grasslands

**Wen Yiduo's Words and Deeds**
- Values academic work and political activities seriously

**Sun Quan Urges L¨u Meng to Study**
- Encourages studying by explaining The Six Secret Teachings

**Ah Chang and the Classic of Mountains and Seas**
- Ah Chang buys book for Lu Xun due to shared interest in ancient Chinese culture

**Deng Jiaxian's Contributions**
- Great scientist who made significant contributions to China's nuclear weapons program

**Impressions of Mr. Lu Xun (Xiao Hong's Recollections)**
- Great literary figure, deeply influential on Chinese literature.

### D.3 HUMAN TURING EVALUATION QUESTIONNAIRE

**Human Turing Evaluation Questionnaire**

**Experiment Overview**:
- Inspired by the classic Turing test
- Aimed to evaluate whether LVSA could emulate human language expression and cognitive levels
- Participants acted as "judges" to determine if generated dialogues resembled natural human student responses

**Experimental Steps**:
1. **Questionnaire completion**: Participants judged whether student responses resembled real ones or not
2. **Think-aloud protocol**: Participants verbalized their thought process during the questionnaire
3. **Semi-structured interviews**: Participants elaborated on factors affecting their evaluations

**Participant Recruitment**:
- 35 participants were recruited, including pre-service and in-service teachers
- Participants had experience teaching junior high school Chinese and expressed willingness to participate
- Detailed information of 10 selected participants was provided in the table

**Pre-survey of Evaluators**:
- Participants were asked to provide personal information, teaching experience, and familiarity with junior high Chinese textbooks
- They also indicated their willingness to participate in think-aloud exercises and semi-structured interviews

**Experiment Procedure**:
- Experimenter provided detailed explanations and guidelines to ensure scientific rigor
- Participants were asked to judge 120 teacher-student dialogues, some involving virtual students
- Participants' verbalizations during the questionnaire were recorded for analysis

**Evaluation Criteria**:
- Based on the Big Five personality traits of students (Carefulness, Cooperation, Nervousness, Low Tolerance, and High Extraversion)
- Participants ranked student descriptions based on these traits

**Think-aloud Protocol**:
- Participants verbally explained their thought process while answering questions
- Recorded for analysis to determine the criteria used in distinguishing virtual from real student language performances

**Semi-structured Interviews**:
- Participants elaborated on factors affecting their evaluations
- Responses were coded for consistency and used to develop a scientific evaluation framework

**Compensation**:
- Participants received appropriate compensation and a small token of appreciation for their contributions.

### D.4 HUMAN TURING EVALUATION RESULTS

**Human Turing Evaluation Results**

**Table A6:**
- Evaluator performances for distinguishing virtual students from real ones
  - Columns: HE, HN, LO, HA, LC (High Extraversion, High Neuroticism, Low Openness, High Agreeableness, Low Conscientiousness)
  - Rows: Evaluators' judgments before and after interaction with students
- Evaluator1 to Evaluator10 results

**Evaluation Metric:**
- Probability of identifying fine-tuned virtual students as real students

**Findings:**
- Average recognition probability for fine-tuned virtual students exceeds 0.9 across personality dimensions
- Some virtual students with specific traits (high neuroticism, low conscientiousness, low openness) become indistinguishable from real students
- Language generation for these virtual students is more convincing and human-like in teaching scenarios.

**Fleiss's Kappa:**
- Measure of inter-rater agreement used to evaluate participants' judgments of virtual versus real students
- A value between 0.6 and 0.8 indicates substantial agreement, suggesting high consensus among participants
- Experiment result showed a Fleiss’s Kappa value of **0.6917**, indicating strong agreement in participants' judgments.

**Impact:**
- Virtual students based on large language models have significant potential to emulate human language expression effectively in teaching scenarios.

### D.5 GPT-4 LARGE-SCALE EVALUATION PROMPT

**GPT-4 Large-Scale Evaluation Prompt**

**Coding of Interview Content**:
- Two students coded evaluator interview transcripts using ATLAS.ti software
- Employed a two-level coding method to ensure thoroughness and accuracy
- Identified four primary codes and 15 secondary codes capturing multidimensional factors evaluators focused on when distinguishing virtual from real students
- Coding dimensions included:
    - **Cognitive Level**: Complexity, Reasonableness, Logicality
    - **Psychological State**: Suspicions, Interaction, Nervousness, Reflection
    - **Verbal Expression**: Personalization, Sentence Structure, Oral Language, Fluency, Pronoun Usage, Length, Emotional Integration

**Prompt Design**:
- Coding dimensions were selected as key evaluation criteria in prompt design
- Adopted a "chain-of-thought" (CoT) reasoning approach to guide GPT-4's step-by-step assessment of virtual student responses
- Integrated coded evaluation dimensions into prompts to ensure consistent evaluation outcomes between GPT-4 and human evaluators

**Comparison of Human Evaluation and GPT-4 Evaluation**:
- Consistency between GPT-4 and human evaluators was quantified using Fleiss's Kappa coefficient
- GPT-4's average evaluation performance reached 0.978, demonstrating high alignment with human evaluators
- Evaluations for certain personality types exceeded human assessments, while others fell below
- Overall, the study found a "substantial agreement" (Fleiss's Kappa consistency coefficient = 0.6806) between GPT-4 and human evaluators

### D.6 BAD CASE EXAMPLES

**Bad Case Examples of LLMs' Performance**

**Limitations of LLMs in Role-Playing Contexts**:
- Anomalies identified: limited improvement in low-conscientiousness personalities, absence of fine-tuning effects across all question types in some models, and relatively weaker performance of LLaVa compared to other models post-fine-tuning.

**Low Conscientiousness Personalities**:
- Poor fine-tuning performance: attributed to sparse distribution of relevant data in original training data
- Increased likelihood of "hallucination phenomena" where model struggles to maintain semantic coherence
- Complicating factors: negative traits linked to antisocial behavior or marginalized groups, and measures to avoid reinforcing such characteristics during training.

**Comparison of Closed-Ended and Open-Ended Questions**:
- Inconsistent fine-tuning effects across question types: no statistically significant differences for InternVL, Qwen, and MiniCPM (p-values > 0.05)
- Structural differences between questions: closed-ended require precise recall, open-ended involve reasoning, emotional expression, and creative thinking.

**LLaVa Model Performance**:
- Lower overall post-fine-tuning accuracy of around 67.85% compared to other models
- Likely due to pretraining predominantly on English data, limiting performance in Chinese-language applications
- Future research: incorporating more extensive Chinese-language pretraining or customized fine-tuning tailored to Chinese contexts.

