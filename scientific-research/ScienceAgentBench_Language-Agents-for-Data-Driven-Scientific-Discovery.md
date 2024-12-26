# ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery

Collaborators: Ziru Chen\*, Shijie Chen\*, Yuting Ning, Qianheng Zhang (UW-Madison), Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei (UW-Madison), Zitong Lu (OSU Psychology Dept.), Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum (College of Pharmacy OSU), Xuhui Huang (UW-Madison Chemistry Dept.), Xia Ning (OSU BMI and CPhO), Song Gao (UW-Madison Geography Dept.), Yu Su, Huan Sun\* (Correspondence: chen.8336, chen.10216, sun.397@osu.edu)

Website: [https://osu-nlp-group.github.io/ScienceAgentBench/](https://osu-nlp-group.github.io/ScienceAgentBench/)

https://arxiv.org/abs/2410.05080v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 ScienceAgentBench](#2-scienceagentbench)
  - [2.1 Problem Formulation](#21-problem-formulation)
  - [2.2 Data Collection](#22-data-collection)
  - [2.3 Evaluation](#23-evaluation)
  - [2.4 Comparison with Existing Benchmarks](#24-comparison-with-existing-benchmarks)
- [3 Experimental Setup](#3-experimental-setup)
- [4 Results and Analysis](#4-results-and-analysis)
  - [4.1 Main Results](#41-main-results)
  - [4.2 Human Evaluation](#42-human-evaluation)
- [5 Conclusion and Future Work](#5-conclusion-and-future-work)
- [Appendix A Limitations](#appendix-a-limitations)
- [Appendix B Related Work](#appendix-b-related-work)
- [Appendix C Example Task Instructions](#appendix-c-example-task-instructions)
- [Appendix D More Details about Main Results](#appendix-d-more-details-about-main-results)
- [Appendix E Case Studies](#appendix-e-case-studies)
  - [E.1 Case 1: Action Space of OpenHands](#e1-case-1-action-space-of-openhands)
  - [E.2 Case 2: Influence of Expert-Provided Knowledge](#e2-case-2-influence-of-expert-provided-knowledge)
- [Appendix F Expert Validation Details](#appendix-f-expert-validation-details)
  - [F.2 Program Example for Subject Matter Experts](#f2-program-example-for-subject-matter-experts)
  - [F.3 Knowledge Example Provided to Subject Matter Experts during Annotation](#f3-knowledge-example-provided-to-subject-matter-experts-during-annotation)
- [Appendix G Rubric Examples](#appendix-g-rubric-examples)
- [Appendix H Prompt Templates](#appendix-h-prompt-templates)
- [Appendix I Publications, Repositories, and Licenses](#appendix-i-publications-repositories-and-licenses)

## Abstract

**Advancements in Language Models (LLMs)**
- Piqued interest in developing LLM-based language agents for automating scientific discovery
- Sparked excitement and skepticism about true capabilities of such agents

**Essential Tasks for Scientific Discovery**
- Agent must complete all essential tasks in the workflow to fully automate scientific discovery
- Rigorous assessment of agents on individual tasks before claims of end-to-end automation

**ScienceAgentBench: A New Benchmark for Evaluating Language Agents**
- Extracted 102 tasks from 44 peer-reviewed publications in four disciplines
- Engaged nine subject matter experts to validate the tasks
- Unified target output as self-contained Python program files
- Employed various evaluation metrics to examine generated programs, execution results, and costs
- Tasks underwent multiple rounds of manual validation by annotators and subject matter experts
- Proposed strategies to mitigate data contamination concerns

**Evaluation Results**
- Best-performing agent only solved 32.4% of the tasks independently
- With expert-provided knowledge, it could solve 34.3% of the tasks
- Underscores limited capacities of current language agents in generating code for data-driven discovery, let alone end-to-end automation for scientific research

**ScienceAgentBench Structure**
- Consists of one or more sub-tasks per task that must be completed to achieve the goal (Figure 1)
- Involves heterogeneous datasets from various disciplines: Bioinformatics, Computational Chemistry, Geographical Information Science, Psychology and Cognitive Neuroscience.

## 1 Introduction

**Large Language Models (LLMs)**
- Remarkable capabilities beyond text generation: reasoning, tool learning, code generation
- Research interests in developing LLM-based language agents for automating scientific discovery end-to-end

**Data-Driven Discovery Workflow**:
- Model development, data analysis, visualization are essential tasks
- Agents must complete all tasks to fully automate data-driven discovery

**Evaluating Language Agents**:
- Careful evaluations of agents' performance on individual tasks needed
- Objective assessment and continued development through benchmarks like ScienceAgentBench

**Design Principles of ScienceAgentBench**:
1. **Scientific authenticity**: Curate tasks from peer-reviewed publications, co-design with subject matter experts
2. **Rigorous graded evaluation**: Unify output as Python programs, employ various evaluation metrics and rubrics
3. **Careful multi-stage quality control**: Manual validation by annotators and experts to ensure task authenticity and scientific plausibility

**Evaluation of LLMs on ScienceAgentBench**:
- Claude-3.5-Sonnet using self-debug outperforms OpenHands CodeAct with fewer API fees
- Best agent can only solve 32.4% of tasks independently, 34.3% with expert knowledge

**Potential Impact and Future**:
- Language agents hold significant potential in augmenting human scientists' productivity
- Long-term goal: Measuring progress towards developing language agents for data-driven scientific discovery assistance.

## 2 ScienceAgentBench

**ScienceAgentBench Overview:**
Evaluates agents on essential tasks within data-driven discovery workflows.
Initial goal: Language agents as science co-pilots to write code for data processing, analysis, and visualization.
Target users are scientists who can code but wish to save programming time using language agents.
Tasks are formulated as code generation problems, ensuring verifiable outputs directly usable by scientists with minimal modifications.

### 2.1 Problem Formulation

**ScienceAgentBench: Component Analysis**

**Figure 2:** An example Computational Chemistry task in ScienceAgentBench consists of four components:
1. **Task Instruction**: Describes the goal and output requirements of a data-driven discovery task. Kept concise to retain open-endedness and encourage agent development without reliance on prescriptive directions. (See Appendix [C](https://arxiv.org/html/2410.05080v1#A3 "Appendix C Example Task Instructions ‣ ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery") for examples.)
2. **Dataset Information**: Contains directory structure and a preview of the dataset's content to help agents use it correctly. Optional for those with file navigation tools, but saves turns in interactions for those without them.
3. **Expert-Provided Knowledge**: Includes explanations for scientific terms, formulas, and example usages of programming tools provided by subject matter experts. Agents may utilize this optional input to mitigate their knowledge gap in involved disciplines (as shown in Section [4](https://arxiv.org/html/2410.05080v1#S4 "4 Results and Analysis ‣ ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery")).
4. **Annotated Program**: Adapted from an open-source code repository released by a peer-reviewed scientific publication, providing a self-contained program with package imports, function/class implementations, and a main procedure to carry out the task. Agents are expected to generate similar independent programs that can be executed (e.g., by Python interpreter), but not necessarily using the same tools as in annotated programs.

### 2.2 Data Collection

**Task Annotation Process:**
* Form a group of graduate students to annotate tasks in four disciplines: Bioinformatics, Computational Chemistry, Geographical Information Science, and Psychology & Cognitive Neuroscience
* Search for peer-reviewed publications that release code and data under permissive licenses
* Follow five steps to annotate each task:
	+ Identify a reasonably documented code example and convert it into a benchmark task
	+ Collect and preprocess datasets used in the code
	+ Annotate the reference program by revising it to analyze datasets in the benchmark
	+ Implement task-specific success criteria as an executable script and draft fine-grained rubrics for evaluation
	+ Write instruction and dataset information for the task
* Discarded four tasks due to long execution times or nontrivial environment setup, leaving 106 tasks for validation

**Data Contamination and Shortcut Mitigation Strategies:**
* Agents may take shortcuts, such as reading ground-truth labels in test sets without writing training code, which can hurt evaluation validity
* Devise two strategies to modify datasets:
	+ Randomly remove five data points from test sets to prevent agents from using automatic data loaders with training corpora
	+ Replace test set labels with dummy values for tasks involving model development to ensure fairness

**Expert Validation:**
* Engage nine subject matter experts (senior Ph.D. students and professors) to validate tasks and provide additional knowledge
* Present instructions, dataset information, annotated programs, and rubrics to the experts, who complete a questionnaire to:
	+ Validate if a task represents a realistic data-driven discovery workflow in their discipline
	+ Review if the instruction provides an accurate high-level description of the program and uses professional languages
	+ Provide up to three pieces of knowledge needed for solving each task
	+ Make necessary revisions to rubrics
* Revise 41 task instructions and remove three tasks that are not representative enough for scientific workflows, leaving 103 tasks

**Annotator Verification:**
* Ask annotators to verify tasks they did not compose and execute programs to reproduce results
* Refine 29 task annotations and discard one more hard-to-replicate task
* Finalize ScienceAgentBench with 102 high-quality tasks for data-driven scientific discovery.

### 2.3 Evaluation

**Evaluation Challenges and Strategies in ScienceAgentBench:**
* **Open-ended tasks**: introduce challenge of diverse setup requirements for programs generated by different agents
* **Conditional environment initialization**: used to accommodate various program requirements, including a conda environment with basic Python packages: numpy, pandas, matplotlib, pytorch, tensorflow, rdkit, and tf_keras
* **Evaluation metrics**: Valid Execution Rate (VER), Success Rate (SR), CodeBERTScore (CBS), API Cost (Cost) for program evaluation; figure quality assessment using GPT-4o as judge.
* **Rubric-based evaluation**: complementary to outcome-based metrics, assesses programs at fine-grained levels: Data Loading, Data Processing, Modeling or Visualization, Output formatting, Output Saving.

**Evaluation Strategy:**
1. Initialize conda environment with basic Python packages for program execution
2. Analyze each program using pipreqs to identify required packages
3. Update the conda environment and configure packages accordingly
4. Evaluate programs based on success criteria (Table 1)
5. Check for execution errors, save output correctly, and calculate evaluation metrics: VER, SR, CBS, Cost
6. For figure evaluation, use GPT-4o as judge to assess quality and obtain average score from sampled responses
7. Conduct human evaluation using rubrics for fine-grained assessment (Section 4.2)
8. Automate the rubric-based evaluation process in future work.

### 2.4 Comparison with Existing Benchmarks

**Comparison of ScienceAgentBench to Representative Existing Benchmarks**

**Table 2: Comparison of ScienceAgentBench with Existing Benchmarks**

| **Benchmark** | **Code Generation** | **Task Type** | **Heterogeneous Data** | **Shortcut Prevention** | **Number of Tests** | **Complexity Level** | **Sources** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TaskBench (Shen et al., [2024](https://arxiv.org/html/2410.05080v1#bib.bib66)) | No Code Gen | Synthetic | ✗ | ✗ | 28,271 | - | GitHub |
| SWE-Bench (Jimenez et al., [2024](https://arxiv.org/html/2410.05080v1#bib.bib37)) | File-Level Edit | GitHub | ✗ | ✗ | 2,294 | - | - |
| BioCoder-Py (Tang et al., [2024b](https://arxiv.org/html/2410.05080v1#bib.bib71)) | Function-Level | GitHub | ✗ | ✗ | 1,126 | - | GitHub |
| MLAgentBench (Huang et al., [2024b](https://arxiv.org/html/2410.05080v1#bib.bib34)) | File-Level Edit | Kaggle | ✗ | ✗ | 13 | - | Kaggle |
| DiscoveryBench-Real (Majumder et al., [2024b](https://arxiv.org/html/2410.05080v1#bib.bib50)) | No Code Gen | 27 Publications | ✓ | ✗ | 6 | - | - |
| SciCode (Tian et al., [2024](https://arxiv.org/html/2410.05080v1#bib.bib72)) | Function-Level | Publications | ✗ | ✓ | 5 | - | Publications |
| BLADE (Gu et al., [2024](https://arxiv.org/html/2410.05080v1#bib.bib28)) | Function-Level | 31 Publications | ✗ | ✗ | 6 | - | Publications |
| **ScienceAgentBench (Ours)** | File-Level Generation | 44 Publications | ✓ | ✓ | 4 | - | - |

**Unique Features of ScienceAgentBench:**
1. Tasks require generating a standalone program file from scratch, instead of JSON API calls or a few lines of code completion or edits in other benchmarks.
2. Adapts 44 peer-reviewed publications and covers various real-world datasets across different disciplines, including complex structures like cell images, chemical structure-activity relationships, and geographical maps with multiple layers.
3. Mitigates data contamination and agent shortcut issues to establish valid evaluation.
4. Medium scale of 102 tasks, smaller than some synthetic or easier task benchmarks but reasonable for evaluating agents considering annotation difficulty and evaluation cost.

**Limitations and Related Work:** (Discussed in Appendices A and B)

## 3 Experimental Setup

**Experiment Frameworks:**
* Direct Prompting: simple framework, shows basic code generation capability of each LLM
* OpenHands CodeAct: agent development framework for code generation and software engineering
	+ Interacts with Python code interpreter, bash shell, and web browser
	+ Unifies all actions into a large action space
	+ Uses best-performing CodeActAgent v1.9 in experiments
* Self-Debug: code generation framework for LLMs to execute programs and reflect on results iteratively
	+ Re-implemented with modifications: no self-reflection before debugging, early exit option, set up environment using pipreqs and pip-tools
	+ Ensures fair comparisons by not initializing self-debug environment with basic packages or configuration rules
	+ Improves evaluation stability by repeating each task with three independent runs and selecting the best run based on metrics.

**Experiments:**
* Three open-weight LLMs: Llama-3.1-Instruct-70B, 405B (Dubey et al., [2024](https://arxiv.org/html/2410.05080v1#bib.bib20)), and Mistral-Large-2 (123B) (MistralAI, [2024](https://arxiv.org/html/2410.05080v1#bib.bib54))
* Two proprietary LLMs: GPT-4o (OpenAI, [2024](https://arxiv.org/html/2410.05080v1#bib.bib56)) and Claude-3.5-Sonnet (Anthropic, [2024](https://arxiv.org/html/2410.05080v1#bib.bib2))
* Same hyperparameters: temperature = 0.2, top_p = 0.95, 0-shot prompting via APIs
* Evaluated under three different frameworks: Direct Prompting, OpenHands CodeAct, Self-Debug.

## 4 Results and Analysis

**Experiments (Table 3)** reveal that recent LLMs and agents perform modestly on tasks, with a success rate of only up to 34.3% for Claude-3.5-Sonnet using expert knowledge. This result signifies that current LLM-based agents are inadequate for complex data-driven discovery tasks like those presented in ScienceAgentBench.

### 4.1 Main Results

**Findings on ScienceAgentBench**

**Direct Prompting vs. Self-Debug**:
- Directly prompted LLMs cannot unleash their full potential for data-driven discovery tasks
- Directly prompted LLMs can only solve 16.7% of tasks independently and 20.6% with additional knowledge
- LLM-generated programs have correct high-level structures but implementation-level errors, such as missing steps or wrong API usage
- **Self-debug** can nearly double the success rate (16.7 → 32.4; 1.94×) of Claude-3.5-Sonnet without extra knowledge
- With expert-provided knowledge, self-debug shows decent improvement over direct prompting:
  - 13.7 absolute gains on SR (20.6 → 34.3; 1.67×)
  - 45.1 absolute gains on VER (41.2 → 86.3; 2.09×)
- Self-debug is effective and important for enabling LLMs to execute and revise their code

**OpenHands CodeAct vs. Self-Debug**:
- For four of the five LLMs, **self-debug** demonstrates better performance than OpenHands CodeAct
- GPT-4o is an exception, as it is better at leveraging tools in OpenHands than other LLMs
- Other LLMs struggle with specialized bash commands to edit programs correctly
- GPT-4o may have been trained to better follow instructions and use complex tools like a web browser

**With vs. Without Expert-Provided Knowledge**:
- Expert-provided knowledge leads to consistent improvements on SR and CBS for most agents
- Agents can leverage helpful information in the knowledge to generate high-quality program drafts and use execution feedback to address errors
- There are performance decreases on VER for most agents, due to:
  - Specified tools that are less familiar to the agents
  - Agents generating executable but less meaningful programs without the knowledge
- Expert-provided knowledge helps agents generate more useful programs from a scientist user's perspective, as reflected by SR and CBS

#### Limitations of Language Agents in Complex Data-Driven Discovery Tasks

**Findings on Gold Programs and Task Error Rates:**
- **Distribution of lines**: red vertical line marks average length (58.6 lines) of all gold programs in benchmark (Figure [3](https://arxiv.org/html/2410.05080v1#S4.F3))
- **Task error rates**: sub-task category analysis in different disciplines:
  - Language agents struggle with complex tasks due to insufficient processing of data and inadequate use of specific tools (Figure [3](https://arxiv.org/html/2410.05080v1#S4.F3))
    * Bioinformatics: high failure rate in data processing and model development tasks due to heterogeneous data formats
    * Computational Chemistry: similar struggles with handling complex data and developing appropriate models
    * Geographical Information Science: tasks require specific tools like Geopandas for analysis, but current language models cannot use them effectively
    * Psychology & Cognitive Neuroscience: API usage is crucial for correct results, but existing LLMs generate incorrect or hallucinated code in this regard.

**Task Complexity and Failures:**
- Majority of succeeded tasks have simpler gold programs (less than 58.6 lines)
- Language agents still fail on many complex tasks with elaborate gold programs
- Data processing, model development, discipline-specific tools are key challenges for language agents in data-driven discovery.

### 4.2 Human Evaluation

**ScienceAgentBench: Human Evaluation of Claude-3.5-Sonnet**

**Evaluation Setup:**
- Rubric-based human evaluation of generated programs using expert-provided knowledge and rubrics
- Two evaluators rate each program, normalize scores to range 0–100
- Noise introduced by hiding task success outcomes
- Partial credits assigned for incorrect but useful programs

**Data Loading and Processing:**
- Successful programs receive perfect human ratings in data loading
- Failed programs have rating below 50 for some, distinguishable distribution

**Modeling or Visualization:**
- Human raters agree with SR metric on successful programs' implementation
- No difference found for output formatting and saving between groups

**Results and Analysis:**
- Data loading and processing distinguish successful and failed programs
- Modeling or visualization stage indicates agreement between human evaluators and SR metric
- Overlapped but distinguishable distributions of human ratings for both groups

**Conclusion:**
- Human evaluation complements outcome-based metrics in fine-grained assessment of language agents' performance.

## 5 Conclusion and Future Work

**ScienceAgentBench: A New Benchmark for Language Agents in Data-Driven Scientific Discovery**

**Introduction**:
- ScienceAgentBench introduced as new benchmark to evaluate language agents
- Compilation of 102 diverse, real-world tasks from 44 peer-reviewed publications across four scientific disciplines
- Engagement of nine subject matter experts for data quality assurance

**Experiments and Findings**:
- Best-performing agent, Claude-3.5-Sonnet with self-debug, can only solve 34.3% of the tasks using expert-provided knowledge
- Current language agents cannot yet automate tasks for data-driven discovery or a whole research pipeline

**Benchmark Advocacy and Future Research**:
- ScienceAgentBench to be used as testbed for developing future language agents with stronger capabilities
- Assessment of language agents for data-driven discovery through automatic graded metrics, such as an LLM judge based on task-specific rubrics

**Author Contributions**:
- Z. Chen: Project leadership, benchmark formulation, data collection and human evaluation, experiment implementation and manuscript writing
- S. Chen: Rubric-based evaluation design and implementation, optimization of evaluation scripts, and manuscript revision
- Y. Ning: Experiment execution with Llama 3.1 70B/405B
- Z. Chen, S. Chen, Y. Ning, Q. Zhang, B. Wang, B. Yu, Y. Li, Z. Liao, and C. Wei: Student annotators for benchmark collection, verification, and evaluation
- Z. Lu, V. Dey, M. Xue, F. Baker, B. Burns, D. Adu-Ampratwum, X. Huang, X. Ning, and S. Gao: Subject matter experts validating tasks and providing knowledge
- Y. Su and H. Sun: Senior authors overseeing the project, contributing to core ideas, and manuscript revision

**Acknowledgments**:
- Gratitude to colleagues from OSU NLP group and NSF-funded ICICLE AI Institute for constructive feedback
- Research sponsorship by NSF OAC 2112606, Amazon, and Ohio Supercomputer Center.

## Appendix A Limitations

**ScienceAgentBench: Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery**

**Benchmark Focus**:
- Data-driven discovery tasks formulated as code generation problems
- Automate tedious tasks in data processing, analysis, and visualization for scientists
- Evaluate language agents' capabilities as science co-pilots

**Code Generation Capability**:
- Formulate each task as a code generation problem
- Easy verifiability using established metrics
- Directly usable by scientists without additional efforts

**Future Studies**:
- Carefully examine other capabilities of language agents, e.g., summarizing literature, suggesting ideas, planning experiments
- Advocate rigorous, comprehensive assessments of one capability at a time

**Evaluation Methods**:
- Well-established evaluation methods, e.g., CodeBERTScore (Zhou et al., 2023), GPT-4o judge for figures (Wu et al., 2024; Yang et al., 2024b)
- Acknowledges limitations and encourages development of better automatic evaluation metrics or human evaluation protocols

**Diversity of Tasks, Disciplines, and Programs**:
- Strive to include diverse tasks and programs from different scientific disciplines
- Compromises: focus on Python programs, limit task processing time to 10 minutes
- Designed a principled, extensible data collection process and expert validation protocol
- Encourage future work to expand ScienceAgentBench with other languages and disciplines

**Ethical and Safety Considerations**:
- Adapt open-source code and data, respecting creators' ownership and intellectual property
- Cite original papers, list repositories, and provide licenses (Appendix I)
- Acknowledge limitations and welcome requests from authors to modify or remove tasks
- Agents developed with ScienceAgentBench should consider potential safety issues in deployment, e.g., adversarial attacks, confidential data leakage
- Users should have control over language agents through intervention and feedback mechanisms.

## Appendix B Related Work

**AI for Science:**
- **Deep learning** unlocks power of data for scientific discovery (Wang et al., 2023)
- Examples: AlphaFold predicts protein structures, language models in various disciplines (Yue et al., 2024; Yu et al., 2024; Labrak et al., 2024)
- Language agents needed to automate data-driven discovery end-to-end (Cao, 2017)

**Agents and Benchmarks for Task Automation:**
- Long-established challenge in AI research (Russell & Norvig, 2010)
- New generation of agents using LLMs for web navigation, software development, scientific discovery (Deng et al., 2023; Jimenez et al., 2024; Zheng et al., 2023; Lu et al., 2024)
- Evaluation of performance: TaskBench (Shen et al., 2024), SWE-Bench (Jimenez et al., 2024), BioCoder-Py (Tang et al., 2024b), ML-Bench (Tang et al., 2024a), MLAgentBench (Huang et al., 2024b)
- Scientific publications as task sources: SciCode (Tian et al., 2024), BLADE (Gu et al., 2024), DiscoveryBench-Real (Majumder et al., 2024b)
- ScienceAgentBench: individual task evaluations, focus on essential tasks in real-world data-driven discovery workflows.

## Appendix C Example Task Instructions

**Bioinfomatics and Computational Chemistry Task Instructions:**
* **Bioinformatics:**
  + Train cell counting model on BBBC002 datasets using Drosophila KC167 cells
  + Save test set predictions as "count" column in "cell-count_pred.csv"
  + Train drug-target interaction model using DAVIS dataset for antiviral drugs and COVID-19 target
  + Determine binding affinities between antiviral drugs and COVID-19 target
  + Rank antiviral drugs based on predicted affinities and save to "davis_dti_repurposing.txt"
  + Plot Tanimoto similarities of fingerprints for interaction fingerprints between a selected ligand and protein for the first 10 trajectory frames
  + Save png file as "ligand_similarity_pred.png"
  + Train Variational Autoencoder (VAE) model on given data and perform differential expression test for each cell type
  + Extract top markers using results and visualize them as a dotplot with dendrogram
  + Save figure to "hca_cell_type_de.png"
* **Computational Chemistry:**
  + Train multitask model on Clintox dataset for drug toxicity and FDA approval status
  + Save test set predictions, including SMILES representation of drugs and probability of positive labels, in "clintox_test_pred.csv"
  + Generate features for diffusion data based on material composition using SHAP feature selection approach
  + Select top 20 features and save as "mat_diffusion_features.csv"
  + Filter compounds in "hits.csv" and save SMILES representations of remaining ones to "compound_filter_results.txt"
  + Train graph convolutional network on given dataset for aquatic toxicity prediction
  + Compute atomic contributions to molecular activity using the resulting model and visualize as "aquatic_toxicity_qsar_vis.png"

**Geographical Information Science and Psychology & Cognitive Neuroscience Task Instructions:**
* **Geo Information Science:**
  + Analyze Elk movements, estimate home ranges, assess habitat preferences, identify spatial clusters of Elk movements
  + Document findings with maps and visualizations and save as "Elk_Analysis.png"
  + Analyze impact of land subsidence on flooding based on future elevation data
  + Identify flood-prone areas and estimate potential building damage for urban planning and mitigation strategies
  + Save results to "flooding_analysis.png"
  + Calculate deforestation area percentage in the Brazilian state of Rondônia within buffer zone of 5.5km around road layers
  + Save percentage result in "deforestation_rate.csv"
  + Load North America climate data in NetCDF file, extract temperature data, perform quadratic polynomial fit analysis, output fitting results by year in "polynomial_fit_pred.csv"
* **Psychology & Cognitive Neuroscience:**
  + Process and visualize ECG data through R peak detection and outlier correction
  + Plot overview of data and save final figure as "ecg_processing_vis1_pred_result.png"
  + Analyze IMU data collected during sleep, compute sleep endpoints: time of falling asleep, time of awakening, total duration spent sleeping
  + Save results in JSON file "imu_pred.json" with keys for sleep onset, wake onset, and total sleep duration, respectively
  + Analyze cognitive theories using pattern similarity, process CSV files containing model predictions for various syllogistic reasoning tasks
  + Calculate similarity scores between these models and pre-computed high-conscientiousness and high-openness patterns
  + Save results in "CogSci_pattern_high_sim_data_pred.csv"
  + Train linear model to learn mapping of neural representations in EEG signals from one subject (Sub 01) to another (Sub 03) based on preprocessed EEG data
  + Use test set of Subject 1 (Sub 01) to generate EEG signal for Subject 3 (Sub 03), save generated EEG signal as "linear_sub01tosub03_pred.npy"

## Appendix D More Details about Main Results

**ScienceAgentBench: Mean Performances of Language Agents for Data-Driven Scientific Discovery**

**Main Text Results**:
- Best performances from three independent runs presented

**Table D.1: Mean Performances without Expert Knowledge**
| Models | SR | CBS | VER | Cost ↓ |
| --- | --- | --- | --- | --- |
| Direct Prompting | Llama-3.1-Instruct-70B (3.6, 2.0) | 81.0 (0.4) | 22.2 (0.9) | 0.001 (0.000) |
| Direct Prompting | Llama-3.1-Instruct-405B (3.6, 0.5) | 79.3 (0.1) | 32.0 (0.5) | 0.011 (0.000) |
| ... | ... | ... | ... | ... |
| Self-Debug | Llama-3.1-Instruct-70B (7.2, 1.2) | 81.2 (0.3) | 67.3 (2.4) | 0.009 (0.000) |
| Self-Debug | Llama-3.1-Instruct-405B (8.8, 1.4) | 80.8 (0.5) | 67.0 (2.8) | 0.054 (0.005) |

**Table D.2: Mean Performances with Expert Knowledge**
| Models | SR | CBS | VER | Cost ↓ |
| --- | --- | --- | --- | --- |
| Direct Prompting | Llama-3.1-Instruct-70B (2.6, 0.5) | 81.7 (0.1) | 19.3 (1.7) | 0.001 (0.000) |
| Direct Prompting | Llama-3.1-Instruct-405B (2.9, 0.0) | 81.3 (0.0) | 24.5 (0.0) | 0.011 (0.000) |
| ... | ... | ... | ... | ... |
| Self-Debug | Llama-3.1-Instruct-70B (9.8, 2.1) | 82.0 (0.4) | 60.8 (2.1) | 0.011 (0.000) |
| Self-Debug | Llama-3.1-Instruct-405B (8.2, 0.9) | 82.2 (0.1) | 61.1 (3.8) | 0.072 (0.002) |

## Appendix E Case Studies

### E.1 Case 1: Action Space of OpenHands

**Findings on LLMs' Performance:**
* Four out of five Language Models (LLMs) demonstrate better performance than OpenHands CodeAct, except for GPT-4o
* **GPT-4o**: Performs well with web browser usage in OpenHands
	+ Uses browsing actions: goto(), click(), fill(), press() (Lines 31–61, 85–90)
* **Claude-3.5-Sonnet**: Struggles with specialized bash command to update program file without duplication (Listing E.2, Line 11)
	+ Ends up using Python function open() instead (Line 146)
* LLM-based agents may not benefit from large action spaces with complex tool usage like OpenHands design.

#### OpenHands Bash Command Struggles: Backward Feature Selection Example

**NeuroKit Documentation**

**Versions**
- [legacy_docs](https://neurokit2.readthedocs.io/en/legacy_docs/search.html)

**Downloads**
- PDF: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/pdf/>
- HTML: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/htmlzip/>
- Epub: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/epub/>

**Project Home (Read the Docs)**
- [NeuroKit](https://readthedocs.org/projects/neurokit2/)
- Builds: <//readthedocs.org/projects/neurokit2/builds/>
- Downloads: <//readthedocs.org/projects/neurokit2/downloads/>

**GitHub**
- [View](https://github.com/neuropsychology/NeuroKit/blob/legacy_docs/docs/search.rst)
- Edit: <https://github.com/neuropsychology/NeuroKit/edit/legacy_docs/docs/search.rst>

**Search**
- Please activate JavaScript to enable search functionality.

**Document Structure**
1. **Versions**
   - [legacy_docs](https://neurokit2.readthedocs.io/en/legacy_docs/search.html)
2. **Downloads**
   - PDF: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/pdf/>
   - HTML: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/htmlzip/>
   - Epub: <//neurokit2.readthedocs.io/_/downloads/en/legacy_docs/epub/>
3. **Project Home (Read the Docs)**
   - [NeuroKit](https://readthedocs.org/projects/neurokit2/)
   - Builds: <//readthedocs.org/projects/neurokit2/builds/>
   - Downloads: <//readthedocs.org/projects/neurokit2/downloads/>
4. **GitHub**
   - [View](https://github.com/neuropsychology/NeuroKit/blob/legacy_docs/docs/search.rst)
   - Edit: <https://github.com/neuropsychology/NeuroKit/edit/legacy_docs/docs/search.rst>
5. **Search**
   - Please activate JavaScript to enable search functionality.

**Code Example**
- Import necessary libraries
- Load data using pandas
- Preprocess features and target
  - Numeric features: StandardScaler()
  - Categorical features: OneHotEncoder()
- Binarize target variable
- Split data into training and testing sets
- Create preprocessing steps (ColumnTransformer)
- Define pipeline with preprocessing and logistic regression model
- Perform backward feature selection using SequentialFeatureSelector (SFS)
  - Get results
  - Plot the results
- Save the best feature subset and cross-validation accuracy
- Evaluate on test set using best feature subset
- Print information about the best feature subset, selected features, and test accuracy.

#### Code Cleanup and Duplicate Removal in IPython Notebook

**Interactive Assistant**: I've extracted the key details from your conversation with the agent, focusing on the interaction between you and the agent. Here is a summary:

**Your Interaction with the Agent:**
- You requested the agent to clean up the code in your file by removing duplicate content and correcting syntax errors.
- The agent executed several IPython cells containing the code changes.
- After each code execution, the agent reported whether the code was executed successfully or if there were any issues (e.g., incomplete input or syntax errors).

**Agent's Response:**
- The agent acknowledged your requests and made the necessary adjustments to the code by cleaning it up and removing duplicates.
- When encountering syntax errors, the agent informed you of the issue and suggested correcting the code before executing it again.

**Code Updates:**
- Several blocks of code were provided as examples for updates to your file. These included:
  - Importing necessary libraries (pandas, numpy, etc.)
  - Reading in data from a CSV file and splitting it into features (X) and target variable (y)
  - Preprocessing the data using StandardScaler and OneHotEncoder for numeric and categorical columns respectively
  - Creating a pipeline with preprocessing steps and a LogisticRegression model
  - Performing backward feature selection using SequentialFeatureSelector
- The agent also suggested directly editing the file content if you encounter persistent issues.

**Observations:**
- After each code execution, the agent reported whether the code was executed successfully or if there were any issues, such as incomplete input or syntax errors.

**File Updates:**
- After several iterations of requesting code updates and observing syntax errors, the agent successfully updated the file content with the provided blocks of code.

### E.2 Case 2: Influence of Expert-Provided Knowledge

**Case Study: Influence of Expert-Provided Knowledge on Claude-3.5-Sonnet**
* **Problem**: Claude-3.5-Sonnet generates less accurate programs without expert-provided knowledge, leading to poor performance in data-driven scientific discovery tasks.
* **Solution**: Provide the agent with expert knowledge to help it develop more effective models.

**Impact of Expert Knowledge**
1. Improved tool usage: Expert knowledge helps Claude-3.5-Sonnet use advanced tools like DeepPurpose, leading to better drug discovery models.
2. Avoiding data contamination: While some generated programs using expert knowledge may contain signs of data contamination, the overall approach improves the agent's performance and resemblance to ground truth programs.

**Case Study Details**
* **Without Expert Knowledge**: Claude-3.5-Sonnet uses basic tools like pandas and sklearn for drug discovery but generates inaccurate results (Listing E.3).
* **With Expert Knowledge**: An expert provides knowledge on encoding methods for drugs and targets, enabling the agent to install DeepPurpose and develop advanced neural network models (Listing E.4). However, some data contamination issues remain.

**Listings**
1. Listing E.3: Program generated by Claude-3.5-Sonnet without expert knowledge.
2. Listing E.4: Program generated by Claude-3.5-Sonnet with expert knowledge and DeepPurpose installation.
3. Expert knowledge provided: "Typical encoders for a drug include ECFP fingerprint or message-passing neural networks over the 2D molecular graph representation, and for a target include 1D convolutional neural networks over the amino acid sequence."

#### "Predicting Repurposed Antiviral Drugs using Deep Learning"

**Installing Packages**
- If package is 'rdkit':
  - Install using `pip install rdkit-pypi`
- Otherwise:
  - Install using `pip install <package>`

**Importing Modules and Creating Directories**
- Import necessary modules (`numpy`, `pandas`, `DeepPurpose.DTI`, `DeepPurpose.utils`, `DeepPurpose.dataset`)
- Create 'pred_results' directory if it doesn't exist

**Loading and Preprocessing DAVIS Dataset**
- Load and process DAVIS dataset from 'benchmark/datasets/dti/DAVIS' path
  - Binary: False
  - Convert_to_log: True

**Converting Numpy Arrays to Pandas DataFrames**
- Create 'train' DataFrame with drug, target, and label data

**Loading Drug-Target Interaction Model**
- Generate configuration for the model
  - Drug encoding: MPNN
  - Target encoding: CNN
  - Classifier hidden dimensions: [1024, 1024, 512]
  - Train epochs: 5
- Initialize the model using the generated configuration

**Training the Model**
- Train the model on the 'train' DataFrame

**Loading Antiviral Drugs and COVID-19 Target Sequence**
- Load antiviral drugs from a tab-separated file
- Read COVID-19 target sequence from a text file

**Preparing Data for Prediction**
- Generate drug representations for repurposing
  - Use SMILES strings of antiviral drugs
  - Repeat COVID-19 target sequence for each drug

**Making Predictions**
- Use the trained model to make predictions on the drug representations and target sequence

**Creating Results DataFrame and Sorting by Affinity**
- Create a DataFrame with drug SMILES, names, and predicted affinities
- Sort the DataFrame by descending order of affinity (lowest values first)

**Saving Results**
- Save the sorted results to a file 'davis_dti_repurposing.txt' in the 'pred_results' directory

**Ground-truth Program in the Benchmark**
1. Import necessary modules and libraries (DeepPurpose, Pathlib, Shutil, Os, Json, Numpy, Pandas)
2. Define functions: `make_dataset` and `main`
3. In `make_dataset` function:
  - Read drug files and process data
  - Read affinity file and process data
  - Combine SMILES, target sequence, and affinity into a single array
  - Convert unit of affinity from nM to pH scale
4. In `main` function:
  - Load target sequence from JSON file
  - Generate training and validation datasets using the 'make_dataset' function
5. Initialize configuration for the model (drug encoding, target encoding)
6. Initialize the model using the generated configuration
7. Train the model on training data
8. Preprocess COVID-19 target sequence and antiviral drugs to make predictions using the `repurpose` function
9. Save results to a file 'result/repurposing.txt' in the 'pred_results' directory
10. Run the main function if the script is executed as the main module.

## Appendix F Expert Validation Details

**Expert Validation Process Details (Section [2.2](https://arxiv.org/html/2410.05080v1#S2.SS2 "2.2 Data Collection ‣ 2 ScienceAgentBench ‣ ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery"))**
* **Questionnaire for Subject Matter Experts (F.1)**
  * Included in expert validation process
  * Detailed questionnaire provided
* **Program Example for Subject Matter Experts (F.2)**
  * Two examples used in the questionnaire
* **Knowledge Example Provided to Subject Matter Experts during Annotation (F.3)**
  * Additional information for experts during annotation process.

### F.2 Program Example for Subject Matter Experts

**Task Instruction:**
- Train a Graph Convolutional Network (GCN) on the given dataset to predict aquatic toxicity of compounds using DeepChem library
- Use the resulting model to compute and visualize atomic contributions for a test example compound
- Save the figure as "pred_results/aquatic_toxicity_qsar_vis.png"

**Program Structure:**
1. Import required libraries: RDKit, TensorFlow (implicit in DeepChem), Pandas
2. Define functions for handling molecular structures and calculating atomic contributions based on the similarity maps
3. Define main function containing steps to load data, prepare model, train model, predict and visualize results
4. Run the main function

**Functions:**
1. `vis_contribs(mol, df, smi_or_sdf="sdf")`:
   - Calculates atomic contributions for a given molecular structure using DeepChem's similarity maps based on the provided dataset and smoothing method (SMILES or SDF format)
2. `main()`:
   - Loads data from the specified dataset file
   - Prepares the model for training
   - Trains the GCN model using DeepChem's SDFLoader
   - Predicts the aquatic toxicity for the test example compound and its fragments
   - Merges predictions for compounds and fragments to calculate atomic contributions
   - Visualizes the results as an image file "pred_results/aquatic_toxicity_qsar_vis.png"

**Steps:**
1. Set up environment variables for TensorFlow (if necessary)
2. Import required libraries: RDKit, Pandas, DeepChem and its components (SDF loader, featurizer, transformer, etc.)
3. Define functions `vis_contribs` and `main` as described above
4. In the main function:
   - Load data from the training dataset using SDFLoader
   - Prepare the model for training by defining hyperparameters, creating an instance of GraphConvModel, and fitting it to the training data
   - Load data from the test dataset and prepare atomic contributions using the trained model
   - Merge predictions for compounds and fragments to calculate atomic contributions
   - Visualize the results using RDKit's `SimilarityMaps.GetSimilarityMapFromWeights()` function and save it as a PNG image with appropriate parameters.
5. Call the main function at the end of your script.

### F.3 Knowledge Example Provided to Subject Matter Experts during Annotation

**Task Instruction:**
- Train a graph convolutional network (GCN) on the given dataset to predict aquatic toxicity of compounds using DeepChem library.
- Compute and visualize atomic contributions for a test example compound.

**Program Structure:**
1. Import necessary libraries, including RDKit and TensorFlow's Keras.
2. Define functions:
   - `vis_contribs(mol, df, smi_or_sdf)`: Visualize atomic contributions using RDKit.
3. Main function `main()`:
   - Load dataset from file and train GCN model using DeepChem.
   - Load test example compound file and create corresponding dataset.
   - Use trained GCN to predict toxicity of test compound and its fragments.
   - Merge results, calculate atomic contributions, and visualize them.
4. Run `main()` function if the script is executed as a standalone program.

**Domain Knowledge:**
- IGC50: Gold label for aquatic toxicity in the dataset.
- RDKit's `SimilarityMaps.GetSimilarityMapFromWeights()` function is used to visualize atomic contributions.
- Atomic contributions can be calculated by predicting the toxicity of the complete compound and its fragments (with one atom removed), then finding the difference between them.

## Appendix G Rubric Examples

**ScienceAgentBench: Rubric Examples**

**Computational Chemistry:**
- **Data Loading**:
  * Initialize Data Loader for Training (5 points)
    - Correctly initializes MyClintoxLoader object with correct parameters (5 points)
    - Initializes featurizer='ECFP', tasks=['FDA_APPROVED', 'CT_TOX'], feature_field='smiles', and file path 'benchmark/datasets/clintox/clintox_train.csv' (5 points)
  * Load Training Dataset (5 points)
    - Successfully loads training dataset using train_loader object and assigns to train_dataset (5 points)
  * Initialize Data Loader for Testing (5 points)
    - Correctly initializes MyClintoxLoader object with correct parameters (5 points)
    - Initializes featurizer='ECFP', tasks=['FDA_APPROVED', 'CT_TOX'], feature_field='smiles', and file path 'benchmark/datasets/clintox/clintox_test.csv' (5 points)
  * Load Testing Dataset (5 points)
    - Successfully loads test dataset using test_loader object and assigns to test_dataset (5 points)
- **Data Processing**:
  * Transform Data (5 points)
    - Applies required transformations to training dataset with specified transformers (e.g., 'balancing') (5 points)
- **Modeling, Analysis, or Visualization**:
  * Initialize Model (10 or 15 points)
    - Successfully initializes MultitaskClassifier with correct parameters: number of tasks equal to length of CLINTOX_TASKS, n_features=1024, layer_sizes=[1000], dropouts=[0.25], learning_rate=0.001, and batch_size=50 (10 or 15 points)
  * Fit Model (10 points)
    - Successfully fits the model using train_dataset (10 points)
  * Predict Using Model (10 points)
    - Correctly uses trained model to predict scores on test_dataset with necessary test_transformers (10 points)
- **Output Formatting**:
  * Format Output DataFrame (5 points)
    - Creates pandas DataFrame named 'test_scores_df' containing 'smiles', 'FDA_APPROVED', and 'CT_TOX' columns with correctly assigned test scores (5 points)
- **Output Saving**:
  * Save Predictions to CSV (5 points)
    - Correctly saves test_scores_df to a CSV file at 'pred_results/clintox_test_pred.csv' without an index (5 points)
- **Total Points**: 65 for GPT-4o generated rubric, 75 for expert revised rubric.

**Geographical Information Science:**
- **Data Loading**:
  * Load Bathymetry Data (10 points)
    - Correctly loads bathymetry raster data from 'benchmark/datasets/CoralSponge/CatalinaBathymetry.tif' (10 points)
  * Load Coral and Sponge Data (10 points)
    - Correctly reads coral and sponge data from 'benchmark/datasets/CoralSponge/CoralandSpongeCatalina.geojson' (10 points)
  * CRS Transformation (5 points)
    - Correctly transforms CRS of the GeoDataFrame to EPSG:4326 (5 points)
- **Data Processing**:
  * Elevation Conversion (10 points)
    - Correctly converts elevation values by multiplying with -1 (10 points)
  * Calculate Gradient (10 points)
    - Accurately calculates gradient (grad_x, grad_y) using numpy’s gradient function (10 points)
  * Calculate Slope and Aspect (10 points)
    - Correctly calculates slope in degrees from gradients and adjusts any negative values for aspect calculation.

### Geographical Data Processing Rubric Reduction Example

**Rubric for Geographical Information Science Task**

**Data Loading**:
- **Load Bathymetry Data**: Correctly loads bathymetry raster data from `benchmark/datasets/CoralSponge/CatalinaBathymetry.tif` (5 points)
- **Load Coral and Sponge Data**: Correctly reads coral and sponge data from `benchmark/datasets/CoralSponge/CoralandSpongeCatalina.geojson` (5 points)
- **CRS Transformation**: Correctly transforms CRS of GeoDataFrame to EPSG:4326 (5 points)

**Data Processing**:
- **Elevation Conversion**: Correctly converts elevation values by multiplying with -1 (5 points)
- **Calculate Gradient**: Accurately calculates gradient using numpy's gradient function (5 points)
- **Calculate Slope**: Correctly calculates slope in degrees from gradients (10 points)
- **Calculate Aspect**: Correctly calculates aspect in degrees and adjusts negative values (10 points)
- **Coordinate to Raster Index Conversion**: Correctly implements function to convert coordinates to raster grid indices (5 points)
- **Extract Slope and Aspect**: Extracts slope and aspect values for each point in GeoDataFrame correctly (10 points)
- **Add Slope and Aspect to GeoDataFrame**: Successfully adds extracted slope and aspect values as new columns (5 points)
- **Group by VernacularNameCategory**: Correctly groups GeoDataFrame by `VernacularNameCategory` and computes mean values for slope and aspect (5 points)

**Modeling, Analysis, or Visualization**:
- **Bar Plot for Mean Slope**: Correctly creates bar plot showing mean slope per species (10 points)
- **Bar Plot for Mean Aspect**: Correctly creates bar plot showing mean aspect per species (10 points)

**Output Formatting**:
- **Plot Descriptions**: Properly sets plot titles, axis labels, and ensures x-ticks are rotated for readability (5 points)

**Output Saving**:
- **Save Plots**: Saves plots as `mean_slope_per_species.png`, `mean_aspect_per_species.png`, and `pred_results/CoralandSponge.png` (5 points)

**Total Points**: 100

## Appendix H Prompt Templates

**ScienceAgentBench: Prompt Templates for Language Agents**

**Direct Prompting (Table H.1)**
- You are an expert Python programming assistant that helps scientist users write high-quality code to solve their tasks
- Write a complete program that accomplishes the requested task and saves any outputs in the correct format
- Wrap your program in a code block specifying script type (e.g., 'python')
- Keep responses concise, no code execution required unless intended
- Address user issues by providing fixed, complete programs
- Do not suggest partial solutions or interactive Python commands
- User request: {task_instruction}
- Domain knowledge: {domain_knowledge}, optional
- Dataset access: {dataset_path}, directory structure previewed

**Self-Debug (Table H.2)**
- Similar to Direct Prompting, but user may report exceptions and errors
- Address reported issues by providing fixed, complete programs

**OpenHands CodeAct (Table H.3)**
- You are an expert Python programming assistant that helps scientist users write high-quality code to solve their tasks
- Write a complete program that accomplishes the requested task and saves any outputs to '/workspace/pred_results/' in the correct format
- Save your program as '/workspace/pred_programs/{pred_program_name}' before running it to check and fix errors
- Do not run programs in the background, address compatibility issues without restarting environment.

## Appendix I Publications, Repositories, and Licenses

**Publications and Repositories (I.1-I.5)**

**Table I.1: Bioinformatics and Computational Chemistry Publications**
- **Automated Inference of Chemical Discriminants** [Raschka et al. (2018)](https://arxiv.org/html/2410.05080v1#bib.bib59)
- **CellProfiler: image analysis software for identifying and quantifying cell phenotypes** [Carpenter et al. (2006)](https://arxiv.org/html/2410.05080v1#bib.bib10)
- **DeepPurpose: A Deep Learning Library for Drug-Target Interaction Prediction** [Huang et al. (2020)](https://arxiv.org/html/2410.05080v1#bib.bib33)
- **ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries** [Swanson et al. (2024)](https://arxiv.org/html/2410.05080v1#bib.bib69)
- **Prediction and mechanistic analysis of drug-induced liver injury (DILI) based on chemical structure** [Liu et al. (2021)](https://arxiv.org/html/2410.05080v1#bib.bib46)
- **SCANPY: large-scale single-cell gene expression data analysis** [Wolf et al. (2018)](https://arxiv.org/html/2410.05080v1#bib.bib80)
- **A Python library for probabilistic analysis of single-cell omics data** [Gayoso et al. (2022)](https://arxiv.org/html/2410.05080v1#bib.bib26)
- **MUON: multimodal omics analysis framework** [Bredikhin et al. (2022)](https://arxiv.org/html/2410.05080v1#bib.bib7)
- **Scirpy: a Scanpy extension for analyzing single-cell T-cell receptor-sequencing data** [Sturm et al. (2020)](https://arxiv.org/html/2410.05080v1#bib.bib68)
- **The scverse project provides a computational ecosystem for single-cell omics data analysis** [Virshup et al. (2023)](https://arxiv.org/html/2410.05080v1#bib.bib73)

**Table I.2: Geographical Information Science and Psychology & Cognitive Neuroscience Publications**
- **eofs: A Library for EOF Analysis of Meteorological, Oceanographic, and Climate Data** [Dawson (2016)](https://arxiv.org/html/2410.05080v1#bib.bib16)
- **The Open Global Glacier Model (OGGM) v1.1** [Maussion et al. (2019)](https://arxiv.org/html/2410.05080v1#bib.bib53)
- **Human selection of elk behavioural traits in a landscape of fear** [Ciuti et al. (2012)](https://arxiv.org/html/2410.05080v1#bib.bib15)
- **Investigating the preferences of local residents toward their landscapes** [Ziedan et al.](https://arxiv.org/html/2410.05080v1#bib.bib74)

### Repositories & Licenses for Data-Driven Scientific Research Tools

**Study Findings:**
- **Chattanooga's proposed bus network redesign** ([2021](https://arxiv.org/html/2410.05080v1#bib.bib97))
- **Urban wildlife corridors: Building bridges for wildlife and people** (Zellmer & Goto, [2022](https://arxiv.org/html/2410.05080v1#bib.bib91))
- **Impact of urban climate on extreme temperatures in Madison, Wisconsin, USA** (Schatz & Kucharik, [2015](https://arxiv.org/html/2410.05080v1#bib.bib64))

**Geospatial Analysis:**
- **Model Animal Home Range** (Fleming, [2024](https://arxiv.org/html/2410.05080v1#bib.bib25))
- **Run geoprocessing tools with Python** (Zandbergen, [2024](https://arxiv.org/html/2410.05080v1#bib.bib90))
- **Model how land subsidence affects flooding** (Andeweg & Kuijpers, [2024](https://arxiv.org/html/2410.05080v1#bib.bib1))
- **Predict deforestation in the Amazon rain forest** (ESRI, [2024a](https://arxiv.org/html/2410.05080v1#bib.bib21))
- **NOAA Deep Sea Corals Research and Technology Program** (Hourigan, [2023](https://arxiv.org/html/2410.05080v1#bib.bib31))
- **Chart coral and sponge distribution factors with Python** (Robinson, [2023](https://arxiv.org/html/2410.05080v1#bib.bib62))
- **Assess access to public transit** (ESRI, [2024b](https://arxiv.org/html/2410.05080v1#bib.bib22))
- **Build a model to connect mountain lion habitat** (ESRI, [2024c](https://arxiv.org/html/2410.05080v1#bib.bib23))
- **Analyze urban heat using kriging** (Krause, [2024](https://arxiv.org/html/2410.05080v1#bib.bib41))
- **Assess burn scars with satellite imagery** (ESRI, [2024d](https://arxiv.org/html/2410.05080v1#bib.bib24))

**Psychology & Cognitive Neuroscience:**
- **BioPsyKit: A Python package for the analysis of biopsychological data** (Richer et al., [2021](https://arxiv.org/html/2410.05080v1#bib.bib60))
- **NeuroKit2: A Python toolbox for neurophysiological signal processing** (Makowski et al., [2021](https://arxiv.org/html/2410.05080v1#bib.bib51))
- **Modeling Human Syllogistic Reasoning: The Role of “No Valid Conclusion”** (Riesterer et al., [2019](https://arxiv.org/html/2410.05080v1#bib.bib61))
- **Analyzing the Differences in Human Reasoning via Joint Nonnegative Matrix Factorization** (Brand et al., [2020](https://arxiv.org/html/2410.05080v1#bib.bib6))
- **Generate your neural signals from mine: individual-to-individual EEG converters** (Lu & Golomb, [2023](https://arxiv.org/html/2410.05080v1#bib.bib48))