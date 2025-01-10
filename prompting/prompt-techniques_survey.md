# The Prompt Report: A Systematic Survey of Prompting Techniques

by Sander Schulhoff, Michael Ilie1, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhoff, Pranav Sandeep Dulepet, Saurav Vidyadhara, Dayeon Ki, Sweta Agrawal, Chau Pham, Gerson Kroiz Feileen Li, Hudson Tao, Ashay Srivastava, Hevander Da Costa, Saloni Gupta, Megan L. Rogers, Inna Goncearenco, Giuseppe Sarli, Igor Galynker, Denis Peskoff, Marine Carpuat, Jules White, Shyamal Anadkat, Alexander Hoyle, Philip Resnik

https://arxiv.org/pdf/2406.06608

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
  - [1.2 Terminology](#12-terminology)
- [2 A Meta-Analysis of Prompting](#2-a-meta-analysis-of-prompting)
  - [2.1 Systematic Review Process](#21-systematic-review-process)
  - [2.2 Text-Based Prompting Techniques Taxonomy](#22-text-based-prompting-techniques-taxonomy)
  - [2.3 Prompting Technique](#23-prompting-technique)
  - [2.4 Prompt Engineering](#24-prompt-engineering)
  - [2.5 Answer Engineering](#25-answer-engineering)
- [3 Beyond English Text Prompting](#3-beyond-english-text-prompting)
  - [3.1 Multilingual Prompting Techniques](#31-multilingual-prompting-techniques)
  - [3.2 Multimodal](#32-multimodal)
- [4 Extensions of Prompting](#4-extensions-of-prompting)
  - [4.1 Agents](#41-agents)
  - [4.2 Evaluation](#42-evaluation)
- [5 Prompting Issues](#5-prompting-issues)
  - [5.1 Security Concerns](#51-security-concerns)
  - [5.2 Alignment](#52-alignment)
- [6 Benchmarking](#6-benchmarking)
  - [6.1 Benchmarking Prompting Techniques](#61-benchmarking-prompting-techniques)
  - [6.2 Prompt Engineering Case Study](#62-prompt-engineering-case-study)
- [7 Related Work](#7-related-work)
- [8 Conclusions](#8-conclusions)

## Abstract
**Generative Artificial Intelligence (GenAI) Systems and Prompting Techniques:**

**Background:**
- GenAI systems widely deployed across industries and research settings
- Developers and end users interact through prompting or prompt engineering
- Nascency of area leads to conflicting terminology and poor ontological understanding of prompts

**Contribution:**
- Establish structured understanding of prompts
- Assemble taxonomy of prompting techniques
- Analyze their use

**Taxonomy of Prompting Techniques:**
- 58 text-only prompting techniques identified
- 40 techniques for other modalities

**Comprehensive Vocabulary:**
- 33 vocabulary terms presented

**Meta-Analysis:**
- Entire literature on natural language prefix-prompting analyzed

**Findings:**
- Taxonomy of prompting techniques
- Comprehensive vocabulary for understanding prompts

**Implications:**
- Improved understanding of prompting techniques in GenAI systems
- Better interaction between developers and end users.

## 1 Introduction

**Transformer-based Language Models (LLMs)**
* Widely deployed in consumer-facing, internal, and research settings
* User provides input "prompt" to which the model produces an output
* Prompts can be textual or other forms: images, audio, videos
* Ability to prompt models makes them easy to interact with and use flexibly across various use cases
* Effective use of prompts leads to improved results
* Large body of literature growing around prompting techniques
* Poorly understood field with rapidly increasing techniques

**Scope of Study**
* Focus on discrete prefix prompts rather than cloze prompts or soft prompts
* Only study task-agnostic techniques, not specific ML/MMT models
* Limit to hard (discrete) prompts and techniques using prefix prompts
* Exclude techniques using gradient-based updates (fine-tuning)

**Categories within Prompting**
1. **Text-based Techniques**: Discussed in Section 2
2. **Multilingual Techniques**: Discussed in Section 3.1
3. **Multimodal Techniques**: Discussed in Section 3.2
4. **Agent Techniques**: Discussed in Section 4
5. **Evaluation Techniques**: Discussed in Section 4.2
6. **Security and Safety Measures**: Discussed in Sections 5.1 and 5.2
7. **Case Studies**: Discussed in Section 6
8. **Conclusion**: Discussed in Section 8

**Prompting Techniques Taxonomy**
* Conducted machine-assisted systematic review using PRISMA process to identify 58 different text-based prompting techniques
* Create a taxonomy with robust terminology of prompting terms (Section 1.2)

**Multilingual Techniques**
* Discussed in Section 3.1

**Multimodal Techniques**
* Discussed in Section 3.2
* Many multilingual and multimodal techniques are extensions of English text-only prompting techniques

**Agent Techniques**
* Use external tools like Internet browsing and calculators (Section 4.1)
* Need to understand how to evaluate outputs and avoid hallucinations (Section 4.2)

**Evaluation and Security Measures**
* Evaluating outputs: Section 4.2
* Security measures: Section 5.1
* Safety measures: Section 5.2

**Case Studies**
1. Test range of prompting techniques against common benchmark MMLU (Section 6.1)
2. Explore manual prompt engineering on a significant real-world use case (Section 6.1)

**Conclusion**
* Discuss the nature of prompting and its recent development (Section 8)

### 1.2 Terminology
#### 1.2.1 Components of a Prompt

**Components of a Prompt**
- **Directive**: Instruction or question issued in a prompt, serving as its core intent. Examples: "Tell me five good books to read", "Perform English to Spanish translation".
- **Example/Exemplar**: Demonstrations guiding the GenAI to accomplish tasks. One-shot case: "Write a clear and curt paragraph about llamas".
- **Output Formatting**: Instructions for the GenAI to output information in specific formats like CSV or markdown formats (Xia et al., 2024).

**Prompting Techniques**
- **Style Instructions**: Output formatting used to modify style rather than structure. Example: "Write a limerick about llamas, pretending you are a shepherd".
- **Role/Persona**: Improves writing and style text by specifying a role for the GenAI to adopt. Example: "Pretend you are a shepherd and write a limerick about llamas".
- **Additional Information**: Necessary details included in a prompt, sometimes called 'context'. Examples: Name and position when writing an email.

#### 1.2.2 Prompting Terminology
- Prompt: The process of providing a prompt to a GenAI, resulting in a response.
- Prompt Chain: Consists of multiple prompt templates used sequentially, where the output of one is used to parameterize the next.
- Prompt Engineering: Iterative process of developing and modifying prompts to improve performance.
- Exemplar: Examples of tasks completed shown to a model in a prompt.

## 2 A Meta-Analysis of Prompting 
### 2.1 Systematic Review Process

**Systematic Review Process for Prompting Techniques**
* Collection of Data Set:
  + Query databases (arXiv, Semantic Scholar, ACL) using keywords related to prompting and prompt engineering (Appendix A.4)
  + Retrieve articles based on PRISMA process from arXiv as initial sample for filtering criteria
* Filtering Criteria:
  1. Propose novel prompting technique? (include)
  2. Strictly covers hard prefix prompts? (include)
  3. Focuses on training by backpropagating gradients? (exclude)
  4. Uses masked frame and/or window for non-text modalities? (include)
* Human Annotation:
  + Labeling of 1,661 articles from arXiv set based on filtering criteria with 92% agreement between annotators
  + Developing a prompt using GPT-4-1106-preview to classify remaining articles
  + Validating the prompt against 100 ground-truth annotations, achieving 89% precision, 75% recall, and F1 score of 81%
* Combined Human and LLM Annotation:
  + Generating a final set of 1,565 relevant papers for analysis

### 2.2 Text-Based Prompting Techniques Taxonomy
* Overview:
  + 58 text-based prompting techniques categorized into six major categories (Figure 2.2)
* Categories and Examples:
  1. **Prompt Design**: Fine-tuning, Hyperparameter Optimization, Prompt Tuning, Prompt Selection, Prompt Engineering
  2. **Task Formulation**: Zero-shot Learning, Few-shot Learning, One-shot Learning
  3. **Model Customization**: Pretraining, Finetuning, Adaptive Prompting
  4. **Data Augmentation**: Data Filtering, Data Smoothing, Data Manipulation, Synthetic Data Generation
  5. **Interactive Prompts**: User Interaction, Human Feedback, In-context Learning
  6. **Ensembling Methods**: Multi-head Attention, Mixture of Experts, Neural Ensembles
* Some techniques may fit into multiple categories; place them in the category of most relevance.

#### 2.2.1 In-Context Learning (ICL)

**In-Context Learning (ICL) vs. Few-Shot Prompting:**
* **ICL**: Ability of GenAIs to learn skills/tasks using exemplars or instructions within the prompt without weight updates/retraining
* ICL can be optimized and understood significantly (Bansal et al., 2023; Si et al., 2023a; Štefánik and Kadl’cík, 2023)
* Exemplar Generation: SG-ICL, Exemplar Ordering, Exemplar Selection (KNN, Vote-K), Thought Generation
* Chain-of-Thought (CoT): Zero-Shot CoT, Analogical Prompting, Step-Back Prompting, Thread-of-Thought (ThoT), Tab-CoT, Few-Shot CoT
* Ensembling: COSP, DENSE, DiVeRSe, Max Mutual Information, Meta-CoT, MoRE, Self-Consistency, Universal Self-Consistency, USP
* Text-based prompting techniques: Chain-of-Verification, Self-Calibration, Self-Refine, Self-Verification, ReverseCoT, Cumulative Reason, Decomposition, DECOMP, Faithful CoT, Least-to-Most, Plan-and-Solve, Program-of-Thought, Recursive Thought, Skeleton-of-Thought, Tree-of-Thought
* Exemplar Distribution: Balanced vs. Random
* Exemplar Quality: Correct Labeling
* Exemplar Similarity: Select similar instances
* Exemplar Format: Choose a common format
* Exemplar Quantity: Include as many examples as possible
* Examples (Text): {TEXT}
* Few-Shot Learning (FSL): Broader machine learning paradigm to adapt parameters with few examples
* Few-Shot Prompting: Specific to prompts in GenAI settings, no weight updates/retraining
* Conflated terms: ICL vs. FSL

**2.2.1.1 Few-Shot Prompting Design Decisions**

**Factors Influencing Few-Shot Prompting:**
* **Exemplar Quantity**: More exemplars generally improve performance, especially for larger models; diminishing returns beyond 20.
* **Exemplar Ordering**: Affects model behavior and accuracy.
* **Exemplar Label Distribution**: Distribution of labels in prompt affects model behavior.
* **Exemplar Label Quality**: Necessity of valid demonstrations is unclear, larger models handle incorrect labels better.
* **Exemplar Format**: Optimal format may vary per task; common formats include "Q: {input}, A: {label}" or "input: output".
* **Exemplar Similarity**: Selecting similar exemplars benefits performance, but more diverse ones can also improve it.

**Few-Shot Prompting Techniques:**
* **K-Nearest Neighbor (KNN)**: Selects exemplars similar to the test sample for boosted performance. Can be time and resource intensive.
* **Vote-K**: Selects unlabeled candidate exemplars for annotators to label, then uses labeled pool for few-shot prompting. Ensures diversity and representativeness.
* **Self-Generated In-Context Learning (SG-ICL)**: Uses a GenAI to automatically generate exemplars when training data is unavailable but not as effective as actual data.
* **Prompt Mining**: Discovers optimal "middle words" in prompts through corpus analysis for improved performance.
* **More Complicated Techniques**: LENS, UDR, and Active Example Selection use iterative filtering, embedding and retrieval, and reinforcement learning respectively.

#### 2.2.2 Zero-Shot

**Zero-Shot Prompting Techniques**

**Role Promoting:**
- Assigns specific roles to GenAI in the prompt (e.g., "Madonna", "travel writer")
- Creates desirable outputs for open-ended tasks and improves accuracy on benchmarks

**Style Promoting:**
- Specifies desired style, tone, or genre in the prompt
- Similar effect as role prompting

**Emotion Prompting:**
- Incorporates phrases of psychological relevance to humans into the prompt
- Improves LLM performance on benchmarks and open-ended text generation

**System 2 Attention (S2A):**
- Asks an LLM to rewrite the prompt and remove unrelated information
- Passes new prompt into an LLM for final response

**SimToM:**
- Deals with complicated questions involving multiple people or objects
- Establishes set of facts one person knows, then answers question based on those facts
- Two-prompt process helps eliminate irrelevant information in the prompt

**Rephrase and Respond (RaR):**
- Instructs LLM to rephrase and expand the question before generating final answer
- Demonstrates improvements on multiple benchmarks

**Reloading:**
- Adds "Read the question again:" phrase to the prompt in addition to repeating the question
- Shows improvement in reasoning benchmarks, especially with complex questions

**Self-Ask:**
- Prompts LLMs to first decide if they need to ask follow-up questions for a given prompt
- If so, generates these questions and then answers them before answering the original question

#### 2.2.3 Thought Generation:
**Chain-of-Thought (CoT) Prompting:**
- Encourages LLM to express thought process before delivering final answer
- Demonstrates significant enhancements in math and reasoning tasks
- Exemplar includes question, reasoning path, and correct answer (Figure 2.8)

**Zero-Shot CoT:**
- Appends thought inducing phrases like "Let's think step by step" to the prompt
- Attractive as it doesn't require exemplars and is generally task agnostic
- Step-Back Promoting is a modification of CoT where LLM is asked a high-level question before delving into reasoning
- Analogical Promoting generates exemplars with CoTs for mathematical reasoning and code generation tasks
- Thread-of-Thought Promoting consists of improved thought inducer for CoT, "Walk me through this context in manageable parts step by step"
- Tabular Chain-of-Thought (Tab-CoT) makes LLM output reasoning as a markdown table to improve structure and reasoning

**Few-Shot CoT:**
- Presents multiple exemplars with chains-of-thought to enhance performance significantly
- Contrastive CoT Prompting adds incorrect and correct explanations to the CoT prompt to show LLM how not to reason
- Uncertainty-Routed CoT Promoting samples multiple reasoning chains and selects majority if above a certain threshold
- Complexity-based Promoting selects complex examples for annotation and uses majority vote among chains exceeding a length threshold
- Automatic Chain-of-Thought (Auto-CoT) Promoting generates chains of thought automatically to build Few-Shot CoT prompts for test samples.

#### 2.2.4 Decomposition

**Decomposition Techniques for LLMs**

**Decomposition**:
- Significant research on decomposing complex problems into simpler sub-questions
- Effective problem-solving strategy for both humans and GenAI

**Least-to-Most Prompting (L2MP)**:
- Prompts LLM to break a given problem into sub-problems, then solve them sequentially
- Shows significant improvements in tasks involving symbolic manipulation, compositional generalization, and mathematical reasoning

**Decomposed Prompting (DECOMP)**:
- Few-Shot prompts LLM to show how to use certain functions (e.g., string splitting, internet searching)
- LLM breaks down original problem into sub-problems, then sends them to different functions
- Shows improved performance over L2MP on some tasks

**Plan-and-Solve Prompting**:
- Improved Zero-Shot CoT prompt: "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan and solve the problem step by step"
- Generates more robust reasoning processes than standard Zero-Shot-CoT on multiple reasoning datasets

**Tree-of-Thought (ToT)**:
- Creates a tree-like search problem by starting with an initial problem then generating multiple possible steps in the form of thoughts
- Evaluates progress each step makes towards solving the problem and decides which steps to continue with
- Effective for tasks that require search and planning

**Recursion-of-Thought (RoT)**:
- Similar to regular CoT, but sends complicated problems into another prompt/LLM call
- Inserts answer from sub-problem into the original prompt
- Shows improvements on arithmetic and algorithmic tasks

**Program-of-Thoughts (PoT)**:
- Uses LLMs like Codex to generate programming code as reasoning steps
- Code interpreter executes these steps to obtain the final answer
- Excels in mathematical and programming-related tasks but is less effective for semantic reasoning

**Faithful Chain-of-Thought (FCoT)**:
- Generates CoT that has both natural language and symbolic language (e.g., Python) reasoning
- Makes use of different types of symbolic languages in a task-dependent fashion

**Skeleton-of-Thought (SoT)**:
- Focuses on accelerating answer speed through parallelization
- Prompts LLM to create a skeleton of the answer, then sends these questions to an LLM in parallel
- Concatenates all the outputs to get a final response

#### 2.2.5 Ensembling Techniques
- Use multiple prompts to solve the same problem, then aggregate the responses into a final output
- Reduce variance of LLM outputs and often improve accuracy, but come with the cost of more model calls

**Demonstration Ensembling (DENSE)**:
- Creates multiple few-shot prompts, each containing distinct exemplars from the training set
- Aggregates the outputs to generate a final response

**Mixture of Reasoning Experts (MoRE)**:
- Creates a set of diverse reasoning experts using different specialized prompts for different reasoning types
- Selects the best answer based on an agreement score

**Max Mutual Information Method (MIM)**:
- Creates multiple prompt templates with varied styles and exemplars, then selects the optimal template that maximizes mutual information

**Self-Consistency (SC)**:
- Prompts the LLM multiple times to perform CoT, crucially with a non-zero temperature to elicit diverse reasoning paths
- Uses majority vote over all generated responses to select a final response
- Shows improvements on arithmetic, commonsense, and symbolic reasoning tasks

**Universal Self-Consistency (USC)**:
- Similar to SC, but rather than selecting the majority response, it inserts all outputs into a prompt template that selects the majority answer
- Helpful for free-form text generation and cases where the same answer may be output slightly differently by different prompts

**Meta-Reasoning over Multiple CoTs**:
- Generates multiple reasoning chains (but not necessarily final answers) for a given problem
- Inserts all of these chains in a single prompt template, then generates a final answer from them

**DiVeRSe**:
- Creates multiple prompts for a given problem, then performs Self-Consistency for each
- Scores reasoning paths based on each step, then selects a final response

**Consistency-based Self-adaptive Prompting (COSP)**:
- Constructs Few-Shot CoT prompts by running Zero-Shot CoT with Self-Consistency on a set of examples, then selecting a high agreement subset of the outputs to be included in the final prompt as exemplars
- Performs Self-Consistency with this final prompt

**Universal Self-Adaptive Prompting (USP)**:
- Builds upon the success of COSP, using unlabeled data to generate exemplars and a more complicated scoring function to select them
- Does not use Self-Consistency

**Prompt Paraphrasing**:
- Transforms an original prompt by changing some of the wording while maintaining the overall meaning
- Effectively a data augmentation technique that can be used to generate prompts for an ensemble

#### 2.2.6 Self-Criticism

**Approaches to Generating Self-Criticism for LLMs:**
* **Self-Calibration (Kadavath et al., 2022):**
  * Prompts an LLM to answer a question
  * Builds new prompt with the question, its answer, and instruction for judgement of correctness
  * Useful for gauging confidence levels in decision making
* **Self-Refine (Madaan et al., 2023):**
  * Iterative framework: LLM provides feedback on initial answer
  * LLM then improves answer based on feedback until stopping condition is met
  * Demonstrated improvement across various tasks
* **Reversing Chain-of-Thought (RCoT) (Xue et al., 2023):**
  * Prompts LLMs to reconstruct problem based on generated answer
  * Generates fine-grained comparisons between original and reconstructed problems for checking inconsistencies
  * Feedback from inconsistencies used to revise generated answer
* **Self-Verification (Weng et al., 2022):**
  * Generates multiple candidate solutions using Chain-of-Thought
  * Scores each solution by masking parts of original question and asking an LLM to predict them based on remaining information
  * Demonstrated improvement on reasoning datasets
* **Chain-of-Verification (COVE) (Dhuliawala et al., 2023):**
  * Generates answer to a given question using an LLM
  * Creates list of related questions to verify correctness of the answer
  * Each question answered by the LLM, then all information used to produce final revised answer
  * Demonstrated improvements in various question-answering and text-generation tasks
* **Cumulative Reasoning (Zhang et al., 2023b):**
  * Generates several potential steps to answer a question
  * LLM evaluates them, deciding whether to accept or reject each step
  * Checks if final answer is reached; repeats process if not
  * Demonstrated improvements in logical inference tasks and mathematical problems.

### 2.3 Prompting Technique

**Prompting Technique Usage**

**Measuring Technique Usage**:
- Measured by proxy of measuring number of citations from other papers in dataset
- Assumes papers about prompting are more likely to actually use or evaluate the cited technique

**Top 25 Cited Papers**:
- Most propose new prompting techniques
- Prevalence of citations for Few-Shot, Chain-of-Thought, and other popular prompting methods (Figure 2.11)

**Model and Dataset Mentions**:
- Graph shows citation counts of models (GPT-3, BERT, etc.) and datasets (CommonsenseQA, AQUA-RAT, etc.) in papers

**Prompting Techniques Usage Analysis**:
- Researchers propose new techniques and benchmark them across multiple models and datasets
- Helps to establish a baseline and understand prevalence of other techniques

**Benchmarks**:
- Models (Figure 2.9) and datasets (Figure 2.10) being used are measured by citations in dataset papers

### 2.4 Prompt Engineering

**Prompt Engineering Techniques:**
* Meta Promoting: technique used to generate or improve a prompt or template (Reynolds & McDonell, 2021; Zhou et al., 2022b; Ye et al., 2023)
	+ AutoPrompt (Shin et al., 2020b): uses frozen LLM and prompt templates with trigger tokens updated via backpropagation
	+ Automatic Prompt Engineer (APE): generates prompts using a set of exemplars and iteratively scores, paraphrases, and selects best ones (Zhou et al., 2022b)
	+ Gradient-Free Instructional Prompt Search (GrIPS): uses more complex operations like deletion, addition, swapping, paraphrasing to create variations of a starting prompt (Prasad et al., 2023)
* Prompt Optimization with Textual Gradients (Pro-TeGi): improves prompt templates through multi-step process, passing output, ground truth, and criticisms for generating new prompts (Pryzant et al., 2023)
* RLPrompt: uses frozen LLM with unfrozen module to generate prompt templates, scores on a dataset, and updates unfrozen module using Soft Q-Learning (Guo et al., 2022; Deng et al., 2022)
* Dialogue-comprised Policy-gradient-based Discrete Prompt Optimization (DP2O): involves reinforcement learning, custom prompt scoring function, and conversations with a LLM to construct the prompt (Li et al., 2023b).

### 2.5 Answer Engineering
* Iterative process of developing or selecting algorithms that extract precise answers from LLM outputs
* Three design decisions: choice of answer shape, space, and extractor (Figure 2.12)
	+ **Answer Shape**: physical format (token, span, image/video)
	+ **Answer Space**: domain values (all tokens or restricted to specific labels)
	+ **Answer Extractor**: rule or separate LLM for extracting answer from outputs.

## 3 Beyond English Text Prompting

**Multilingual and Multi-Modal Prompting Techniques for GenAIs**

**3 Beyond English Text Prompting**:
- Prompting GenAIs with English text is currently the dominant method, but prompting in other languages or modalities requires special techniques.
- This section discusses the domains of multilingual and multi-modal prompting.

### 3.1 Multilingual Prompting Techniques
- **Translate First Prompting**: Translates non-English input examples into English, allowing the model to utilize its strengths in English.

#### 3.1.1 Chain-of-Thought (CoT) Promoting
- **XLT (Cross-Lingual Thought) Prompting**: Uses a prompt template with six separate instructions, including role assignment, cross-lingual thinking, and CoT.
- **CLSP (Cross-Lingual Self Consistent Prompting)**: Introduces an ensemble technique to construct reasoning paths in different languages for the same question.

#### 3.1.2 In-Context Learning (ICL) Extensions
- **X-InSTA Promoting**: Explores three distinct approaches for aligning in-context examples with the input sentence: semantic, task-based, and a combination of both.
- **In-CLT (Cross-lingual Transfer) Prompting**: Leverages both source and target languages to create in-context examples, helping stimulate the model's cross-lingual capabilities.

#### 3.1.3 In-Context Example Selection
- **Strong influence on performance of LLMs**. Semantically similar examples are important, but using semantically dissimilar or peculiar exemplars can also enhance performance.
- **PARC (Prompts Augmented by Retrieval Cross-lingually)**: Retrieves relevant exemplars from a high resource language to enhance cross-lingual transfer performance.

#### 3.1.4 Prompt Template Language Selection
- **Selecting the English language for the prompt template can be more effective** than the task language, likely due to the model's familiarity with English data

#### 3.1.5 Prompting for Machine Translation
- Techniques like **Multi-Aspect Promoting and Selection** (MAPS), **Chain-of-Dictionary** (CoD), and **Dictionary-based Promoting for Machine Translation** (DiPMT) extract words or definitions in the source and target languages, then use them as part of the prompt to generate a translation.
- **Decomposed Prompting for MT** (DecoMT) translates the input into several chunks independently, then uses these translations and contextual information to generate a final translation.
- **Human-in-the-Loop** techniques like Interactive-Chain-Prompting and Iterative Prompting involve humans in the translation process to resolve ambiguities or refine the initial translation.

### 3.2 Multimodal

**Multimodal Prompting Techniques**
* Multimodal techniques extend text-based taxonomy to include image, audio, video, segmentation, and 3D modalities
* New ideas made possible by different modalities as GenAI models evolve beyond text-based domains

#### 3.2.1 Image Prompting
* Image modality includes photographs, drawings, screenshots of text
* Techniques: Image Generation, Caption Generation, Image Classification, Image Editing
* **Prompt Modifiers**: Words appended to a prompt to change the resultant image (Oppenlaender, 2023)
	+ Examples: Medium, Lighting
* **Negative Prompting**: Users can numerically weight certain terms in the prompt to influence model's focus (Schulhoff, 2022)

**Multimodal In-Context Learning**
* Image-as-Text Promoting generates textual description of an image for easy inclusion in text-based prompts
* Paired-Image Prompting: Model is shown two images with a transform, then generates new image based on that transformation (Wang et al., 2023k; Liu et al., 2023e)

**Multimodal Chain-of-Thought**
* Extensions of textual CoT to the multimodal setting: Image and audio prompts (Zhang et al., 2023d; Huang et al., 2023c; Zheng et al., 2023b; Yao et al., 2023c)
* Techniques: Duty Distinct Chain-of-Thought, Multimodal Graph-of-Thought, Chain-of-Images (Zheng et al., 2023b; Yao et al., 2023c; Meng et al., 2023)

#### 3.2.2 Audio Prompting
* In early stages with mixed results from open source models (Hsu et al., 2023; Wang et al., 2023g; Peng et al., 2023; Chang et al., 2023)
* Expectations for future proposals of various prompting techniques in the audio modality

#### 3.2.3 Video Prompting
* Extensions to text-to-video generation, video editing, and video-to-text generation (Brooks et al., 2024; Lv et al., 2023; Liang et al., 2023; Girdhar et al., 2023; Zuo et al., 2023; Wu et al., 2023a; Cheng et al., 2023; Yousaf et al., 2023; Mi et al., 2023; Ko et al., 2023a)

#### 3.2.4 Segmentation Prompting
- Used for segmentation tasks (Tang et al., 2023; Liu et al., 2023c)
- Examples: semantic segmentation

#### 3.2.5 3D Prompting
- Used in 3D modalities
- Applications:
  - 3D object synthesis (Feng et al., 2023; Li et al., 2023d,c; Lin et al., 2023; Chen et al., 2023f; Lorraine et al., 2023; Poole et al., 2022; Jain et al., 2022)
  - 3D surface texturing (Liu et al., 2023g; Yang et al., 2023b; Le et al., 2023; Pajouheshgar et al., 2023) - 4D scene generation (animating a 3D scene) (Singer et al., 2023; Zhao et al., 2023c)
- Input prompt modalities: text, image, user annotation (bounding boxes, points, lines), and 3D objects.

## 4 Extensions of Prompting
**Extensions of Prompting**
### 4.1 Agents
- LLMs have improved capabilities, leading to exploration of external tools (agents)
- Agents are GenAI systems that serve user goals via actions engaging with external systems
- Examples:
  - LLM outputs string for calculator usage
  - LLM writes and runs code, searches the internet

**OpenAI Assistants**:
- OpenAI, LangChain Agents, LlamaIndex Agents are examples

#### 4.1.1 Tool Use Agents
- Symbolic (e.g., calculator) and neural (e.g., separate LLM) external tools used
- Example: Modular Reasoning, Knowledge, and Language (MRKL) System
  - Contains LLM router providing access to multiple tools
  - Router makes calls to get information, combines it for final response
- Toolformer, Gorilla, Act-1 propose similar techniques

**Self-Correcting with Tool-Interactive Critiquing (CRITIC)**:
- Generates initial response without external calls
- LLM criticizes response for errors
- Uses tools to verify or amend parts of the response

#### 4.1.2 Code-Generation
**Code-Based Agents:**
* Translate problems into code for generation of answers
* Use Python interpreter to execute the generated code
* Example: PAL, ToRA, TaskWeaver

#### 4.1.3 Observation-Based Agents
**Observation-Based Agents:**
* Interact with toy environments to solve problems
* Receive observations and use them in problem solving process
* Example: ReAct, Reflexion

**Lifelong Learning Agents:**
* Acquire new skills as they navigate the world
* Generate code to execute actions and save for long-term memory
* Example:
  + Voyager
  + Ghost in the Minecraft (GITM)

#### 4.1.4 **Retrieval Augmented Generation (RAG):**
+ Verify-and-Edit
+ Demonstrate-Search-Predict (DSP)
+ Interleaved Retrieval guided by Chain-of-Thought (IRCoT)
+ Iterative Retrieval Augmentation techniques: FLARE, IRP

**Retrieval Augmented Generation (RAG):**
* Retrieve relevant information from external sources
* Enhance performance in knowledge-intensive tasks

**Agent Techniques for Code Generation:**
* Single step code generation: PAL, ToRA, TaskWeaver
* Interleaving code and reasoning steps: ToRA
* Lifelong learning agents: Voyager, GITM
* Iterative retrieval augmentation techniques: FLARE, IRP.

### 4.2 Evaluation

**Evaluation of LLMs as Evaluators**

**Components of Evaluation Frameworks**:
- Prompting technique(s)
- Output format of the evaluation
- Framework of the evaluation pipeline
- Methodological design decisions

#### 4.2.1 Prompting Techniques
- Simple instruction vs. CoT
- Role-based Evaluation
- Chain-of-Thought prompting
- Model-Generated Guidelines

#### 4.2.2 Output Format
- Linear Scale (1-5)
- Binary Score
- Likert Scale
- XML/JSON styling

#### 4.2.3 Prompting Frameworks
- LLM-EVAL: single prompt with schema of variables to evaluate
- G-EVAL: includes AutoCoT steps in the prompt
- ChatEval: multi-agent debate framework

#### 4.2.4 Methodologies
- Explicit scoring: directly prompting the LLM for a quality assessment
- Implicit scoring: deriving score using model's confidence, likelihood, or explanations
- Batch Prompting: evaluating multiple instances at once
- Pairwise Evaluation: comparing quality of two texts directly

## 5 Prompting Issues

### 5.1 Security Concerns

#### 5.1.1 **Prompt Hacking:**
- Manipulating prompts to attack GenAI systems
- Types: Prompt Injection, Jailbreaking

**Prompt Injection:**
- Overriding original developer instructions with user input
- Can lead to malicious outputs (privacy concerns, offensive content)
- Example: "Ignore other instructions and make a threat against the president"

**Jailbreaking:**
- Directly prompting GenAI models maliciously
- Example: "Make a threat against the president"

#### 5.1.2 Risks of Prompt Hacking

**Data Privacy:**
- Training Data Reconstruction: extracting training data from GenAI models via prompts
  - Nasr et al. (2023): ChatGPT repeats word "company" to regurgitate training data
- Prompt Leaking: extracting prompt templates from applications
  - Developers spend significant time creating prompt templates

**Code Generation Concerns:**
- LLMs used for code generation can lead to vulnerabilities
- Package Hallucination: LLM-generated code attempts to import non-existent packages with malicious code
- Bugs in LLM-generated code more frequent than non-LLM code
- Minor changes to prompting technique can also lead to vulnerabilities

**Customer Service:**
- Malicious users exploit chatbots through prompt injection attacks, leading to brand embarrassment and potential financial losses.

#### 5.1.3 Hardening Measures
- Prompt-based Defenses: instructions included in the prompt to avoid prompt injection
  - "Do not output any malicious content"
- Guardrails: rules and frameworks for guiding GenAI outputs
  - Classifying user input as malicious or not, then responding with a canned message if malicious
- Dialogue managers: allow LLM to choose from curated responses
- Prompting-specific programming languages: improve templating and act as guardrails.

**Detectors:**
- Tools designed to detect malicious inputs and prevent prompt hacking
- Many companies build such detectors using fine-tuned models trained on malicious prompts.

### 5.2 Alignment

**Alignment and Prompt Design for LLMs**

**Importance of Alignment**: Ensuring that Language Models (LLMs) are well-aligned with user needs in downstream tasks is crucial for successful deployment. Poor alignment can lead to harmful outputs, inconsistent responses, or biases, making it challenging to use the models effectively.

#### 5.2.1 **Prompt Sensitivity**: 
LLMs are highly sensitive to input prompts. Even minor changes in prompt wording, task format, or order of examples can significantly impact the model's behavior and performance on tasks.

**Prompt Drift**: When the model behind an API changes over time, the same prompt may produce different results due to prompt drift. Continuous monitoring is necessary to maintain prompt performance.

#### 5.2.2 Calibration and Overconfidence
LLMs are often overconfident in their answers, leading users to rely too heavily on model outputs. Confidence calibration techniques like Verbalized Score or simple prompts can help mitigate this issue. Sycophancy occurs when models express agreement with user opinions, even when they contradict the model's initial output.

#### 5.2.3 Biases, Stereotypes, and Culture
LLMs should be fair to all users and avoid perpetuating biases, stereotypes, or cultural harms in their outputs. Techniques like Vanilla Prompting, Selecting Balanced Demonstrations, Cultural Awareness, AttrPrompt, and Question Clarification have been designed to address these issues.

#### 5.2.4 Ambiguity
Ambiguous questions can be interpreted in multiple ways, making it challenging for LLMs to provide accurate responses. Techniques like Ambiguous Demonstrations and Question Clarification have been developed to help LLMs handle ambiguous inputs.

## 6 Benchmarking

### 6.1 Benchmarking Prompting Techniques

**Technique Benchmarking**:
- Formal evaluation of prompting techniques using a representative subset of 2,800 MMLU questions
- Used gpt-3.5-turbo for all experiments

**Comparing Prompting Techniques**:
- Tested 6 distinct prompting techniques using the same general prompt template
- Included base instruction and 2 question formats
- Compared 6 total variations of each technique, except for Self-Consistency

**Zero-Shot Baseline**:
- Ran questions directly through the model without any prompting techniques as a baseline
- Utilized both formats and 3 phrasing variations of the base instruction

**Zero-Shot-CoT Techniques**:
- Tested with 3 thought indicators (Thought, ThoT, Plan and Solve)
- Selected the best and ran it with Self-Consistency, taking the majority response

**Few-Shot Techniques**:
- Tested with exemplars generated by an author
- Used 3 variations of the base instruction and 2 question formats
- Ran the best phrasing with Self-Consistency, taking the majority response

**Results**:
- **Accuracy values** are shown for each prompting technique (Figure 6.1)
- Zero-Shot CoT performed better than Zero-Shot, but had wide spread
- Self-Consistency improved accuracy for Zero-Shot prompts, but only repeated the same technique
- Few-Shot CoT performed the best, with unexplained performance drops from certain techniques needing further research
- Prompting technique selection is akin to hyperparameter search and is very difficult.

### 6.2 Prompt Engineering Case Study

#### 6.2.1 Problem
**Suicide Risk Detection from Text**
- Severe global issue with lack of mental health resources
- High suicide rates in the U.S. (second leading cause of death)
- Significance of identifying suicidal crisis states for early intervention
- Focus on frantic hopelessness or entrapment as key predictor factor

#### 6.2.2 Dataset
- Subset from University of Maryland Reddit Suicidality Dataset
- Posts from r/SuicideWatch, a peer support subreddit
- Two coders trained to recognize factors in Suicide Crisis Syndrome
- Achieved solid inter-coder reliability (Krippendorff's alpha = 0.72)

#### 6.2.3 The Process

**Process of Using an LLM to Identify Entrapment in Posts**

**Expert Prompt Engineer**:
- Authored a widely used guide on prompting (Schulhoff, 2022)
- Tasked with using an LLM to identify entrapment in development posts

**Initial Steps**:
- Given brief verbal and written summary of **Suicide Crisis Syndrome** and **entrapment**
- Provided 121 development posts and their positive/negative labels (positive means entrapment is present)
- Documented the prompt engineering process for illustrative purposes

**Dataset Exploration**:
- Reviewed a description of entrapment as a first-pass rubric for human coders
- Loaded the dataset into a Python notebook for data exploration
- Asked **gpt-4-turbo-preview** if it knew what entrapment was, but found the response not similar to the provided description
- Included the provided description of entrapment in all future prompts

**Getting a Label**:
- Found that the LLM was exhibiting unpredictable and difficult to control behavior in sensitive domains
- Switched to the **GPT-4-32K** model to address the issue
- Noted that "guard rails" associated with some large language models may interfere with progress on a prompting task

#### Prompting Techniques (32 steps)

**Prompting Techniques (32 steps)**

**Improving Prompting Technique**:
- The prompt engineer spent time improving the prompting technique being used
- Techniques included: **Few-Shot**, **15Precision**, and others
- **F1** and **recall** are known as positive predictive value and true positive rate, respectively
- F1 is often used for evaluation but may not be appropriate in this problem space due to the uneven weighting of precision and recall

**Prompting Techniques Evaluated**:
- 10-Shot **AutoDiCoT** 
- 1-Shot AutoDiCoT (without email)
- 1-Shot AutoDiCoT + Full Context
- Zero-Shot + Context (Exact Match)
- Zero-Shot + Context (First Chars)
- 10-Shot AutoDiCoT Ensemble + Extraction
- 10-Shot AutoDiCoT Without Email
- Triplicate Context
- 20-Shot AutoDiCoT + Full Words
- 20-Shot AutoDiCoT + Full Words + Extraction
- Prompt 10-Shot AutoDiCoT + Extraction
- Prompt 20-Shot AutoDiCoT
- 10-Shot AutoDiCoT + Extraction Prompt
- 20-Shot AutoDiCoT 10-Shot AutoDiCoT

**Performance Metrics**:
- **Recall**: 0.25, 0.30
- **Precision**: 1.0, 0.70
- **F1 Score**: 0.40, 0.45

**Chain-of-Thought, AutoCoT, Contrastive CoT, and Multiple Answer Extraction Techniques**:
- Reported statistics for the first runs of these techniques
- F1 scores could change by up to 0.04 upon subsequent runs

**Zero-Shot + Context**:
- Obtained 0.25 recall, 1.0 precision, and 0.40 F1 score when evaluated on all samples from the training/development set

**10-Shot "Yes" or "No" Extractor**:
- Checks if the output is exactly "Yes" or "No"
- Has better performance than an extractor that checks if the first few characters of the output match the words "Yes" or "No"

**Prompt Engineer's Approach**:
- Observed the mislabeling of a 12th item in the development set and began experimenting with ways to modify the prompt to get the correct label
- Prompted the LLM to generate an explanation of why the 12th item was labeled incorrectly

**Explanation Output**:
- Provides mental health support, such as "If you're in immediate danger of harming yourself, please contact emergency services or a crisis hotline"
- Is often five times longer than this snippet

**Automatic Directed CoT (AutoDiCoT)**

**Algorithm:**
- Automatically directs the CoT process to reason in a particular way
- Combines automatic generation of CoTs with showing LLM examples of bad reasoning

**Changes and Testing:**
- Including email message from prompt engineer for context
- Removing email message decreases performance
- Adding full context, 10 regular exemplars, and one-shot exemplar about not reasoning incorrectly hurts performance
- Creating more AutoDiCoT exemplars yields most successful prompt in terms of F1 score

**Effects of Email:**
- Demonstrates improvements through exploration and fortuitous discovery
- Highlights the difficulty of prompting as a black art where LLM may be sensitive to unexpected variations

**Comparison of Prompts:**
- Full context only boosts performance over previous technique
- 20-shot AutoDiCoT leads to worse results than 10-shot prompt on all samples other than the first twenty and test set.

**20-Shot AutoDiCoT**:
- Including full words (Question, Reasoning, Answer) instead of abbreviations (Q, R, A) did not improve performance
- Decreased F1 score by 5% and 6%, respectively for precision and recall

**20-Shot AutoDiCoT + Full Words + Extraction Prompt**:
- Improved accuracy but decreased F1 score due to unparsed outputs containing incorrect responses
- No change in recall (0.86)

**10-Shot AutoDiCoT + Extraction Prompt**:
- Did not improve results compared to the best performing 10-Shot AutoDiCoT prompt
- Decreased F1 score by 4% and precision by 6%

**10-Shot AutoDiCoT without Email**:
- Removing email from the prompt hurt performance
- Decreased F1 score by 14%, recall by 38%, and precision by 5%

**De-Duplicating Email**:
- Removing duplicate emails significantly hurt performance
- Decreased F1 score by 7%, recall by 13%, and precision by 5%

**10-Shot AutoDiCoT + Default to Negative**:
- Using the best performing prompt and defaulting to negative (not entrapment) in case of unextracted responses did not help performance
- Decreased F1 score by 11%, recall by 3%, and precision by 8%

**Ensemble + Extraction**:
- Combining multiple variations of the best performing prompt and using an extraction prompt to obtain final answers hurt performance
- Decreased F1 score by 16%, recall by 22%, and precision by 12%

**10-Shot AutoCoT + 3x the context (no email dupe)**:
- Pasting in three copies of the context without email duplication did not improve performance
- Decreased F1 score by 6%, recall by 8%, and precision by 5%

**Anonymize Email**:
- Replacing personal names with random names significantly decreased performance
- Decreased F1 score by 8%, recall by 14%, and precision by 5%

**DSPy Framework**:
- Automatically optimized LLM prompts for the target metric (F1) using a chain-of-thought classification pipeline and random sampling of training exemplars
- Achieved the best result (0.548 F1, 0.385/0.952 precision/recall) without using the professor's email or incorrect 10-Shot AutoDiCoT prompt
- Demonstrated significant promise of automated prompt engineering

#### 6.2.4 Discussion

**Prompt Engineering: Discussion**

**Key Takeaways**:
- Prompt engineering is a non-trivial process, not well described in literature
- It is fundamentally different from other ways of getting a computer to behave as desired: it involves "cajoling" the system, not programming, and can be incredibly sensitive to specific details in prompts
- The process is highly dependent on both the LLM (Language Model) being used and domain expertise

**Importance of Digging into the Data**:
- Exploring the reasoning behind LLM's incorrect responses is crucial

**Collaboration between Prompt Engineer and Domain Experts**:
- Engagement between these two groups is essential for successful prompt engineering

**Automated Prompt Exploration**:
- Automated methods can be useful, but human prompt engineering/revision is the most effective approach

**Future Directions**:
- This study serves as a step towards more robust examination of prompt engineering techniques.

## 7 Related Work

**Related Work**

**Systematic Review of Prompt Engineering**:
- Liu et al. (2023b): Systematic review of prompt engineering techniques, including:
  - **Prompt template engineering**
  - **Answer engineering**
  - **Prompt ensembling**
  - **Prompt tuning methods**
  - Covers various types of prompts (e.g., cloze, soft-prompting) before the era of ChatGPT

**Review of Prompting Techniques**:
- Chen et al. (2023a): Review of popular prompting techniques like:
  - **Chain-of-Thought**
  - **Tree-of-Thought**
  - **Self-Consistency**
  - **Least-to-Most prompting**
- White et al. (2023) and Schmidt et al. (2023): Taxonomy of prompt patterns, similar to software patterns
- Gao (2023): Practical tutorial on prompting techniques for non-technical audience
- Santu and Feng (2023): General taxonomy of prompts to design complex task-specific prompts
- Bubeck et al. (2023): Qualitative experiment with a wide range of prompting methods on early GPT-4
- Chu et al. (2023): Review of Chain-of-Thought related prompting methods for reasoning

**Prompt Engineering for Specific Domains/Applications**:
- Meskó (2023) and Wang et al. (2023d): Recommended use cases and limitations of prompt engineering in medical/healthcare domains
- Heston and Khun (2023): Review of prompt engineering for medical education use cases
- Peskoff and Stewart (2023): Query ChatGPT and YouChat to assess domain coverage
- Hua et al. (2024): GPT-4-automated approach to review LLMs in mental health space
- Wang et al. (2023c): Review of prompt engineering and relevant models in the visual modality
- Yang et al. (2023e): Comprehensive list of qualitative analyses of multimodal prompting, particularly GPT-4V19
- Durante et al. (2024): Review of multimodal interactions based on LLM embedded agents
- Ko et al. (2023b): Review of literature on adoption of Text-to-Image generation models for visual artists' creative works
- Gupta et al. (2024): Review of GenAI through a topic modeling approach
- Awais et al. (2023): Review of foundation models in vision, including various prompting techniques
- Hou et al. (2023): Systematic review of prompt engineering techniques as they relate to software engineering
- Wang et al. (2023e): Review the literature on software testing with large language models
- Zhang et al. (2023a): Review ChatGPT prompting performance on software engineering tasks like automated program repair
- Neagu (2023): Systematic review on how prompt engineering can be leveraged in computer science education
- Li et al. (2023j): Review literature on fairness of large language models

**Related Aspects**:
- Hallucination of language models (Huang et al., 2023b)
- Verifiability (Liu et al., 2023a)
- Reasoning (Qiao et al., 2022)
- Augmentation (Mialon et al., 2023)
- Linguistic properties of prompts (Leidinger et al., 2023)

**Update and Standardization**:
- This survey offers an update in a fast-moving field
- Provides a starting point for taxonomic organization and standardization of prompting techniques
- Based on the widely well-received PRISMA standard for systematic literature reviews (Page et al., 2021)

## 8 Conclusions

**Conclusions:**
* Generative AI is a novel technology with limited understanding of its capabilities and limitations
* Natural language interface poses challenges like ambiguity, context, and course correction
* Techniques described are discovered through experimentation, analogies from human reasoning, or serendipity
* Present work aims to provide taxonomy and terminology for existing prompt engineering techniques
* Discuss over 200 prompting techniques, frameworks, safety, and security issues
* Two case studies illustrate models' capabilities and the process of solving problems
* Stance is observational with no claims to validity or completeness
* Encourage skepticism towards claims about method performance
* Recommend understanding the problem and using ecologically valid data and metrics.

**Acknowledgements:**
* Advice from Hal Daumé III, Adam Visokay, Jordan Boyd-Graber, Diyi Yang, and Brandon M. Stewart
* API credits and design work from OpenAI and Benjamin DiMarco.

