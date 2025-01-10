# A Comprehensive Survey of AI-Driven Advancements and Techniques in Automated Program Repair and Code Generation
source: https://arxiv.org/html/2411.07586v1
by Avinash Anand, Nishchay Yadav, Akshit Gupta, Shaurya Bajaj 

## Contents
- [Abstract.](#abstract)
- [1. Introduction](#1-introduction)
- [2. Survey Methodology](#2-survey-methodology)
  - [2.1. Research Questions](#21-research-questions)
  - [2.2. Methods Employed](#22-methods-employed)
- [3. Research Questions](#3-research-questions)
  - [3.1. How have AI techniques, especially large language models (LLMs), improved software debugging and bug fixing? What are some recent trends and common challenges in using AI for these tasks?](#31-how-have-ai-techniques-especially-large-language-models-llms-improved-software-debugging-and-bug-fixing-what-are-some-recent-trends-and-common-challenges-in-using-ai-for-these-tasks)
  - [3.2. What recent trends have emerged for using AI in APR tools?](#32-what-recent-trends-have-emerged-for-using-ai-in-apr-tools)
  - [3.3. What challenges are being faced in APR right now?](#33-what-challenges-are-being-faced-in-apr-right-now)
- [4. How do modern debugging tools and benchmarks help evaluate the effectiveness of bug-fixing methods? What are the common gaps or limitations in these tools?](#4-how-do-modern-debugging-tools-and-benchmarks-help-evaluate-the-effectiveness-of-bug-fixing-methods-what-are-the-common-gaps-or-limitations-in-these-tools)
  - [4.1. Modern Debugging tools](#41-modern-debugging-tools)
  - [4.2. Modern Benchmark tools](#42-modern-benchmark-tools)
  - [4.3. Common Gaps and Limitations in Modern Debugging Tools and Benchmarks](#43-common-gaps-and-limitations-in-modern-debugging-tools-and-benchmarks)
- [5. How do different AI models and tools compare in handling various code-related tasks? What are the main strengths and weaknesses of these models?](#5-how-do-different-ai-models-and-tools-compare-in-handling-various-code-related-tasks-what-are-the-main-strengths-and-weaknesses-of-these-models)
- [6. How do AI models trained on general language datasets, specialized code datasets, or self-supervised learning approaches differ in their ability to handle code-related tasks?](#6-how-do-ai-models-trained-on-general-language-datasets-specialized-code-datasets-or-self-supervised-learning-approaches-differ-in-their-ability-to-handle-code-related-tasks)
- [7. Conclusion](#7-conclusion)

## Abstract.

**Automated Program Repair (APR) and Large Language Models (LLMs)**

**Background:**
- Bug fixing and code generation research topics transformed by LLMs
- Recent explosion of Large Language Models in these areas

**Group 1: APR and LLM Integration**
* New methods for bug detection and repair
	+ Locating semantic errors, security vulnerabilities, runtime failure bugs
	+ Emphasizes role of LLMs in reducing manual debugging efforts
	+ Innovations boosting accuracy and efficiency in automatic debugging
* Context-aware fixes using LLMs:
	+ Toward functional correctness and security
	+ APR research focuses on enabling iterative code improvement
	+ Open-source models utilized for feedback loops

**Group 2: Code Generation using LLMs**
* Overview of general-purpose LLMs fine-tuned for programming
* Task-specific models presented
* Improving code generation methods:
	+ Identifier-aware training
	+ Fine-tuning at instruction level
	+ Incorporating semantic code structures

**Methodologies in APR and Code Generation:**
- Using LLMs as a basis for research
- Discussing challenges of achieving functional correctness and security
- Outlining future directions for research in LLM-based software development

**Publication Information:**
- Copyright: ACM
- Year: 2024
- DOI: XXXXXXX.XXXXXXX
- CCS categories: Computing methodologies, Artificial intelligence; Software and its engineering, Software creation and management.

## 1. Introduction

**Automated Programming with Large Language Models (LLMs)**

**Background:**
- LLMs have gained popularity in automated software engineering for bug fixing and code generation
- Increased usage over past decade [^9] [^11]
- Improved quality and speed of automating programming tasks
  - Summarizing code
  - Generating code based on natural language requests
  - Fixing bugs in pre-existing code
  - Understanding complex repositories

**Tools and Research:**
- Many tools developed for APR and NLP-based code generation [^16] [^22] [^7]
  - Implement Abstract Syntax Trees (ASTs)
  - Use heuristics for ranking plausible patches, patterns, and context-matching
- LLMs have natural advantage due to large datasets and billions of parameters
- Performance significantly better than training models from scratch [^19] [^20] [^26]

**Complexities and Challenges:**
- Repair scenarios: syntactic errors, semantic errors [^27]
- Repair techniques: recompilation, binary rewriting [^28]
- Testing of repairs: patch generation, input testing, coevolution [^29]
- Benchmarking and evaluation of performance [^30]
- Difficulties in building language-agnostic APR tools [^31]

**Survey Paper Goals:**
1. Collect research on LLMs for code generation and summarize achievements
2. Elucidate repair scenarios and programming languages used
3. Describe integration of LLMs into workflow for repairing and generating code
4. Identify limitations and ongoing challenges in the field.

## 2. Survey Methodology

We implemented measures to conduct the survey by searching and collecting models, research papers, and journals relevant to our purpose through various methods. 

[Figure 1. Papers Included in Every Domain](https://arxiv.org/html/2411.07586v1/extracted/5992088/figures/PapersIncludedinEveryDomain.png) 

### 2.1. Research Questions

**Research Questions**
1. **How have AI techniques, especially large language models (LLMs), improved software debugging and bug fixing?**:
   - What are recent trends in using AI for these tasks?
   - What are common challenges faced in this area?
2. **How do modern debugging tools and benchmarks help evaluate the effectiveness of bug-fixing methods?**:**
   - What are common gaps or limitations in these tools?
3. **What are the key differences between popular code generation models?**:
   - How do these models perform on tasks like code completion and code summarization?
4. **How do different AI models and tools compare in handling various code-related tasks?**:**
   - What are the main strengths and weaknesses of these models?

### 2.2. Methods Employed

**Methods Used for Selecting and Bifurcating Resources:**
1. **Systematic Literature Review**:
   - Review available literature on topic of interest and related work
   - Clear inclusion/exclusion criteria
   - Search relevant papers in databases, apply criteria, analyze and categorize under Code Generation and Automated Program Repair
2. **Taxonomy Development**:
   - Classification of AI techniques, tools, and methods for debugging, bug fixing, and code generation
   - Comparative analysis based on objectives, techniques, and outcomes
   - Identify overlaps and distinctions between categories
3. **Comparative Analysis**:
   - Compare selected papers using performance, accuracy, etc. criteria
   - Tabular and graphical description for better visualization of comparison
   - Between model and tools comparison
4. **Trend Analysis and Gap Identification**:
   - Identify open challenges and research gaps in AI-driven techniques/methods for APR and code generation
   - Figure out common themes, recurring challenges, under-explored areas across all categories
   - See future trends and potentials in similar field
5. **Survey of Benchmarks and Evaluation Metrics**:
   - Study on benchmarks and evaluation metrics for evaluating models and tools
   - Identify similarities, differences, describe each benchmark, analyze strengths/weaknesses, suggest improvements or new metrics.

## 3. Research Questions

### 3.1. How have AI techniques, especially large language models (LLMs), improved software debugging and bug fixing? What are some recent trends and common challenges in using AI for these tasks?

**Automated Program Repair (APR) for Fixing Security Bugs:**
- **Template Based Patching**: APR tools use existing templates to fix security vulnerabilities like SQL injection [^19] [^20].
- **Dynamic Analysis for Security Bugs**: Techniques like fuzzing and symbolic execution help discover unnoticed bugs, which are then patched through testing [^13] [^18] [^26].
- **Search-Based APR for Security**: Tools output likely patches that are tested to find the optimal one meeting given conditions [^18] [^26].
- **Specification-Guided APR for Security Protocols**: APR tools obtain precise grammar and specifications for protocols, then patch checks are conducted on inputs [^20] [^26].
- **Input Sanitization and Validation Patches**: APR detects missing input validation, adding required functions to prevent harmful inputs [^26].
- **Repairing Memory Safety Bugs**: APR identifies unsafe memory accesses and applies corrective measures like bounds checking or safer allocation techniques [^13] [^20].

**Automated Program Repair (APR) for Fixing Semantic Bugs:**
- **Pattern-Based Patching for Semantic Bugs**: APR systems address logical errors using pattern-based repairs [^27] [^20] [^13].
- **Fuzzing Techniques for Semantic Bugs**: Greybox fuzzing combines black and white box testing to reveal difficult-to-detect bugs [^24] [^26] [^13].
- **Search-Based APR for Semantic Bugs**: Evolutionary algorithms search repair options covered by test suites or specifications [^20] [^26].
- **Specification-Guided Repair**: Formal models of expected behavior help identify deviations and produce valid patches [^8] [^26].

**Automated Program Repair (APR) for Fixing Syntactic Bugs:**
- **Pattern-Based Patching for Syntactic Bugs**: APR systems consider syntax violations and use correction patterns [^6].
- **Grammar-Based Fuzzing for Syntax Validation**: Tools produce valid inputs based on Abstract Syntax Trees (ASTs) to find bugs related to incomplete or incorrectly structured code [^8].
- **Search-Based Techniques for Syntactic Repairs**: Methods look through many code modifications and test them to obtain a valid syntactic fix.

### 3.2. What recent trends have emerged for using AI in APR tools?

**Recent Trends in Automated Program Repair:**
- **Pre-trained Models**: Pre-training on large programming datasets (Codex, CodeT5) is gaining traction [^13].
- **Transfer Learning**: LLMs are becoming more adaptable for transfer learning, fine-tuning pre-trained models for specific bug-fixing tasks.
- **Self-Supervised Learning**: Allows training on unlabeled datasets, taking advantage of the fact that there are a lot of code repositories without annotations [^18] [^26].
- **Explainable AI (XAI)**: Making AI-driven bug fixing more interpretable and transparent for developers [^26].
- **Interactive Debugging Systems**: Integrating active learning where AI requests human help to resolve unclear cases.
- **Multi-modal Models**: Incorporating code, comments, documentation, and logs enhances the AI's ability to understand context [^18] [^26].

**Advancements in Automated Program Repair:**
1. **Pre-trained models**: Using fine-tuned pre-trained models like GPT-4 on programming datasets [^13].
2. **Neural Networks for Fault Localization**: Deep analysis of code properties (temporal, control flow) to identify bugs using Abstract Syntax Trees (ASTs) [^19] [^20] [^26].
3. **Automated Test Generation**: Creating and running test cases to check the correctness of potential solutions [^18] [^26].
4. **Test Coverage Improvement**: Dissecting code modifications to guarantee new tests encompass all relevant elements, reducing overfitting [^18] [^26].

**Workflow of Automated Program Repair Tools:** (Refer to Figure 3 in the provided text)

### 3.3. What challenges are being faced in APR right now?

**Challenges with Automated Program Repair (APR)**
- **Accuracy and Reliability**:
  * Incorrectly identifying faulty code as fine
  * Human verification required before implementation
- **Context Sensitivity**:
  * Difficulties understanding large codebases with dependencies
  * Reasons for incorrect repairs that are technically correct but not for the entire codebase
- **Resource Overhead**:
  * High demands for memory and computing power
  * Disruption of user's workflow and productivity decrease

**Limitations of Language Learning Models (LLMs) in APR:**
- **Generalization**:
  * Difficulty with new, unseen bugs or domain-specific code
  * Challenges in systems with unconventional architectures or libraries
- **Scalability**:
  * Debugging large, complex systems is difficult for LLMs due to potential interactions and dependencies
- **Limited Understanding of Context**:
  * May make mistakes in understanding full business logic or domain-specific intricacies behind bugs
  - Resulting in incomplete or incorrect fixes
- **Security Concerns**:
  * AI-generated fixes may introduce security vulnerabilities or fail to address existing ones
- **Bias in Training Data**:
  * LLMs may inherit biases from the data they were trained on, leading to over-reliance on common patterns and overlooking edge cases
- **Overfitting to Benchmarks**:
  * Models trained mostly on benchmarks might not perform well in real-life scenarios
- **Ethical Concerns**:
  * Copyright issues when LLMs are trained on proprietary code
  * Concerns about reproducing code without proper credit.

**Targeted Programming Languages in Surveys:**
(Refer to Table 1 for details)

## 4. How do modern debugging tools and benchmarks help evaluate the effectiveness of bug-fixing methods? What are the common gaps or limitations in these tools?

Benchmarking is crucial for identifying repair scenarios APR can handle effectively and improving debugging tools. Papers surveyed have rigorously benchmarked against existing state-of-the-art debugging tools, helping researchers explore new ways to improve their solutions.

### 4.1. Modern Debugging tools

**Software Engineering Debugging Tools**

**Debugging Assistance**:
- Automated defect detection, location, and repair on code

**Fuzzing-Based Approaches**:
- **FuzzRepair** and **AFLNet**: Integrate code fuzzing with program repair methods
- Generate diverse, reachable inputs to find bugs in hard-to-reach areas
- Check fixes across various scenarios for thoroughness

**Machine Learning Approaches**:
- **CoCoNuT**, **SequenceR**, and **Tufano19**: Develop deep learning patches based on bug-fixing patterns
- Evaluate generalization of bug-fixing methods across codebases

**Semantic Code Analysis Tools**:
- **CodeBERT**, **GraphCodeBERT**, and **CugLM** : Embed semantic code representations for potential fault targeting
- Utilize natural language processing and code understanding models

**Structural Relationship Learning Tools**:
- **TreeBERT** and **T5** : Learn and understand structural relationships within code
- Ensure bug fixes are syntactically correct and logically consistent

**Evolutionary/Template-Based Repair Tools**:
- **ARJA-e**, **REWARDREPAIR**, and **EVOREPAIR** : Use genetic algorithms, reward-based learning for patch generation
- Measure efficiency and scalability of repair techniques

**Template-Based Fix Generation Tools**:
- **TBAR** and **Darjeeling** : Generate fixes based on predefined templates
- Analyze effectiveness of rule-based or human-guided patch generation

**Probabilistic Modeling Tools**:
- **Prophet** : Predicts success probability of bug fix before production application
- Provides method to evaluate bug fix success rate

**Fit in Program Context Checking Tools**:
- **Fix2Fit** and **CPR** : Ensure fixes are right and fit well in program context
- Decrease side effects leading to new bugs

**Comprehensive Bug-Fixing Assessment Tools**:
- Provide methods like fuzzing, template-based repair, machine learning, and semantic analysis for thorough bug-fixing assessment.

### 4.2. Modern Benchmark tools

**Bug-Fixing Techniques Evaluation using Modern Benchmarks**

**Importance of Modern Benchmarks**:
- Measuring efficiency of bug-fixing techniques
- Controlled settings for thorough testing and verification

**Benchmark Categories**:
- **Code Generation Benchmarks**:
  - HumanEval: determines code generation abilities, fixes issues
  - MBPP: checks efficiency of bug-fixing methods in generating/correcting code
- **Fuzzing Benchmarks**:
  - ProFuzzBench: tests vulnerability detection and addressing of network protocols
  - SCTBench: detects concurrency problems, verifies bug-fixing techniques' performance
- **Debugging Benchmarks**:
  - DebugBench: assesses large language model debugging tasks
  - VulnLoc: focuses on automatic vulnerability localization
- **Defects4J**: provides real Java bugs data for testing bug-fixing methods
- **TransCoder**: enables code translation between programming languages for bug testing

**Benefits of Modern Benchmarks**:
- Conduct various tests covering software quality aspects
- Improve functionality and fault tolerance of software systems.

### 4.3. Common Gaps and Limitations in Modern Debugging Tools and Benchmarks

**Automated Program Repair (APR) and Code Generation: Limitations and Challenges**

**Generalization and Dataset Overfitting**:
- Limited number of benchmarks for APR systems, not reflecting real-world complexity
- Issues with randomness and incompleteness in fuzzing approach (e.g., QFuzz)
- Possible overfitting when tested on specific datasets like Defects4J

**Bias in Tool Selection and Benchmarks**:
- Convenient or popular tools and benchmarks, rather than diverse sets
- Neglect of real-world bugs and issues in focus (e.g., security vulnerabilities)
- Limited applicability across different programming languages

**Non-Determinism and Overfitting in Repair Tools**:
- Difficulty achieving consistent results due to non-deterministic features
- Overfitting, where tools generate "plausible" patches that may still be wrong
- Test case evaluation does not guarantee correctness

**Limited Handling of Security and Non-Test Bugs**:
- APR tools struggle with security vulnerabilities and non-test bugs
- Difficulty fixing memory safety bugs in languages like C and C++ (e.g., buffer overflows)
- Limited applicability in systems requiring robust security measures

**Challenges with Input Validation and Memory Bugs**:
- APR systems have trouble generating patches for complex input validation functions
- Fixing memory bugs like use-after-free or null pointer dereference remains difficult
- Ensuring correct memory allocation techniques is a challenge in languages like C and C++

**Limitations in Evaluation Metrics and Test Suites**:
- Limited test suites and evaluation metrics not accurately reflecting real-world bugs
- Accessibility issues and biases from using external platforms for testing (e.g., LeetCode)
- Simulated bugs may not be an exact reflection of actual production system bugs

**Difficulties with ML-Based Approaches**:
- Challenges generating large patches or rare function names using current ML methods
- Inaccurate predictions from ML models leading to compound correction issues.

## 5. How do different AI models and tools compare in handling various code-related tasks? What are the main strengths and weaknesses of these models?

**AI Models for Code-Related Tasks:**
* **Codex (OpenAI)**: Fast and fluent in completing codes but lacks accuracy in complex tasks
* **CodeT5 (Salesforce)**: Reliable code reading tool for smaller tasks with slower real-time performance
* **GraphCodeBERT (Microsoft)**: Processes complicated code structures but slower processing speed
* **Phind.com/CodeLlama** and **DeepSeek-Coder**: Standout in generating accurate code from prompts, handling unfinished snippets, and completing tasks across multiple languages
* **StarCoder2**: High mastery of 16 different languages but troubles with C++ and fill-in-the-middle tasks
* **Smaug** and **Zephyr**: Important role in code completion with Smaug leading on HuggingFace Open LLM Leaderboard and Zephyr being strong in bug detection and scalability
* **Magicoder** and **WizardCoder**: Weaknesses in debugging tasks, Magicoder leading in a certain benchmark over GPT-3.5 but not as good as CodeLlama-Python (34B) in large scale bug fixing tasks
* **GPT-4**: Superior natural language skills for writing summaries of long and complicated code but slower performance compared to smaller models.

**Code-Related Tasks:**
* **Code completion**: Codex, CodeT5, GraphCodeBERT, Phind/CodeLlama, DeepSeek-Coder, Smaug, Zephyr, Magicoder, WizardCoder, SPT Code, StarCoder2, GPT-4 excel in code completion tasks.
* **Bug fixing**: Codex, CodeT5, GraphCodeBERT, Phind, DeepSeek-Coder perform differently based on their ability to detect and fix bugs in various situations.
* **Code summarization**: CodeT5, GraphCodeBERT, Codex, GPT-4, SPT Code, StarCoder2 excel in code summarization tasks with different strengths and weaknesses.
* **Code translation and refactoring**: Codex, GPT-4 perform well in these areas but have varying speeds and accuracy levels.

## 6. How do AI models trained on general language datasets, specialized code datasets, or self-supervised learning approaches differ in their ability to handle code-related tasks?

**AI Models for Code-Related Tasks:**
* **Pre-trained on general language datasets**: Codex (powered by Github Copilot), CodeT5, GraphCodeBERT, Phind
	+ Codex: pre-trained on natural language data and publicly available code from GitHub
		- Enhances iterative problem solving capability but inconsistent in treating variables for complex tasks
	+ CodeT5: uses a unified encoder-decoder architecture to approach tasks using both programming and natural languages
		- Improves understanding of code semantics through developer-assigned identifiers, making it the top performer on various tasks from CodeXGLUE benchmark
	+ GraphCodeBERT: enters the world of data flow graphs for representation of code semantics
		- Capable of performing tasks like code search and clone detection due to knowledge of value transmission between variables
	+ Phind: fine-tuned on proprietary dataset of about 80,000 structured programming problems using advanced training techniques
		- Achieves a high pass rate on the HumanEval benchmark with rigorous decontamination process
* **Pre-trained to specialized code datasets**: OpenCodeInterpreter, StarCoder2, SPT Code, Magicoder
	+ OpenCodeInterpreter: picks up coding queries from specialized datasets and converts them into multi-turn dialogues for effective learning
	+ StarCoder2: exploits a language model fine-tuned on programming data to convert text into code snippets with great success
	+ SPT Code: encodes source code sequences to learn nice representations for code completion and translation tasks
	+ Magicoder: utilizes OS code snippets through OSS-INSTRUCT method for the creation of realistic code instructions
* **Models with self-supervised or bootstrapped methods**: DeepSeek-Coder, WizardCoder, Mixtral, Smaug
	+ DeepSeek-Coder: fine-tuned on a large corpus of open-source code to respond effectively to specific coding tasks and generate meaningful outputs
	+ WizardCoder: introduces the Evol-Instruct method for automatic generation of complex instructions, increasing performance on difficult tasks
	+ Mixtral: uses Sparse Mixture of Experts architecture and tuning of parameters to optimize performance by activating only some of its 42 billion parameters for inference
	+ Smaug: a complex model that may inherit biases from seed snippets, leading to low-quality outputs despite having the ability to perform well on mathematical problems.

## 7. Conclusion

This paper summarizes recent research on using Large Language Models (LLMs) and Artificial Intelligence (AI) in automated program repair and code generation. It highlights state-of-the-art developments, effective use cases, and current challenges. We also provide comparisons among open-source tools to help select the best models for future improvements.

