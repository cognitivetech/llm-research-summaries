# AFLOW : AUTOMATING AGENTIC WORKFLOW GENERATION

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, Chenglin Wu
https://arxiv.org/pdf/2410.10762

## Contents
- [Abstract](#abstract)
- [1 INTRODUCTION](#1-introduction)
- [2 RELATED WORK](#2-related-work)
- [3 PRELIMINARY](#3-preliminary)
  - [3.1 PROBLEM FORMULATION](#31-problem-formulation)
  - [3.2 AFLOW OVERVIEW](#32-aflow-overview)
- [4 THE DESIGN DETAILS OF AFLOW](#4-the-design-details-of-aflow)
- [5 EXPERIMENTS \> 5.1 EXPERIMENTAL SETUP](#5-experiments--51-experimental-setup)
  - [5.2 EXPERIMENTAL RESULTS AND ANALYSIS](#52-experimental-results-and-analysis)
- [6 CONCLUSION](#6-conclusion)
- [A APPENDIX](#a-appendix)
- [B CASE STUDY](#b-case-study)
- [C COMPLETE OPTIMIZATION TRAJECTORY OF THE MATH DATASET](#c-complete-optimization-trajectory-of-the-math-dataset)

## Abstract
**AFLOW: Automating Agentic Workflow Generation**

**Abstract**:
- Large language models (LLMs) have demonstrated remarkable potential in solving complex tasks across diverse domains
- Construction of agentic workflows requires significant human effort, limiting scalability and generalizability
- Recent research has sought to automate workflow generation and optimization, but existing methods rely on initial manual setup and fall short of fully automated and effective workflow generation

**Approach**:
- Reformulate workflow optimization as a search problem over code-represented workflows
- Introduce **AFLOW**: an automated framework that efficiently explores this space using Monte Carlo Tree Search, iteratively refining workflows through code modification, tree-structured experience, and execution feedback

**Evaluation**:
- Empirical evaluations across six benchmark datasets demonstrate AFLOW's efficacy, yielding a 5.7% average improvement over state-of-the-art baselines
- AFLOW enables smaller models to outperform GPT-4o on specific tasks at 4.55% of its inference cost

**Conclusion**:
- The code will be available at https://github.com/geekan/MetaGPT

## 1 INTRODUCTION

**Introduction:**
* Large Language Models (LLMs) have become powerful tools across various domains
* Rapid advancement relies on manually designed agentic workflows, which require significant human effort
* Recent efforts focus on automating the discovery of effective agentic workflows to reduce reliance on human intervention
* Automated methods struggle to capture full diversity of workflows and optimize performance within limited iterations

**Challenges:**
1. Difficulty representing diverse requirements, operations, and dependencies for each task
2. Virtually boundless search space for possible workflows, making efficient exploration challenging

**Proposed Framework: AF LOW**
* Models the workflow as a sequence of interconnected LLM-invoking nodes
* Nodes represent actions; edges define logic, dependencies, and flow between actions
* Workflow modeled as graph or network, capturing complex interactions

**Enhancements:**
1. Operators: Predefined, reusable combinations of nodes for common agentic operations
2. MCTS algorithm to navigate infinite search space
3. Soft mixed-probability selection mechanism for node exploration
4. LLM-driven node expansion to introduce new possibilities
5. Execution evaluation to assess workflow performance
6. Backpropagation of experience to refine future search iterations

**Key Contributions:**
1. Unified framework for future research on workflow optimization at both node and method levels
2. AF LOW: MCTS-based method that automatically discovers effective workflows across multiple domains with minimal human intervention
3. Extensive evaluation demonstrating superior performance compared to manually designed methods and existing automated approaches, enabling smaller LLMs to outperform larger models for better cost-performance efficiency.

## 2 RELATED WORK

**Related Work: Agentic Workflow vs Autonomous Agents**

**Agentic Workflow**:
- Represented by static tasks completed through predefined processes
- Multiple LLM invocations used for solving problems
- Categorized into general and domain-specific types
  - General workflows: Universal problem-solving approaches
  - Domain-specific workflows: Effective processes to solve specific problems (e.g., code generation, data analysis)

**Autonomous Agents**:
- Distinct paradigm from agentic workflow
- Dynamic problem solving through flexible autonomous decision making
- Require specific actions and design patterns for the environment

**Existing Work on Agentic Workflows**:
- Manually discovered numerous effective workflows
- Challenging to exhaust various tasks across different domains
- Importance of automated workflow generation and optimization

**Automated Agentic Optimization**:
- Three types: prompt optimization, hyperparameter optimization, and automated workflow optimization
  - **Prompt optimization**: LLMs optimize prompts within fixed workflows
  - **Hyperparameter optimization**: Focuses on optimizing predefined parameters
  - **Automated workflow optimization**: Optimizes entire workflow structures
    - Offers more potential for fully automated generation
- Recent works explore diverse representations and methods: GPTSwarm, ADAS, AFLOW

**GPTSwarm**:
- Uses graph structures with reinforcement learning
- Struggles to represent workflows with conditional states due to graph structure limitations

**ADAS**:
- Utilizes code structures to represent workflows
- Stores historical workflows in a linear list structure
- Challenged by the efficiency of its search algorithm and simplistic representations

**AFLOW**:
- Uses code to represent workflows
- Introduces named node structure with various LLM invocation parameters
- Provides operators for predefined node combination functions
- Employs a specially designed MCTS algorithm for automated workflow optimization
  - Leverages tree-structured experience and execution feedback to efficiently discover effective workflows.

## 3 PRELIMINARY

**Section Overview:**
- Formulate automated agentic workflows generation problem in Section 3.1
- Discuss design considerations for AF LOW in Section 3.2
- Example explanation in Figure 2

**Role:** Helpful assistant
**Approach:** Reason and act based on context
**Deliverable:** Generate answer based on provided context

**Tempertrue: [0,1]**
- Models NodeOperator
- Generate Node Ensemble
- Review Node
- Judge Node
- Revise Node
- Multi-Agent Debate
- History Conditions
- Self Refine Conditions
- Self Consistency

**Figure 2:** Example of nodes, operators, and edges
- Optional parameters for Nodes
- Structure of some Operators
- Common representations of Edges

### 3.1 PROBLEM FORMULATION

**Agentic Workflow**
- **Workflow**: Sequence of LLM-invoking nodes (N) representing specific operations performed by an LLM
- Each node:
  - Characterized by parameters: Model M, Prompt P, Temperature τ, Output format F
  - Connected by edges E representing the sequence of execution
- **Edge Structures**:
  - Graph: Flexible structure representing hierarchical, sequential, or parallel relationships between nodes
  - Neural Network: Represents complex, non-linear relationships between nodes
  - Code: Comprehensive representation expressing linear sequences, conditional logic, loops, and network structures

**Automated Workflow Optimization**
- Given a task T and evaluation function G, the goal is to discover a workflow W that maximizes G(W, T)
- Search process where an algorithm A explores the search space S to determine the optimal workflow configuration
- **Search Space**: Encompasses all possible configurations of node parameters and edge structures
  - N: {(M, τ, P, F)|M∈ M, τ∈[0,1], P∈ P, F∈ F}
  - E: Representing sets of possible language models, prompts, output formats, and edge configurations
- **AF LOW Framework**:
  - Sets the search space to nodes with only prompt parameters as flexible
  - Uses MCTS-based search within this space to iteratively execute Soft Mixed Probability Selection, LLM-Based Expansion, Execution Evaluation, and Experience Backpropagation until maximum iterations or convergence criteria are met

### 3.2 AFLOW OVERVIEW

**AFLOW Overview**
* Addresses limitations of previous workflow optimization methods
* Uses Large Language Models (LLMs) within Monte Carlo Tree Search (MCTS) to explore full range of possible agentic workflows
* Represents nodes N and edges E through code, ensuring completeness in search space
* Variant of MCTS iteratively explores workflow search space, evaluates configurations, and backpropagates experiences for refinement
* Simplifies search by fixing key parameters: model M, temperature τ, format F
* Operators O encapsulate common agentic operations for efficient utilization
	+ Generate, Format, Review/Revise, Ensemble, Test, Programmer (see Appendix A.4 for detailed structures)
	+ Easy to expand for various tasks or perform searches with an empty Operator Set
* Optimization problem formalized as SAFlow: {(P1, ... , Pn, E, O1, ... , On)| Pi∈ P, E∈ E, Oi∈ O} (1)
	+ W∗ = AFLOW(SAFlow, G, T ) (2)
* Applies to reasoning tasks with easily obtainable evaluation functions.

## 4 THE DESIGN DETAILS OF AFLOW

**The Design Details of AFLOW+**

**Core Concept**:
- Employ Large Language Models (LLMs) as optimizers to modify code-represented workflows within a search structure based on Monte Carlo Tree Search (MCTS) variant.

**Iterative Process**:
1. Soft mixed probability selection
2. LLM-based optimization expansion
3. Execution evaluation
4. Experience backpropagation
5. Dynamic convergence assessment
6. Repeat until maximum iterations or meets convergence criteria

**Existing Workflow Optimization Methods**:
- Iteratively use past workflow structures to prompt LLMs to discover new structures
- Struggles due to information loss during accumulation and vast search space, reducing search efficiency

**Key Idea**:
- Leverage the tree structure of MCTS to preserve node-based exploration experiences in workflow optimization
- Prevent local optima by introducing a special selection mechanism allowing generation from a blank template at any round

**Initialization**:
- Start with a template workflow that provides a framework for invoking nodes and operators
- Randomly partition the dataset into a validation set (20%) and a test set (80%)
- Execute the blank template 5 times on the validation dataset to select a subset of problems with high variance in scores

**Selection**:
- Evaluate an empty workflow on the validation set as the initial node
- Continuously select workflows based on a soft mixed probability selection strategy

**Expansion**:
- Employ an LLM optimizer to create new workflows, leveraging the selected workflow's experience to modify node connections
- Maximizes insights from past iterations by including all modifications and their corresponding improvements or failures

**Evaluation**:
- Directly execute workflows to obtain feedback through explicit evaluation functions
- Test each generated workflow 5 times on the validation set, computing mean and standard deviation

**Backpropagation**:
- Store performance information and optimizer's modifications for use in the selection phase
- Add performance score to the global performance record

**Terminal Condition**:
- Implement an early stopping mechanism to avoid unnecessary costs after optimization reaches its limit.

## 5 EXPERIMENTS
### 5.1 EXPERIMENTAL SETUP

**Experiments: Automated Workflow Optimization (AFLOW) vs Manually Designed Methods**

**Datasets:**
- GSM8K, HumanEval, MBPP, HotpotQA, DROP, MATH used for experiments
- Validation and test sets divided using a 1:4 ratio
- Full datasets for GSM8K, HumanEval, MBPP
- Randomly selected 1,000 samples each for HotpotQA and DROP
- 617 problems from four typical problem types in MATH at difficulty level 5

**Benchmarks:**
- Comparison of performance between manually designed methods and workflows generated by AFLOW with various executor LLMs: GPT-4o-mini ("Ours") and DeepSeek-V2.5 ("Ours*")
- All workflows tested thrice on the divided test set, with average results reported

**Baselines:**
- Comparison against manually designed methods for LLMs: IO (direct LLM invocation), Chain-of-Thought (CoT), Self Consistency CoT (5 answers), MultiPersona Debate, Self-Refine, and MedPrompt
- Comparison against automated workflow optimization method ADAS

**Implementation Details:**
- AFLOW uses different models for optimization and execution: Claude-3.5-sonnet as optimizer, DeepSeek-V2.5, GPT-4o-mini-0718, Claude-3.5-sonnet-0620, GPT-4o-0513 as executors
- All models accessed via APIs
- Temperature set to 1 for DeepSeek-V2.5 and 0 for other models
- Iteration rounds set to 20 for AFLOW, 30 for ADAS

**Metrics:**
- Solve Rate (%) as primary metric for GSM8K and MATH lv5*
- Pass@1 metric for HumanEval and MBPP to assess code accuracy
- F1 Score for HotpotQA and DROP
- Cost calculated by tracking token usage to construct a pareto front, demonstrating performance-cost trade-offs between different methods.

### 5.2 EXPERIMENTAL RESULTS AND ANALYSIS

**Experimental Results and Analysis**

**Main Experimental Results:**
- AF LOW outperforms manually designed methods by an average of 5.7%
- Surpasses contemporary automated workflow optimization methods by 19.5%
- Achieves an average performance of 80.3% across six datasets in QA, Code, and Math domains
- Improves over ADAS on MATH lv5∗and MBPP tasks by 57%, demonstrating robustness on complex datasets

**Cost Analysis:**
- AF LOW can identify workflows that allow weaker models to outperform stronger models on the pareto front of cost-effectiveness
- Eliminates human labor costs previously required for automated workflow optimization
- Opens up further possibilities for widespread adoption by achieving superior performance at lower costs compared to stronger models

**Ablation Study:**
- AF LOW with operators discovers better-performing workflows within the same number of iterations, exhibiting a trend of multiple small improvements
- Operators effectively boost search efficiency by introducing a constrained search space
- Even without operators, AF LOW achieves 93.1% performance, surpassing other manually designed methods
- Autonomously develops an ensemble-like structure, demonstrating its advantage as an optimizer for searching code-represented edges

**Case Study:**
- AF LOW evolves from a blank template to the structure presented in Figure 5(B) through single-step modifications
- Unsuccessful exploration nodes introduce custom review and verification nodes that decreased accuracy
- Demonstrates advantage as an optimizer for searching code-represented edges, enabling it to independently design efficient structures for problems.

## 6 CONCLUSION

**Automated Workflow Optimization: AF LOW Framework**

**Conclusion:**
- Introduced AF LOW, a novel framework for automated workflow optimization
- Formulated the problem and established foundational structure for future research
- Leveraged Monte Carlo Tree Search and code-represented workflows to navigate search space efficiently
- Demonstrated effectiveness of AF LOW on six benchmarks:
  - Outperformed manually designed methods and existing automated optimization approaches
  - Enabled weaker models to outperform stronger ones on Pareto front of cost-effectiveness
- Potential for enhancing LLMs' problem-solving capabilities while optimizing computational costs.

## A APPENDIX

**LLM-Based Expansion:**
* **Graph and Prompt Optimization**: Reconstruct and enhance LLM graph and corresponding prompt for problem solving.
* Use XML tags for modifications in responses to avoid runtime failures.
* Incorporate critical thinking methods like review, revise, ensemble (multiple answers generation through different/similar prompts, voting/integrating/checking the majority), self-Ask.
* Consider Python loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), or machine learning techniques for optimization.
* Limit graph complexity to 10.
* Include all required prompts in `prompt_custom`.
* Generate only necessary prompts within `prompt_custom`, not those already built-in.
* Ensure generated prompts do not contain placeholders.

**Node Structure:**
* **ActionNode**: Fill method to process node based on context, LLM, and schema format (text, json, markdown, xml).
* Determine example and output format based on format passed in the fill() method call.

**Workflow Structure:**
* **Workflow**: Initialize name, dataset type, LLM config, and create LLM instance.
* Implement workflow logic by subclassing Workflow class and overriding __call__ method.

**Operators:**
* **ContextualGenerate**, **CodeGenerate**, **Format**, **Review**, **Revise**, **Ensemble**, **Test**, and **Programmer**: Predefined operators to enhance search efficiency in AFLOW.

**MCTS Algorithm (AFLOW):**
* Detailed explanation of the AFLOW algorithm with initial workflow, evaluator, dataset, number of rounds, operators, top k, and early stopping rounds required.
* Select high variance instances for validation based on scores from previous round.
* Optimize workflow modification using LLM as optimizer.
* Execute new workflow on dataset to obtain score and cost.
* Repeat process for specified number of rounds, updating best score and results accordingly.
* If top k workflows remain unchanged in n rounds, return the optimal workflow.

## B CASE STUDY

**Case Study: AFlow's Workflow Optimization using Custom Operators**

**AFlow's Workflow for Mathematical Problem Solving:**
- Generates code solutions using `custom_code_generate` operator (Code Generate Prompt)
- Ensembles best solutions using `sc_ensemble`
- Tests solutions and fixes errors if necessary
  - If test fails: uses `custom` to fix the error and retest
- Combines initial response with refined solution for comparison
- Selects most accurate solution using `compare_and_select` prompt

**AFlow's Workflow for HotpotQA:**
- Generates solutions using diverse approaches: algebraic (Solve Approach1), visual/diagrammatic (Solve Approach2), or estimation/approximation techniques (Solve Approach3)
- Compares and selects the most accurate solution using `compare_and_select` prompt

**Optimal Workflow Generation:**
- AFLOW generates an ensemble of solutions for given problem input
- Each solution is evaluated based on correctness, completeness, and consistency with the problem statement
- The best solution is selected as final answer

**AFlow's Flexibility in Tailoring Workflows:**
- AFLOW adapts workflows to different problem domains
- Maintains sophisticated problem-solving structures while maintaining flexibility

**Comparison with ADAS:**
- In contrast to ADAS, AFlow designs optimal workflows that reduce human effort and improve efficiency.

**ADAS Workflow for HotpotQA:**
- Initial reasoning by diverse expert agents (Reading Specialist, Logic Specialist, Generalist)
- Iterative refinement with external knowledge integration
  - Retrieve relevant information from a knowledge base
  - Verify the relevancy and accuracy of the retrieved information
  - Refine insights using verified knowledge
- Final synthesis to provide a final answer.

## C COMPLETE OPTIMIZATION TRAJECTORY OF THE MATH DATASET

**Math Dataset Optimization Trajectory**

**Operators**:
- **"3":**
    - **score:** 0.5277310924369748
    - **success:**
        * **14**: Modify Custom operator for more detailed solution, add new Custom operator to refine answer
    - **failure:**
        * **13**: Modify Custom operator for more detailed solution, add new Custom operator to format answer
- **"5":**
    - **score:** 0.5512605042016807
    - **success:**
        * Generate detailed step-by-step solution, compare and select best one from multiple approaches using ScEnsemble operator
    - **failure:**
        * No modifications suggested
- **"9":** (twice)
    - **score:** 0.5378151260504201
    - **success:**
        * Generate detailed step-by-step solution, compare and select best one from multiple approaches using ScEnsemble operator
    - **failure:**
        * No modifications suggested
- **"10":** (failure)
    - **score:** 0.5042016806722688
    - **failure:**
        * Generate step-by-step solution, compare and select best one from multiple approaches using ScEnsemble operator
- **"13":** (failure)
    - **score:** 0.5193277310924369
    - **failure:**
        * Modify Custom operator for more detailed solution, add new Custom operator to refine and format answer
- **"4":**
    - **score:** 0.0
    - **failure:**
        * No modifications suggested
- **"11":** (failure)
    - **score:** 0.5159663865546219
    - **failure:**
        * Add new Custom operator for comprehensive solution approach, incorporate into ensemble process
- **"12":** (failure)
    - **score:** 0.0
    - **failure:**
        * Generate multiple solution approaches, select best one using ScEnsemble operator
- **"15":** (failure)
    - **score:** 0.5243697478991596
    - **failure:**
        * Add new Custom operator to generate multiple solutions, select best one using ScEnsemble operator
- **"16":** (failure)
    - **score:** 0.5210084033613446
    - **failure:**
        * Generate multiple solution approaches, select best one using ScEnsemble operator
- **"17":** (deepseek failure)
    - **score:** 0.0
    - **failure:**
        * Add ScEnsemble operator to generate multiple solutions and select the best one
- **"18":** (failure)
    - **score:** 0.5176470588235293
    - **failure:**
        * Modify Custom operator for more detailed solution, compare and select best one from generated solutions using ScEnsemble operator
- **"19":** (deepseek failure)
    - **score:** 0.5445378151260505
    - **failure:**
        * Add new Custom operator for simplified solution, incorporate into ensemble process with existing detailed solution

