# Agent S: An Open Agentic Framework that Uses Computers Like a Human

**Authors**: Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, Xin Eric Wang (Equal Contributions)
https://arxiv.org/html/2410.08164v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Agent S](#3-agent-s)
  - [3.1 Experience-augmented Hierarchical Planning](#31-experience-augmented-hierarchical-planning)
  - [3.2 Memory Construction and Update](#32-memory-construction-and-update)
  - [3.3 Agent-Computer Interface](#33-agent-computer-interface)
- [4 Experiments](#4-experiments)
  - [4.1 Experimental Setup](#41-experimental-setup)
  - [4.2 Main Results](#42-main-results)
  - [4.3 Ablation Study](#43-ablation-study)
  - [4.4 Error Analysis](#44-error-analysis)
  - [4.5 Generalization to Different Operating Systems](#45-generalization-to-different-operating-systems)
- [5 Conclusion](#5-conclusion)

## Abstract

**Agent S: An Open Agentic Framework for Automating Computer Tasks**

**Overview**:
- Agent S is an open agentic framework that enables autonomous interaction with computers through a Graphical User Interface (GUI)
- Aims to transform human-computer interaction by automating complex, multi-step tasks
- Addresses three key challenges:
    - **Acquiring domain-specific knowledge**
    - **Planning over long task horizons**
    - **Handling dynamic, non-uniform interfaces**

**Key Components**:
- **Experience-augmented hierarchical planning**: Learns from external knowledge search and internal experience retrieval at multiple levels, facilitating efficient task planning and subtask execution
- **Agent-Computer Interface (ACI)**: Elicits the reasoning and control capabilities of GUI agents based on Multimodal Large Language Models (MLLMs)

**Performance**:
- Outperforms baseline by 9.37% on success rate (an 83.6% relative improvement)
- Achieves new state-of-the-art on the OSWorld benchmark

**Generalizability**:
- Demonstrates broad generalizability to different operating systems on a newly-released WindowsAgentArena benchmark

**Availability**:
- Code available at [https://github.com/simular-ai/Agent-S](https://github.com/simular-ai/Agent-S)

## 1 Introduction

**Autonomous GUI Agents for Interactive Systems**

**Significance of Digital Revolution**:
- Douglas Engelbart: "The digital revolution is far more significant than the invention of writing or even of printing."

**GUI Agents**:
- Offer promise in solving specific user queries through direct UI interaction using mouse and keyboard
- Can boost efficiency and improve accessibility for individuals with disabilities

**Challenges in Automating Computer Tasks**:
1. Require up-to-date domain knowledge and ability to learn from open-world experience
2. Need for long-horizon, multi-step planning with interdependent actions in specific sequence
3. Navigating dynamic, non-uniform interfaces while processing large volumes of visual and textual information

**Experience-Augmented Hierarchical Planning**:
- Proposed approach to enhance GUI agent's capabilities in solving diverse tasks
- Leverages Online Web Knowledge and Narrative Memory to decompose complex tasks into manageable subtasks

**Agent-Computer Interface (ACI)**:
- Abstraction layer to improve grounding, safety, and efficiency for MLLM-based GUI agents
- Defines a dual-input strategy and bounded action space for effective interaction

**Performance Improvement**:
- Agent S shows remarkable improvement in overall performance on OSWorld benchmark (from 11.21% to 20.58%)
- Consistent improvements across five broad computer task categories over the OSWorld agent
- Performance improvement from 13.3% to 18.2% on WindowsAgentArena

**Contributions**:
- Introduce Agent S, a new agentic framework with experience-augmented hierarchical planning, self-supervised continual memory update, and ACI for MLLM-based GUI agents
- Propose an experience-augmented hierarchical planning method using external web knowledge and internal memory
- Extend the concept of ACI to GUI agents, allowing high-level primitive actions for effective interaction
- Conduct extensive experiments on OSWorld to show effectiveness of individual components and establish new state-of-the-art in automating computer tasks.

## 2 Related Work

**MLLM Agents:**
* Multimodal Large Language Models (MLLMs) used as reasoning backbone in Agentic Systems
* Augmented with Memory, Structured Planning, Tool Use, and ability to Act in external environments
* Show promise in various domains: embodied simulators, video games, scientific research, software engineering
* Proposed Agent-Computer Interface (ACI) for more efficient and reliable interaction
* Individual modules integrated into new MLLM agent framework

**GUI Agents:**
* Applied to execute natural language instructions in web and OS environments
* Early research focused on web navigation tasks using behavioral cloning, reinforcement learning, etc.
* Shifted to OS-level environments with benchmarks like OSWorld, WindowsAgentArena, DiGIRL, AndroidWorld
* Offer broader control capabilities beyond single-browser contexts
* Methodologically varied: behavioral cloning, in-context trajectory examples, state-dependent offline experience, reusable skill generation
* Recent work proposes cognitive architectures for video games and OS tasks
* Our work contributes unique modules like experience-augmented hierarchical planning and ACI, with a novel continual memory update framework

**Retrieval-Augmented Generation (RAG):**
* Improves reliability of MLLM inference by augmenting input with external knowledge
* MLLM agents benefit from retrieving task exemplars, state-aware guidelines, past experiences
* Our use of experience for augmentation differs: 1) hierarchical planning uses full and subtask experience; 2) full task experience summarized into abstractive textual reward; 3) subtask experience assessed by self-evaluator before storage in memory.

## 3 Agent S

**Agent S Framework:**
* Integrates three main strategies: experience-augmented hierarchical planning, continual update of narrative and episodic memory, Agent-Computer Interface (ACI) for precise perception and action on GUIs.

**Experience-Augmented Hierarchical Planning:**
- Breaks down complex tasks into manageable subtasks
- Allows high-level planning and low-level execution to draw from external web-based experience and internal task-specific experience.

**Continual Update of Narrative and Episodic Memory:**
- Stores and retrieves self-evaluated task experience
- Enables improvement over time and adaptation to changes in open-world desktop environment.

**Agent-Computer Interface (ACI):**
- Provides vision-augmented accessibility tree observation
- Contains all valid GUI elements
- Constrains chosen action to a bounded discrete space of valid actions.

**Description of Components:**
* Experience-augmented hierarchical planning: breaks down tasks into manageable subtasks for effective planning and execution, using external and internal experience.
* Continuous learning through narrative and episodic memory: enables improvement over time by storing self-evaluated task experiences.
* Agent-Computer Interface (ACI): ensures grounding by providing precise perception and action capabilities on GUIs within a constrained environment.

### 3.1 Experience-augmented Hierarchical Planning

**Manager (G): Fusing External Knowledge and Internal Experience for Planning**
* The Manager G is the primary plan generator module that receives:
	+ User task Tu from user
	+ Initial environment observation O0 (Annotated Accessibility Tree + Screenshot) from ACI
* Formulates an **observation-aware query Q** based on user instruction and observation in "How to do X" format
* Uses query for:
	+ Online Web Search through Perplexica Search Engine to get external knowledge Kweb
	+ Retrieval of similar task experience summary Enu from Manager's own Narrative Memory Mn
* Fuses retrieved information into a single **fused guideline** using Experience Context Fusion submodule
* Uses fused knowledge for detailed, topologically sorted queue of subtasks ⟨s0..sn⟩ to accomplish user instruction
* Generates associated context Csi for each subtask si

**Worker: Learning from Subtask Experience and Trajectory Reflection**
* Worker modules w0..wn execute sequential subtasks si generated by Manager G
* Retrieve similar subtask experience Esi from Worker's Episodic Memory based on ⟨Q,si,Csi⟩ query
* Episodic Memory includes complete plans with specific grounding actions and summaries designated as DONE or successful
* Trajectory Reflector TRi observes entire episode and provides reflective advice to agent
* Experience Esi and reflection used by Action Generator inside Worker to generate structured response: previous action status check, observation analysis, semantic next action, grounded next action
* Grounded action aj passed to ACI for implementation in Desktop Environment
* If subtask completed successfully (DONE), Self-Evaluator S generates learning summary of strategy used which is fed back into Worker's episodic memory Me
* If maximum number of steps limit reached or user-provided task complete, Self-Evaluator generates learning summary of entire task completion process and saves it in Manager's narrative memory Mn

**Self-Evaluator: Summarizing Experiences as Textual Rewards**
* Responsible for generating experience summaries r as textual rewards for Manager and Worker modules
* Generates learning summary of strategy used by worker to complete subtask when DONE signal received
* Feeds back learning summary to update Worker's episodic memory Me
* Generates learning summary of entire task completion process upon successful completion of all subtasks or maximum number of steps limit
* Saves learning summary in Manager's narrative memory Mn

**Figure 4: Memory Construction and Update Pipeline**
* Contains two phases: Self-supervised Exploration and Continual Memory Update
* Initial Narrative & Episodic Memory constructed through random curated tasks during exploration phase
* Updated based on inference tasks continually

### 3.2 Memory Construction and Update

**Initial Memory Construction via Self-supervised Exploration**

**Self-supervised exploration on synthetically generated tasks**:
- Agent S conducts self-supervised exploration on a set of tasks
- Two types of exploration tasks: environment-independent and environment-aware

**Environment-independent tasks**:
- Generated using a task generator from applications in OSWorld and WindowsAgentArena
- Top 50 most common tasks are selected

**Environment-aware tasks**:
- Initial environments of tasks in OSWorld and WindowsAgentArena are used to generate new tasks
- Tasks are prompted to be different based on the environment

**Running Agent S on exploration tasks**:
- Collected full task experiences (Narrative Experience En) and subtask experiences (Episodic Experience Ee)
- Keys for memory storage:
  - Narrative Memory Mn: query Q
  - Episodic Memory Me: query Q concatenated with subtask information ⟨Q, si, Csi⟩

**Continual Memory Update**:
- As Agent S interacts with new tasks, it continually updates the Narrative Memory Mn and Episodic Memory Me
- Enables Agent S to learn even during inference and retrieve learned knowledge for new tasks

### 3.3 Agent-Computer Interface

**Agent-Computer Interface (ACI) for MLLM Agents:**
* Designed to accommodate two user types: human users and software programs
* Inadequate for MLLM agents due to unique operational constraints
* Inspired by ACI developed for Software Engineering agents
* Bridges gap between MLLM agent requirements and GUI control tasks

**Perception and Grounding:**
- Current MLLMs can reason about elements but lack internal coordinate system
- Agents need to interact with fine UI elements and constantly observe environment
- Desktop environments provide Accessibility Tree for parsing coordinate information
- Dual-input strategy: image input observes salient details, accessibility tree input reasons and grounds elements
- Tag each element in the accessibility tree with unique integer tags
- Augment accessibility tree with textual blocks from screenshot using OCR module

**Constrained Action Space:**
- Traditional desktop automation relies on APIs and scripts, unsuitable for MLLM agents due to unbounded action space
- Compromises safety and precision by allowing arbitrary executable code as actions
- Incorporates bounded action space with primitive actions (click, type, hotkey)
- Agents refer to elements by tagged IDs, ACI translates into executable Python code
- Agents perform only one discrete action at each time step for immediate feedback.

## 4 Experiments
### 4.1 Experimental Setup

**Evaluation of Agent S**
- **OSWorld Benchmark**:
    - Tests multimodal agents' capability to execute various computer tasks
    - Includes OS, Office, Daily, Professional, and Workflow apps on Ubuntu
    - Uses GPT-4o and Claude-3-Sonnet as backbone models for Agent S evaluation
    - Inputs: accessibility tree and screenshot
    - Baseline: takes coordinates-based accessibility tree and screenshots as input, generates action with coordinates at each step

- **WindowsAgentArena Benchmark**:
    - Evaluates Agent S' generalization on Windows operating system
    - Includes 154 tasks for GPT-4o as backbone model
    - Uses PaddleOCR2 toolkit for OCR and text-embedding-3-small embedding model for retrieval
    - Inputs: accessibility tree and screenshot
    - Baseline: NAVI, which utilizes accessibility tree, OCR, and proprietary models to process screenshots and create Set-of-Marks as input. Its action space includes a constrained set of primitives but allows multiple actions to be chained together.

### 4.2 Main Results

**Performance Comparison of Agent S and Baseline Models (OSWorld)**
* **Table 1**: Comparison of Success Rates (%)**
  * Agent S vs GPT-4o: +9.37% overall, doubling performance in Daily and Professional tasks
  * Agent S outperforms all baselines on "Daily" and "Professional" tasks
* **Claude-3.5-Sonnet** and GPT-4o perform better than baselines in majority of tasks
* Enhanced capability of Agent S in handling diverse, complex tasks

**Qualitative Examples (OSWorld's Thunderbird App)**
* Task: Remove account "anonym-x2024@outlook.com"
* Agent S completes tasks by interacting with desktop through a combination of actions: `agent.click()`
* Successful examples demonstrated in Appendix D.1

**Figure 5**: Qualitative Examples (Thunderbird Task)
* Open Account Settings: agent.click(41, 1, “left”)
* Remove the Account: agent.click(86, 1, “left”)
* Remove the Account: agent.click(149, 1, “left”)

### 4.3 Ablation Study

**Agent S Ablation Study Findings**

**Background:**
- Subset of 65 instances (OSWorld tests) used for study
- GPT-4o as LLM backbone for all ablation studies

**Impact of Experience Components:**
| Component | Successful Rate (%)|
| --- | --- |
| baseline (OSWorld Agent) | 10.77 |
| Agent S (with all components) | 26.15 |
| - w/o Web Knowledge | 16.80 |
| - w/o Narrative Memory | 21.41 |
| - w/o Episodic Memory | 18.46 |
| - w/o All | 13.85 |

**Effects of ACI Module:**
- Enhances reasoning abilities of LLMs
- Supports better agentic learning
- Improves performance significantly when added to Agent S (ACI-only)

**Importance of Hierarchical Planning:**
- Models long-horizon workflows effectively
- Significant performance improvement with full Agent S setup

**Impact of Exploration, Continual Memory Update, and Self-Evaluator:**
- Self-supervised exploration: crucial for memory construction
- Continual memory update: important for updating memories
- Self-evaluator: stores experience as summaries instead of unfiltered trajectories
  - Reveals benefits in Figure [7](https://arxiv.org/html/2410.08164v1#S4.F7) when compared to full trajectories.

**Ablation Findings:**
- Removing exploration: performance drop to 20.00%
- Removing continual memory update: performance drop to 20.00%
- Removing self-evaluator: performance drop to 13.85%

**Conclusion:**
- Learning from experience enhances domain knowledge for GUI agents
- Each component plays a critical role in Agent S's performance improvement.

### 4.4 Error Analysis

**Error Analysis of Agent S in OSWorld**

**Types of Errors:**
- **Planning Error**: Unsuitable plans generated for tasks:
  * Inaccuracies in plans
  * Misleading subtask information
  * Misalignment of subtask sequence with task requirements
- **Grounding Error**: Failure to interact accurately with target elements despite visibility and correct reasoning:
  * Incorrect element selection
  * Inaccurate coordinate selection due to limitations of action space (e.g., selecting center instead of precise part)
- **Execution Error**: Incorrect decisions or failure to adjust behavior during task execution:
  * Repetitive actions
  * Diverging from subtask goals
  * Delays in transitioning between subtasks
  * Violation of established protocols by combining multiple actions into one

**Error Statistics:**
- **Table 3**: Error Rate (%) for Agent S on OSWorld tasks:
  | Error Metric | Office | Daily | Professional | Workflow | Overall |
  | --- | --- | --- | --- | --- | --- |
  | Planning Error | 25.00% | 30.00% | 66.67% | 66.67% | 34.69% |
  | Grounding Error | 0.00% | 75.00% | 50.00% | 66.67% | 53.06% |
  | Execution Error | 87.50% | 100.00% | N/A | 71.43% | 79.59% |
  * Single task may contain multiple errors
- **Subtask Failure Rate**: Average percentage of failed subtasks relative to total attempts
- **Error Rate**: Proportion of tasks exhibiting a specific error type

**Observed Error Types:**
- Execution and grounding errors are most common across various task categories.

**Detailed Error Analysis**: (Appendix D.2) for case study on error occurrence.

### 4.5 Generalization to Different Operating Systems

**Agent S Framework Testing Results**

**WindowsAgentArena Benchmark**:
- Agent S outperforms the Navi agent on this Windows OS benchmark
- Comparison with similar GPT-4o configuration and input methods

**Performance Metrics**:
- Successful rate (%) on 154 test examples across various tasks: Office, Web Browser, Windows System, Coding, Media & Video, Windows Utils

**Comparison to Navi Agent**:
- Agent S performs better overall (18.2%) compared to Navi agent's performance (13.3%)

**Individual Task Performance**:
| Method | Office | Web Browser | Windows System | Coding | Media & Video | Windows Utils | Overall |
|---|---|---|---|---|---|---|---|---
| **Agent S** | 0.0% | 13.3% | 45.8% | 29.2% | 19.1% | 22.2% | 18.2% |
| NAVI (Bonatti et al., [2024](https://arxiv.org/html/2410.08164v1#bib.bib3)) | 0.0% | 20.0% | 29.2% | 9.1% | 25.3% | 0.0% | 13.3% |

## 5 Conclusion

**Agent S Framework:**
* **Autonomous GUI agents** that perform user queries by controlling keyboard and mouse
* Demonstrates benefits of Learning from Experience for Task-oriented agents
* Introduces concept of Agent Computer Interface (ACI) for GUI domain
	+ Allows MLLM agents to perceive and reason at language level with rich feedback
* Uses Experience-Augmented Hierarchical Planning, Online Web Knowledge, and ACI
* Achieves SOTA performance on OSWorld benchmark and generalizability across OS
* Agents can learn from external sources and environment without human or environmental feedback in the GUI agents domain

**Future Work:**
* Address unaddressed metrics: number of agent steps and wall clock time for task completion
* Consider shortest-path navigation formulation of GUI control to evaluate Pareto-optimality on dimensions of time and accuracy
* Extend ideas of experiential learning and Agent Computer Interface to smaller, open-source MLLMs for fine-tuning.

