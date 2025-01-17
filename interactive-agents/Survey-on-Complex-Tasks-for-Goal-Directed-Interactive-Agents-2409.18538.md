# A Survey on Complex Tasks for Goal-Directed Interactive Agents

by Mareike Hartmann and Alexander Koller
https://arxiv.org/pdf/2409.18538

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Scope of this Survey](#2-scope-of-this-survey)
- [3 Example Tasks](#3-example-tasks)
  - [3.2 Digital Assistance](#32-digital-assistance)
- [4 Structuring the Task Landscape](#4-structuring-the-task-landscape)
  - [4.2 Goals](#42-goals)
  - [4.3 World and Knowledge States](#43-world-and-knowledge-states)
  - [4.4 Actions](#44-actions)
  - [4.5 Observations](#45-observations)
  - [4.6 Task Evaluation](#46-task-evaluation)
  - [4.7 General Properties of the Environment](#47-general-properties-of-the-environment)
- [5 Discussion and Future Directions](#5-discussion-and-future-directions)

## Abstract
**Goal-Directed Interactive Agents: A Survey on Complex Tasks**

**Introduction:**
- Goal-directed interactive agents assist humans in daily life
- Recent advances in LLMs led to new, challenging tasks for evaluation
- This survey compiles and structures relevant tasks and environments based on challenges

**Background:**
- Agents can complete tasks through interactions with environment
- Surge of new tasks due to advanced language models
- Importance of understanding challenges posed by these tasks

**Resources:**
- Up-to-date compilation available at: https://coli-saar.github.io/interactive-agents

**Compilation and Structuring:**
- Relevant tasks and environments for goal-directed interactive agents
- Organized along dimensions relevant to current obstacles.

**Conclusion:**
- Proper contextualization of performance across tasks essential for understanding challenges.

## 1 Introduction

**Tasks for Goal-Directed Agents Interacting with Task Environments: A Survey**

**Introduction**:
- Recent work on large language models (LLMs) and tool use aims to fundamentally change human-computer interaction
- Previously, users had to issue commands or click GUI elements to execute computer actions
- Future users can delegate high-level tasks to the computer, which decomposes into direct commands/actions

**Progress in LLM Agents**:
- Rapid advancements in:
  - Reasoning over contexts (Wei et al., 2022; Yao et al., 2023)
  - Decomposing problems (Prasad et al., 2024)
  - Deciding on tools/actions to use (Schick et al., 2023) or take (Li et al., 2022)
- Mostly based on large in-context learning LLMs (Brown et al., 2020)

**Development of Tasks**:
- Increasing breadth, naturalness, and difficulty for agent evaluation
- Various tasks:
  - Managing email conversations with friends (APPWORLD, Trivedi et al., 2024)
  - Answering complex questions (HOTPOTQA REACT, Yang et al., 2018)
  - Online shopping (MIND 2WEB, Deng et al., 2023)
  - Performing complex tasks in situated environments (MINEDOJO, Fan et al., 2022)
- Rapid development of tasks drives understanding of agent abilities/limitations and advancement of architectures

**Need for Survey**:
- Tracking, interpreting results, and understanding challenges for each task becomes difficult with rapid developments
- Offering a survey on the current landscape of tasks for goal-directed agents complements existing surveys on agent architectures (Liu et al., 2024b; Ma et al., 2024a; Xu et al., 2023; Wang et al., 2024b)
- Aim to keep task survey up-to-date through a companion website for contributions from task developers

**Scope of the Survey**:
- Focuses on tasks where agents interact with their environment
- Differs in modality of the environment, action spaces, observability, rewards, and evaluation metrics
- Properties greatly affect modeling choices for successful agents

**Methodology**:
- Provide examples to specify task scope in Section 2
- Discuss structuring dimensions for goal-oriented tasks in Section 4
- Offer findings and future directions in Section 5

## 2 Scope of this Survey

**Scope of this Survey**
- Comprises tasks for **goal-directed interactive agents**
- Agents receive explicit **goal specifications**, e.g., natural language instructions or questions
- Derive **goal conditions** from NL instructions, like "Checkmate the king" vs. "Win the game"
- Focuses on challenging tasks with multiple actions to achieve a goal
- Limits scope to single autonomous agent tasks without human intervention

**Extensions and Related Surveys**
- Interaction with humans: Lin et al., Huang et al. (discussed in Section 5)
- Collaboration between multiple agents: Zhou et al., Huang et al. (discussed in Section 5)
- Comprehensive surveys on LLM-based agents: Mialon et al., Qin et al., Xi et al., Gao et al., Wang et al., Cheng et al.
- Memory components for LLM-based agents: Zhang et al.
- Agents based on multi-modal foundation models: Xie et al.
- Multi-agent interaction paradigm: Guo et al., Sun et al., Zhang et al.
- Tools and APIs for augmenting LLMs: Wang et al. (detailed overview)
- Applications of LLM-based agents: Peng et al.

**Related Surveys on LLM-Based Agents:**
- Mialon et al. (2023), Qin et al. (2023a), Xi et al. (2023), Gao et al. (2023), Wang et al. (2023b), Cheng et al. (2024): single agent paradigm with a focus on modeling aspects and common applications
- Zhang et al. (2024b): memory components for LLM-based agents
- Xie et al. (2024a): multi-modal foundation models
- Guo et al. (2024a), Sun et al. (2024), and Zhang et al. (2024a): multi-agent interaction paradigm
- Wang et al. (2024d): tools for augmenting LLMs, including benchmarks for tool/API use
- Peng et al. (2024): overview of applications for LLM-based agents as part of a pipeline for holistic evaluation of LLMs.

## 3 Example Tasks

**Navigation & Object Manipulation Tasks:**
- Agents navigate and interact with physical objects in simulations of physical environments (2D or 3D)
- Examples: GRIDLU, MINIGRID, MINERL, MINEDOJO, ALFRED, EMBODIEDQA
- Based on various game engines and environments like Minecraft, text worlds, or AI2-Thor
- Agents required to perform tasks such as object arrangement, collection, or answering questions about rooms and objects.

**Text Worlds:**
- Environments represented via textual descriptions
- Examples: ALFWORLD, SCIENCEWORLD, JERICHO AGENTBENCH
- Based on the TextWorld engine, providing a partial description of the environment and events for agents to react to.

**PDDL Planning Problems:**
- Tasks specified in the Planning Do-main Description Language (PDDL)
- Small action and state spaces that are usually fully observable
- Examples: BLOCKSWORLD APBENCH, GOLDMINER APBENCH
- Current LLM-based agents either directly consume PDDL statements or verbalized domain descriptions.

### 3.2 Digital Assistance

**Digital Assistance: Interaction with Tools, APIs, GUIs, and Code**

**Interaction with Tools and APIs:**
- Digital assistants operate external software via Application Programming Interfaces (APIs) or tools
- Action spaces correspond to valid tool calls
- APIs can be called in isolation or embedded in code
- Examples: HOTPOTQA REACT, GSM 8KTOOLQA, GQA, IMGEDIT, GAIA, M&M'S, TOOLBENCH, RESTBENCH, and TOOLALPACA

**Interaction with Graphical User Interfaces (GUIs):**
- Digital assistants interact with websites or apps through graphical interfaces
- Action spaces comprise mouse and keyboard actions like click, type, press, swipe
- Observations: screenshots, HTML, accessibility trees, annotated screenshots
- Examples: MIND 2WEB, AITW, and OSWORLD

**Interaction with Code Interpreters:**
- Digital agents can directly interact with code interpreters like python
- Action space is the set of valid statements in a programming language
- Observations directly correspond to outputs of interpreter
- Example: SQL_DATABASE AGENTBENCH, APPWORLD.

## 4 Structuring the Task Landscape

**Task Formalization for Agent Performance Evaluation**

**Characteristics of Complex Tasks**:
- Agent's objective: come up with course of action to achieve goal in a given environment
- Interaction with environment: take actions, observe effects on environment
- Task instance defined as Partially Observable Markov Decision Process (POMDP)
  - **S**: set of states
  - **A**: set of admissible actions
  - **T**: state transition function
  - **O**: observation function
  - **Ω**: goal specification
- **G**: goal specification, varies by type and directness of expressing goal conditions
- Agent's objective: come up with sequence of actions from A to complete G by interacting with the environment

**Formal Definitions**:
- **Task instance**: (POMDP)<(S,A), T, O, Ω>, S0, G
- **Actions**: A set of admissible actions available in a given state
- **States**: Correspond to situations the agent can modify via its actions
- **Observations**: Agent receives information about current state through observations from set O

### 4.2 Goals

**Goals and Goal Specification**

**Goal specification**:
- Conveys information about the conditions for task completion
- Expressed as an instruction or question in natural language (NL)
- Can range from direct NL translations of goal states to less direct specifications
- Less direct goal specifications increase task difficulty as agent cannot directly reach goal state
- Require agents with mechanisms for task decomposition

**Two Goal Types**:
1. Reach a specific world state:
   - Goal specification maps to a set of goal states SG
   - Achieved if current state s is in SG
2. Answer a question:
   - Goal specification maps to a subset of actions AG
   - Achieved by selecting action a from AG, often to submit an answer

**Stopping Criteria**:
- Agents must perform final action indicating goal conditions were met (stop or answer action)
- Some environments recognize task as completed upon reaching goal state

### 4.3 World and Knowledge States

**World States vs. Knowledge States**

**Manipulating World States**:
- Tasks requiring agents to change a situation (e.g., manipulate objects, modify database states)
- Agent can transition to unsolvable states by executing irreversible world state changes
- Examples: GOLDMINER APBENCH (destroying gold), APPWORLD (deleting DB entries)

**Observability of World States**:
- **Fully observable**: Agent has perfect information about the current state
- **Partially observable**: Agent needs to gather more information through actions
- Rare in realistic tasks, mostly found in synthetic tasks with small/low-dimensional world states (e.g., GRIDLU, BLOCKSWORLD APBENCH)

**Manipulating Knowledge States**:
- Tasks requiring agents to retrieve or transform information without changing the world state
- Actions serve to acquire or transform information, which does not render any follow-up actions inadmissible.

**Environment-Based State Changes**:
- World state changes independent of agent actions (e.g., bird hatching from an egg, day-to-night progression)
- Adds complexity for the agent to maintain the world model and master the environment.

### 4.4 Actions

**Parameterized Action Spaces**
* Many tasks include parameterized action spaces
* Can be continuous, discrete, or too large to enumerate
* Large action spaces: cannot enumerate all possible actions
* Filtering top-k candidate actions at each time step
* Restricting action space based on goal specification

**Action Preconditions and Effects**
* Agent needs knowledge about transition function T
* Includes action preconditions (constraints for valid execution)
* Can learn through interaction, build into agent directly
* Some tasks provide full specifications of transition function
* Planning time effects: abstract information about functions' workings
* Execution time effects: effect observed at execution time
* Building all planning time effects infeasible for large action spaces
* Some tasks provide functionality to retrieve descriptions on demand.

### 4.5 Observations

**Observations in Task Execution**

**Definition of Observation**: Information exposed to an agent due to action execution.
- Includes immediate effects and feedback on execution failures or progress towards completion (intermediate rewards).

**Examples**:
* **Blocksworld**: Unstacking blocks, seeing changes in the environment.
* **ALFWorld**: Arriving at a location and observing items on shelves.
* **Toolbench Function**: Finding personal details from a dictionary output.
* **APPWorld**: Searching contacts for phone numbers with customized error messages.
* **Blocksworld vs. Inadmissible Actions**: Differences in observations upon executing admissible or inadmissible actions.
- Inadmissible actions: No effect on state or detailed feedback on reasons for failure.
* **Intermediate Rewards**: Feedback on progress towards task completion (manually annotated or automatically calculated).
* **Modality of Observation Space**: Representations in various modalities, affecting the type of models that can be used to process the information.
- Examples: Visual information vs NL descriptions, structured text outputs like JSON dictionaries.
- Difficulties in extracting relevant information from observations: HTML for webpage observations.
* **Active Research**: Identifying the best modality for representing observations and how it impacts agent architectures.

### 4.6 Task Evaluation

**Task Evaluation Methods for AI Agents**

**Reference-based evaluation of final answers:**
- Determines if agent successfully completed a task by comparing predicted answer to a reference answer: exact match, fuzzy match, or based on reference answer's rank in predicted ranking.
- Suitable for most QA tasks and goal states with objective evaluation criteria.
- Infeasible for creative or subjective tasks and non-controllable data sources.

**Reference-based evaluation of final states:**
- Evaluates agent's transitioned state based on reference goal state satisfaction: checks constraints or partial completion based on distance from the goal state.
- Applicable to tasks specifying objective goal conditions.

**Reference-based evaluation of action sequences:**
- Predicted sequences compared with human-annotated references: exact match or fuzzy match.
- Calculate precision and recall of predicted actions against set-based representation of reference sequence (Ma et al., 2024b).
- Conservative metric as it assumes a single correct sequence for satisfying goal conditions.

**Reference-free evaluation:**
- Agents' outputs evaluated by humans or LLMs instead of references.
- Milani et al. (2023) train classifier on human evaluator judgments to compare predicted trajectories.
- Recent work shows large variance in correlation between LLM judgments and human judgments (Bavaresco et al., 2024).

### 4.7 General Properties of the Environment

**General Properties of the Environment**

**Task Difficulty:**
- Xie et al. (2024b) define as human completion time
- Subjective annotations by authors or LLM useful for model analysis
- Objective measures: length of gold trajectory, size of action space, number of actions required to solve a task, number of objects to interact with

**Domain Specificity:**
- Impact on (mis-)alignment between task-relevant knowledge and pre-training data in LLMs
- Common knowledge vs. domain-specific knowledge
- More challenging for agents as specialized knowledge might not be stored in their parameters

**Data Availability:**
- Determines learning paradigms and additional steps required to apply them
- Major paradigms: online reinforcement learning, supervised learning, in-context learning
- Interactive environments release transition functions for a more comprehensive approach

**Task Generation:**
- Action space and transition function created manually
- Goal specification and manual annotation of corresponding conditions
- Some datasets provide problem generators to automatically generate tasks for training or evaluating agents.

## 5 Discussion and Future Directions

**Discussion and Future Directions**

**Importance of Agent-User Interaction:**
- Current tasks focus on user involvement only at goal specification
- User interaction required for realistic scenarios
- Struggles with current agent architectures (Lin et al., 2024)
- Combining goal-directed environment interaction and agent-user interaction needed
  - Insights from task-oriented dialogue and collaborative games

**Evaluation of Agent Behavior:**
- Several works introduce dedicated data splits for specific aspects:
  * Generalization to unseen actions (Qin et al., 2024; Li et al., 2023)
  * Compositional generalization (Furuta et al., 2024)
- Separate evaluation of intermediate steps
  * Agents' understanding of websites (Liu et al., 2024a)
  * Identifying user intents based on GUI interactions (Berkovitch et al., 2024)

**Formalizing Observations:**
- Develop frameworks for studying agent abilities and limitations
- Contribute to a better understanding of agent capabilities and weaknesses.

**Standardizing Environments:**
- Lack of standardized held-out splits and evaluation scripts in some benchmarks (Kapoor et al., 2024)
- Dependence on external tools or APIs hinders reproducibility
  * GAIA relies on GPT4 plugins, TOOLBENCH on web-based APIs
- Stable version of TOOLBENCH introduced by Guo et al. (2024b)
- Importance of rendering environments and evaluations more reproducible.

