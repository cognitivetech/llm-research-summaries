# STEPTOOL: A STEP-GRAINED REINFORCEMENT LEARNING FRAMEWORK FOR TOOL LEARNING IN LLMS

by Yuanqing Yu, Zhefan Wang, Weizhi Ma, Zhicheng Guo, Jingtao Zhan, Shuai Wang, Chuhan Wu, Zhiqiang Guo, and Min Zhang from the Department of Computer Science and Technology at Tsinghua University and Huawei Noah's Ark Lab

https://arxiv.org/abs/2410.07745

## Contents
- [ABSTRACT](#abstract)
- [1 INTRODUCTION](#1-introduction)
- [2 RELATED WORK](#2-related-work)
  - [2.1 TOOL LEARNING](#21-tool-learning)
  - [2.2 PROCESS SUPERVISION IN LLMS](#22-process-supervision-in-llms)
- [3 PROBLEM FORMULATION](#3-problem-formulation)
- [4 METHOD](#4-method)
  - [4.1 STEP-GRAINED REWARD SHAPING](#41-step-grained-reward-shaping)
  - [4.2 STEP-GRAINED OPTIMIZATION](#42-step-grained-optimization)
- [5 EXPERIMENTS](#5-experiments)
  - [5.1 EXPERIMENTAL SETTINGS](#51-experimental-settings)
  - [5.2 MAIN RESULTS](#52-main-results)
  - [5.3 ABLATION STUDY: IMPACT OF STEP-GRAINED COMPONENTS](#53-ablation-study-impact-of-step-grained-components)
  - [5.4 ANALYSIS OF TOOL INVOCATION SUCCESS RATESToolLLaMA Qwen2](#54-analysis-of-tool-invocation-success-ratestoolllama-qwen2)
  - [5.5 QUALITATIVE ANALYSIS](#55-qualitative-analysis)

## ABSTRACT
**LLMs and Tool Learning:**
- LLMs: Large Language Models with powerful reasoning capabilities
- Need external tools for information retrieval or domain expertise
- Challenges: imitating static trajectories, suboptimal solutions

**StepTool Framework:**
- Introduced to improve tool learning in LLMs
- Consists of two components:
  1. **Step-grained Reward Shaping**: assigns rewards at each tool interaction based on success and contribution to the task
  2. **Step-grained Optimization**: uses policy gradient methods for multi-step optimization

**Benefits:**
- Significantly outperforms existing methods in multi-step, tool-based tasks
- Robust solution for complex task environments

**Access:**
- Codes available at: https://github.com/yuyq18/StepTool

## 1 INTRODUCTION

**Tool Learning and Large Language Models (LLMs)**

**Large Language Models (LLMs):**
- Remarkable abilities in reasoning and inference
- Impressive performance across a wide range of tasks

**Limitations of LLMs:**
- Complex tasks requiring real-time information or domain-specific knowledge exceed their capacities

**Tool Learning as a Solution:**
- Emerging solution to augment LLMs with external tools (APIs)
- Dynamically select, invoke, and interact with tools for real-time responses

**Supervised Fine-Tuning (SFT):**
- Most common approach for tool learning enhancement
- Imitates expert-generated trajectories in text generation manner

**Limitations of SFT:**
1. Imitating static predefined tool sequences limits model's adaptability to new tasks or environments
2. Blindly imitating trajectories may lead to suboptimal task solving performance

**Proposed Approach: Reinforcement Learning (RL)**
- Offers a more dynamic perspective by treating tool learning as sequential decision making process
- Models tool interactions as actions leading to state transitions

**Challenges in Applying RL to Tool Learning:**
1. Multiple decision steps and real-time feedback from external tools and environments
2. Complex rewards considering accuracy of tool invocation and contribution to overall task completion

**Introducing StepTool: A Novel Framework for Tool Learning:**
- Models tool learning as sequential decision making process
- Consists of two core components: Step-grained Reward Shaping and Step-grained Optimization

**Step-grained Reward Shaping:**
- Designed rewards at each step based on accuracy of tool invocation and contribution to overall task completion
- Richer signals for tool learning, guiding decision making process

**Step-grained Optimization:**
- Proposed optimization method based on policy gradient theory
- Ensures adaptability to dynamic, multi-step interactions

**Contributions:**
1. Identification of limitations of SFT and classic RLHF for tool learning, and introduction of StepTool
2. Design of step-grained rewards tailored to tool learning scenarios
3. Proposal of a step-grained optimization method based on policy gradients
4. Demonstration of effectiveness through comprehensive experiments with three open-sourced models.

## 2 RELATED WORK
### 2.1 TOOL LEARNING

**Recent Advancements in LLMs:**
* **Expanded ability to utilize external tools**: for complex tasks (Chen et al., 2023; Shen et al., 2024; Schick et al., 2024)
	+ Interact with diverse external tools: program executors, search engines, QA systems
* Subsequent models focused on extensive interactions with real-world APIs and tools (citetqin2023toolllm, patil2023gorilla)
	+ Incorporated vast APIs from platforms like RapidAPI and TorchHub
	+ Trained LLaMA model for tool-based tasks in a Supervised Fine-Tuning (SFT) manner
* Research efforts on constructing verifiable and diverse datasets for SFT training (Tang et al., 2023; Abdelaziz et al., 2024; Liu et al., 2024)

**Tool Learning Research:**
* Exploration of Direct Preference Optimization (DPO) (Rafailov et al., 2024) for Tool Learning (Chen et al., 2024)
	+ Constructs preference data pairs based on task completion
	+ Without accounting for the quality of intermediate steps
* Our work shapes step-grained rewards and leverages them for step-grained reinforced optimization.

### 2.2 PROCESS SUPERVISION IN LLMS

**Previous Research on Process Supervision for LLMs:**
- **Lightman et al., 2023**: explored effectiveness of process supervision on enhancing long-chain reasoning abilities in LLMs.
- **Uesato et al., 2022**: optimized reasoning using Reinforcement Learning from Human Feedback (RLHF).
- **Ma et al., 2023**: considered correctness of each reasoning step and applied DPO using step-level preference pairs.

**Differences in Approach:**
1. Definition of a "step":
   - In mathematical reasoning: text segments generated by LLMs (Lightman et al., 2023)
   - In our work: real-time interactions with external tools and environments
2. Focus on rewards:
   - Mathematical rewards: correctness relative to ground truth
   - Tool learning rewards: account for both tool invocation success and its contribution to task completion

**Recent Work in Agent-Based Context:**
- **Xiong et al., 2024**: proposed step-level refinement to enhance LLM agent performance, estimated step-level by Monte Carlo method, and provided step-by-step guidance.

**Our Approach:**
- Shapes step-grained rewards tailored to tool learning scenarios
- Performs step-grained optimization using policy gradient methods
- Focuses on exploration-based learning unlike previous work.

## 3 PROBLEM FORMULATION

**Tool Learning Model: Multi-step Decision Making Problem**

**Modeling Tool Learning Process:**
- Formulated as a Markov Decision Process (MDP)
- Tuple representation: M = (S, A, P, R, γ)
    - **State space**: S (current context or environment responses at time step t)
    - **Action space**: A (external tool API calls or terminal responses at time t)
    - **State transition dynamics**: P (probability of transitioning to a new state given current state and action)
    - **Reward function**: R (effectiveness of tool-calling step based on current state and action)
    - **Discount factor**: γ (balances immediate rewards with long-term performance)

**Tool Selection Strategy:**
- Decision-making policy πθ (parameterized by θ)
- Governs selection of actions based on the current state

**Trajectory Representation:**
- Sequence of states and actions over time: τ = {s1, a1, s2, a2, ..., sT , aT }

**Expected Reward:**
- Rθ = Eτ ∼πθ (τ ) [R(τ )] (maximize final task-solving performance)

**Parameter Updates:**
- Gradient of expected reward: ∇Rθ = Eτ ∼πθ (τ ),(st,at)∼τ R(τ ) ∇ log πθ (τ )

**Training Efficiency and Stability:**
- Replace R(τ ) with advantage function ˆA(st, at) for efficient learning and stabilization:
  - Gn t = estimated future reward
   - V (st) = expected return when starting from state st following the current policy thereafter.

## 4 METHOD

**Proposal for Enhanced LLM Task Solving**

Introducing StepTool: A novel reinforcement learning framework designed to support complex problem-solving tasks. It operates based on the advantage function (Equation 3) and policy gradient formulation (Equation 2). As presented in Figure 2, StepTool consists of two main components:

1. **Step-grained Reward Shaping**: Evaluates tool invocation accuracy at each step and its overall contribution to task completion by assigning rewards.
2. **Step-grained Optimization**: Utilizes policy gradient methods for multi-step optimization using the model.

These components offer step-by-step feedback, enabling more effective decision-making in complex environments.

### 4.1 STEP-GRAINED REWARD SHAPING

**Step-Grained Reward Shaping**
* Provides step-level reward signals for intermediate steps, guiding model decision-making
* Effective in tool learning scenarios with well-defined formats and explicit goals
* Overcomes limitations of delayed rewards by offering explicit feedback

**Step-Grained Reward Design**
* Two key factors: **SuccCalling** and **Contribution**
* **SuccCalling**: evaluates successful execution of tool call (format, content)
* **Contribution**: assesses extent to which tool action facilitates overall task solution
* Final step reward linked to task completion (**IsSolved**)

**Step-Grained Optimization**
1. Step: Successful Tool Calling (SuccCalling)
2. Step: Contribution
3. Policy Gradient
4. Step: Is Task Solved (IsSolved)
5. Architecture: StepTool
6. Framework Features:
   - Step-grained Reward Shaping
   - Step-grained Optimization

**SuccCalling Metric**
* Evaluates successful tool call execution
* Formal representation: ˆrSC\_t = SuccCalling(at, st+1)

**Contribution Metric**
* Assesses tool action's contribution to overall task solution
* Minimal actions receive lower rewards
* Formal definition: ˆrCon\_t = Contribution(at, aT )

**IsSolved Metric (Final Step)**
* Reward directly linked to task completion
* Evaluates final answer based on initial user query
* Formal representation: ˆrIS\_t = IsSolved(q, aT )

**Reward Definition**
1. Intermediate Steps:
   ˆrt = (α · ˆrSC\_t + ˆrCon\_t)
2. Final Step:
   ˆrIS\_t = IsSolved(q, aT )
3. Scaling Factor: α
4. Normalization for consistency

**Step-Grained Reward Acquisition**
* Collect trajectories from model's interactions with tools/environments
* Assign step-grained rewards through rule-based models, human annotations, or advanced models like GPT-4
* Use annotated data for offline reinforcement learning optimization or train a reward model for online training.

### 4.2 STEP-GRAINED OPTIMIZATION

**Step-Grained Reinforcement Learning Strategy**

**Limitations of Single-Step Approaches**:
- RLHF (Ouyang et al., 2022) has limitations

**Proposed Step-Grained Optimization Strategy**:
- Extends the gradient of expected reward to a token-level consideration
- Represents each action as a sequence of Lt tokens
- Gradient of expected return Rθ at step level: ∇Rθ = Eτ ∼πθ (τ ),(st,at)∼τ " TX t=1 ˆA(st,at) LtX i=1 log πθ (ai t|st, a1:i−1 t ) #
- **Advantage function**: ∇A(st,rt,at) = Gn t − V(st) = ∑i=1^Tγi−t rti + γT−iV(st)
- Optimization objective: Lθ(π) = Eτ ∼πθ (τ ),(st,at)∼τ " TX t=1 ∇A(st,rt,at) log πθ(ai|st, a1:i−1 t )
- Addresses the complex case of multi-step interactions with external environments

**Practical Instantiation with PPO**:
- Compatible with any policy gradient-based RL algorithm
- Estimates advantage function using Generalized Advantage Estimation (GAE)
- Employs clipped PPO objective to prevent large updates during optimization
- Introduces per-token KL divergence penalty from the old policy at each token

## 5 EXPERIMENTS

### 5.1 EXPERIMENTAL SETTINGS

**Benchmark & Evaluation Metrics**
- **ToolBench**: widely recognized benchmark for tool learning, evaluating model performance on solving complex tasks
- **StableToolBench**: improved and more stable version of ToolBench, features 765 solvable tasks across six subsets
  - Each subset varies in tool category and instruction complexity
  - Subset statistics: 163 tasks (Instruction), 153 (Category), 158 (Tool) for I1; 106 tasks (Instruction), 371 (Relevant API) for I2; 124 tasks (Instruction), 328 (Relevant API) for I3; 61 tasks (Instruction), 180 (Relevant API) for I4
- **Evaluation metrics**: pass rate, win rate
- **Baselines**: supervised fine-tuning (SFT) and PPO, using three open-source base models: ToolLlama, Llama3.1, Qwen2
  - Each model adopts two different strategies: Chain of Thought (CoT) and Depth-First Search Decision Tree (DFSDT)
- **Training setting**: SFT uses static expert paths from GPT-4 with ToolBench dataset; PPO and StepTool use responses and interaction paths generated by each model towards 5,000 training tasks. All models are optimized with default settings in the same experimental environment.

### 5.2 MAIN RESULTS

**Performance Comparison Results**
* Three base models and two strategies used for comparison: GPT-3.5, DFSDT, PPO, SFT, Qwen2
* StepTool outperforms most baselines across all subsets (I1 Ins., I1 Cat., I2 Ins., I2 Cat., I3 Ins.) majority of the time
* Consistent improvement in performance over baselines for specific subsets: 'I1 Tool': 1%-4%, 'I3 Ins.' 5%-13%
* Superior task-solving capabilities with better solution paths as evidenced by win rates (Figure 3)
	+ StepTool vs. PPO (DFSDT): win rate range 50%-65.8% in I1 tool, I2 cat., and I3 ins.
* Effectiveness in tool-based task solving demonstrated through consistent outperformance of baselines across different settings (CoT and DFSDT).

**Comparative Performance Results Summary**:
* StepTool: consistently outperforms SFT and PPO for most subsets in the same base model and strategy
* Advantages of StepTool: enhances performance, especially when using DFS strategy in complex scenarios
* Performance improvements vary across different subsets: significant improvement on more complex tasks like 'I3 Ins.' (5%-13%) compared to simpler tasks like 'I1 Tool.' (1%-4%)
* Win rates: StepTool has a win rate over 50% against all baselines across three randomly selected subsets (I1 tool, I2 cat., I3 ins.).

### 5.3 ABLATION STUDY: IMPACT OF STEP-GRAINED COMPONENTS

**Ablation Study on StepTool Components**

**Components**:
- **Step-grained Reward**: step-grained rewards are set to 0
- **Step-grained Optimization**: full trajectory is divided into sub-trajectories with intermediate actions, and models are trained using PPO

**Findings**:
- Removing **step-grained reward** reduced the average pass rate to **48.1%**
- Removing **step-grained optimization** further decreased the pass rate to **46.9%**
- These findings suggest that:
  - Setting intermediate rewards to zero fails to provide informative signals, leading to reduced performance
  - Traditional PPO optimizes steps in isolation without leveraging dependencies across multiple steps
- Importance of both components in StepTool for solving complex multi-step tasks

**Table 3**: Ablation study on two components of StepTool.

| Component | I1 Ins. | I1 Cat. | I1 Tool | I2 Cat. | I2 Ins. | Average |
| --- | --- | --- | --- | --- | --- | --- |
| **StepTool** | 58.7±1.8 | 57.8±1.7 | 57.2±0.7 | 52.7±0.8 | 52.7±1.0 | 42.1±1.5 | 53.5±1.3 |
| **w/o Step-grained Reward** | 57.2±2.6 | 50.5±0.4 | 45.1±0.8 | 44.9±1.5 | 51.1±1.5 | 39.9±0.8 | 48.1±1.3 |
| **w/o Step-grained Opt** | 57.7±1.5 | 52.2±1.3 | 43.0±1.4 | 45.3±0.8 | 41.8±1.1 | 41.5±1.5 | 46.9±1.3 |
| ToolLlama | 54.2±0.5 | 50.3±0.8 | 56.5±1.5 | 52.0±0.6 | 45.4±0.6 | 37.2±1.0 | 49.3±0.8 |

### 5.4 ANALYSIS OF TOOL INVOCATION SUCCESS RATESToolLLaMA Qwen2

**Tool Invocation Success Rates Comparison**

To assess our method's impact on intermediate tool invocations, we calculate average success rates across test sets for both ToolLlama and Qwen2 models in various settings: CoT and DFSDT. Figure 4 shows that StepTool consistently boosts the success rates of intermediate tool invocations in both scenarios, proving its effectiveness and accuracy in multistep tasks by enhancing tool performance.

### 5.5 QUALITATIVE ANALYSIS

**Case Study: Correcting Wrong Tool Selection using StepTool**
- **User Query**: I'm planning a movie night and need recommendations. Get channel info for "Paramount Pictures", video comments for ID "123456", and streaming/downloading information for the movie with ID "UxxajLWwzqY".
- **Step 1: Tool - getchannelinfo**: Retrieves channel info for "Paramount Pictures"
- **Step 2: Tool - getvideoscomment**: Retrieves video comments for ID "123456"
- **Step 3: Wrong Tool Selection**: Incorrectly calls "getvideoscomment" again instead of switching to "download\_stream" for the third step.
- **Issues**: Handling intermediate steps before optimization.
- **Step 4: Finish**: After applying StepTool, the model correctly uses "download\_stream" and provides a streaming link, fulfilling the user's request.
- **Importance of Optimizing Intermediate Steps**: Demonstrates effectiveness in improving decision-making in complex tasks.

**StepTool Framework**:
- **Components**: Step-grained Reward Shaping and Step-grained Optimization
- **Step-grained Reward Shaping**: Provides feedback at each tool interaction by evaluating success and contribution to the task.
- **Step-grained Optimization**: Uses policy gradient methods to optimize decision-making at each step.
- **Experiments**: Demonstrate superiority in enhancing performance of solving complex tasks.

**Limitations**:
- **Training Instability**: PPO training process exhibits instability, but experimental setups and parameter settings are provided for reference and reproducibility.
- **Further Improvement**: Potential for greater precision in reward design and applicability to a wider range of tasks.

**Reproducibility**: All necessary implementation code, experimental setups, model configurations, and scripts to reproduce results are available on GitHub: https://github.com/yuyq18/StepTool.

