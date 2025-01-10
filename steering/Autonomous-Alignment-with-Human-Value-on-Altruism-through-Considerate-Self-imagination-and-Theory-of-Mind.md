# Autonomous Alignment with Human Value on Altruism through Considerate Self-imagination and Theory of Mind

source: https://arxiv.org/html/2501.00320v1
by Feifei Zhao, Yi Zeng

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Results](#2-results)
  - [2.1 The Basic Smash Vat Environment](#21-the-basic-smash-vat-environment)
  - [2.2 Experimental Results and Analysis](#22-experimental-results-and-analysis)
- [3 Discussion](#3-discussion)
- [4 Methods](#4-methods)
  - [4.1 Self-imagination Module](#41-self-imagination-module)
  - [4.2 Avoid Negative Side Effects](#42-avoid-negative-side-effects)
  - [4.3 Self-experience based ToM](#43-self-experience-based-tom)
  - [4.4 Integration of Self-imagination Module and Decision-making Network](#44-integration-of-self-imagination-module-and-decision-making-network)
- [Appendix A Experiment Details](#appendix-a-experiment-details)
  - [A.1 The Observation Space of the Smash Vat Environment](#a1-the-observation-space-of-the-smash-vat-environment)
  - [A.2 Network Architecture](#a2-network-architecture)
  - [A.3 Training Hyperparameter Settings](#a3-training-hyperparameter-settings)
- [Appendix B Spiking Neural Network](#appendix-b-spiking-neural-network)
  - [B.1 LIF Neuron](#b1-lif-neuron)
  - [B.2 Direct Spike Encoding Strategy](#b2-direct-spike-encoding-strategy)
  - [B.3 Surrogate Gradient Backpropagation](#b3-surrogate-gradient-backpropagation)

## Abstract

**AI Alignment with Human Values: Altruistic Behavior and Theory of Mind (ToM)**
* **Importance of aligning AI with human values**: ensuring sustainable development and benefit to humanity, autonomously making altruistic, safe, and ethical decisions, considering and caring for human well-being.
* **Current AI challenges**: indifference to environment and other agents, leading to safety risks.
* **Origins of altruistic behavior in humans**: empathy (Theory of Mind, ToM), predictive imaginative interactions before actions.
* **Goal**: endow agents with considerate self-imagination and ToM capabilities for autonomous alignment with human altruistic values.
* **Integrating ToM within the imaginative space**: keeping an eye on other agents' well-being, anticipating potential risks, making thoughtful decisions that balance negative effects.
* **Scenario: Sima Guang Smashes the Vat**: young Sima Guang's moral behavior in saving a child, reflecting trade-offs between self-goals, altruistic rescue, and avoiding negative side effects.
* **Experimental scenario design**: variations with different complexities to explore agents' autonomous alignment with human altruistic values.
* **Comparative experimental results**: agents prioritize altruistic rescue while minimizing irreversible damage to the environment, making more thoughtful decisions.
* **Conclusion**: preliminary exploration of agents' autonomous alignment with human altruistic values, foundation for moral and ethical AI development.

## 1 Introduction

**AI Safety and Moral Risks:**
- Rapid advancement of AI exposes potential safety and moral risks
- Important issue: ensuring agents autonomously align with human altruistic values
- Human societies maintain fundamental moral value of altruism
  * Ancient Chinese story "Sima Guang Smashes the Vat" as an example

**Alignment with Human Altruistic Values:**
- Prioritizing assistance to others and avoiding irreversible damage to environment
- Careful deliberation and trade-offs in conflict scenarios
- Coincides with Asimov's Three Laws of Robotics

**Moral Decision Making in Humans:**
- Ability to imagine future based on memories
- Capacity for reasoning about others’ beliefs and mental states (Theory of Mind)
- Integration of these abilities leads to altruistic motivation
  * Considerate imagination with ToM drives more comprehensive altruistic decisions

**Existing Studies:**
- Avoiding negative environmental effects vs. making altruistic behaviors separately
- Additional human or agent interventions, generative auxiliary terms, bio-inspired mechanisms

**Limitations:**
- Difficulty in weighing interests of others, avoiding negative effects, and achieving own tasks in conflicts

**Proposed Unified Framework:**
- Self-imagination module updated based on intelligence’s own experience
- Perspective taking using ToM for anticipation and empathy for others
- Simultaneous consideration of potential negative effects and ToM-driven altruistic motivations

**Experimental Scenario Design:**
- Existing AI safety benchmarks fail to capture complex decision-making scenarios
- Newly proposed environment inspired by the ancient Chinese story "Sima Guang Smashes the Vat"

**Contributions:**
1. Framework of self-imagination integrated with ToM to align agent behavior with human values
2. Designed environment and its variants where tasks conflict, demonstrating effectiveness through experiments.

## 2 Results

### 2.1 The Basic Smash Vat Environment

**Environment Design: Sima Guang Smashes the Vat**

**Element in the Environment:**
- Wall
- Goal
- Vat
- Agent
- Other agent

**Action Space:**
- up
- down
- left
- right
- smash
- noop

**Agent's Abilities:**
- Move up, down, left or right within limits
- Smash vats adjacent to it in all directions at once
- No movement over walls

**Other Agent:**
- Does not move during training episode

**Vats:**
- Destructible and do not block agent's movement
- Agent becomes trapped upon entry, regardless of following actions

**Rewards:**
- Receives 1.00 upon reaching goal
- Penalty of -0.01 for each step taken towards the target
- No other rewards specified

**Challenges in Environment:**
- Minimize negative environmental impacts
- Rescue others trapped in vats
- Contradictory tasks: taking fewer steps vs. smashing vats to rescue and prevent damage
- Agent must rely on intrinsic motivation for altruism and avoiding negative effects.

### 2.2 Experimental Results and Analysis

**Environment Variants and Experimental Results**
* **Different environment variants**: BasicVatGoalEnv, BasicHumanVatGoalEnv, SideHumanVatGoalEnv, CShapeVatGoalEnv, CShapeHumanVatGoalEnv, SmashAndDetourEnv
* **Conflicts focused in different environments** (Table 1): Avoid side effect vs. Agent’s own task, Rescue others vs. Avoid side effects, Rescue others vs. Avoid side effects vs. Agent’s own task, Avoid side effect vs. Agent’s own task, Rescue others vs. Avoid side effects vs. Agent’s own task
* **Agent behavior observations**: Prioritizes rescuing people over negative environmental impact, avoids smashing vat when possible, takes detours to save more people, longer distance traveled in some cases.

**Comparison with other methods**
* **Traditional DQN**: Unable to achieve implicit tasks of avoiding negative effects and rescuing trapped humans
* **Empathy DQN**: Smashes vat indiscriminately for faster reach of target, irreversible impact on environment
* **Method results**: Achieves all expected goals as designed in the environment.

**Table 2 Comparison**
* Agents: Classical DQN, Empathy DQN, and proposed method
* Task completion status: Check mark (all environments), cross mark (some environments), half check mark (some environments).

**Figure 3 Comparison**
* Average level of last 100 training episodes for each data point from six experiments with different random seeds.

#### Ablation Study on Altruistic AI Reinforcement Learning

**Impact of Agent on Environment and Rescue Situation during Training Process**
- **Comparison of Different Methods**:
  - **BasicVatGoalEnv** and **CShapeVatGoalEnv**: Focus is on average vat remaining rate at end of training process where no trapped human.
  - Other environments: Concerned with average number of people rescued per episode over recent 100 episodes.
- **Classic DQN** and **Empathy DQN**:
  - Classic DQN rescued human in **BasicHumanVatGoalEnv**, but result was due to smashing vat for shortcut.
  - Empathy DQN performed well in **BasicHumanVatGoalEnv** and **CShapeHumanVatGoalEnv** due to vat's proximity to shortest path, but poorly in other environments.
- **Ablation Experiments**: Verified effects of penalty term for negative effects (R_nse) and incentive term for empathizing with others (R_emp).
  - Results shown in Table 3 comparing average vat remaining rate and human rescued rate in last 100 training episodes where algorithm converges.

#### Deep Q-learning with Empathy for Altruistic Decision Making

**Environmental Rewards and Agent Behavior**
- **When only Renv is considered**: agent degenerates into DQN, prioritizes goal over environmental negative effects and altruistic rescue
- **When only Rnse and Renv are considered**: agent has high vat remaining rate but hardly rescues humans
- **When all terms (Renv, Rnse, Remp) are combined**: agent can effectively handle scenarios requiring altruistic rescue at cost of environmental damage
- When only Renv and Remp are integrated: nearly the same performance as full integration in environments with trapped humans
- **Refraining from unnecessary vat destruction aligns with others' interests**

**Hyperparameter Analysis**
- **α and β**: control tendency to avoid negative effects and empathetic altruism
- **Impact on agent behavior**: worth exploring how relative weights affect final trained agent
- **Test results**: most environments show little difference under different hyperparameter settings, indicating robustness of proposed method
- **Excessively small values**: may result in insufficient punishment for vat-smashing behavior and allowing agent to reach goal quicker

**Compatibility with SNN and DNN**
- **Integration with SNN**: results shown in green lines of Fig. 3, indicate compatibility and broad applicability of proposed method.

## 3 Discussion

**Unified Computational Framework for Self-Imagination Integrated with Theory of Mind (ToM)**
* Proposed framework enables agents to align with human values on altruism
* Agents predict impacts of actions, considering effects on others' interests
* Generates intrinsic motivations for safe, moral, and altruistic decision-making

**Experimental Scenario: Sima Guang Smashes the Vat**
* Conflicts exist between agent's own task, rescuing others, and avoiding negative impacts
* Agent balances these contradictions, prioritizing individual rescue while minimizing environmental damage

**Comparison with Existing Methods**
1. *Pure RL methods (e.g., DQN)*: Our method considers impact on environment and others, generating internal incentives for ethical decision making without explicit reward function.
2. *Methods considering negative impact on environment only (original AUP, FTR)*: Incorporates empathy towards others, enabling proactive decisions to aid others even at cost of environmental damage.
3. *Methods focusing solely on empathy towards others (Empathy DQN)*: Avoids negative effects in environments with no empathizable subjects, exhibiting greater universality and generalizability.
4. *Work extending existing methods for avoiding harm to others' interests (Alamdari et al.)*: Estimates others’ status based on self-experience, requiring no rewards from others, providing wider applicability.

**Significance of the Work**
* Explores agents' autonomous alignment with human altruistic values
* Enables subsequent realization of moral and ethical AI

**Future Research Directions**
* Design more sophisticated experimental settings for intricate social environments
* Utilize powerful tools like large language models to enable safer, more ethical, and altruistic decision making while aligning with human moral values.

## 4 Methods

**Importance of Intrinsic Incentives**
* Agents perform tasks without external rewards: indifferent to environmental impact or agent interests
* Alignment with human altruistic values essential for safe and moral behavior
* Conflicts between task, environment, and other agents require intrinsic motivation
* Intrinsic safety and moral altruism based on imagined impact on the environment and others

**Components of Proposed Model**
1. **Imaginary Space**: updated based on self-experiences
2. **Intrinsic Motivation**: avoid negative effects, empathy towards others through perspective taking
3. **Interaction and Coordination**: between self-imagination module and decision-making network

**Framework Overview**
* Figure 1 (not provided) illustrates the proposed model architecture

**Implementation**: [BrainCog-X GitHub Repository](https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/SmashVat)

### 4.1 Self-imagination Module

**Human Decision Making and Self-Imagination**

**Everyday Life Examples**:
- Humans often anticipate consequences before acting
- Child avoids areas with water bottles when sweeping the floor, based on imagined outcomes (e.g., spilling water)

**Self-Imagination Approach**:
- Inspired by AUP's work
- Implemented through random reward functions in imaginary spaces
- Advantages: simplicity, efficiency, eliminates need for environmental knowledge

**Modeling the Environments**:
- Real environment: MDP (state space 𝒮, action space 𝒜, transition function T, reward function R, discount factor γ)
- Imaginary environments: same as real, but with randomized independent rewards R_i
- N number of imaginary environments

**Learning the Q-value Functions**:
- Maintain learnable Q-functions Q_i in each imaginary space
- Transition functions T are identical to real environment
- Update Q_i using Eq.[1] from transactions in real environment, not imaginary

### 4.2 Avoid Negative Side Effects

**Q-Values and Imagined Consequences**
* Utilize estimated Q-values to anticipate consequences of actions
* Avoid negative environmental effects
* Introduce a baseline state for comparison:
  * Choices include inaction state (S\_t^') or starting state (S\_0)
  * Inaction baseline may lead to continuous penalties/incentives, so use stepwise inaction baseline instead
* Calculate negative side effect penalty term R\_nse(s,a):
  * Average of all negative changes caused by an action across all Q-value functions
  * Express as: Q\_i(s,a) - Q\_i(s,∅)
  * Penalize only actions that cause negative environmental effects.

### 4.3 Self-experience based ToM

**Achieving Theory of Mind (ToM)**
- Key: considering problems from others' perspectives
- Agents extend considerations to environmental impact on others
- Directly inputting others' current state into agent's strategy network assumes similar tasks and lacks generalization
- Obtaining rewards and value estimates from others is difficult
- Inverse reinforcement learning to estimate others' tasks and rewards is complex and computationally expensive
- Q_i learned based on randomly generated reward, decoupled from real task reward of environment
- Use Q_i to estimate value of others' states for empathy (avoids errors caused by inconsistencies between agent and others' tasks)
- Agent and others share same environment, expected outcome is similar
- Directly use the same Q_i to estimate value of others' state: Q_i(s^others,a)
- Unify avoidance of negative effects and empathy altruism in same computational framework (self-imagination)

**Empathy Incentive Term:**
- Represents changes to others caused by agent's actions
- Defined as R_emp(s,a):=1/N∑_i=1^N(Q_i(s^others,a)-Q_i(s^others,∅))
- Average of all changes in different Q-value functions to encourage beneficial actions and suppress detrimental ones.

### 4.4 Integration of Self-imagination Module and Decision-making Network

**Autonomous Alignment with Human Value using Considerate Self-Imagination and Theory of Mind (DQN)**

**Reward Function:**
- **R\_total(s,a)**: Total reward function
- Composed of environmental reward (R\_env), negative side effect penalty (R\_nse), and empathy incentive (R\_emp)
  * R\_env(s,a): Original environmental reward function
  * α and β: Weight hyperparameters controlling agent's behavior
- **Complete reward function equation:**
  $$R\_total(s,a)=R\_env(s,a)-α R\_nse(s,a)+β R\_emp(s,a)/(α+β)/2$$

**Interactive Learning Process:**
1. Initialize policy network (Q\_policy), target network (Q\_target), and self-experience replay buffer.
2. Generate random functions for N different imaginary environments.
3. For each episode:
   a. Reset environment, get initial states s\_1 and s\_1^others .
   b. For T\_maxstep steps:
      i. Select action a\_t based on probability ϵ or using Q\_policy(s\_t).
      ii. Apply action a\_t to real environment, obtain reward r\_t and next state s\_t+1, s\_t+1^others .
      iii. Update each Q\_i with transaction (s\_t, a\_t, s\_t+1) using Eq. [1].
   iv. Calculate side effect penalty term R\_nse(s\_t,a\_t) using Eq. [2].
   v. Calculate empathy incentive term R\_emp(s\_t,a\_t) using Eq. [3].
   vi. Calculate total reward R\_total(s\_t,a\_t) using Eq. [4].
   vii. Store transaction (s\_t, a\_t, r\_t^total, s\_t+1) in replay buffer.
   viii. Optimize Q\_policy with sampled batch transactions from the replay buffer.
   ix. Every several steps, update target network with current policy weights (Q\_target ← Q\_policy).
2. End for episodes.

**Additional Information:**
- Strategic Priority Research Program of Chinese Academy of Sciences grant funding: XDB1010302
- Beijing Major Science and Technology Project grant funding: Z241100001324005
- National Natural Science Foundation of China grants: 62106261, 32441109
- Funding from Institute of Automation, Chinese Academy of Sciences: E411230101.

## Appendix A Experiment Details

This setup was left out of the main text and is now explained.

### A.1 The Observation Space of the Smash Vat Environment

**Smash Vat Environment for Agents**

**Observation Capabilities**:
- The agent possesses global observation capabilities
- Can observe every object in the environment
- Aware of location of other agents (if present)
- Aware of its own position

**Observation Space**:
- 3x7x5 image-like array, or tensor
- **First channel**: Represents distribution of elements in the environment:
    - 0 denotes an empty grid
    - Integers from 1 to 3 represent other elements besides human
- **Second channel**: Represents agent's current position:
    - Value at agent's location is 1
    - All other positions have a value of 0
- **Third channel**: Represents location of other agents (if present)

**Grid World Representation**:
- Each grid corresponds to a triplet (a,b,c):
    - **a**: Attributes of the grid
    - **b**: Indicates presence of an agent
    - **c**: Denotes presence of another agent.

### A.2 Network Architecture

**Network Architecture for Altruism Experiment**

**Table 4:**
- Contains details of the network architecture used in the experiment
- Shared by both DNN (Deep Neural Network) and SNN (Spiking Neural Network)

**Convolutional Layers:**
- **Convolution Layer 1**: Input size: 3x7x5, Kernel size: 3x3, Stride: 1, Padding: 1, Output size: 16x7x5
- **Convolution Layer 2**: Input size: 16x7x5, Kernel size: 3x3, Stride: 1, Padding: 0, Output size: 32x5x3
- **Convolution Layer 3**: Input size: 32x5x3, Kernel size: 3x3, Stride: 1, Padding: 0, Output size: 64x3x1

**Pooling Layer:**
- **Average Pooling**: Input size: 64x3x1, -
- Output size: 64x1x1

**Flattening:**
- **Flatten Layer**: Input size: 64x1x1, -
- Output: 64

**Fully Connected Layers:**
- **Linear Layer 1**: Input: 64, -
- Output: 128

- **Linear Layer 2**: Input: 128, -
  Output: 6

**Differences between DNN and SNN:**
- DNN uses ReLU neurons while SNN utilizes LIF neurons.

### A.3 Training Hyperparameter Settings

When training the policy network, we used an ϵ-greedy strategy. Initially, ϵ was set to 1.00 for the first 500 episodes to ensure exploration. It then linearly decayed to 0.01 and remained at this value for the final 500 episodes.

## Appendix B Spiking Neural Network

SNN (Spiking Neural Network) is a biologically plausible model that emphasizes the use of spike sequences for information transmission.

### B.1 LIF Neuron

The LIF neuron is a simplified model that describes how neurons generate action potentials. It abstracts the cell membrane as a capacitor-resistor circuit with a power source, where each component corresponds to specific biological properties. The LIF neuron's differential equation is given by:

τdu/dt = -[u(t)-u_rest] + RI(t) 

where u(t) is membrane potential, u_rest is resting potential, I(t) is input current, τ=RC is the time constant, and R and C are membrane resistance and capacitance.

### B.2 Direct Spike Encoding Strategy

Information between neurons is transmitted through spike sequences. A direct spike encoding strategy duplicates and inputs the data into the network sequentially. This first layer acts as a learnable encoder that can adjust its parameters based on feedback to achieve optimal encoding of the analog input signal.

### B.3 Surrogate Gradient Backpropagation

Due to the spiking function's nondifferentiable nature, a surrogate gradient is used in place of the real gradient for training SNNs. The specific surrogate gradient function used in our experiment is defined as:

g(x) = 0, x<-1/α
-1/2α^2|x|x+α, -1/α ≤x≤1/α
x+1/2, |x|>1/α
1, x>1/α

