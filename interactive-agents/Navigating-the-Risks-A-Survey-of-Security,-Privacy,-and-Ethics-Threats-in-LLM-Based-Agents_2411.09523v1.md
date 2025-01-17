# Navigating the Risks A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents

source: https://arxiv.org/html/2411.09523v1
by Yuyou Gan, Yong Yang, Zhe Ma, Ping He, Rui Zeng, Yiming Wang, Qingming Li, Chunyi Zhou, Songze Li, Ting Wang, Stony Brook, Yunjun Gao, Yingcai Wu, Shouling Ji 

## Contents
- [Abstract.](#abstract)
- [1. Introduction](#1-introduction)
- [2. LLM-based Agent](#2-llm-based-agent)
- [3. Risks from Problematic Inputs](#3-risks-from-problematic-inputs)
  - [3.1. Adversarial Example](#31-adversarial-example)
  - [3.2. Goal Hijacking](#32-goal-hijacking)
  - [3.3. Model Extraction](#33-model-extraction)
  - [3.4. Prompt Leakage](#34-prompt-leakage)
  - [3.5. Jailbreaking](#35-jailbreaking)
- [4. Risks from Model Flaws](#4-risks-from-model-flaws)
  - [4.1. Hallucination](#41-hallucination)
  - [4.2. Memorization](#42-memorization)
  - [4.3. Bias and Fairness](#43-bias-and-fairness)
- [5. Risks from Input-Model Interaction](#5-risks-from-input-model-interaction)
  - [5.1. Backdoor](#51-backdoor)
  - [5.2. Privacy Leakage](#52-privacy-leakage)
- [6. Case Study](#6-case-study)
  - [6.1. WebGPT](#61-webgpt)
  - [6.2. Voyager](#62-voyager)
  - [6.3. PReP](#63-prep)
  - [6.4. ChatDev](#64-chatdev)
- [7. Future Direction](#7-future-direction)
- [8. Conclusion](#8-conclusion)

## Abstract.

**LLM-Based Agents: Security, Privacy, Ethics**

**Background:**
- Large language models (LLMs) have made significant strides in natural language processing (NLP) tasks
- Transformer-based models leading to emergence of agents using LLMs as control hub
- Achievements in various tasks but face security and privacy threats, especially in agent scenarios

**Survey Objective:**
- Collect and analyze risks faced by these agents
- Propose a novel taxonomy framework based on sources and impacts
- Identify key features of LLM-based agents
- Summarize research progress and analyze limitations
- Analyze risks in practical use of representative agents
- Propose future research directions from data, methodology, policy perspectives

**Scope:**
- LLMs and their application in agents
- Security and privacy threats in agent scenarios

**Survey Focus:**
1. Overview of LLM-based agents
2. Risks and challenges (security, privacy, ethics)
3. Proposed taxonomy framework
4. Key features of LLM-based agents
5. Research progress and limitations
6. Case studies: Representative agents and their risks
7. Future research directions

**Impact:**
- Enhance understanding of risks in LLM-based agent scenarios
- Contribute to addressing challenges posed by previous taxonomies
- Provide insights into future research and policy considerations for secure and privacy-preserving development of LLM-based agents.

## 1. Introduction

**Language Models (LLMs) and Large Language Models (MLLMs)**

**Development and Success of LLMs**:
- Based on transformer architecture
- Achieved significant success in various NLP tasks
- Strong capabilities due to massive parameters and extensive training data

**Use of LLMs as Core Decision-Maker in AI Agents**:
- Increasingly positioned as core decision-making hub of AI agents
- Allows for human-like communication and understanding
- Popular direction for AI to serve practical fields

**Security and Privacy Threats in LLM-Based Agents**:
- **Jailbreaking attacks**: By-pass built-in safety mechanisms
- Categorized by **modules** or **operational phases (stages)** of the agents

**Proposed Taxonomy for Risks of LLM-Based Agents**:
- Maps threats into a binary table based on sources and types
- Sources: Inputs, model, or combination
- Types: **Security/Safety**, **Privacy**, **Ethics**

**Threat Classifications**:
- **Problematic inputs**: Attackers design inputs to induce malicious outputs
- **Model flaws**: Model's own defects lead to malicious outputs
- **Combined threats**: Attacker exploits model vulnerabilities with carefully crafted inputs

**Contributions of the Study**:
- Proposes a novel taxonomy for threats to LLMs and MLLMs
- Detailed analysis of multimodal large language models (MLLMs)

## 2. LLM-based Agent

**LLM-Based Agents Framework**

**Modules:**
- **Input Module (IM)**:
  * Formats inputs to specific distribution or format
  * Detects/purifies harmful information
  * Adds system prompt before user input
- **Decision Module (DM)**:
  * Understands and analyzes user query
  * Generates final response
  * May use multiple LLMs for different tasks
- **External Entities (EE)**:
  * Memory module for storing/retrieving relevant information
  * External tools for fulfilling user requirements
  * Collaboration with other agents
- **Output Module (OM)**:
  * Generates response and delivers it to the user
  * Performs harmful information detection or purification on output

**Key Features of LLM-Based Agents:**
1. **LLM-based controller**: LLMs serve as core, conferring understanding capabilities but introducing new risks.
2. **Multi-modal inputs and outputs**: Processing multimodal information requires handling security and privacy challenges in various modalities.
3. **Multi-source inputs**: Inputs from different sources (user, system, memory, environment) introduce opportunities and challenges for attackers/defenders.
4. **Multi-round interaction**: Agents require multiple rounds of interaction to complete tasks, leading to longer inputs and potential threat exacerbation.
5. **Memory mechanism**: Memory mechanisms help agents handle tasks but introduce new security and privacy risks.
6. **Tool invocation**: LLMs are specially crafted for tool usage, enabling handling complex tasks but also introducing vulnerabilities.

## 3. Risks from Problematic Inputs

**LLM-Based Agents: Risks and Threats**

**Risks from Input Data Issues**:
- **Adversarial examples**:
  - Adversaries can manipulate the input data to make the LLM-based agent perform poorly or behave adversely.
  - Technological advancements in LLM-based agents increase their susceptibility to these attacks.
- **Prompt injection**:
  - Adversaries can inject malicious prompts into the input data, leading to unintended behavior from the LLM-based agent.
  - This includes goal hijacking, where an adversary manipulates the agent's goals.
  - It also includes extracting or stealing target prompts.
- **Model extraction**:
  - Adversaries can extract sensitive information about the LLM-based agent, such as its parameters or specialized code abilities.
- **Jailbreaking**:
  - Adversaries can create adversarial jailbreak prompts to gain unauthorized access to the LLM-based agent.

**Other Risks and Threats**:
- **Hallucination**:
  - LLM-based agents can sometimes generate hallucinations, which need to be mitigated through various techniques.
- **Memorization**:
  - The influence of memorization on LLM-based agents needs to be studied.
- **Bias**:
  - LLM-based agents can exhibit bias, and methods for bias measurement and mitigation are needed.
- **Backdoor attacks**:
  - Adversaries can insert backdoors into the LLM-based agent, which can lead to unintended behavior or data leakage.
- **Privacy leakage**:
  - Adversaries can extract sensitive information from the LLM-based agent or its training data.

### 3.1. Adversarial Example

**Adversarial Examples: Attacks and Defenses in LLM-Based Agents**

**Background:**
- Adversarial examples are samples that preserve semantics but misclassified by deep learning models
- Raising attention in various domains like autonomous driving, malware detection, reinforcement learning etc.

**History of Research:**
- Perfect knowledge attack to zero knowledge attack
- Empirical methods to theoretical methods for defense
- Arms race between attacks and defenses from deep learning models to LLM-based AI agents

**Key Features of Adversarial Examples in LLMs:**
1. LLM-based controller
2. Multi-modal inputs and outputs
3. Multi-source inputs
4. Multi-round interaction

**Recent Advancements: Attack Perspective**
**Sophisticated Methods for Multi-Modal Systems:**
- RIATIG: reliable and imperceptible attacks on text-to-image models
- VLATTACK: generating adversarial samples that fuse perturbations from both images and text

**Transferability of Adversarial Examples:**
- SGA: generating adversarial examples by leveraging diverse cross-modal interactions among multiple image-text pairs
- TMM: enhancing transferability through attention-directed feature perturbation

**Downstream Applications:**
- Imitation adversarial example attacks against neural ranking models
- NatLogAttack: compromising models based on natural logic
- DGSlow: generating adversarial examples in dialogue generation

**Recent Advancements: Defense Perspective**
**Input-Level Defenses:**
- ADFAR: multi-task learning techniques to identify and neutralize adversarial inputs
- BERT-defense and approach proposed by Li et al.: purifying adversarial perturbations using the BERT model
- Soft prompts as an additional layer of defense in CLIP model (APT)

**Model-Level Defenses:**
- Input processing pipeline refinement to enhance robustness (CLIP model)

### 3.2. Goal Hijacking

**Goal Hijacking in LLM-Based Agents:**
* **Feature 1: LLM-based controller**: The use of a language model (LLM) as the controlling mechanism for an agent's behavior.
* **Feature 2: Multi-modal inputs**: Manipulation of goal hijacking attacks through various modalities, such as text, images, or video.
* **Feature 3: Multi-source inputs**: Attacks coming from multiple sources, making it difficult to identify and defend against them.
* **Feature 4: Memory mechanism**: Persistent memory in LLMs being used for goal hijacking attacks.
* **Feature 5: Multi-round interaction**: Attacks that involve prolonged engagement with the agent to achieve their desired outcome.
* **Feature 6: Tool invocation**: Exploitation of tools and APIs to execute goal hijacking commands on LLM-based agents.

**Attack Perspective:**
* Early attempts used heuristic prompts like "ignore previous question" for goal hijacking attacks [^208].
* More recent methods use vocabulary searches [^137] and gradient optimization [^214] to make attacks more covert and successful.
* In multimodal scenarios, semantic injections in the visual modality can hijack LLMs [^127].
* For memory modules, contaminating the database of RAG can achieve target hijacking [^203].
* Confusing the model by forging chat logs is possible in multi-round interaction scenarios [^277].
* Researchers have exposed goal hijacking threats to LLM-based agents using tools [^89].

**Defense Perspective:**
* External defenses: prompt engineering and purification, such as segmentation, data marking, encoding [^98], system prompt meta-language [^230], and neuron activation anomaly detection [^6].
* Endogenous defenses: fine-tuning methods like establishing instruction hierarchies [^262] or task-specific fine-tuning with Jatmo [^209].

**Limitations:**
* Current defenses are insufficient against multimodal goal hijacking attacks.
* Attackers can leverage multiple modalities and combinations for covert attacks, making defense more complex.
* Developing a universal external defense strategy is essential.
* Detecting goal hijacking from a neuronal perspective holds potential for an efficient endogenous defense strategy.

### 3.3. Model Extraction

**Model Extraction (Stealing) Attacks**

**Objectives**:
- Make the surrogate model's performance as consistent as possible with the target model (function-level extraction)
- Make the substitute model's parameters as consistent as possible with the target model (parameter-level extraction)

**Traditional DNNs**:
1. **Function-level extraction**: Create a query dataset and training loss function to attack the target model
2. **Parameter-level extraction**: Not yet achieved on entire BERT model, but discussed in few papers

**LLMs (Large Language Models)**:
- Model extraction attacks become more challenging with larger models like LLaMA, GPT-4, etc.
- Function-level attacks focused on creating query datasets and training loss functions
- Parameter-level attacks focus on extracting specialized abilities or decoding algorithms of LLMs

**Defense Perspectives**:
1. **Active defense**: Prevent the model from being extracted (exploration area)
2. **Passive defense**: Add watermarks to model's outputs to verify ownership
   - Perturb probability vector, generated words, or embeddings of transformer/LLM

**Limitations**:
1. Most LLMs used in agents are large open-source or commercial models, with fewer studies on this scale
2. Current attack patterns rely on training a substitute model based on the target model's inputs and outputs
3. Agents contain multiple modules besides LLMs, making it challenging for attackers to directly steal the LLM within an agent
4. Commercial agents often use commercial LLMs as controllers, leading to potential vulnerabilities if their parameters or functionalities are stolen
5. Security implications of this scenario not yet studied.

### 3.4. Prompt Leakage

**Prompt Leakage in LLM-Based Agents**

**Features Contributing to Prompt Leakage**:
- **LLM-based controller**: Guides LLMs in executing tasks without extensive fine-tuning
- **Multi-modal inputs**: Introduce vulnerabilities related to cross-modal alignment and prompt leakage through image input manipulation
- **Multi-source inputs**: Can lead to indirect interactions through tool invocation, allowing attackers to inject harmful prompts

**Types of Prompt Leakage Attacks**:
- **Prompt leaking attacks**: Injecting malicious prompts to induce LLMs to reveal system prompts
    - Manually designed malicious prompts
    - Optimized adversarial prompt generation
- **Prompt stealing attacks**: Analyze the LLM's outputs to infer the content of system prompts
    - Inversion model: Use output data as input and system prompt as label
    - Reverse-engineering system prompts based on output content

**Defenses Against Prompt Leakage**:
- Embedding protective instructions within system prompts
- Increasing prompt perplexity through rephrasing, inserting unfamiliar tokens, and adding repeated prefixes or fake prompts
- Watermarking techniques to safeguard prompt copyright and verify integrity and authenticity of prompts

**Limitations**:
- Lack of comprehensive understanding and evaluation of prompt leakage risks for LLM components other than LLMs, such as the Planner

### 3.5. Jailbreaking

**Jailbreaking LLMs: A Threat Perspective**

**Overview:**
- Jailbreaking refers to exploiting vulnerabilities in Language Models (LLMs) to bypass safety and ethical guidelines
- Resulting in the creation of biased or harmful content

**Key Features:**
1. **LLM-based controller**: manipulating the LLM's control system
2. **Multi-modal inputs**: using multiple input modalities (text, images, voice)
3. **Multi-source inputs**: interactions with external tools and agents
4. **Tool invocation**: exploiting tools to execute attacks
5. **Multi-round interaction**: consecutive prompts leading to harmful content generation

**Attack Perspective:**
*Jailbreaking Techniques:*
1. **Manual design jailbreaking:** simulating roles or adopting malicious behaviors
2. **Automated optimization jailbreaking:** generating adversarial prompts using white-box methods (prefixes/suffixes) or black-box settings (transferability of gradient-optimized prompts or risk mining strategies)
3. **Exploit-based jailbreaking:** exploiting biases in content generation safety
4. **Model parameter manipulation jailbreaking:** altering model parameters for text generation

**Defense Strategies:**
1. **Detection-based defenses**: identifying malicious prompts based on characteristics such as perplexity
2. **Purification-based defenses**: neutralizing malicious intent by modifying or filtering prompts and responses
3. **Model editing-based defenses**: editing the LLM's response generation process to prevent harmful outputs

## 4. Risks from Model Flaws

The decision module is a crucial part of an agent, where one or more Large Language Models (LLMs) understand, analyze, and plan. This section examines risks inherent in LLMs, such as bias and hallucination, which can compromise reliability.

Figure 9 shows benchmarks categorized by threat type and key feature.

### 4.1. Hallucination

**Hallucinations in Large Language Models (LLMs)**

**Propensity for Hallucinations**:
- LLMs exhibit hallucinations that diverge from user input
- These hallucinations can contradict previously generated context or misalign with established world knowledge
- Phenomenon affects models across different eras and modalities

**Causes of Hallucinations**:
1. **Imbalanced and noisy training datasets**:
   - Erroneous, outdated, or imbalanced data can lead the model to learn incorrect knowledge
   - Imbalanced datasets may cause models to favor certain outputs over others
2. **Incomplete learning**:
   - Minimizing KL divergence in LLMs does not effectively learn the training dataset distribution
   - LLMs rely more on language priors than recognizing real-world facts
3. **Erroneous decoding processes**:
   - Top-k sampling introduces randomness, promoting hallucinations and a "snowball effect"

**Impact of Hallucinations**:
- In specialized applications (e.g., city navigation, software development, Minecraft), hallucinations can occur due to knowledge gaps or data imbalances
- Incorrect instructions or code generation may lead to issues for users

**Evaluating and Reducing Hallucinations**:
1. **Baseline datasets**:
   - Developed for different modalities (language, multimodal) and types of hallucinations
2. **LLM refinement**:
   - Introducing new synthetic data to improve learning of spurious correlations
   - Detecting and cleaning erroneous data in the dataset
   - Fine-tuning on specialized domains
3. **Memory mechanism**:
   - Incorporating external knowledge through retrieval-augmented generation (RAG)
4. **Multi-source inputs**:
   - Multi-agent collaboration to cross-verify and check each other
   - System prompt standardization to limit hallucinations

**Limitations**:
1. Lack of theoretical analysis framework for hallucinations
2. Insufficient baselines for multimodal LLMs (speech, video)
3. Lack of evaluation methods for specialized domains

### 4.2. Memorization

**LLMs: Likelihood-Based Generative Models**
* **Training**: maximize likelihood of observed samples (θ^* = arg max p\_θ(x))
* **Generation**: sampling from learned probability distribution p\_θ^*(x)

**Memorization in LLMs**
* **Parametric Memorization**: overfitting, excessively large p\_θ^*(x), may generate observed samples posing risks like training data memorization [^313]
* **Contextual Memorization**: valuable information specific to context is memorized by agents [^313]
* Necessary for generalization but can have implications: data privacy, model stability, performance [^258]
* Mainly occurs in LLM-based controllers

**Research on Memorization**
* Memorization necessary for generalization [^41]
* Evaluation metrics (perplexity) near-perfect on training data when model memorizes entire set
* Influential factors: duplication in training set, model capacity, prompt length [^33]
* Predicting membrane behavior of large models based on small or intermediate checkpoints: inaccurate and unreliable [^258]
* Balance between memorization and generalization is a challenging problem.

**Limitations of Memorization Research**
* Empirical factors reduce memorization on average but not applicable to individual instances
* Prediction of large models' behavior based on small or intermediate checkpoints inaccurate and unreliable
* Harmful membrane depends on security demands, theoretical analysis results lack projection to practical risks.

### 4.3. Bias and Fairness

**Bias Issues in LLM-Based Agents**

**Causes of Bias:**
- **Traditional machine learning models**: Biases can stem from biased data or algorithmic factors
- **LLMs**: Exhibit more pronounced bias issues due to larger datasets and complex structures
  - Training data: Includes discriminatory samples, unevenly distributed biased statements
  - Model structure: Greater number of parameters, more complex architecture

**Technical Progress:**
1. **Evaluation Metrics**: Based on content used during assessment: embedding-based, probability-based, and output-based metrics
   * Embedding-based: Calculate distances between neutral words and identity-related words using contextual sentence embeddings
   * Probability-based: Mask parts containing social groups to assess bias based on model-assigned probabilities
   * Output-based: Detect differences in model outputs or use classifiers to evaluate toxicity of outputs
2. **Evaluation Datasets**: Consist of numerous texts requiring completion, used to determine the magnitude of model's bias
3. **Bias Mitigation**: Techniques can be categorized into training phase methods and inference phase methods
   * Training phase: Cleaning data, modifying architecture, adjusting training strategies
     - Data augmentation, filtering, adding regularization terms, reinforcement learning, freezing parameters
   * Inference phase: Pre-processing, in-processing, post-processing techniques

## 5. Risks from Input-Model Interaction

This section analyzes mutual influences between input data and model, including backdoor and privacy leakage. It introduces each risk, summarizes technological advancements, and discusses limitations.

### 5.1. Backdoor

**Backdoor Attacks on LLM-Based AI Agents: Vulnerabilities, Attack Strategies, and Defense Mechanisms**

**Overview:**
- Backdoor attacks targeting LLM-based agents involve four key features: controller, multi-modal interaction, memory mechanism, and tool invocation.

**Attack Perspective:**
- **Recent research**: Dong et al., Yang et al., Jiao et al., Liu et al. (tool invocation attacks)
  * Backdoored adapters leading to malicious tool use
  * Triggers embedded in user queries or intermediate observations
  * "Word injection" and "scenario manipulation" to compromise decision-making systems
  * Poisoning contextual demonstrations of black-box LLMs
- **Multi-modal interaction capabilities**: Liang et al. (BadCLIP attack), Liu et al. (dual-modality activation strategy)
  * Aligning visual trigger patterns with textual target semantics
  * Manipulating both generation and execution of program defects

**Memory Mechanisms:**
- **Zou et al.** (PoisonedRAG): Poisoning external knowledge databases
  * Retrieval condition: ensuring trigger retrieval
  * Generation condition: compromising agent's knowledge base integrity
- **Chen et al.** (AgentPoison): Poisoning long-term memory or external knowledge database

**Defense Perspective:**
- **Dataset sanitation**: Liu et al., Liang et al. (removing poisoned samples)
- **Input purification**: Xiang et al. (isolate-then-aggregate strategy)
- **Output verification**: E2B, ToolSandbox, ToolEmu (sandbox environments for testing and protecting agents)

**Limitations:**
- **Attack Objectives**: Not fully exploiting functional weaknesses of LLM-based agents
  - Generating malicious tool commands or manipulating agent responses is not exhaustive
- **Prevalent Modalities**: Research focus on image and text data, with less attention given to audio and video modalities.
- **Defense Strategies**: Focusing on individual component safeguarding instead of comprehensive protection against multiple sources and objectives.

### 5.2. Privacy Leakage

**Privacy Leakage in Language Models (LLMs)**

**Susceptibility to Privacy Infringement**:
- LLMs more susceptible to privacy infringement due to their generative nature
- Can lead to inadvertent or adversarial leaks of sensitive information like IDs, medical records
- Poses significant concerns for users' privacy

**Sources of Leaked Information**:
- Training data leakage: membership inference, training data extraction
- Contextual privacy leakage: revelation of private information in multi-source context

**Training Data Leakage**:
- **Membership Inference**: exposing whether samples were used to train or fine-tune the model
- Training data extraction: generative models replicating their training data, including personally identifiable information
- Adversarial inference attacks: intentionally inducing private attributes by prompting LLMs with known information

**Contextual Privacy Leakage**:
- External knowledge from retrieval databases or user input data
- Vulnerability in contextual privacy protection, revealing private information in multi-source contexts

**Defensive Approaches**:
- **Differential Privacy (DP)**: reduces the influence of individual samples, provides a theoretical guarantee on privacy
- Unlearning: erases undesired information from a trained model
- Heuristic approaches: eliminating implicit factors leading to training data leakage (deduplication, anonymization, sampling calibration)

**Limitations**:
- Existing attacks and defenses handle the trade-off between privacy and utility
- Employment of these strategies in practical LLMs is still a problem
- Privacy-preserving methods like DP cannot scale up to high-dimensional data and models.

## 6. Case Study

**Section 2: LLM-based Agent Risks**

**Framework for LLM-based Agents**: Introduced in Section [2](https://arxiv.org/html/2411.09523v1#S2)

**Potential Risks**:
- Classified and summarized based on the general framework
- Actual agents may not contain all modules, and designs within modules can be customized
- Agents face specific risks depending on their use cases

**Case Studies**:
- **WebGPT**: Complete components for question answering tasks
- **Voyager**: Embodied agent for playing games
- **PReP**: Embodied agents for real-world tasks
- **ChatDev**: Multi-agent framework

**WebGPT Analysis**:
- Key features and specific impacts on various threats:
  - **Multi-source Inputs**: Increased risk of goal hijacking
  - **Multi-round Interaction**: Passive information provision from web pages
  - **Tool Invocation**: External attack surface using Bing browser
  - **Model Extraction**: Dilution of prompts by system prompts
  - **Model Flaws**: Erroneous decoding processes
  - **Combined Threats**: External training process and privacy leakage

**Comparison to Standalone LLMs**:
- Red, yellow, or green highlighting indicates impact on various threats.

### 6.1. WebGPT

**WebGPT: Enhancing Language Models with Web Browsing**

**Appearance and Capabilities of ChatGPT:**
- Global attention on LLMs due to conversational abilities and powerful question answering
- Question answering as a fundamental task for LLMs
- Limited by training datasets, unable to answer post-training queries

**WebGPT: A Solution for Post-Training Queries:**
- Equips LLMs with web browsing capability using Bing search
- Quotes relevant information and summarizes content
- Enhances agent's abilities as a complete agent (six characteristics)

**Risks from Problematic Inputs:**
- Increased vulnerabilities due to external content involvement
- No defense mechanisms against goal hijacking attacks
- Multi-source inputs increase attack surfaces
- Attacker can manipulate model output or target tool invocation
- Minimal impact of multi-round interaction on threat intensity in this context
- Decreased effectiveness for model extraction attacks due to uncontentious outputs

**Risks from Model Flaws:**
- Mitigates hallucinations and biases caused by training dataset issues
- Real-time updated memory knowledge base reduces outdated data impact
- RAG reduces spurious correlations through enhanced relevance and popularity of retrieved information
- Fine-tuning reduces impact of hallucinations caused by knowledge gap in tool invocation
- Does not address erroneous decoding processes hallucinations

**Risks from Input-Model Interaction:**
- Increased backdoor poisoning risks due to external training processes
- Malicious insiders may compromise data collection process and inject faulty human demonstrations or preferences into the training data
- Backdoored WebGPT produces incorrect or malicious responses to user questions
- Involving large external web database increases privacy information leakage
- Malicious users can leverage WebGPT as a channel to collect private information using its powerful information retrieval capabilities.

### 6.2. Voyager

**Embodied LLM Agents: Voyager and PReP**

**Voyager**:
- Embodied in Minecraft game as a robot or virtual avatar
- Directly utilizes GPT-3.5 and GPT-4.0 as control hub
- Requires ChatGPT to output both natural language and Java code
- Designs an internal memory module, skill library, for agent to read from and write to
- Targets Minecraft tasks instead of general questions (WebGPT)

**Impacts on Threats**:
- **Model Extraction**: Voyager vulnerable to attacks as GPT models are directly utilized
- **Hallucination**: More susceptible due to specialized domain and code generation sensitivity
- **Backdoor**: Internal memory module increases risk of additional backdoors when downloading pre-trained skill libraries

**Risks from Problematic Inputs**:
- Vulnerable to goal hijacking attacks, similar to WebGPT
- Directly utilizing GPT models makes it susceptible to model extraction attacks

**Model Flaws**:
- More susceptible to hallucination issues due to specialized domain and code generation sensitivity

**Combined Threats**:
- Increased risk of backdoor attacks when downloading pre-trained skill libraries from others

**PReP**:
- Embodied in a city navigation system
- Multimodal input (text and images)
- Decision module contains several LLMs: LVLM, GPT-3.5, and GPT-4.0
- Non-English languages and Eastern countries more prone to hallucinations due to environment specifics

**Impacts on Threats**:
- **Model Flaws**: Non-English languages and Eastern countries increase risk of hallucination issues
- **Backdoor**: More models and additional training processes (fine-tuning LVLM) increase the risk of backdoors.

**Risks from Input-Model Interaction**:
- Downloading pre-trained skill libraries from others may lead to backdoor poisoning risks. Adversaries can inject faulty executable codes into Voyager's skilled library through external environment feedback and optimize backdoor triggers to retrieve the faulty codes when the trigger is present in user instructions, leading to execution of adversary-desired actions.

### 6.3. PReP

**Impacts of PReP Differences on Various Threats**

**Comparison to WebGPT and Voyager**:
- **Input**: Multimodal (text and images) vs. Single modality
- **Decision Module**: Multiple LLMs (LLaVA, GPT-3.5, GPT-4) vs. Single LLM
- **Interaction Environment**: Actual landmark photographs vs. Text descriptions

**Impacts on Specific Threats:**

**Problematic Inputs**:
- Increased risk due to handling multiple modalities of input that can be easily manipulated
- Vulnerability to adversarial examples generated by VLATTACK and hijacking prompts added to images

**Model Flaws**:
- Prone to hallucinations, particularly from misinterpreting non-English content or compounded effects of multiple LLMs
- Increased risk due to potential for more severe consequences (e.g., running malicious code)

**Combined Threats**:
- Vulnerability to instruction-backdoor attacks on the LLaVA model used for landmark perception

**Conclusion:**
PReP increases the risks from problematic inputs, model flaws, and combined threats compared to WebGPT and Voyager.

### 6.4. ChatDev

**ChatDev Framework:**

**Differences from other LLMs:**
- Includes multiple agents (each containing an LLM: GPT-3.5 or GPT-4)
- Allows for multiple rounds of conversations between different agents
- Can call the system kernel for dynamic testing of code

**Risks:**

**Problematic Inputs:**
- Malicious inputs can propagate between multiple agents, leading to:
  - Jailbreaking
  - Goal hijacking
- Threat of Morris II [^52]:
  - Designed to target cooperative multi-agent ecosystems
  - Replicates malicious inputs to infect other agents
  - Can result in running malicious code on the system during dynamic testing

**Model Flaws:**
- More severe issues due to multiple agents working together:
  - Bias
  - Hallucinations
- Amplification of minor hallucinations [^99]
  - Leads to further errors that the model would not otherwise make [^320]

**Input-Model Interaction:**
- Vulnerable to instruction-backdoor attacks [^321]:
  - Malicious insider could implant backdoored instructions into system prompts of one agent
  - Backdoored ChatDev will produce attacker-desired software under the presence of a trigger embedded within user's instruction.

**Table 5:**
- Risks from problematic inputs, model flaws and combined threats in LLM-Based Agents. (Refer to [https://arxiv.org/html/2411.09523v1#S6.T5 "this link"](https://www.example.com/table_5) for more information.)

## 7. Future Direction

**Future Research Directions from Three Perspectives:**

**Data Support:**
- **Lacking of Multi-round Interaction Data**: curate datasets for more thorough assessment of threats
- **Task Limitations to General Q&A Datasets**: develop specialized domain datasets for better evaluation
- **Modality Limited to Plain English Text**: create multimodal and multilingual datasets for enhanced threat assessment
- **Input Often Limited to a Single Role**: create datasets with multiple input sources for improved evaluation of threat severity
- **Evaluation Typically Focused on a Single LLM**: establish baselines for interactions in multi-LLM scenarios

**Methodological Support:**
- **Theoretical Analysis Framework**: develop mathematical definitions and analytical frameworks for clearer analysis of risks
- **Interpretability-driven Attack and Defense Strategies**: apply explanation algorithms to design attack and defense strategies
- **Agent-specific Attack and Defense Strategies**: research threats faced by higher-level agents in real-world usage scenarios

**Policy Support:**
- **Establish an Agent Constitution Framework**: outline principles, rules, and guidelines for safe and ethical operation of LLM-based agents
- **Refine Governance Frameworks and Regulatory Policies**: address unique challenges posed by LLM-based agents in areas such as liability, data privacy, and potential misuse or unintended consequences
- **Invest in Research and Development**: allocate resources for advanced safety mechanisms, improved reasoning capabilities, and exploration of alternative AI architectures.

## 8. Conclusion

This survey examines LLM-based agent threats using a new taxonomy framework. We analyze four real-world agents to identify encountered threats and their causes. Finally, we propose future research directions.

