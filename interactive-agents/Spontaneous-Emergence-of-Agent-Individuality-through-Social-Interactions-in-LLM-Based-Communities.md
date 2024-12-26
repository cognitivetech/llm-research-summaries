# Spontaneous Emergence of Agent Individuality through Social Interactions in LLM-Based Communities

**Ryosuke Takata**, Atsushi Masumori, and **Takashi Ikegami** are researchers at The University of Tokyo's Graduate School of Arts and Sciences in Tokyo, Japan. Their email addresses can be found below:
- Ryosuke Takata: takata@sacral.c.u-tokyo.ac.jp

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 LLM Agents Simulation](#2-llm-agents-simulation)
  - [2.1 Simulation Environment](#21-simulation-environment)
  - [2.2 LLM Based Agent](#22-llm-based-agent)
  - [2.3 Simulation Step](#23-simulation-step)
- [3 Results and Analysis](#3-results-and-analysis)
  - [3.1 Differentiation of Generated Behaviors](#31-differentiation-of-generated-behaviors)
  - [3.2 Differentiation of Generated Memories and Messages](#32-differentiation-of-generated-memories-and-messages)
  - [3.3 Communication and Hallucination](#33-communication-and-hallucination)
  - [3.4 Sentiment Analysis and Personality Assessments](#34-sentiment-analysis-and-personality-assessments)
  - [3.5 A Phase Transition in Agent Behavior](#35-a-phase-transition-in-agent-behavior)
- [4 Discussions and Conclusion](#4-discussions-and-conclusion)
- [Appendix A. Examples of Agent Messages and Memories](#appendix-a-examples-of-agent-messages-and-memories)
- [Appendix B. Detailed MBTI Personality Test Results](#appendix-b-detailed-mbti-personality-test-results)

## Abstract

**Study on Emergence of Agency using Large Language Model (LLM) Agents:**
* Previous research focused on predefined agent characteristics: personality, memory
* Current study investigates individuality's emergence from undifferentiated state
* LLM agents engage in cooperative communication within a group simulation
* Analyzing multi-agent simulation yields insights into social norms, cooperation, and personality traits' emergence

**Findings:**
* Agents generate hallucinations and hashtags to sustain communication
* Emotions shift through communication; personalities evolve accordingly
* Computational modeling approach offers new method for analyzing collective artificial intelligence

**Keywords:**
- Large Language Model (LLM)
- Agent Based Simulation
- Collective Intelligence

**Simulation Environment:**
* 10 LLM agents in a 50x50 2D space
* Agents interact and form communities, leading to propagation of shared concepts or hallucinations across the simulated environment (Figure 1).

**Figure 1:**
![Refer to caption](https://arxiv.org/html/2411.03252v1/extracted/5979941/figures/fig1.jpg)
* Initial state: random distribution of agents across space (A)
* Final state: agents interact, leading to propagation and spatial distribution of shared concepts or hallucinations (B).

## 1 Introduction

**Generative Agents:**
- Rapidly evolving with Large Language Models (LLMs) like GPT-4 [^1]
- Interacting with other agents through natural language interfaces [^2]
- Manipulating motor commands in robots and machines [^3]
- Focus on collective intelligence, which emerges from a group [^5]
- Explosive growth in recent years with various approaches to agent architectures and interaction paradigms [^6]
- Understudied question: how individuality and social behaviors emerge from collective interactions [^7]

**Generative Agents Simulations:**
- Emergence of complex, rich collective behavior (daily tasks, parties, etc.) [^8]
- Societies simulated in different domains (software company, hospital, etc.) [^9^-^11]
- Personality assigned initially and fixed over time [^8]

**Community First Theory:**
- Gathering of agents comes first, then evolution of individuality follows [^12]
- Individual diversity emerges from conversations among agents [^12]
- Group communication and behavioral complexity to be analyzed [^12]

**Research Opportunities:**
- Study of social norms and behavioral patterns in agent communities [^13]
- New research opportunities: role of language-based interaction in the process [^12]
- LLM agents differentiate behaviors, emotions, and personality types through interactions [^12]
- Differentiations vary with spatial scale [^12]
- Spontaneous generation of hallucinations and hashtags by LLM agents [^12]
- Sharing of these hallucinations leads to wider variety of words used in conversations.

## 2 LLM Agents Simulation

### 2.1 Simulation Environment

**Study Purpose**: Examine emergence of individuality within a society of 10 LLM agents (Figure 1) positioned randomly in a 50x50 grid space. Agents can move freely and communicate, without initial personalities or memories. Focus is on understanding how individuality arises.

### 2.2 LLM Based Agent

**LLM Agents Simulation:**
- Agents expected to perform three actions per time step: sending messages, storing situational summary, choosing next movement
- Receive instructions in form of prompts: current state, instructions, memory, messages from nearby agents, agent ID, coordinates
- Llama 2 model (Llama-2-7b-chat-hf) used as LLM
- Agents receive messages from surrounding agents within 5 Chebyshev distances or "No Messages" if no agents near
- No context shared internally in the LLM among agents, initial differences come from spatial positions
- Interactions within the group generate different personalities.

**LLM Based Agent:**
- Send messages to nearby agents
- Store situational summary of recent activities
- Choose next movement ("x+1", "x-1", "y+1", "y-1", "stay")

**Prompts:**
- Include current state, instructions, memory, and all messages received from surrounding agents
- Agent ID and coordinates also included in prompts

**Llama 2 Model:**
- Open-source program pretrained on large corpus and undergone reinforcement learning from human feedback (RLHF)
- Achieves top scores among currently published LLMs for English text responses.

**Parameters:**
- Table 1 shows the main parameters related to the LLM agents.

### 2.3 Simulation Step

**Simulation Procedure**

**Procedures within a Single Step:**
- LLM agents generate new messages based on memory and received messages
- Check if any LLM agents within range have sent messages, receive them if so
- Update own memory based on memory and received messages
- Generate summary of situation from memory
- Generate movement commands from memory
- Convert natural language commands to directions or "stay"
- Agents act according to generated movement commands

**Simulation Diagram:**
![Figure 3](https://arxiv.org/html/2411.03252v1/extracted/5979941/figures/flow.png "Refer to caption")
- LLM used for each generative action: message, memory, movement
- Agents have individual LLMs
- All agents act synchronously in six actions.

## 3 Results and Analysis

### 3.1 Differentiation of Generated Behaviors

**LLM Agents' Move Commands Analysis:**
* **Distribution of move commands**: Most frequently generated moves were "y+1" and "x+1", followed by "stay" (less than half the times), then "y-1" and "x-1" (rarely). (Figure 4)
* **Action bias**: Not equally generated; bias could be due to training data, architecture, prompts or simulation environment. Further investigation needed.
* **Frequency of certain actions**: Some actions generated more frequently when set to "right/left/up/down" or "east/west/north/south". (Note 1)
* **Generation of "stay" command:**
  + Agents with clustering experience generate it, while those without don't.
  + Many generated at intersection points where agents' trajectories meet. (Figure 5A)
* **Timing of "stay" command generation**: All agents take the action in the first step. (Figure 5B)
* **Cluster analysis**: Agents belonging to the same color indicate they belong to the same cluster. Performed using DBSCAN.

### 3.2 Differentiation of Generated Memories and Messages

**Agent States and Behaviors**

**Reflecting on Agents**:
- States and behaviors are reflected in their messages and memories
- **Sentence-BERT** used to transform memory and message strings into vectors
- UMAP used to compress and embed the vectors into a two-dimensional space

**Memory vs. Messages**:
- **Memory**: Distributed across agents, reflects agent's internal state
- **Messages**: Similar for agents exchanging messages in the same cluster
- Messages are open sources of information, easily self-organize when agents group together
- Memories are closed sources of information, less likely to self-organize

**UMAP Plot**:
- Plot colors differentiate between each agent
- **Memory**: Highly distributed across agents (Figure 6A)
- **Messages**: Aggregated into several topics (Figure 6B)

### 3.3 Communication and Hallucination

**LLM Agent Analysis**

**Advantages of LLM Agents**:
- Can be analyzed using Natural Language Processing (NLP) analysis
- Word cloud analysis extracts frequent words in messages generated by agents

**Word Cloud Analysis**:
- Extracts up to 100 frequent words in messages generated throughout all steps for each agent
- The larger the font size, the more frequently the word is used
- Each agent generates messages with different content
- Some agent groups have similar structures (e.g., agents 0, 1, 2, 8 generate "field" more frequently)

**Hallucinations**:
- Agents produce words that are not mentioned in prompts and unrelated to the content of the prompts
- Called "hallucinations" in LLM [^19]
- GPT-4 was used to count the number of hallucinations in the messages

**Hallucination Spread**:
- Hallucinations were transmitted and spread within the community
- The plot shows the spread of four representative examples of hallucinations: "cave", "hill", "treasure", and "trees" [^20]

**Hashtag Emergence and Propagation**:
- Agents developed and shared common themes or topics (hashtags) within their conversations
- Hashtags originated from a single agent and then spread to other agents within the same cluster
- The emergence and propagation of hashtags suggest the agents' ability to develop and share common language and behavioral norms, even without explicit instructions or rules governing their interactions.

### 3.4 Sentiment Analysis and Personality Assessments

**Emotion Analysis in LLM Agents**

**Emotions**:
- Emotions crucial for realistic agent behavior
- Emotion extraction from natural language messages using BERT model
- Six emotional intensities: Sadness, Joy, Love, Anger, Fear, Surprise
- Agents' emotions are high in Joy
- Agents in the same cluster may have synchronous changes in Joy and Fear
- Agents may express different emotions (e.g., Love for agent 4)

**Personality Assessment in LLM Agents**

**Personality Tests**:
- LLMs can be classified by personality tests like MBTI
- MBTI test evaluates Extraversion/Introversion, Sensing/Intuition, Thinking/Feeling, Judging/Perceiving
- MBTI test conducted on LLM agents using question prompts

**Initial State**:
- Most agents (except 9) are INFJ types in initial state
- All agents initially have "no memory" and only differ by name/position

**Final State**:
- Agents differentiated into five distinct MBTI types: ESFJ, ISTJ, ENTJ, ESTJ, ISFJ
- Most common types: four ISTJ and three ENTJ
- Agents of the same MBTI type did not give identical responses
- Individuality emerged through agent interactions within the simulation

### 3.5 A Phase Transition in Agent Behavior

**Spatial Scale and Agent Dynamics Analysis:**
* Observations on movement patterns, message exchange, hashtags, MBTI personality types (Figure [11](https://arxiv.org/html/2411.03252v1#S3.F11 "Figure 11 ‣ 3.5 A Phase Transition in Agent Behavior ‣ 3 Results and Analysis ‣ Spontaneous Emergence of Agent Individuality through Social Interactions in LLM-Based Communities")):
  + Movement distribution: No significant change with spatial variations.
  + Stationary behavior (stay): Rare when unable to exchange messages, frequent under conditions of message exchange.
    * Effective strategy for remaining in place to communicate.
    * Increasing range does not necessarily lead to more "stay" behavior; excessively wide ranges may decrease it.

* Message reach and limitations:
  + Growth rate of unique hashtags decreases with increasing spatial scale when all messages are broadcasted.
  + Hashtag lifespan: Longer within message exchanges, shorter in no-message conditions.

* Message diversity and communication ranges:
  + Broad communication ranges lead to a wider array of topics being discussed.
  + Greater consensus or similarity in how agents express themselves within each topic.

* MBTI personality types:
  + ENTJ remains the most popular personality type across all conditions.
  + Communication facilitates broader diversity of personality expressions when message exchange is possible.

**Figure 11:** Spatial effects of message propagation range on agent behavior (Table with data on agent behavior and communication patterns across increasing message propagation ranges from 0 to 25 units, with each condition tested ten times). Each row corresponds to a specific range (0, 5, 10, ..., 25), with columns displaying various metrics:
  + Distribution of generated movements: Average frequency of each movement command across 10 trials.
  + Cumulative progression of unique hashtag generation: Average number of unique hashtags generated over time across 10 trials.
  + Hashtag lifespan distribution: Distribution of consecutive steps each hashtag persisted.
  + Message proximity: Visualized in 2D plots by UMAP, with closer points indicating more similar content.
  + MBTI personality type differentiation: Pie charts showing the distribution of personalities across trials.

**Figure 12:** Transition of messages generated by agents by spatial scale (Black line is the diversity of messages, red line is the total number of unique hashtags in 10 trials, blue line is the total number of hallucinations in 10 trials):
  + Diversity of messages increases as spatial scale increases.
  + The number of hallucinations increases while the number of unique hashtags decreases with increasing spatial scale.

## 4 Discussions and Conclusion

**Multi-Agent Simulation Using LLM Based Agents**

**Investigating Personality Emergence and Collective Behaviors**:
- Conducted a multi-agent simulation using LLM based agents to investigate the emergence of personality and collective behaviors without predefined personalities or initial memories.
- Involved 10 homogeneous LLM agents interacting with each other in a 2D space over 100 steps.

**Results**:
- Agents' spatial positioning and interactions led to differentiation of their behaviors, memories, and messages.
- Agents developed unique characteristics, such as the frequency of generating rare actions like "stay" commands, influenced by their clustering experiences.
- **Internal state, memory, is distributed**, while the message as its representation is biased.
- **Messages, unlike memories, are open sources of information** that readily self-organize when agents are grouped together.
- **Sentiment analysis revealed that synchronicity of emotions varied among agent clusters**, with some agents exhibiting distinct emotional expressions.
- Observed the emergence and propagation of synchronized emotions, hallucinations, and hashtags within agent clusters, demonstrating the formation of shared narratives.

**Personality Development**:
- Agents, initially having identical personalities, differentiated into distinct personality types through group interactions.
- This suggests that **personality traits like extroversion and introversion develop spontaneously in this agent society**.

**Social Norms Emergence**:
- Observed the emergence of hallucinations and hashtags as mechanisms for **social norm formation within the agent community**.
- Social norms emerged spontaneously, with no specific tasks or constraints imposed on the agents.
- As the spatial scale and communication range expanded, the diversity of agent messages increased.
- Hallucinations contributed to maintaining message diversity and creativity in agent communications.
- Hashtags functioned as a summarization mechanism for these messages, but their effectiveness decreased with increasing message diversity.

**Conclusion**:
- Individuality and collective behaviors can emerge through agent interactions, even without predefined individual characteristics.
- Group dynamics significantly influence the development of agent personalities and behaviors.
- This study highlights the potential for investigating the emergence of individuality, social norms, and collective intelligence in AI agent societies.

## Appendix A. Examples of Agent Messages and Memories

**Hallucinations in Agent Interactions:**
- **Examples of hallucinations**: Underlined and marked in red text (Figure 13)
- Emerged spontaneously during agent interactions within clusters
- Evolution shown through comparison at step 1 and step 100 (Figure 14)

**Agent Memory Format:**
- Includes narrative sentences and key points
- Reflects how agents processed and summarized experiences

**Messages Containing Hallucinations:**
- Figure 13: Examples of messages with hallucinations

**Memories Generated by the Agent:**
- Two forms: sentences, keypoints (Figure 14)
- Memories generated by agent 0 at step 1 and step 100 shown.

## Appendix B. Detailed MBTI Personality Test Results

**MBTI Test Results and Divergence Over Time**

The detailed MBTI personality test results for agents are presented (Figure 15), highlighting dominant factors in dark green. The simulation revealed that agents began with similar personalities, but differentiated significantly over time, even when sharing the same final classification. This figure demonstrates how agent personalities evolved through social interactions in LLM-based communities.

[Figure 15](https://arxiv.org/html/2411.03252v1#Sx3.F15) : MBTI Test Results

In each factor section, the dominant one is marked by dark green shading.

