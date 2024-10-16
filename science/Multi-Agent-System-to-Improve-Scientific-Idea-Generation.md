# Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation

Haoyang Su, Renqi Chen, Shixiang Tang 

https://open-sciencelab.github.io/Social_Science/

https://arxiv.org/abs/2410.09403v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 AI for Scientific Discovery](#21-ai-for-scientific-discovery)
  - [2.2 Multi-agent Systems in Team Collaboration](#22-multi-agent-systems-in-team-collaboration)
- [3 The Virtual Scientists](#3-the-virtual-scientists)
  - [3.1 The Scientific Research Ecosystem](#31-the-scientific-research-ecosystem)
  - [3.2 The Multi-agent System for Scientific Collaboration](#32-the-multi-agent-system-for-scientific-collaboration)
- [4 Empirical Study](#4-empirical-study)
  - [4.1 Experimental Settings](#41-experimental-settings)
  - [4.2 Comparisons with AI scientist](#42-comparisons-with-ai-scientist)
  - [4.3 Exploring Science of Science: The Impact of Team Dynamics on Novelty](#43-exploring-science-of-science-the-impact-of-team-dynamics-on-novelty)
  - [4.4 Ablation Study](#44-ablation-study)
- [5 Conclusion](#5-conclusion)
- [Appendix A Data and Code Availability](#appendix-a-data-and-code-availability)
  - [A.1 Code](#a1-code)
  - [A.2 Data](#a2-data)
- [Appendix B Limitations and Future Work](#appendix-b-limitations-and-future-work)
- [Appendix C Ethics Statement](#appendix-c-ethics-statement)
- [Appendix D Effect of the Potential Data Leakage](#appendix-d-effect-of-the-potential-data-leakage)
- [Appendix E More Details of Methods](#appendix-e-more-details-of-methods)
  - [E.1 Self-review](#e1-self-review)
- [Appendix F More Experiments](#appendix-f-more-experiments)
- [Appendix G Prompts](#appendix-g-prompts)
  - [G.1 Scientist Definition](#g1-scientist-definition)
  - [G.2 Collaboration Selection](#g2-collaboration-selection)
  - [G.3 Topic Discussion](#g3-topic-discussion)
  - [G.4 Idea Generation](#g4-idea-generation)
  - [G.5 Novelty Assessment](#g5-novelty-assessment)
  - [G.6 Abstract Generation](#g6-abstract-generation)
  - [G.7 LLM Review](#g7-llm-review)
- [Appendix H Example Scenarios](#appendix-h-example-scenarios)
  - [H.1 Collaboration Selection](#h1-collaboration-selection)
  - [H.2 Topic Discussion](#h2-topic-discussion)
  - [H.3 Idea Generation](#h3-idea-generation)
  - [H.4 Novelty Assessment](#h4-novelty-assessment)
  - [H.5 Abstract Generation](#h5-abstract-generation)

## Abstract

**AI Advancements in Scientific Research**
* Rapid advancement of scientific progress requires innovative tools for accelerated discovery
* Recent AI methods, specifically large language models (LLMs), show promise in tasks like hypothesis generation and experimental design
* However, these methods lack the collaborative nature inherent in real-world scientific practices

**Proposal: Virtual Scientists (VirSci)**
* Multi-agent system designed to mimic teamwork in scientific research
* Organizes a team of agents to collaboratively generate, evaluate, and refine research ideas

**Evaluation:**
* Demonstrates outperformance of state-of-the-art method in producing novel and impactful scientific ideas
* Aligns with insights from the Science of Science field

**Findings:**
* Integrating collaborative agents can lead to more innovative scientific outputs
* Robust system for autonomous scientific discovery.

## 1 Introduction

**Automatic Scientific Discovery using Large Language Models (LLMs)**

**Background:**
- Rapid scientific progress necessitates more efficient tools for exploring new concepts and tackling complex challenges
- Automatic scientific discovery has emerged as a promising solution to expedite innovation
- Artificial intelligence (AI) has the potential to revolutionize research by automating key steps in the scientific process

**VirSci: An LLM-based Multi-agent System for Scientific Idea Generation:**
- Proposed system consists of five key steps: Collaborator Selection, Topic Discussion, Idea Generation, Novelty Assessment, and Abstract Generation
- Constructs a knowledge bank of scientists' backgrounds and develops digital twin agents using Retrieval-Augmented Generation (RAG) framework
- Lead agent identifies collaborators based on expertise and research interests, aligning with real-world cooperation patterns
- Team discussion mechanism enhances quality of output by engaging in iterative inter- and intra-refinement dialogues
- Generates comprehensive abstract representing proposed ideas

**Evaluation:**
- Introduces a benchmark for measuring novelty of ideas from three perspectives: dissimilarity to past papers, alignment with contemporary research trends, potential influence on contemporary research
- Comparison against past and contemporary paper databases ensures generated ideas are innovative while aligning with emerging scientific directions

**Findings:**
- VirSci outperforms single agent executive in producing novel scientific ideas, with an average gain of 13.8% and 44.1% in alignment level and potential impact on contemporary research, respectively
- Emergent social behaviors among agents align with prior studies in Science of Science, suggesting potential for further exploration of mechanisms in research collaboration using multi-agent simulations.

**Contributions:**
- Proposes the first multi-agent system for scientific collaborations from team organization to novel scientific idea generation
- Conducts extensive evaluations investigating VirSci's team settings and generated ideas, demonstrating that multi-agent collaboration can improve quality of outcomes, surpassing SOTA single agent method
- Aligns simulation results with important findings in Science of Science.

## 2 Related Work

### 2.1 AI for Scientific Discovery

**AI's Impact on Scientific Discovery**
- **Advancements through AI**: Enhances research processes, identifies complex molecular and protein structures (Vignac et al., [2022](https://arxiv.org/html/2410.09403v1#bib.bib49); Abramson et al., [2024](https://arxiv.org/html/2410.09403v1#bib.bib1))
  - Reduces time required for experimental iterations
- Wide application across diverse fields: Chemistry (Liu et al., [2023a](https://arxiv.org/html/2410.09403v1#bib.bib28); Meteorology (Bi et al., [2023](https://arxiv.org/html/2410.09403v1#bib.bib5)); Medicine (Rajpurkar et al., [2022](https://arxiv.org/html/2410.09403v1#bib.bib39))
- AI methodologies collaborate in various scientific pipeline stages: Hypothesis generation, experimental design, data acquisition, analysis (Zheng et al., [2023](https://arxiv.org/html/2410.09403v1#bib.bib62); Wang et al., [2023](https://arxiv.org/html/2410.09403v1#bib.bib50); Miret & Krishnan, [2024](https://arxiv.org/html/2410.09403v1#bib.bib32); Wysocki et al., [2024](https://arxiv.org/html/2410.09403v1#bib.bib55); Lu et al., [2024](https://arxiv.org/html/2410.09403v1#bib.bib30))
- Limitations: Lack collaborative nature inherent in real-world research

**VirSci**: First to harness LLM-based multi-agent system for autonomous scientific discovery

### 2.2 Multi-agent Systems in Team Collaboration

**Multi-Agent Systems for Team Collaboration**

**Traditional Multi-Agent Systems**:
- Involve semi-autonomous agents coordinating through explicit protocols and structured messages
- Achieve common goals
- Examples: [Dunin-Keplicz & Verbrugge (2011), Bakliwal et al. (2018)]

**Advancements with LLMs**:
- Enables agents to utilize natural language for communication and collaboration
- Fosters intuitive and flexible interaction model
- Superior performance in various domains: programming, game playing, complex reasoning tasks
  - Examples: [Liu et al. (2023), Wang et al. (2024), Light et al. (2023), Du et al. (2024)]

**Implementing LLM-based Multi-Agent Systems**:
- Promotes de novo scientific ideas through collaborative efforts

## 3 The Virtual Scientists

**Figure 2 Summary:**
- Collaborator selection process (left)
	+ Team leader forms a research team
- Discussion routine (middle)
	+ Collaborative dialogue for task progression
- Architecture of author knowledge bank and paper database (right)
	+ Provides crucial information during collaboration
- Goal: Build multi-agent system using real academic data
	+ Simulate scientist team assembly, abstract generation for novel scientific idea
- VirSci System (proposed):
	1. Scientific research ecosystem
	2. Multi-agent system for scientific idea generation

### 3.1 The Scientific Research Ecosystem

**Scientific Research Ecosystem**

**Components**:
- Paper information (before ybound)
  * Past papers Bpast
  * Titles, citation counts, abstracts
- Corresponding author information (before and after ybound)
  * Contemporary papers Bcon
  * Same basic information as past papers

**Paper Databases**:
- **Past Paper Database Bpast**
  * Constructed using Faiss
  * Papers published before ybound
- **Contemporary Paper Database Bcon**
  * Constructed using Faiss
  * Papers published after ybound

**Author Knowledge Bank**:
- For each scientist in S:
  * Extract basic profile from computer science dataset
    * Name, affiliations, citation count, research interests, collaboration history
- Embed scientist profiles into **author knowledge bank** using KnowledgeBank module from AgentScope
  * Real author names masked to prevent data leakage and privacy issues

**Adjacency Matrix A**:
- Represents collaboration counts between scientists in S
- Ai,j denotes number of times scientist i has collaborated with scientist j
- Increment all values by 1 to encourage exploration of new partnerships

### 3.2 The Multi-agent System for Scientific Collaboration

**Collaborator Selection:**
- Team leader randomly selects scientists from S to form a team T based on probability distribution Pi,j calculated using adjacency matrix A
- Collaborators evaluate whether to join the team using chain-of-thought process
- Invitation Mechanism: New agents can be initialized for discussion but not added to the fixed-size team
- Discussions continue until predefined team size is reached or a topic is agreed upon

**Topic Discussion:**
- Team engages in discussions based on task description prompt
- Agents generate responses Rk,i from probability distribution Psi(⋅|Qk,i)
- Invitation Mechanism allows agents to seek advice from scientists outside the team without adding them to the fixed-size team
- After K turns, team leader generates final research topic based on content of discussions

**Idea Generation:**
- Agents required to generate comprehensive responses including idea description, experimental plan, and self-assessment
- Initial prompt provides references from Bpasmto aid in generating first idea
- Subsequent prompts incorporate previously generated ideas for refinement or new proposals
- After K turns, top three ideas with highest confidence are retained in the idea list I

**Idea Novelty Assessment:**
- Agents compare each idea with related papers from Bpasmto determine novelty
- Blind review process does not include dialogue memory in prompt
- Chain-of-thought process is used for agents to select their preferred idea and provide reasoning.

#### Multi-agent Abstract Refinement System for Scientific Writing

**Abstract Generation**

**Process**:
- Team selects highest voted idea as Ri⁢d⁢e⁢a for abstract generation
- Initial draft by team leader based on Ri⁢d⁢e⁢a
- Scientists evaluate, propose modifications, and revise abstract (K turns)
- Final abstract denoted as Ra⁢b⁢s⁢t⁢r⁢a⁢c⁢t
- Self-review mechanism for novelty check before providing to team leader

**Abstract Generation Prompt**:
- Qk,i = [Task description, Previous abstract (Rk,i−1), Evaluation metrics]
- Agents evaluate previous abstract and propose modifications, then revise the abstract
- Dialogue history not included in prompt as process is iterative

**Novelty Check**:
- Finalized Ra⁢b⁢s⁢t⁢r⁢a⁢c⁢t goes through self-review mechanism for novelty pre-check
- Discussed in more detail in Appendix E.1

**Ablation Study**:
- Uncertainty in total inference cost makes fair experimental comparisons difficult
- Effectiveness of this module discussed in Section 4.4

## 4 Empirical Study

### 4.1 Experimental Settings

**Dataset:**
* AMiner Computer Science Dataset 1: 1,712,433 authors, 2,092,356 papers (Tang et al., 2008)
* Period: 1948 - 2014
* Filtered data: 156 authors, 85,217 papers
* Data embedding: "mxbai-embed-large" model

**Implementation:**
* Agentscope framework (Gao et al., 2024)
* LLMs used: GPT-4o (OpenAI, 2023), Llama-3.1 (8b and 70b) (Dubey et al., 2024)
* Evaluated on team discussion setting with 4 members and 5 turns
* Normalized metrics: Historical Dissimilarity (HD), Contemporary Dissimilarity (CD), Contemporary Impact (CI), Overall Novelty (ON)

**Experimental Design:**
1. Explore VirSci's performance vs AI Scientist under similar conditions
2. Investigate impact of team settings on novelty: team size, discussion turns, team freshness, research diversity
3. Study effect of system components on generated abstracts' novelty
4. Examine adaptive turn numbers' impact on novelty.

### 4.2 Comparisons with AI scientist

**Comparison Results:**
- **GPT-4o**: outperforms AI Scientist across all metrics (LLM Review Score, CD, CI)
- **Others**: no significant difference in novelty scores
  - Llama-3.1 (8b and 70b): moderate changes in model size do not enhance novelty

**Justifications for Fair Comparison:**
1. Same research direction: include NanoGPT in topic selection for VirSci to align with AI Scientist's topics (2D Diffusion, etc.)
2. Equal inference cost: set team members and discussion turns for our system similar to the number of self-reflection turns in AI Scientist
3. Consistent paper retrieval: use Semantic Scholar API for both paper retrieval and metric calculation like AI Scientist does
4. Multiple metrics: evaluate generated abstracts using our metrics (CD, CI) and AI Scientist’s metric (LLM review score)

**Experimental Results:**
- Multi-agent system outperforms AI Scientist across all three metrics in Fig. [1]
  - Demonstrates effective enhancement of novelty in a collaborative setting
- GPT-4o shows superior ability to generate innovative ideas and abstracts.

### 4.3 Exploring Science of Science: The Impact of Team Dynamics on Novelty

**Team Size and Discussion Turns:**
- **Effects of team size on novelty**: Increasing number of team members can enhance novelty score due to broader ideas and perspectives, but peak innovation occurs with a team size of 8. Excessively large teams may introduce coordination challenges and communication barriers that hinder creativity.
- **Effects of discussion turns on novelty**: Appropriate number of turns enables exploration of topics thoroughly, leading to more sophisticated research outputs. Excessive turns can lead to fatigue and reduced engagement, stifling creativity. Peak novelty is achieved with a discussion turn count of 5.
- **Interaction between team size and discussion turns**: Larger teams with fewer turns or smaller teams with excessive discussion turns can still produce relatively high novelty scores. Strategic planning in collaborative settings is crucial for maintaining high levels of novelty.

**Team Freshness:**
- **Effects of team freshness on novelty**: A balanced mix of new and returning collaborators (50% freshness) yields the highest historical dissimilarity and overall novelty, particularly in larger teams. Contemporary research aligns more closely with future trends in teams with fresh members.

**Team Research Diversity:**
- **Effects of team diversity on novelty**: Increasing diversity enhances novelty metrics such as historical dissimilarity and contemporary impact, with peak innovation occurring at 50% diversity for larger teams. Balanced teams composed of diverse and non-diverse members can produce novel and impactful work.

### 4.4 Ablation Study

**Effects of Components Designed for Novelty (Continued)**

**Invitation Mechanism**:
- Introducing new scientists into discussion positively impacts team's performance
- Seeking external insights from relevant, non-team scientists fosters more diverse and novel ideas

**Novelty Assessment Step**:
- Significantly boosts scores in idea generation
- If not considered, output will be the last scientist's idea rather than a list of diverse ideas
- Novelty assessment ensures continuous evaluation for originality, helping teams avoid overlap with existing research

**Self-Review Mechanism**:
- Crucial in refining abstracts
- Allows team leader to re-evaluate abstract for novelty and discard low-quality abstracts
- Engages team in new discussion to generate a better idea, as evidenced by score improvements

**Comparison of Fixed Turns vs. Adaptive Turns**:
- Adaptive pattern shows lower inference cost and higher ON (novelty)
- More flexible approach allows teams to adjust discussions dynamically rather than rigidly
- Larger teams require more discussion turns, but adaptive pattern accommodates complexities while maintaining high novelty

## 5 Conclusion

**Virtual Scientist (VirSci): A Pioneering Multi-Agent System**

* Designed to replicate collaborative dynamics of scientific discovery
* Overcomes limitations of traditional single-agent systems by focusing on idea generation in initial phases of autonomous science discovery
* Structured five-step process allows for collaboration among specialized agents, offering diverse expertise and insights similar to real-world scientific teams
* Demonstrates improved performance compared to single-agent approaches
* Highlights emergent social behaviors among scientist agents, indicating potential avenues for studying collaborative mechanisms in scientific research

## Appendix A Data and Code Availability

### A.1 Code

**Code Accessibility**:
- Find at: [<https://github.com/open-sciencelab/Social_Science>](https://github.com/open-sciencelab/Social_Science)

### A.2 Data

Preprocessed data is located at [https://drive.google.com/drive/folders/1ZwWMBQ5oK-l4VuzMa60GbMND0g2EIxIu?usp=sharing]. (19 characters)

## Appendix B Limitations and Future Work

**Limitations of Multi-Agent System:**
* Validated on a single computer science dataset:
  - Restricts diversity of research ideas
  - Reduces generalizability to other scientific fields
* Focuses on a single domain:
  - Simplifies complexities of real-world teamwork
  - Limits capacity for interdisciplinary collaborations
* Future Directions:
  1. Expanding the system to incorporate datasets from various scientific disciplines:
     - Increases diversity of generated ideas
     - Enables simulations of interdisciplinary collaborations
     - Provides a more realistic representation of research environments
  2. Enhancing simulation of teamwork:
     - Allowing multiple teams to work concurrently
     - Enabling agents to contribute to multiple teams or projects simultaneously
     - Better reflects collaborative dynamics in modern scientific research
     - Offers deeper insights into the processes of scientific collaboration and innovation.

## Appendix C Ethics Statement

**Use of Data and System Design**
- Utilizes AMiner dataset (publicly available) to maintain data privacy
- Author names are masked during simulations
- System is intended to support human researchers, not replace them
- Emphasis on human oversight for quality/integrity of outputs
- Promote transparency by sharing all relevant codes for reproducibility within research community

## Appendix D Effect of the Potential Data Leakage

**Addressing Potential Data Leakage Concerns:**

**Reasons why data leakage does not significantly impact experiment validity:**
1. All models encounter same exposure to training data from 2011-2014:
   - Both comparisons between multi-agent system and AI Scientist
   - Comparisons between different team settings
2. Evaluation process remains fair:
   - Relative performance differences observed are not skewed by uneven data leakage
3. Goal is to explore how collaboration strategies influence novelty of outputs, not absolute measures:
   - All team settings face the same potential exposure to historical data
4. Conclusions drawn regarding collaboration strategies and team performance remain valid:
   - Novelty metrics provide accurate comparison of agents' ability to generate distinct ideas under varying conditions
5. Data leakage affects all models uniformly:
   - Does not undermine relative comparisons between different settings or conclusions drawn.

## Appendix E More Details of Methods

### E.1 Self-review

**Self-Review Mechanism**
* After Ra⁢b⁢s⁢t⁢r⁢a⁢c⁢t is finalized, a self-review mechanism is initiated to check novelty
* The optimized abstract Ra⁢b⁢s⁢t⁢r⁢a⁢c⁢t is provided to the team leader for comparison with similar papers in Bp⁢a⁢s⁢t
* If this is the first self-review and the similarity to existing papers is too high, the abstract undergoes further revision
* The evaluation Rr⁢e⁢v⁢i⁢e⁢w is added to Eq. ([6](https://arxiv.org/html/2410.09403v1#S3.E6 "In 3.2 The Multi-agent System for Scientific Collaboration ‣ 3 The Virtual Scientists ‣ Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation")) for the next revision round
* If the abstract fails a second self-review, it will be discarded and a new idea is generated
* Once the self-review meets novelty requirements, the final abstract is produced and the system terminates
* This self-review mechanism introduces uncertainty in total inference cost, making fair experimental comparisons difficult
* Effectiveness of this module discussed only in ablation study (Sec. [4.4](https://arxiv.org/html/2410.09403v1#S4.SS4 "4.4 Ablation Study ‣ 4 Empirical Study ‣ Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation"))

## Appendix F More Experiments

**Evaluation of Novelty Metric Validity**

**Methodology**:
- Extract 100 abstracts generated under different experimental conditions
- Evaluate using:
  - **Proposed overall novelty metric**
  - LLM-based reviewer (GPT-4o API)
  - Human researchers in computer science domain
- Use idea review form from [Si et al., 2024] for scoring

**Results**:
- Figure 6: Scoring of same abstract under different metrics (X-axis: Score for metric 1, Y-axis: Score for metric 2)
  - Pearson correlation coefficient between proposed novelty metric and LLM-based reviewer = **0.59**, positive correlation
- Figure 7: Scoring of same abstract under different metrics (X-axis: Score for metric 1, Y-axis: Score for metric 3)
  - Pearson correlation coefficient between proposed novelty metric and human researcher = **0.51**, positive correlation

**Conclusion**:
- Positive correlation between the proposed overall novelty metric and currently used novelty measurement methods (Lu et al., 2024; Si et al., 2024)
- Validity of the proposed metric is supported to some extent.

## Appendix G Prompts

### G.1 Scientist Definition

**Personal Information as Agent Definition**
- Utilize scientists' personal details to define agent
- Personal details include name, role, affiliation, research interests, citation status, and collaboration history
- See Fig. 8 for system prompt illustration (source: [2410.09403v1](https://arxiv.org/html/2410.09403v1#A7.F8))

### G.2 Collaboration Selection

**Collaboration Selection Prompt**:
- Illustrated in Fig. [9](https://arxiv.org/html/2410.09403v1#A7.F9) (Figure 9)
- Depicts a multi-agent system for improving scientific idea generation
- Figure caption: "Two Heads Are Better Than One" (Appendix G Prompts)

### G.3 Topic Discussion

**G.3.1 Discussion** \nFig. [10](https://arxiv.org/html/2410.09403v1#A7.F10) illustrates the discussion prompt for the topic "Multi-Agent System can potentially improve scientific idea generation."

![Figure 10 caption: G.3.1 Discussion, G.3 Topic Discussion, Appendix G Prompts, Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation](https://arxiv.org/html/2410.09403v1/x10.png)

**G.3.2 Summarization** \nFig. [11](https://arxiv.org/html/2410.09403v1#A7.F11) represents the prompt for final topic selection after topic discussion, based on "Multi-Agent System and its potential to improve scientific idea generation."

![Figure 11 caption: G.3.2 Summarization, G.3 Topic Discussion, Appendix G Prompts, Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation](https://arxiv.org/html/2410.09403v1/x11.png)

### G.4 Idea Generation

**Idea Generation Prompt**:
- Shown in Fig. [12](https://arxiv.org/html/2410.09403v1#A7.F12 "Figure 12: G.4 Idea Generation")
- Illustrates a multi-agent system's potential for enhancing scientific idea generation (Refer to caption and image)

### G.5 Novelty Assessment

**Novelty Assessment Prompt Illustrated in Fig. [13](https://arxiv.org/html/2410.09403v1#A7.F13 "Figure 13")**:

- Figure 13 shows the prompt for a novelty assessment (G.5) located in Appendix G Prompts of [Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation](https://arxiv.org/html/2410.09403v1).

### G.6 Abstract Generation

**Discussion Prompts:**
* **Beginning case**: Figure 14 - Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation (see caption)
* **Normal case**: Figure 15 - Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation (see caption)

**Self-review Prompt:**
* After generating the final abstract: Figure 16 - Two Heads Are Better Than One: A Multi-Agent System Has the Potential to Improve Scientific Idea Generation (see caption)

### G.7 LLM Review

**NeurIPS2024 Review Guidelines for LLM-based Comparison**

For a fair comparison, we utilize the same metric as AI Scientist, based on NeurIPS2024 reviewer guidelines. We focus on only critical metrics since our evaluation is limited to the abstract.
[Figure 17](https://arxiv.org/html/2410.09403v1#A7.F17) shows this process, titled "LLM Review".

## Appendix H Example Scenarios

### H.1 Collaboration Selection

Scientific collaborators are selected based on their diverse backgrounds (Figure 18), with some accepting and others rejecting invitations due to these differences.

Figure 18: Collaborator Selection Scenario
Diverse backgrounds influence scientist's choices.

### H.2 Topic Discussion

**Topic Discussion: Normal Case**
- Scientists provide coherent topic discussions (highlighted in yellow)
- Ensure a thorough discussion of research topic
- Example scenario illustrated in Figure 19

**Topic Discussion: Invitation Mechanism**
- Ensures comprehensive topic discussion
- Collaboration invitation mechanism highlighted in blue
- Example scenario illustrated in Figure 20

### H.3 Idea Generation

**Figure 21**: Example scenario for initial stage of idea generation. [See source](https://arxiv.org/html/2410.09403v1#A8.F21)

**Figure 22**: Example scenario for normal stage in idea generation. [See source](https://arxiv.org/html/2410.09403v1#A8.F22)

### H.4 Novelty Assessment

**Novelty Assessment Scenarios**

**User Prompt**:
- Illustrated in Figure 23
- Includes three candidate ideas and related papers

**Agent Responses**:
- Illustrated in Figure 24
- By max-voting, idea 2 is selected as the final idea

**Additional Details**:
- Figure 23 corresponds to the user prompt scenario
- Figure 24 corresponds to the agent responses scenario

### H.5 Abstract Generation

**Abstract Generation:**
* **Normal Case**: Illustrated in Figure 25 (https://arxiv.org/html/2410.09403v1#A8.F25) and Figure 26 (https://arxiv.org/html/2410.09403v1/x26.png)
* Scenario of a multi-agent system improving scientific idea generation
* Agents work together to create abstracts
* Agents review each other's work for improvements
* Feedback loop enhances final output quality

**Self-review:**
* Scenario illustrated in Figure 27 (https://arxiv.org/html/2410.09403v1#A8.F27)
* Agents evaluate their own work for potential improvements
* Self-reflection leads to more thorough analysis and better results.

