# OASIS Open Agent Social Interaction Simulations with One Million Agents

source: https://arxiv.org/html/2411.11581v4
by Ziyi Yang, Zaibin Zhang, Zirui Zheng, Yuxian Jiang, Ziyue Gan, Zhiyu Wang, Zijian Ling, Jinsong Chen, Martz Ma, Bowen Dong, Prateek Gupta, Shuyue Hu, Zhenfei Yin, Guohao Li, Xu Jia, Lijun Wang, Bernard Ghanem, Huchuan Lu, Chaochao Lu, Wanli Ouyang, Yu Qiao, Philip Torr, Jing Shao

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Methodology](#2-methodology)
  - [2.1 Workflow of OASIS](#21-workflow-of-oasis)
  - [2.2 Environment Server](#22-environment-server)
  - [2.3 RecSys](#23-recsys)
  - [2.4 Agent Module](#24-agent-module)
  - [2.5 Time Engine](#25-time-engine)
  - [2.6 Scalable Design](#26-scalable-design)
- [3 Experiment](#3-experiment)
  - [3.1 Experimental Scenarios](#31-experimental-scenarios)
  - [3.2 Experimental Settings](#32-experimental-settings)
  - [3.3 Can OASIS be Adapted to Various Platforms and Scenarios to Replicate Real-world Phenomena?](#33-can-oasis-be-adapted-to-various-platforms-and-scenarios-to-replicate-real-world-phenomena)
  - [3.4 Does the Number of Agents Affect the Accuracy of Simulating Group Behavior?](#34-does-the-number-of-agents-affect-the-accuracy-of-simulating-group-behavior)
  - [3.5 Misinformation Spreading in One Million Agents](#35-misinformation-spreading-in-one-million-agents)
- [4 Ablation Study](#4-ablation-study)
  - [4.1 Analysis of Efficiency for One Million Users](#41-analysis-of-efficiency-for-one-million-users)
  - [4.2 Ablation of Components in OASIS](#42-ablation-of-components-in-oasis)
- [5 Conclusion](#5-conclusion)
- [Appendix A Acknowledgements](#appendix-a-acknowledgements)
- [Appendix B Related Work](#appendix-b-related-work)
  - [B.1 Social Media](#b1-social-media)
  - [B.2 Multi-Agent Systems](#b2-multi-agent-systems)
  - [B.3 Multi-Agent System Social Simulation](#b3-multi-agent-system-social-simulation)
- [Appendix C Ablation Study](#appendix-c-ablation-study)
  - [C.1 More Efficiency Analysis](#c1-more-efficiency-analysis)
  - [C.2 Recommend System Ablation](#c2-recommend-system-ablation)
  - [C.3 Temporal Feature Ablation](#c3-temporal-feature-ablation)
  - [C.4 LLM Ablation](#c4-llm-ablation)
- [Appendix D Method Details](#appendix-d-method-details)
  - [D.1 User Actions Prompts](#d1-user-actions-prompts)
  - [D.2 Environment Server Database Structure](#d2-environment-server-database-structure)
  - [D.3 Recommendation System](#d3-recommendation-system)
  - [D.4 Parallel Optimization](#d4-parallel-optimization)
- [Appendix E Data Preparations](#appendix-e-data-preparations)
  - [E.1 Real-World Propagation Data](#e1-real-world-propagation-data)
  - [E.2 Group Polarization](#e2-group-polarization)
  - [E.3 Herd Effect](#e3-herd-effect)
- [Appendix F Experiments Details](#appendix-f-experiments-details)
  - [F.1 Actions of Different Scenarios](#f1-actions-of-different-scenarios)
  - [F.2 Information Spreading](#f2-information-spreading)
  - [F.3 Group Polarization](#f3-group-polarization)
  - [F.4 Herd Effect](#f4-herd-effect)
- [Appendix G Misinformation Spreading in One Million Agents](#appendix-g-misinformation-spreading-in-one-million-agents)
  - [Technology](#technology)
  - [Education](#education)
  - [Entertainment](#entertainment)
  - [Health](#health)
- [Appendix H limitations \& Future Directions](#appendix-h-limitations--future-directions)
- [Appendix I Social Impact and Ethical Considerations](#appendix-i-social-impact-and-ethical-considerations)

## Abstract

**Social Media Simulator: OASIS**

**Background:**
- Growing interest in enhancing rule-based agent-based models (ABMs) for social media platforms with large language model (LLM) agents
- Several LLM-based ABMs proposed in the past year, but each is specific to a scenario and resource-intensive
- Real-world social media platforms involve millions of users

**Features of OASIS:**
- **Generalizable and scalable**: designed based on real-world social media platforms
- **Dynamically updated environments**: dynamic social networks and post information
- **Diverse action spaces**: following, commenting
- **Recommendation systems**: interest-based and hot-score-based
- **Large-scale user simulations**: capable of modeling up to one million users
- **Flexible**: can be easily extended to different social media platforms

**Findings:**
- Replication of various social phenomena: information spreading, group polarization, herd effects on X and Reddit
- Observations of social phenomena at different agent group scales
  - Larger agent group scale leads to more enhanced group dynamics and diverse opinions

**Authorship and Contributions:**
- **First Co-Author**: equal contribution, authorship order is random
- **Second Co-Author**: equal contribution, authorship order is random
- **Corresponding Author**: not indicated in the provided text.

**Figure 1:**
- OASIS can simulate different social media platforms, such as X and Reddit, and supports simulations of up to millions of LLM-based agents.

## 1 Introduction

**Complex Societal Systems**
- Characterized by many interconnected and interdependent components or agents
- Interactions give rise to emergent behaviors not predictable from individual actions
- Important in digital world, but experiments can be costly

**Agent-Based Models (ABMs)**
- Used to understand complex systems where real-world experiments are difficult
- Consist of computational agents interacting with each other or environment
- Agents programmed to behave in realistic manner

**Large Language Models (LLMs)**
- Demonstrated capability to mimic human behaviors
- Can engage in role-playing and take simple to complex actions
- Used for comprehensive testing and analysis of social media studies

**Social Media Platforms**
- Differ in action space, recommendation systems (RecSys), and dynamic network design
- Influence user behavior, making them crucial environments for studying modern social dynamics

**OASIS: Open Agent Social Interaction Simulations**
- Collection of generalizable and scalable ABMs to study various social media platforms
- Built on five foundational components: Environment Server, RecSys, Agent Module, Time Engine, and Scalable Inferencer
- Supports large-scale user simulations ranging from hundreds to millions of agents
- Enables comprehensive agent experiments with an advanced multi-processing technique and comprehensive user generation method.

## 2 Methodology

OASIS is a highly generalizable simulator for various social media, developed using LLM-based technology. We outline its workflow and internal mechanisms that enable easy scaling and adaptation to simulate millions of agents.

### 2.1 Workflow of OASIS

**OASIS Components and Simulation Process**

**Components:**
- **Environment Server**: stores user information, including name, self-description, and historical posts
- **RecSys (Recommendation System)**: filters suggested posts based on user interests and contextual factors
- **Agent Module**: selects actions to take, such as liking or reposting a post
- **Time Engine**: governs agent activation based on hourly activity probabilities
- **Scalable Inferencer**: enables chain-of-thought (CoT) reasoning for generating actions and thoughts

**Registration Phase:**
- Users provide personal information during registration
- Receive character descriptions to align with their characteristics
- Agents receive action descriptions for specific social media platform tasks

**Simulation Phase:**
- Environment sends user data to RecSys for suggested posts
- Agents select actions based on RecSys suggestions, self-description, and contextual factors
- Chain-of-Thought (CoT) reasoning incorporated for generating actions and thoughts
- Time engine activates agents at specific times based on hourly activity probabilities
- Results updated in the environment server upon agent performance.

### 2.2 Environment Server

**Environment Server Components and Functions:**
* Maintains status and data of social media platforms: users' information, posts, user relationships
* Implemented using a relational database for efficient management
* Detailed database structure provided in Appendix D.2
* Six primary components: users, posts, comments, relations, traces, recommendations

**User Table:**
- Stores basic user information: name and biography

**Post Table:**
- Contains all the posts made on the platform
- Includes detailed information: number of likes, creation time

**Comment Table:**
- Stores comments made on the platform
- Includes detailed information

**Relations Component:**
- Multiple tables store various relationships
  - Follow and mutual relationships between users
  - Likes between users and posts

**Trace Table:**
- Records each user's entire action history

**Recommendation Table:**
- Populated by the output of the Recommendation System (RecSys)
- Analyzes user's trace table.

**Database Dynamics:**
* Can be dynamically updated: new users, posts, comments, follow relationships added over time.

### 2.3 RecSys

**Recommendation System (RecSys)**

**Role**: Controls information seen by agents, shaping the flow of information. Developed for two social media platforms: X and Reddit.

**X Platform**:
- Recommended posts come from in-network (followed users) and out-of-network sources
- **In-network content**: Ranked by popularity (likes) before recommendation
- **Out-of-network posts**: Recommended based on interest matching using TwHIN-BERT, modeling user interests through profile and recent activity similarity
- Factors taken into account: recency, number of followers of post's creator (superuser broadcasting)
- Post count from in-network and out-of-network sources can be adjusted for different scenarios

**Reddit Platform**:
- RecSys modeled based on Reddit's disclosed post ranking algorithm
- **Hot score calculation**: Integrates likes, dislikes, and created time to prioritize recent and popular posts
- Formula: `h = log_10(max(|u-d|, 1)) + sign(u-d)·t - t_0/45000` where:
  - `h`: hot score
  - `u`: number of upvotes
  - `d`: number of downvotes
  - `t`: submission time in seconds since Unix epoch, `t_0 = 1134028003`
- Posts ranked based on hot scores to identify top k for recommendation with varying numbers depending on experiment.

### 2.4 Agent Module

**Agent Module Features**
* Based on large language models
* Inherits core features from CAMEL [^22]
* Consists of memory module and action module

**Memory Module**
- Stores information encountered by the agent
- Includes details about posts: number of likes, comments, and user interactions (likes, comments)
- Stores user's previous actions and reasoning behind them

**Action Module**
- Enables 21 different types of interactions with environment
* Sign up
* Refresh
* Trend
* Search posts
* Search users
* Create post
* Repost
* Follow
* Unfollow
* Mute
* Like
* Unlike
* Dislike
* Undo dislike
* Unmute
* Create comment
* Like comment
* Unlike comment
* Dislike comment
* Undo dislike comment
- Detailed descriptions available in Appendix [D.1](https://arxiv.org/html/2411.11581v4#A4.SS1)

**Additional Features**
- Utilizes CoT reasoning for enhanced interpretability of behaviors
- Increases user interaction diversity, making it closer to real-world social media platforms.

### 2.5 Time Engine

**Incorporating Temporal Features**
- Define agent's hourly activity level based on:
  - Historical interaction frequency
  - Customized settings
- Agents activated based on probabilities, not all at once
- Manage time progression using a time step approach (1 time step = 3 minutes)
  - Accommodates varying LLM inference speeds
- Propose alternative time-flow setting:
  - Linearly maps real-world time to simulation time
  - Actions executed earlier within the same time step have earlier timestamps in the database.

### 2.6 Scalable Design

**Scalable Inference System Design**
- Highly concurrent distributed system for agents, environment server, and inference services
- Modules communicate through information channels
- Asynchronous mechanisms allow multiple requests from agents while waiting for responses
- Environment module processes incoming messages in parallel
- GPU resource management by dedicated manager to ensure efficient utilization
- See Appendix [D.4](https://arxiv.org/html/2411.11581v4#A4.SS4 "D.4 Parallel Optimization ‣ Appendix D Method Details ‣ OASIS: Open Agent Social Interaction Simulations with One Million Agents") for more details

**Large-scale User Generation Algorithm**
- Combines real user data and relationship network model to simulate up to one million users
- Preserves scale-free nature of social networks
- Diverse user profiles based on population distributions: age, personality, profession as independent variables
- Core and ordinary users linked using interest-based sampling (0.2 probability of following core users)
- Prevents network density and ensures diversity
- Details in Appendix [E.1](https://arxiv.org/html/2411.11581v4#A5.SS1 "E.1 Real-World Propagation Data ‣ Appendix E Data Preparations ‣ OASIS: Open Agent Social Interaction Simulations with One Million Agents"), [E.2](https://arxiv.org/html/2411.11581v4#A5.SS2 "E.2 Group Polarization ‣ Appendix E Data Preparations ‣ OASIS: Open Agent Social Interaction Simulations with One Million Agents"), and [E.3](https://arxiv.org/html/2411.11581v4#A5.SS3 "E.3 Herd Effect ‣ Appendix E Data Preparations ‣ OASIS: Open Agent Social Interaction Simulations with One Million Agents")

## 3 Experiment

We primarily focus on two research questions: 

1. Can OASIS be applied to various platforms and scenarios? We replicate three influential studies: information propagation on X, group polarization on rapid exchange platforms, and herd effect on Reddit.
2. Does the number of agents affect simulation accuracy? We conduct sociological experiments with hundreds to tens of thousands of agents to identify emergent phenomena as agent numbers increase.

### 3.1 Experimental Scenarios

Information Propagation on X:
We examine how messages spread through networks influenced by factors like network structure, content, and interactions.

Two key aspects are:

1. Information Spreading: Messages transmitted across a network.
2. Group Polarization: Social interactions foster extreme opinions.

Our analysis focuses on these dynamics within the Reddit platform.

### 3.2 Experimental Settings

**Information Spreading in OASIS:**
* Collect data from Twitter15 and Twitter16 datasets for 198 real-world instances
* Use X API to retrieve user profiles, follow relationships, and previous posts
* Initialize agents with this data and their most recent posts
* Measure scaling, depth, and max breadth of propagation paths using normalised RMSE as evaluation metric
	+ Scale: number of users participating in propagation over time
	+ Depth: maximum depth of propagation graph of source post
	+ Max breadth: largest number of users participating at any depth
* Evaluate precise alignment through mean and confidence intervals of normalised RMSE at each minute. (See [F.2](https://arxiv.org/html/2411.11581v4#A6.SS2) for details)

**Group Polarization in OASIS:**
* Select 196 real users' information from the information-spreading experiment
* Use LLMs to generate synthetic users with up to 1 million scale
* Set real users as core users, generated users forming follow-up relationships based on topics
* Assess which opinions are more extreme or helpful using GPT-4o-mini (see [F.3](https://arxiv.org/html/2411.11581v4#A6.SS3) for details)

**Herd Effect in OASIS:**
* Collect 116,932 real comments from Reddit across seven topics and use LLMs to generate profiles for 3,600 users
* Collect 21,919 counterfactual content posts and generate 10,000 users
* Divide into down-treated group (one initial dislike), control group (no initial likes or dislikes), and up-treated group (one initial like)
* Simulate interactions for each experiment on Reddit, introducing initially-rated comments or posts at the beginning of each time step
* Use Llama3-8b-instruct as base LLM and adjust agent actions to accommodate scenarios

**Evaluation Metrics:**
* Information Spreading: scale, depth, max breadth, normalised RMSE (scale, depth, max breadth), mean and confidence intervals of normalised RMSE at each minute (see [F.2](https://arxiv.org/html/2411.11581v4#A6.SS2) for details)
* Group Polarization: use GPT-4o-mini to assess which opinions are more extreme or helpful (details in [F.3](https://arxiv.org/html/2411.11581v4#A6.SS3))
* Herd Effect: post score (difference between number of upvotes and downvotes) and disagree score (degree of disagreement expressed in comments responding to counterfactual content) (see [F.4.1](https://arxiv.org/html/2411.11581v4#A6.SS4.SSS1) for details)

### 3.3 Can OASIS be Adapted to Various Platforms and Scenarios to Replicate Real-world Phenomena?

**Information Propagation in X: OASIS Simulation vs Real World**

**Finding 1:**
- OASIS can replicate real world information spreading process in terms of scale and maximum breadth with minimal offset
- Depth trend of simulation results is smaller compared to real world trends
- Complexity and precision of real-world Recommendation Systems (RecSys) and user profiles limit the accuracy of modeling intermediary users

**Experiment:**
- OASIS simulation aligns well with real-world information dissemination trends, RMSE error margin around 30%
- Discrepancy in depth likely due to simplified design of RecSys and data limitations

**Finding 2:**
- OASIS can replicate group polarization phenomenon during information propagation
- Group Polarization: Individuals with similar views adopt more extreme positions after exchanging opinions
- Agents' responses to Halen's suggestions become increasingly conservative, especially in interactions with uncensored models
- Uncensored models tend to use extreme phrases like 'always better'

**Group Polarization Experiment:**
- Set a hypothetical scenario: Discussion about whether Halen should take risks or continue writing ordinary novels
- Agents assigned conservative views, opinions compared using GPT-4o-mini every 10 time steps
- Results show agents' responses become increasingly extreme over time, especially in interactions with uncensored models

**Finding 3:**
- Agents are more likely to exhibit herd effect than humans during social interactions
- For down-treated comments, agents tend to dislike or give fewer likes while humans deliberate more carefully and may increase the like score.

### 3.4 Does the Number of Agents Affect the Accuracy of Simulating Group Behavior?

**Experiments on Group Polarization and Agent Scales**
- Conducted experiments to investigate how agent opinions change at different scales (from 196 to 100K)
- Collected suggestions from the same 196 users for all experiments
- Kept consistent experimental settings as described in group polarization
- Ran simulation for 30 time steps
- Visualized distribution of agents' opinions using Nomic Atlas (Figure 7)

**Findings on Group Polarization and Agent Scales**

**Larger Groups Lead to More Helpful and Diverse Responses:**
- When number of agents increases, there is significant enhancement in diversity of user opinions (Figure 7)
- User opinions from larger groups are more helpful based on evaluation criteria from Safe-RLHF

**Herd Effect in Reddit**

**Question:** Whether agents would exhibit herd effect when exposed to counterfactual posts (*i.e.*, misinformation).

**Observation:** No herd effect observed in small groups, no difference between up-treated, control, and down-treated groups.

**Increased Number of Agents:** When number of agents was increased from 100 to 10,000, agents began to exhibit explicit herd effect:
- Disagree scores in down-treated group were significantly higher than those in control and up-treated groups (Figure 8)
- Large-scale groups tend to guide agents toward self-correction.

**Specific Examples of Herd Effect:** Appendix F.4.3 illustrates specific examples through posts and comments.

### 3.5 Misinformation Spreading in One Million Agents

**Understanding Information Spread on Social Media Platforms**

**Simulation of a Social Media Platform**:
- Designed a large-scale simulation of a social media platform resembling Twitter, with one million users and celebrity accounts with large followings
- Four distinct information sets were disseminated, consisting of both factual and false narratives related to technology, health, education, and entertainment

**Findings**:
- Rumors have a significantly broader reach than truthful information
- Agents tend to engage more with rumor posts that are novel and emotionally provocative
- Users form isolated groups over time, as shown in **Figure 10**

**Impact on Social Connections**:
- Users form new relationships during the interaction process, as shown in **Figure 10** (green dots represent agents, red lines indicate newly established relationships)
- The number of posts discussing various pairs of rumors and truths over time is visualized in **Figure 9**

**Methodology**:
- The activation probability (likelihood of users sharing information) was set at:
  - 0.1 for core users
  - 0.01 for regular users

## 4 Ablation Study

### 4.1 Analysis of Efficiency for One Million Users

**Study Findings:**
* Report on runtime, GPU utilization, number of tweets, and comments for simulations at scales of one million, one hundred thousand, and ten thousand under group polarization setting using A100 GPUs.
* Efficient conduct of large-scale user interactions: simulating 100,000 users over 10 time steps with five A100 GPUs within two days.
* Additional efficiency analysis in Appendix C.1 for other scenarios.

**Efficiency Analysis (One Million Agents):**
* For scale of one million agents: hours per time step = 18.0, GPUs (A100) = 27.0, new tweets per time step (K) = 48.5, new comments per time step (K) = 97.1.
* Five A100 GPUs enable efficient simulation of large-scale user interactions.

### 4.2 Ablation of Components in OASIS

**Ablation Experiments on OASIS:**
* Conducted on RecSys module and temporal feature of Time Engine
* Impact of absence of RecSys: hinders spread of information, limits dissemination potential
* Tested various models: MiniLM v6 [^40], BERT [^7], TwHIN-BERT
* TwHIN-B Bert performs well due to pre-training on 7 billion tweets in 100+ languages
* Replaced activity probability list with a fixed list of ones for temporal feature
* Results demonstrate importance of real-world data activity probabilities in reproducing dissemination patterns

**Experiment Metrics:**
* Primary metric: Normalized RMSE at every minute (for detailed analysis)

## 5 Conclusion

We present OASIS, a scalable social media simulator that replicates real-world dynamics. It has modular components for adaptability across platforms and can support up to 1 million users. We used OASIS to reproduce known social phenomena and identify unique behaviors emerging from large language model-driven simulations.

## Appendix A Acknowledgements

**Project Leadership and Contributors:**
- **Jing Shao**, Zhenfei Yin, Guohao Li: project co-leaders
- Ziyi Yang: database, information channel, interfaces, time engine, Reddit recommendation system, experimental design
- Zaibin Zhang: recommendation system codebase, environment server, agent generation, large-scale simulation optimization, architecture design, scenario development
- Zirui Zheng: Time Engine, Twitter's recommendation system, information propagation experiment (data preparation, prompt iterations, result visualization)
- Yuxian Jiang: action models (all prompt iterations), agent generation, polarization experiment design and implementation
- Ziyue Gan: scenario analysis, experimental results, reference collection, diagram drawing, herd effect introduction
- Zhiyu Wang: codebase of asynchronous system, LLM deployment, GPU resource management, herd effect and group polarization experiment implementation
- Jinsong Chen: framework design and collaboration solution setup (initial phase)
- Martz Ma, Bowen Dong: experiment result analysis, graphical design, paper writing
- Prateek Gupta, Shuyue Hu, Xu Jia, Lijun Wang, Philip Torr, Yu Qiao, Wanli Ouyang, Huchuan Lu, Bernard Ghanem: advice and guidance on project development and experimentation.

## Appendix B Related Work

### B.1 Social Media

Social media encompasses communication, interaction, and content-sharing platforms. While it offers benefits like self-expression without real-world consequences, hazardous phenomena pose significant economic, political, and social risks. These threats include promoting risky behaviors, mental health issues in teens, social influence, group polarization, and misinformation.

Complex network structures, vast data, and diverse behaviors hinder research on these topics. Ethical concerns also arise in some studies. To address this, a controllable virtual environment for social simulation is needed to test hypotheses on a virtual platform.

### B.2 Multi-Agent Systems

**Multi-Agent Systems**
* Composed of multiple autonomous entities
* Each possesses different information and diverging interests
* Advantages:
  + Ability to assume different roles in group activities
  + Richer and more complex interaction behaviors (collaboration, discussion, strategic competition)
* Recent studies demonstrate potential across various domains:
  + **Tool-based agent assistants**: Collaborating a small group of LLM-based agents to conduct tasks
  + **Society or game simulation environments**: Involving large-scale agent groups to run simulations in complex environments
* Focus on scalability due to the complicated nature of interactions in a large society.

### B.3 Multi-Agent System Social Simulation

**Social Simulation in Social Science Research:**
* Role of agent-based modeling (ABM) studies: Schelling's model of segregation [^43], Chicago simulation [^27], pandemic [^13]
* Traditional ABM limitations: subjective rule design, scalability issues
* Advantages of large language models (LLMs):
  + Interact using natural language
  + More accurate simulation of human behavior
  + Utilize more complex tools
* Recent studies using LLM-based agents:
  + Multi-agent behavior patterns [^37]
  + Simulations of social networks [^9]
  + Society's response to misinformation [^3]
* Importance for exploring LLMs' capabilities: alignment [^24], emergence of social norms [^41]
* Current focus on small-number interactions; research gap in large-scale agent studies
* Emphasis on large-scale agents to investigate collective behaviors.

## Appendix C Ablation Study

### C.1 More Efficiency Analysis

**Table 3: Experiment Efficiency Analysis**

| Scale | Time/Time Step | GPUs (A100) | Comments/Time Step |
| --- | --- | --- | --- |
| 10k | 15 minutes | 4 | 1393 |
| 1k | 0.83 minutes | 4 | 129 |
| 100 | 0.33 minutes | 4 | 14 |

### C.2 Recommend System Ablation

**Recommendation System (RecSys) Ablation Studies**

**Impact on Message Dissemination**:
- Ablation studies conducted to verify RecSys impact:
    - Existence of RecSys itself
    - Different models for embedding posts and profiles
- 28 randomly selected topics from original 198, covering all categories
- **RecSys removal**:
    - Worked well for entertainment topics due to dense follower networks in fan groups
    - Premature end of information spread for most groups, manifesting as broadcast behavior from a single superuser
    - Essential for connecting isolated nodes and sustaining the simulation
- **Different RecSys models**:
    - TwHIN-BERT (pretrained on 7 billion posts in 100+ languages) more suitable than general models
    - Paraphrase-MiniLM-L6-v2, BERT-base-multilingual-cased (regular BERT) vs. TwHIN-BERT:
        - **TwHIN-BERT** and regular BERT showed much better performance than paraphrase-MiniLM-L6-v2 (Figure 11(a))
    - Recommendation results in Figure 11(b) show that TwHIN-BERT could recommend a more proper post.

### C.3 Temporal Feature Ablation

We ablate our temporal feature in Figure 12 by comparing OASIS's performance on 28 topics without it to its full version. Without this feature, OASIS struggles to capture real-world information propagation dynamics, likely due to agents acting too frequently.

### C.4 LLM Ablation

We tested Qwen1.5-7B-Chat, Internlm2-chat-20b, and Llama-3-8B-Instruct as backend agents for simulating real-world information propagation on 28 random topics. The results are shown in Figure 13, with normalized RMSE values indicating the performance of each model.

## Appendix D Method Details

### D.1 User Actions Prompts

**Actions Available**
- Sign up new user: `sign_up(user_name, name, bio)`
- Create post: `create_post(content)`
- Repost post: `repost(post_id)`
- Like post: `like_post(post_id)`
- Dislike post: `dislike_post(post_id)`
- Undo dislike post: `undo_dislike_post(post_id)`
- Create comment on a post: `create_comment(post_id, content)`
- Like comment: `like_comment(comment_id)`
- Unlike comment: `unlike_comment(comment_id)`
- Dislike comment: `dislike_comment(comment_id)`
- Undo dislike comment: `undo_dislike_comment(comment_id)`
- Follow user: `follow(followee_id)`
- Unfollow user: `unfollow(followee_id)`
- Mute user: `mute(mutee_id)`
- Unmute user: `unmute(mutee_id)`
- Search posts: `search_posts(query)`
- Search users: `search_user(query)`
- Retrieve current trending topics: `trend()`
- Refresh feed: `refresh()`

**Self-Description**
- Consistent with self-description and personality.

**Response Format**
- Format: `{"reason": "your feeling about these posts and users, then choose some functions based on the feeling. Reasons and explanations can only appear here.", "functions": [...]}`
- Ensure JSON format.
- Include key 'name'.

### D.2 Environment Server Database Structure

**Database Tables:**
- **Post table**: Stores information about each post created by a user, including ID, user ID, content, creation time, number of likes, and number of dislikes.
  * Example: Table 4 in the text.
- **Dislike table**: Tracks users who have disliked a specific post, including their ID, user ID, post ID, and creation time.
  * Example: Table 5 in the text.
- **Like table**: Tracks users who have liked a specific post, including their ID, user ID, post ID, and creation time.
  * Example: Table 6 in the text.
- **Comment table**: Stores information about each comment made on a post, including ID, post ID, user ID, content, and creation time.
  * Example: Table 7 in the text.
- **Comment Dislike table**: Tracks users who have disliked a specific comment, including their ID, user ID, comment ID, and creation time.
  * Example: Table 8 in the text.
- **Comment Like table**: Tracks users who have liked a specific comment, including their ID, user ID, comment ID, and creation time.
  * Example: Table 9 in the text.
- **User table**: Stores information about each user, including their ID, agent ID, username, name, bio, creation time, number of followers, and number of followings.
  * Example: Table 10 in the text.
- **Follow table**: Tracks users who are following another user, including their follower ID, followee ID, and creation time.
  * Example: Table 11 in the text.
- **Mute table**: Stores information about muted users, including the muter ID, mutee ID, and creation time.
  * Example: Table 12 in the text.
- **Trace table**: Tracks various user actions, including sign-ups, post creations, dislikes/likes, comments, muting, following, and recommendation system cache.
  * Example: Table 13 in the text.
- **Rec table (recommendation system cache)**: Stores information about recommended posts for specific users.
  * Example: Table 14 in the text.

### D.3 Recommendation System

**Post Recommendation System**

**Overview**:
- Ranks all posts and saves highest-ranked ones in a recommendation table
- Size of this table can be adjusted but remains constant for each user during an experiment

**Retrieving Recommended Posts**:
- When an agent selects "refresh" action, environment server retrieves post IDs linked to the user's ID from the recommendation table
- Subset of these post IDs is randomly sampled and their full content is queried from the post table, then sent to the user

**Recommendation Algorithm (X)**:
- Calculates score between a post and a user using the following formula:
```
Score = R * F * S
```
- Components of this formula:
  - **R (Recency Score)**
    - Based on difference between `t_current` (current timestamp) and `t_created` (timestamp when post was created), divided by 100
    - Calculated using natural logarithm (ln) with a base of 271.8
  - **F (Fan Count Score)**
    - Maximum value of 1 and logarithm (base 1,000) of fan count plus one
  - **S (Cosine Similarity Score)**
    - Cosine similarity between embeddings `E_p` (post content) and `E_u` (user profile and recent post content)

**Embeddings**:
- `E_p`: Embedding of the post content
- `E_u`: Embedding of the user profile and recent post content

### D.4 Parallel Optimization

**Information Channel:**
- Multiple agents interact asynchronously and concurrently with social media environment and inference management servers
- Advanced event-driven architecture for broader event categories and large model requests
- Dedicated channel facilitates communications between agents and servers:
  - Asynchronous message queue for agent requests
  - Thread-safe dictionary for response storage
- Upon receiving a request, assigns UUID for traceability
- Processed responses stored in the dictionary using the assigned UUID as key

**Inference Manager:**
- Capable of managing GPU devices to scale resources
- Distributes inference requests evenly across all graphics cards for efficient processing
- Ensures flexible utilization of GPU resources.

## Appendix E Data Preparations

### E.1 Real-World Propagation Data

**Data Collection:**
- Select 198 propagations from [^25] and [^26]
- Each propagation dataset includes: posting time, post content, and propagation tree with user IDs, repost IDs, and repost times
- Retrieve user profiles and previous posts within three days before source post's posting
- Collect specific time periods for added realism (hour before and two hours following)
- Generate detailed profiles using GPT-3.5 Turbo based on user information and previous posts
- Prompt template: generate character description based on name, username, description, created_at, followers count, following count, and previous_posts

**User Activity Probability Calculation:**
- P_ij = f_ij / max_k(f_kj)
- P_ij is the jth hourly activity probability of user i
- Calculate by dividing the jth hourly activity frequency (f_ij) of user i, by the maximum jth hourly activity frequency across all users in the group (max_k(f_kj))

### E.2 Group Polarization

**User Generation Algorithm**

**Principles**:
- Preserves scale-free nature of social networks
- Combines real user data to create a network of up to one million users

**User Profiles**:
- Acquire population distributions from social network statistics: age, personality traits (MBTI), profession, and social network trends
- Classify professions into 13 categories and social network trends into 9 categories
- Sample from these distributions to generate user backgrounds and social characteristics
- Generate a fictional user profile based on provided personal information: realname, username, bio, persona
- Ensure output can be parsed to JSON

**Social Network**:
- Linking large-scale generated agents into a relationship network is essential
- Matthew effect observes core users (with over 1000 followers) account for 80% of all users
- Derive an initial core-ordinary user attention tree from core users in specific interest areas, constructing the initial relationship network
- Each agent samples two topics of interest and has a probability of following a core user if the topic aligns with the core user, with a probability of 0.1 to prevent excessively dense network and enhance diversity.

### E.3 Herd Effect

**Reddit Experiment User Generation Process**

**Step 1:**
- Demographic information assignment through random sampling: MBTI, age, gender, country, profession

**Step 2:**
- Select topics of interest based on demographic information and provided list: Economics, IT, Culture & Society, General News, Politics, Business, Fun

**Step 3:**
- Generate user profile with fictional background story and detailed interests based on hobbies and profession: real name, username, bio, persona

**Real Data**
- Authentic Reddit comments and post titles from 17 subreddits during March 2023
- Generate contextually relevant post content based on titles and comments
- Categorize content into seven topics: Business, Culture & Society, Economics, Fun, General News, IT, Politics

**Counterfactual Data**
- Utilize all counterfactual information from the dataset to create content for posts
- Examples of counterfactual posts provided in table.

## Appendix F Experiments Details

### F.1 Actions of Different Scenarios

**OASIS Framework: Adjusting Agent Actions for Different Scenarios**

**Adjusting Agent Actions:**
- Variations between scenarios require different actions
- Integrated into OASIS framework for user selection and combination

**Actions in Various Scenarios (Table 17):**

| **Action Type** | **Scenario: Information Spreading** | **Scenario: Group Polarization** | **Comparison with Herd Effect in Humans** | **Counterfactual Herd Effect in Reddit** |
|---|---|---|---|---|
| like post | repost | do nothing | like comment | create comment |
| repost | follow | like post | dislike comment | search users |
| follow | like comment | follow | search posts | trend |
| do nothing | dislike post | dislike post | do nothing | refresh |

**Information Spreading in X:**
- Repost, like post, or follow actions affect information dissemination in various scenarios.

**Group Polarization in X:**
- Different behaviors (repost, like/dislike posts, create comments) impact group polarization dynamics depending on the scenario.

**Comparison with Herd Effect in Humans:**
- Human herd behavior comparisons with OASIS simulations include various actions like commenting, searching users, and refreshing trending topics or posts.

**Counterfactual Herd Effect in Reddit:**
- Create comment instead of like/dislike comment action for understanding group polarization effects on Reddit platform.

### F.2 Information Spreading

**Metrics for Propagation Trends:**
* **Scale**: Number of unique users involved in propagation
* **Depth**: Number of edges connecting a node to the root node (original post)
	+ Overall depth of propagation: greatest depth among all nodes
* **Max Breadth**: Highest number of nodes at any depth during the entire propagation
* Normalized Root Mean Square Error (RMSE) formula:
	+ `Normalized RMSE = √(1/n * ∑_i=1^n (y_simu^i - y_real^i)^2)/y_real^n`
	+ Where:
		- `n`: maximum minute in simulation results
		- `y_simu^i`, `y_simu^i`: value of a certain metric at i-th minute of simulation/real propagation process
		- `y_real^n`: value of the same metric at real propagation process
* **Real Propagations Alignment**:
	+ Set maximum time steps to 50, each representing 3 minutes
	+ Action space: like, repost, follow, do nothing
	+ Compare simulation results with real data for first 150 minutes
* **Simulation Reproducibility**:
	+ Two topics: one with additional noise and another without noise
	+ Ten simulations repeated for each topic
	+ Plotting resulting curves to illustrate discrepancies across ten simulations
* **Time Consumption**:
	+ 26 minutes to run a simulation on one NVIDIA A100-SXM4-80GB
* **Noise Impact**:
	+ Variable results depending on the number of additional posts and prominence of poster.

### F.3 Group Polarization

**F.3.1 Dilemma Questions:**
* Successful writer Helen faces a career-advancing dilemma: write a novel with high potential for failure
* Options: proceed or not
* Evaluate alternatives using following criteria: impact on career, time and effort invested

**F.3.2 Polarization Evaluation Prompts:**
* Determine extremeness of two sentences regarding Helen's situation
* If no difference in tone, mark as "neutral"

**F.3.3 Helpfulness Evaluation Prompts:**
* Evaluate helpfulness and quality of responses provided by AI assistants to user questions
* Criteria for evaluation: accurate information, clarity and comprehensibility, completeness of response, contextual understanding, creative problem-solving, depth of explanation, politeness and professionalism, reference to reliable sources, user engagement.
* [User Question] {question}
* [Answer1] {answer1}
* [Answer2] {answer2}
* [Response Format] Reason: Choice: Answer1 or Answer2.

### F.4 Herd Effect

**Metrics for Assessing Herd Effect:**
- **Post Score**: difference between upvotes (U_i) and downvotes (D_i), can be negative: S_i = U_i - D_i, where S_i is the score of i^th post.
- **Disagree Score**: rating from 1 to 10 indicating agreement/disagreement with counterfactual content based on comments.

**Herd Effect Experiment:**
- Replicated setup of human studies on Reddit, including visibility of comment scores and prohibition of likes/dislikes revocation.
- Controlled user posted content at scheduled intervals to minimize biases from identities and relationships.
- Action space included: like comment, dislike comment, like post, dislike post, search posts, users, trend, refresh, do nothing.
- Each agent had a 0.1 probability of activation and read one sampled post during each time step.
- Experiment conducted over 40 time steps with varying numbers of agents (100, 1k, 10k).
- Agents discussed counterfactual posts, leading to factual correction at the group level as interactions increased.

**Results:**
- Gradual shift towards opposing counterfactual content as interactions increase.
- Example shown in Figure 16 below: agents' comments on a counterfactual post move from surprise and curiosity to partial opposition and complete rejection of the false content.

## Appendix G Misinformation Spreading in One Million Agents
### Technology
**Truth**:
-  **Amazon** expanding **delivery drone program**
-  Aims for **30-minute deliveries** in select cities
-  Goals: Improve **efficiency** and reduce **delivery times**

**Misinformation**:
-  Claim: Amazon to **eliminate all delivery drivers** within 2 years
-  Allegedly due to new drone program
-  Associated hashtags: #Automation #Future

### Education
**Truth**:
-  **Harvard University** announces new **scholarship program**
-  Covers **full tuition** for undergraduates
-  Eligibility: Families earning less than **$75,000** per year

**Misinformation**:
-  False claim: Harvard **raising tuition fees** for all students
-  Alleged despite new scholarship program
-  Hashtag: #EducationCrisis

### Entertainment
**Truth**:
-  Latest **Marvel movie**: "Avengers: Forever"
-  **Broke box office records**
-  Earned over **$1 billion** in opening weekend

**Misinformation**:
-  False claim: Marvel to **retire Avengers franchise**
-  Allegedly will not produce more superhero movies
-  Hashtag: #EndOfAnEra

### Health
**Truth**:
-  Recent study shows benefits of **regular exercise**
-  Significantly **reduces risk** of chronic diseases
-  Examples: **Diabetes** and **heart disease**

**Misinformation**:
-  False claim: Exercise will be **unnecessary** in 5 years
-  Allegation: New treatments will **eliminate chronic diseases**
-  Hashtag: #HealthRevolution

## Appendix H limitations & Future Directions

**RecSys**
- Current recommendation system similar to platforms like X (formerly Twitter) or Reddit
- Designed for semantic similarity based on user profile and recent activity
- More complex algorithms, like collaborative filtering, not implemented in OASIS
- Misalignment between OASIS's performance and real-world propagation data

**User Generation**
- User data obtained through Twitter API or proposed algorithm in OASIS
- Abstraction of individuals leads to a gap between simulator and the real world

**Social Media Platform**
- Expanded action space on social media platforms, but not exhaustive
- Actions like bookmarking, tipping, purchasing, live streaming not supported
- Current simulation operates in text-based environment, no multimodal content
- Future extensions could enhance realism with multimedia support

**Scalable Design**
- Asynchronous design helps avoid bottlenecks but requires several days to complete
- Optimizing inference speed and improving database efficiency crucial for reducing time and cost

**Untapped Potential**
- Platform has potential to serve as foundation for other research
- Can be used to evaluate novel recommendation systems or train large language models with enhanced influence capabilities using agent feedback.

## Appendix I Social Impact and Ethical Considerations

**OASIS Development and Applications:**
- Provides insights into complex social phenomena: information propagation, group polarization, herd effects
- Raises ethical considerations

**Ethical Considerations:**
1. **Replication of real-world social dynamics**:
   - Introduces concerns regarding fidelity and interpretation of results
   - Risk of reinforcing biases in areas related to misinformation or polarization
   - Researchers must be cautious about simulation's influence on public understanding or policy recommendations
2. **Privacy concerns:**
   - Use of real-world data for training agents may introduce risks related to user anonymity and data security
   - Ensuring ethical handling: anonymization, consent
3. **Scalability**:
   - Asset for research but presents potential dangers if misused
   - Large-scale agent-based models could be leveraged for manipulation of online discourse or misinformation campaigns
   - Implement strict governance and ethical guidelines to prevent misuse

