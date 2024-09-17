# WEBLINX: Real-World Website Navigation with Multi-Turn Dialogue

by Xing Han Lù, Zdeněk Kasner, Siva Reddy

[Code, data, and models available for research](https://mcgill-nlp.github.io/weblinx)

## Abstract
**Overview**:
- Proposed problem: conversational web navigation
- Digital agent controls web browser
- Follows user instructions to solve real-world tasks in multi-turn dialogue fashion

**Benchmark**:
- Introducing WEBLINX: large-scale benchmark of 100K interactions across 2300 expert demonstrations
- Covers broad range of patterns on over 150 real-world websites
- Can be used to train and evaluate agents in diverse scenarios

**Challenges**:
- **Large Language Models (LLMs) cannot process entire web pages in real-time**
- To solve this bottleneck, they propose a retrieval-inspired model that efficiently prunes HTML pages by ranking relevant elements

**Evaluation**:
- Use selected elements, screenshots, and action history to assess various models for their ability to replicate human behavior in web navigation
- Experiments with small text-only to proprietary multimodal LLMs
- Findings: smaller finetuned decoders surpass zero-shot LLMs (including GPT-4V), but all finetuned models struggle to generalize to unseen websites

## 1 Introduction

**Conversational Web Navigation: Real-World Problem and Benchmark Introduction**

**Background:**
- Conversational assistants can navigate websites through plugins (OpenAI, 2023d)
- Limitations: plugins must be developed for each website, may not cover all functionality

**Research Question:**
- Can models behind conversational assistants navigate websites directly in the user's browser?

**Conversational Web Navigation Problem Definition:**
- Given initial instruction, an agent must complete a real-world task inside a web browser while communicating with the user via multi-turn dialogue

**Relevance:**
- Enhances smart speakers and digital assistants with voice-controlled web navigation
- Improves productivity of knowledge workers by reducing repetitive steps

**WEBLINX: First Benchmark for Conversational Web Navigation (Table 1)**
| Columns       | Description                                |
|------------------|------------------------------------------|
| Chat          | Use of multi-turn dialogue               |
| Gener.        | If tasks are general or specialized     |
| Browse        | Use of a web browser                    |
| # Dom.        | Number of app/website domains             |
| # Inst.       | Number of instances                      |
| Avg. # El.    | Average number of HTML elements per page |
| Avg. # Turns  | Average number of turns per instance       |

**Unique Aspects:**
- First large-scale benchmark for conversational web navigation
- Evaluates agents' generality to realistic scenarios, including new websites and categories

**Methods: Dense Markup Ranking (§5.1) and Evaluation Metrics (§4)**

**Conclusions:**
- Existing methods may struggle with large DOMs and generalizing to new settings
- Significant effort needed for progress in conversational web navigation.

## 2 Related Work

**Related Work and Background**

**Web Navigation Agents**:
- Previous work focused on building web agents for a single task (e.g., MiniWoB++)
- Reinforcement learning approaches reached human-level performance in simulated environments (Liu et al., 2018; Humphreys et al., 2022)
- Limited transferability to realistic settings despite environment extensions and sample-efficient methods (Gur et al., 2021; Kim et al., 2023)
- Other works explored language commands, question answering on Wikipedia, or iterative tool resolution for crowdsource platforms (Pasupat et al., 2018; Li et al., 2020; Burns et al., 2022; Xu et al., 2021; 2024)
- WebShop: e-commerce environment with over 12K human-written task instructions (Yao et al., 2022)
- LLM-based navigation services like Adept, Multi-On, and HyperWrite (Nakano et al., 2021; 2023; 2023)
- Large-scale resources for autonomous navigation agents: VisualWebArena, WebArena, WEBLINX (Koh et al., 2024; Zhou et al., 2023; Furuta et al., 2023)

**Website Representations**:
- Efficiently representing real-world websites is a long-standing challenge in web understanding (Wu et al., 2023)
- Approaches for simplifying or compressing textual representation of the website include rule-based algorithms, accessibility tree representations, graph embeddings, and model-based approaches (Zhou et al., 2021; Asouel et al., 2023; Wang et al., 2022; Deng et al., 2022; Aghajanyan et al., 2022; Gur et al., 2024)
- Previous works for visual information of the web page rely on feature extraction (Liu et al., 2010; Cormer et al., 2017)
- Dense markup ranker to select relevant DOM elements and optionally combine with high-resolution browser screenshots (Deng et al., 2023)

**Conversational Interfaces**:
- Conversational interfaces are the basis of task-oriented dialogue (Chen et al., 2017; Zhang et al., 2020b)
- End-to-end solutions show promising results, but use of LLMs remains under scrutiny (Hudeˇcek & Du ˇsek, 2023)
- Dialog2API: interface for interacting with API-based services (Shu et al., 2022)
- META-GUI: dataset focused on automating actions in mobile apps rather than general websites (Sun et al., 2022)
- RUSS: first dialogue-centric dataset designed to support services through annotated demonstrations (Xu et al., 2021)
- WEBLINX: covers a wide range of real-world tasks with longer demonstrations due to dynamic topic switching (Adlakha et al., 2022).

## 3 WEBLINX

**Weblinx Benchmark Overview**
- **WEBLINX**: large-scale conversational web navigation benchmark with 2337 demonstrations and an average of 43 turns
- Contains interactions between a human user (instructor) and human assistant (navigator) on 155 real-world websites in 8 categories and 50 subcategories

**Action Space**
- **Actions**: click, load URL, say, submit form, text input
- Detailed description of each action in Table 3

**Dataset Statistics**
- Total demonstrations: ~2300
- Breakdown by category and split: Figure 2
- Additional statistics in Appendix A.1 and A.2

**Demonstration Framework**
- Recorded real-time interactions between instructor and navigator
- Each demonstration (D): sequence of states (s) and actions (a)
- State (st): representation of website, elements, screenshot, etc.
- Model (m) predicts action based on state and prompt template

**Data Collection**
- Professional data labeling company with 8 expert annotators
- Instructor interacts with navigator in a web browser
- App and processing pipeline record demonstrations
- Validation by different annotator under original navigator's supervision

**Evaluation Splits**
- TRAIN split for training model
- VALID and TEST IID: assess in-domain generalization
- 4 out-of-domain splits for various scenarios

**Representing Actions and States for Modeling**
- State (st): contains current DOM tree, screenshot, utterance, viewport size, interaction history
- Model (m) predicts action based on state and prompt template
- Interaction history: set of past five actions and utterances

**Parsing Action Output**
- Action consists of intent and argument in textual format
- Follows predefined structure for parsing into a structured form
- Can be executed using tools like Selenium.

## 4 Evaluation Framework

**Evaluation Framework**

**Metrics**:
- **Task success rate**: Measures the proportion of demonstrations where the model reached the desired final state (not applicable here due to evolving objective)
- **Intent Match (IM)**: Indicates if the predicted action's intent matches the reference's intent: `1` for a match, `0` otherwise
- **Element Similarity using IoU**: Computes intersection over union (IoU) between bounding boxes of reference and predicted elements, favoring high visual overlap and penalizing large/small discrepancies
- **Text Similarity using F1**: Calculates character n-gram matches (default `n=6`) between text arguments, scaling by the intent match score
- **URLF**: Applied to load intent URLs with consistently segmentable structures

**Turn-level and Overall Scores**:
- **Element group (EG)**: Includes click, textinput, and submit; evaluated using IoU
- **Text group (TG)**: Encompasses load, say, and textinput; evaluated using F1 score
- Turn-level score: Determined by intent match and element overlap for EG actions or text similarity for TG actions. Micro-averaged to compute overall score.

## 5 Methods

**Methods for Selecting Candidate Elements and Modeling Actions**

**Candidate Selection:**
- **Dense Markup Ranking (DMR)** proposed as an efficient alternative to previous methods
  * Simplified element representation to reduce computational overhead
  * Dual encoder-based approach
  * Similarity-based learning between text and HTML elements
* Faster than previous methods, but slightly lower recall
* Reduces processing time for real-time interactions

**Input Representation:**
- **Truncation strategy**: leverages hierarchical nature of input to determine which subsection to truncate
- Includes full HTML attributes, viewport size, XML Path, and bounding boxes of candidate elements

**Modeling Actions:**
- Combine most promising candidates with remaining information for predicting action strings
- Examine 19 models (zero-shot and finetuned) with different input modalities: image-only, text-only, and both
- Categorize action models by input modality: text-only, image-to-text, multimodal

**Text-Only Models:**
- MindAct (Flan-T5 finetuned on WEBLINX)
- LLaMA-2 and Sheared-LLaMA
- GPT-3.5 Turbo (zero-shot and finetuned)
- GPT-4T (zero-shot)

**Image-to-Text Modeling:**
- Pix2Act: encoder-decoder model purely finetuned on pixels using Pix2Struct backbone

**Multimodal Models:**
- Fuyu-8B (base model pretrained on browser screenshots)
- GPT-4V (OpenAI's variant with vision capabilities)

## 6 Experimental Results

**Experimental Results**
- Report of results from Section 5 experiments on groups defined in Section 4.2
- Aggregated results for 11 models presented in Table 4
- Discussion of:
  - **MindAct** vs. **Flan-T5 finetuned using DMR-based input representation** (§5.1)
    * MindAct trails behind Flan-T5 due to lack of exposure to multi-turn dialogue
    * Flan-T5 never trained on any navigation actions
    - Important role of **DMR-based representation** in achieving better performance
  - **LLaMa-based models** vs. **Flan-T5 and MindAct**
    * Outperform both Flan-T5 and MindAct despite Sheared-LLaMa being smaller than Flan-T5
    * May be due to high quality training on a large number of instruction-following tasks compared to Flan-T5
    - Equal performance between **Sheared-LLaMa** and LLaMA-2 13B is intriguing
  - **Image-to-text vs. multimodal models**: **Pix2Act (1.3B param.)** vs. **Fuyu-8B**
    * Fuyu outperforms Pix2Act due to ability to receive text as input and greater parameter count
    - Trails behind Pix2Act for intent matching and text prediction
  - **Comparison of multimodal with chat-based models**: **Fuyu-8B** vs. **LLaMA chat-based text-only models**
    * Chat-based LLaMA models outperform Fuyu-8B, indicating that multimodal models fine-tuned on screenshots are still behind chat-based models optimized for instruction-based finetuning
  - **Comparison with proprietary models**: **GPT-3.5T** and **GPT-4T** (zero-shot) vs. **LLaMA-2** (finetuned)
    * Proprietary models outperform LLaMA-2 in zero-shot setting, but GPT-3.5F is outperformed by Sheared-LLaMA and LLaMA-2 when finetuned
    - Cause for GPT-3.5F's underperformance is unclear due to limited access to hyperparameters
    * Similar performance between **GPT-4V** and **GPT-4T**, suggesting existing multimodal models may not effectively use screenshots for predicting actions
  - **Generalization capabilities**: Comparison of TEST OOD vs. TEST IID results highlights weaknesses of fine-tuned models in generalizing to unseen websites
    * **LLaMa-13B** achieves poor results on TEST CAT, indicating difficulty with new subcategories

**Qualitative Assessment**
- Examination of two models: **GPT-4V** and **LLaMA-2-13B** (finetuned) to understand performance gap between zero-shot and finetuned models
- Focus on scenarios where models make poor predictions despite correctly predicted intents: click , textinput, say
  - **Click**: GPT-4V selects incorrect tabs or less optimal options; LLaMA-2 can still fail by clicking on irrelevant elements
  - **Textinput**: GPT-4V writes email subject instead of title, shares irrelevant links; LLaMA-2 may attempt to click instead of textinput and omit titles
  - **Say**: Different writing styles between GPT-4V and LLaMA-2; LLaMA-2 provides unhelpful responses by sharing irrelevant links

## 7 Discussion 
**Experimental Findings**
- Larger multimodal models surpass smaller image-only models when finetuned but lag behind text-only models
- DMR-based representation leads to better performance for both finetuned and zero-shot models
- Text-only decoders perform closely with smaller variants on out-of-domain splits, while zero-shot models are consistently surpassed by their finetuned counterparts
- Qualitative assessments show that best zero-shot models can make simple and unjustified errors

**Limitations**
- Benchmark contains static demonstrations, limiting evaluation of alternative trajectories
- Architectures have inherent limitations, such as text-only models not being able to draw or describe images

**Conclusion**
- Introduced WEBLINX, a large-scale expert-built benchmark for conversational web navigation on real-world websites
- Evaluated finetuned and zero-shot models with various modalities and found that chat-based decoder models achieve the best results but still struggle to generalize
- Suggest future directions: designing multimodal architectures, evaluating models in wider ranges of scenarios, expanding beyond browser tasks, leveraging reward-based methods, and alternative training approaches.

