# Do great minds think alike? Investigating Human-AI Complementarity in Question Answering with caimira 

by Maharshi Gor, Hal Daumé III, Tianyi Zhou, Jordan Boyd-Graber
https://arxiv.org/html/2410.06524v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background and Preliminaries](#2-background-and-preliminaries)
  - [2.1 Quizbowl: Where Trivia Nerds Practice](#21-quizbowl-where-trivia-nerds-practice)
  - [2.2 A review of Item Response Theory (irt)](#22-a-review-of-item-response-theory-irt)
- [3 Bootstrapping irt with caimira](#3-bootstrapping-irt-with-caimira)
  - [3.1 Introducing question _relevance_](#31-introducing-question-relevance)
  - [3.2 Zero Centering of _difficulty_](#32-zero-centering-of-difficulty)
  - [3.3 Content-Aware Transformations](#33-content-aware-transformations)
- [4 Experimental Setup](#4-experimental-setup)
  - [4.1 Human Agents](#41-human-agents)
  - [4.2 ai Agents](#42-ai-agents)
  - [4.3 caimira Setup](#43-caimira-setup)
- [5 Question and Agent Analysis](#5-question-and-agent-analysis)
  - [5.1 Latent aspects and Agent skills](#51-latent-aspects-and-agent-skills)
  - [5.2 Which Questions are most difficult?](#52-which-questions-are-most-difficult)
- [6 Related Work](#6-related-work)
- [7 Conclusions](#7-conclusions)
- [8 Limitations](#8-limitations)
- [9 Ethical Considerations](#9-ethical-considerations)
- [Appendix A Quizbowl Dataset](#appendix-a-quizbowl-dataset)
- [Appendix B caimira Setup.](#appendix-b-caimira-setup)
- [Appendix C qa Agents in our study](#appendix-c-qa-agents-in-our-study)
- [Appendix D Question Features for Logistic Regression Study](#appendix-d-question-features-for-logistic-regression-study)
- [Appendix E Question Difficulty](#appendix-e-question-difficulty)

## Abstract

**Recent Advancements of Large Language Models (LLMs)**
- Claims of AI surpassing humans in **Natural Language Processing (NLP) tasks** such as textual understanding and reasoning

**Introduction of CAIMIRA Framework**
- Rooted in **Item Response Theory (IRT)**
- Enables quantitative assessment and comparison of problem-solving abilities of Question-Answering (QA) agents: humans and AI systems

**Findings from Over 300,000 Responses**
- From ~70 AI Systems and 155 Humans across thousands of quiz questions
- Distinct proficiency patterns in **knowledge domains** and **reasoning skills**

**Comparison of Human vs. AI Performance**
- Humans outperform AI in:
    - **Knowledge-grounded abductive reasoning**
    - **Conceptual reasoning**
- State-of-the-art LLMs like GPT-4-Turbo and LLMA-3-70b show superior performance on:
    - Targeted information retrieval
    - Fact-based reasoning, particularly when gaps are well-defined and addressable through pattern matching or data retrieval

**Conclusion and Future Work**
- Need for future QA tasks to focus on questions that challenge higher-order reasoning, scientific thinking, nuanced linguistic interpretation, and cross-contextual knowledge application
- Advance AI developments that better emulate or complement human cognitive abilities in real-world problem-solving.

## 1 Introduction

**The NLTK Community's Focus on Human Behavior Emulation vs. Model Supremacy**
- The NLP community has focused on human behavior emulation: treating human performance as a ceiling for models
- Recent LLMs (large language models) are purportedly acing tests that many humans find challenging
  - Examples include IBM Watson's Jeopardy! performance in 2010
  - A thorough, quantitative examination of the relative strengths and weaknesses of human vs. computer in question answering (QA) remains absent

**Using Item Response Theory (IRT) to Address this Gap**
- IRT is a statistical framework developed in psychometrics for constructing effective standardized tests
  - It models the interaction between individuals and test items (questions)
  - Allows assessment of abilities of respondents (humans and AI systems) and characteristics of test items

**Introducing CAIMIRA: Content-aware, Identifiable, Multidimensional Item Response Analysis**
- Overcomes key challenges of applying IRT to QA
- Uses question text to infer characteristics, enabling generalization to new questions without prior responses
- Applied to responses from 155 human trivia players and a wide range (~70) of QA systems over thousands of carefully crafted QA questions

**Findings from CAIMIRA Analysis**
- Humans and AI systems' skills are strikingly different across latent axes:
  - Humans: superior in interpretative abilities, instinctive thinking, cognitive flexibility, conceptual and knowledge-grounded abductive reasoning
  - LLMs: superior in retrieving specific information about events and locations, extensive parametric memory
- CAIMIRA also reveals questions that challenge most LLMs in extracting the final answer, featuring complex sentence structures and semantic relationships

**Conclusion**
- This study provides insights into strengths and weaknesses of human and AI QA, laying groundwork for future developments better emulating or complementing human cognitive abilities
- It underscores the need for sophisticated benchmarks to distinguish proficient from less capable QA systems in areas demanding deeper understanding.

## 2 Background and Preliminaries

**Source of Quizbowl QA Data**: Section 2.1 ([Link](https://arxiv.org/html/2410.06524v1#S2.SS1))

**Preliminaries of IRT and MIRT**: Section 2.2 ([Link](https://arxiv.org/html/2410.06524v1#S2.SS2))

**Foundation of Caimira**: Section 3 ([Link](https://arxiv.org/html/2410.06524v1#S3))

### 2.1 Quizbowl: Where Trivia Nerds Practice

**Goals:**
- Identify similarities and differences between system and human responses to diverse, challenging questions
- Use depth-testing "probing" questions over information seeking ones
- Focus on Quizbowl dataset for trivia questions with human answers and varying expertise levels

**Background:**
- Significance of research agendas in developing human-like intelligent QA systems (Manchester paradigm)
- ProtoBowl: open source QA dataset based on Quizbowl, a trivia game with sentence-clues and difficulty levels

**Data Source:**
- Quizbowl: questions across various categories from history to literature
- Human players of varying expertise answering questions
- Open source dataset for research purposes.

### 2.2 A review of Item Response Theory (irt)

**Comparing Humans and AI Systems using Item Response Theory (IRT)**

**Item Response Theory (IRT):**
- Framework used to understand question quality and participant strengths
- Widely adopted in psychometrics, medical education, and other fields for developing standardized tests
- Captures skills of subjects (humans or AI systems) based on their responses to a set of questions

**IRT Assumptions:**
1. A set of question-answer pairs
2. Subjects spanning humans and QA systems
3. Binary correctness rulings of their responses
4. Response correctness predicted by the subject's skill level (si) and the question's difficulty (dj)

**Modeling Response Correctness:**
p(Ui,j=1 | si, dj) = σ(si − dj), where σ is the sigmoid function

**Learning Objective:**
- Model skill and difficulty parameters that best fit assumed priors given observed response data
- Use Bayesian inference for fitting models

**Limitations of Existing IRT Applications in NLP:**
1. One-dimensional modeling of item characteristics, assuming a linear hierarchy in difficulty and skill levels
2. Monotonicity assumption restricting the model's ability to distinguish agents in NLP tasks
3. Non-identifiability issues when using hierarchical priors for resolution, making optimization unstable for higher dimensions
4. Dependence on question identifiers hinders generalization and fails to identify differences based on content

**Multidimensional Latent IRT (mirt):**
- Developed to relax the monotonicity assumption and model multi-factor characteristics
- Models two question characteristics: a scalar difficulty dj and an m-dimensional discriminability 𝜶𝒋 that interacts with the m-dimensional skill vector 𝐬𝐢
- Objective: p(Ui,j=1 | 𝐬𝐢, dj, 𝜶𝒋) = σ(𝐬𝐢⊺⁢𝜶𝒋 − dj)
- Discriminability 𝜶𝒋 captures how sensitively the correctness probability changes with each dimension of the agent skill 𝐬𝐢.

## 3 Bootstrapping irt with caimira

**Caimira Workflow**

**Introduction:**
- Figure 3 illustrates the Caimira workflow (refer to caption)
- Predicts probability of agent correctly answering question using model in Eq. ([3](https://arxiv.org/html/2410.06524v1#S3.E3))

**Question Processing:**
- **Raw relevance** 𝐫𝐣': multidimensional, computed from question embedding 𝐄jq and learnt linear transformations
- **Raw difficulty** 𝐝𝐣': multidimensional, raw reference improved by zero centering to address non-identifiability issue

**Caimira Framework:**
- Addresses limitations of Mirt (Section 2.2)
- Introduces:
  - **Relevance** (𝐫𝐣): novel concept for each item j
  - **Zero-centered difficulty** (𝐝𝐣)
  - **Learnable content-aware transformations** (fR and fD) that produce 𝐫𝐣 and 𝐝𝐣 from raw questions

**Response Prediction Model:**
- Probability of agent i correctly answering question j: [Equation 3](https://arxiv.org/html/2410.06524v1#S3.E3)
- Agent skills: **𝐬𝐢** (m-dimensional)
- Question relevance and difficulty: **𝐫𝐣**, **𝐝𝐣** (m-dimensional)

### 3.1 Introducing question _relevance_

**Item Response Analysis in caimira:**
- Includes an item characteristic for each question that captures relevance of latent aspect (𝐫𝐣) towards estimating correct answer likelihood (p⁢(Ui,j))
- Relevance measures alignment between agent skills (𝐬𝐢) and question difficulty (𝐝𝐣) across m-dimensions.
- Proportion assigned to each dimension based on its importance in determining answer probability.
  - Physics knowledge, analytical reasoning given greater relevance for Thermodynamics question.
- Precise and contextually appropriate likelihood estimate obtained through targeted aggregation of differences across relevant dimensions.

**Connection to Topic Models:**
- caimira's questions are admixtures of latent aspects or dimensions with relevance 𝐫𝐣 indicating each dimension’s contribution.
- Similar to per-document allocation in topic models where documents are mixtures of topics.

### 3.2 Zero Centering of _difficulty_ 

**Normalizing Question Difficulty for Comparison**

To overcome the non-identifiability issue of shared skill and difficulty values (𝐬𝐢−𝐝𝐣) due to aggregation across dimensions (Eq [3](https://arxiv.org/html/2410.06524v1#S3.E3)), raw question difficulties are normalized to have a zero mean for each dimension (Equation [7](https://arxiv.org/html/2410.06524v1#S3.E7)). This normalization limits skill and difficulty ranges, making comparisons across dimensions possible.

### 3.3 Content-Aware Transformations

**CAIMIRA: Improving upon MIRT for Question Answering**

**Question Text Mapping**:
- Caimira maps question text into **relevance** and **difficulty** values using learnable functions fR,fD
- Transforms a question qj from the space of question texts Q into raw relevance (𝐫𝐣′) and raw difficulty (𝐝𝐣′) vectors
- Modeled as linear transformations over a pre-trained embedder fE:Q→ℝn (e.g., BERT), which represents qj in an n-dimensional space as an embedding 𝐞𝐣

**Normalization and Transformation**:
- Raw values are normalized to obtain final relevance (𝐫𝐣) and difficulty (𝐝𝐣) values using softmax normalization for relevance
- Normalization ensures that the values sum to 1 across m dimensions, reflecting the relative importance of each latent aspect

**Agent Skills**:
- Caimira learns an agent skill embedding matrix 𝐄a∈ℝna×m
- Agent i's skill vector is the ith row of this matrix (𝐬𝐢=𝐄ia)
- Allows Caimira to learn a compact representation of each agent's skills and question characteristics (difficulty and relevance), across m dimensions, for use in response prediction model

**Learning Objective**:
- To optimize Caimira's parameters (Θ), which include the agent skill embedding matrix 𝐄a and linear transformation parameters fR,fD,
- Use **maximum a posteriori** (MAP) based loss
- Combines cross-entropy loss (ℒCE) with regularization terms (ℒreg):
  - ℒce: Cross-entropy loss between true label x and predicted probability y
  - ∥⋅∥1: ℓ1 norm
  - λd, λs: Regularization hyperparameters
- Total loss: ℒcaimira = ℒce + ℒreg

## 4 Experimental Setup

**Collecting Responses from Humans and QA Systems**

**Protobowl Logs**:
- Collect player logs from "Protobowl" platform over quiz bowl (qb) questions spanning various categories
- Player logs record:
  - Question metadata: category, time taken to answer, answer string, correctness ruling by the platform
- Best players have deep knowledge and excellent lateral thinking skills

**Constructing QA Dataset**:
- qb questions are multi-sentence (5 sentences), with each sentence serving as a distinct clue for the answer
- Each item is formed by cumulatively adding clues from a qb question:
  - First item contains initial clue only
  - Subsequent items incorporate an additional clue each
- Cumulative clue addition provides insight into how progressively revealing information affects agents' response accuracy

**Mapping Player Responses to Cumulative Clues**:
- Player responses are mapped to cumulative clue items to analyze the effectiveness of each clue set in eliciting correct answers
- Responses to a question after the first clue are recorded under its corresponding clue number (e.g., q31_1 for initial clue, q31_2 for second clue, etc.)
- This mapping is refined through a backfilling process:
  - If a player correctly answers at clue t, they are also marked correct for earlier clues (t′>t)
  - If a human answers incorrectly at clue t, they are marked incorrect for later clues (t′<t)
- This results in a total of 3042 entries in the refined dataset.

### 4.1 Human Agents

**Challenges in Exploring QA Abilities of Human and AI**

**Sparse Individual Human Data**:
- Most human players only engage with a few dozen questions
- To address this, synthetic human agents are formed by grouping individual human players

**Purposes of Group Formation**:
1. Accumulate dataset where agents have attempted substantial portion of questions
2. Mitigate issue of non-representativeness from few power users

**Grouped Human Agents**:
- Synthetic construct amalgamating responses from multiple human players with similar skill levels
- Group sizes: 1, 5, 10, and 15, creating 20 human agents spanning 155 distinct players

**Group Formation Mechanism**:
- Human players grouped based on coverage of questions attempted
- In cases of majority rule not reached, a response is sampled based on votes

**Human Participants**:
- All fluent in US English, experienced Quiz Bowl players
- May not encompass full diversity of broader population
- Expertise in trivia games allows contrasting nuanced skill sets with AI capabilities

### 4.2 ai Agents

**Evaluating QA Systems: Differentiating Skills and Models**

**Retrievers**:
- **Context Retrievers**:
    - Use sparse (e.g., bm25) or dense (grit-lm, contriever) methods to fetch k most relevant context documents
    - Evaluated on recall: if answer appears within retrieved documents
- **Title Retrievers**:
    - Predict answer based on title(s) of retrieved document(s)
    - Evaluated on recall: if answer appears in the title

**Large Language Models (LLMs)**:
- Zero-shot in-context learning
- Includes base models (OPT, GPT-Neo, Pythia), instruction-tuned models (OPT-IML, T0, Flan-T5, Flan-UL2), very large scale models (llama-3-70b, Falcon40B, Cohere’s cmd-r+ 9, Mixtral 8x7b), and closed-source APIs (gpt-4o, gpt-4-turbo, Gemini-family)

**Retriever-Augmented Generative Models (RAG)**:
- Combine retrievers with generative models for answer production
- Using FlanT5-XL and exploring Flan-UL2, cmd-r+

**Answer Match Equivalence**:
- Traditional exact match often misses alternative answers with different wording but same semantic sense
- Adopt fuzzy match evaluation using answer aliases: if character level matching rate between predicted and gold answer exceeds certain threshold, prediction is considered correct.

### 4.3 caimira Setup

**Ablation Study on caimira Model**
* Ablation study on varying latent dimensions m:
	+ Performance plateaus beyond m=5 (Figure 4)
	+ Train a 5-dimensional caimira model using all-mpnet-base-v2 and sbert as question embedder fE
	+ Supplement sbert's text input with answer and Wikipedia page summary to capture information gaps between questions and answers
	+ Minimize ℒcaimira using Adam optimizer, learning rate 0.005, batch size 512, λd=λs=1e−5
* Interpretation of latent dimensions:
	+ Use Logistic Regression to predict binary relevance label (𝐫𝐣𝐤>0.6) for each dimension k
	+ Topical categories and linguistic properties used for question features
	+ Report classification accuracy and significant features (Figure 5)
	+ Demonstrates efficacy of predicting relevance from sbert embedding
* Distribution of skills across agent types:
	+ Interpretations given in Figure 5
	+ Red dashed line indicates mean _effective difficulty_ of each dimension (Equation 13)

**Interpreting Latent Aspects using Logistic Regression**
* Use Logistic Regression as an interpretative tool to study latent dimensions of caimira
* Build upon Benedetto et al. for item difficulty parameters and Gor et al. for relevance dimensions
* For each latent dimension (k), logistic regression predicts if 𝐫𝐣𝐤>0.6 as a function of interpretable features from questions
* Features span topical question subcategories, clue counts, temporal expression mentions, Wikipedia match scores, and linguistic features (normalized to have zero mean and unit variance)
* Most contributing, statistically significant features for each dimension listed in Figure 5
* Maintain random guess accuracy at 50% to make learned coefficients comparable across dimensions.

## 5 Question and Agent Analysis

**Interpretation of Caimira's Latent Aspects**
- Analyzing caimira's underlying characteristics that differentiate agent skills
- Examining patterns in question difficulty and agent performance

### 5.1 Latent aspects and Agent skills

**caimira's Latent Aspects and Their Corresponding Skills:**
* **Abductive Recall**: Bridges indirect clues and vague references to formulate information gaps and recall specific entities to fill the gap (Bhagavatula et al., 2019; Shi et al., 2024). Requires multi-hop reasoning and deduction. Humans excel at this aspect due to intuition, while AI models struggle.
* **History and Events**: Involves historically grounded questions with clearer information gaps that challenge participants to synthesize multiple pieces of information and infer connections between events (Example in Fig 3). Bigger LLMs excel in this category as they have effective recall and application of historical information through their parametric memory.
* **Scientific Facts**: Focuses on domain-specific conceptual knowledge, often featuring questions from scientific domains (Figure 7). Retrieval-based systems fare well when allowed to retrieve sufficient documents, but instruction-tuned LLMs outperform base models in this category. Humans and large-scale LLMs excel in scientific facts as do closed-source systems like gpt-4-turbo.
* **Cultural Records**: Represents questions focusing on prominent figures such as authors, composers, artists, and leaders, testing direct knowledge recall of well-known facts (Figure 8). These questions are relatively easy and accessible due to high WikiMatchScore.
* **Complex Semantics**: Pertains to questions about popular events with complex semantic relationships and detailed sentences (Figure 9). Despite their intricacy, they are particularly retriever-friendly due to high WikiMatchScores, enabling retrievers to locate correct documents. However, agents without retrieval abilities or large parametric memories struggle in this aspect.

### 5.2 Which Questions are most difficult?

**Analyzing Question Difficulty using Effective Difficulty Metric (𝐝𝐣,𝐤)**
* **Effective difficulty**: 𝐝𝐣(𝐞) represents contribution of latent aspect k to question j's difficulty
* Calculated as 𝐫𝐣,𝐤⁢𝐝𝐣,𝐤 according to Equation [3] in document
* **Clustering questions**: KMeans on effective difficulty vectors (5-dimensional)
* **Question analysis**: Mean relevance and mean effective difficulty per cluster
* **Difficulty groups**: _Abduction_ (V.Hard) and _Mixed Bag_ identified as most challenging
	+ Complex semantics, indirect phrasing, single clue
	+ Human accuracy significantly higher than ai systems like GPT-4-Turbo
* **Observed trends**: Fewer clues and lower WikiMatchScore lead to higher difficulty questions.

**Question Difficulty Analysis using caimira:**
* Mean relevance 𝐫𝐣,𝐤 and mean effective difficulty 𝐝𝐃,μ𝐤(𝐞) per cluster on each dimension (Figure 10)
* _Abduction_ (V.Hard) and _Mixed Bag_ emerge as hardest categories for human-AI complementarity in question answering.

## 6 Related Work

**NLP Evaluation Methods: Ideal Point Models (IPM) and Item Response Theory (IRT)**

**Background:**
- Current QA evaluation paradigms lack segmentation of datasets for assessment of relative differences between items
- Lalor et al. (2019) propose IRT as a framework to address this issue in NLP
- Rodriguez et al. argue for its adoption as standard, demonstrating benefits
- Byrd and Srivastava use IRT to estimate question difficulty and model skills

**IRT vs IPM:**
- **IRT**: One-dimensional traits, used in educational assessments (ability gauging)
- **IPM**: Multi-dimensional, used in political science (position evaluation on spectra)
- Both employ probabilistic methods for binary outcomes estimation

**Advancements and Applications:**
- Research focuses on augmenting human skills with language models using AI in creative writing and QA
- Collaborative approaches: human writers use GPT-3 for suggestions or modifying user-selected text spans
- Teamwork between experts, novices, and AI for trivia and information retrieval
- Our approach aims to identify latent factors highlighting distinct capabilities of humans and AI in QA assessment.

## 7 Conclusions

**Comparison of AI Systems to Humans:**

**CAIMIRA**:
- Enables discovery and interpretation of latent aspects in QA datasets
- Highlights skills of various QA agents

**Disparities between AI and Human Performance**:
- **GPT-4-Turbo** and **Gemini Pro**: Excel at direct, context-rich queries
  - Connecting events and figures
- Struggle with indirectly phrased questions lacking explicit entity references
- **Human Acumen**: Shines in these areas

**Assessing AI Abilities**:
- GPT-4-Turbo's performance on complex, knowledge-intensive tasks does not indicate superhuman abilities
  - Training data and publicly available quiz questions are potential factors
- Future research needed to:
  - Develop stronger evaluations for gauging AI systems' understanding of implicit contexts
  - Systematically contrast their skills with those of humans

**Implications and Further Research**:
- Opens up new avenues for research on estimating agent skills in multi-agent systems and collaborations
- Becomes crucial as NLP evolves toward conversational agents and real-world problem solving.

## 8 Limitations

**Dataset and Task Limitations**
- **Limited language diversity**: English-only dataset restricts generalizability to other languages
- **Lack of diverse task types**: Rely solely on trivia-based questions, lacking non-trivia datasets with human responses in competitive settings
- **Absence of multilingual trivia benchmarks**: Lack multilingual trivia datasets with human responses and performance benchmarks

**Challenges in Interpreting Near-Perfect Scores**
- Gpt-4-turbo's near-perfect performance on complex tasks may not necessarily indicate superhuman reasoning:
  - Protobowl questions have been public since 2011
  - Full training data for gpt-4-turbo is unknown
- Need for more robust evaluation methods to accurately assess AI systems' understanding and reasoning abilities compared to humans

**Lack of Information on Specific Human Players**
- Limited access to information about specific human players due to the nature of Protobowl platform
- Future work can focus on collecting such information while hiding user identity

**Non-Extensibility of a Trained Model to a New AI System**
- Caimira's performance is not extensible to new agent without retraining the model:
  - Requires having a feature set for human players as well, which is not available at the moment

**Static Representation from S Bert**
- Use static dense representation of question text from S Bert instead of finetuning the model to adapt to Caimira's objective:
  - Future work can explore using parameter efficient fine-tuning (PEFT) for better results.

## 9 Ethical Considerations

**Ethical Guidelines**
- Respected privacy and obtained informed consent from human participants
- Anonymized their data to comply with ethical standards
- Committed to ethical research practices in advancing NLP technologies

**Tools Used**
- Utilized GitHub Copilot for low-level coding and writing assistance (plotting codes, prose editing)

**Ethical Considerations**
- Acknowledged the carbon footprint of training and running large-scale language models
- Trained a small model with 25000 parameters for 20 minutes on a single A4000 GPU
- Used a pre-trained SBERT model to encode question text

## Appendix A Quizbowl Dataset

**Quizbowl: Incremental Question Answering**

**Description**:
- Trivia game with decreasing difficulty clues leading to a "giveaway" hint
- Sequence of clues reveals more information and helps disambiguate references (Figure 11)

**Example Quizbowl Questions**:
1. Religion: **q832_5**
   - Written by Sahabas after leader's death
   - Clarification of meaning and significance: tafsir
   - Hundred and fourteen chapters called suras (recite)
   - Sacred text for Muslims: **Answer:** Quran / Koran
2. Music: **q622_3**
   - Commissioned by Paul Wittgenstein
   - Invented by Bartolomeo Cristofori
   - Improvement over clavichord and harpsichord: Piano / Pianoforte
3. Mathematics: **q2443_1**
   - Equals a number (4 times infinite sum)
   - Answer: pi / 3.14 / π

**Characteristics**:
- Discriminates players' skills with early answers being better
- Opposed to "all or nothing" question answering
- Multiple opportunities for agents to answer a question by creating entries per clue (q622_1, q622_2, etc.)

## Appendix B caimira Setup.

**Caimira Learning Objective**

**Revised caimira objective**:
- `p(Ui,j=1|𝐬𝐢,𝐫𝐣,𝐝𝐣)=σ((𝐬𝐢−𝐝𝐣)⊺𝐫𝐣)`
- Where:
  - `𝐬𝐢`: Agent skills (`m` dimensions)
  - `𝐫𝐣, 𝐝𝐣`: Question relevance and difficulty (`m` dimensions each)
  - `𝐝𝐢` and `𝐫𝐣` are functions of question representation `𝐄jq` defined as:
    - `𝐫𝐣′ = 𝐖R⁢𝐄jq+𝐛R`
    - `𝐝𝐣′ = 𝐖D⁢𝐄jq`
    - `𝐫𝐣 = softmax(𝐫𝐣′)`
    - `𝐝𝐣 = 𝐝𝐣′−1nq⋅∑j=1nq𝐝𝐣′`
- These, along with the embedding matrix `𝐄a` of agent skills (`𝐬𝐢=𝐄ia`) are the parameters trained for caimira using a regularized cross entropy objective.

**Hyperparameters**:
- Mini-batch stochastic gradient descent to minimize `ℒcaimira` (Equation 11)
- `λd` and `λs` set to 1e−5
- Adam optimizer without weight decay, learning rate of 0.005
- Batch size of 512

## Appendix C qa Agents in our study

**Agents Used in the Study**

**Retrievers**:
- Dense retriever: Contriever, pretrained on Wikipedia and CCNet data, fine-tuned on MS-MARCO
- Sparse retriever: BM25 algorithm with Anserini's implementation
- Title-retriever: Assumes the document title is the query answer

**Evaluation**:
- Retrievers evaluated on recall-based accuracy
- Retriever scores a point if the answer appears within the top k documents for context retrievers, or in the title of the top k documents for title-retriever

**Large Language Models (LLMs)**:
- Base Models: OPT, GPT-Neo, Pythia (unsupervised CausalLM objective)
- Benchmark Instruction Tuned Models: OPT-IML, T0, T0pp, Flan-T5, Flan-UL2
- Very Large-Scaled Models: Llama-2 (70 billion parameters), Falcon (40 billion parameters), their instruction tuned variants

**Retriever-Augmented Generative Models (RAG)**:
- Retrieve Wikipedia documents relevant to the questions using dense and sparse retrievers
- Use a generator model, such as FlanT5-XL or Flan-UL2, for short answer generation

**Evaluation**:
- Adopt a fuzzy match evaluation using multiple-answer aliases to better handle alternative answers with different wordings but the same semantic meaning.

## Appendix D Question Features for Logistic Regression Study

**Logistic Regression Study Features**

**Question Category Features**:
- Binary features indicating question category
- Categories: fine arts, cultural geography, geography, physical/technical geography, ancient history, history, cultural/exploration history, military history, scientific history, social history, language, author and works, literature, genre and style, literary terms, plot and characters, music, mythology, political events, politics, political figures, institutions, theory, religion, astronomy, science, biology, chemistry, earth science, materials, mathematics, physics, other

**Linguistic Features**:
- **LingFeat**: Python research package for linguistic feature extraction
- Extracts 255 linguistic features across 5 broad branches:
  - Advanced Semantic (AdSem): Measures meaning complexity
  - Semantic Richness, Noise, and Clarity: From trained LDA models
  - Discourse (Disco): Measures coherence and cohesion
  - Syntactic (Synta): Evaluates grammar and structure
  - Lexico Semantic (LxSem): Measures word/phrasal-specific difficulty
  - Shallow Traditional (ShTra): Assesses text difficulty through basic counts and formulas

**Time-Based Features**:
- **t_range**: 1 if question is about certain time period or range, 0 otherwise
- **t_range**: 1 if question refers to an event related to another event, 0 otherwise

**Other Features**:
- **o_TRASH**: 1 if question inquires about specific pop culture events, 0 otherwise
- **o_Records**: 1 if question mentions superlative forms (e.g., "most recent"), 0 otherwise

## Appendix E Question Difficulty

**Heatmaps of Question Clusters Across Latent Factors**
- **Mean relevance 𝐫𝐣**,𝐤: heatmap showing the average relevance of question clusters across five latent factors (k) [Figure 19]
- **Mean effective difficulty 𝐝𝐃,μ𝐤(𝐞)** : heatmap showing the mean effective difficulty of question clusters across five latent factors (k) [Figure 19]

**Agent Accuracies Across All Question Clusters**
- Figure 20: full set of agent accuracies for all question clusters defined in Figure 19.

**Color Scheme and Figures**
- Same color scheme as in **Figure 9**: Complex Semantics, Latent aspects and Agent skills [Figure 9]
- Examples of questions from different clusters provided in Figures 21-31.

