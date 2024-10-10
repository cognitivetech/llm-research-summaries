# ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

by Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong, Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu and Ge Yu
https://arxiv.org/html/2402.13547v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Methodology](#3-methodology)
  - [3.1 Preliminary of Retrieval-Augmented Generation (RAG) Models](#31-preliminary-of-retrieval-augmented-generation-rag-models)
  - [3.2 ActiveRAG: RAG with Active Knowledge Learning](#32-activerag-rag-with-active-knowledge-learning)
  - [3.3 Knowledge Construction from Different Learning Views](#33-knowledge-construction-from-different-learning-views)
- [4 Experimental Methodology](#4-experimental-methodology)
- [5 Evaluation Result](#5-evaluation-result)
  - [5.1 Overall Performance](#51-overall-performance)
  - [5.2 Ablation Study](#52-ablation-study)
  - [5.3 Characteristics of Different Knowledge Construction Mechanisms](#53-characteristics-of-different-knowledge-construction-mechanisms)
  - [5.4 Generalization Ability of Knowledge Construction Outcomes](#54-generalization-ability-of-knowledge-construction-outcomes)
  - [5.5 Case Studies](#55-case-studies)
- [6 Conclusion](#6-conclusion)
- [A.1 License](#a1-license)
    - [Restoring Social Engagement through Cranial Nerve Techniques](#restoring-social-engagement-through-cranial-nerve-techniques)
    - [Finding Marie-Hélène Aubert's Birthplace: A Case Study in ActiveRAG](#finding-marie-hélène-auberts-birthplace-a-case-study-in-activerag)
    - [ActiveRAG: Finding "The Woods" Producer Using Cognitive Nexus](#activerag-finding-the-woods-producer-using-cognitive-nexus)
- [A.2 Prompts Used in ActiveRAG](#a2-prompts-used-in-activerag)
- [A.3 Prompts Used in Baseline Models](#a3-prompts-used-in-baseline-models)
- [A.4 Additional Case Studies](#a4-additional-case-studies)

## Abstract

**Retrieval Augmented Generation (RAG)**
- Introduces a new paradigm for Large Language Models (LLMs)
- Aids in resolution of knowledge-intensive tasks

**Current RAG models**:
- Position LLMs as passive knowledge receptors
- Restricts their capacity for learning and comprehending external knowledge

**ActiveRAG**:
- Innovative RAG framework
- Shifts from passive knowledge acquisition to active learning mechanism
- Uses **Knowledge Construction mechanism**:
  - Develops a deeper understanding of external knowledge
  - Associates it with previously acquired or memorized knowledge
- Utilizes **Cognitive Nexus mechanism**:
  - Incorporates outcomes from both chains of thought and knowledge construction
  - Calibrates the intrinsic cognition of LLMs

**Experimental results**:
- ActiveRAG surpasses previous RAG models
- Achieves a 5% improvement on question-answering datasets

**Availability**:
- All data and codes available at [https://github.com/OpenMatch/ActiveRAG](https://github.com/OpenMatch/ActiveRAG)

## 1 Introduction

**Constructivism and Retrieval-Augmented Generation (RAG) Models:**

**Background:**
- Learners construct new understandings and knowledge through experience and social discourse
- Large Language Models (LLMs) have strong emergent abilities but suffer from hallucination problem and outdated parametric memories

**Retrieval-Augmented Generation (RAG) Models:**
- Retrieve knowledge from external corpus, build brute-force RAG architecture by feeding passages to LLMs
- Effectiveness influenced by noise from retrieved knowledge
- Recent research focuses on mitigating noise through self-reflection and self-refining: filter out irrelevant passages or conduct summarized noting to extract key point knowledge

**Limitations of Passive Learning:**
- Violates Constructivism, a philosophy theory of education that emphasizes active learning
- Neglects the nature of active learning, which builds associations between external knowledge and prior learned knowledge

**ActiveRAG:**
- Leverages active learning to augment knowledge comprehension by bridging the gap between prior knowledge and retrieved information
- Enhances active learning capability of LLMs without further fine-tuning
- Three-step pipeline: Retrieval, Knowledge Construction, Cognition Nexus

**ActiveRAG vs Vanilla/Chain-of-Note RAG Models:**
- Inaccurate answers produced by vanilla and Chain-of-Note models due to limitations of passive knowledge acquisition
- ActiveRAG leverages active learning to enable LLMs to produce accurate answers.

**Experiments:**
- Evaluation results demonstrate over a 5% improvement compared to baseline models across various question answering datasets
- Robustness maintained across different datasets and varying numbers of retrieved passages
- Knowledge construction outcomes can be generalized to different LLM architectures, aiding them in leveraging external knowledge.

## 2 Related Work

**Retrieval-Augmented Generation (RAG) Models**

**Background:**
- RAG models aim to retrieve external knowledge and enhance language models
- Effective in various NLP tasks: question answering, dialog understanding, code generation, etc.

**Components:**
- Dense retrievers as knowledge-seeking modules
- Language models generate results based on retrieved knowledge

**Early RAG Models:**
- Optimize models to leverage external knowledge
- Develop architecture of generator to use external knowledge fully
- Train more accurate retriever using generation model feedback
- Jointly train retriever and generator

**Advancements with Large Language Models (LLMs):**
- Leverage external knowledge to reduce generation perplexity in NLP tasks
- Alleviate hallucination problem of LLMs by updating outdated or long-tail knowledge

**Challenges:**
- Noise from retrieved contexts can affect effectiveness of RAG models

**Approaches to Mitigate Noise:**
- Eliminate irrelevant contexts: natural language inference models, summarization model, conditional cross-mutual information
- Self-RAG: LLMs filter out irrelevant passages via self-reflection

**Recent Work:**
- Adaptively retrieve passages or build noting mechanism to summarize key point knowledge from retrieved passages.

**Constructivism Theory:**
- Associates external knowledge with prior knowledge, which is often overlooked by LLMs.

## 3 Methodology

**Introduction to ActiveRAG Method**

We introduce our ActiveRAG method, which prompts Language Models (LLMs) to actively read, understand, and use external knowledge for generation. The process begins by introducing Retrieval-Augmented Generation (RAG) models [3.1](https://arxiv.org/html/2402.13547v1#S3.SS1). Subsequently, we describe our ActiveRAG method that encourages LLMs to learn actively from retrieved passages [3.2](https://arxiv.org/html/2402.13547v1#S3.SS2).

### 3.1 Preliminary of Retrieval-Augmented Generation (RAG) Models

**Retrieval-Augmented Generation (RAG)**

**Existing RAG Models**:
- Guu et al. ([2020](https://arxiv.org/html/2402.13547v1#bib.bib8)) and Izacard et al. ([2023](https://arxiv.org/html/2402.13547v1#bib.bib13}) utilize retrieval models to search passages **D={d1,…,dn}** and enhance the generation ability of language models by grounding these retrieved passages.

**Vanilla RAG**:
- Earlier RAG models usually employ the **retrieval-generation architecture**, which directly feeds the retrieval passages **D={d1,…,dn}** to language models to generate the answers for the given query **q**.
- These retrieved passages contain noise, and the **brute-force RAG approach** tends to constrain the benefits of RAG modeling.
- It has sparked discussions on augmenting LLMs using the retrieval results or model-generated outputs.

**RAG with Self-Refining**:
- Recent work employs the **retrieval-refining-generation architecture** to empower the capability of RAG using LLMs.
- **Self-RAG** and **Chain-of-Note** are two representative methods:
  - **Self-RAG**: Focuses on finetuning LLMs to adaptively retrieve passages on-demand and controlling the information flows from retrievers to generators.
  - **Chain-of-Note**: Designs the instructions to self-refine the retrieved passages and forms the query-focused summarization, which can be applied to black-box LLMs like GPT-3.5.
- However, they neglect the nature of active learning, where the learner actively constructs their understanding and knowledge by integrating new information with prior knowledge.

### 3.2 ActiveRAG: RAG with Active Knowledge Learning

**ActiveRAG: Revealing the Treasures of Knowledge via Active Learning**

**Architecture**:
- ActiveRAG builds on constructivism learning theory to teach LLMs to actively acquire knowledge
- Three-step pipeline: retrieval, **knowledge construction**, and cognitive nexus

**Knowledge Construction**:
- Regards LLM as a cognitive structure for receiving, understanding, and transforming external knowledge from passages
- Constructs four distinct agents for knowledge acquisition:
  - Anchoring
  - Logical reasoning
  - Cognition
  - Association
- Details described in Sec. 3.3 of the paper

**Cognitive Nexus**:
- Employs the cognitive nexus mechanism to assist LLMs in utilizing external knowledge for augmentation
- Facilitates the fusion of constructed knowledge understanding with intrinsic cognitive processes of LLMs
- Prompts LLMs to generate an initial chain-of-thought, then integrates outcomes of knowledge construction into the chain-of-thought
- Emphasizes incorporating knowledge construction into self-aware cognitive reasoning

### 3.3 Knowledge Construction from Different Learning Views

**Constructivism Learning Theory**
- **Active process**: learning involves construction of internal mental representations
- Encompasses both structural knowledge and extensive non-structural experiential backgrounds (Piaget [1970])

**Prompting LLMs for Active Learning:**
- **Semantic Association**: integrates familiar information, consolidates foundational and advanced pieces
  - Expands model's cognitive boundaries
  - Deepens understanding of query and knowledge
- **Epistemic Anchoring**: represents previously unknown knowledge
  - Establishes foundational understanding
  - Incorporates new concepts relevant to the question
- **Logical Reasoning**: draws logical conclusions, refines problem-solving capabilities (constructivism learning theory)
- **Cognitive Alignment**: adjusts existing mental structures to incorporate new knowledge
  - Prevents factual errors
  - Mitigates hallucination of LLMs

**Data Statistics:**
- Table 1: Data Statistics
  - Each dataset consists of randomly sampled 500 questions from the raw dataset
- PopQA, TriviaQA, NQ, WebQ datasets used for analysis.

## 4 Experimental Methodology

**ActiveRAG: Revealing the Treasures of Knowledge via Active Learning**

**Datasets**:
- Natural Questions (NQ) Kwiatkowski et al. ([2019](https://arxiv.org/html/2402.13547v1#bib.bib19))
- PopQA Mallen et al. ([2023](https://arxiv.org/html/2402.13547v1#bib.bib23))
- TriviaQA Joshi et al. ([2017](https://arxiv.org/html/2402.13547v1#bib.bib16))
- WebQ Berant et al. ([2013](https://arxiv.org/html/2402.13547v1#bib.bib4))
- 500 questions randomly sampled from each QA dataset used for evaluation

**Evaluation Metrics**:
- Accuracy (Acc) to evaluate model performance in open domain QA
- String matching between golden answer and model prediction for calculating accuracy

**Baselines**:
- Prompt learning: vanilla answer generation, Chain-of-Thought, Guideline models
- RAG modeling: vanilla RAG, Chain-of-Note, Self-Rerank, Self-Refine models

**Implementation Details**:
- Access GPT models via OpenAI API for inference
- Use T5-ANCE to retrieve top-k relevant documents from KILT-Wikipedia for each question
- Set temperature at 0.2 during generation and employ gpt-3.5-turbo-1106 as the foundation model

## 5 Evaluation Result

**Overall Performance of ActiveRAG**
- Analysis of ActiveRAG performance
- Characteristics of knowledge construction mechanisms examined
- Evaluation of generalization capability of knowledge construction outcomes
- Presentation of case studies

### 5.1 Overall Performance

**ActiveRAG Performance on Open-Domain QA Datasets**
* Comparison of ActiveRAG with various baseline models: LLMs w/o RAG, vanilla RAG model, self-refined RAG models, and ChatGPT-3.5 model (Table [2](https://arxiv.org/html/2402.13547v1#S4.T2 "Table 2"))
* Baseline models' performance:
  * Vanilla RAG model shows distinct improvements on some datasets compared to ChatGPT-3.5 but not all (Yu et al., [2022](https://arxiv.org/html/2402.13547v1#bib.bib45))
  * Refinement of passages for LLMs leads to decreased performance in both TrivialQA and WebQ datasets (ChatGPT-3.5)
* ActiveRAG outperforms all baseline models with over 5% improvements, particularly on the WebQ dataset
* Significant improvements of ActiveRAG over ChatGPT-3.5 across all datasets, indicating its capacity to guide LLMs in uncovering valuable knowledge through active learning.
* Experiments conducted using 5 and 10 top-ranked passages:
  * All baseline RAG models exhibit consistent improvements when provided with more retrieved passages for generation
  * ActiveRAG shows nearly identical performance regardless of the number of provided passages, indicating that information extracted from top 5 passages is sufficient to prompt LLMs to generate correct answers.
* ActiveRAG effective in uncovering necessary knowledge from retrieved passages with help of knowledge construction.

### 5.2 Ablation Study

**Experimental Methodology**
* Ablation studies conducted to demonstrate effectiveness of cognitive nexus method
* Different knowledge construction methods integrated with chain-of-thought: CoT w. Passage, CoT w. Note, ActiveRAG
* Baseline models: raw retrieved passages (CoT w. Passage), refined passages as notes (CoT w. Note)
* Oracle reranking to show upper bound performance of knowledge construction methods

**Results:**
* CoT w. Passage improves vanilla RAG model performance, establishing cognitive nexus between external knowledge and intrinsic cognition of LLMs
* Significantly enhances quality of generated outputs for LLM employing CoT method
* Mitigates hallucination problem in LLMs by utilizing a cognitive nexus to refine raw chain-of-thought with external knowledge
* ActiveRAG outperforms other methods across all datasets, highlighting efficacy of knowledge construction approaches
* Associate knowledge construction method demonstrates necessity for leveraging external knowledge to deepen understanding of LLMs.

**Oracle Reranking:**
* Potential effectiveness of integrating different knowledge construction methods indicated by evaluation results
* Leveraging diverse methods aids LLMs in answering various kinds of questions.

**Figure 3:**
* Text Similarity between Associate and Other Knowledge Construction Methods
* BLEU-2 score used for evaluation in experiments.

### 5.3 Characteristics of Different Knowledge Construction Mechanisms

**Evaluation of Associate-based Knowledge Construction Method**

**Observations**:
- The Associate method exhibits less similarity to chain-of-note approach, indicating a divergence from simple summarization
- The Associate method yields results more akin to Anchoring, suggesting both mechanisms primarily extract knowledge from retrieved passages
- The results of the Associate method diverge more significantly from those of Cognition, indicating counterfactual cognition leading to disparate outcomes
- LLMs have the capability to construct knowledge and understanding from diverse perspectives

**Evaluation Results**:

| Method     | NQ  | TriviaQA | PopQA   | WebQ    |
| -----------|------|----------|---------|---------|
| LLaMA2-7B |       |         |         |         |
| Vanilla RAG | 26.4 | 41.8    | 30.8    | 24.8    |
| Chain-of-Note | 22.0 | 45.2    | 38.1    | 21.0    |
| CoT        | 32.2 | 66.0    | 25.2    | 41.8    |
| LLM w/o RAG | 26.8 | 53.2    | 21.6    | 40.2    |
| Associate   | 50.6 | 87.4    | 55.8    | 49.6    |
| Anchoring   | 52.4 | 88.6    | 51.6    | 54.0    |
| Logician   | 51.2 | 87.6    | 59.6    | 50.2    |
| Cognition  | 54.4 | 87.2    | 58.4    | 50.6    |
| PPL-Rerank | 53.4 | 87.6    | 56.2    | 50.8    |

**Generalization Effectiveness of Outcomes**:
- The table shows the generalization effectiveness of outcomes from chain-of-note and ActiveRAG, using the top-ranked passages for augmentation

**Case Study**:
- Detailed analysis of a case where the Associate method was used to answer a question about the first sports event held in Worcester, Massachusetts in 1927
- The case includes passage retrieval, knowledge construction using Associate Agent, and cognitive nexus steps.

### 5.4 Generalization Ability of Knowledge Construction Outcomes

**Experiment Findings on Knowledge Construction Outcomes**
- **ActiveRAG**: Revealing the Treasures of Knowledge via Active Learning
- Table [4](https://arxiv.org/html/2402.13547v1#S5.T4 "Table 4 ‣ 5.3 Characteristics of Different Knowledge Construction Mechanisms ‣ 5 Evaluation Result ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning") shows results using different LLMs:
    - **Prompting weaker LLMs (LLaMA2-7B/13B) with outcomes of ActiveRAG**:
        - Decreased performance for vanilla RAG and Chain-of-Note compared to LLM without RAG
            * Retrieved passages become noise, adversely impacting model performance
            * Limited capacity of LLaMA2-7B/13B to analyze and locate knowledge causes misleading
        - Significant improvements over 20% for LLaMA2-7B/13B across all QA datasets
    - Knowledge construction method:
        - Diverges from Chain-of-Note approach
        - Establishes knowledge understanding by connecting retrieved passages with previously acquired knowledge
        - Embodies teacher's thinking process, serving as cognitive roadmap for student
    - Enhancing outcomes of various knowledge construction mechanisms:
        - Quality assessed through calculation of query-conditioned perplexity (PPL)
        - LLM capable of selecting suitable method for obtaining precise answers.

### 5.5 Case Studies

**ActiveRAG's Effectiveness**:
- Shown through analysis of one case from Table [5](https://arxiv.org/html/2402.13547v1#S5.T5 "Table 5 ‣ 5.3 Characteristics of Different Knowledge Construction Mechanisms ‣ 5 Evaluation Result ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")
- Case demonstrates **knowledge understanding results** formed with learned knowledge, such as foundational and advanced information
- Extracted information provides **essential clues for answering questions**
- Analysis of knowledge construction offers **brief summaries** and illustrates the application of external knowledge
- Related knowledge from passages (e.g., "sports events in 1927", "golf", "1925 U.S. Open") makes thoughts more **knowledgeable and detailed**
- Effectiveness of ActiveRAG in supporting Language Models (LLMs) to combine external knowledge and intrinsic cognition for **answer generation**

## 6 Conclusion

**ActiveRAG Paper Proposal**:
- Introduces an architecture combining retrieval and augmentation (ActiveRAG)
- Incorporates active learning for knowledge construction
- Inspired by Constructivism theory
- Builds mechanisms for knowledge construction and cognitive nexus to integrate retrieved knowledge with LLMs' intrinsic cognition.

## A.1 License

**Role: Associate**
- **Fact-checking**: process of confirming accuracy of statements or claims through reliable sources
- **Model illusion**: overconfidence response of AI due to inadequate training data
- Extract foundational knowledge and advanced information from passages for deepening model's understanding
- Encourage thorough comprehension of questions and expanding knowledge boundaries

**Role: Anchoring**
- Provide relevant background and unknown knowledge to help the model better understand a question
- Retrieve unfamiliar knowledge from authoritative sources

**Role: Logician**
- Enhance causal reasoning and logical inference abilities of the model by extracting relevant information
- Use retrieved knowledge to improve reasoning process

**Role: Cognition**
- Update language model's knowledge and prevent factual errors through fact-checking
- Retrieve knowledge from most authoritative sources to ensure accuracy.

**Prompt Templates**:
- Associate: Enhance reasoning process using retrieved knowledge for deepening understanding
- Anchoring: Help the model better understand a question by providing relevant, unfamiliar knowledge
- Logician: Improve causal reasoning and logical inference abilities using retrieved knowledge
- Cognition: Update language model's knowledge to prevent factual errors and alleviate illusions.

**Method: Vanilla ChatGPT-3.5+**
* Provide concise answers to questions
* Avoid unnecessary details

**Question:** Who was the first British golfer to win the US Masters?

**Reasoning Process:**
1. The US Masters is a prestigious golf tournament held in the United States.
2. The first British golfer to win the US Masters would have had to have won it before any other British golfer.
3. Sandy Lyle was the first British golfer to win the US Masters, with victories in 1986 and 1988.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters.

---

**Method: Chain of Thought**
* Think and reason step by step to answer questions
* Consider relevance of given information

**Question:** Who was the first British golfer to win the US Masters?

**Reasoning Process:**
1. The US Masters is a prestigious golf tournament held in the United States.
2. Sandy Lyle became the first British player to win the U.S. Masters, marking a significant milestone for British golfers in the history of the tournament.
3. Other notable achievements in golf history include Gary Player being the first foreigner to win the American Masters Tournament and Ben Hogan being the only golfer to win the Masters, U.S. Open, and British Open in the same calendar year.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters. This achievement marked a significant milestone for British golfers in the history of the tournament.

---

**Method: Guideline**
* Follow a step-by-step process to guide students in solving problems
* Consider relevance and usefulness of given information

**Prompt:** You are a knowledgeable and patient professor whose role is to guide students in solving problems correctly. The following question requires detailed analysis: Who was the first British golfer to win the US Masters?

**Reasoning Process:**
1. Sandy Lyle became the first British player to win the U.S. Masters, marking a significant milestone for British golfers in the history of the tournament.
2. This achievement is mentioned in two out of five provided passages (Passage 1 and Passage 2). These passages are relevant and useful in answering the question.
3. Other notable achievements in golf history include Gary Player being the first foreigner to win the American Masters Tournament and Ben Hogan being the only golfer to win the Masters, U.S. Open, and British Open in the same calendar year.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters. This achievement marked a significant milestone for British golfers in the history of the tournament.

---

**Method: RAG (Relate, Analyze, Guarantee)**
* Follow a step-by-step process to answer questions using provided passages
* Consider relevance and usefulness of given information

**Passages:**
1. Sandy Lyle becomes the first British player to win the U.S. Masters. Lyle sweeps his approach shot to the last green out of a fairway bunker to within, and sinks the resulting birdie putt for a one-shot victory over Mark Calcavecchia.
2. 10 April 2013 Golfer Sandy Lyle becomes the first British winner of the US Masters.
3. In 2007, Virginia Tech golfer Drew Weaver became the first American to win the British Amateur golf tournament since 1979. Weaver edged out 2006 Australian Amateur champion Tim Stewart and earned an invitation to the 2007 Open Championship.
4. Gary Player of South Africa became the first foreigner to win the American Masters Tournament, taking the event by one stroke. On the very last hole, the leader, Arnold Palmer had to take six strokes.

**Reasoning Process:**
1. Sandy Lyle is mentioned as the first British player to win the U.S. Masters in Passages 1 and 2. These passages are relevant and useful for answering the question.
2. Other notable achievements in golf history include Gary Player being the first foreigner to win the American Masters Tournament and Ben Hogan being the only golfer to win the Masters, U.S. Open, and British Open in the same calendar year. These achievements are not directly related to answering the original question but provide additional context for understanding golf history.
3. Passages 3 and 4 do not contain any information about the first British golfer to win the US Masters. They focus on different aspects of golf history, so they are irrelevant to answering the original question.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters. This achievement marked a significant milestone for British golfers in the history of the tournament.

#### Finding Marie-Hélène Aubert's Birthplace: A Case Study in ActiveRAG

**Marie-Hélène Aubert's Birthplace**
- Marie-Francine Hébert, a Canadian author from Quebec, is an unfamiliar name that does not provide information about Marie-Hélène Aubert's birthplace.
- Marie Marguerite Françoise Hébert was a figure in the French Revolution and was born in Paris, but she is also not Marie-Hélène Aubert.
- Michel Pierre Alexis Hébert was born in Granville, Manche, which is also not where Marie-Hélène Aubert was born.

**Coretta Scott King's Death**
- Coretta Scott King died on January 30, 2006, at the rehabilitation center in Rosarito Beach, Mexico.
- The main cause of her death is believed to be respiratory failure due to complications from ovarian cancer.
- King's decline in health began after suffering a stroke in August 2005 and being diagnosed with ovarian cancer.
- The clinic where she died, Hospital Santa Monica (licensed as Clinica Santo Tomas), was shut down due to operating without proper licensing and controversial medical practices.

**The Woods Television Series**
- The Woods is a television series produced by Matthew Lessner.

#### ActiveRAG: Finding "The Woods" Producer Using Cognitive Nexus

**Producer of "The Woods" Film**

**Passage Retrieval**:
- Walter Wood (producer) - not relevant to the film "The Woods"
- "The Woods is a 2011 film written and directed by Matthew Lessner and starring Toby David and Justin Phillips."
- "Robert Woods is a classical music producer. Woods is a founder and former president of the independent record label Telarc International Corporation."
- "The Wood is a 1999 American coming of age film directed by Rick Famuyiwa and starring Omar Epps, Richard T. Jones and Taye Diggs."

**Knowledge Construction using Cognition Agent**:
- Contradictory information: 
    - Mention of "Robert Woods" as a classical music producer, which does not align with the question about the producer of the film "The Woods"
- Analyzing the role of information:
    - This information could update the model's knowledge and prevent factual errors by adding the correct producer of "The Woods" film, which is Matthew Lessner

**Cognitive Nexus**:
- Contradictory information:
    - "Robert Woods" mentioned as a classical music producer (not relevant to "The Woods" film)
- Correct information:
    - "The Woods is a 2011 film written and directed by Matthew Lessner."
- Knowledge enhancement:
    - Add the correct producer of "The Woods" film, which is Matthew Lessner

**Answer**: The producer of "The Woods" is **Matthew Lessner**.

## A.2 Prompts Used in ActiveRAG

ActiveRAG prompts for building the model:
- Knowledge construction (Table 6): {question}, {passages} \n
- Cognitive nexus (Table 7): {question}, {passages} \n
- Response from chain-of-thought model: {chain_of_thought_reply}\n
- Results of knowledge construction models: Associate, Anchoring, Logician, and Cognition (respectively)

## A.3 Prompts Used in Baseline Models

**Table 8**: Shows prompts used in our baseline models: ChatGPT-3.5, Guideline, Vanilla RAG, Chain-of-Note, Self-Rerank, and Self-Refine. The template for Chain-of-Note follows Yu et al.'s (2023b) description, while the prompt for Vanilla RAG is based on Ram et al. (2023b).

## A.4 Additional Case Studies

**ActiveRAG Results: Case Studies**

**ActiveRAG with Associate Model**:
- Illustrated in Table [9](https://arxiv.org/html/2402.13547v1#A1.T9 "Table 9 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")
- Initial chain-of-thought leads to erroneous inference
- Associate-based knowledge construction method extracts related knowledge
- LLMs identify incorrect reasoning process and augment it
- Yield accurate answers

**ActiveRAG with Anchoring Model**:
- Illustrated in Table [10](https://arxiv.org/html/2402.13547v1#A1.T10 "Table 10 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")
- Anchoring model identifies unfamiliar information as "Marie-Francine Hébert"
- Accurately extracts related information from passages
- Produces accurate response: "Nantes, France"

**ActiveRAG with Logician Model**:
- Illustrated in Table [11](https://arxiv.org/html/2402.13547v1#A1.T11 "Table 11 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")
- Logician-based knowledge construction generates rational knowledge from retrieved passages
- Recognizes that "The primary cause of Coretta Scott King’s death is attributed to respiratory failure resulting from complications related to ovarian cancer"
- Provides accurate evidence directly

**ActiveRAG with Cognition Model**:
- Illustrated in Table [12](https://arxiv.org/html/2402.13547v1#A1.T12 "Table 12 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")
- Cognition-based knowledge construction results illustrate that knowledge about " Robert Woods" does not align with query needs
- Significance of knowledge construction in discerning distinctions, like "Woods" film vs. producer "Walter Wood"
- Facilitates LLMs in generating precise responses.

