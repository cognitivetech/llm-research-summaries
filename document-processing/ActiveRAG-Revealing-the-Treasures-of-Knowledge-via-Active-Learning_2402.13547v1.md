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
- [A.2 Prompts Used in ActiveRAG](#a2-prompts-used-in-activerag)
- [A.3 Prompts Used in Baseline Models](#a3-prompts-used-in-baseline-models)
- [A.4 Additional Case Studies](#a4-additional-case-studies)

## Abstract

**Retrieval Augmented Generation (RAG)**
- New paradigm for Large Language Models (LLMs) aiding in resolution of knowledge-intensive tasks

**Current RAG models:**
- Position LLMs as passive knowledge receptors
- Restricts capacity for learning and comprehending external knowledge

**ActiveRAG:**
- Innovative RAG framework
- Shifts from **passive knowledge acquisition** to **active learning mechanism**

**Active Learning Mechanism in ActiveRAG:**
- Utilizes **Knowledge Construction mechanism**
  - Develops deeper understanding of external knowledge
  - Associates it with previously acquired or memorized knowledge

- Designs **Cognitive Nexus mechanism**
  - Incorporates outcomes from both chains of thought and knowledge construction
  - Calibrates the intrinsic cognition of LLMs

**Experimental Results:**
- ActiveRAG surpasses previous RAG models
- Achieves a **5% improvement on question-answering datasets**

**Availability:**
- All data and codes available at [https://github.com/OpenMatch/ActiveRAG]()

## 1 Introduction

**Constructivism in Language Models**

**Learners' Knowledge Acquisition**:
- Learners construct new understandings and knowledge through experience and social discourse

**Large Language Models (LLMs)**:
- LLMs, e.g., GPT-4 OpenAI and LLaMA, have shown strong emergent abilities and convincing performance in NLP tasks
- However, they suffer from the **hallucination problem**, **outdated parametric memories**, and **unreliable outputs**

**Retrieval-Augmented Generation (RAG) Models**:
- RAG models retrieve knowledge from external corpus and build a brute-force RAG architecture by directly feeding passages to LLMs
- However, the effectiveness of retrieval-augmented models is highly influenced by the noise from retrieved knowledge

**ActiveRAG**:
- ActiveRAG focuses on mitigating the effect of noise from retrieved passages by self-reflection and self-refining
- It filters out irrelevant passages or conducts summarized noting to extract key point knowledge, which is effective in assisting LLMs to generate more accurate answers

**Limitations of Passive Knowledge Acquisition**:
- Vanilla RAG and Chain-of-Note models yield inaccurate answers, highlighting the limitations of passive knowledge acquisition
- Vanilla RAG is misled by certain ambiguous entities; Chain-of-Note employs shallow relevance modeling

**ActiveRAG**:
- ActiveRAG leverages active learning to augment knowledge comprehension by bridging the gap between prior knowledge and retrieved information
- It enhances the active learning capability of LLMs without requiring further fine-tuning
- ActiveRAG uses a three-step pipeline: Retrieval, Knowledge Construction, and Cognition Nexus stages

**Experiments and Results**:
- Evaluation results demonstrate the effectiveness of ActiveRAG across various question answering datasets, yielding over 5% improvement compared to baseline models
- ActiveRAG exhibits robustness and maintains stable performance across different datasets and varying numbers of retrieved passages
- Knowledge construction outcomes can be generalized to different LLM architectures, aiding them in leveraging external knowledge and achieving improvements exceeding 20%

## 2 Related Work

**Retrieval-Augmented Generation (RAG) Models**

**Background:**
- RAG models aim to retrieve external knowledge and enhance language models
- Strong effectiveness in various NLP tasks: question answering, dialog understanding, code generation, etc.
- Utilize dense retrievers as knowledge-seeking modules, prompting language models to generate results based on retrieved knowledge

**Earlier RAG Models:**
- Optimize models to leverage external knowledge and serve knowledge-intensive tasks
- Develop architecture of generator to fully use external knowledge
- Train more accurate retriever using feedback from generation models
- Jointly train retriever and generator

**RAG in Large Language Models (LLMs):**
- Leverage external knowledge to reduce perplexity in general NLP tasks
- Alleviate hallucination problem by updating outdated or long-tail knowledge

**Challenges:**
- Noise from retrieved contexts challenges effectiveness of RAG models

**Approaches to Mitigate Noise:**
- Eliminate irrelevant contexts: use natural language inference, summarization models, conditional cross-mutual information
- Self-RAG: LLMs filter out irrelevant passages via self-reflection
- Adaptively retrieve passages or build noting mechanism to summarize key point knowledge from retrieved passages

**Constructivism Theory:**
- Associates external knowledge with prior knowledge
- RAG models continue to passively acquire additional knowledge, often overlooking the importance of this connection.

## 3 Methodology

Introduction to ActiveRAG method:
- This section presents our ActiveRAG method (Sec. [3.2](https://arxiv.org/html/2402.13547v1#S3.SS2 "3.2 ActiveRAG"))
- Guides LLMs to actively learn knowledge from retrieved passages (Sec. [3.2](https://arxiv.org/html/2402.13547v1#S3.SS2 "3.2 ActiveRAG"))
- Based on Retrieval-Augmented Generation (RAG) models (Sec. [3.1](https://arxiv.org/html/2402.13547v1#S3.SS1 "3.1 Preliminary of RAG Models"))

(Note: ActiveRAG stands for Revealing the Treasures of Knowledge via Active Learning)

### 3.1 Preliminary of Retrieval-Augmented Generation (RAG) Models

**Retrieval-Augmented Generation (RAG)**

**Existing RAG Models**:
- Utilize retrieval models to search passages `D={d1,…,dn}`
- Enhance the generation ability of language models by grounding these retrieved passages

**Vanilla RAG**:
- Employs the **retrieval-generation architecture**
- Directly feeds the retrieved passages `D` to language models to generate answers for query `q`
- Retrieved passages contain noise, and this **brute-force approach** tends to constrain the benefits of RAG modeling

**RAG with Self-Refining**:
- Avoids the effect of the noise from retrieved passages
- Employs the **retrieval-refining-generation architecture** to empower the capability of RAG using LLMs

**Self-RAG and Chain-of-Note**:
- **Self-RAG**: Summarizes knowledge from retrieved passages, focusing on finetuning LLMs to adaptively retrieve and control information flows
- **Chain-of-Note**: Designs instructions to self-refine the retrieved passages and forms query-focused summarization, applicable to black-box LLMs like GPT-3.5
- Neglect the nature of active learning, where the learner actively constructs their understanding and knowledge by integrating new information with prior knowledge

### 3.2 ActiveRAG: RAG with Active Knowledge Learning

**ActiveRAG Architecture for Active Learning**

**Background:**
- Inspired by constructivist learning theory: Steffe & Gale (1995)
- LLMs actively acquire knowledge through ActiveRAG pipeline

**Components:**
1. Retrieval
2. Knowledge Construction
3. Cognitive Nexus

**Knowledge Construction:**
- Four agents for knowledge acquisition: anchoring, logical reasoning, cognition, association
- Human learning behavior mimicking
- Detailed descriptions in Sec. 3.3

**Cognitive Nexus:**
- Facilitates fusion of constructed knowledge understanding with intrinsic cognitive processes of LLMs
- Employed to assist LLMs in utilizing external knowledge for augmentation
- Fusion occurs during the generation of a chain-of-thought
- Encourages reflection and rectification of potential factual errors
- Differs from RAG models: emphasis on incorporating knowledge construction into cognitive reasoning.

### 3.3 Knowledge Construction from Different Learning Views

**Constructivism Learning Theory**
- **Active process**: learning involves construction of internal mental representations
- **Structural knowledge** and experiential backgrounds encompassed
- **Four agents to acquire knowledge:** Semantic Association, Epistemic Anchoring, Logical Reasoning, Cognitive Alignment

**Semantic Association (Associate)**
- Encourages integration of familiar information
- Consolidation of foundational and advanced pieces of knowledge
- Expands cognitive boundaries for a comprehensive understanding

**Epistemic Anchoring (Anchoring)**
- Represents previously unknown knowledge
- Establishes foundational understanding
- Incorporates new concepts relevant to the question

**Logical Reasoning (Logician)**
- Draws logical conclusions from structured information
- Refines problem-solving capabilities
- Constructivism learning theory: not merely about absorbing information but active construction of knowledge through reasoning

**Cognitive Alignment (Cognition)**
- Deals with knowledge that contradicts pre-existing understanding
- Prevents factual errors and mitigates hallucination of Language Models (LLMs)

**Data Statistics**
- Each dataset consists of 500 questions randomly sampled from the raw dataset
- Percentage of queries used: PopQA (3.5%), TriviaQA (5.7%), NQ (5.7%), WebQ (24.6%)

## 4 Experimental Methodology

**Dataset**:
- Four datasets used for open domain question answering: Natural Questions (NQ), PopQA, TriviaQA, and WebQ
- Subset of 500 questions randomly sampled from each dataset due to inference cost

**Evaluation Metrics**:
- Accuracy (Acc) used to evaluate model performance
- Lowercase conversion and string matching between model prediction and golden answer

**Baselines**:
- Prompt learning: Vanilla answer generation, Chain-of-Thought (CoT), Guideline
- RAG modeling: Vanilla RAG, Chain-of-Note, Self-Rerank, Self-Refine

**Implementation Details**:
- Access GPT models via OpenAI API using gpt-3.5-turbo-1106 as foundation model and temperature 0.2 during generation
- Use T5-ANCE for retrieving top-k relevant documents from KILT-Wikipedia using OpenMatch toolkit

## 5 Evaluation Result

**Overall Performance of ActiveRAG**
- Presented in this section

**Analyses of Knowledge Construction Mechanisms**
- Exploration of characteristics within various mechanisms

**Generalization Capability of Knowledge Construction Outcomes**
- Investigation into the generalizability of outcomes

**Case Studies**
- Demonstration through specific examples

### 5.1 Overall Performance

**Performance Comparison: ActiveRAG vs Baseline Models**
- **ActiveRAG**: shows significant improvements over all baseline models on different open-domain QA datasets (Table [2](https://arxiv.org/html/2402.13547v1#S4.T2))
- **Vanilla RAG model**: exhibits distinct performance compared to ChatGPT-3.5 on WebQ dataset, nearly equivalent on TrivialQA and NQ datasets (Yu et al., [2022](https://arxiv.org/html/2402.13547v1#bib.bib45))
- **ChatGPT-3.5 model**: decrease in performance on WebQ dataset, nearly equivalent on TrivialQA and NQ datasets (Yu et al., [2022](https://arxiv.org/html/2402.13547v1#bib.bib45))
- **Self-refined RAG models**: performance improvements when provided with more retrieved passages for generation (baseline RAG models consistently exhibit improvements)
- **ActiveRAG**: nearly identical performance when provided with top 5 or top 10 passages, indicating the effectiveness of knowledge construction in uncovering necessary knowledge from retrieved passages.

### 5.2 Ablation Study

**Experimental Methodology: ActiveRAG**

**Ablation Studies**:
- Demonstrate effectiveness of cognitive nexus method through different knowledge construction methods integrated with chain-of-thought (CoT)

**Baseline Models**:
- CoT with Passage: process retrieved passages
- CoT with Note: refine passages as notes
- ActiveRAG: knowledge construction approaches

**Oracle Reranking**:
- Illustrates potential effectiveness of integrating different knowledge construction methods

**Results**:
- CoT with Passage improves performance of vanilla RAG model, establishing cognitive nexus between external knowledge and intrinsic cognition of LLMs
- CoT with Passage significantly improves quality of generated outputs for CoT-employed LLM
- Mitigates hallucination problem in LLMs by using a cognitive nexus to refine raw chain-of-thought with external knowledge

**Comparative Analysis**:
- ActiveRAG outperforms various knowledge representation methods, highlighting efficacy of knowledge construction methods
- Associate method outperforms others, demonstrating necessity of leveraging external knowledge for deepening LLM understanding

**Figure 3: Text Similarity between Associate and Other Knowledge Construction Methods**:
- Employs BLEU-2 score for evaluation

### 5.3 Characteristics of Different Knowledge Construction Mechanisms

**Associate-based Knowledge Construction Method Characteristics**
* Conduct experiments to demonstrate Associate method's capabilities (Figure 3)
* Diverges from chain-of-note approach in knowledge construction, showing less similarity
* Yields results akin to Anchoring, extracting knowledge from retrieved passages
* Counterfactual cognition leads to disparate outcomes compared to Cognition
* LLMs can construct knowledge and understanding from various perspectives

**Evaluation Results**
| Method        | NQ | TriviaQA | PopQA  | WebQ   |
|---------------|-----|----------|---------|--------|
| LLaMA2-7B     |    |          |         |        |
| Vanilla RAG   | 26.4| 41.8      | 30.8    | 24.8   |
| Chain-of-Note | 22.0| 45.2      | 38.1    | 21.0   |
| CoT          | 32.2| 66.0      | 25.2    | 41.8   |
| LLM w/o RAG  | 26.8| 53.2      | 21.6    | 40.2   |
| w. Associate  | 50.6| 87.4      | 55.8    | 49.6   |
| w. Anchoring  | 52.4| 88.6      | 51.6    | 54.0   |
| w. Logician  | 51.2| 87.6      | 58.4    | 50.6   |
| w. Cognition | 54.4| 87.2      | 58.4    | 50.6   |
| PPL-Rerank   | 53.4| 87.6      | 56.2    | 50.8   |
| LLaMA2-13B   |    |          |         |        |
| Vanilla RAG   | 29.0| 39.0      | 22.8    | 25.2   |
| Chain-of-Note | 22.0| 46.0      | 30.6    | 22.8   |
| CoT          | 34.0| 73.2      | 31.6    | 48.8   |
| LLM w/o RAG  | 30.6| 57.2      | 32.2    | 44.6   |
| w. Associate  | 53.2| 89.6      | 57.4    | 52.6   |
| w. Anchoring  | 51.6| 88.2      | 50.6    | 53.8   |
| w. Logician  | 57.8| 90.3      | 65.4    | 53.0   |
| w. Cognition | 51.2| 88.6      | 52.4    | 50.6   |

**Table 4: Generalization Effectiveness of Outcomes from Chain-of-Note and ActiveRAG**
* First Ryder Cup held at Worcester, Massachusetts in 1927 (Answer)
* Passage retrieval for augmentation

**Step 1: Passage Retrieval**
1. Think about sports events that could have been held in Worcester, Massachusetts in 1927.
2. Consider the popularity of different sports at that time.
3. Recall any historical events or records related to sports in Worcester in 1927.

**Passages:**
- Worcester now hosts the Massachusetts Pirates (Indoor football team) and previously was home to New England Surge (Defunct Continental Indoor Football League).
- Worcester Country Club is a private country club that hosted the first Ryder Cup in 1927. It has also hosted the men's and women's U.S. Open, as well as the first official U.S. Open qualifying round in 1924.
- Worcester, Massachusetts is home to minor league sports teams and professional franchises. Many historic and local sporting events have occurred in Worcester, including the first official Ryder Cup golf tournament at Worcester Country Club in 1927.

**Step 2: Knowledge Construction using Associate Agent**
* Foundational Knowledge: Worcester, Massachusetts has a history of hosting various sports events, including golf tournaments and minor league teams. The Worcester Country Club hosted the first Ryder Cup in 1927, making it a significant location for the event.
* Advanced Information: Details about specific sports events and teams that have been hosted or based in Worcester, such as the Worcester Busters, Worcester Boosters, and Worcester Panthers baseball teams.
* Analyzing the Role of Information: Worcester, Massachusetts has a rich sports history and has been a significant location for various sports events and teams. The hosting of the first Ryder Cup in 1927 at the Worcester Country Club highlights the city's importance in the world of golf. Additionally, the presence of minor league sports teams and professional franchises demonstrates the city's active participation in a wide range of sports. This information deepens the model's understanding of the question by providing context about the sports culture in Worcester and expanding its knowledge boundaries about the specific events and teams associated with the city.

**Step 3: Cognitive Nexus**
1. Identify popular sports events in 1927, considering Worcester, Massachusetts as the location.
2. Worcester, Massachusetts has a rich history in sports, particularly in baseball and golf. The Worcester Country Club hosted significant events such as the first Ryder Cup in 1927, the 1925 U.S. Open, the 1960 U.S. Women's Open, and the first-ever U.S. Open qualifying round in 1924.
3. The Worcester Country Club is an elite group of three golf courses in the United States to host the men's and women's U.S. Open and the Ryder Cup. This further solidifies its importance in the sports world.
4. Considering the historical significance of the Worcester Country Club in hosting major golf events, it is likely that the sports event first held at Worcester, Massachusetts in 1927 was the first Ryder Cup.

### 5.4 Generalization Ability of Knowledge Construction Outcomes

**Study Findings on Knowledge Construction Outcomes using Different LLMs:**
* **ActiveRAG**: reveals quality of knowledge construction outcomes for various LLMs (Table [4](https://arxiv.org/html/2402.13547v1#S5.T4 "Table 4"))
* **Prompting weaker models** with ActiveRAG: LLaMA2-7B/13B
* Results:
	+ Decreased performance for vanilla RAG and Chain-of-Note compared to LLM without RAG (LLaMA2-7B/13B misled by retrieved passages due to limited capacity)
	+ Significant improvements over 20% across all QA datasets for LLaMA2-7B/13B, demonstrating effectiveness
* Differences from Chain-of-Note: knowledge construction method connects retrieved passages with previously acquired knowledge, establishing cognitive roadmap
* Enhancing outcomes of various knowledge construction mechanisms through query-conditioned perplexity (PPL) assessment.

### 5.5 Case Studies

**ActiveRAG Effectiveness:**
- Shown through analysis of Table [5](https://arxiv.org/html/2402.13547v1#S5.T5 "Table 5 ‣ 5.3 Characteristics of Different Knowledge Construction Mechanisms ‣ 5 Evaluation Result ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning") case study
- Forms knowledge understanding results using learned information (foundational, advanced)
- Essential clues for answering questions provided
- Brief summaries and application of external knowledge illustrated
- Incorporates knowledge construction outcomes into cognitive nexus module
- Calibrates raw chain-of-thought results to generate correct answers
- Effectiveness demonstrated through use of related knowledge from passages (e.g., "sports events in 1927", "golf", "1925 U.S. Open") for more detailed and knowledgeable thoughts.

## 6 Conclusion

**ActiveRAG Paper**:
- Introduces an active retrieval-augmentation architecture for learning
- Based on Constructivism theory, integrates knowledge construction and cognitive nexus mechanisms in LLMs to combine external knowledge with intrinsic cognition.

### Limitation

ActiveRAG effectively leverages external knowledge for LLM generation tasks without finetuning, requiring calls to the ChatGPT API three times: once for initial chain-of-thought construction, again for processing knowledge construction results, and lastly for generating the final answer. However, this process may increase response time latency and API calling costs due to the lengthy inputs caused by including extensive retrieved passages and knowledge construction results.

## A.1 License

**Fact-Checking and Addressing Model Illusions in AI**

**Prompt Templates for Knowledge Construction**:
- **Associate**: Extract foundational knowledge or advanced information to deepen the model's understanding of the question
  - Retrieve passages from reliable sources
  - Verify for errors
  - Enhance reasoning process using retrieved knowledge
  - Generate answer based on enhanced reasoning process

**Anchoring**: Provide relevant background and unknown knowledge to help the model better understand the question
- Retrieve passages from reliable sources
- Identify information that can update the model's knowledge
- Verify for errors
- Enhance reasoning process using retrieved knowledge
- Generate answer based on enhanced reasoning process

**Logician**: Extract content to enhance causal reasoning and logical inference abilities
- Retrieve passages from reliable sources
- Identify information that can improve causal reasoning and logical inference
- Verify for errors
- Enhance reasoning process using retrieved knowledge
- Generate answer based on enhanced reasoning process

**Cognition**: Update the model's knowledge to prevent factual errors and alleviate model illusions
- Retrieve passages from most authoritative knowledge repositories
- Identify information that can update the model's knowledge
- Verify for errors
- Enhance reasoning process using retrieved knowledge
- Generate answer based on enhanced reasoning process

#### "Cognitive Nexus Approach for Question Answering"

**Method: Vanilla ChatGPT-3.5**
* Concise answers to questions
* Avoid unnecessary details
* Please provide detailed analysis step by step

**Question:** {question}

**Method: Chain of Thought**
* Think and reason step by step
* Analyze problem thoroughly
* Provide reasoning process
* Label passages as useful or useless, relevant or irrelevant

**Step 1: Passage Retrieval**
- Relevant passages related to the question
- Gather information from passages

**Step 2: Knowledge Construction using Associate Agent**
- Consolidate gathered information
- Deepen understanding of the question
- Expand knowledge boundaries in relevant field

**Foundational Knowledge:**
- Sandy Lyle was the first British golfer to win the US Masters
- Significant achievement for British golfers in tournament history

**Advanced Information:**
- Gary Player: first foreigner to win American Masters Tournament
- Ben Hogan: only golfer to win Masters, U.S. Open, and British Open in same calendar year

**Step 3: Cognitive Nexus**
- Enhance reasoning process using retrieved knowledge
- Deepen understanding of question through familiarity with basic and advanced information.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters, marking a significant milestone for British golfers in tournament history.

**Method: Self-Refine**
* Analyze reasoning process and refine as needed
* Provide detailed analysis based on passages

**Step 1: Passage Retrieval**
- Relevant passages related to the question
- Gather information from passages

**Step 2: Knowledge Construction using Associate Agent**
- Consolidate gathered information
- Deepen understanding of the question
- Expand knowledge boundaries in relevant field

**Foundational Knowledge:**
- Sandy Lyle was the first British golfer to win the US Masters
- Significant achievement for British golfers in tournament history

**Advanced Information:**
- Gary Player: first foreigner to win American Masters Tournament
- Ben Hogan: only golfer to win Masters, U.S. Open, and British Open in same calendar year

**Step 3: Cognitive Nexus**
- Enhance reasoning process using retrieved knowledge
- Deepen understanding of question through familiarity with basic and advanced information.

**Answer:** Sandy Lyle was the first British golfer to win the US Masters, marking a significant milestone for British golfers in tournament history.

#### Marie-Hélène Aubert's Birthplace: Nantes

**Role:** Anchoring

**Question:** Where was Marie-Hélène Aubert born?

**Answer:** Nantes, France

**Reasoning process:**
1. Marie-Hélène Aubert is a French politician and writer.

#### Coretta Scott King's Cause of Death: Respiratory Failure due to Ovarian Cancer

**Coretta Scott King's Death**

**Background:**
- Coretta Scott King: civil rights leader, wife of Martin Luther King Jr., passed away on January 30, 2006
- Cause of death believed to be respiratory failure due to complications from ovarian cancer
- Decline in health began after suffering a stroke and diagnosis of ovarian cancer in August 2005

**Passage Retrieval:**
- Passage 1: Coretta Scott King hospitalized with stroke, diagnosed with ovarian cancer (Passage 1)
- Passage 2: Coretta Scott King died from respiratory failure due to complications from ovarian cancer at a rehabilitation center in Mexico (Passage 2)
- Passage 3: Coretta Scott King's mother died on her son's birthday (Passage 3)
- Passage 4: Coretta Scott King, civil rights leader and widow of Martin Luther King Jr., died from ovarian cancer at the age of 78 (Passage 4)

**Knowledge Construction:**
- Coretta Scott King's health declined after a stroke and diagnosis of ovarian cancer in August 2005
- The main cause of her death was respiratory failure due to complications from advanced-stage ovarian cancer
- She was hospitalized and undergoing holistic therapy for her stroke and ovarian cancer at the time of her death
- The clinic where she died, Hospital Santa Monica (licensed as Clinica Santo Tomas), was shut down due to operating without proper licensing and controversial medical practices

**Cognitive Nexus:**
- Coretta Scott King passed away on January 30, 2006, at the rehabilitation center in Rosarito Beach, Mexico
- Her declining health began after suffering a stroke in August 2005 and being diagnosed with ovarian cancer
- The main cause of her death was respiratory failure due to complications from advanced-stage ovarian cancer
- Additional impact from the blood clot lodged in her brain contributed to her ongoing medical issues.

#### Understanding Who Produced "The Woods" through Cognitive Analysis

**Producer of "The Woods"**

**Question:** Who was the producer of The Woods?

**Answer:** Matthew Lessner

**Steps:**
1. **Passage Retrieval**:
   - Passage 1: Walter Wood (producer)
   - Passage 2: The Woods is a 2011 film written and directed by Matthew Lessner...
   - Passage 3: Robert Woods is a classical music producer...
   - Passage 4: The Wood is a 1999 American coming of age film produced by Rick Famuyiwa and Todd Boyd...
2. **Knowledge Construction using Cognition Agent**:
   - Contradictory Knowledge:
      * Mention of "Robert Woods" as a classical music producer does not align with the question about the producer of "The Woods" film
   - Analyzing the Role of Information:
      * This information could update the model's knowledge and prevent factual errors by adding the correct producer of "The Woods" film, which is Matthew Lessner.
3. **Cognitive Nexus**:
   - Contradictory Information: The mention of "Robert Woods" as a classical music producer does not align with the question about the producer of "The Woods" film
   - Correct Producer: Matthew Lessner (mentioned in Passage 2)
   - Updating Model's Knowledge: Adding the correct producer, Matthew Lessner, to prevent factual errors.

## A.2 Prompts Used in ActiveRAG

**ActiveRAG Prompt Templates for Model Building:**

Table 6 and Table 7 in Appendix A display the templates used for our ActiveRAG model. The prompts are:
- {question} (given question)
- {passages} (retrieved passages)
- {chain\_of\_thought\_reply} (output of chain-of-thought model)
- {Associate\_knowledge\_constrcution\_reply}, {Anchoring\_knowledge\_constrcution\_reply}, {Logician\_knowledge\_constrcution\_reply}, and {Cognition\_knowledge\_constrcution\_reply} (outcomes of the respective knowledge construction models)

In simpler terms, the templates include a question, passages retrieved for context, an answer from the chain-of-thought model, and replies generated by four knowledge construction models: Associate, Anchoring, Logician, and Cognition.

## A.3 Prompts Used in Baseline Models

**Table 8**: Shows the prompts used by our baseline models, including vanilla ChatGPT-3.5, Guideline, Vanilla RAG, Chain-of-Note, Self-Rerank, and Self-Refine. The Chain-of-Note prompt template follows Yu et al.'s (2023b) description, while the Vanilla RAG prompt template is derived from Ram et al. (2023b).

**Note**: This passage has been made more concise by reducing redundant phrases and simplifying the language.

## A.4 Additional Case Studies

**ActiveRAG Cases and Results:**
* **Table [9](https://arxiv.org/html/2402.13547v1#A1.T9 "Table 9 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")**:
  + Generated by Associate model
  + Initial chain-of-thought leads to erroneous inference
  + LLMs identify raw chain-of-thought as incorrect reasoning process
  + Augmenting reasoning process yields accurate answers
* **Table [10](https://arxiv.org/html/2402.13547v1#A1.T10 "Table 10 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")**:
  + Generated by Anchoring model
  + Identifies unfamiliar knowledge as "Marie-Francine Hébert"
  + Extracts related information to produce accurate response ("Nantes, France")
* **Table [11](https://arxiv.org/html/2402.13547v1#A1.T11 "Table 11 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")**:
  + Generated by Logician model
  + Recognizes accurate knowledge about Coretta Scott King's cause of death
  + Provides rational knowledge from retrieved passages
* **Table [12](https://arxiv.org/html/2402.13547v1#A1.T12 "Table 12 ‣ A.1 License ‣ Appendix A Appendix ‣ ActiveRAG: Revealing the Treasures of Knowledge via Active Learning")**:
  + Generated by Cognition model
  + Knowledge about Robert Woods does not align with question needs
  + Significance of knowledge construction in discerning distinctions for LLMs.

