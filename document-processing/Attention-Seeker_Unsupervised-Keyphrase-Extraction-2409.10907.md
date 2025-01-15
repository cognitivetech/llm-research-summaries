# Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Keyphrase Extraction

by Erwin D. López Z., Cheng Tang, Atsushi Shimada

https://arxiv.org/pdf/2409.10907

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 Methodology](#3-methodology)
  - [3.1 Attention-Seeker](#31-attention-seeker)
  - [3.2 Extraction of the Self-Attention Maps](#32-extraction-of-the-self-attention-maps)
  - [3.3 Attention Scores Estimation: Short Documents](#33-attention-scores-estimation-short-documents)
  - [3.4 Attention Scores Estimation: Long Documents](#34-attention-scores-estimation-long-documents)
- [4 Experiments and Results](#4-experiments-and-results)
  - [4.1 Datasets and Evaluation Metrics](#41-datasets-and-evaluation-metrics)
  - [4.2 Baselines](#42-baselines)
  - [4.3 Results](#43-results)
- [5 Ablation](#5-ablation)
- [6 Conclusion](#6-conclusion)
- [7 Limitations](#7-limitations)

## Abstract

**Proposed Method**:
- Leverages self-attention maps from a Large Language Model to estimate importance of candidate phrases
- Identifies specific components (layers, heads, attention vectors) where the model pays significant attention to key topics in text
- Uses attention weights provided by these components to score candidate phrases

**Distinguishing Feature**:
- Dynamically adapts to input text without any manual adjustments
- Enhances practical applicability compared to models requiring manual tuning (e.g., selection of heads, prompts, hyperparameters)

**Evaluation and Results**:
- Evaluated on four publicly available datasets: Inspec, SemEval2010, SemEval2017, Krapivin
- Outperforms most baseline models, achieving state-of-the-art performance on three out of four datasets
- Excels in extracting keyphrases from long documents

**Source Code**:
- Available at https://github.com/EruM16/Attention-Seeker

## 1 Introduction

**Keyphrase Extraction**

**Importance of Keyphrase Extraction**:
- Critical task in NLP
- Facilitates efficient and accurate information retrieval
- Valuable for RAG, document categorization, text segmentation, and topic modeling

**Classification of Keyphrase Extraction Methods**:
- **Supervised methods**: High performance, but require large amounts of labeled data and are often domain-specific
- **Unsupervised methods**: Adaptable across various domains, rely on information extracted from the document itself

**Categories of Unsupervised Keyphrase Extraction Methods**:
1. Statistical methods: Use in-corpus statistical information
2. Graph-based methods: Leverage co-occurrence patterns
3. **Embedding-based methods**: Analyze similarities between documents and phrases within a PLM's embedding space
4. Prompt-based methods: Use the PLM's decoder log-its to estimate the probability of generating a candidate phrase
5. Self-attention-based methods: Examine the PLMs' Self-Attention Map (SAM) to identify which candidate phrases receive the most attention

**Limitations of Existing Unsupervised Methods**:
- Embedding-based approaches struggle to accurately estimate document similarities due to anisotropic PLM embedding spaces
- Prompt and self-attention-based methods circumvent these limitations but introduce new complexities

**Proposed Approach: Attention-Seeker**:
- Extends SAMRank, introducing a module that selects the most relevant SAMs to score candidate phrases effectively
- Defines the characteristics of an optimal SAM and uses it to estimate the relevance of each SAM
- Evaluates the relevance of individual attention vectors for short documents and document segments for long documents
- Demonstrates SOTA-level performance on four benchmark datasets without requiring parameter tuning

## 2 Related work

**Unsupervised Keyphrase Extraction: Related Work**

**Statistic-Based Methods**:
- Divided into statistical metrics derived from text
  - TF-IDF method (Spärck Jones, 2004) uses word frequencies
  - YAKE (Campos et al., 2020) incorporates additional factors like co-occurrences, positions, and casings

**Graph-Based Methods**:
- Represent documents as graphs with words as nodes
  - TextRank (Mihalcea and Tarau, 2004) adapts PageRank algorithm for keyphrase extraction
    - SingleRank (Wan and Xiao, 2008): incorporates information from neighboring documents
    - TopicRank (Bougouin et al., 2013): clusters nodes within a topic space
    - PositionRank (Florescu and Caragea, 2017): integrates position bias into PageRank algorithm
- Recent advances use Pretrained Language Models (PLMs) for semantic feature extraction:
  - **Embedding-based methods**: focus on generating embedding vectors from documents and candidate keyphrases
    - EmbedRank (Bennani-Smires et al., 2018): uses Doc2Vec and Sent2Vec embeddings, cosine similarity for ranking
    - SIFRank (Sun et al., 2020): integrates ELMo embeddings and the SIF model to refine document embeddings
    - JointGL (Liang et al., 2021): applies similar strategy using BERT embeddings, incorporating local similarities derived from document graphs
    - AttentionRank (Ding and Luo, 2021): introduces cross-attention with BERT embeddings, generates document vectors using self-attention maps (SAMs) for further refinement
  - **Prompt-based methods**: use PLMs to predict keyphrases directly through prompt conditioning
    - PromptRank (Kong et al., 2023): estimates likelihood of candidate phrases using T5 PLM logits, specifically prompted for keyphrase extraction
  - **Self-attention-based methods**: focus on using SAMs from PLMs
    - SAMRank (Kang and Shin, 2023): aggregates attention vectors from a specific SAM to measure the attention a candidate phrase receives from other tokens

**Self-Attention Map**:
- Scaled Dot-Product Attention mechanism by Vaswani et al. (2017)
- Query (Q), Key (K), and Value (V) are linear transformations of the embedding representation of the same input sentence
- Softmax function assigns attention scores to tokens in Key based on relevance to Query, resulting in Self-Attention Map (SAM)
- SAM is used to compute weighted sum of Value vectors, transformed into new embedding representation for further processing by attention mechanism in subsequent layers.

## 3 Methodology

### 3.1 Attention-Seeker

**Four Main Steps:**
1. **Candidate Generation:**
   - Tokenize and POS tag words using the Stanford CoreNLP tool
   - Extract noun phrases ("NN", "NNS", "NNP", "NNPS", "JJ") using NLTK's RegexpParser
   - Define these phrases as document keyphrase candidates
2. **Self-Attention Maps (SAMs) Extraction:** 
3. **Estimation of Attention Scores:** Proposed new approach for estimating attention scores from all SAMs instead of relying on manual selection (parameter tuning)
4. **Scoring Candidate Phrases:** Importance scored based on attention scores

### 3.2 Extraction of the Self-Attention Maps

**Extraction of Self-Attention Maps (SAMs)**
- **Step 1**: Extract SAMs from each layer and head of Large Language Model LLAMA 3-8B using Huggingface library
- **Short Documents**: Input entire text directly into the LLM for SAM extraction
- **Long Documents**:
  - Input document's abstract into the LLM first
  - Divide remaining text into segments with consistent token count
  - Process each segment separately for SAM extraction
- **Handling Long Documents vs Short Documents**:
  - Different methods used depending on document length
- **Use of Huggingface Library**

### 3.3 Attention Scores Estimation: Short Documents

**Attention Score Estimation**

**Calculation of Attention Scores**:
- Determines relevance of SAMs (sentence attention mechanisms) based on relevance to candidate phrases
- Calculates weighted average of attention vectors across all relevant SAMs
- Final vector **B** is a weighted combination of attention scores from most relevant parts of SAMs

**Defining Hypothesized Vector**:
- Initial hypothesis: SAMs focused on keyphrases pay attention to candidate phrases (Equation 2)
- Ignore attention vectors for non-candidate tokens as they may introduce noise (Clark et al., 2019)

**Relevance of Attention Vectors**:
- Calculated as dot product between attention vector **Alh i** and hypothesized vector **H** (Equation 3)
- Set to **0** for tokens not belonging to candidate phrases

**Calculating Relevance Scores**:
- Average relevance score of individual attention vectors in a SAM (Equation 4)
- Final attention vector **B** is a weighted average of all attention vectors across all SAMs, with weights based on their relevance scores (Equation 5)

**Refining Attention Scores**:
- Obtained vector B converted into refined vector **B** by setting values corresponding to non-candidate tokens to **0**.

### 3.4 Attention Scores Estimation: Long Documents

**Long Document Attention Scores Estimation**
* Long documents are segmented into abstracts and equally sized segments
* For each segment:
  * Calculate relevance scores of attention vectors (Rlh) using average of relevance scores of corresponding attention vectors
  * Apply l1-normalization to weighted average of all SAMs in the segment (SAMs)
  * Obtain attention score vector Bs for the segment by summing rows in the average SAM
* Define segment relevance Ts as the dot product between attention score vector Bs and hypothesized vector Hs
* For candidate phrase calculation:
  * Sum attention scores Bi (corresponding to candidate tokens)
  * Divide final score by frequency for single word candidates
* Calculate final score of each candidate by summing its scores across all segments.

## 4 Experiments and Results
### 4.1 Datasets and Evaluation Metrics

**Benchmark Datasets for Keyphrase Extraction:**
* **Inspec**: Abstracts from scientific papers, average words: 135, sentences: 6, key phrases: 9 (Krapivin, 2003)
* **SemEval2017**: Paper abstracts, average words: 194, sentences: 7, key phrases: 17, unigrams: 25.7%, bigrams: 34.4% (Augenstein et al., 2017)
* **SemEval2010**: Full papers, average words: 8545, sentences: 312, key phrases: 6, unigrams: 17.8%, bigrams: 62.2% (Kim et al., 2010)
* **Krapivin**: Full papers, average words: 8154, sentences: 380, key phrases: 15, unigrams: 20.5%, bigrams: 53.6% (Krapivin et al., 2009)

**Performance Metrics:**
* Evaluated based on F1 score at the top 5 (F1@5), 10 (F1@10), and 15 (F1@15) keyphrases predicted by the model.

**Statistical Analysis:**
* Table presents statistical information about each dataset such as number of documents, average words, sentences, and key phrases.
* Evaluated methods include TF-IDF, YAKE, TextRank, SingleRank, TopicRank, PositionRank, EmbedRank (d2v and s2v), SIFRank, AttentionRank, MDERank, JointGL, PromptRank, Self-Attention Map-based Methods (SAMRank), and Attention-Seeker.
* Bold indicates the best performance and underline indicates two best performances on each dataset.

### 4.2 Baselines

**Baselines Used in Evaluation:**
- **Statistics-based methods**: TF-IDF (Spärck Jones, 2004), YAKE (Campos et al., 2020)
- **Graph-based methods**: TextRank (Mihalcea and Tarau, 2004), SingleRank (Wan and Xiao, 2008), TopicRank (Bougouin et al., 2013), PositionRank (Florescu and Caragea, 2017)
- **Embedding-based methods**: EmbedRank (Bennani- Smires et al., 2018), SIFRank (Sun et al., 2020), AttentionRank (Ding and Luo, 2021), MDERank (Zhang et al., 2022), JointGL (Liang et al., 2021)
- **Prompt-based methods**: PromptRank (Kong et al., 2023)
- **Self-attention-based models**: SAMRank (average of all SAMs extracted with LLAMA 3-8B model, non-parametric method)

**Implementation Notes:**
- To ensure fair comparison: implemented SAMRank using the average of all SAMs and non-parametric method.
- Omitted proportional score as it led to lower performance.

### 4.3 Results

**Performance Comparison of Attention-Seeker to Baseline Models:**
* **Table 2**: Summarizes performance of Attention-Seeker vs. baselines on Inspec, SemEval2010, SemEval2017, and Krapivin datasets.
* **Results:**
  + Attention-Seeker achieves SOTA on Inspec, SemEval2010, and Krapivin; second best on SemEval2017.
  + Non-parametric SAMRank performs well, highlighting effectiveness of Self-Attention Maps (SAMs) for unsupervised keyphrase extraction.
* **Long Document Datasets**: Performance of self-attention-based methods suggests promising direction for long document keyphrase extraction using SAMs from LLMs.
* **Consistent Outperformance of Attention-Seeker over SAMRank**: Demonstrates advantages of considering relevance of different SAMs and document segments.

**Comparison with PromptRank:**
* **PromptRank**: Remains best performing method on SemEval2017, but requires hyperparameter tuning for each specific benchmark.
* **Attention-Seeker**: Achieves second-best performance without parameter tuning.
* **Variability of Input Texts**: Attention-Seeker adapts to variations in SAMs triggered by input texts, positioning it as a robust and adaptable solution for unsupervised keyphrase extraction.

## 5 Ablation

**Impact of Attention-Seeker Components on Performance**

**Modules Tested**:
- Relevance score for each SAM attention vector (Slh)
- Relevance score for each SAM (Rlh)
- Filtering step applied to non-candidate tokens of Slh

**Performance Comparison**:
- Base method: All attention vectors and SAMs assigned equal relevance, no filtering applied
- Configurations incorporating modules: B (includes relevance scores), B+Slh+Rlh+f (applies filters)

**Findings for Short Documents**:
- Slh score provides the most significant improvements in performance
- Filtering attention vectors from certain tokens can be beneficial, as they may function as "non-ops" and cause random distribution of attention scores
- Current filtering mechanism may lead to slight performance degradation in some cases

**Findings for Long Documents**:
- Segmenting document into abstracts and equal-length segments slightly differs from base method
- Relevance score for each segment (Ts) provides the most significant improvements
- Refining hypothesized vector Hs further improves performance of Ts scores
- Some performance degradation compared to base model due to difference in length between abstract and other segments
- Further research needed to explore new methods for estimating segment relevance without compromising base model's performance.

**Relevance of SAMs**:
- Attention-Seeker's Rlh score identifies the most relevant SAMs for keyphrase extraction, eliminating the need for labeled data to select an optimal SAM
- Analysis reveals that SAMs from intermediate layers (9-15 out of 32) and a few from first/last layer contribute significantly to task performance.

## 6 Conclusion

**Attention-Seeker**:
- Self-attention-based method for unsupervised keyphrase extraction
- Eliminates need for manual parameter tuning
- Assumes that certain **Self-Attention Maps (SAMs)** are specialized to focus on most important phrases within a document
- Reframes keyphrase extraction as identifying crucial information in SAMs

**Long Documents**:
- Method extended to identify most relevant segments for attention score extraction

**Performance**:
- Attention-Seeker outperformed most baselines on four benchmark datasets
- Demonstrated effectiveness of approach

**Scoring Mechanism**:
- Scoring mechanism is simple, has potential for improvement
- Suggested improvements: more sophisticated methods for relevance estimation

**Future Research**:
- Explore more advanced relevance estimation methods to improve performance
- Optimizing these estimates may provide insights into internal LLM processes and contribute to efficient pretrained language model design.

## 7 Limitations

**Limitations of Attention-Seeker**

**1. Dot product approach**:
- Ineffective use of dot product between desired vector and attention vectors
- Can be improved using normalization techniques: l2-normalization, softmax normalization, or softmax with tempered scaling

**2. Simplistic hypothesized vectors (H)**:
- Initial implementation uses binary vectors based on candidate phrases
- May introduce noise into the attention distribution of candidates
- Future work could explore more sophisticated hypothesized vectors

**3. Requirement for abstracts**:
- Long document version requires the first segment to be an abstract
- Limits application to documents with abstracts only
- Negatively affected overall performance in ablation study
- Further research needed to define hypothesized vector (H) without relying on information from abstracts

**Possible solutions**:
- Use more effective methods for measuring alignment between attention vectors and hypothesized vectors
- Extract most important tokens from all document segments, assuming equal relevance
- Define hypothesized vector H based on these important tokens.

