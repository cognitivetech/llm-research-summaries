# LongKey: Keyphrase Extraction for Long Documents

Jeovane Honorio Alves and Cinthia Obladen de Almendra Freitas (PPGD Law, Pontifícia Universidade Católica do Paraná)

Contact: cinthia.freitas@pucpr.br

Jean Paul Barddal (PPGIa Informatics, Pontifícia Universidade Católica do Paraná)

Contact: jean.barddal@ppgia.pucpr.br

## Contents
- [Abstract](#abstract)
- [I Introduction](#i-introduction)
- [II Proposed Approach](#ii-proposed-approach)
  - [II-A Word Embedding](#ii-a-word-embedding)
  - [II-B Keyphrase Embedding](#ii-b-keyphrase-embedding)
  - [II-C Candidate Scoring](#ii-c-candidate-scoring)
- [III Experimental Setup](#iii-experimental-setup)
  - [III-A Datasets](#iii-a-datasets)
  - [III-B Experimental Settings](#iii-b-experimental-settings)
- [IV Results and Discussion](#iv-results-and-discussion)
  - [IV-A LDKP Datasets](#iv-a-ldkp-datasets)
  - [IV-B Unseen Datasets](#iv-b-unseen-datasets)
  - [IV-C Component Analysis](#iv-c-component-analysis)
  - [IV-D Performance Evaluation](#iv-d-performance-evaluation)
  - [IV-E Short Documents](#iv-e-short-documents)
- [V Conclusion](#v-conclusion)

## Abstract

**LongKey: Keyphrase Extraction from Lengthy Documents**

* Addresses the challenge of manually annotating vast documents in the era of information overload
* Proposes LongKey framework for extracting keyphrases from long-context documents using an encoder-based language model to capture intricacies
* Uses a max-pooling embedder to enhance keyphrase candidate representation
* Outperforms existing unsupervised and language model-based keyphrase extraction methods on LDKP datasets and six diverse, unseen datasets
* Demonstrates versatility and superior performance in keyphrase extraction for varied text lengths and domains.

## I Introduction

**Keyphrase Extraction (KPE)**
* Essential for effective information retrieval in vast data volumes
* Identifies representative keyphrases enhancing document comprehension, retrieval, and management
* Keyword vs keyphrase: interchangeable, applicable to terms of any length
* Categorized based on underlying principles: unsupervised methods (TF-IDF, RAKE, TextRank), supervised learning with pre-trained BERT embeddings (KeyBERT), and recent approaches (PatternRank, JointKPE, HyperMatch, GELF)

**LongKey: Keyphrase Extraction for Long Documents**
* Novel framework addressing challenges of keyphrase extraction in long documents
	+ Expands token support for encoder models like Longformer
	+ Introduces new strategy for keyphrase candidate embedding capturing context across the document

**Methodology (Section II)**
* Detailed explanation of the LongKey methodology

**Experimental Setup (Section III)**
* Description of experimental setup for evaluating LongKey on long documents

**Results and Discussion (Section IV)**
* Presentation and analysis of results obtained using LongKey on long documents

**Conclusion (Section V)**
* Summary of the study and its findings, including implications for future research in keyphrase extraction for long documents.

## II Proposed Approach

**LongKey Methodology**

The proposed methodology is named LongKey. It consists of three main stages: initial word embedding, keyphrase candidate embedding, and candidate scoring (see Figure 1). Each stage enhances the selection and evaluation of keyphrases.

Figure 1 illustrates the overall workflow for the LongKey approach.

![Figure 1](https://arxiv.org/html/2411.17863v1/x1.png)

### II-A Word Embedding

**Longformer Model for Long-Context Documents**

**Model**: Longformer model is used to generate embeddings for long-context documents. It supports extended contexts through:
- Sliding local windowed attention with a default span of 512 tokens
- Global attention mechanism

**Architecture**:
- Encoder-type language model
- Twelve attention layers, each producing output embedding size of 768
- Positional embedding size of 8,192 (duplicated from RoBERTa)

**Processing Long Documents**:
- Large documents are split into chunks (maximum size: 8192 tokens) for processing by Longformer
- Chunked embeddings are concatenated to create a unified representation of the entire text's tokens

**Document Representation**:
- Input document is converted to a numeric representation using a tokenizer
- Longformer model processes token representation to generate embeddings, capturing contextual details

**Global Attention**:
- Initial token ([CLS]) is designated for global attention
- Each document token attends to every other document token and vice-versa

**Encoding Process**:
- Encoder-type model generates token embeddings: E^T = {e_1,1, e_1,2, ... , e_N,M_N}
- Where e_i,j represents the embeddings of jth token from ith word in document D
- Each embedding e_i,j has a size of 768 (omitted for clarity)

### II-B Keyphrase Embedding

**LongKey Approach to Keyphrase Extraction for Long Documents**

**Keyphrase Embeddings**:
- Context-sensitive: same keyphrase can have different embeddings based on environment
- Combined into unique embeddings for each keyphrase candidate, taking into account document's thematic and semantic landscape
- Single embedding created for words with multiple tokens using only first token embeddings
- Word embeddings used as input to keyphrase embedding module

**Keyphrase Embedding Module**:
- Convolutional network used to construct embeddings for potential n-gram keyphrases
- N distinct 1-D convolutional layers, each with kernel size k corresponding to n-gram size, and no padding
- Generates keyphrase embeddings from pre-generated word embeddings

**Keyphrase Embedding Pooler**:
- Aggregates diverse embeddings of keyphrase candidate's occurrences into a single, comprehensive representation
- Max pooling operation used to emphasize most contextually significant keyphrases

**Candidate Embeddings**:
- Generated for each unique keyphrase found in the document, from unigrams to n-grams
- Maximized values from all occurrences of the keyphrase are combined into a singular, comprehensive representation (C^l)

**Overall Process**:
- Multiple instances of keyphrases are encapsulated into a cohesive representation, enhancing evaluation of relevance for accurate ranking.

### II-C Candidate Scoring

**LongKey Approach**

**Candidate Embeddings**:
- Each candidate embedding is assigned a ranking score
- Higher scores indicate more accurate representation of document's content
- Ranking and chunking losses optimized during training to align with ground-truth keyphrases

**Ranking Score**:
- Calculated using linear layer on candidate embeddings
- Single value representing the ranking score for each candidate keyphrase

**Chunking Score**:
- Calculated using linear layer on keyphrase embeddings
- Each keyphrase embedding has a singular chunking score

**Margin Ranking Loss**:
- Used to optimize each candidate's score through MR_loss function
- Enhances distinction between positive and negative samples by elevating true keyphrase scores

**Cross-Entropy Loss**:
- Used for keyphrase chunking, similar to JointKPE
- Represents the likelihood of a sample belonging to positive class (p+)
- Calculated using binary classification loss formula: BCE_loss

**LongKey Objectives**:
- Refines embeddings of keyphrase candidates for overall precision and contextual sensitivity in keyphrase extraction
- Distinct from JointKPE's focus on optimizing individual keyphrase instances' embeddings

## III Experimental Setup

**Empirical Evaluation of LongKey Method Summary**
- Overview of experimental datasets utilized
- Description of analysis configurations employed

### III-A Datasets

**Long Document Keyphrase Identification Datasets (LDKP)**
- **Large datasets** required for training language models and evaluating keyphrase extraction from input documents
- LDKP formulated specifically for extracting keyphrases from full-text papers

**Datasets in LDKP:**
- **LDKP3K**: Approximately 100 thousand samples, average of 6027 words per document (similar to KP20K)
- **LDKP10K**: Over 1.3M documents, averaging 4384 words per sample

**Additional Datasets:**
- **Zero-shot evaluation**: Assessing methods on different domains and patterns

**Datasets for zero-shot evaluation:**
- **Krapivin**: 2,304 full scientific papers from computer science domain (ACM)
- **SemEval2010**: 244 ACM scientific papers across distributed systems, information search, distributed artificial intelligence, and social sciences
- **NUS**: 211 scientific conference papers with keyphrases annotated by student volunteers
- **FAO780**: 780 documents from agricultural sector labeled by FAO staff using the AGROVOC thesaurus
- **NLM500**: 500 biomedical papers, annotated with terms from MeSH thesaurus
- **TMC**: 281 chat logs related to child grooming from Perverted Justice project

**Evaluating LDKP Datasets:**
- Compared on two popular short-context datasets: KP20k and OpenKP.

**Short Context Datasets:**
- **KP20k**: Over 500 thousand abstracts of scientific papers (validation and test subsets each)
- **OpenKP**: Over 140 thousand real-world web documents with human-annotated keyphrases

### III-B Experimental Settings

**Experiment Setup:**
- **Hardware**: NVIDIA RTX 3090 GPUs (24GB VRAM each)
- **Training Regimen**: AdamW optimizer, cosine annealing learning rate scheduler with a learning rate of 5x10^-5 and a warm-up for initial 10% iterations
- **Gradient Accumulation**: Achieved an effective batch size of 16 in the training phase to circumvent VRAM constraints
- **Maximum Token Limit**: 8,192 during training to accommodate document lengths
- **Positional Embedding**: Expanded to 8,192 for longer chunks in inference mode (tested up to 96K)
- **Keyphrase Length**: Limited to a maximum of five words (k=[1,5]) to maintain computational efficiency and align with standard practices

**Evaluation:**
- **Model Training**: Longformer model trained on LDKP3K for 25 thousand iterations, LDKP10K for 78,125 iterations (almost an entire epoch), and BERT model with chunking to extend training to 8,192 tokens
- **Performance Metric**: F1-score, harmonic mean between precision and recall, for the most significant K keyphrase candidates (F1@K)
- **Metrics Calculation**: Precision@K, Recall@K, and F1@K using the predicted keyphrases sorted by their ranking scores in a decreasing order and ground-truth keyphrases
- **Additional Evaluation**: F1@Best, which evaluates the K with the best harmonic mean between recall and precision for a specific method and dataset (threshold of K≤100 to not deviate strongly from default Ks)
- **Stemming**: The Porter Stemmer was employed for all experiments, but no lemmatization was applied. Both candidate and ground-truth keyphrases underwent stemming. Duplicated ground-truth keyphrases were cleaned, removing the possibility of duplicates improving F1-score.

## IV Results and Discussion

**Performance Analysis**

* Focuses on two main datasets for investigation
* Includes zero-shot learning scenarios and domain adaptability analysis
* Investigates the role of keyphrase embedding pooler, performance estimation, and inference in short-context documents

### IV-A LDKP Datasets

**Results on LDKP3K Test Subset:**
* LongKey: Keyphrase Extraction for Long Documents
	+ F1@5: **39.55%**
	+ F1@𝒪: **41.84%**
* All fine-tuned methods used BERT (on LDKP3K) and Longformer architecture
* LongKey8K identified without chunking, max of 8192 tokens
* Among evaluated methods, LongKey8K was the best performer
* LongKey maintained lead even with domain shift to LDKP10K: F1@5 = **31.94%**, F1@𝒪 = **32.57%**
* Performance metrics on LDKP10K test subset provided in Table I
* LongKey was the leading method with F1@5 of **41.81%**
* LongKey (LDKP3K) outperformed other models trained on same dataset but scored significantly lower on LDKP10K, indicating dataset-specific variations in effectiveness.

**Evaluation without Chunking:**
* LongKey identified without chunking, max of 8192 tokens as LongKey8K
* LongKey8K emerged as the best performer with F1@5: **39.55%**, F1@𝒪: **41.84%**

**Performance on LDKP10K Test Subset:**
* LongKey maintained lead even on broader LDKP10K test subset: F1@5 = **31.94%**, F1@𝒪 = **32.57%**

**Comparison with GELF:**
* GELF score reported without specific K value in its paper.

### IV-B Unseen Datasets

**LongKey Method Evaluation**

**Performance Across Diverse Domains**:
- LongKey outperformed other methods in nearly all tested datasets, except for SemEval2010 and TMC
- See Tables [II](https://arxiv.org/html/2411.17863v1#S4.T2) and [III](https://arxiv.org/html/2411.17863v1#S4.T3) for results on unseen datasets

**LDKP Training Dataset**:
- Choice of LDKP3K or LDKP10K training dataset significantly influenced performance across unseen datasets
- LDKP3K-trained models excelled in every dataset except NLM500
- LDPK3K had overall longer samples (average 6,027 words) compared to LDKP10K (average 4,384 words)
- Further studies encouraged to assess influence of study areas and sample size

**Balance between BERT and Longformer-based Methods**:
- Both showed potential for improvements regarding long-context encoders
- Figure 2 shows performance of LongKey and JointKPE on LDKP3K dataset, categorized by document length
- LongKey consistently outperformed other models in nearly all benchmarks, showcasing broad applicability and strength in keyphrase extraction across varied domains.

### IV-C Component Analysis

**Component Analysis: LongKey vs JointKPE**

**Assessing KEP contribution**:
- Evaluated LongKey approach with different aggregation functions: average, sum, maximum
- Compared improvement obtained using KEP (Keyphrase Embedding Pooler) compared to JointKPE
- Experimental settings: each model configuration underwent 12,500 iterations

**Results**:
- Table IV shows average and standard deviation results of five runs for each configuration
- **Overall JointKPE F1@5 score**: around 36 with a high std dev. of 0.50
- Using KEP (average reduction): F1@5 score of **29.15** , lower std dev. of 0.23
- Summation aggregator: improved F1@5 slightly to **32.76%±0.20**
- Max pooling: best F1@5 score of almost **39** with the lowest std. dev. of 0.07

**Impact of KEP**:
- Substantial impact on LongKey's success
- Appropriate reduction choice is crucial
- **Max aggregator** especially highlights salient features in different occurrences of a keyphrase, contributing to effective extraction of representative keyphrases.

### IV-D Performance Evaluation

**LongKey Performance Evaluation**

**Evaluation Metric**: Performance calculated based on documents per second using a single RTX 3090 GPU.

**Results** (Table V):
- LongKey: Keyphrase Extraction for Long Documents
    * Inferior to supervised methods due to keyphrase embedding pooler
    * Minimal performance loss compared to overall F1 score increase
- CPU-only methods marked with an asterisk (*)
- Longformer-based and BERT-based methods:
    * Inferior results in some cases
    * Slight boost in performance compared to LongKey

**Conclusion**:
- LongKey performs slightly inferior to supervised methods, but this loss is minimal compared to the overall F1 score increase.
- Encouraged to develop robust approaches with minimal bottlenecks.

### IV-E Short Documents

**Results and Discussion: Short Document Datasets**

**Competitive Methods**:
- RankKPE
- JointKPE
- LongKey

**Performance on KP20k and OpenKP Datasets**:
- JointKPE was superior on KP20k, which has a high correlation with LDKP3K
- Models trained on LDKP10K were generally better for OpenKP
- RankKPE and SpanKPE had similar results to the other methods for OpenKP

**Performance on Long-Context Datasets**:
- LongKey improvements not seen in short-context documents, except for TMC dataset
- Improvements likely related to proposed keyphrase embedding pooler
- LongKey may be more biased toward long-context documents

**Future Experiments**:
- Increase length and content variability in training stage to evaluate capabilities of keyphrase embedding pooler.

## V Conclusion

**LongKey: A Novel Framework for Keyphrase Extraction in Long Documents**

**Importance of Keyphrase Extraction**:
- Crucial for summarizing and navigating content within documents
- Prevalent methods fail to analyze long-context texts like books and technical reports comprehensively

**Introducing LongKey**:
- Novel keyphrase extraction framework specifically designed for the intricacies of extensive documents
- Robustness stems from its innovative architecture and rigorous validation on long-context datasets

**Validation and Results**:
- Simple component analysis and assessments of LDKP datasets
- Testing across 6 diverse, previously unseen long-context datasets and 2 short-context datasets
- Empirical results highlight LongKey's capability in long-context Key Phrase Extraction (KPE)
- Sets a new benchmark for the field and broadens the horizon for its application

**Choosing Appropriate Training Datasets**:
- Crucial for LongKey's performance on unseen data
- Highlights the need for strategic modifications to improve generalization without sacrificing keyphrase extraction effectiveness

**Limitations and Future Work**:
- Inferior results in short-context datasets indicate the necessity of improvements for better generalization
- Restriction on maximum keyphrase length focuses the method on specific lengths
- Need to explore accommodating longer keyphrases without sacrificing performance
- Context size limitation may restrict LongKey's ability to fully capture and process extensive document content
- Any plans to expand this limit must balance increased computational demands with available resources

**Conclusion**:
- LongKey sets a new benchmark in keyphrase extraction for long documents
- Combines adaptability with high accuracy across various domains
- Superior embedding strategy contributes to its effectiveness, suggesting significant potential for enhancing document indexing, summarization, and retrieval in diverse real-world contexts.

