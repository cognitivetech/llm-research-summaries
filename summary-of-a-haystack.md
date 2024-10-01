# Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems

by Philippe Laban, Alexander R. Fabbri, Caiming Xiong, Chien-Sheng Wu
https://arxiv.org/html/2407.01370v1

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Summary in a Haystack Framework](#3-summary-in-a-haystack-framework)
- [4 Evaluation Protocol](#4-evaluation-protocol)
- [5 Results](#5-results)
  - [5.2 Benchmark Results](#52-benchmark-results)
  - [5.3 Estimating Human Performance](#53-estimating-human-performance)
  - [5.4 Position Bias Sensitivity](#54-position-bias-sensitivity)
- [6 Discussion](#6-discussion)
- [7 Conclusion](#7-conclusion)

## Abstract 
- Evaluating long-context tasks like Needle-in-a-Haystack is challenging due to lack of complexity
- Summarization can play a central role in such evaluation
- Procedure designed to synthesize Haystacks, ensuring specific insights repeat across documents
- "Summary of a Haystack" (SummHay) task requires system to process the Haystack and generate summary identifying relevant insights and precisely citing source documents
- Automatic evaluation can score summaries on two aspects - Coverage and Citation
- Large-scale evaluation of 10 LLMs and 50 RAG systems in conversation and news domains

**Key Findings**:
- SummHay is an open challenge for current systems, as they lag human performance (56%) by 10+ points on a Joint Score
- Long-context LLMs like GPT-4o and Claude 3 Opus score below 20% without a retriever on SummHay
- SummHay can also be used to study enterprise RAG systems and position bias in long-context models

**Conclusion**:
- The authors hope future systems can equal and surpass human performance on SummHay.

## 1 Introduction

**Long-Context Language Models vs. Retrieval Augmented Generation (RAG)**
* Long-context language models: capable of processing sequences of hundreds of thousands to millions of tokens
* Recent progress: Beltagy et al. (2020), Raffel et al. (2020), Lewis et al. (2019) vs. latest models like Claude-3 and Gemini-1.5-pro
* Evaluation challenge: Needle-in-a-Haystack task not complex enough for differentiation
* Proposed evaluation method: Summarization as testbed
* Summarization requires reasoning over long context, understanding content importance
* Prior work limited to single documents or around 10k tokens, e.g., Laban et al. (2020), Fabbri et al. (2021a), Bhandari et al. (2020), Liu et al. (2022)
* Evaluation limitations: low-quality reference summaries and automatic metrics that don't correlate with human judgments
* First contribution: generating Haystacks for conversations and news articles using data synthesis programs
	+ Carefully designed pipeline to ensure feasibility and validity of the task
	+ Each Haystack contains approximately 100 documents on a topic, totaling around 100k tokens
	+ Total of 10 Haystacks with roughly 10 queries for a sum of 92 SummHay tasks
* Second contribution: developing SummHay's evaluation protocol focusing on Coverage of reference insights and Citation quality
	+ Manual annotation shows strong reproducibility among knowledgeable annotators (0.77 correlation)
	+ LLM-based evaluation finds lower level of correlation but reduced evaluation cost by a factor of almost 50
* Third contribution: estimating human performance on SummHay and large-scale evaluation of 50 RAG systems and 10 long-context LLMs
	+ Findings: all models significantly below human performance, even with oracle signals; trade-offs between RAG pipelines and LLMs; advanced RAG components boost end-to-end performance; lost in the middle phenomenon confirmed.
* Open-sourcing dataset and evaluation methodology.

## 2 Related Work

**Long-Context Summarization Evaluation**

**1. Related Work: Long-Context Evaluation**

**Diagram Illustrating Steps to Synthesize a Haystack of Documents:**
- Subtopic and insight creation followed by document generation

**Evaluation in Long-Context Settings**
- Existing work focused on relevance or coverage evaluation for short input, single document setting
- Recent studies: coherence in book summarization, content selection for book summarization, correlating LLM metrics in coverage

**Long-Context LLMs Evaluation:** Needle-in-a-Haystack
- Assessing long-context recall ability of LLMs
- Effects analyzed: needle placement, multi-needle, modal variations, and data contamination
- Synthetic data creation for long input tasks: Bai et al. (2023), Shaham et al. (2023)

**Attribution Evaluation:** Long Input Summarization
- Several benchmarks studied ability of LLMs to ground generation with citations
- AttributionBench aggregates 7 existing attribution datasets
- Previous studies: Hagrid, AttrEval-GenSearch, knowledge graphs, long-form question answering
- Seahorse: study on summary attribution in short context setting.

**Our Paper:**
- Long input summarization with synthetic data for evaluation
- Focus on relevance, long context, and attribution (citation) evaluation.

## 3 Summary in a Haystack Framework

**Preliminaries**
- **Long-form answer generation task**: Haystack (Topic + Subtopics & Insights)
- Similar to long-form question answering and query-focused summarization tasks

**Haystack Synthesis**
1. **Subtopic and Insight Generation**:
   - Controls information distribution in documents
   - Topic: "three students discuss strategies for an upcoming exam"
   - Subtopics: distinctive, unique, expandable into 3 distinct insights
   - Quality assurance ensures satisfaction of requirements
2. **Document Generation**:
   - One document at a time generated with selected insights
   - Instructions for generating a document including all selected insights
   - Variable number of insights per document based on domain (750 words or roughly 1,000 tokens)
   - Quality assurance ensures realistic and unique documents
3. **Evaluation**:
   - Precise knowledge of mapping between insights and documents is crucial for task evaluation
   - Five domain-specific verification processes during synthesis to ensure sound expected mapping
   - Manual inspection and high human annotator performance provide evidence of quality

**Summary of a Haystack Task**
1. **Subtopic Query Generation**: transform each subtopic into a query (e.g., "What do the students discuss regarding stress management?")
2. **System Instructions**: generate summary to answer query, bullet-point format, number of bullet points matches number of insights for that subtopic, cite sources in bracketed format using document identifiers provided in Haystack
3. **Benchmark**: two domains (conversations and news) with 5 Haystacks each, average length: 93k tokens, consists of on average 9.20 subtopics, each averaging 6.75 insights, totaling 62 insights per topic
4. **Evaluation Metrics**: Coverage correlation, linking accuracy, and evaluation cost (manual annotation vs automated methods)

**Table 1: Reproducibility and cost comparison of manual and automated evaluation for SummHay.**
| Method | Coverage Correlation | Link Accuracy | Cost ($) | Manual Annotation |
|---|---|---|---|---|
| Gemini-1.5-pro | 0.751 | 89.3 | $15.1 | N/A |
| GPT-4o (9FS) | 0.719 | 89.2 | $26.1 | N/A |
| GPT-4o | 0.716 | 88.9 | $6.9 | N/A |
| Claude 3 Opus | 0.677 | 87.9 | $23.8 | N/A |
| Claude 3 Haiku | 0.498 | 87.7 | $0.4 | N/A |
| GPT3.5 | 0.495 | 86.7 | $1.3 | N/A |

## 4 Evaluation Protocol

**Evaluation Protocol for Summarization Models**

**Coverage Metric**:
- Measures overlap between candidate subtopic summary bullet points and reference insights
- Iterate over each reference insight, assessing whether it is fully, partially, or not covered in the bullets
- For each insight, the summary receives a score of 100 for full coverage, 50 for partial coverage, and 0 otherwise
- The final **Coverage Score** is the average of all insights' scores (ranges from 0 to 100)

**Citation Metric**:
- Given a partially or fully covered reference insight, measure precision and recall between generated citations and gold-standard cites
- **Citation Score** is calculated as F1 score of precision and recall
- A system must be both precise and thorough in its citing to achieve high Citation Score
- The **Citation Score** of an entire summary is the average insight's Citation Score

**Joint Metric**:
- Pieces together Coverage and Citation Scores
- Calculated by iterating over each reference insight, multiplying its coverage score and citation scores (assigning 0 citation if not covered)
- Joint Score of a summary ranges from 0 to 100

**Evaluation Metrics Table**:
| Summarizer | Rand | Vect | LongE | KWs | RR3 | Orac | Coverage | Citation | Joint | #Words/Bullet |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT3.5 | 36.2 | 45.8 | 46.0 | 48.4 | 51.9 | 56.2 | 57.9 | 50.3 | 21.8 | 11-12 words/bullet |
| Claude 3 Haiku | 9.3 | 15.2 | 15.0 | 15.9 | 16.8 | 23.0 | 62.3 | 5.5 | 34.1 | 6-7 words/bullet |
| GPT4-turbo | 28.2 | 36.2 | 31.1 | 37.9 | 40.4 | 53.8 | 57.9 | 30.9 | 35.1 | 8-9 words/bullet |
| Command-r | 17.1 | 24.6 | 24.0 | 25.7 | 29.3 | 33.9 | 37.8 | 16.2 | 22.3 | 7-8 words/bullet |
| Gemini-1.5-flash | 21.0 | 31.6 | 32.8 | 43.6 | 49.7 | 51.7 | 59.4 | 32.8 | 25.5 | 7-8 words/bullet |
| Command-r+ | 44.2 | 56.4 | 53.1 | 56.2 | 58.9 | 61.0 | 50.3 | 51.6 | 34.4 | 9-10 words/bullet |
| Claude 3 Sonnet | 55.8 | 70.6 | 69.7 | 72.1 | 73.1 | 77.7 | 73.6 | 76.2 | 43.4 | 10-11 words/bullet |
| Claude 3 Opus | 56.5 | 72.4 | 69.6 | 72.5 | 76.5 | 81.4 | 76.2 | 72.8 | 40.3 | 9-10 words/bullet |
| GPT-4o | 54.0 | 67.1 | 67.8 | 66.6 | 70.4 | 76.6 | 66.1 | 72.3 | 45.9 | 10 words/bullet |
| Gemini-1.5-pro | 53.0 | 63.5 | 64.9 | 63.6 | 68.4 | 67.6 | 70.0 | 72.0 | 44.8 | 10 words/bullet |
| Human Perf. | - | - | - | - | - | - | 74.5 | N/A | N/A | N/A |

## 5 Results

**Experimental Settings**

**Long-Context LLMs**:
- Evaluating: long-context LLMs that directly access the full Haystack and RAG systems
- Longer than an individual Haystack

**Participating LLMs**:
- Cohere’s Command-R and Command-R+
- Google’s Gemini-1.5-pro and Gemini-1.5-flash Reid et al. (2024)
- OpenAI’s GPT4-turbo and GPT-4o
- Anthropic’s Claude3 models: haiku, sonnet, and opus
- Including GPT-3.5 exclusively in the RAG setting

**Retrieval-Augmented Summarization (RAG)**:
- Reduce Haystack input size by filtering with a retriever
- **Retrievers**:
  - Receive query and all Haystack documents
  - Produce query relevance score for each document
  - Documents sorted in reverse order based on relevance score
  - Select first 15k tokens (for generators with 16k context window)
- **Retriever implementations**:
  - KWs: number of overlapping keywords between document and subtopic query, extracted using NLTK
  - Embedding methods: Vect (SentenceTransformers), LongE (extends standard embedders for longer inputs), Rerank3 (Cohere)
- **Lower and upper bounds**:
  - Rand baseline: random relevance scores
  - Oracle: number of subtopic insights in a document

**Figure 1 (right)**: Illustration of long-context LLMs and RAG systems.

### 5.2 Benchmark Results

**Benchmark Results for Summarization Models**

**Summary:**
- Coverage scores range from 36.2% to 81.4%, impacted by choice of retrieval method
- Oracle retriever achieves best coverage with most summarizers, but not necessary for strong performance
- Citation scores improve with better retrieval, except for Gemini-1.5-pro in full context setting
- Joint score provides complete system performance on SummHay test

**Impact of Retrieval:**
- Oracle retriever evaluates all models' capabilities
- RR3 retriever outperforms simpler ones due to advanced features for enterprise search and RAG settings
- Room for improvement in RAG systems as they underperform Oracle setting

**Model Performance:**
- Claude3 Opus, GPT-4o, and Gemini-1.5-pro are top performers with Joint Scores between 30.8 and 36.0
- Different trade-offs in Coverage, Citation: room for growth in a system achieving high Coverage and citation quality

**Average Bullet Point Length:**
- Average bullet point length varies among systems (29.7 to 38 words)
- Succinct methods can achieve strong performance on SummHay

**Citation Scores Analysis:**
- No system excels at both precision and recall but trade-offs observed, e.g., Claude models favor higher precision over recall.

**Additional Information:**
- Appendix A.3 confirms that verbosity does not bias evaluation.
- Appendix A.5 breaks down Citation Scores into Precision and Recall.

### 5.3 Estimating Human Performance

**Estimating Human Performance on SummHay Task:**
* Estimated by recruiting two annotators for task performance
* Participants performed in Oracle setting, reducing text volume by a factor of 5-6
* Sessions conducted for conversational and news domains, representing unbiased estimate
* Figure 3 aggregates results across ten sessions, showing progress during sessions (Figure)
* Citation Score corresponds to F1 measure; Precision: avg. 80.0, Recall: rising steadily
* Human performance outperforms LLMs and RAG systems in Table 2 (56.1 vs. 44.6)
* Reader should not consider human performance as an upper bound but a reference point for achievable performances.

**Document Order Summarizer:**
* Three different document orders for evaluating performance: Top, Bottom, Random
* Results in Table 3 from various LLMs (GPT-4o, Claude3, Opus, Gemini) based on document order.

### 5.4 Position Bias Sensitivity

**Position Bias Sensitivity:**
* **Full Context Experiment Results**: Documents in Haystack ordered arbitrarily, relevant ones found at top, middle, and bottom portions (Table 2)
* **Prior Work**: Models exhibit position bias, focusing on extremities of context window (Huang et al., 2023; Chang et al., 2023; Chen et al., 2023b; Ravaut et al., 2023)
* **Position Bias Experiment**: SummHay framework used to study position bias (Table 3)
	+ Top three models ran on sorted Haystacks with relevant docs at Top or Bottom of context window
	+ All models exhibit position bias: GPT-4o and Claude3 Opus favor bottom, Gemini-1.5-pro top
* **Position Sensitivity Score**: Maximum absolute difference in Joint Score between Random ordering and Top/Bottom conditions (future systems should aim for minimal sensitivity)

## 6 Discussion

**Discussion**

**Quality Control:**
- Data pipeline details: insight verification, inclusion checks (Section 3.2 & Appendix A.1)
- Preventing overlap, ensuring insights presence
- Errors may occur due to LLMs in data synthesis
- Human performance simplified setting vs Oracle setting

**Assumptions in Data Synthesis:**
- Independent document generation: no dependencies or cross-references
- Realistic assumptions can increase difficulty and improve accuracy (future work)

**Controlling for Verbosity:**
- Specifying desired number of insights
- Longer summaries may result in higher coverage
- More difficult task without controlling verbosity
- Trade-offs between verbosity, human preference, and overall scores for future study

**Evaluation Metrics:**
- Room for improvement in coverage and linking evaluation
- Gemini-1.5-pro: rate limit, more costly than GPT-4o (Appendix A.3)
- Chen and Eger (2023), Liu et al. (2023): untested non-LLM based NLI metrics
- Potential improvement in future studies with different evaluation methods

**Model Choice:**
- Closed-source models generally outperform open-sourced models, but performance comparison task-dependent (Chen et al., 2023a)
- Exclusion of high-performing open-sourced models due to minimal context window for RAG experiments
- Balancing reduction of input size and feasibility of the task with up to 15k tokens in RAG setting.

## 7 Conclusion

**Conclusion**
- Introduced SummHay benchmark task to assess long-context LLMs and RAG systems ability to precisely summarize large sets of documents
- Reveals current models struggle with this task, even in oracle document settings
- Believes SummHay provides a robust framework for evaluating long-context systems and driving progress towards human performance

**Ethical Considerations**
- Project reflects English-speaking culture and may contain biases from datasets used
- Ensured fair compensation, communication, pace, and ability to quit for professional annotators
- Verified all datasets and models are publicly released with proper permission for research use.

