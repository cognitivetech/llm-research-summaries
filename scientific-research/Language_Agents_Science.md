# LANGUAGE AGENTS ACHIEVE SUPERHUMAN SYNTHESIS OF SCIENTIFIC KNOWLEDGE

by Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, Andrew D. White

https://storage.googleapis.com/fh-public/paperqa/Language_Agents_Science.pdf

## Contents
- [LANGUAGE AGENTS ACHIEVE SUPERHUMAN SYNTHESIS OF SCIENTIFIC KNOWLEDGE](#language-agents-achieve-superhuman-synthesis-of-scientific-knowledge)
- [2 Answering scientific questions](#2-answering-scientific-questions)
- [3 Performance analysis of PaperQA2](#3-performance-analysis-of-paperqa2)
- [4 Summarizing scientific topics](#4-summarizing-scientific-topics)
- [5 Detecting contradictions in the literature](#5-detecting-contradictions-in-the-literature)
- [6 Conclusions](#6-conclusions)

## LANGUAGE AGENTS ACHIEVE SUPERHUMAN SYNTHESIS OF SCIENTIFIC KNOWLEDGE

**Language Agents Exceed Human Performance in Scientific Research**

**Background:**
- Language models (LLMs) have potential for scientific research but lack factuality, attention to detail, and appropriate benchmarks.
- Previous benchmarks don't compare human and AI performance on real tasks or consider entire literature.

**Approach:**
- Develop rigorous comparison methodology for retrieval, summarization, and contradiction detection tasks.
- Create PaperQA2 agent that exceeds PhD student/postdoc performance in these tasks.

**PaperQA2 Features:**
- Schematic: Figure 1A (agent's toolset and action representations).
- Performance: Figure 1B (exceeding human performance in question answering, summarization, contradiction detection).

**Retrieval Task:**
- PaperQA2 gathers evidence from entire literature using Search APIs.
- Collects background summaries and filters top responses based on score.

**Summarization Task:**
- Proposes answers with cited article summaries in Wikipedia style.
- Achieves 85.2% accuracy compared to human-written articles.

**Contradiction Detection Task:**
- Extracts claims from papers and checks them against literature.
- Identifies contradictions in biology papers, like rs1344706's effect on schizophrenia risk.

**Contributions:**
- First robust procedure for evaluating a single AI system across multiple real-world literature search tasks.
- PaperQA2 exceeds PhD student/postdoc performance in retrieval and summarization tasks.
- Applies PaperQA2 to contradiction detection task, identifying inconsistencies at scale.

## 2 Answering scientific questions

**LitQA2: Evaluating AI Systems for Retrieval from Scientific Literature**

**Creating LitQA2**:
- Set of 248 multiple choice questions designed to require retrieval from scientific literature
- Answers appear in main body of papers, not abstracts
- Ideally appear only once in all scientific literature
- Enforced criteria enable evaluation of response accuracy by matching system's cited source DOI with original creator's DOI
- Excluded questions where AI or human could answer using alternative sources

**Precision and Accuracy Metrics**:
- Precision: fraction of correct answers when a response is provided
- Accuracy: fraction of correct answers over all questions
- Recall: percentage of total questions with system's answer attributed to correct source DOI

**Designing an AI System for Scientific Literature: PaperQA2**
- **Retrieval-augmented generation (RAG)**: current paradigm for eliciting factually-based responses from LLMs
- RAG provides additional context, but identifying correct snippet is a challenge
- **PaperQA2**: RAG agent that decomposes RAG into tools to revise search parameters and generate/examine candidate answers
- **Paper Search Tool**: transforms user request into keyword search to identify candidate papers
- **Gather Evidence Tool**: ranks paper chunks using dense vector retrieval, LLM reranking, and contextual summarization (RCS)
- RCS prevents irrelevant chunks from appearing in RAG context by summarizing and scoring relevance
- **Generate Answer Tool**: uses top ranked evidence summaries inside prompt to an LLM for final response
- **Citation Traversal Tool**: exploits citation graph as a form of hierarchical indexing to add additional relevant sources

**Performance Evaluation of PaperQA2**:
- Parsed and utilized an average of 14.5 ± 0.6 papers per question on LitQA2
- Precision: 85.2 percent ± 1.1 percent (mean ± SD, n = 3)
- Accuracy: 66.0 percent ± 1.2 percent (mean ± SD, n = 3)
- System chose "insufficient information" in 21.9 percent ± 0.9 percent of answers
- Outperforms other RAG systems on the LitQA2 benchmark in both precision and accuracy
- All RAG systems tested, with exception of Elicit, outperform non-RAG frontier models in both precision and accuracy
- Accuracy on new set of 101 questions did not differ significantly from original set of 147 questions, indicating optimizations generalized well to new questions.

## 3 Performance analysis of PaperQA2

**Performance Analysis of PaperQA2**

**Non-Agentic Version**:
- Created a non-agentic version (No Agent) that had hard-coded actions
- This version had significantly lower accuracy than the agent-based system
- Attributed the performance difference to the agent's better recall, which allows it to return and change keyword searches after observing the amount of relevant papers found

**Components and Categories**:
- PaperQA2 performance on LitQA2 across different technologies (Figure 1B)
- Performance studies and ablations across component categories (Figure 1C, Figure S5)
- **Search Recall**: DOIs found via "Paper Search" or "Citation traversal" tools
- **Top-k Ranking**: All DOIs with similarity rankings below the top-k cutoff (30)
- **RCS**: All DOIs selected by RCS
- **Attribution**: All DOIs cited in the "Generate Answer" tool

**Error Bars and Mean**:
- LitQA2 runs had 1.26 ± 0.07 (mean ± SD) searches per question, and 0.46 ± 0.02 (mean ± SD) citation traversals per question
- This shows that the agent will sometimes return to an additional search or traverse the citation graph to gather more papers

**Contexts in "Generate Answer" Tool**:
- The number of contexts or text chunks included in the final text generation step is a balance between improving accuracy and increasing irrelevant context
- Varying the number of contexts from 15 to 5 showed that 15 gave highest precision but 5 gave highest accuracy

**Citation Traversal Tool**:
- Hypothesized that papers found as either citers or citees of existing relevant chunks would be an effective form of hierarchical indexing
- Ablation of "Citation Traversal" tool showed increased accuracy (t(2.55) = 2.14, p = 0.069) and significantly increased DOI recall (t(3) = 3.4, p = 0.022) at all stages of PaperQA2
- This tool mirrors the way scientists interact with the literature

**Variations in "Generate Answer" Tool and RCS**:
- Performed experiments varying the model choice for "Generate Answer" tool and the RCS step in "Gather Evidence" tool
- No RCS Model ablation showed that adding RCS to a traditional RAG text generation step significantly increases retrieval accuracy (t(3.92) = 9.29, p < 0.001)
- However, smaller models (GPT-3.5-Turbo and Llama3) decreased overall accuracy when used for RCS relative to not using a model at all
- This indicated there is a comprehension threshold that must be met for effective summarization and relevance evaluation
- GPT-4-Turbo significantly outperformed other models on LitQA2 accuracy when used in the RCS step (t(3.47) = 6.14, p = 0.003)
- Claude-Opus had the highest LitQA2 precision, though it was not a significant increase over Gemini-1.5-Pro and GPT-4-Turbo

**Lowering Top-k Ranking Depth**:
- Significant increase in accuracy with increasing depth (i.e. more document chunks entering into the RCS ranking) from 1 to 10 (t(2.15) = 5.44, p = 0.014)
- Diminishing performance gains from 10 to 30 (the default)
- Having a deep (>10) RCS ranking list with a high-performing LLM is crucial in achieving human-level accuracy on LitQA2

**Parsing Quality and Chunk Sizes**:
- Parsing quality did not significantly increase precision, accuracy, or recall on LitQA2
- This is likely specific to being a retrieval task, as there is often only a single passage needed from a paper's body
- Anecdotally, better parsings were crucial for extracting data from tables in WikiCrow

## 4 Summarizing scientific topics

**WikiCrow System for Generating Wikipedia-Style Articles on Protein Coding Genes**

**Overview:**
- WikiCrow generates articles on human protein coding genes using PaperQA2 calls for topics like structure, function, interactions, clinical significance
- Average article length: 1,219 words (vs. 889.6 for Wikipedia)
- Cost per article: $4.48 (including search and LLM API costs)
- Compared against human-written Wikipedia with expert grading on statements

**Performance:**
- Fewer "cited and unsupported" statements in WikiCrow than Wikipedia (13.5% vs. 24.9%)
- Lower uncited statement rate for WikiCrow (3.5% vs. 13.6%)
- Higher precision of cited statements in WikiCrow (86.1% vs. 71.2%)
- Fewer reasoning errors but similar attribution errors compared to Wikipedia

**FAM83H Gene:**
- Located on chromosome 8 in humans, encodes a pseudoenzyme protein with a PLD-like domain
- Regulates keratinization in epithelial cells for skin formation and integrity
- Linked to autosomal dominant hypocalcified amelogenesis imperfecta (ADHCAI), a genetic disorder affecting enamel development
- Interacts with various proteins, essential for cellular processes like recruiting CK-Ia to keratin filaments.

## 5 Detecting contradictions in the literature

**Contradiction Detection in Scientific Literature**

**Background:**
- PaperQA2 used to detect contradictions at scale
- Many versus many problem instead of one versus many
- Previous work focused on claim verification or fact checking, often in news contexts
- ContraCrow system built using PaperQA2 for automatic contradiction detection

**ContraDetect Benchmark:**
- Derived from LitQA2 with incorrect and correct statements
- Evaluated ContraCrow's ability to detect contradictions
- Obtained ROC curve and AUC of 0.842, setting threshold at 8 resulted in 73% accuracy, 88% precision, 7% false positives

**Evaluation:**
- Demonstrated ability to handle "no-evidence statements" with high accuracy
- Applied to set of 93 biology papers identifying an average of 35.16 claims per paper and 2.34 contradictions per paper (setting Likert scale threshold at 8)
- Human expert evaluation agreed on 70% of claims, F1 score 0.82

**Concerns:**
- Annotators might exhibit bias towards agreeing with the model or being influenced by its reasoning
- Further evaluated ContraCrow through "contradiction detection" task, experts agreed with each other more than with ContraCrow (75.5% vs 60.42%)
- Overconfidence on ContraCrow's part may be a primary driver of lack of agreement with human annotators.

## 6 Conclusions

**PaperQA2 System: Conclusions**
- Developed methodology to compare AI systems against human performance for scientific research
- PaperQA2 outperforms human experts on answering questions across all scientific literature
- Produces summaries that are, on average, more factual than Wikipedia summaries
- Can be deployed to identify contradictions in scientific literature at scale
- Hardest task reported by a human expert who performed the task was the contradiction detection task
- PaperQA2 is expensive compared to lower accuracy commercial systems but inexpensive in absolute terms ($1-$3 per query)
- Scaling up PaperQA2 and other literature-enabled agents like WikiCrow and ContraCrow empowers greater use of latent knowledge in literature.

**Data Availability:**
- Code for replicating results or modifying the algorithm is available on Github via paperqa.
- Data including evaluator responses, contradiction detection claims, litQA questions, and WikiCrow candidate statements are provided in supplementary materials.
- All generated WikiCrow articles for this study are available in a public Google Cloud bucket: https://storage.googleapis.com/fh-public/wikicrow2/.

