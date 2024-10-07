# Towards Efficient Methods in Medical Question Answering using Knowledge Graph Embeddings

by Saptarshi Sengupta, Connor Heaton, Suhan Cui, Soumalya Sarkar†, Prasenjit Mitra
https://arxiv.org/pdf/2401.07977

## Contents
- [Abstract](#abstract)
- [I. INTRODUCTION](#i-introduction)
- [II. RELATED WORK](#ii-related-work)
- [III. PROPOSED METHODOLOGY](#iii-proposed-methodology)
- [IV. RESULTS](#iv-results)
- [V. DISCUSSION](#v-discussion)
- [VI. CONCLUSIONS AND FUTURE WORK](#vi-conclusions-and-future-work)

## Abstract
**Background**:
- Natural Language Processing (NLP) task: Machine Reading Comprehension (MRC)
- Modern language models like BioBERT, SciBERT, and ChatGPT trained on medical corpora for medical domain
- In-domain pre-training expensive in terms of time and resources

**Proposed Approach**:
- Resource-efficient method to inject domain knowledge into a model without relying on such domain-specific pre-training
- Use of Multi-Layer Perceptrons (MLPs) for aligning and integrating embeddings extracted from medical knowledge graphs with the embedding spaces of pre-trained language models (LMs)
- Aligned embeddings fused with open-domain LMs BERT and RoBERTa fine-tuned for MRC tasks: span detection (COVID-QA) and multiple-choice questions (PubMedQA)

**Comparison to Prior Techniques**:
- Compare method to techniques relying on vocabulary overlap for embedding alignment
- Circumvent requirement of vocabulary overlap to deliver better performance
- Allow BERT/RoBERTa to perform on par (occasionally exceed) or show improvements in general over prior techniques

**Conclusion**:
- Signal an alternative method to in-domain pre-training for domain proficiency.

## I. INTRODUCTION

**Machine Reading Comprehension (MRC)**
- Machine Reading Comprehension: model answers a question based on context
- Requires identifying entities, supporting facts, question intent
- LLMs like ChatGPT expected to advance in MRC performance but struggle with question answering tasks across domains
- BERT models more capable for MRC in medical domain due to massive pre-training on unlabelled corpora (expensive)
- Alternative: using Knowledge Graph Embeddings (KGE) as domain knowledge injection

**Proposed Approach:**
- Fuse entity KGE into question representation during fine-tuning phase for MRC
- Vocabulary overlap not required unlike existing approaches
- Homogenization technique inspired by work on feed-forward neural networks (FFNNs) to align embeddings spaces

**Challenges:**
- Existing approaches rely on vocabulary overlap between knowledge graph entities and language model vocabularies
- Domain terms may span multiple subwords in a language model, requiring homogenization technique for alignment

**Contributions:**
1. Proposed domain-agnostic strategy using FFNNs to align embeddings spaces without relying on vocabulary overlap or the presence of phrases or pseudo-words (entry-cluster, related_to).
2. Demonstrated that open-domain models can perform similar to domain-specific models by avoiding expensive pre-training over large in-domain corpora or showing improvements with method's homogenization technique.
3. Released a cleaned version of the COVID-QA dataset for future research.

## II. RELATED WORK

**Related Work**

**Text Integration**:
- Inject dictionary definitions for rare words using a custom loss function
- Incurs overhead by additional pre-training

**Knowledge Graph Triples**:
- Define as information tuples (subject, predicate, object) in a pseudo-language
- COMET: Trained on knowledge graph triples for commonsense reasoning, but limits generalization to natural text
- K-BERT: Expands identified entities with one-hop knowledge graph-triples and fine-tunes BERT using updated representations

**Embedding Integration**:
- Use external embeddings trained on a relevant domain for fine-tuning transformer models
- **E-BERT**: Fine-tuned BERT using external embeddings for question-answering, but requires substantial overlap between entities in the knowledge source and LM's vocabulary
- **Medical Inference**: Concatenated external KGE embeddings with BioELMo embeddings and used with ESIM model

**Embedding Alignment**:
- Linear objective function for cross-lingual embedding alignment proposed by Mikolov, et al.
- Requires mapping dictionary and common dimensionality between source and target embeddings
- Challenges in integrating embeddings from knowledge bases due to differences in vocabulary and word tokens

## III. PROPOSED METHODOLOGY

**Methodology for COVID-QA and PubMedQA**

**Overall Pipeline**:
- Entity linking
- KGE homogenization
- Definition embedding generation
- Fine-tuning with external knowledge infusion

**Resources Used**:
1. **COVID-QA**: SQuAD style dataset with 2,019 question-answer pairs based on 147 scientific articles and annotated by 15 biomedical experts.
2. **PubMedQA**: Multiple-choice QA performed on the PubMedQA benchmark, which has a collection of 1k expert-annotated instances of yes/no/maybe biomedical questions.
3. **UMLS**: Metathesaurus (a collection of various biomedical terminologies) from the UMLS to extract entity definitions.
4. **Pre-Trained UMLS KGE**: Trained 3.2M 50-dimensional entity embeddings using knowledge graph triples from the UMLS metathesaurus and semantic network 2.
5. **MetaMap**: Used as entity identifier/linker, works in tandem with UMLS to break down input sentences according to UMLS entities.

**Preprocessing (COVID-QA Cleanup)**:
- Identified syntactical and encoding issues: excess spaces, missing spaces, uncapitalized acronyms, repeated words, spelling mistakes, and grammatical issues.
- Used Grammarly for identifying 1020 questions (50.5%) with these issues and National Laboratory of Medicine’s replace_UTF8 tool to address Unicode characters.

**Entity Linking**:
- Ran MetaMap on COVID-QA and PubMedQA questions, revealing 1,897 and 2,782 entities respectively.
- Among the common entities with pre-trained KGEs, only 1,452 and 2,078 had definitions in the metathesaurus and were chosen for homogenization.

**KGE Homogenization**:
- Proposed a method to learn homogenized RdLM vectors from RdKGE ones using an FFNN with a single hidden layer and dropout regularization.
- Trained on 10,000 samples from the pre-trained KGE without overlap with COVID-QA and PubMedQA dataset entities.

**Definition Embeddings**:
- Hypothesized that KGEs alone would not lead to significant performance gains, so incorporated entity definitions for added external knowledge.
- Vectorized definitions by passing them through respective LMs in a feature extraction mode using the model-specific pooler output.

**Fine-Tuning with Improved Question Representation**:
- Obtained two embeddings per entity: homogenized KGE and definition embedding.
- Average the two embeddings to form the final external knowledge vector instead of adding them separately.
- Explored BERTRAM concatenation (add external embeddings using / separator) and DEKCOR concatenation (concatenate without tampering the original text).

## IV. RESULTS

**COVID-QA and PubMedQA Experiment Results**

**COVID-QA**:
- Results from experiments with models fine-tuned on SQuAD (COVID-QA) and SNLI (PubMedQA) are presented
- Two general-purpose models, BERTBASE and RoBERTaBASE, as well as domain-specific Bio/Sci-BERT models were used
- Models were first trained on SQuAD or SNLI before being fine-tuned for COVID-QA or PubMedQA, respectively
- Metrics reported: average F1 and EM for COVID-QA, and average accuracy and F1 for PubMedQA across all folds
- Baseline results using Mikolov's E-BERT strategy were also provided
- Randomizing external embeddings before fine-tuning was conducted to gauge the model's attention towards additional knowledge signals
- **DEKCOR concatenation** of KGE and definition embedding led to best performance for non-domain-specific variants of BERT and RoBERTa, improving F1 and EM scores over regular fine-tuning
- Non-domain-specific models outperformed the Mikolov baseline, with improvements in F1 (0.6% - 1.2%) and EM (0.4% - 3.1%)
- Domain-specific models, BioBERT and SciBERT, showed some improvement over vanilla fine-tuning but did not outperform the non-domain-specific models

**PubMedQA**:
- Results for PubMedQA experiments with models fine-tuned on SNLI are presented
- As in COVID-QA, two general-purpose models (BERTBASE and RoBERTaBASE) and domain-specific Bio/Sci-BERT models were used
- Models were first trained on SNLI before being fine-tuned for PubMedQA
- Metrics reported: average accuracy and F1 for PubMedQA across all folds
- Baseline results using Mikolov's E-BERT strategy were also provided
- Randomizing external embeddings before fine-tuning was conducted to gauge the model's attention towards additional knowledge signals
- For BERT, KGE and definition embedding with BERTRAM concatenation led to the best accuracy across all configurations, while F1 was best for E-BERT and definition embedding + BERTRAM
- RoBERTa did not show any performance gains from the proposed method, but homogenized KGE with BERTRAM concatenation yielded the best accuracy (1.6% over regular fine-tuning)
- Compared to the best E-BERT baseline, BERT saw a 0.26% improvement in accuracy while F1 remained unchanged
- RoBERTa's best-performing model showed a 5.1% increase in accuracy over E-BERT, demonstrating the effectiveness of utilizing the entire model vocabulary

**Ablation Studies**:
- Replacing entity tokens with homogenized forms would alter sentence semantics, so only concatenation experiments were conducted
- For COVID-QA, KGE alone improved F1 and EM by 1.5% and 4.7%, respectively, while definition embeddings improved them by 1.7% and 6%, respectively
- For RoBERTa, KGE alone decreased F1 but increased EM, while definition embeddings decreased F1 but increased EM
- Conjectured that the benefit from definition embedding was due to its resemblance to transformer vectors, while the homogenized KGE had some benefit but added noise
- For PubMedQA, the best KGE model yielded a 5.1% and 6.7% increase in accuracy and F1, respectively, over regular fine-tuning
- With only definition embeddings, similar improvements were observed in F1 (11.1%) and accuracy (4.8%), while RoBERTa's best model obtained a 0.6% improvement in F1 and 11.1% in accuracy using the homogenized KGE

## V. DISCUSSION

**Study Findings**
* RoBERTa outperforms BERT-based models when integrating external knowledge through concatenation due to its tokenization scheme:
  * RoBERTa's vocabulary includes spaces, altering language decomposition and presentation to the model
  * Performance improvement for COVID-QA in terms of EM is more pronounced for domain-specific models because it helps pinpoint answers
* PubMedQA sees overall enhancements for both metrics (accuracy and F1) for BERT and RoBERTa based models upon fine-tuning
* Integrating external embeddings provides performance improvements for non-domain-specific models
* Adding random entity embeddings even improves performance over vanilla fine-tuning of domain-specific models, likely because they denote relevant terms without significant vocabulary overlap
* Homogenization method allows scaling well to domains with minimal vocabulary overlap

**Comparisons between Models**
* BERTBASE vs. RoBERTa:
  * RoBERTa's tokenization scheme includes spaces that alter language decomposition and presentation when external knowledge is included, leading to performance disparities
* Fine-tuning UMLS Embeds (KGE) and E-BERT:
  * Performance improvements for both models when integrating external embeddings
* BioBERT, SciBERT:
  * Significant improvement in accuracy/F1 over vanilla fine-tuning on PubMedQA dataset

**Complexity Analysis**
* Training an FFNN to homogenize embeddings requires learning additional parameters compared to the Mikolov baseline
* This computational overhead is mitigated by modern machines' capability to optimize simple networks without significant energy consumption
* Retraining the network for each model to be aligned could become a bottleneck if there are many entity embeddings to homogenize, but scalability remains an advantage.

## VI. CONCLUSIONS AND FUTURE WORK

**Conclusions and Future Work on External Knowledge Embedding Integration for MRC (Machine Reading Comprehension)**

**Findings:**
- Proposed approach shows potential for adding domain-specific information to input representation of a non-domain-specific model
- Demonstrates benefits without lengthy pre-training process

**Limitations and Future Research:**
1. **Alternative Strategies for Incorporating External Embeddings**:
   - More research needed into figuring out alternative strategies
   - Alternatives include training adapter layers, but present processing overhead and complexity
2. **Benefit of Pre-Training on Domain-Specific Corpora:**
   - Gap between performance on COVID-QA vs PubMedQA indicates need for further investigation
   - Higher scores on PubMedQA suggest benefits, but low scores on COVID-QA make definitive claim uncertain
3. **Further Investigation into Poor Performance of Models:**
   - Issues may lie in underlying architecture or semantic disconnect between medical and open-domain corpora
4. **Conclusion:**
   - More research needed to understand the effects of pre-training on dense, straightforward question-answer pairs for true domain generalization.

