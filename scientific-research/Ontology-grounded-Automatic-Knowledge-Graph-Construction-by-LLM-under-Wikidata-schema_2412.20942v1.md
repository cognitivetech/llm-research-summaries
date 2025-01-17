# Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema

source: https://arxiv.org/html/2412.20942v1
Xiaohan Feng, Xixin Wu, Helen Meng 

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Literature Review](#2-literature-review)
- [3 Method: Ontology-grounded KG Construction](#3-method-ontology-grounded-kg-construction)
  - [3.1 Competency Question (CQ)-Answer Generation](#31-competency-question-cq-answer-generation)
  - [3.2 Relation Extraction and Ontology Matching](#32-relation-extraction-and-ontology-matching)
  - [3.3 Ontology Formatting](#33-ontology-formatting)
  - [3.4 KG Construction](#34-kg-construction)
- [4 Experiments and Discussion](#4-experiments-and-discussion)
  - [4.1 Experiment settings](#41-experiment-settings)
  - [4.2 Result](#42-result)
  - [4.3 Discussion](#43-discussion)
- [5 Conclusion](#5-conclusion)

## Abstract

**Ontology-Grounded Knowledge Graph Construction**

We use Large Language Models to construct Knowledge Graphs from a knowledge base, grounded in an authored ontology. CQs are generated to discover the knowledge scope, and equivalent relations are replaced with Wikidata counterparts. The resulting KG is consistent and interpretable, demonstrating competitive performance on benchmark datasets. This scalable pipeline requires minimal human intervention and yields high-quality, interoperable KGs expandable via Wikidata semantics.

## 1 Introduction

**Knowledge Graphs (KGs)**
- **Structured representations of information**: capture entities and their relationships in a graph format
- Enable intelligent applications: semantic search, question answering, recommendation systems, decision support
- Critical to harness the power of these technologies across various domains

**Challenges with Manual Curation**:
- Time-consuming, expensive
- Difficult to scale to large, evolving domains

**Automatic Methods for KG Construction**:
- **Large Language Models (LLMs)**: generate fluent natural language, memorize and recall factual knowledge

**Challenges with LLM-Generated KGs**:
- May generate inconsistent or redundant facts
- Limited coverage of the target domain due to reliance on pre-training data
- Difficulty integrating LLM-generated KGs with existing knowledge bases

**Proposed Approach**:
- Harnesses reasoning power of LLMs and structured schema of Wikidata
- Discovering scope of knowledge: generate **Competency Questions (CQ)** and answers from unstructured documents
- Summarize relations and properties into an ontology, matching against Wikidata schema
- Ground transformation of CQ-answer pairs into a structured KG on the same ontology
- Reduces redundancy, leverages implicit knowledge, ensures interoperability with public knowledge bases

**Main Contributions**:
1. Proposed novel approach: LLM-based KG construction using ontology based on Wikidata schema
2. Pipeline combining CQ generation, ontology alignment, and KG grounding to construct high-quality KGs
3. Demonstration of effectiveness through experiments, showing improvements in KG quality and interpretability/utility

## 2 Literature Review

**Knowledge Graph Construction: Approaches and Challenges**

**Background:**
- Active research area in recent years
- Variety of approaches for extracting structured knowledge from unstructured data sources

**Early Methods:**
- Rule-based systems and handcrafted features for entity identification and relation extraction
- Heavily relied on manual creation of training data

**Deep Learning Approaches:**
- Neural network-based approaches enable more flexible and scalable KG construction
- Prominent line of work: distant supervision for automatic generation of training data
  * Assumes entities mentioned together in a sentence are likely related
  * Suffers from noise and incomplete coverage

**Unsupervised and Semi-supervised Methods:**
- Reduce reliance on labeled data through techniques like bootstrapping, graph-based inference, and representation learning
- Struggle with consistency and quality control issues

**Large Language Models (LM):**
- Take advantage of knowledge captured in pretrained LMs for KG generation
- Promising but only produces triplets without canonicalization
- Some methods rely on vector-based similarity measures for relationship deduction
  * Yields good performance but falls short on interpretability

**Current State:**
- Significant progress in KG construction and LLM applications
- Performance, interpretability, coverage of proprietary documents, and interaction with other knowledge bases remain issues

**Our Pipeline:**
- Grounded in ontology based on Wikidata schema
- Ensures human-readable output KG for easier integration with Wikidata or other KGs.

## 3 Method: Ontology-grounded KG Construction

Our proposed approach for building knowledge graphs using LLMs consists of four stages:

1. Competency Question Generation
2. Relation Extraction and Ontology Matching
3. Ontology Formatting
4. KG Construction

Figure 1 illustrates this pipeline.

### 3.1 Competency Question (CQ)-Answer Generation





"We generate competency questions (CQs) using an LLM, guided by instructions and examples. This step scopes the KG construction task within the domain, ensuring the resulting KG aligns with intended use cases. It also enables ontology expansion through user-submitted questions, which can be refined through our proposed pipeline."

### 3.2 Relation Extraction and Ontology Matching

**Preliminary Experiments with LLMs:**
- Spontaneous recall of Wikidata knowledge observed in LLMs during experiments
- Behavior also observed in small 7B/14B models

**Second Step: Extracting Relations from CQs:**
- Prompt LLMs to extract properties from CQ and write brief descriptions on usage
- Match extracted properties against Wikidata properties for elicitation of model memories

**Property Description and Matching:**
- Pre-populate candidate property list with all Wikidata properties, filtering out external database/knowledge base IDs
- Construct sentence embedding representation for each property description
- Perform vector similarity search between descriptions to find top 1 closest candidate property match
- Validate matches by LLM for deduplication and semantic similarity check

**Final Property List:**
- Add validated matched properties to final property list
- Keep newly minted properties if expanding from Wikidata's candidate property list
- Discard new properties when requiring a subset of the candidate property list

**Scenarios:**
1. No prior schema known for the domain with some new properties outside common ontology expected
2. Known target list of possible properties.

### 3.3 Ontology Formatting

We generate an OWL ontology using LLM based on matched and new properties. We copy description, domain, and range fields from Wikidata semantics properties. For new properties, LLM infers and summarizes classes for domain and range to output a complete OWL ontology, following the format of copied Wikidata properties.

### 3.4 KG Construction

In the final stage, we use an LLM to construct a knowledge graph (KG) from given CQs and their related answers, grounded by the previous stage's ontology. The LLM extracts entities and maps them to the ontology using defined properties, outputting RDF triples that form the KG.

## 4 Experiments and Discussion

### 4.1 Experiment settings

**Evaluation of Ontology-Grounded Approach to Knowledge Graph Construction (KGC)**
* Three datasets used for evaluation: Wiki-NRE, SciERC, WebNLG
* Included SciERC for more robust assessment as it contains relation types not equivalent to properties in Wikidata
* Used a subset of Wiki-NRE's test dataset (1,000 samples, 45 relation types) due to cost constraints
* SciERC test set: 974 samples, 7 relation types
* WebNLG test set: 1,165 samples, 159 relation types
* Evaluation metrics: partial F1 on KG triplets (based on standards in [18])
* All experiments conducted with one-pass processing

**Limitations of Previous Reports**:
* Incomplete annotation in KGC reports regarding possible relation types and KG triplets [19, 20]

**Evaluation Settings**:
* **Target schema constrained**: Match all relation types to their closest equivalents in Wikidata, constrict ontology to relation universe in test set
* **No schema constraint**: Do not filter matched ontology if not in schema of test dataset

**Property Conjunction Evaluation**:
* Select the closest properties proposed by LLM based on subjective opinion for SciERC's "for, compare, feature of"

**Knowledge Graph Parsing and Extraction**:
* Parse output KG with RDF parser to extract valid triples for each document in test set
* Present triplets to evaluation script for assessment

**Baseline Systems**:
* Non-LLM Baseline: GenIE, PL-Marker, ReGen (for Wiki-NRE, SciERC, WebNLG respectively)
* LLM-based systems: Results reported in [19] for Wiki-NRE and WebNLG using Mistral model, GPT-4 results in [3] for SciERC.

**Note**: It is unlikely that Mistral-7B poses an advantage over an earlier version of GPT-4 when interpreting the result of SciERC.

### 4.2 Result

**Method Comparison: F1 Scores on Test Datasets**

**Baselines**:
- Non-LLM Baseline: 0.484 (Wiki-NRE), 0.532 (SciERC), 0.767 (WebNLG)
- LLM Baseline: 0.647 (Wiki-NRE), 0.07 (SciERC), 0.728 (WebNLG)

**Proposed Methods**:
- Mistral: 0.66/0.60 (Wiki-NRE/SciERC), 0.74/0.68 (WebNLG)
- GPT-4o: 0.71/N/A (Wiki-NRE/SciERC), 0.76/N/A (WebNLG)

**Table 1: Partial F1 Scores on Test Datasets**

| **Method** | **Wiki-NRE** | **SciERC** | **WebNLG** |
|---|---|---|---|
| Non-LLM Baseline | 0.484 | 0.532 | 0.767 |
| LLM Baseline | 0.647 | 0.07 | 0.728 |
| Proposed (Mistral) | **0.66**/0.60* | **0.73**/0.58* | **0.74**/0.68* |
| Proposed (GPT-4o) | 0.71/N/A | N/A/0.77** | 0.76/N/A |

* Target schema constrained setting
* SciERC and WebNLG: performance improvement without schema constraint, but regression compared to baselines
* Wiki-NRE: exceeds all baselines under target schema constrained setting

**Results**:
- Proposed approaches outperform baselines on Wiki-NRE and SciERC with target schema constraint
- Maintain competitiveness against fine-tuned SOTA on WebNLG when constrained to target schema
- Performance improvement when using GPT-4o.

### 4.3 Discussion

**Performance Discrepancy on Different Grounding Ontology:**
- Lower performance on no schema constraint setting across all datasets due to discovery of richer ontology
- Expanded schema may capture additional relevant information but hinder extraction performance against limited target schema
- Trade-off between schema completeness and strict adherence to predefined ontology
- Performance deficit in absence of schema constraints cannot be evaluated against dataset directly, as ontology is not entirely covered by test set annotations
- Pipeline demonstrates ability to provide coverage of properties in test set when also capturing ontology outside of schema, which may be more useful for novel document sets with no expert knowledge
- Manual evaluation on full captured ontology a future work
- Marginal performance deficit leaves room for improvement

**Utility of Generated Knowledge Graph:**
- Correctness of extracted triples evaluated, but KG can do more than that
- Explicitly generating KG provides path to audit LLM knowledge and reduce hallucination
- Pipeline may serve as foundation for interpretable QA system where LLM autonomously extracts ontology and deduces retrieval query based on ontology
- Interpretability arises from the fact that KG and query could be understood and verified by users
- Usage of Wikidata schema offers potential interoperability with whole Wikidata knowledge base, expanding knowledge scope of QA system
- Research continuation on significant direction

**Computational Resources:**
- Growing concern of sustainability in LLM applications due to intensive requirement on computational resources
- Pipeline consumes three separate LLM calls per document, plus one call per extracted relation
- Comparison with Non-LLM baselines not straightforward due to model fine-tuning requirements and larger size of Mistral-7B model
- Advantage in resource cost compared to Non-LLM baselines for small number of documents with no training requirement
- Performance burden, proposing techniques for fine-tuning and guided decoding to achieve better performance with smaller models.

## 5 Conclusion

We've shown that our ontology-grounded approach to knowledge graph construction with Large Language Models is effective. By combining Wikidata's structured knowledge with LLMs and generated ontologies, we can build high-quality KGs across various domains. This allows for the creation of interpretable QA systems that access both common and proprietary knowledge bases.

Acknowledgement: This work was supported by Centre for Perceptual and Interactive Intelligence (CPII) Ltd under the InnoHK scheme.
