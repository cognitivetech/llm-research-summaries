# Harnessing Large Language Models: Fine-tuned BERT for Detecting Charismatic Leadership Tactics in Natural Language

by Yasser Saeid, Felix Neubürger, Stefanie Krügl, Helena Hüster, Thomas Kopinski, Ralf Lanwehr
https://arxiv.org/abs/2409.18984

## Contents
- [Abstract](#abstract)
- [I Introduction](#i-introduction)
- [II Related Works](#ii-related-works)
- [III Dataset description](#iii-dataset-description)
- [IV Methods and Experimental Setup](#iv-methods-and-experimental-setup)
- [V Experimental Results and Discussion](#v-experimental-results-and-discussion)
- [VI Future Research](#vi-future-research)

## Abstract
**Background:**
- Researchers investigated identifying Charismatic Leadership Tactics (CLTs) in natural language
- Used a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for detection
- Based on an extensive corpus of CLTs generated and curated for this task

**Methodology:**
- Trained machine learning model to accurately identify presence of these tactics in natural language
- Performance evaluation conducted to assess effectiveness of the model in detecting CLTs

**Findings:**
- Total accuracy over detection of all CLTs: 98.96%
- Significant implications for research in psychology and management
  - Potential methods to simplify assessment of charisma in texts

**Keywords:** Generative AI, Charismatic Leadership, Political Speeches, Natural Language Processing, Computational Linguistics, Large Language Models.

## I Introduction

**Charismatic Leadership**

**Introduction**:
- Charismatic leadership is a significant force influencing human behavior
- Defined as "values-based, symbolic, and emotion-laden leader signaling" [2]
- Signals include things one does to communicate [3]
- Essence: building strong connections with followers
- Utilize charisma signals to convey leadership abilities [4], express values [5, 6], demonstrate understanding of right/wrong [2]
- Leaders amplify charisma following crises [5, 7]
- Positive outcomes: enhanced follower productivity [8], increased social media influence [9], improved adherence to COVID-19 orders [10]

**Charismatic Leadership Tactics (CLTs)**:
- Play a crucial role in understanding charismatic leadership
- Previous studies labor-intensive, as analysis required manual coding [11]
- Recent advancements in NLP techniques enable automated analysis using transformer models like BERT [12]

**Objective of the Study**:
- Identify linguistic features underlying CLTs
- Improve understanding of charisma's linguistic foundations
- Develop analytical tools for studying charismatic leadership

**Related Concepts**:
- Ethical leader signals (ELSs) and emotional intelligence (EI) [13]
- Communicate ethical values, intentions, emotional competence among leaders
- Effect on followers' perceptions and organizational outcomes

**Methodology**:
- Fine-tune a BERT model on a corpus of CLTs
- Apply it to a new corpus for identifying CLTs within sentences

**Contribution of the Study**:
- Provide a foundation for future research on linguistic means of charismatic leadership
- Develop tools for analysis and understanding charismatic leadership.

## II Related Works

**NLP Techniques for Analyzing Charismatic Leadership**
* Recent years: NLP techniques used to analyze charisma in speeches
	+ Identify patterns determining level of charisma
	+ Distinguish between charismatic leaders and non-charismatic ones
* Linguistic features prevalent in charismatic speeches
	+ Enthusiasm, optimism, rhetorical devices
* Machine learning algorithms used for classification: political speeches based on charisma content
	+ Certain linguistic features, such as enthusiasm and positivity, more frequent in charismatic speeches
* Deep learning techniques analyze emotional tone of political speeches
	+ Charismatic leaders use more positive emotions, fewer negative emotions
* Advancement of transformer-based models like BERT: captures subtle nuances in language usage
	+ Researchers fine-tune pre-trained language models to classify texts based on sentiment, emotion, persuasive appeal

**Developing a BERT-Based Model for Charismatic Leadership**
* Goal: Develop a model that can accurately detect CLTs in sentences modeled after gubernatorial speeches
* Training data: Large corpus of AI generated sentences based on gubernatorial speeches
	+ Insights into how charismatic leadership manifests in spoken language across different contexts and leader types

**Ethical Leadership Research: Recent Literature**
* Previous study: Critically evaluated current landscape of ethical leadership
	+ Identified conceptual conflation, lack of knowledge regarding causes and consequences
	+ Proposed refined conceptualization based on signaling theory
		- Ethical leadership behavior defined as leader's signaling towards stakeholders
			* Enactment of prosocial values
			* Expression of moral emotions
* Banks et al. used multifaceted approach to delve into ethical leadership through signaling theory
	+ Constant comparative analysis, preregistered experiments, and data science techniques
	+ Identified verbal ELSs associated with emotions such as righteous anger and pride
	+ Demonstrated impact on evaluations of ethical leadership
		- Countering counterproductive behavior
		- Enhancing task performance
		- Encouraging extra role behavior
	+ Introduced DeepEthics, a machine learning algorithm for scoring texts for ELSs.

**Importance of Ethical Leadership Research**
* Nuanced distinction between leader behaviors and subjective follower evaluations
* Clearer conceptualizations and innovative methodologies needed in ongoing exploration of ethical leadership.

## III Dataset description

**Dataset Description III-A Content:**
- Charismatic leaders have better outcomes due to charisma [11]
- Operationalized through twelve CLTs: frame, delivery, substance [11]
	+ **Frame**: metaphors, stories, anecdotes, etc. [11]
	+ **Delivery**: body gestures, facial expressions, tone [11]
	+ **Substance**: moral convictions, setting ambitious goals [11]
- CLTs enable leaders to signal ability and willingness to lead [11]
- Signaling charisma can evoke positive experiences with charismatic leaders

**III-B Dataset Generation using Large Language Models:**
- Aiming for comprehensive dataset of 10,000 samples per tactic [11]
- Using ChatGPT API to generate diverse and well-balanced data [19]
	+ Example prompt: generate sentences expressing ambitious goal-setting [11]
- Manually evaluating generated output for desired diversity and quality [2]
- Iteratively generating text samples using prompts [6 months]
	+ Segmented data generation into smaller batches for organization
- Rigorous curation of text samples to ensure adherence to charismatic tactics, grammatical accuracy, and overall consistency.

**Data Collection and Curation:**
1. Prompt Development: carefully designed prompts for diverse range of results [2]
	+ Understanding linguistic nuances and pragmatic elements fundamental to charismatic communication
2. Output Evaluation: manual evaluation until desired output achieved [1]
3. Data Generation: methodically produced text samples using ChatGPT API [6 months]
4. Data Curation: rigorous checks for adherence to charismatic tactics, grammatical accuracy, and overall consistency with prompt instructions
5. LLM Evaluation and Comparison: comparative analysis of various LLMs in generating charismatic text (out of scope for this paper)
6. Data Organization and Analysis: batching data generation requests, streamlined data management and analysis [2]
7. Quality Control: exhaustive manual examination to ensure overall quality and accuracy
8. Use of VU Amsterdam Metaphor Corpus to extend corpus with established linguistic resources.

## IV Methods and Experimental Setup

**BERT Algorithm for Multilabel Sentence Classification**

**Background:**
- Demonstrated efficacy in multilabel sentence classification for natural language processing (NLP)
- Transformer-based model that redefined NLP tasks, including sentence classification
- Advanced capabilities: bidirectional context understanding, capturing intricate dependencies within text data

**Foundation of BERT:**
1. Pre-training phase: exposed to vast amounts of unlabeled text data
2. Unsupervised learning: predict masked words within sentences
3. Develop deep understanding of language nuances and semantic relationships
4. Subsequent tasks: sentence classification, multilabel sentence classification for CLTs

**Implementation with BERT:**
1. Tokenize sentences using bert-base-uncased tokenizer from huggingface library
2. Create contextualized embeddings for entire sentence
3. Combine vectors to encapsulate nuanced meaning and context
4. Predict multiple labels or categories for given sentence (multilabel nature of the task)
5. Split data into training and test sets, with internal validation strategy
6. Finetune pretrained BERT model for sequence classification using AdamW optimizer from pytorch library
7. Learning rate: lr=5⋅10−5
8. Training for 100 epochs

**Resources:**
- Code and further materials available on GitHub 1

## V Experimental Results and Discussion

**Experimental Results and Discussion**
- **Multilabel classification task**: evaluated using standard machine learning metrics: accuracy, precision, f1-score
- Metrics were weighted with classweights and averaged due to unbalanced dataset
- Classification evaluation shown in Table I, averaged scores displayed in Table II
  - Total accuracy: 98.96%
- Examination of confusion matrix (Figure 1) provides nuanced understanding of model's performance:
  - Each row corresponds to actual class; each column represents predicted class
  - Reveals frequency of correct vs. misclassified tactics
  - Identifies patterns of confusion, potential challenges for model in distinguishing similar CLTs
- Performance aligns with human interpretation complexities within field

**Quantitative Metrics (Table I)**
- **Classification metrics for different charismatic leadership tactics**: precision, recall, f1-score, support for each tactic

**Averaged Classification Metrics (Table II)**
- Macro average: unweighted mean of precision, recall, and f1-score, treating each class equally
- Weighted average: considers class imbalance by computing mean with each class's contribution weighted by its prevalence in the dataset

**Conclusion**: comprehensive evaluation of multilabel classification model shows robust performance despite linguistic nuances of CLTs. Potential applicability in analyzing and understanding charismatic leadership communication.

## VI Future Research

**Future Research Directions in Charismatic Leadership and NLP**

**Application to Gubernatorial Speeches:**
- Apply multilabel sentence classification techniques to (US)-gubernatorial speeches
- Insights into dynamics of leadership communication
- Valuable findings through advanced NLP models
- Potential extension of current framework to political contexts

**Using the Generated Corpus for Other NLP Research:**
- Contribute to broader research community by making corpus accessible
- Detailed documentation and insights into construction process
- Ethical and legal considerations outlined in separate publication
- Fostering collaboration among researchers from diverse backgrounds

**Methods for Generation of NLP Corpora (Using Other LLMs):**
- Explore alternative methods for generating NLP corpora
- Leverage other language models for corpus generation
- Impact on quality and diversity of corpora investigated
- Insights into optimizing creation processes and enhancing robustness of datasets

**Usage and Comparison of LLMs for Sequence Classification:**
- Continuously evolving field of NLP with emerging new language models
- Assess performance of various LLMs in capturing contextual nuances
- Subtle variations in leadership communication considered
- Refinement of NLP applications through model comparisons.

