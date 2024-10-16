# VERITAS-NLI: Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference

Arjun Shah, Hetansh Shah, Vedica Bafna, Charmi Khandor

https://github.com/Arjun-254/VERITAS-NLI 

https://arxiv.org/abs/2410.09455v1

## Contents
- [Abstract](#abstract)
- [I Introduction](#i-introduction)
- [II Literature Survey](#ii-literature-survey)
- [III Methodology](#iii-methodology)
  - [III-A Training Dataset](#iii-a-training-dataset)
  - [III-B Evaluation Dataset](#iii-b-evaluation-dataset)
  - [III-C Classical Model Performance Benchmarks](#iii-c-classical-model-performance-benchmarks)
- [IV Proposed Solution](#iv-proposed-solution)
  - [IV-A Web Scraping Techniques](#iv-a-web-scraping-techniques)
  - [IV-B Question-Answer Pipeline](#iv-b-question-answer-pipeline)
  - [IV-C Small Language Model Pipeline](#iv-c-small-language-model-pipeline)
  - [IV-D Article Pipeline](#iv-d-article-pipeline)
  - [IV-E NLI-based Models for Headline Inconsistency Detection](#iv-e-nli-based-models-for-headline-inconsistency-detection)
- [V Results](#v-results)
  - [V-A How effective is our Proposed Solution for the given task?](#v-a-how-effective-is-our-proposed-solution-for-the-given-task)
  - [V-B Which NLI-based Model performs best for Headline Inconsistency Detection?](#v-b-which-nli-based-model-performs-best-for-headline-inconsistency-detection)
  - [V-C Which Small Language Models (SLMs) perform best, and how effective are the generated questions?](#v-c-which-small-language-models-slms-perform-best-and-how-effective-are-the-generated-questions)
  - [V-D What are the performance characteristics and relationships between the Top-performing Classical models, BERT-based models and our Proposed Solution?](#v-d-what-are-the-performance-characteristics-and-relationships-between-the-top-performing-classical-models-bert-based-models-and-our-proposed-solution)
  - [V-E How efficient are our Proposed Pipelines?](#v-e-how-efficient-are-our-proposed-pipelines)
- [VI Conclusion](#vi-conclusion)

## Abstract

**Fake News Detection**
- **Challenges**: Rapid spread of information through online platforms, reliance on training data, inability to generalize on unseen headlines
- Proposed solution: Leveraging web-scraping techniques and Natural Language Inference (NLI) models for verification

**Web-Scraping**:
- Retrieves external knowledge necessary for verifying the accuracy of a headline

**Natural Language Inference (NLI) Models**:
- Used to find support for a claimed headline in the corresponding externally retrieved knowledge

**Evaluation**:
- Diverse self-curated dataset spanning multiple news channels and broad domains
- Best performing pipeline achieves an accuracy of 84.3%
- Surpasses best classical Machine Learning model by 33.3% and Bidirectional Encoder Representations from Transformers (BERT) by 31.0%

**Conclusion**:
- Combining dynamic web-scraping with Natural Language Inference is effective for fake news detection

## I Introduction

**Impact of Fake News**
- Exacerbation of digital platforms as primary sources of information
- Undermines trust in reliable sources
- Potential to skew public opinion [[1](https://arxiv.org/html/2410.09455v1#bib.bib1)]

**Spread and Erosion of Misinformation**
- False narratives spread easily across social media and online platforms
- Urgent need to address issue [[2](https://arxiv.org/html/2410.09455v1#bib.bib2)]
- Proliferation of fake news in 2016 US elections [[3](https://arxiv.org/html/2410.09455v1#bib.bib3)]
- Individuals overwhelmed by authenticity of information

**Click-Bait Articles and Misinformation**
- Google's Adsense as primary metric for news site success [[4](https://arxiv.org/html/2410.09455v1#bib.bib4)]
- Erosion of democracy, justice, and public trust [[5](https://arxiv.org/html/2410.09455v1#bib.bib5)]
- Intentionally created and disseminated as misinformation campaigns [[1](https://arxiv.org/html/2410.09455v1#bib.bib1)]
- Decline in user interactions with misinformation on Facebook, rise on Twitter [[6](https://arxiv.org/html/2410.09455v1#bib.bib6)]

**VERITAS-NLI Solution**
- Leverages state-of-the-art Natural Language Inference Models (NLI) for verifying claims
- Compares claims against externally retrieved information from reputable sources via web scraping techniques [[1](https://arxiv.org/html/2410.09455v1#bib.bib1)] [[2](https://arxiv.org/html/2410.09455v1#bib.bib2)]
- Employs small language models to generate questions based on headlines, enhancing verification process through a question-answering approach
- Consistency between scraped article and input headlines assessed using NLI models of varying granularity
- Evaluation dataset constructed by manually curating real headlines from reputable sources and synthetically generating fake headlines to simulate misinformation scenarios
- Enables model to adapt to dynamic content, as it does not rely on static, outdated data
- Continuously validates claims against real-time information to ensure model remains relevant and effective in rapidly changing news environments

**Identifying Indicators of Fake News**
- Analyzes language, sources, factual claims, pronoun-swapping, sentence-negation, named-entity preservation, and overall credibility of information presented
- Cross-references claims made in news content with latest reliable and authoritative sources.

## II Literature Survey

**Fact-Checking False Information**

**Initial Approaches:**
- Manual fact-checking by humans: site factchecker.in [[9](https://arxiv.org/html/2410.09455v1#bib.bib9)]
- Resource-intensive and limited scalability
- Potential bias and subjectivity

**Machine Learning Approaches:**

**Classical Machine Learning Algorithms:**
- Support Vector Machines (SVM) [[10](https://arxiv.org/html/2410.09455v1#bib.bib10), [11](https://arxiv.org/html/2410.09455v1#bib.bib11)]
  * Higher performance in detecting fake news
- Naïve Bayes [[12](https://arxiv.org/html/2410.09455v1#bib.bib12), [13](https://arxiv.org/html/2410.09455v1#bib.bib13)]
  * Effective for detecting fake news
- Random Forest Classifier [[14](https://arxiv.org/html/2410.09455v1#bib.bib14)]
  * Achieved an accuracy of 68% on PolitiFact and GossiCop datasets
- K-Nearest Neighbor (KNN) [[15](https://arxiv.org/html/2410.09455v1#bib.bib15)]
  * Obtained an accuracy of 79% from BuzzFeed News dataset

**Deep Learning Approaches:**
- Deep learning network [[20](https://arxiv.org/html/2410.09455v1#bib.bib20)]
  * Detects patterns in language of fake news
  * Trained on data up to 2016, suggested for use with automated fact-checkers
- Graph-based models [[21](https://arxiv.org/html/2410.09455v1#bib.bib21)]
  * Self-supervised contrastive learning and Bayesian graph Local extrema Convolution (BLC)
  * Aggregates node features while accounting for uncertainties in social media interactions
  * High accuracy on Twitter datasets but struggles with feature learning challenges in power-law distribution of node degrees
- Neural network utilizing graph attention networks, KGAT [[22](https://arxiv.org/html/2410.09455v1#bib.bib22)]
  * Performs fine-grained fact verification by analyzing relationships between claim and evidence presented as a graph

**Transformer-based models:**
- BERT [[23](https://arxiv.org/html/2410.09455v1#bib.bib23), [24](https://arxiv.org/html/2410.09455v1#bib.bib24)]
  * Outperforms other models using contextualized embeddings
  * Achieves high accuracy across multiple datasets
- DistilBERT and SHAP (Shapley Additive exPlanations) [[26](https://arxiv.org/html/2410.09455v1#bib.bib26)]
- FakeBERT [[28](https://arxiv.org/html/2410.09455v1#bib.bib28)]
  * Employs bidirectional learning to enhance contextual understanding of articles

**Hybrid Models:**
- Combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks [[30](https://arxiv.org/html/2410.09455v1#bib.bib30)]
  * Achieved an accuracy of 96% on a dataset sourced from Kaggle
- Combination of supervised and unsupervised learning paths with CNNs [[31](https://arxiv.org/html/2410.09455v1#bib.bib31)]
  * Performs well even with limited labeled data
- Self-serving mechanism to improve detection over traditional methods [[32](https://arxiv.org/html/2410.09455v1#bib.bib32)]

**Natural Language Processing (NLP) Techniques:**
- Difficulties in using current NLP techniques [[33](https://arxiv.org/html/2410.09455v1#bib.bib33)]
- Large language models like GPT-3 [[34](https://arxiv.org/html/2410.09455v1#bib.bib34)]
  * Used for generating questions, responses and verifying facts.

### "Advances and Challenges in Fake News Detection Research"

**Limitation of Early Fake News Detection Methods**
- Depends on initial set of questions generated by Language Models (LLMs)
- Operates on small-scale datasets, limiting scalability and generalizability
- Focuses mainly on specific claim styles or periods (e.g., coronavirus)

**Advancements in Fake News Detection Research**
- Utilizes Small Language Models (SLMs) for comparable results
- Emphasizes importance of social context for detection
- Incorporates user engagement metrics and hybrid approaches

**Datasets and Evaluation Metrics for Fake News Detection**
- FEVER dataset: fact extraction and verification benchmark [Thorne et al.]
- Data collection site with ground-truth labels [Hassan et al.]
- ClaimBuster model and Bart-large filtering system [Reddy et al.]

**Summary of Fake News Detection Research Progress**
- Significant progress in developing methods and models for Fake News Detection (FND)
- Several research gaps: lack of real-world implementation, scalability, robustness, ethical implications

**Future Directions**
- Address practical issues related to scalability, robustness, and ethical implications
- Explore novel Machine Learning algorithms specifically for FND that can keep up with current news in a dynamic environment.

## III Methodology

### III-A Training Dataset

**LIAR Dataset for Fake News Detection**

**Source**: Pulitfact.com (POLITIFACT), publicly available resource
- Compiled from PolitiFact, a Pulitzer Prize-winning website
- Multiple human evaluated statements
- Authentication of POLITIFACT editor

**Data**:
- News claims from various multimedia sources: news releases, TV/radio interviews, campaign speeches, advertisements, social media platforms (Twitter, Facebook)
- Societal and government related news topics: economy, taxes, campaign-biography, elections, education, jobs

**Characteristics**:
- Statements collected over a ten-year period
- Each statement accompanied by metadata: speaker, context, detailed judgment report

**Fields in the Dataset**:
1. **Label Description**: Truthfulness rating assigned to each statement
   - Multilabel system: "True", "Mostly True", "Half True", "Barely True", "False", "Pants on Fire"
2. **Label Assignment**: Rationale or explanation for the label assignment by fact-checkers
   - Evidence and reasoning based on research, reports, data analysis
3. **Label Distribution**: Figure 1 illustrates the label distribution in the LIAR dataset

**Preprocessing**:
- Converted 6 classes into two distinct categories (True/False)
- Mitigates subjective disagreements by reducing ambiguity
- Consistent labeling standards: Mapping = { “true”: True, "mostly-true": True, "half-true": True, "barely-true": False, "false": False, "pants-fire": False}

**Usage**:
- Training of classical machine learning models and transformer-based models for binary classification using the entire transformed LIAR dataset.

### III-B Evaluation Dataset

**Evaluating Model Performance**

**New Dataset for Headline Analysis:**
- Consists of latest headlines from various domains (science, sports, pop culture, politics, business)
- Collected from trusted sources (CNN, USNews, The Guardian, BBC, NYT, Reuters, Times of India, CNBC)

**Dataset Specifications:**
- Contains 346 credible headlines
- Generates corresponding unreliable headlines using Small Language Model (SLM), microsoft/Phi-3-mini-4k-instruct
- Total of 692 headlines in evaluation dataset

**Unreliable Headline Generation:**
- Prompted to modify reliable headlines through: sentence negation, number swaps, replacing named entities, noise injection
- Resulting in convincing fake news headlines for model evaluation

**Zero-shot Prompt Approach:**
- "Given the true headline, you have to make changes to it using a combination of Sentence negation, Number swaps, Replacing Named Entities and Noise Injection to generate a fake news headline. The output should be only the generated fake news headline."

**Example:**
- Figure 2 shows an example of an unreliable headline generated using Microsoft’s Phi-3 with our zero-shot prompting approach.

### III-C Classical Model Performance Benchmarks

**Classical Machine Learning Models for Fact-Checking News Articles**

**Preprocessing Steps**:
- **Stopword Removal**: Removing common words with little meaning
- **Token Filtering**: Excluding non-alphabetic tokens
- **Lemmatization**: Reducing words to root form using NLTK's WordNetLemmatizer
- **TF-IDF Vectorizer**: Representing word importance based on frequency in document and corpus

**Classical Machine Learning Models**:
- **Linear SVC**: Support Vector Classifier that finds a hyperplane for maximum distance between classified samples
- **Multinomial Naive-Bayes**: Probabilistic classifier based on Bayes' Theorem, suitable for document classification
- **Random Forest Classifier**: Ensemble model creating decision trees to improve predictive accuracy
- **Logistic Regression**: Model for modeling relationship between dependent and independent variables, predicting likelihood of group membership

**Transformer Models**:
- **BERT**: Pre-trained model based on Transformer architecture, capturing context of words through left and right contexts
- Fine-tuning BERT on LIAR dataset provides an advantage due to its deep understanding of language patterns and context

**Model Performance Metrics**:
- Accuracy, Precision, Recall, F1-Score for models trained and tested on LIAR and Evaluation datasets

**Importance of Transformers**:
- Effectively handle polysemous words with bidirectional context understanding
- Can be fine-tuned with a small number of task-specific examples to achieve high performance

#### BERT Fine-Tuning for Fake News Detection

**Text Normalization:**
- Convert all text to lowercase (for BERT-base-uncased)
- Perform unicode normalization

**Punctuation Splitting:**
- Split the text at punctuation, adding spaces around each punctuation mark

**WordPiece Tokenization:**
- Break words into subword units based on a trained vocabulary
- Unknown words are split into smaller subwords that exist in the vocabulary

**Adding Special Tokens:**
- Add special tokens such as [CLS], [SEP] for specific tasks (e.g., classification, question answering)

**Fine-Tuning:**
- **Initialization**: Fine-tuning begins with pre-trained models initially trained on a large corpus
- **Task-Specific Adjustments**: Tweak the last few layers of the model to align outputs with task requirements
- **Hyperparameter Optimization:**
  - Number of epochs: A single epoch is often sufficient, but using greater numbers can lead to overfitting
  - Batch sizes: Smaller batch sizes help maintain a balance between memory usage and effective learning
  - Warmup steps: Implement a warmup phase at the beginning of training where learning rates are gradually increased
  - Learning rate: The most suitable value for many models is 5e-3 (learing\_rate) and 0.001 (weight\_decay)
  - FP16: Enables faster training and reduced memory usage while maintaining precision

## IV Proposed Solution

**Proposed Solution: VERITAS-NLI**

* VERITAS-NLI utilizes web-scraping, Small Language Models (SLM), and Natural Language Inference (NLI) models to mimic human cognitive processes in evaluating claim reliability.
* Three pipelines - Question-Answer Pipeline, Article Pipeline, and Small Language Model Pipeline - showcase this intuition and demonstrate effectiveness.

### IV-A Web Scraping Techniques

**Web Scraping Techniques for Fact-Checking News Headlines:**
* Retrieve external knowledge using: web scraping, chromedriver, XML Tree traversal
* Approaches to scrape relevant information:**Fact-Based Headline Verification**: check Google Search 'Quick Answer'
  + **Top-K Articles Retrieval**: get titles and contents from top search results
  + **People Also Asked, **SLM Generated Questions**: generate questions using Small Language Models
* Sources used: trusted news outlets, reputable sources
* Scraping rules: follow 'robots.txt' directives to access content responsibly.

### IV-B Question-Answer Pipeline

**Google Search Methods: Amalgamation of Techniques**
- **Quick Answer**: Dependent on whether input headline is a hard fact or not
- Incorporated **"People Also Asked",   - Heavy user interaction dependence for uncommon occurrences
  - May not be represented in the PAA section
- **Terminating fallback**: Naive scraping of top-k articles if previous methods fail

**Scraping and Claim Verification Process (Pipeline #3)**
1. Web scraping
2. Passed scraped content to headline-verifying Natural Language Understanding (NLI) model
3. Input headline is claim (hypothesis), scraped content is premise
4. Model generates prediction and confidence score
5. Represents how well claims in headline are supported by external information.

### IV-C Small Language Model Pipeline

**Small Language Models for Question Generation**
- **Pipeline**: Small Language Models (SLM) used to generate questions for more relevant content scraping
- **Approach**: Identical to Question-Answer Pipeline with an added SLM question generation step
- **Transform**: User entered input headline into a context-aware question using Open Source SLMs: Mistral-7B-Instruct-v0.3 and Microsoft’s Phi-3-mini-4k-instruct
- **Benefits**: Retrieve more direct answers from the web for fact-checking claims against external sources
- **Question Generation**: Transforms input headline into a question using SLMs to increase probability of obtaining a direct response on the web

**Example Question Generation Prompt**:
- You’re tasked with fact-checking news headlines for authenticity. Given a headline, generate 1 question that needs to be true to verify its accuracy using a Google search and scrape the answer from the quick search box. Ask the crucial questions first.

**Pipeline 2**: Question Generation Process
- Input headline: "India wins 2023 ICC Men’s Cricket World Cup"
- Prompt: Generate 1 question that needs to be true to verify the authenticity of this headline using a Google search and scrape the answer from the quick search box.
- Question generated by Mistral-7B-Instruct-v0.3: "Who won the 2023 ICC Men’s Cricket World Cup?"
- This question simplifies the scraping workflow and helps retrieve a direct answer to verify the veracity of the claim.

### IV-D Article Pipeline

**Article Pipeline Methodology:**
- Gathers data from online sources systematically using web scraping [[43](https://arxiv.org/html/2410.09455v1#bib.bib43)]
- Extracts relevant content based on Top K links to articles
- Acquires headings and content from these links for verification process
- Structures pipeline efficiently and reliably
- Naive approach leads to more accurate outcomes

**Pipeline Components:**
1. Web scraper: acquires headings and content of articles related to a specific input headline
2. Model: verifies the claim based on retrieved information
3. Natural Language Inference (NLI) models: FactCC [[7](https://arxiv.org/html/2410.09455v1#bib.bib7)] and SummaC [[8](https://arxiv.org/html/2410.09455v1#bib.bib8)] for analyzing potential inconsistencies within claimed headlines

**Benefits:**
- Efficient and reliable process
- Rich contextual understanding of topic at hand from article content
- Improves model performance compared to incomplete or misguiding summaries like 'Quick Answer' and 'People Also Asked'

**Improved Results:**
- Scraping right amount of information (quantity to quality) leads to best output correlation.

### IV-E NLI-based Models for Headline Inconsistency Detection

**FactCC Approach for Ensuring Factually Consistent Summaries:**
- **VERITAS-NLI**: leverages FactCC [[7](https://arxiv.org/html/2410.09455v1#bib.bib7)], a model ensuring factual consistency in text summarization
- **Architecture of FactCC**: built upon BERT, operates under weakly-supervised learning framework, trained on synthetic data, and uses multi-task learning for Consistency Identification, Support Extraction, and Error Localization
- **Impact on Text Summarization**: ensures that generated summaries remain factually consistent with their source documents

**Comparison of Headline and Source Text:**
- Inconsistencies detected using **Natural Language Inference (NLI)** models: Entailment, Contradiction, or Neutral
- NLI models perform well thanks to large training datasets like MNLI [[52](https://arxiv.org/html/2410.09455v1#bib.bib52)], SNLI [[53](https://arxiv.org/html/2410.09455v1#bib.bib53)], and VitaminC [[54](https://arxiv.org/html/2410.09455v1#bib.bib54)]
- Modern attention-based architectures exhibit human performance for this task [[8](https://arxiv.org/html/2410.09455v1#bib.bib8)]

**FactCC Output:**
- Input Headline, generated question(s), retrieved article or direct answer, and NLI-Model Output Score are presented for comparison
- Example of correct outputs: Max Verstappen wins 2023 F1 world title -> Entailment (or high NLI score)

**SummaC Approach:**
- Obtains state-of-the-art results for inconsistency detection in summarization of documents
- Segments longer documents into sentence units, changing the granularity at which NLI models are applied
- Calculates likelihood of relationships between each summary and document-sentence pair using an NLI model
- Aggregates these scores to produce final output: SummaC-ZS (maximum value per column) or SummaC-Conv (1D-convolution on histograms)
- Determines optimal decision threshold for binary predictions through calibration process.

## V Results

**Study on Fake News Detection:**
- **Finds significant performance differences** between Machine Learning (ML) models, BERT models, and proposed Transformer-based pipelines for classifying news headlines
- **Classical ML models**: rely heavily on static training data, severely limiting their ability to adapt in dynamic environments
  - Average F1-score: 0.675, accuracy: 60.04% (LIAR dataset)
  - Decline in performance when tested on larger corpus of news headlines: average F1-score: 0.537, accuracy: 49.86%
- **BERT models**: initially outperform classical ML models by ∼ 3.5 percent but also suffer from reliance on static training data

**Performance Metrics:**
- Table III shows results for pipelines using various NLI (Natural Language Inference) models: FactCC, SummaC, Mistral-7B, Phi-3, and Article Pipeline
  - Each pipeline is evaluated based on precision, recall, F1 score, and accuracy
- SLM Pipelines perform well but still rely solely on static training data, which limits their adaptability in dynamic environments.

### V-A How effective is our Proposed Solution for the given task?

**Performance of Proposed Solutions (VERITAS-NLI)**
* Significantly outperforms prior approaches: highest accuracy = 84.3% (Article-Pipeline with SummaC-ZS)
* Substantial gains over best classical machine learning model (MultinomialNB): 33.3%
* Transformer-based BERT model: 31.0%
* Ten out of twelve pipeline configurations demonstrated superior accuracy compared to all baseline models
* Novel web-scraping techniques for dynamic information retrieval
* Robust solution for verification of claims without having to retrain the model on newer data
* Explainability by providing externally retrieved knowledge and Top-K articles (Figure 5)
* Detailed performance view in Table II, Figure 6: Common and Unique Correct-Incorrect Decisions Between Pipelines.

### V-B Which NLI-based Model performs best for Headline Inconsistency Detection?

**SummaC vs FactCC for Headline Inconsistency Detection**

**Performance:**
- SummaC consistently outperforms FactCC due to:
  - Application of NLI models on sentence-level pairs
  - Enhanced granularity for detecting subtle inconsistencies

**Approaches:**
- SummaC designed for fine-grained inconsistency analysis (sentence level)
- FactCC focuses on document-level fact checking

**Variants and Performance:**
- SummaC has two variants: SummaC-ZS (zero-shot), SummaC-Conv (Convolution)
- Both outperform FactCC across all evaluation metrics
- Superior performance due to unique approach of leveraging NLI pair-matrix

**Results:**
- Average accuracy for pipelines: FactCC = 59.83%, SummaC-ZS = 77.15%, SummaC-Conv = 69.33%
- SummaC pipelines outperform FactCC by 17.32% and 9.50% each

**Drawbacks:**
- SummaC-ZS: sensitivity to extreme values due to mean calculation
  - No longer a hindrance for news headlines (predominantly one sentence)

### V-C Which Small Language Models (SLMs) perform best, and how effective are the generated questions?

**Study Overview:**
- Two small language models used for question generation to scrape specific information and verify claims
- Mistral-7b pipelines have an average accuracy of 64.93%, Phi-3 pipelines have 63.37%
- Both models exhibit similar proficiency, despite Phi-3 having fewer parameters
- Small language model pipelines reach a maximum accuracy of 75.80%, while Article pipeline achieves 84.3%
- Quality of evidence retrieval is dependent on the initial questions generated by the language models (FLEEK [34])
- This could be the reason for lower performance compared to the naive article approach

### V-D What are the performance characteristics and relationships between the Top-performing Classical models, BERT-based models and our Proposed Solution?

**Research Question Insights:**
- Assessing performance characteristics of legacy models and proposed solution through Venn diagrams
- Comparison of distinct correct and incorrect decisions:
  - Legacy Models vs. Best-Performing Pipeline (Multinomial Naive Bayes, Logistic Regression, BERT-base-uncased, Article Pipeline with SummaC Zero-Shot)
    * Figure 6A and 6B show that our solution outperforms legacy models in both correct decisions and lowest incorrect decisions
    * Significant overlap in incorrect decisions among legacy models indicates lack of generalization for latest news headlines
  - Comparison between Best Performers (Article Pipeline with SummaC Zero-Shot, QNA_SummaC_ZS, Mistral_SummaC_ZS, Article_SummaC_ZS)
    * Figure 6C and 6D: Question-based pipelines uniquely classify more distinct correct decisions than article-based pipelines
    * Both types of pipelines have some unique incorrect decisions, with question-based pipelines exhibiting a higher number due to their focus on nuanced questions
    * Article_SummaC_ZS has significantly fewer distinct incorrect decisions compared to its counterpart, Article_FactCC.

**Additional Insights:**
- Execution times for various steps of the proposed pipeline are illustrated in Figure 7 using boxplots. (Note: This information is not covered in the initial text.)

### V-E How efficient are our Proposed Pipelines?

**Performance Assessment**
- Calculated average processing time through **stratified sampling** to ensure balanced subset of true and false headlines
- All experiments conducted on Google Colab:
  - NVIDIA Tesla T4 GPU (15.4GB GDDR6 memory)
  - Intel(R) Xeon(R) CPU @ 2.00GHz with 12.7GB RAM
- Divided pipeline into two stages: **web-scraping stage** and **NLI for inconsistency detection stage**
- Time taken for each sub-task visualized as box-plot in Figure 7
- Detailed information on time metrics for each pipeline in Table IV

**Web Scraping Stage**
- Most time-consuming task:
  - SLM-Phi3: 2.9 seconds
  - SLM-Mistral: 4.1 seconds
  - Question-Answer: 5.7 seconds
  - Article: 6.7 seconds

**NLI Stage (Inconsistency Detection)**
- Average inference time across pipelines: 0.047s (FactCC), 0.188s (SummaC-ZS), 0.101s (SummaC-Conv)
  * Increased time due to increased granularity and accuracy improvement
- Pipeline-specific average inference times:Article: 0.192s
  + SLM Mistral-7B: 0.0486s
  + SLM Phi-3: 0.0502s
  + Question-Answer: 0.040s
- Highest average times for pipelines with larger text volumes, 
## VI Conclusion

**Study on Fake News Detection**

- Investigates Machine Learning and Transformer Model efficacy for fake news detection
- Classical models' inadequate performance due to reliance on static training data
- Proposes VERITAS-NLI, a solution that combines advanced web-scraping techniques and Natural Language Inference Models
- Finds significant improvement in accuracy (over 30%) compared to benchmark models
- Enhancements essential for advancing the field and maintaining automated systems' pace with evolving fake news strategies
- Aims to safeguard public discourse integrity.
