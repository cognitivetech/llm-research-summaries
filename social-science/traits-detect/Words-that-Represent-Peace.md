# Words that Represent Peace

https://arxiv.org/abs/2410.03764v1
by Tushar Prasad, Larry S. Liebovitch, Melissa Wild, Harry West, Peter T. Coleman

## Contents
- [Abstract](#abstract)
- [I Introduction](#i-introduction)
- [II Related Works](#ii-related-works)
- [III-A Data](#iii-a-data)
- [III-B Preprocessing](#iii-b-preprocessing)
- [III-C Modeling](#iii-c-modeling)
- [III-D Qualitative Inference](#iii-d-qualitative-inference)
- [IV-A Data and Preprocessing](#iv-a-data-and-preprocessing)
- [IV-B Modeling Experiments](#iv-b-modeling-experiments)
- [IV-C Qualitative Results](#iv-c-qualitative-results)

## Abstract

**Peace Measurement using News Media: Methodology**

**Data Source:**
- LexisNexis for news media data

**Identifying Higher vs Lower Peace Countries:**
1. Analyze news themes to distinguish between higher and lower peace countries
2. Characteristics of Higher Peace News: finance, daily activities, health
3. Characteristics of Lower Peace News: politics, government, legal issues
4. Provides a starting point for measuring levels of peace
5. Identifying underlying social processes

**Data Analysis:**
- Use Natural Language Processing (NLP), Machine Learning algorithms like Random Forests and Support Vector Machines for analysis.

## I Introduction

**The Role of Language in Peace and Conflict**

**Communication and Reality**:
- Communication through language shapes our perception of reality
- Plays a pivotal role in human interactions, including conflicts

**Destructive Power of "Hate Speech"**:
- Incites violence and deepens societal divisions
- Researchers are leveraging data science and NLP to monitor and predict conflicts

**Understanding Conditions for Sustained Peace**:
- Highly peaceful societies exhibit specific norms, values, and stability
- Increasing focus on "positive peace" - the active social forces that contribute to harmonious coexistence

**Peace Speech**:
- Linguistic structures that promote and sustain peaceful interactions
- Empirical studies on specific features and effects are scarce

**Research Approach**:
- Machine learning techniques to identify key linguistic features of peace speech
- Analyzing most frequently used words in countries with varying levels of peace

**Journalists' Role**:
- Propagating paranoia or fostering peace through narrative and themes
- System to inform journalists about "peacefulness" of society and suggest adjustments to promote peace

## II Related Works

**Peace Research Findings:**
* Fry et al. [[4](https://arxiv.org/html/2410.03764v1#bib.bib4)] and Coleman et al. [[5](https://arxiv.org/html/2410.03764v1#bib.bib5)]:
  * Societal peacefulness reveals many groups choose peace over war, an untapped resource for understanding pathways to sustainable peace.
  * Peace understood through ratio of positive to negative intergroup reciprocity, stable over time.
* Liebovitch et al. [[6](https://arxiv.org/html/2410.03764v1#bib.bib6)]:
  * Peace extends beyond absence of conflict, defining it as the systems that foster and sustain harmonious societies (positive peace).
  * Complex systems analysis using physics-inspired techniques to explore factors contributing to peace.
* Fry et al. [[4](https://arxiv.org/html/2410.03764v1#bib.bib4)]:
  * Some human societies not only avoid war but also actively foster positive intergroup relationships (peace systems).
  * Peace systems exhibit higher levels of common identity, interconnectedness, interdependence, and peace leadership compared to other social units.
* Liebovitch et al. [[7](https://arxiv.org/html/2410.03764v1#bib.bib7)]:
  * Language plays a crucial role in both causing and consequence of peace or conflict.
  * Machine learning models trained on word frequencies from countries at the extremes of peace can be used to compute a quantitative peace index for intermediate-peace countries.

## III-A Data

**Peace Speech vs Hate Speech Analysis**

**Dataset**:
- Provided by LexisNexis through a partnership with Elsevier and Columbia University
- Approximately 2,000,000 media articles from 2010 to 2020
- Written in English, from 20 different countries
- Variety of sources: "24 Hours Toronto", "BBC Monitoring: International Reports", "Marie Claire" etc.

**Measuring Peace**:
- Utilized peace studies listed below to train the model
- Averaged values from these indices to determine peaceful vs non-peaceful countries
    - **Global Peace Index** [[8](https://arxiv.org/html/2410.03764v1#bib.bib8)]
    - **Positive Peace Index** [[9](https://arxiv.org/html/2410.03764v1#bib.bib9)]
    - **Human Development Index** [[10](https://arxiv.org/html/2410.03764v1#bib.bib10)]
    - **World Happiness Index** [[11](https://arxiv.org/html/2410.03764v1#bib.bib11)]
    - **Fragile States Index** [[12](https://arxiv.org/html/2410.03764v1#bib.bib12)]
    - **Inclusiveness Index** [[13](https://arxiv.org/html/2410.03764v1#bib.bib13)]
    - **Gini Coefficient** [[14](https://arxiv.org/html/2410.03764v1#bib.bib14)]
- Peaceful countries: Austria, Australia, Belgium, Czech Republic, Denmark, Finland, Netherlands, New Zealand, Norway, Sweden
- Non-peaceful countries: Afghanistan, Congo, Guinea, India, Iran, Kenya, Nigeria, Sri Lanka, Uganda, Zimbabwe

## III-B Preprocessing

**Preprocessing Steps for Analyzing Peace Speech vs. Hate Speech in Media Articles:**

**I. Extraction of Word Occurrences:**
- Initial extraction focusing on words relevant according to Human Development Indexes
- Frequency of each word within articles from both peaceful and non-peaceful countries

**II. Removal of Proper Nouns:**
- Achieve unbiased results by preventing specific entities or names from skewing analysis
- Prevents disproportionate influence of certain names on the results

**III. Selective Stop Word Removal:**
- Eliminate noise while maintaining potentially insightful linguistic markers
- Targeted removal of articles, auxiliary verbs, conjunctions, and prepositions
- Retained some stop words to reveal meaningful patterns in peaceful versus non-peaceful societies

**IV. Reduction to Top Words:**
- Balance between maintaining relevant words that are reasonably frequent and avoiding overfitting or information loss
- Reduced the number of words per country to 1,000

**V. Normalization of Word Occurrences:**
- Account for variations in total word counts across different countries
- Crucial step to prevent bias due to disproportionate numbers of words within countries
- Calculate normalized word count for each country and word: `Normalized Word Count = (Number of Occurrences of a Word in a Country / Total Number of Words in that Country)`
- Average the normalized frequencies across all countries within peaceful and non-peaceful datasets separately.

These preprocessing steps help prepare the dataset for deeper machine-learning analysis to identify key features of peace speech.

## III-C Modeling

**Linguistic Analysis for Predicting Country Peacefulness**

**Data Preprocessing**:
- Concatenated normalized data from peaceful and non-peaceful countries

**Machine Learning Techniques**:
- Logistic Regression
- Support Vector Machines (SVMs)
- Decision Trees
- Random Forests
- Selected for their unique strengths in classification tasks

**Evaluation Methodology**:
- Cross-verified insights gained to ensure robustness
- Adopted a novel leave-one-out cross-validation approach
- Used one country at a time as holdout set
- Trained models on remaining 19 countries
- Repeated for each country, predicting peace/non-peacefulness based on linguistic profile

**Performance Metrics**:
- Evaluated using accuracy
- Averaged accuracies from each iteration to provide comprehensive view of generalization across diverse contexts

**Detailed Analysis**:
- Compiled results into confusion matrices for each model
- Helped understand strengths and limitations of each technique in predicting peace status from linguistic features.

## III-D Qualitative Inference

**Peaceful vs. Non-peaceful Countries: Linguistic Characteristics Analysis**

**III-D1 Word Clouds:**
- Created to represent linguistic characteristics of peaceful and non-peaceful countries
- Size of each word corresponds to its frequency of occurrence in respective group
- Enabled identification and comparison of prominent themes and terms
- Intuitive exploration of data, recognizing specific linguistic elements in different societies

**III-D2 Dimensionality Reduction:**
- Analyzed relationships and similarities among words using large language model (LLM) for semantic meaning representation
- Employed Principal Component Analysis (PCA) to reduce dimensionality of word embeddings to 2D space
- Enabled visual interpretation of relationships and clusters among words

**III-D3 Clustering:**
- Identified distinct groups or clusters of words sharing similar characteristics using K-means clustering
- Revealed underlying patterns not immediately apparent from word clouds alone
- Manually examined semantic connections between words in different countries

**III-D4 Semantic Segmentation using a Large Language Model:**
- Utilized LLM for semantic segmentation of words collected from word clouds
- Aimed to validate manual clustering results and explore potential of AI in capturing nuanced linguistic relationships
- Dual-method analysis reinforced findings, tested human intuition against AI system's text understanding capabilities.

## IV-A Data and Preprocessing

**Methodology**:
- Top 1,000 words from both peaceful and non-peaceful countries selected.
- Applied all preprocessing steps as outlined in the methodology.
- Remaining unique words: 1,270 post-preprocessing.
- Log-transformation applied to minimize differences between frequencies of certain words after normalization.

## IV-B Modeling Experiments

**Model Evaluation Results**
- **Models Used**: Logistic Regression, SVM, Decision Trees, Random Forests
- **Evaluation Metrics**: Precision, Recall, Accuracy

**Logistic Regression**
- 100% precision
- 100% recall
- 100% accuracy

**SVM**
- 100% precision
- 100% recall
- 100% accuracy

**Decision Tree**
- 100% precision (95% recall)
- 95% accuracy

**Random Forest**
- 100% precision (95% recall)
- 95% accuracy

**Model Selection and Evaluation Methodology**:
- Leave-one-out cross-validation approach
  - One country used as holdout set
  - Remaining countries used for training models
- Hyperparameter tuning using Optuna package
- Primary metric: Accuracy

**Observations and Decision**:
- Good results from all models after hyperparameter tuning
- Logistic Regression and SVM showed higher accuracy, but SVM with linear kernel selected for qualitative analysis
- Random Forest included to provide broader perspective on linguistic features in predicting peace status.

## IV-C Qualitative Results

**IV-C1 Word Clouds**

**Peaceful Word Cloud**:
- Characterized by a discourse focused on:
  - Economic stability
  - Financial markets
  - Growth
  - Terms like "transaction," "aggregate," "year," and "stock" are prominent

**Non-peaceful Word Cloud**:
- Marked by discussions on:
  - Violence
  - Conflict
  - Political instability
  - Prominent terms include "corruption," "kill," "constitution," and "afghan"

**Random Forest Model Word Clouds**:
- Top 50 words based on feature importance
- Words appearing in both peaceful and non-peaceful countries highlighted in blue

**Supporting Information**:
- Figures 1 and 2: Peaceful and Non-peaceful Word Clouds, respectively
- Provided by Climate School and Data Science Institute at Columbia University for student researchers
- Support provided by Toyota Research Institute

**SVM Model Word Clouds**:
- Top 75 words contributing to classification of peaceful and non-peaceful countries based on coefficients
- Words reflect frequency of occurrence in respective group

**IV-C2 Dimensionality Reduction and Clustering**:
- Embedded words using multilingual e5-large model
- PCA transformed word embeddings into 2D space for visual interpretation
- K-means clustering identified distinct groups of words:
  - **Peaceful**: Finance, Personal Social, Health Wellbeing, Tech Development, Markets
  - **Non-peaceful**: Government Public, Legal Conflict, Geopolitical, Media, Analysis Decision

**IV-C3 AI Semantic Segmentation**:
- Aligns with qualitative analysis:
  - Peaceful: Finance, Personal Social, Health Wellbeing, Tech Development, Markets
  - Non-peaceful: Political and Social Issues, Legal Processes, Governance, Leadership, Natural Resources

