# Is Personality Prediction Possible Based on Reddit Comments?

by Robert Deimann, Till Preidt, Shaptarshi Roy, Jan Stanicki
https://arxiv.org/abs/2408.16089

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Data Collection and Processing](#2-data-collection-and-processing)
- [3 Related Work](#3-related-work)
- [4 Myers Briggs Type Indicator](#4-myers-briggs-type-indicator)
- [5 Data Collection](#5-data-collection)
- [6 Method](#6-method)
- [7 Evaluation](#7-evaluation)
- [8 Analysis](#8-analysis)
- [9 Future Work](#9-future-work)

## Abstract
**Myers-Briggs Personality Type Prediction Based on Reddit Comments**

**Team Roles and Responsibilities**:
- **Shaptarshi Roy (1643514)**: Data collection, knowledge acquisition, summarizing overviews, providing information for informed decisions.
- **Jan Stanicki (1644014)**: Data collection and processing, reading papers.
- **Robert Deimann (1615725)**: Programming classifiers, evaluating them, analyzing results, contributing to conclusion.
- **Till Preidt (1416911)**: Basic analysis, including language distributions and bag of words representation comparison.

## 1 Introduction
- Natural Language Processing (NLP) covers various tasks related to lexicology, semantics, and pragmatics.
- Predicting an author's personality from their written statements is possible based on NLP techniques.
- Social networks provide a large amount of data for generating psychological profiles.
- Inspiration for the project came from previous work on this topic.

**Myers-Briggs Type Indicator (MBTI)**:
- MBTI is a psychological test used to categorize personality types into 16 classes based on statements like comments.
- The authors collected user comments from Reddit, focusing on comments posted in subreddits related to the MBTI system.
- They distinguished between MBTI data (comments posted in MBTI subreddits) and non-MBTI data (comments posted elsewhere).
- Pre-processing steps included removing HTML artifacts.

## 2 Data Collection and Processing
- The authors collected comments from users who had posted at least one comment in an MBTI subreddit within the last 5 years.
- They distinguished between MBTI and non-MBTI data based on the user's associated subreddit.
- They performed general pre-processing steps, including removing HTML artifacts.

**Classification Techniques**:
- The authors used various classification techniques to compare results and find the best method.

## 3 Related Work

**Related Work on Personality Prediction**

**Abidin et al.** (2020):
- Proposed a Random Forest classifier for personality prediction based on Twitter data
- Dataset: 50 tweets each for 8,657 users, MBTI types derived from forum and questionnaire responses
- Supervised algorithm using decision trees trained on various sets of observations
- Outperformed Logistic Regression and Support Vector Machines

**Amirhosseini and Kazemian** (2020):
- Used Extreme Gradient Boosting for personality prediction, an improvement over basic Gradient Boosting
- Dataset: Same as Abidin et al. (2020)
- Divided classification task into four binary classifiers specialized in each dimension of personality types
- Performed best in recognizing differences between introverted and extroverted dimensions

**Keh and Cheng** (2019):
- Proposed a fine-tuned BERT model pre-trained on sequence classification
- Derived data directly from the Personalitycafe forum, divided into 16 sections for each personality type
- Used a transformer pre-trained on Masked Language Model task to generate personality type-based language
- More successful in imitating extroverted types due to more vocal and active users on the forum

**Sang et al.** (2022a):
- Introduced multi-view multi-row BERT classifier for personality prediction of movie characters
- Inputs are longer than intended for original BERT, handled non-verbal inputs differently
- Dataset contains 3,543 characters from 507 movies and 36.676 subreddits by 13,631 users
- Automatically parsed scripts to fundamental parts using a statistical parser for personality type prediction

**Gjurkovic and Šnajder** (2018):
- Introduced MBTI data set and subset, openly accessible for personality prediction models
- Dataset divided by posts and comments: 22.934.193 comments (583.385.564 words) from 36,676 subreddits by 13,631 users and 354 posts (921.269 words) from 20,149 subreddits by 9,872 users
- Excluded comments from MBTI-related subreddits and replaced mentions of personality types with placeholders for training set
- Largest openly accessible data set for this task, remedied problems related to non-anonymity, limited expression space, and bias towards personality topics.

## 4 Myers Briggs Type Indicator

**The Myers Briggs Type Indicator (MBTI)**

**Background:**
- Developed by Katherine Briggs and her daughter Isabel Briggs Myers in the 1950s
- Based on Carl Jung's theory of psychological types
- Widely used in psychology for self-understanding and personal growth

**Carl Jung's Theory:**
- Psychological types reflect unique ways of experiencing the world
- Two main attitudes: introversion and extraversion
- Four main functions: thinking, feeling, sensation, intuition
- Understanding dominant function crucial for self-awareness and growth

**MBTI Extensions:**
- Standardized framework for identifying personality differences
- Four-letter code distinguishing 16 types (I/E N/S T/F P/J)
- Includes secondary, tertiary, and inferior functions

**Function Descriptions:**
- Introverted vs. Extraverted: direction of dominant function
- Intuition (N): perceives world abstractly
- Sensing (S): grounded in real world
- Thinking (T): bases decisions on rational thought
- Feeling (F): trusts gut feeling for bigger decisions

**Function Interplay:**
- Introverted feelers (INFP, ISFP) have rich emotional inner worlds
- Extroverted feelers (ESFJ, ENFJ) empathetic and direct feelings outwardly

**Acceptance in Academia:**
- Not fully accepted but still widely used
- Proven that friends can accurately guess MBTI type

**Research Approach:**
- Investigating whether MBTI can be guessed from written text
- Building classifiers based on full 16 types, dominant functions (8 labels), and first two functions (also results in 8 labels)
- Comparing results with existing classifiers for binary classifiers of the respective letters.

## 5 Data Collection

**Data Collection Process**

**Reddit Data**:
- Community discussion of MBTI on /r/MBTI subreddit
- Users display their personality type as a "flair"
- Collect data using Pushshift API instead of Reddit's API due to limitations

**Data Retrieval**:
- Collected 1879 users with corresponding labels from the MBTI subreddit
- Enriched data by collecting comments of these users across all of reddit
- Focused on retrieving comments rather than main post submissions

**Preprocessing**:
- Removed 90k "deleted" and "removed" comments, as well as short (<50 char) comments and comments starting with "http" or "r/"

**Dataset Characteristics**:
- Around 6.6M collected comments, reduced to 4.06M after preprocessing
- **630k** comments originate from MBTI-related subreddits
- Masked user labels in comments to avoid model learning label information

**Limitations of the Dataset**:
- Includes real-world data with spelling mistakes and colloquial language
- Population distribution differs from general population, as shown in **Amirhosseini and Kazemian (2020)**
- Users may express irony or sarcasm that could lead to false conclusions about personality type
- Low-quality comments exist in the dataset

**Sampling Approaches**:
- Proportionate sampling: each class represented relative to its occurrence
- Disproportionate sampling: equal number of instances per class, despite imbalance in occurrence

## 6 Method

**Methodology**
- Utilize ALBERT (Lan et al., 2019) for classification: 'albert-base-v2' tokenizer, base model
- Sequence classification with AlbertForSequenceClassification pipeline
- Train on different datasets to evaluate improvement in predicting results
- Learning rate: 2e-5, AdamW optimizer, default value of 12 hidden layers
- Epochs adapted based on best generalizing model
- Collect data from MBTI subreddits and discussions forums

**Data Samples and Training Options**
- Total data set contains 4.06M comments
- Subsets with equal frequency of each personality type
- Distinction between comments from MBTI-related subforums and those excluding them
- Nine options for training classifiers: six subsets and three types of classification (16-fold, 8-fold, binary)
- Total combinations: 54
- Selected a subset and trained 18 classifiers
- Same base ALBERT model used for valid comparisons between fine-tuned models

**Model Configuration**
- Experimented with different training set sizes (64, 10k, 20k examples)
- Batch size: 10 for binary classifiers and 32 for 16-fold MBTI classifier
- Larger batch size beneficial in multi-label classification due to seeing all classes during each training step
- Smaller batch sizes worked reasonably well with binary classifiers

## 7 Evaluation
**Evaluation Metrics and Results**
- Predictions evaluated on balanced test set and imbalanced dataset (original distribution)
- Best F1-score for 16-fold classifier: 16% (significant improvement over random distribution of 6.25%)
- Balanced 16-fold classifier outperformed proportional data set in F1-score
- Merging opposite MBTI classes improved prediction accuracy, but not as significant as desired
- Four binary classifiers: F1-scores between 54 and 61% (highest on T/F axis)
- Training on MBTI subset of comments improved F1-score by 2 percentage points
- Better results achieved using balanced data sets for all training procedures in retrospect.

## 8 Analysis

**Bag-of-Words (BoW) Analysis Approach**

**Background:**
- BoW method requires preprocessing steps: stemming and stop word removal
- Language-dependent issues faced due to different languages used in comments
- Focus on English comments for simplicity
- Other language elements removed manually

**Distribution of Languages:**
- Figure 4 (INTP class) and Figure 6 (ESFJ class) show overrepresentation of English comments
- Similar distribution pattern for all classes

**Relevant Words:**
- Highly represented words in INTP: "people," "would," "think," "make," and "know"
  - Polite requests, self-related formulations, intuitive expressions or introversion
- ESFJ class: "feel," "think," "people," "would," and "know"
  - Aspects of extroversion but also sense and feeling

**Classification Analysis:**
- Improved performance when training on balanced data set (Figure 7)
  - Rarest class ESFJ represented more frequently in balanced training
  - Model recognizes opposite types as very different, validating MBTI concept
- Major classes INTP, INTJ, and INFP predicted most often after training on proportionate data set
- Difficulty predicting personality based on a single comment.

**Improvements:**
- Considerations: more representative data, handling multiple languages more effectively
- Possible improvements discussed in section 9.

## 9 Future Work
**Future Work and Improvements**

**Adjusting Model:**
- Fine-tuning models: changing learning rate, optimizer, activation functions
- Adjusting architecture based on hardware and time expenditure
- Capturing context information for irony detection
- Larger model to process more tokens without being limited like BERT
- Multi-GPU training for efficiency

**Improving Preprocessing:**
- Dealing with emojis, special characters, different comment lengths, languages
- Balancing datasets to achieve a compromise between equal sizes and proportionality of classes

**Focusing on User Behavior:**
- Classifying multiple or all comments of a user instead of single ones
- User-based distinction at a previous stage
- Including context in the model for improved accuracy predictions

**Conclusion:**
- Predicting personality types based on Reddit comments using ALBERT classifier
- Encounters with issues during data collection and classification process
- Observed significant tendency towards correct prediction but questionable if all aspects of personality can be captured by text snippets alone.
