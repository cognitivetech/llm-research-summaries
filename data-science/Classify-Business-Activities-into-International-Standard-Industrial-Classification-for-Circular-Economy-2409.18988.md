# Unified Framework to Classify Business Activities into International Standard Industrial Classification through Large Language Models for Circular Economy 

by Xiang Li , Lan Zhao, Junhao Ren, Yajuan Sun, Chuan Fu Tan, Zhiquan Yeo, Gaoxi Xiao
https://arxiv.org/pdf/2409.18988

## Contents
- [Abstract](#abstract)
- [I. INTRODUCTION](#i-introduction)
- [II. BACKGROUND](#ii-background)
- [III. METHODOLOGY](#iii-methodology)
- [IV. EXPERIMENT AND DISCUSSION](#iv-experiment-and-discussion)
- [V. CONCLUSION](#v-conclusion)

## Abstract
**Background:**
- Effective information gathering essential for circular economy recommendation systems
- Lack of standardized framework to represent diverse economic activities hinders construction of centralized knowledge repository

**Approach:**
1. Create a centralized knowledge repository cataloguing waste-to-resource transactions
2. Use LLMs to classify textual data describing economic activities into ISIC system (globally recognized economic activity classification)
3. Categorize any global business activity descriptions into unified ISIC standard

**Challenges:**
- Significant barrier to constructing a centralized knowledge repository due to absence of standardized framework for representing disparate geographical regions' economic activities

**Proposed Solution:**
- Utilize LLMs, such as GPT-2, for multi-class classification task
- Train the model on large datasets of textual data describing various economic activities and their corresponding ISIC codes

**Benefits:**
1. Standardized foundation for knowledge codification and recommendation systems
2. Facilitates creation of a centralized knowledge repository
3. Enables cross-regional implementation of circular economy practices
4. Achieves high accuracy rate (95% on test dataset with 182 labels)
5. Contributes to global efforts in fostering sustainable circular economy practices.

## I. INTRODUCTION

**Introduction:**
- Waste recycling and reusing: promising practices for Circular Economy (CE)
- Industrial Symbiosis (IS): reusing waste or by-products between companies
- Engages industries in collective approach, reducing waste and need for virgin materials
- Potential solution to resource scarcity and environmental degradation
- Existing IS parks showcase benefits and adaptability to different contexts

**Challenges:**
- Lack of a universally standardized framework for economic activity classification
- Countries use different industrial classification systems (e.g., SSIC, NACE)
- Hinders broader adoption of IS and limits opportunities for cross-border collaboration

**Approach:**
- Leverage Large Language Models (LLMs) to classify textual data into the International Standard Industrial Classification (ISIC) framework

**Rationale for ISIC:**
1. Globally recognized, transparent, and inclusive categorization for economic activities
2. Facilitates collection and presentation of statistics for economic scrutiny, decision-making, and policy formulation
3. Adopted by a significant number of nations and used in various statistical domains (e.g., national accounts, enterprise demographics, employment)
4. Incorporated in functional areas like fiscal assessments and business accreditation processes
5. Extensive adoption facilitates comparison of economic activity data on a global scale

**Methodology:**
- Use LLMs for ISIC classification: two-stage framework
1. Identify most suitable model from various candidates by comparing their performance in a simpler task (pre-trained states)
2. Adapt and fine-tune the selected model with an additional trainable classification layer for ISIC code prediction

**Contributions:**
- Studying a novel problem of predicting ISIC codes to enable waste-to-resource matching across regions
- Fine-tuning an LLM with an additional classification layer to predict ISIC codes
- Demonstrating the effectiveness of fine-tuned model through extensive experiments on real-world datasets.

## II. BACKGROUND

**Benefits of Creating a Centralized Knowledge Repository for Universal Waste-to-Resource Matching**

**Waste-to-Resource Matching**:
- Industrial symbiosis is a promising approach to achieve circular economy by reusing wastes or by-products from one company as a resource for another
- Through waste-to-resource exchange, the need for virgin resources and the production of waste could be reduced, leading to both economic and environmental benefits

**Importance of a Database**:
- A database comprising successful historical waste-to-resource matches is beneficial
- These databases, such as MAESTRI and ISDATA, contain information about:
  - Industrial sectors (ISIC / SSIC / NACE) of waste providing and receiving companies
  - Details of the wastes exchanged
- Unifying these historical cases helps analyze various aspects of industrial symbiosis, including influences, emerging mechanisms, and driving factors
- Provides valuable insights for identifying new potential waste-to-resource matches

**Challenges**:
- Regional differences in industrial classification standards (e.g., SSIC vs. NACE) require manual conversion among different codes
- Integrating data into a unified database structure is labor-intensive and time-consuming
- Companies must classify their economic activities into the same standard as used in the database

**ISIC Classification Framework**:
- The ISIC framework is methodically arranged in a hierarchical manner spanning four levels: Sections, Divisions, Groups, and Classes
- The goal is to classify any activity description into the finest ISIC code, i.e., the 4-digit ISIC code

**Use of LLMs for Domain-Specific Text Processing**:
- **Large Language Models (LLMs)**, such as GPT and BERT, represent a significant advancement in Natural Language Processing (NLP)
- Directly exploiting or fine-tuning pre-trained models have become a new paradigm for various domain-specific applications
- Examples include customizing LLMs' tokenizer with equipment data and technical documents to recognize domain-specific terminologies, using BERT as a classification module, and fine-tuning DistiIlBERT model with domain-specified data
- In this paper, the authors propose to fine-tune an LLM with domain-specific data to enable it to capture subtle differences in various activity descriptions and adapt it to ISIC classification

## III. METHODOLOGY

**Methodology Framework for LLMs Deployment in ISIC Code Classification**
- **Data Collection**:
    - Utilizes information from EcoInvent database on activity names and their corresponding ISIC classifications
    - Reduces number of categories from 182 to 48 by using only the first two digits of ISIC codes for model selection phase
- **LLMs Deployment**:
    - Two-phase framework:
        - Phase 1 (Model Selection):
            * Utilizes a spectrum of advanced language models and performs cosine similarity-based classification of 48 ISIC categories
            * Selects the best performing model for further fine-tuning in Phase 2
        - Phase 2 (Model Fine-Tuning):
            * Freezes the selected pre-trained model and adds a new trainable classification layer on top
            * Fine-tunes the model using the original dataset with all 182 categories to improve performance
- **Evaluation Method**:
    - Calculates standard metrics like True Positive, True Negative, False Positive, and False Negative for each class
    - Computes overall accuracy by dividing total number of correct predictions by all predictions
    - Applies weighted macro-averaging to account for label imbalance in the context of ISIC classification's 182 categories.

## IV. EXPERIMENT AND DISCUSSION

**Experiment and Discussion**

**Model Selection**:
- Table 2 shows performance of various candidate models: multi-qa-mpnet-base-cos-v1 (18.20% accuracy), all-mpnet-base-v2 (7.58%), paraphrase-MiniLM-L6-v2 (21.23%), paraphrase-albert-small-v2 (17.79%), RoBERTa (11.35%), and **GPT-2** (27.60%)
- Suboptimal performance can be attributed to:
  - Models not being fine-tuned on the ISIC dataset or nuances of economic activity descriptions
  - Relying solely on semantic similarities in a nuanced domain like economic activities
- **GPT-2** identified as the most promising model for fine-tuning

**Model Fine-tuning**:
- Focus shifts towards fine-tuning the GPT-2 model using a new classification layer
- Implemented **Cross Entropy Loss** and used the **Adam optimizer** with a learning rate of 0.001, trained for 30 epochs
- Model converged at the end of training as shown in Figure 2
- Post-refinement, model showed remarkable improvement:
  - Notable accuracy (95.28%) on test set
  - High precision (95.37%), recall (95.28%), and F1-score (95.27%)
- Fine-tuned GPT-2 model can classify economic activity descriptions into the ISIC framework with high accuracy, enabling automatic classification and waste-to-resource matching globally

## V. CONCLUSION

**Conclusion**
- Research explored novel problem: predicting ISIC (International Standard Industrial Classification) codes for economic activities
- Fine-tuned LLMs used to solve unique task through additional classification layer
- Extensive experiments conducted using real-world datasets, revealing practical applicability and effectiveness in predicting ISIC codes with high accuracy and reliability
- Outcomes indicate significant advancement in field, contributing to improved economic data analysis and classification practices.

