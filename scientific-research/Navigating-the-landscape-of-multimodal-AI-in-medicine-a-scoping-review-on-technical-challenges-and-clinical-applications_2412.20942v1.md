# Navigating the landscape of multimodal AI in medicine a scoping review on technical challenges and clinical applications

source: https://arxiv.org/html/2411.03782v1
by Giulia Nicoletti

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Search Criteria](#2-search-criteria)
- [3 Overview of multimodal medical AI](#3-overview-of-multimodal-medical-ai)
  - [3.1 Modalities and data types](#31-modalities-and-data-types)
  - [3.2 Organ systems, medical tasks and AI functions](#32-organ-systems-medical-tasks-and-ai-functions)
- [4 Methodology](#4-methodology)
  - [4.1 Importance of public data](#41-importance-of-public-data)
  - [4.2 Feature encoding and modality fusion](#42-feature-encoding-and-modality-fusion)
  - [4.3 Network architectures](#43-network-architectures)
  - [4.4 Handling missing modalities](#44-handling-missing-modalities)
  - [4.5 Validation](#45-validation)
- [5 Clinical applications](#5-clinical-applications)
  - [5.1 Nervous system (n=122)](#51-nervous-system-n122)
  - [5.2 Respiratory system (n=93)](#52-respiratory-system-n93)
  - [5.3 Digestive system (n=43)](#53-digestive-system-n43)
  - [5.4 Reproductive system (n=43)](#54-reproductive-system-n43)
  - [5.5 Sensory system (n=25)](#55-sensory-system-n25)
  - [5.6 Integumentary system (n=24)](#56-integumentary-system-n24)
  - [5.7 Cardiovascular (n=20)](#57-cardiovascular-n20)
  - [5.8 Urinary system (n=11)](#58-urinary-system-n11)
  - [5.9 Musculoskeletal system (n=9)](#59-musculoskeletal-system-n9)
  - [5.10 Multiple systems (n=27)](#510-multiple-systems-n27)
  - [5.11 Miscellaneous (n=15)](#511-miscellaneous-n15)
- [6 Towards clinical implementation](#6-towards-clinical-implementation)
- [7 Discussion](#7-discussion)
- [8 Conclusion](#8-conclusion)

## Abstract

**Multimodal AI in Healthcare:**

**Background:**
- Recent technological advances led to growth in patient data
- Artificial intelligence (AI) models analyze individual data modalities
- Recognition of multimodal AI for enhanced clinical decision making

**Scope:**
- Review of deep learning-based multimodal AI applications
- Publication analysis between 2018 and 2024

**Findings:**
- Multiple complementary data sources enhance decision making
- Increasing number of studies on multimodal AI in medical domain
- Consistent outperformance of multimodal models over unimodal ones (average improvement: 6.2 percentage points in AUC)

**Development:**
- Various architectural approaches
- Fusion strategies
- Application areas include various medical disciplines

**Challenges:**
- Cross-departmental coordination
- Heterogeneous data characteristics
- Incomplete datasets

**Technical and Practical Challenges:**
- Developing multimodal AI systems for clinical implementation
- Addressing technical and practical concerns

**Commercially Available Models:**
- Overview of available multimodal AI models for clinical decision making

**Factors Driving Development:**
- Identification of key factors accelerating the field's maturation

**Future Directions:**
- Thorough understanding of current state, challenges, and future directions of multimodal AI in medicine.

## 1 Introduction

**Healthcare Landscape Evolution:**
* Data-centric approach to patient care and decision making [^82]
* Digital pathology [^65], biosensors [^80], next-generation sequencing [^83]
* Complementary data from various modalities
* Macro level overview or single-cell resolution
* Increasing data quantity and quality
* Complexity in selecting optimal treatment
* Traditional approach: multidisciplinary boards [^56]
* Limitations to scalability
* Artificial Intelligence (AI) as a solution [^83]

**Artificial Intelligence in Healthcare:**
* Multimodal AI for decision making [^83]
* Combines multiple streams of information effectively
* Improved predictions based on broader context [^83]
* Few studies, unimodel models are standard

**Challenges in Developing Multimodal AI Models:**
* Cross-departmental nature [^87]
* Variable data modalities and characteristics [CNN vs transformers]
* Inconsistent availability of all modalities for each patient
* Missing data impact on performance [population selection bias]
* Handcrafted features with expert clinicians involved [^36]
* Recent accelerated interest in multimodal AI development [^78]
* Simplified feature extraction using unsupervised learning and deep neural networks as encoders.

## 2 Search Criteria

**Scoping Review on Multimodal AI Models in Medical Field**

**Multimodality Criteria**:
- Integration of diagnostic CT scan and subsequent tissue biopsy slides
- Combines domains of radiologist and pathologist
- Example: T1- and T2-weighted MRI scans not considered multimodal

**Inclusion Criteria for Review**:
- Studies using deep neural networks
- Development of multimodal models for specific medical tasks
- Exclusion of generic visual question answering

**Search Strategy**:
- Literature search in PubMed, Web of Science, Cochrane, and Embase
- Initial search: 12856 results (journal and conference papers)
- Subsequent update: up-to-date overview
- Applied inclusion/exclusion criteria and deduplication

**Screening Process**:
- **Title and Abstract (TiAb) Screening Phase**:
  - Team of six reviewers
  - Reviewed title and abstract based on inclusion criteria
  - Included papers verified by a second reviewer
  - Discrepancies resolved through discussion or involving a third reviewer
  - Criteria: medical domain, in scope, multimodality, deep neural networks
  - Exclusion: non-medical, out of scope, no clinical question
- **Full-Text Screening Phase**:
  - Single reviewer confirmed adherence to inclusion criteria
  - Doubts resolved with a second reviewer
  - Total of 432 studies included in final analysis

## 3 Overview of multimodal medical AI

**Multimodal Artificial Intelligence (AI) in Medical Research**

**Expansion of Multimodal AI**:
- Growing number of articles on integrating multiple modalities between 2018 and 2024
- Rapid increase, from 3 papers in 2018 to 150 papers in 2024

**Organization of Review**:
- **Data Modalities**: distributed according to data types
- **Sections**:
  - **State-of-the-art algorithm designs for multimodal medical AI (section [4](https://arxiv.org/html/2411.03782v1#S4 "4 Methodology ‣ Navigating the landscape of multimodal AI in medicine: a scoping review on technical challenges and clinical applications"))
  - **Clinical tasks (section [5](https://arxiv.org/html/2411.03782v1#S5 "5 Clinical applications ‣ Navigating the landscape of multimodal AI in medicine: a scoping review on technical challenges and clinical applications"))**
    - Detailed analysis of multimodal AI approaches categorized per organ system
  - **Challenges and opportunities for clinical adoption (section [6](https://arxiv.org/html/2411.03782v1#S6 "6 Towards clinical implementation ‣ Navigating the landscape of multimodal AI in medicine: a scoping review on technical challenges and clinical applications"))**
    - Discussion of multimodal AI's challenges and opportunities for clinical adoption
- **Conclusion**: High-level discussion of findings and guidelines for future research directions.

### 3.1 Modalities and data types

**Multimodal Medical AI: Modalities Used and Trends Over Time**

**Categories of Modalities:**
- **Image-based**: radiology (CT, MRI, ultrasound, X-rays, nuclear imaging), pathology (stained histology images), clinical images (OCT, fundus photography, dermatoscopy)
  * If too few papers for medical specialty: grouped as "other images"
- **Non-image-based**: text (structured and unstructured), omics data (genomics, transcriptomics, proteomics), Electroencephalography (EEG), Electrocardiography (ECG) signals

**Prevalence of Modalities:**
1. **Radiology**: 30%
2. **Text**: 30%
3. **Omics**: 12%
4. **Pathology**: 12%

**Trends in Combining Modalities:**
- Preference for integrating one imaging modality with structured or unstructured text data: radiology/text (n=206)
- Pathology/omics/text (n=19), radiology/text/omics (n=15), etc. demonstrate attempts to span multiple scales of biological organization

**Complex Combinations:**
- Radiology/pathology/text (n=7)
- Radiology/pathology/omics/text (n=3)

**Figure 2**: Overview of multimodal medical AI, including all modalities and their respective subtypes.

### 3.2 Organ systems, medical tasks and AI functions

**Multimodal Medical AI Studies: Organ Systems and Medical Tasks**

**Organ Systems:**
- Nervous system: 122 studies
- Respiratory: 93 studies
- Reproductive: 43 studies
- Digestive: 43 studies
- Sensory: 25 studies
- Integumentary: 24 studies
- Miscellaneous: 15 studies
- Multiple organ systems: 27 studies

**Medical Tasks:**
- Diagnosis: 45% to 91% of medical tasks across all organ systems
- Survival prediction: 18% of all medical tasks
  - **Subdivided into:**
    * Survival analysis
    * Disease progression analysis
    * Treatment response analysis
- Other tasks (not uniformly represented)

**Data Modalities:**
- Figure 2 illustrates the distribution, growth trends, and combination trends of data modalities used in reviewed articles.

**Medical Disciplines and Tasks:**
- Figure 3 shows the distribution of medical tasks and data sources across organ systems.
- **Nervous system** and **respiratory system** lead in public data uses.

## 4 Methodology

### 4.1 Importance of public data

**Data Availability Challenge for Multimodal Medical AI**
- **Key challenge**: development of multimodal medical AI is influenced by data availability (Figure 3.2)
- Strong correlation between:
  - Number of models for specific organ system/modality combination
  - Availability of public data sources

**Utilization of Public Datasets in Multimodal AI Research for Medical Applications:**
- Widespread practice, with 61% of data sources coming from public portals:
  - The Cancer Genome Atlas (TCGA): 14%
  - Alzheimer’s Disease Neuroimaging Initiative (ADNI): 8%
  - Medical Information Mart for Intensive Care (MIMIC): 5%
  - The Cancer Imaging Archive (TCIA): 2%
- Other publicly shared data: 15% (GitHub, publisher’s website)
- Private datasets not shared publicly: 24%
- Data from other portals used by fewer than ten reviewed papers: "other data portals" (20%)

### 4.2 Feature encoding and modality fusion

**Multimodal Medical AI Development:**
* Deep neural networks simplified feature extraction/encoding for individual modalities
* Each modality encoded by its deep neural network, results combined for downstream tasks
* Self-supervised learning techniques (DINO, SimCLR) enabled training of feature encoders without labels

**Encoder Types:**
* CNNs: dominant in the landscape (82%), strong correlation with image-based data modalities
* Other: 32%, includes graph neural networks, multi-layer perceptrons
* MLPs: 21%
* Handcrafted feature encoders, RNNs, Transformers used for non-image modalities

**Fusion Stages:**
* Intermediate fusion (79%): data sources fused after encoding but before final layers
* Concatenation method: most common (69%)
* Attention mechanism: 12%
* Late fusion: second most common (14%)
	+ Combine results or predictions of unimodal models, no interaction between modalities
	+ Training data for unimodal models does not have to be paired
* Early fusion: least applied (6%), challenges include input data alignment and extensive preprocessing

**Fusion Techniques:**
* Concatenation of feature vectors
* Attention mechanism to optimally learn from complementary information
* Average or train a model on top of unimodal predictions

**Advantages of Early Fusion:**
* Modalities available in the feature encoding stage, potentially allowing deep neural networks to optimize feature design.

### 4.3 Network architectures

Multimodal AI models' architecture is chosen based on purpose and data availability. Key steps include:

1. Feature extraction
2. Information fusion
3. Final processing

Some models introduce explainability, while others compare different architectures to assess effectiveness. Usage of individual modality encoders has shifted: vision transformers are gaining popularity, while MLPs decrease in usage, and CNNs remain steady. Traditional machine learning methods still appear in 5% of papers for structured text data extraction.

### 4.4 Handling missing modalities

**Handling Missing Data in Multimodal AI Models**

**Data Completeness Assumption**:
- Most multimodal AI models assume data completeness: all modalities must be available for every entry.
- This is often not feasible due to issues like:
  - **Data silos**: Incompatible data archival systems (e.g., PACS)
  - Retrospective nature of studies
  - Data privacy concerns

**Handling Missing Data**:
- Commonly excluding entries with missing modalities and using only complete entries
  - Reduces sample size
  - Limits model inference to complete-modality data
  - Introduces selection biases

**Imputation Techniques**:
- **Non-learning-based approaches**:
  - Primarily used for structured data (clinical, test results)
  - Impute using measures of central tendency or moving average for continuous variables
  - Use mode or add new category for categorical variables
  - Introduce bias and reduce data variability

- **Learning-based approaches**:
  - Leverage machine learning methods to predict missing values
  - Common methods: k-NN, weighted nearest neighbor, linear regression, random forests, neural networks, CART algorithm, MICE, XGBoost
  - Some use generative models (auto-encoders, GANs) or directly predict missing data
  - Others introduce dropout modules or specific loss functions to account for missing modalities

### 4.5 Validation

**Multimodal AI Model Validation:**

**Observations on Validation Schemes:**
- **Internal validation**: most studies (82%) limited to this scheme (papers not using external data for validation)
- Public datasets often used as both training and external validation data
- Comparison with unimodal baseline essential for performance gains assessment

**Study Selection:**
- Subset of 48 papers chosen based on multimodal vs unimodal baseline comparison
- Average improvement in AUC: 6.2 percentage points
- Previously reported improvement: 6.4 percentage points ([^36])

**Ablation Experiments:**
- Majority of studies (72%) report improvement over unimodal models
- No notable improvement for some papers (5%)
- No reported comparison with a unimodal baseline for others (22%)

**Significance Testing:**
- Improvements rarely tested for significance, potentially leading to optimistic bias.

## 5 Clinical applications

Multimodal AI research papers are categorized by applied organ system. Figure [3] shows their distribution among systems. Some studies overlap multiple systems (n=27), while others don't fit into any category (n=15). Below, key contributions in each area are summarized.

### 5.1 Nervous system (n=122)

**Neurodegenerative Disorders and Cancer Diagnosis:**
- **Primary focus**: diagnosis and disease progression of neurodegenerative disorders (Alzheimer's disease) and cancer diagnosis and survival prediction
- **Few studies**: on Parkinson's disease, autism, stroke, schizophrenia, depression
- **Modality used**: MRI is the primary modality for most studies
- **Data integration**: integrating clinical data (n=43), 'omics data (n=18), or both (n=8)
- **Additional modalities**: CT with clinical variables (n=10), pathology with MRI (n=6), or 'omics (n=5)
- **Publicly available datasets**: most frequently used datasets include Alzheimer’s Disease Neuroimaging Initiative (ADNI), The Cancer Genome Atlas (TCGA), CPM-RadPath, and Brain Tumor Segmentation Challenge (BraTS)

**Model Validation:**
- **Limited external validation**: only a few studies conducted external validation (n=15)
- **Internal and cross-validation**: majority employed internal validation (n=40) and cross-validation (n=63)

**Integrating Multiple Modalities:**
- **Enhanced performance**: integrating multiple modalities has enhanced performance in most studies (n=83)

**Innovative Studies:**
- **Patient-wise feature transfer model**: [^46] proposed a patient-wise feature transfer model that learns the relationship between radiological and pathological images, enabling inference using only radiology images while linking the prediction outcomes directly to specific pathological phenotypes.
- **Histological images and genomic data**: [^12] integrates histological images and genomic data and can handle patients with partial modalities by leveraging a reconstruction loss computed on the available modalities.
- **Four different modalities**: [^13] combined radiology images, pathology images, genomic, and demographic data to predict glioma tumor survival and reached a c-index of 0.77 on a 15-fold Monte Carlo cross-validation.

**Advanced Frameworks:**
- **Transformer architectures**: [^51] handled missing data and externally validated their findings using multiple transformer architectures to integrate imaging data and clinical variables, reaching an AUC value of 0.984 on external validation.
- **Sociodemographic data and genetic data**: [^72] proposed a unique type of early fusion where sociodemographic data and genetic data are branded on the MRI scan and used to develop a model able to diagnose early and subclinical stages of Alzheimer’s Disease.
- **Genomic, imaging, proteomic, and clinical data**: [^44] developed a novel framework for AD diagnosis that fuses genomic, imaging, proteomic, and clinical data, validating the proposed method on the ADNI dataset and outperforming other state-of-the-art multimodal fusion methods.
- **Diverse data integration**: [^97] presented an AI model that integrates diverse data, including clinical data, functional evaluations, and multimodal neuroimaging, to identify 10 distinct dementia etiologies across 51,269 participants from 9 independent datasets, even in the presence of incomplete data.

### 5.2 Respiratory system (n=93)

**Respiratory System Research:**

**Focus Areas:**
- Diagnosis (n=19)
- Survival (n=12)
- Treatment response prediction (n=3)
- Disease progression (n=4) for Cancer and Covid-19

**Common Strategies:**
- Combining clinical variables with X-ray or CT imaging of the thorax (n=18 & n=27)
- Integrating chest X-rays with clinical reports for X-ray report generation (n=11)

**Datasets:**
- MIMIC-CXR dataset (n=9)
- The Cancer Genome Atlas (TCGA) (n=5)
- National Lung Screening Trial (NLST) dataset (n=3)

**Performance:**
- Many studies performed better with multimodal integration (n=69)
- Limited external validation (n=15)

**Notable Studies:**
- [^31]: Multimodal graph-based approach combining imaging and non-imaging information to predict COVID-19 patient outcomes.
- [^20]: Outperformed state-of-the-art models on three external validation sets in predicting risk of indeterminate pulmonary nodules using CT imaging and clinical variables.
- [^21]: Proposed a multi-path multimodal missing network integrating multiple data types for lung cancer risk prediction.
- [^93]: Novel lung cancer survival analysis framework using multi-task learning, incorporating histopathology images and clinical information.
- [^39]: Multimodal fusion approach to detect lung disease using X-ray and clinical information.
- [^53]: Examined multimodal fusion types for classifying chest diseases with radiological images and associated text reports.

### 5.3 Digestive system (n=43)

**Digestive System Studies:**
* Most studies focused on malignancies diagnosis (n=25)
* Fewer addressed survival, treatment response, disease progression prediction
* Primary malignancies: colorectal (n=12), liver (n=10), stomach, esophagus, duodenum (n=9)
* No dominant modality combination; clinical variables (n=31) and histopathology slides (n=17) were common
* Multimodal data integration improved performance in 38/39 papers over unimodal approaches
* Highest proportion of externally validated studies (n=11), but limited external validation overall
* Some studies used publicly available datasets, mainly TCGA (n=11) for training rather than external validation

**Noteworthy Studies:**
* [^14] developed a multimodal AI model using endoscopic ultrasonography images and clinical information to distinguish carcinoma from noncancerous lesions of the pancreas, improving diagnostic accuracy for novice endoscopists and reducing skepticism among experienced ones.
* [^11] introduced a deep learning model combining radiology, pathology, and clinical data to predict treatment responses to anti-HER2 therapy or anti-HER2 combined immunotherapy in patients with HER2-positive gastric cancer, managing missing modalities through learnable embeddings.
* [^108] developed and externally validated a multimodal model using PET/CT features, histopathology slides, and clinical data to predict treatment response to bevacizumab in liver metastasis of colorectal cancer patients, achieving an AUC of 0.83 in the external validation set.

### 5.4 Reproductive system (n=43)

**Reproductive Domain:**
- Encompasses studies focused on malignancies diagnosis and prognosis prediction
- Predominant cancers: breast (n=32), prostate (n=5), ovaries, cervix, placenta (n=5)
- No predominant combination of modalities
  - Clinical variables with MRI (n=9) or histopathology (n=6) explored equally
  - Omics data with histopathology (n=5) or clinical data (n=3)
- Most studies demonstrated improved performances when integrating multiple modalities (n=33)
- Few studies performed external validation (n=9)
- Some studies used public datasets for training and internal validation, such as TCGA-BRCA dataset (n=11)

**Interesting Studies:**
- **[^58]**: Deep learning framework that fuses image-derived features with genetic and clinical data to perform survival risk stratification of ER+ breast cancer patients. Improved AUC value ranging from 0.13 to 0.37 compared to six state-of-the-art models.
- **[^25]**: Evaluated various methods for fusing MRI and clinical variables in breast cancer classification, achieving an AUC value of 0.989 on a validation cohort of 4909 patients.
- **[^91]**: Explored multiple instance learning techniques to combine histopathology with clinical features to predict the prognosis of HER2-positive breast cancer patients. Found that simpler modality fusion techniques, such as concatenation, were ineffective in enhancing the model’s performance.

### 5.5 Sensory system (n=25)

**Sensory System Research Focus**
- Most studies focus on **ophthalmology** (n=23)
- Remaining studies address **otology** (n=2)

**Ophthalmology Subspecialties**
- Glaucoma: n=7
- Retinopathy: n=4

**Modality Usage in Studies**
- Optical coherence tomography
- Color fundus photography
- Clinical data

**Multimodal Model Performance**
- More effective than unimodal models: n=19
- Few studies include external validation: n=3

**Notable Studies**
- [^63] Developed deep learning system for diabetic retinopathy progression using color fundus images and risk factors.
  - Used over 160,000 eyes for development
  - Internally and externally validated with around 28,000 and 70,000 eyes, respectively
- [^107] Combined four imaging modalities and free-text lesion descriptions for uncertainty-aware classification of retinal artery occlusion using Transformer architecture.
  - Model can handle incomplete data.

### 5.6 Integumentary system (n=24)

**Dermatology Studies within Integumentary Category**
* Focus on diagnosing skin lesions using dermatoscopic images and clinical variables (n=12)
* Majority compared multimodal approaches to unimodal baselines (n=14)
* Improved performance with added modalities: lesion location, patient demographics
* Four studies conducted external validation, others relied on internal or cross-validation
* Publicly available datasets: ISIC challenge (n=9), HAM10000 (n=4), Seven-point Criteria Evaluation (SPC) (n=3)

**Interactive Dermatology Diagnostic Systems**
* SkinGPT-4: multimodal large language model for dermatology diagnosis
* Based on image and textual descriptions, interactive treatment recommendations
* Evaluated on 150 real-life cases with board-certified dermatologists

**Fusion Schemes for Diagnosing Skin Lesions**
* Proposed by [^85]: two fusion schemes integrating dermatoscopic images, clinical photographs, and clinical variables (n=8 types of skin lesions)

**Multimodal Model Performance**
* [^109]: multimodal model incorporating clinical images and high-frequency ultrasound performed on par or better than dermatologists in diagnosing 17 different skin diseases.

### 5.7 Cardiovascular (n=20)

**Cardiovascular Research Studies:**
- **Diagnosis focus**: Research focused exclusively on this area (n=15)
- Some studies included survival, treatment response, and disease progression prediction (n=4)

**Models Incorporating Multimodalities:**
- Most studies combined clinical variables with a second modality (n=10)
- Radiology imaging was often involved
- Better results when comparing unimodal models to multimodal ones (n=15)
- Only three studies externally validated their results

**Datasets Used:**
- MIMIC, JSRT Database, Montgomery County X-ray Set, GEO, and UK Biobank datasets employed for research

**Notable Study:**
- [^27] study introduced Attention-Based Cross-Modal (ABCM) transfer learning framework
- Merged diverse data types: clinical records, medical imaging, genetic information through attention mechanism
- Achieved high accuracy with an AUC value of 0.97 on validation set
- Surpassed traditional single-source models.

### 5.8 Urinary system (n=11)

**Urogenital System Studies:**
- **Disorders Focused On**: kidney (n=8), bladder (n=2), adrenal gland (n=1)
- **Purpose**: diagnosis (n=7), survival prediction (n=4)
- **Malignancies Primary**: oncological (n=9)
- **Modality Combinations**: CT + second/third modality (n=9)
  - clinical data, histopathology images, omics
- **Performance Improvement**: multimodal models outperformed unimodal ones in 8 studies
- **Validation Approaches**: only a few externally validated results (2/11), others used internal or cross-validation
- **Datasets**: The Cancer Genome Atlas (TCGA), The Cancer Imaging Archive (TCIA)

**Study on Renal Diseases:**
- **Focus**: renal clear cell carcinoma prognostic model development
- **Modality Integration**: deep features from computed tomography and histopathological images, eigengenes derived from genomic data.

### 5.9 Musculoskeletal system (n=9)

**Musculoskeletal System Studies:**
- **Diagnosing Bone Diseases**: focus on osteoarthritis, deep caries and pulpitis, bone fractures, and aging (n=6)
- Radiology images with clinical data or unstructured text data used (n=6 & n=2 respectively)
- Improved performance with multimodal models compared to unimodal ones (n=6)
- No studies externally validated results, employing internal validation approaches instead
- Public datasets from Osteoarthritis Initiative (OAI) and Pediatric Bone Age Challenge used for validation:
  * Over 24000 samples for diagnosing Prosthetic joint infection from CT scan and patients' clinical data [^45]
  * Multicentre dataset of 72 Swedish radiology departments for detecting atypical femur fractures using X-ray imaging and clinical information [^79]

**Study Details:**
- Employed unidirectional selective attention mechanism and graph convolutional network for diagnosing Prosthetic joint infection from CT scan and patients' clinical data, reaching an AUC value of 0.96. [^79]
- Developed multimodal model based on X-ray imaging and clinical information for detecting atypical femur fractures. [^79]

### 5.10 Multiple systems (n=27)

**Multimodal AI Systems Evaluated Across Multiple Organs**

**Studies**:
- Evaluated multimodal system performance on multiple organs (up to 8 types)
- Benefited from publicly available datasets

**Examples of Studies**:
1. **[^3]**: Developed an interpretable multimodal modeling framework for prognostication of 8 cancer types using DNA methylation, gene expression, and histopathology data.
2. **[^9]**: Built a deep learning model to jointly examine pathology whole-slide images and molecular profile data from 14 cancer types to predict outcomes, discovering features correlating with poor and favorable outcomes. Included an explainability component with heatmaps for histopathology and SHAP values for genomic markers.
3. **[^8]**: Constructed a neural network-based model to predict the survival of patients for 20 different cancer types using clinical data, mRNA expression data, microRNA expression data, and histopathology whole slide images (WSIs).
4. **[^88]**: Presented a multimodal deep learning method for long-term pan-cancer survival prediction applied to data from 33 different cancer types.

**Commonality**:
- Reliably demonstrated model performance on wide range of cancer types
- All studies benefited from publicly available TCGA data

### 5.11 Miscellaneous (n=15)

**Multimodal Model Applications (II)**

**Category of Miscellaneous Studies:**
- Diagnosis, recurrence prediction for thyroid carcinoma
- Prediction of Type II diabetes, severe acute pancreatitis, immunotherapy response for DLBCL
- Abnormality detection from chest images
- Fetal birth weight prediction

**Interesting Contributions:**
1. **Autoencoder with Multiple Encoders**: [^35] proposed a model for thyroid cancer recurrence probability prediction using hormonal and pathological data. Significantly improves performance compared to unimodal models.
2. **Unsupervised Learning in Histopathology and Clinical Data**: [^42] combined histopathology and clinical data through knowledge distillation to derive a unimodal histopathology model for predicting immunochemotherapy response of DLBCL patients.
3. **Transformer-based Neural Network Architecture**: [^33] developed a multimodal neural network integrating imaging and non-imaging data for diagnosing up to 25 pathologic conditions using chest radiographs and clinical parameters. Significantly improves diagnostic accuracy compared to imaging alone or clinical data alone.
4. **Pre-trained Multilevel Fusion Network**: [^5] presented a model combining Vision-conditioned reasoning, Bilinear attentions, Contrastive Language-Image Pre-training (CLIP), and stacked attention layers for enhancing feature extraction from medical images and questions. Reduces language bias and improves accuracy on three benchmark datasets.

## 6 Towards clinical implementation

**Multimodal AI Models in Healthcare: Review of Research and Clinical Application**

**Background:**
- Search for multimodal AI models in FDA database and Health AI register
- Preselection based on mention of "multi-modal" or "multimodal"
- Manual inspection for inclusion criteria
- Lack of certified multimodal AI models found

**Findings:**
1. **Prostate Cancer Risk Stratification**: [Miller et al., 2021] developed a multimodal model integrating digital pathology slides and clinical variables for risk stratification in prostate cancer.
   - Employs ResNet50 pretrained model to extract image features from pathology slides
   - Concatenates image feature vector with clinical variable vector for final prediction
   - Obtained higher AUC (0.837) compared to unimodal imaging model (0.779) in predicting distant metastasis after 5 years
   - Incorporated into NCCN guidelines, not FDA-certified
2. **COVID-19 Prognosis and Treatment Selection**: [Chen et al., 2021] developed a multimodal model to predict prognosis and required interventions for COVID-19 patients using chest X-ray abnormality detection and clinical/laboratory data.
   - AUC of 0.854 for multimodel vs. 0.770 for unimodal imaging model
   - Feature importance analysis revealed that age, dyspnea, etc., played a significant role in prediction
   - Not available for clinicians yet but proved potential benefits of current CE-certified unimodal models when used in multimodal pipelines.

**Challenges:**
1. **Data Silos**: Medical data is stored separately in systems like PACS, IMS, and EHRs; interoperability required for AI implementation
2. **Privacy Concerns**: Increasing multimodal data can lead to re-identification of individuals within large datasets, managing these issues crucial to protect patient confidentiality
3. **Explainable AI (XAI)**: Addressing the "black box" nature of AI and ensuring end-users understand and trust decision making processes for widespread adoption in clinical practice.

## 7 Discussion

**Multimodal AI Development in Healthcare**

**Advancements and Challenges**:
- Exponential growth in multimodal AI models highlights recent research efforts in healthcare data integration
- Integrating multiple data modalities leads to notable performance boosts, as revealed by a previous review [^36]
- Significant disparities exist across various medical disciplines, tasks, and data domains
- Applications are most common for the nervous and respiratory systems, with fewer for musculoskeletal and urinary systems
- Radiology imaging data and text data integration outnumbers other combinations
- Variations in model applications, including automated diagnosis vs. disease progression/survival prediction

**Data Collection and Curating**:
- High-quality public datasets availability is a significant bottleneck for multimodal AI development
- Data collection from diverse sources and departments requires extensive curation efforts
- Establishing new multimodal datasets can substantially accelerate AI development in specific domains

**Technical Design Challenges**:
- Most studies focus on developing effective modality fusion methods rather than designing novel encoder architectures
- The diversity of modalities warrants thorough investigation of optimal data fusion methodologies
- Emergence of foundational models may accelerate research in optimal data fusion and generate stronger encoders for handling missing data

## 8 Conclusion

This review provides an extensive overview of multimodal AI development across various medical fields and data domains. While multimodal models show significant performance gains by considering a broader view of patients, their creation poses unique challenges. We hope this review highlights these challenges and offers potential solutions to guide future advancements.

