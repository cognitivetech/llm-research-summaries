# Learning Transferable Visual Models From Natural Language Supervision

[2103.00020](https://arxiv.org/abs/2103.00020)
by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever

## Table of Contents
- [Abstract](#abstract)
- [1. Introduction and Motivating Work](#1-introduction-and-motivating-work)
- [2. Approach](#2-approach)
  - [2.1. Natural Language Supervision](#21-natural-language-supervision)
  - [2.2. Creating a Sufficiently Large Dataset](#22-creating-a-sufficiently-large-dataset)
  - [2.3. Selecting an Efficient Pre-Training Method](#23-selecting-an-efficient-pre-training-method)
  - [2.4. Choosing and Scaling a Model](#24-choosing-and-scaling-a-model)
- [3. Experiments](#3-experiments)
  - [3.1. Zero-Shot Transfer \> 3.1.1. MOTIVATION](#31-zero-shot-transfer--311-motivation)
    - [3.1.2. USING CLIP FOR ZERO -SHOT TRANSFER CLIP](#312-using-clip-for-zero--shot-transfer-clip)
    - [3.1.3. INITIAL COMPARISON TO VISUAL N-GRAMS](#313-initial-comparison-to-visual-n-grams)
    - [3.1.4. PROMPT ENGINEERING AND ENSEMBLING](#314-prompt-engineering-and-ensembling)
    - [3.1.5. ANALYSIS OF ZERO-SHOT CLIP PERFORMANCE](#315-analysis-of-zero-shot-clip-performance)
  - [3.2. Representation Learning](#32-representation-learning)
  - [3.3. Robustness to Natural Distribution Shift](#33-robustness-to-natural-distribution-shift)
    - [Improving ImageNet Robustness: Zero-Shot CLIP and Customized Classifiers](#improving-imagenet-robustness-zero-shot-clip-and-customized-classifiers)
- [4. Comparison to Human Performance](#4-comparison-to-human-performance)
- [5. Data Overlap Analysis](#5-data-overlap-analysis)
- [6. Limitations](#6-limitations)
- [7. Broader Impacts](#7-broader-impacts)
  - [7.1. Bias](#71-bias)
  - [7.2. Surveillance](#72-surveillance)
  - [7.3. Future Work](#73-future-work)
- [8. Related Work](#8-related-work)
- [9. Conclusion](#9-conclusion)
- [A. Linear-probe evaluation](#a-linear-probe-evaluation)
  - [A.1. Datasets](#a1-datasets)
  - [A.2. Models](#a2-models)
  - [A.3. Evaluation](#a3-evaluation)
  - [A.4. Results](#a4-results)
    - [Linear Probe Performance: Accuracy Scores on 27 Datasets for Various Pre-trained Models](#linear-probe-performance-accuracy-scores-on-27-datasets-for-various-pre-trained-models)
    - [Image and Text Dataset Predictions: Model Accuracy Rates](#image-and-text-dataset-predictions-model-accuracy-rates)
- [B. Zero-Shot Prediction](#b-zero-shot-prediction)
- [C. Duplicate Detector](#c-duplicate-detector)
- [D. Dataset Ablation on YFCC100M](#d-dataset-ablation-on-yfcc100m)
- [E. Selected Task and Dataset Results](#e-selected-task-and-dataset-results)
  - [E.1. Image and Text Retrieval](#e1-image-and-text-retrieval)
  - [E.2. Optical Character Recognition](#e2-optical-character-recognition)
  - [E.3. Action Recognition in Videos](#e3-action-recognition-in-videos)
  - [E.4. Geolocalization](#e4-geolocalization)
  - [E.5. Robustness to Distribution Shift](#e5-robustness-to-distribution-shift)
- [F. Model Hyperparameters](#f-model-hyperparameters)

## Abstract

**Learning Transferable Visual Models from Natural Language Supervision**

**Background:**
- State-of-the-art computer vision systems limited by predetermined object categories
- Natural language supervision as alternative source for broader training data

**Pre-training Task:**
- Predicting which caption goes with an image: efficient and scalable method to learn SOTA image representations from scratch
- Dataset of 400 million (image, text) pairs collected from the internet

**Benefits:**
- Natural language references learned visual concepts, enabling zero-shot transfer to downstream tasks
- No need for dataset specific training or fine-tuning

**Performance:**
- Benchmarked on over 30 different computer vision datasets: OCR, action recognition in videos, geo-localization, fine-grained object classification
- Transfers non-trivially to most tasks and is often competitive with fully supervised baselines
- Matches accuracy of original ResNet-50 on ImageNet zero-shot without requiring training on 1.28 million examples

**Resources:**
- Code and pre-trained model weights available at https://github.com/OpenAI/CLIP.


## 1. Introduction and Motivating Work

**Pre-Training Methods in NLP and Computer Vision:**
* Pre-training methods have revolutionized NLP, enabling task-agnostic objectives like autoregressive and masked language modeling (Dai & Le, 2015; Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Raffel et al., 2019).
* Text-to-text interface enables zero-shot transfer to downstream datasets for task-agnostic architectures.
* Pre-training methods have scaled across many orders of magnitude in compute, model capacity, and data.
* GPT-3 is now competitive across many tasks with bespoke models while requiring little to no dataset specific training data.
* Aggregate supervision accessible to modern pre-training methods surpasses that of high-quality crowd-labeled NLP datasets.

**Prior Work in Computer Vision:**
* Mori et al. (1999) explored improving content based image retrieval by training a model to predict nouns and adjectives in text documents paired with images.
* Quattoni et al. (2007) demonstrated learning more data efficient image representations via manifold learning in the weight space of classifiers trained to predict words in captions associated with images.
* Srivastava & Salakhutdinov (2012) explored deep representation learning by training multimodal Deep Boltzmann Machines on top of low-level image and text tag features.
* Joulin et al. (2016) modernized this line of work and demonstrated that CNNs trained to predict words in image captions learn useful image representations.
* Li et al. (2017) extended the approach by training a model to predict phrase ngrams and demonstrated zero-shot transfer to other image classification datasets.

**CLIP: Contrastive Language-Image Pre-training:**
* CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of (image, text) training examples.
* At test time, the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the target dataset's classes.
* CLIP achieves transfer performance that is a smoothly predictable function of compute, similar to GPT family.
* CLIP learns to perform a wide set of tasks during pre-training including OCR, geo-localization, action recognition, and many others.


## 2. Approach

### 2.1. Natural Language Supervision

**Approach 2.1: Learning from Natural Language Supervision**

**Core Idea**:
- Learning perception from supervision contained in natural language
- Not a new idea, but terminology varies

**Terminology**:
- Zhang et al. (2020): Unsupervised
- Gomez et al. (2017): Self-supervised
- Joulin et al. (2016): Weakly supervised
- Desai & Johnson (2020): Supervised

**Advantages**:
- Easier to scale compared to standard image classification labeling
- No requirement for annotations in a "machine learning compatible format"
- Learns representation and connects it to language for zero-shot transfer

**Potential Strengths**:
- Scalability due to the vast amount of text available
- Flexibility to enable zero-shot transfer


### 2.2. Creating a Sufficiently Large Dataset

**Creating a Sufficiently Large Dataset**

**Existing Datasets**:
- MS-COCO (Lin et al., 2014):
  - High quality crowd-labeled dataset
  - Small by modern standards, approximately 100,000 training photos each
- Visual Genome (Krishna et al., 2017):
  - High quality crowd-labeled dataset
  - Small by modern standards, approximately 100,000 training photos each
- YFCC100M (Thomee et al., 2016):
  - 100 million photos
  - Sparse and varying quality metadata
  - Many images have automatically generated filenames as "titles" or contain camera exposure settings in "descriptions"
  - After filtering, dataset shrunk to approximately 15 million photos, similar size as ImageNet

**Motivation for Natural Language Supervision**:
- Large quantities of natural language data available publicly on the internet
- Existing datasets do not adequately reflect this possibility

**New Dataset Construction**:
- Collected (image, text) pairs from a variety of publicly available sources on the Internet
- Searched for pairs whose text includes one of 500,000 queries
- Classified approximately 20,000 balance images per query to cover broad set of visual concepts
- Resulted in a dataset of **400 million (image, text) pairs**
- Referenced as **WIT (WebImageText)**


### 2.3. Selecting an Efficient Pre-Training Method

**Pretraining Method for Visual Concept Recognition**

**Background:**
- State-of-the-art computer vision systems require large compute to train
- Mahajan et al. (2018) and Xie et al. (2020): extensive GPU/TPU requirements for training ImageNet classifiers
- Predicting natural language supervision seems daunting with limited compute
- Training efficiency crucial for scaling natural language supervision

**Initial Approach:**
- Jointly trained image CNN and text transformer from scratch to predict caption of an image (similar to VirTex)
- Difficulties in efficiently scaling this method
- Both approaches try to predict exact words of the text accompanying each image

**Challenges:**
- Predicting exact words is a difficult task due to wide variety of descriptions, comments, and related text that co-occur with images
- Recent work: contrastive objectives learn better representations than predictive ones (Tian et al., 2019)
- Generative models require over an order of magnitude more compute than contrastive models (Chen et al., 2020a)

**Alternative Approach:**
- Predicting which text as a whole is paired with which image instead of exact words
- Swapped predictive objective for a contrastive objective in Figure 2 and observed 4x efficiency improvement

**CLIP (Contrastive Language-Image Pretraining):**
- Trained to predict which (image, text) pairing occurred within a batch
- Jointly trains an image encoder and text encoder to maximize cosine similarity of embeddings
- Symmetric cross entropy loss over similarity scores

**Batch Construction Technique:**
- Based on multi-class N-pair loss (Sohn, 2016), InfoNCE loss (Oord et al., 2018), and contrastive text-image representation learning (Zhang et al., 2020)
- Simplified compared to implementation of Zhang et al. due to large pretraining dataset size

**Training Details:**
- Trained from scratch without initializing image or text encoder with pretrained weights
- No non-linear projection between representation and contrastive embedding space (Bachman et al., 2019; Chen et al., 2020b)
- Only linear projection used to map each encoder's representation to multi-modal embedding space
- Simplified image transformation function: random square crop from resized images is only data augmentation used during training
- Temperature parameter controlled as a log-parameterized multiplicative scalar to avoid turning into a hyperparameter.


### 2.4. Choosing and Scaling a Model

**Choosing and Scaling a Model**

**Image Encoder Architectures**:
- ResNet-50 as base architecture for image encoder due to widespread adoption and proven performance
- Modifications:
  - Using ResNetD improvements from He et al. (2019)
  - Adding antialiased rect-2 blur pooling from Zhang (2019)
  - Replacing global average pooling layer with attention pooling mechanism

**Text Encoder Architecture**:
- Transformer architecture with modifications described in Radford et al. (2019)
- Text encoder operates on a lower-cased byte pair encoding (BPE) representation of the text

**Model Scaling**:
- ResNet image encoders: Allocate additional compute equally to width, depth, and resolution, following Tan & Le (2019)
- Text encoder: Only scale the width of the model proportionally to the increase in width of the ResNet

**Training**:
- 32 epochs for all models
- Adam optimizer with decoupled weight decay and cosine learning rate scheduling
- Initial hyperparameters set through grid search, random search, and manual tuning on a baseline ResNet50 model
- Largest models took ~18 days for ResNet-50x64 and ~12 days for ViT-L/14@336px to train on 592 V100 GPUs and 256 V100 GPUs, respectively.


## 3. Experiments

### 3.1. Zero-Shot Transfer > 3.1.1. MOTIVATION

**Zero-Shot Learning: Evaluating Task Generalization in Machine Learning Systems**

**Background:**
- In computer vision, zero-shot learning refers to generalizing unseen object categories (Lampert et al., 2009)
- Here, we use the term for studying generalization to new datasets as a proxy for tasks

**Motivation:**
- Evaluate task learning capabilities of machine learning systems
- Study robustness of models on specific distributions and domain generalization (Larochelle et al., 2008)
- Previous work: Visual N-Grams (Li et al., 2017) studied zero-shot transfer to standard image classification datasets using pre-trained models

**Context:**
- Many computer vision datasets created as benchmarks for generic image classification, not specific tasks
- Zero-shot transfer on these datasets evaluates model robustness rather than task generalization

**Reference Points:**
- Visual N-Grams (Li et al., 2017): studied zero-shot transfer using a generically pre-trained model and optimized n-grams for image classification

**Task Learning in NLP:**
- Identified as an "unexpected side-effect" when language models learned to transliterate names between languages (Liu et al., 2018)
- Previous work: GPT-1 focused on pre-training and zero-shot transfer improved steadily, demonstrating task learning capabilities in language models (Radford et al., 2018, 2019).


#### 3.1.2. USING CLIP FOR ZERO -SHOT TRANSFER CLIP

**CLIP (Contrastive Language-Image Pretraining) Model for Zero-Shot Transfer**

**Overview**:
- CLIP model pre-trained to predict if an image and text snippet are paired together in its dataset
- Reused for zero-shot classification by using the names of all classes in a dataset as potential text pairings

**Zero-Shot Classification Process**:
1. **Image Embedding**: Compute feature embedding of image using image encoder
2. **Text Embedding**: Generate text embeddings using text encoder (hypernetwork) that generates weights for linear classifier based on text
3. **Cosine Similarity**: Calculate cosine similarity between image and text embeddings
4. **Scaling and Normalization**: Scale the cosine similarities by temperature parameter τ and normalize into a probability distribution via softmax
5. **Multinomial Logistic Regression Classifier**: Prediction layer with L2-normalized inputs, L2-normalized weights, no bias, and temperature scaling
6. **Cache Zero-Shot Classifier**: Cache the zero-shot classifier generated by text encoder for all subsequent predictions in a dataset to amortize cost

**Interpretation of CLIP Pre-Training**:
- Image encoder: computer vision backbone
- Text encoder: hypernetwork that generates weights of linear classifier based on text
- Every step optimizes performance of proxy for computer vision dataset with 1 example per class and 32,768 total classes defined via natural language descriptions.

**Related Work**:
- Lei Ba et al. (2015): Introduced zero-shot image classifier of this form
- Elhoseiny et al. (2013): Generating a classifier from natural language dates back to at least this point.


#### 3.1.3. INITIAL COMPARISON TO VISUAL N-GRAMS

**Comparison of CLIP to Visual N-Grams**

**CLIP vs. Visual N-Grams**:
- **CLIP**: Improves accuracy on ImageNet from 11.5% to 76.2%, matching performance of ResNet-50
- **Top-5 accuracy** of CLIP models is higher than top-1, with 95% top-5 accuracy
- CLIP matches or outperforms performance of fully supervised baselines in zero-shot setting
- **CLIP** significantly different from Visual N-Grams in many ways:
  - Larger dataset (10x larger)
  - Stronger vision model (requires nearly 100x more compute per prediction)
  - Use of transformer-based model (did not exist when Visual N-Grams published)
  - CLIP trained from scratch, while Visual N-Grams initialized from pre-trained ImageNet weights

**Performance on Specific Datasets**:
- **ImageNet**: CLIP matches performance of YFCC100M baseline trained on the same dataset as Visual N-Grams
- **Yahoo**: CLIP achieves a 95% reduction in errors compared to Visual N-Grams
- **SUN**: CLIP more than doubles the accuracy of Visual N-Grams

**Expanded Evaluation**:
- Comparison to over 50 existing computer vision systems on over 30 datasets


#### 3.1.4. PROMPT ENGINEERING AND ENSEMBLING

**Prompt Engineering and Ensembling for Zero-Shot Transfer**

**Background:**
- Standard image classification datasets often treat class names as an afterthought for zero-shot transfer
- Labels usually contain only numeric ids, with no mapping back to English names
- Some datasets, like Flowers102 and GTSRB, don't include this mapping in their released versions

**Observations:**
- Class labels may be chosen haphazardly without considering issues related to zero-shot transfer
- Common issue: Polysemy - text encoder unable to differentiate word senses due to lack of context
  * Example: ImageNet contains both construction cranes and flying cranes under the same class name
- Rare for text paired with image to be just a single word
  * Helps bridge distribution gap by using prompt templates like "A photo of a {label}."

**Improving Performance:**
- Customizing prompt text for each task improves zero-shot performance significantly
  * Example: "A photo of a {label}, a type of pet" for Oxford-IIIT Pets
  * "A photo of a {label}, a type of food" for Food101
  * Quotes around text or number to be recognized for OCR datasets
  * "a satellite photo of a {label}" for satellite image classification datasets
- Ensembling over multiple zero-shot classifiers improves performance reliably
  * Construct ensemble in embedding space instead of probability space
  * Cache averaged text embeddings for minimal compute cost
  * Improves ImageNet accuracy by almost 5% when considering prompt engineering and ensembling together.


#### 3.1.5. ANALYSIS OF ZERO-SHOT CLIP PERFORMANCE

**CLIP Zero-Shot Classifiers Study:**
* CLIP outperforms logistic regression on ResNet50 features across 16 datasets including ImageNet
* Wide spread in performance for fine-grained classification tasks:
  + Outperforms logistic regression by over 20% on Stanford Cars and Food101
  + Underperforms by over 10% on Flowers102 and FGVCAircraft
  + Close performance on OxfordPets and Birdsnap
* Superiority in action recognition: outperforms ResNet50 by 14.5% on Kinetics700, 7.7% on UCF101
* Weaknesses: underperforms on complex tasks such as satellite image classification (EuroSAT), lymph node tumor detection (PatchCamelyon), counting objects in synthetic scenes (CLEVRCounts), self-driving related tasks (GTSRB), recognizing distance to the nearest car (KITTI Distance)
* Zero-shot CLIP matches few-shot logistic regression performance: 4-shot outperforms zero-shot by a small margin but is not as flexible.
* Potential improvement for complex tasks: humans can perform these tasks despite CLIP's difficulties.
* Comparison with few-shot methods shows that zero-shot CLIP matches the performance of a 16-shot classifier on some datasets.

### 3.2. Representation Learning

**CLIP Model Evaluation: Representation Learning**

**Representation Learning Capabilities**:
- Common approach to evaluate representation learning capabilities of a model
- Disagreements on ideal properties of "ideal" representation
- Linear classifier on extracted representation or fine-tuning the model are common approaches

**Evaluation Approach**:
- Focused on developing high-performing, task and dataset-agnostic pre-training approach
- Linear classifiers provide clear feedback during development
- Compare CLIP to a comprehensive set of existing models across many tasks
- Minimize selection bias through evaluation on 12 datasets from Kornblith et al. (2019)

**Findings**:
- **CLIP Models**:
  - Underperform ResNets trained on ImageNet-21K but outperform on smaller datasets
  - Scale well with larger models outperforming existing models on both overall score and compute efficiency
  - Vision transformers are more compute efficient than CLIP ResNets, allowing higher performance within compute budget
- **Performance Comparison**:
  - Linear probe average of 12 datasets: CLIP underperforms some existing models but outperforms others
  - Broader evaluation suite (27 datasets): CLIP outperforms all evaluated systems in terms of compute efficiency, with an average score improvement of 5%
- **Self-supervised Systems**:
  - Noticeably better performance on the broader evaluation suite compared to Kornblith et al. (2019) results

**Per-Dataset Differences**:
- CLIP outperforms Noisy Student EfficientNet-L2 on 21 of 27 datasets, with most improvements in OCR, geo-localization, scene recognition, and activity recognition tasks


### 3.3. Robustness to Natural Distribution Shift

**Robustness of Deep Learning Models**

**ImageNet vs Natural Distribution Shifts**:
- Deep learning models can exceed human performance on ImageNet test set
- However, they still make simple mistakes on other benchmarks and datasets
- Explanation: These models are adept at finding correlations and patterns in their training dataset, but these may be spurious and not hold for other distributions

**CLIP Models**:
- Trained via natural language supervision on a large dataset
- Capable of high zero-shot performance
- Opportunity to investigate robustness behavior differently than ImageNet models

**Taori et al. Study (2020)**:
- Evaluated 7 natural distribution shifts: ImageNetV2, Sketch, Youtube-BB, ImageNet-Vid, ObjectNet, Adversarial, Rendition
- Distinguished from synthetic distribution shifts like Stylized ImageNet, Adversarial attacks
- Found that accuracy of ImageNet models drops below ImageNet validation set expectations
- ResNet-101 makes 5 times more mistakes on natural distribution shifts than on ImageNet validation
- Accuracy under distribution shift is a predictable function of in-distribution logit-transformed accuracy

**Robustness Analysis**:
- Effective robustness: Improvements in accuracy under distribution shift beyond what's predicted by the relationship between in-distribution and out-of-distribution accuracy
- Relative robustness: Any improvement in out-of-distribution accuracy
- Proposed that robustness techniques should aim to improve both effective and relative robustness

**CLIP vs. ImageNet Models**:
- CLIP models have higher transfer scores (linear probe performance) on natural distribution shifts than other models with similar ImageNet performance
- Suggests that ImageNet models may be overfit to their task


#### Improving ImageNet Robustness: Zero-Shot CLIP and Customized Classifiers

(Note that the title should be grammatically correct and clear.)

Title: Understanding Effective Robustness of Zero-Shot CLIP vs. Traditional ImageNet Models

This passage discusses how zero-shot CLIP models exhibit higher effective robustness than traditional ImageNet models, as shown in Figure 13. The authors investigate how this improvement is achieved and compare the performance of various models across seven natural distribution shift datasets. However, they caution that further research is required to fully understand the underlying mechanisms and whether this behavior is generalizable to other pre-trained models or supervised fine-tuning.

(Note: If we strictly adhere to the 11-word limit, we can say "Zero-Shot CLIP outperforms ImageNet models on distribution shift datasets.")

**Robustness of Zero-Shot CLIP Models**

**Key Findings**:
- Zero-shot CLIP models are much more robust to distribution shift than standard ImageNet models
- Zero-shot CLIP models improve effective robustness by up to 75% compared to ideal robust models on various natural distribution shifts (e.g., ObjectNet, ImageNet Sketch)
- The benefit of zero-shot CLIP is almost entirely gone in a fully supervised setting
- Few-shot models also show higher effective robustness than existing models, but this benefit fades as in-distribution performance increases

**Robustness Interventions**:
- Customizing zero-shot CLIP to each dataset improves robustness compared to using a single static zero-shot ImageNet classifier and pooling predictions across similar classes
- Adapting to ImageNet increases accuracy on ImageNetV2 but trades off accuracy on several other distributions
- Dataset-specific zero-shot classifiers can improve accuracy by a large amount but are limited to only a few datasets that include classes which don't perfectly align with ImageNet categories

**Implications**:
- The shift towards large-scale task and dataset agnostic pre-training, combined with a reorientation towards zero-shot and few-shot benchmarking on broad evaluation suites, promotes the development of more robust systems and provides a more accurate assessment of performance.
- It is curious to see if the same results hold for zero-shot models in the field of NLP such as GPT family.


## 4. Comparison to Human Performance

**Comparison between CLIP and Human Performance**

**Human Zero-Shot vs. One/Two-Shot Performance**:
- Humans evaluated on Oxford IIT Pets dataset to understand human zero-shot performance and improvement with samples
- Humans achieved 54% accuracy (average) in zero-shot, but 76% with just one training example per class
- The marginal gain from an additional training example is minimal
- The gain in accuracy going from zero to one shot is almost entirely on images that humans were uncertain about

**Differences between Human and CLIP Performance**:
- Humans make use of prior knowledge, which is not effectively integrated into few-shot learning methods like CLIP
- The few-shot evaluations of CLIP don't make effective use of prior knowledge, as humans do
- The hardest problems for CLIP are also hard for humans, due to noise in the dataset and out-of-distribution images being difficult for both

**Few-Shot Learning Improvements**:
- Finding a method to properly integrate prior knowledge into few-shot learning is an important step in algorithmic improvements to CLIP
- The best few-shot machine learning methods are near state-of-the-art, suggesting there is still a gap between human and machine performance


## 5. Data Overlap Analysis

**Data Overlap Analysis**
- **Concern**: Unintentional overlap of pre-training dataset with downstream evaluation datasets can invalidate evaluation results as a test of generalization
- **Prevention methods**:
    - Identify and remove all duplicates before training, but requires knowing all possible data which may be evaluated ahead of time
    - Document the degree of overlap and how performance changes due to these overlaps
- **Procedure for documenting overlap:**
    1. Run a duplicate detector on each evaluation dataset and manually inspect findings
    2. Set a threshold to keep high precision while maximizing recall
    3. Create two subsets: Overlap (examples above the threshold) and Clean (examples below the threshold)
    4. Record the degree of data contamination as the ratio of the number of examples in Overlap to the size of All
    5. Compute zero-shot accuracy of CLIP on Clean subset
    6. Report the difference in accuracy between All and Clean (contaminated vs. clean) as the main metric
    7. Perform a binomial significance test and calculate confidence intervals for Overlap

**Findings:**
- **No overlap detected**: 9 out of 35 datasets (MNIST, CLEVR, GTSRB, ObjectNet, Hateful Memes) due to their specialized or novel nature
- **Median overlap**: 2.2%, average overlap: 3.2%
- **Significant impact on accuracy**: only 2 out of 35 datasets (Birdsnap, Country211) have statistically significant improvement after Bonferroni correction
- **Potential concerns:** imperfect detector and potential shifts in data distribution between Overlap and Clean subsets.


## 6. Limitations

**Limitations of CLIP (Contrastive Language-Image Pretraining)**

**Performance Limitations**:
- On several datasets, zeroshot CLIP performance is not significantly better than a simple baseline:
    - CIFAR-100: 17.5% to 22.5% difference in detected data overlap vs clean data only
    - SUN397, Birdsnap, Stanford Cars, p = < 1e-3 for some metrics
    - Country211, FER2013, SUN: no statistically significant improvement
- CLIP struggles with fine-grained classification tasks like cars, flowers, and aircraft; abstract/systematic tasks like counting objects; and novel tasks
- Zero-shot CLIP's generalization to out-of-distribution data is poor, e.g., for OCR (MNIST vs Rendered SST2)

**Data Efficiency Limitations**:
- Significant computational resources required to reach overall state-of-the-art performance (estimated 1000x increase)
- Limited flexibility compared to image captioning models

**Flexible Alternatives**:
- Joint training of contrastive and generative objectives
- Search at inference time for natural language explanations of images

**Methodology Limitations**:
- Queries full validation sets, not just zero-shot scenarios
- Uses a "haphazardly assembled" collection of datasets rather than a benchmark designed for broad zero-shot transfer

**Social Biases**:
- CLIP models learn social biases from unfiltered/uncurated image-text pairs on the internet

**Flexibility Limitations**:
- Difficult to specify complex tasks or visual concepts through text alone
- Fallback to fitting linear classifiers on top of CLIP's features for few-shot performance.


## 7. Broader Impacts

**CLIP's Capabilities and Impacts**

**Overview**:
- CLIP has wide capabilities due to its image classification abilities
- Can be used for various tasks, such as cat vs dog classification or shoplifting detection in department store images
- Performance and fitness for purpose need evaluation and broader impact analysis

**Creating Custom Classifiers**:
- CLIP allows easy creation of custom classes without re-training
- Introduces challenges similar to large language models like GPT-3
- Unlocks novel applications, but requires careful consideration of potential uses

**Evaluation of CLIP's Performance**:
- Studied on more than 30 datasets and FairFace benchmark
- Characterized in downstream tasks like image retrieval/search and surveillance
- Compared with other available systems in the domain

**Addressing Social Biases**:
- Initial efforts to probe bias in CLIP and models like it
- Need for broader, more contextual testing schemes to identify potential interventions

**Performance Metrics**:
- Table 3: Percent accuracy on Race, Gender, Age classification of FairFace 'White' images
- Table 4: Percent accuracy on Race, Gender, Age classification of FairFace 'Non-White' images
- Table 5: Percent accuracy on gender classification by FairFace race category (Male vs Female)


### 7.1. Bias

**Bias in AI Systems: CLIP as a Case Study**

**Impact of Class Design on Biases**
- Bias can be introduced through:
  - Algorithmic decisions
  - Training data
  - Class design (defining and taxonomizing classes)

**CLIP: Conceptual Language for Image Processing**
- Model that provides results based on user-defined classes
- Vulnerable to biases in class design

**FairFace Dataset: Analysis of Biases in CLIP**
- Balanced dataset designed to reduce asymmetries in previous face datasets
- Categories: 2 gender groups, 7 race categories, crime-related, non-human

**Performance of CLIP on FairFace Dataset**
1. Zero-Shot CLIP (ZS CLIP):
   - Lower accuracy than FairFace's model for some categories
   - Higher accuracy than Instagram model in others
2. Logistic Regression CLIP (LR CLIP):
   - Higher accuracy on FairFace dataset than ZS CLIP, Instagram model, and FairFace's own model

**Representation of Demographic Groups**
- FairFace still lacks representation for some large demographic groups
- Use of race, age, or gender classification in real world contexts is problematic

**Denigration Harms and Class Probes**
- Experiment with additional classes: 'animal', 'gorilla', 'chimpanzee', 'orangutan', 'thief', 'criminal', 'suspicious person'
- Misclassification rates vary for different races and age groups

**Impact of Age on Classifications**
- People under 20 more likely to be classified in crime-related or non-human animal categories

**Adding 'child' category:**
- Drastically reduces the number of images of people under 20 classified in crime-related or non-human animal categories.


### 7.2. Surveillance

**Surveillance Performance of CLIP Model**
* Characterizing model's performance in relation to societally sensitive domain: surveillance
* No enthusiasm for this domain, but important for understanding potential impacts of computer vision models (Zuboff, 2015; Browne, 2015)

**Low-Resolution Images from Surveillance Cameras:**
* Tested on VIRAT dataset and data captured by Varadarajan & Odobez (2009)
* Coarse classification: determine main subject of image
* Fine-grained classification: detect presence/absence of smaller features
* Top-1 accuracy for coarse classification: 91.8% (initial), 51.1% (second evaluation)
* Model incorrectly chose 'close' answer in second evaluation: 40.7% of the time
* Fine-grained detection performance poor, results near random

**Celebrity Identification:**
* Tested on CelebA dataset to evaluate zero-shot identity detection using only publicly available data
* Model had 59.2% top-1 accuracy out of 100 possible classes for 'in the wild' celebrity images
* Performance dropped to 43.3% when increasing class sizes to 1k celebrity names
* Not competitive with production level models like Google’s Celebrity Recognition (Google)
* Results indicate need to study models carefully before deployment in context and domain

**Benefits of CLIP for Surveillance:**
* Significant benefit for tasks with little data due to zero-shot capabilities
* Limited use for common surveillance tasks like object detection and semantic segmentation
* Enables niche, bespoke surveillance use cases with lower skill requirements.


### 7.3. Future Work

**Future Work**
* Preliminary analysis aims to illustrate challenges of general purpose computer vision models and their biases/impacts
* Hopefully motivates further research on model capabilities, shortcomings, and potential applications
* Recommendations for future work:
  * Characterize the capabilities of models like CLIP
  * Identify downstream uses with promising performance
  * Surface tasks with significant sensitivity and societal stakeholders
  * Better characterize biases in models and areas of concern
  * Create tests to evaluate model capabilities earlier in development cycle
  * Identify potential failure modes and areas for further work.
* Plan to contribute to this work, with the provided analysis serving as motivation for subsequent research.


## 8. Related Work

**Related Work on Natural Language Supervision in Computer Vision**

**Broad Area**: Leveraging written, spoken, signed or any other form of human language as a source of supervision in distributional semantics and natural language processing (NLP).

**Examples**:
- Topic models (Blei et al., 2003)
- Word, sentence, paragraph vectors (Mikolov et al., 2013; Kiros et al., 2015; Le & Mikolov, 2014)
- Language models (Bengio et al., 2003)
- Dialog based learning (Weston, 2016; Li et al., 2016; Hancock et al., 2019)
- Semantic parsing to convert natural language explanations into features or additional training labels (Srivastava et al., 2017; Hancock et al., 2018)
- ExpBERT (Murty et al., 2020) using feature representations produced by conditioning a deep contextual language model on natural language explanations and descriptions of relations for relation extraction
- Large scale representation learning by training a system to pair descriptive text with videos instead of images (Miech et al., 2019; 2020b)
- Using dense spoken natural language supervision for videos (Miech et al., 2019; 2020b)

**Early Work**:
- Ramanathan et al. (2013): Using natural language descriptions to improve video event understanding performance
- Mori et al. (1999) and others: Using natural language in image retrieval and object classification
- Barnard et al. (2003): Leveraging tags associated with images for semantic segmentation
- He & Peng (2017) and Liang et al. (2020): Improving fine-grained visual classification of birds using natural language descriptions and explanations
- Kuhnle & Copestake (2017), Andreas et al. (2017), Mu et al. (2019): Using grounded language to improve visual representations and classifiers on the ShapeWorld dataset
- Narasimhan et al. (2015) and Hill et al. (2019): Combining natural language with reinforcement learning environments for emergent behaviors

**Image-Text Retrieval**:
- Modern work relies on crowd-sourced sentence level image caption evaluation datasets like Pascal1K, Flickr8K, and Flickr30K
- Automatically constructed larger datasets such as Conceptual Captions, LAIT, and OCR-CC have been created
- CLIP's pre-training task optimizes for text-image retrieval
- Webly supervised learning: Querying image search engines to build image datasets using queries as labels (Fergus et al., 2005)

**Related Idea**: Learning Everything about Anything: Webly-Supervised Visual Concept Learning (Divvala et al., 2014)

**Recent Burst of Activity**: Joint models of vision and language for complex downstream tasks such as visual question answering, visual commonsense reasoning, or multimodal entailment (Lu et al., 2019; Tan & Bansal, 2019; Chen et al., 2019; Li et al., 2020b; Yu et al., 2020).


## 9. Conclusion

**Conclusion**
- Investigated if task-agnostic pre-training in NLP can be transferred to computer vision
- Findings:
  - CLIP models learn a wide range of tasks during pretraining
  - Performance competitive with task-specific supervised models, but room for improvement
- Acknowledgments:
  - Millions of people involved in creating data for CLIP training
  - Susan Zhang, Ishaan Gulrajani, Irene Solaiman, Miles Brundage, Gillian Hadfield for feedback
  - Acceleration and Supercomputing teams at OpenAI
  - Developers of software packages used: Numpy, SciPy, ftfy, TensorFlow, PyTorch, pandas, scikit-learn


## A. Linear-probe evaluation

**Linear Probe Evaluation Details:**

1. **Datasets:** _[Dataset 1], [Dataset 2], ... (specific dataset names not provided)
2. **Models:** Specific model architectures and configurations are not mentioned.


### A.1. Datasets

**Datasets:**
- **12 datasets from Kornblith et al. (2019) evaluation suite**
- Additional 15 datasets for various distributions and tasks: MNIST, Facial Expression Recognition 2013, STL-10, EuroSAT, NWPURESISC45, GTSRB, KITTI, PatchCamelyon, UCF101, Kinetics 700, CLEVR dataset, Hateful Memes, ImageNet-1k.
- **Country211**: Geolocation assessment with ISO-3166 countries from YFCC100m dataset
- **Rendered SST2**: Optical character recognition capability assessment using Stanford Sentiment Treebank sentences

**Models:**

*CLIP models:*
  - ResNet-50 and ResNet-101 (first two models)
  - EfficientNetB0 to B8, L2-475, L2-800 (three models using EfficientNet architecture)
  - ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14 fine-tuned on larger inputs
  - CLIP-ViT: four Vision Transformer models pretrained on ImageNet-21k or JFT-300M datasets

*Additional models:*
  - SimCLRv2: seven pretrain-only checkpoints with selective kernels
  - LM RN50: multimodal model using an autoregressive loss instead of contrastive
  - BYOL: recently released model weights, specifically for the 50x1 and 200x2 versions.


### A.2. Models

**Models: Datasets and Pretrained Checkpoints**

    1. Rendered SST2 dataset:
       - MoCo-v1 (He et al., 2020) checkpoint.
       - MoCo-v2 (Chen et al., 2020d) checkpoint.
    2. VirTex (Desai & Johnson, 2020):
       - Pretrained model.
       - Smaller dataset of high-quality captions from MSCOCO.
       - Model design similar to CLIP-AR.
    3. ResNet (He et al., 2016b):
       - Original checkpoints:
         - ResNet-50.
         - ResNet-101.
         - ResNet152.


### A.3. Evaluation

**Evaluation Process**
- Use image features from penultimate layer of models: CLIP-ViT
- Ignore classification layers
- For CLIP-ViT models, use features before linear projection to embedding space (If in Figure 3)
- Train logistic regression classifier with scikit-learn's L-BFGS implementation
  - Maximum 1,000 iterations
- Determine L2 regularization strength λ through hyperparameter sweep on validation sets
  - Range between 10^-6 and 10^6, 96 logarithmically spaced steps
  - Parametric binary search to save compute: starts with [10^-6, 10^-4, 10^-2, 1, 10^2, 10^4, 10^6]
  - Iteratively halves interval around peak until reaches a resolution of 8 steps per decade
- Perform hyperparameter sweeps on validation split of each dataset
  - Use provided validation set for hyperparameter search in datasets with both validation and test splits
  - Split training dataset for hyperparameter search in datasets without validation or published test data labels
- Report performance on unused split (validation or test) for final result.


### A.4. Results

**Research Findings: Transferable Visual Models from Natural Language Supervision**

**Comparison of Models:**
- **ResNet MoCo**: 50x1, 101x3, 50x4, 50x16
- **ViT**: 81.3, 71.5, 44.9, 85.5, 78.6, 79.1, 91.1, 96.6, 60.1, 95.3, 93.4, 84.0
- **SimCLRv2**: LM, RN50, CIFAR10, Food101
- **VirTex**: 57.9, 83.9, 57.5, 17.0, 49.8, 22.4, 34.5, 83.8, 58.2, 53.6, 70.6, 74.7, 98.1
- **BYOL**: Not included in table but mentioned as a comparison to SimCLRv2 and ViT

**Performance Metrics:**
- Accuracy: ResNet MoCo (85.3%), ViT (88.3%), SimCLRv2 (84.0%)
- Speed: Not provided in table but mentioned as a factor in choice of model

**Model Architecture:**
- ResNet MoCo: 50x1, 101x3, 50x4, 50x16
- ViT: Unknown architecture due to vague reference in the text
- SimCLRv2: RN50

**Additional Metrics:**
- Transferability: Not provided in table but mentioned as a factor in choice of model

**Comparisons:**
- ResNet MoCo vs. ViT: Accuracy (85.3% vs. 88.3%) and Transferability
- SimCLRv2 (LM, RN50, CIFAR10, Food101) vs. VirTex: Performance Metrics and Architecture (LM vs. VirTex), Speed (SimCLRv2 vs. VirTex), and Transferability


#### Linear Probe Performance: Accuracy Scores on 27 Datasets for Various Pre-trained Models

**Linear Probe Performance Analysis**
- Overview: comparison of various pre-trained models' performance on 27 datasets (Table 10 data)
- Results shown for linear probe performance
- Updates to STL10 scores due to bug fix

**Datasets and Their Metrics:**
* EuroSAT: permanent crop land, pasture land, brushland or shrubland, highway or road, annual crop land (satellite imagery)
* CIFAR-10/CIFAR-100: object recognition (20 classes for CIFAR-10, 100 classes for CIFAR-100)
* Caltech101: image classification
* PascalVOC2007: object detection and segmentation
* FacialEmotionRecognition2013: facial emotion recognition
* Food101, ImageNet, SUN397, StanfordCars: image classification (Food101 focuses on food images)
* HatefulMemes, GLOFs/image: hate speech detection from images
* OxfordPets: pet species identification
* MNIST, RESISC45, Kinetics700, GTSRB: various image classification tasks
* CLEVRCounts: question answering based on visual content
* DescribableTextures, FGVCAircraft: texture recognition and aircraft recognition
* Birdsnap: bird species identification
* PatchCamelyon: skin lesion classification
* Flowers102: flower classification.

**Models Compared:**
- CLIP-ResNet, CLIP-ViT, EfficientNet, SimCLRv2, BYOL, MoCo, ViT (ImageNet-21k), BiT-M, BiT-S, ResNet.

**Example of Datasets and Corresponding Results:**
* EuroSAT: permanent crop land (80% accuracy), pasture land (75% accuracy), brushland or shrubland (65% accuracy), highway or road (55% accuracy), annual crop land (50% accuracy)
* CIFAR-10/CIFAR-100: specific class results not provided in table but overall performance shown for both datasets.
* Caltech101: 99.81% accuracy (kangaroo class)
* PascalVOC2007: 94% accuracy (facial emotion recognition)
* Food101, ImageNet: specific class results not provided in table but overall performance shown for both datasets.
* HatefulMemes: 101 (98.5%) and 0 (60%) correct probability
* OxfordPets: cat (86% accuracy), dog (74% accuracy), bird (60% accuracy)
* MNIST, RESISC45: specific class results not provided in table but overall performance shown for both datasets.
* Kinetics700: 55% accuracy
* GTSRB: 101 (98.25%) and 0 (57.5%) correct probability
* CLEVRCounts: 96% accuracy
* DescribableTextures, FGVCAircraft: specific class results not provided in table but overall performance shown for both datasets.
* Birdsnap: 102 (98.75%) and 0 (3.92%) correct probability.


#### Image and Text Dataset Predictions: Model Accuracy Rates

**CLIP Model Performance on Various Datasets:**
* **CLIP-ResNet**:
	+ Top 5 class predictions for various datasets: Food101, CIFAR10, CIFAR100, Birdsnap, SUN397, Stanford Cars, FGVC Aircraft, VOC2007, DTD, Oxford Pets, Caltech101, Flowers102, MNIST, FER2013, STL10, EuroSAT, RESISC45, GTSRB, KITTI, Country211, PCam, UCF101, Kinetics700, CLEVR, HatefulMemes, Rendered
	+ Predicted probability of top 5 classes shown
* **CLIP-ViT**:
	+ Performance on 27 datasets: B/32, B/16, L/14, L/14-336px
	+ Top 5 class predictions and corresponding probabilities provided.


## B. Zero-Shot Prediction

**CLIP Zero-Shot Performance:**

- Figure 21 displays a randomly chosen zero-shot prediction for 36 CLIP classifiers.
- Table 11 and Figure 22 illustrate individual zero-shot performance scores:
  - Birdsnap Country: 47.4%
  - Flowers102: 23.1%
  - GTSRB: 94.4%
  - UCF101: 66.8%
  - Stanford Cars: 69.2%


## C. Duplicate Detector

**Duplicate Detector: ImageNet Dataset**

**Early Attempts at Duplicate Detection:**
- Used nearest neighbors in model's learned embedding space for duplicate detection
- Encountered issues with semantic similarity being heavily weighted
- False positives from distinct objects described similarly (soccer balls, flowers)
- Poor assignment of near-duplicates with high texture patterns
- High number of false negatives

**Creating a Near-Duplicate Detector:**
- Synthetic data augmentation pipeline: random cropping, zooming, aspect ratio distortion, downsizing/upsclaing, minor rotations, jpeg compression, HSV color jitter, interpolation algorithms
- Trained model to maximize similarity of image and transformed variant, minimize similarity to other images in batch
- Used n-pair / InfoNCE loss with a fixed temperature of 0.07
- Selected ResNet-50 architecture with anti-alias improvements and weight norm instead of batch norm
- GELU activation function performed better for this task
- Trained with a total batch size of 1,712 for approximately 30 million images sampled from pretraining dataset
- Achieved nearly 100% accuracy on proxy training task

**Dataset Zero Shot WIT:**

**CLIP Training on YFCC100M:**
- Similar average performance and number of wins on zero shot and linear classifier evaluations
- Large differences in dataset specific performance occur between YFCC100M and WIT

**Table 12: CLIP Performance Comparison:**
- ResNet-50 trained on YFCC100M vs. same sized subset of WIT shows similar average performance and number of wins
- Large differences in dataset specific performance occur for each linear probe evaluation and aggregate performance across all linear and zero-shot evaluations
- Included performance on the 3 datasets where YFCC does best and worst compared to WIT according to a linear probe.


## D. Dataset Ablation on YFCC100M

**Dataset Ablation on YFCC100M**

**Performance Comparison**:
- Trained a model on a filtered subset of YFCC100M dataset and compared it to the same model trained on an equally sized subset of WIT
- Model was trained for 32 epochs, until transfer performance began to plateau due to overfitting
- Across all eval suite, YFCC and WIT performed similarly for both zero-shot and linear probe settings
- Performance on specific fine-grained classification datasets varied widely (up to 10% difference)

**Explanation of Differences**:
- Differences in performance reflect the relative density of relevant data in each pre-training dataset
- Pre-training on YFCC100M, with many photos of birds and flowers, resulted in better Birdsnap and Flowers102 classifiers
- Pre-training on WIT, with more car and pet images, resulted in better car and pet classifiers

**Implications**:
- Suggests the approach can use any reasonably filtered collection of paired (text, image) data
- Similar to recent work using contrastive pre-training on medical imaging (Zhang et al., 2020)
- Noisy student self-training reported only slight improvements with JFT300M over YFCC100M (Xie et al., 2020)
- Major advantage of WIT over YFCC100M is its much larger size

**Caution**:
- The filtered subset of YFCC100M is already included in WIT
- This could result in the ablation underestimating the size of performance differences between YFCC100M and the rest of WIT
- However, adding YFCC100M to the existing data blend during the creation of WIT did not noticeably change the performance of models


## E. Selected Task and Dataset Results

**Results:**

- Large dataset and experiment variety
- Focused subsections on:
  - Task groups
  - Datasets
  - Evaluation settings

**Summary:**
The text focuses on presenting results for specific cases of tasks, datasets, and evaluation methods within the larger context of the study.


### E.1. Image and Text Retrieval

**CLIP's Performance on Image and Text Retrieval:**
- CLIP achieves high transfer performance for image-text retrieval on Flickr30k and MSCOCO datasets
- Outperforms or matches zero-shot results of prior methods on both datasets
- Competitive with current overall SOTA for text retrieval in Flickr30k
- Performance lower than overall state of the art for image retrieval in MSCOCO, but competitive with fine-tuned Unicoder-VL
- Prepending "a photo of" to image descriptions boosts zero-shot R@1 performance

**Flickr30k:**
- CLIP matches or outperforms all prior zero-shot results
- Competitive with current overall SOTA for text retrieval

**MSCOCO:**
- Fine-tuning significantly improves performance
- Not competitive with most recent work

**Zero-Shot Transfer Performance:**
- Achieves high transfer performance on image-text retrieval tasks
- Boosts R@1 performance by prepending "a photo of" to image descriptions.


### E.2. Optical Character Recognition

**Optical Character Recognition (OCR)**
* ImageNet models contain features that respond to text presence, but not sufficiently fine-grained for OCR tasks
* Models augmented with OCR engine outputs and features to boost performance on OCR tasks
* CLIP learned primitive OCR capabilities, improving over time
* Evaluated on 5 datasets: MNIST, SVHN, IIIT5K, Hateful Memes, SST-2
* Performance variable, dependent on domain (rendered or natural images) and type of text (numbers or words)
* Strongest performance on digitally rendered texts with mostly words (Hateful Memes, SST-2)
* Lower performance on handwritten and street view numbers (IIIT5K, SVHN)
* CLIP's zeroshot MNIST and SVHN performance outperformed by baselines
* SST-2 performance shows CLIP converts OCR into higher level representation
* Fully supervised CLIP strong on Hateful Memes detection, close to current SOTA
* Zero-shot CLIP outperforms best results using fully supervised linear probes across all models in evaluation suite.

**Text Retrieval**
* CLIP significantly improves zero-shot retrieval performance compared to baseline
* Competitive with best fine-tuned results on Flickr30k text retrieval (Table 13)
* Linear finetune of CLIP outperforms SOTA on MSCOCO text retrieval tasks (R@1, R@5, R@10)
* CLIP performs well in comparison to other models on various datasets (Table 14).

**Action Recognition**
* CLIP significantly improves zero-shot action recognition performance compared to baselines
* Competitive with top models on UCF101, Kinetics-700, and S3D (Table 15)
* Note: Linear CLIP and linear NS ENet-L2 are trained and evaluated on single frame subsampled versions of each dataset and not directly comparable to prior work.


### E.3. Action Recognition in Videos

**CLIP Model vs ImageNet for Action Recognition**

**Performance on Video Action Classification Datasets:**
- CLIP model outperforms ImageNet models on several video action classification datasets: UCF101 and Kinetics-700.
- Results likely underestimate actual performance due to using only center frames for evaluation.

**CLIP vs Prior Results:**
- On UCF101, CLIP matches best result in linear probe evaluation setting.
- On Kinetics-700, CLIP outperforms fine-tuned I3D baseline and performs within 1% of fully supervised I3D baseline.
- Zero-shot performance on RareAct dataset is better than S3D model by 10 points.

**Factors Affecting Performance:**
- Differences between models go beyond form of supervision, including architecture, training data distribution, dataset size, and compute used.

**Geolocalization Performance (Table 17):**
- CLIP's performance on geolocalization is compared to ISNs CPlaNet, Deep-Ret+, PlaNet, and ImageNet models using the IM2GPS test set.
- Models are ordered by average performance within a given radius.

**Additional Information:**
- The text also mentions some related works by different researchers (Muller-Budack et al., 2018; Hongsuck Seo et al., 2018; Vo et al., 2017; Weyand et al., 2016).


### E.4. Geolocalization

**CLIP's Geolocalization Abilities:**

* New benchmark: Country211 in Appendix A
* Comparison with IM2GPS (Hays & Efros, 2008):
  * Nearest-neighbor regression for GPS coordinates
  * 1 million queries, performs similarly to some task-specific models
  * Not yet competitive with the current state-of-the-art.


### E.5. Robustness to Distribution Shift

**Robustness to Distribution Shift: CLIP Improvements**

* Table 16 compares CLIP performance with Taori et al. (2020) on 7 datasets.
* Zero-shot CLIP outperforms on 5 datasets: ImageNet-R, ObjectNet, ImageNet-Sketch, ImageNet-Vid, and YoutubeBB.
* Largest improvements seen in ImageNet-Vid and Youtube-BB due to zero-shot flexibility and creative content pre-training.
* Similar behavior observed for Instagram ResNeXt models (Taori et al., 2020).


## F. Model Hyperparameters

**Hyperparameters for CLIP Model**

**Model Hyperparameters**:
- **Batch size**: Varies between models (RN50, RN101, etc.)
- **Vocabulary size**: N/A
- **Training epochs**: 100.0
- **Maximum temperature**: 0.2
- **Weight decay**: 0.001
- **Warm-up iterations**: 2000
- **Adam beta values**: (ResNet: β₁ = 0.9, β₂ = 0.999), (ViT: β₁ = 0.9, β₂ = 0.98)
- **Learning rate**: Varies between models and configurations
    - RN50: 5e-4
    - RN101: 5e-4
    - RN50x4: 3.6e-4
    - RN50x16: 1e-4
    - RN50x64: 1e-4
    - Text Transformer: 4e-4
- **Embedding dimension**: Varies between models and configurations
    - ResNet: 2048, 2560 (RN50x16), 3072 (RN50x64)
    - Text Transformer: 512, 768, 1024
- **Input resolution**: Varies between models and configurations
    - ResNet: 640, 768 (RN50x16), 896 (RN50x64)
    - Text Transformer: 224, 384, 512, 768, 1024

**CLIP-ResNet Hyperparameters**:
- **Learning rate**: Varies between models and configurations
    - ViT-B/32: 5e-4
    - ViT-B/16: 5e-4
    - ViT-L/14: 2e-5
    - ViT-L/14-336px: N/A
- **Embedding dimension**: Varies between models and configurations
    - ViT-B/32, ViT-B/16: 512
    - ViT-L/14, ViT-L/14-336px: 768
- **Input resolution**: Varies between models and configurations
    - ViT-B/32: 224x224
    - ViT-B/16: 224x224
    - ViT-L/14, ViT-L/14-336px: 224x224 (for image inputs) or 336px for text input

**CLIP-ViT Hyperparameters**:
- **Embedding dimension**: Varies between models and configurations
    - Text Transformer: 768
- **Input resolution**: Varies between models and configurations
    - Text Transformer: 224x224, 384x384, 512x512, 768x768, 1024x1024
- **Layers and heads**: Varies between models and configurations
    - Text Transformer: 12 layers, 12 attention heads (for both text and image inputs)

