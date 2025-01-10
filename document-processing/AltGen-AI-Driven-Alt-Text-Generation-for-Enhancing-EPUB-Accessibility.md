# AltGen AI-Driven Alt Text Generation for Enhancing EPUB Accessibility

source: https://arxiv.org/html/2501.00113v1
by Yixian Shen, Hang Zhang, Yanxin Shen, Simon Fraser, Lun Wang, Chuanqi Shi, Shaoshuai Du

## Contents
- [Abstract.](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
  - [2.1. Challenges of EPUB Accessibility in Compliance with WCAG Guidelines](#21-challenges-of-epub-accessibility-in-compliance-with-wcag-guidelines)
  - [2.2. AI-Driven Alt Text Generation for EPUB Images](#22-ai-driven-alt-text-generation-for-epub-images)
- [3. Methodology](#3-methodology)
  - [3.1. Data Preprocessing](#31-data-preprocessing)
  - [3.2. Generative AI Model Integration](#32-generative-ai-model-integration)
  - [3.3. Metadata Enrichment](#33-metadata-enrichment)
  - [3.4. File Reconstruction](#34-file-reconstruction)
  - [3.5. Postprocessing and Validation](#35-postprocessing-and-validation)
- [4. Evaluation](#4-evaluation)
  - [4.1. Experimental Setup](#41-experimental-setup)
  - [4.2. Evaluation Metrics](#42-evaluation-metrics)
  - [4.3. Results and Analysis](#43-results-and-analysis)
  - [4.4. Comparative Analysis](#44-comparative-analysis)
- [5. Summary](#5-summary)

## Abstract.

**Accessibility of EPUB Files: Digital Inclusivity through Alt Text Generation**

**Background:**
- Digital accessibility is crucial for inclusive content delivery
- Many EPUB files lack fundamental accessibility standards, especially descriptive alt text for images
- Importance of alt text for visually impaired users and assistive technologies

**Challenges:**
- High resource intensity to generate high-quality alt text at scale
- Achieving contextually relevant and linguistically coherent descriptions

**Introducing AltGen:**
- AI-driven pipeline for automating alt text generation for EPUB files
- Utilizes state-of-the-art generative models: transformer architectures, CLIP, ViT

**AltGen Pipeline:**
1. Data preprocessing: extract and prepare relevant content
2. Visual analysis: computer vision models (CLIP, ViT) for feature extraction
3. Contextual enrichment: integrate visual features with surrounding text information
4. Language model fine-tuning: generate descriptive alt texts based on context
5. Validation: quantitative metrics (cosine similarity, BLEU scores), qualitative feedback from visually impaired users

**Experimental Results:**
- Efficacy of AltGen across diverse datasets
  * 97.5% reduction in accessibility errors
  * High scores in similarity and linguistic fidelity metrics

**User Studies:**
- Practical impact on document usability and comprehension reported by participants

**Comparative Analyses:**
- AltGen outperforms existing approaches in terms of accuracy, relevance, and scalability.

**Keywords:**
- Alt text generation
- Generative AI
- EPUB accessibility
- Computer vision
- Natural language processing
- Digital accessibility
- Transformer models
- Content inclusivity

## 1. Introduction

**Revolutionizing Digital Content Accessibility: An Overview of EPUB Files and Alt Text Generation**

**EPUB Files**:
- Widely adopted standard for digital content due to adaptability and compatibility with assistive technologies
- Positioned as cornerstone of digital content accessibility

**Challenges in EPUB File Accessibility**:
- Significant proportion fall short of meeting accessibility standards, particularly in providing alternative text (alt text) for images
- Absence or inadequacy of alt text severely undermines usability and inclusivity of digital documents

**Importance of Alt Text**:
- Foundational element of digital accessibility
- Serves as textual substitute for images, enabling visually impaired users to comprehend visual content through screen readers

**Manual Alt Text Generation Challenges**:
- Requires domain expertise, contextual understanding, and considerable time
- Infeasible for organizations managing vast digital libraries
- Critical for compliance with accessibility regulations like Web Content Accessibility Guidelines (WCAG)

**Need for Automated Solutions**:
- Recent advancements in generative artificial intelligence offer new avenues to address the alt text generation challenge
- Transformer architecture-based AI models excel at natural language processing and computer vision tasks

**Introducing AltGen Pipeline**:
- Novel AI-driven framework to automate alt text generation for EPUB files
- Integrates advanced generative models, computer vision models, and transformer-based generative models
- Preprocessing stage: extracts images, textual content, and metadata from EPUB files
- Contextualization stage: uses surrounding text to ensure generated descriptions are descriptive and relevant
- Validation stage: combines quantitative metrics (cosine similarity, BLEU scores) and qualitative user feedback to evaluate output.

**Contributions**:
1. Comprehensive framework for AI-driven alt text generation tailored to EPUB files
2. Rigorous evaluation demonstrating significant improvements over existing methods
3. Novel integration of computer vision and natural language processing techniques, setting a new benchmark for accessibility enhancements in digital content.

## 2. Related Work

### 2.1. Challenges of EPUB Accessibility in Compliance with WCAG Guidelines

**Accessibility Challenges with Digital Files: EPUB vs WCAG Standards**

**Background:**
- WCAG standards not consistently adopted for digital accessibility, especially EPUB files [^20]
- Publishers recognizing importance of accessibility but full compliance remains elusive [^5]
- Challenges exist in W3C HTML standards adoption as well [^26]

**EPUB vs PDF:**
- EPUB offers superior accessibility features compared to PDF [^8]
- PDF dominance due to longstanding usage and conversion tools result in non-compliant EPUB files [^8]

**Barriers to WCAG Adoption:**
1. **Recent popularity of EPUB**: Relatively newer format compared to PDF [^8]
2. **Reliance on conversion tools**: Lack of structural and metadata elements for compliance [^8]
3. **Limited adoption of W3C standards**: Similar challenges in HTML [^26]

**Addressing the Gaps:**
- Better tools: Embed accessibility features during EPUB creation [^8]
- Improved workflows: Ensure compliance with global standards [^8]

### 2.2. AI-Driven Alt Text Generation for EPUB Images

**Improving Accessibility in EPUB Files through AI-Generated Alt Text**

**Importance of Automatic Image Description Generation**:
- Critical area for improving accessibility in EPUB files
- Transformative potential of generative AI: creates contextually relevant and high-quality descriptions for diverse image types

**Advancements in AI-Powered Services**:
- **AltText AI**: Leverages large-scale datasets to train sophisticated models that produce accurate and context-sensitive alt text
- Research efforts focus on optimizing efficiency of AI systems: operating at scale with reduced computational overhead

**Validation through User-Centric Evaluations**:
- Measures technical accuracy and practical impact on end-user accessibility
- Feedback methodology aligns with established practices in evaluating AI-generated content

**Integration of Advanced Models**:
- **CLIP**, **ViT**, and **GPT** models ensure visually descriptive and contextually aligned alt text generation.

## 3. Methodology

**Methodology for AI-driven Alt Text Generation in EPUB Files**

**Pipeline Stages**:
1. **Data Preprocessing**:
   - File parsing and image identification

2. **Generative AI Model Integration**:
   - Feature extraction
   - Contextual analysis
   - Alt text generation

3. **Metadata Enrichment**:
   - Language detection
   - Metadata updates

4. **File Reconstruction**:
   - Reassembly
   - Integrity checks

5. **Postprocessing and Validation**:
   - Error verification
   - User feedback

### 3.1. Data Preprocessing

The initial pipeline stage prepares EPUB files for further processing by parsing, identifying images, and analyzing errors. 

1. File parsing uses EbookLib or ZipFile to extract text, images, and metadata.
2. Image identification finds embedded images that need alt text descriptions.
3. Error analysis with Ace Checker detects missing alt text and other accessibility issues.

This sets a foundation for applying generative AI models in subsequent stages.

### 3.2. Generative AI Model Integration

**Pipeline Stages for Generating High-Quality Alt Text Descriptions**

**Image Feature Extraction:**
- Pre-trained computer vision models (CLIP, ViT) analyze visual content of images
- Extract high-dimensional features encapsulating semantic and contextual information

**Contextual Analysis:**
- Incorporate surrounding textual content: captions, adjacent paragraphs
- Integrate image features with relevant EPUB file information
- Ensure generated alt text is not only visually descriptive but also contextually aligned with the document's narrative.

**Alt Text Generation:**
- Leverage transformer-based models (GPT) that have been fine-tuned on large paired image-text datasets
- Synthesize visual and textual inputs to generate human-like, contextually relevant alt text
- Address requirements of accuracy and usability in accessibility enhancement.

### 3.3. Metadata Enrichment

The third pipeline stage enhances EPUB metadata for accessibility and usability. It begins by detecting languages using ensemble models combining statistical, rule-based, and transformer-based approaches. This improves language tag accuracy in multilingual cases.

Next, structural metadata is updated to adhere to accessibility standards like WCAG and EPUB Accessibility 1.0. Key fields such as titles, authors, publication dates, and hierarchies are refined for better usability and processing by assistive technologies.

### 3.4. File Reconstruction

The fourth stage of the pipeline reconstructs EPUB files with integrated alt text and enhanced metadata to ensure accessibility compliance. Libraries like EbookLib efficiently reassemble files while preserving structural integrity. A validation check adheres to EPUB standards and WCAG guidelines, confirming correct implementation and functionality. The result is a compliant, accessible EPUB file ready for end-users.

### 3.5. Postprocessing and Validation

**Error Reduction Verification:**
* To assess effectiveness of alt text generation for EPUB files
* Use accessibility checks with tools like Ace Checker
* Compare AI-generated alt text to ground truth using:
  + Cosine similarity metric: A·B / AB (higher scores mean greater alignment)
  + BLEU score calculation:
    * Compute brevity penalty (BP)
    * Precision for n-grams (p_n)
    * Calculate BLEU score with given formula

**User Feedback Evaluation:**
* Conduct qualitative user studies with visually impaired participants
* Ask them to use assistive technologies and provide feedback on:
  + Relevance of generated alt text
  + Descriptiveness of generated alt text
  + Usability of the EPUB files
* Analyze responses for trends and potential areas for improvement.

## 4. Evaluation





The AltGen pipeline's experimental evaluation assesses its performance in generating alt text for EPUB images, improving accessibility, and scaling up.

### 4.1. Experimental Setup

The evaluation is conducted on 1000 diverse EPUB files sourced from OAPEN and Project Gutenberg. The AltGen pipeline, implemented in Python using EbookLib and PyTorch, is tested on a machine with an NVIDIA RTX 3090 GPU, 64 GB RAM, and Intel Xeon processor.

### 4.2. Evaluation Metrics

**Quantitative Metrics**

* Cosine Similarity: Measures AI-generated alt text similarity to human-written descriptions.
* BLEU Score: Evaluates linguistic fidelity and relevance of generated alt text.
* Error Reduction Rate: Quantifies decrease in accessibility errors post-pipeline application.
* Runtime Efficiency: Assesses average time taken to process each EPUB file.

**Qualitative Metrics**

* Descriptiveness and Relevance: Assessed through user studies with visually impaired participants.
* Overall Usability: Participants rate usability of repaired EPUB files on a 5-point scale.

### 4.3. Results and Analysis

**AltGen Pipeline: Quantitative Results**
- **Table 1**: Illustrates effectiveness of AltGen pipeline for EPUB image alt text generation
- **500 EPUB files evaluated**, diverse content and complexity
- **Cosine Similarity score of 0.93** demonstrates semantic and contextual accuracy
- **BLEU Score of 0.76** emphasizes linguistic fidelity and coherent descriptions
- **Error Reduction Rate of 97.5%** highlights resolution of missing or inadequate alt text
- **Runtime Efficiency of 14 seconds per file** underscores scalability for large-scale processing

**AltGen Pipeline: Qualitative Results**
- **Table 2**: Feedback from user studies with visually impaired participants
- **Average rating of 4.8 out of 5 for "Descriptiveness and Relevance"** highlights pipeline's ability to generate meaningful, contextually appropriate alt text
- Participants reported improved understanding of image content
- **Average rating of 4.7 out of 5 for "Overall Usability"** reflects high satisfaction with ease of navigating EPUB files enhanced by AltGen pipeline
- Users experienced smoother reading experience when using assistive technologies.

### 4.4. Comparative Analysis

**Comparison of AltGen with Baseline Methods**

**Superiority of AltGen**:
- Demonstrated in Table 3 (not provided)
- Outperformed both rule-based approach and machine learning model across all metrics: Cosine Similarity, BLEU Score, and user satisfaction

**Rule-Based Approach**:
- Straightforward method
- Limited ability to produce semantically accurate and linguistically coherent alt text
- Cosine Similarity: 0.65
- BLEU Score: 0.55
- User satisfaction: 3.2 out of 5
- Primary issues: generic and contextually inadequate descriptions

**Machine Learning Model**:
- Leveraged pre-trained models to generate more accurate descriptions
- Reliance on static features limited adaptability to diverse content types in EPUB files
- Cosine Similarity: 0.75
- BLEU Score: 0.68
- User satisfaction: 4.1 out of 5

**AltGen Pipeline**:
- Outperformed both baseline methods across all metrics
- Demonstrated ability to produce highly accurate and contextually relevant alt text
- Cosine Similarity: 0.93
- BLEU Score: 0.76
- User satisfaction: 4.8 out of 5
- Highlighted for its capacity to generate descriptive, context-sensitive alt text that significantly improved the reading experience of visually impaired users.

## 5. Summary

The AltGen pipeline automates alt text generation for EPUB files with remarkable efficiency and accuracy. Leveraging generative AI models and robust evaluation metrics, it produces high-quality descriptions, demonstrated by a Cosine Similarity of 0.93 and BLEU Score of 0.76. User feedback shows significant improvements in usability and comprehension among visually impaired participants, achieving satisfaction ratings of 4.8/5 for descriptiveness and 4.7/5 overall.

