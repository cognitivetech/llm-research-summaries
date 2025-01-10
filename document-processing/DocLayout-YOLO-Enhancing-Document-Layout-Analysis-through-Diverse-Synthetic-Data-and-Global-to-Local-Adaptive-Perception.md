# DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception

Zhiyuan Zhao, Hengrui Kang\* and Bin Wang, Conghui He (Shanghai Artificial Intelligence Laboratory) - Study referenced at "https://arxiv.org/html/2410.12628v1"

\*Authors' affiliation not specified

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 Document Layout Analysis Approaches](#21-document-layout-analysis-approaches)
  - [2.2 Document Layout Analysis Datasets](#22-document-layout-analysis-datasets)
- [3 Diverse DocSynth-300K Dataset Construction](#3-diverse-docsynth-300k-dataset-construction)
  - [3.1 Preprocessing: Ensuring Element Diversity](#31-preprocessing-ensuring-element-diversity)
  - [3.2 Layout Generation: Ensuring Layout Diversity](#32-layout-generation-ensuring-layout-diversity)
- [4 Global-to-Local Model Architecture](#4-global-to-local-model-architecture)
  - [4.1 Controllable Receptive Module](#41-controllable-receptive-module)
  - [4.2 Global-to-Local Design](#42-global-to-local-design)
- [5 Experiments](#5-experiments)
  - [5.1 Experimental Metrics and Datasets](#51-experimental-metrics-and-datasets)
  - [5.2 Comparison DLA Methods \& Datasets](#52-comparison-dla-methods--datasets)
  - [5.3 Implementation Details](#53-implementation-details)
  - [5.4 Main Results](#54-main-results)

## Abstract

**Document Understanding Systems: Document Layout Analysis**
* Crucial for real-world document understanding systems
* Trade-off between speed and accuracy
	+ Multimodal methods (text & visual features): higher accuracy, significant latency
	+ Unimodal methods (visual features only): faster processing speeds, lower accuracy

**Introducing DocLayout-YOLO**
* Novel approach for enhancing accuracy while maintaining speed advantages
* Document-specific optimizations in pre-training and model design

**Robust Document Pre-Training: Mesh-candidate BestFit Algorithm**
* Frames document synthesis as a 2D bin packing problem
* Generates large-scale, diverse **DocSynth-300K dataset** for robust pre-training
* Significantly improves fine-tuning performance across various document types

**Model Optimization: Global-to-Local Controllable Receptive Module**
* Better handling of multi-scale variations in document elements
* Proposed to enhance accuracy in DocLayout-YOLO models

**Document Structure Benchmark (DocStructBench)**
* Complex and challenging benchmark for validating performance across different document types

**Experimental Results:**
* Extensive experiments demonstrate that DocLayout-YOLO excels in both speed and accuracy
* Code, data, and models available at: [https://github.com/opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

## 1 Introduction

**Document Parsing and Document Layout Analysis (DLA)**

**Background:**
- Rapid advancement of large language models and retrieval-augmented generation (RAG) research
- Increasing demand for high-quality document content parsing
- Document Layout Analysis (DLA): precisely locates regions in a document (text, titles, tables, graphics, etc.)

**Approaches to Document Parsing:**
1. **Multimodal methods**: combine visual and textual information
   * Pretraining on document images using unified text-image encoders
   * Achieve higher accuracy but slower due to complexity
2. **Unimodal methods**: rely solely on visual features
   * Faster processing speeds but lack accuracy due to absence of specialized pretraining and model design for document data

**Introducing DocLayout-YOLO:**
- Leverages strengths of both multimodal and unimodal approaches
- Matches speed of unimodal method YOLOv10
- Surpasses existing methods in terms of accuracy on diverse evaluation datasets
   * DocumentSynth-300K: large-scale, diverse pretraining corpus synthesized by Mesh-candidate BestFit algorithm
   * GL-CRM: Global-to-Local Controllable Receptive Module to enhance detection performance

**Contributions:**
1. Proposed DocLayout-YOLO model for diverse layout analysis tasks
2. Introduced Mesh-candidate BestFit algorithm to synthesize diverse document layout data (DocSynth-300K)
3. Designed GL-CRM to enhance model's capability to detect elements of varying scales
4. Extensive experiments on DLA, DocLayNet, and in-house diverse evaluation datasets demonstrate state-of-the-art mAP scores and inference speed for real-time layout analysis on diverse documents.

## 2 Related Work

### 2.1 Document Layout Analysis Approaches

Document Layout Analysis (DLA) involves identifying and locating various components within documents such as text and images. DLA approaches are categorized into unimodal and multimodal methods:
- Unimodal methods approach DLA as a special object detection problem, utilizing generic off-the-shelf detectors [^27].
- Multimodal methods enhance DLA by aligning text-visual features through pre-training, for instance, LayoutLM [^33] offers a unified method with multiple pre-training goals and impressive results on various document tasks. DiT [^21] improves performance via self-supervised pre-training on extensive document datasets. VGT [^6] introduces grid-based textual encoding for extracting text features.

### 2.2 Document Layout Analysis Datasets

**Document Layout Analysis Datasets:**
* IIT-CDIP: 42 million low-resolution, unannotated images; RVL-CDIP subset: 16 classes for 400,000 images [^18, 10]
* PubLayNet: 360,000 pages from PubMed journals [^37]
* DocBank: 500,000 arXiv pages with weak supervision [^23]
* DocLayNet: 80,863 pages from diverse document types [^25]
* D<sup class="ltx_sup" id="S2.SS2.p1.1.1">4</sup>LA: 11,092 images from RVL-CDIP across 27 categories [^6]
* M<sup class="ltx_sup" id="S2.SS2.p1.1.2">6</sup>Doc: diverse collection of 9,080 images with 74 types but not open source due to copyright restrictions [^5]
* DEES200, CHN, Prima-LAD, and ADOPD: not open-sourced or primarily suitable for fine-tuning.

**Document Generation Methods:**
* Most approaches focus on academic papers.

**Limitations of Current Datasets:**
* Significant limitations in diversity, volume, and annotation granularity.
* Lead to suboptimal pre-training models.

## 3 Diverse DocSynth-300K Dataset Construction

**Pre-Training Datasets Limitations**
* Existing datasets have significant homogeneity, mainly academic papers
* Hinders generalization capabilities of pre-trained models
* Need to develop more varied pre-training document dataset for better adaptation

**Diversity in Pre-Training Data**
* **Element diversity**: Variety of document elements (text, tables, etc.)
* **Layout diversity**: Different document layouts (single-column, double-column, multi-column)

**Proposed Methodology: Mesh-candidate BestFit**
* Automatically synthesizes diverse and well-organized documents
* Leverages both element and layout diversity
* Enhances model performance across various real-world document types

**Dataset: DocSynth-300K**
* Resulting dataset from Mesh-candidate BestFit methodology
* Significantly enhances model performance

**Pipeline of Mesh-candidate BestFit**
* Not detailed in provided text

**Document Layout Analysis Datasets**
* Figure 2 (not linked) illustrates document layout analysis datasets
* Enhancing document layout analysis through diverse synthetic data and adaptive perception.

### 3.1 Preprocessing: Ensuring Element Diversity

**Preprocessing Phase:**
- Utilize M<sup>6</sup>Doc test [^5] for diverse document elements: 74 unique elements from around 2800 pages.
- Fragment and categorize the pages to create an element pool per category.
- Maintain diversity within each category through an augmentation pipeline that enlarges data pools of rare categories with less than 100 elements ([A.2.2 Data Augmentation Pipeline](https://arxiv.org/html/2410.12628v1#A1.SS2.SSS2)).

### 3.2 Layout Generation: Ensuring Layout Diversity

**Layout Generation Approach: Bin-packing Inspired Method**

**Random Arrangement Limitations**:
- Disorganized and confusing layouts
- Severely hampers improvement on real-world documents

**Diffusion and GAN Models**:
- Limited to producing homogeneous layouts (e.g., academic papers)
- Insufficient for various real-world document layouts

**Layout Diversity and Consistency**:
- Inspired by 2D bin-packing problem
- Regard available grids as "bins" of different sizes
- Iteratively perform best matching to generate diverse, reasonable document layouts
- Balance layout diversity (randomness) and aesthetics (fill rate, alignment)

**Layout Generation Steps**:
1. **Candidate Sampling**: Obtain subset from element pool based on size for each blank page
2. **Meshgrid Construction**: Construct meshgrid based on layout and filter invalid grids
3. **BestFit Pair Search**: Traverse all valid grids and search for optimal Mesh-candidate pair with maximum fill rate
4. **Iterative Layout Filling**: Repeat steps 2-3 until no valid Mesh-candidates meet size requirement
5. **Random Central Scaling**: Apply to all filled elements

**Generated Document Examples**:
- Comprehensive layout diversity (multiple formats) and element diversity
- Well-organized, visually appealing documents

**Benefits of Generated Documents**:
- Adapt to various real-world document types
- Adhere to human design principles (alignment, density)

**Algorithm of Layout Generation**:
(Details in Algorithm 1 in the paper)

## 4 Global-to-Local Model Architecture

**Caption**: Figure 4: Illustration of Controllable Receptive Module (CRM) extracting and fusing features of varying scales and granularities in document images. To tackle the scale-varying challenge posed by different elements such as title lines and tables, we propose a hierarchical architecture called GL-CRM. This architecture consists of two components: CRM for flexible extraction and integration of multiple scales and granularities, and Global-to-Local Design (GL) featuring a hierarchical perception process from global context to local semantics information.

### 4.1 Controllable Receptive Module

**CRM Architecture (Figure 5)**
- Illustration of Global-to-local design: FIGURE 5 in the text
- Extracting features of different granularities using a weight-shared convolution layer with varying dilation rates (d)
  - Kernel size: k
  - Dilation rates: d=[d_1,d_2,...,d_n]
  - Output: F=[F_1,F_2,…,F_n]
- Integrating and learning to fuse different feature components autonomously:
  - Concatenate features (F̂)
  - Use a lightweight convolutional layer M with kernel size 1 and groups nC for mask extraction
    * Importance weights for different features
  - Apply mask M to the fused features F̂
  - Use a lightweight output projector Conv_out
- Shortcut connection merges integrated feature with initial feature X: X_CRM=X+GELU(BN(Conv_out(M⊗F̂))

**CRM Integration into Conventional CSP Bottleneck (Figure 4)**
- CRM is integrated into the conventional CSP bottleneck: FIGURE 4 in the text
- Controlled by two parameters: k and d
  - Granularity of extracted features
  - Scale of extracted features.

### 4.2 Global-to-Local Design

**CRM Usage for Different Stages:**

- Global stage (rich texture details): CRM with enlarged kernels (k=5) and dilation rates (d=1,2,3). Captures more detail, preserves local patterns for whole-page elements.
- Intermediate stage (downsampled feature map): CRM with smaller kernel (k=3), sufficient dilation rates (d=1,2,3) for medium-scale elements like document sub-blocks.
- Deep stage (semantic information): Basic bottleneck as a lightweight module focusing on local semantic information.

**Figure 6: Examples of Complex Documents:**
Refer to the figure [here](https://arxiv.org/html/2410.12628v1/x7.png) showcasing various document formats and structures in DocStructBench.

## 5 Experiments

### 5.1 Experimental Metrics and Datasets

**Evaluation Metrics:**
- **COCO-style mAP**: metric for accuracy
- **FPS (processed images per second)**: metric for speed

**Evaluation Datasets:**
- **DLA**: 11,092 noisy images in 27 categories from IIT-CDIP across 12 document types
  - Training set: 8,868 images
  - Testing set: 2,224 images
- **DocLayNet**
  - Contains 80,863 pages from 7 document types, manually annotated with 11 categories
  - Split into training/validation/testing sets: 69,103/6,480/4,994 images, respectively
  - Validation set used for evaluation

**DocStructBench:**
- Comprehensive dataset designed for evaluation across various real-world scenario documents
- Consists of 4 subsets: Academic, Textbooks, Market Analysis, Financial
- Data sources diverse, encompassing a broad range of domains from institutions, publishers, and websites
- Total images: 7,310 training + 2,645 testing
- Each image manually annotated across 10 distinct categories: Title, Plain Text, Abandoned Text, Figure, Figure Caption, Table, Table Caption, Table Footnote, Isolated Formula, Formula Caption
- Training on a mixture of all subsets and reporting results separately for each subset.

### 5.2 Comparison DLA Methods & Datasets

**Comparison of DocLayout-YOLO**

- Compared to multimodal methods: LayoutLMv3, DiT-Cascade, VGT
- Unimodal comparison with DINO-4scale-R50 as robust object detector
- For pre-training datasets, we evaluate against DocSynth-300K and public datasets PubLayNet and DocBank

### 5.3 Implementation Details

**Pre-training and Fine-Tuning Strategies for DocLayout-YOLO**

**Pre-training**:
- On DocSynth-300K
- Image longer side resized at 1600
- Batch size of 128
- Learning rate of 0.02 for 30 epochs

**Fine-tuning on D4LA**:
- Longer side resized to 1600
- Learning rate set to 0.04

**Fine-tuning on DocLayNet**:
- Longer side resized to 1120
- Learning rate set to 0.02

**Fine-tuning on DLA**:
- Longer side set to 1600
- Learning rate set to 0.04

**Comparison Models**:
- **DINO**: Pretrained on ImageNet1K, multi-scale training with image longer side of 1280 and AdamW optimizer at 1.0x10^-4
- **LayoutLMv3 and DiT**: Detectron2 Cascade R-CNN training with image longer side of 1333, SGD optimizer of 2.0x10^-4 for 60 k iterations

**Performance Comparison**:
- Table 1: Results of DocLayout-YOLO with different optimization strategies
- Table 2: Performance comparison on D4LA and DocLayNet (v10m++ denotes the original v10m bottleneck enhanced by GL-CRM)
- Table 3: Performance comparison on DocStructBench (v10m++ denotes the original v10m bottleneck enhanced by GL-CRM), FPS tested under different frameworks (Detectron2, MMDetection, and Ultralytics).

### 5.4 Main Results

**Effectiveness of Proposed Optimization Strategies**

**DocSynth-300K**:
- Enhances performance across various document types
- Pre-trained model achieves:
    - 1.2 and 2.6 improvement on D4LA and DocLayNet, respectively
    - Improvement on four subsets of DocStructBench

**DocLayout-YOLO**:
- Achieves significant improvement by combining CRM and DocSynth-300K pre-training
- Results in:
    - 1.7/2.6/1.3/3.5/0.5/0.3 improvements compared to baseline YOLO-v10 model

**Downstream fine-tuning performance**:
- DocSynth-300K pre-trained models show better adaptability across all document types compared to public and synthetic datasets
- Results are summarized in Table 4

**Comparison with Current DLA Methods**:
- **DocLayout-YOLO** outperforms:
    - Robust unimodal DLA methods (e.g., DINO)
        - Improvement of 2.0 on DocLayNet
    - SOTA multimodal methods (e.g., VGT)
        - Achieves 70.3 mAP on D4LA, surpassing second-best VGT's 68.8
- **DocLayout-YOLO** achieves superior performance in three out of four subsets of DocStructBench, surpassing existing SOTA unimodal (DINO) and multimodal (DIT-Cascade-L) approaches
- **DocLayout-YOLO** is significantly more efficient than current DLA methods:
    - 14.3x faster FPS compared to best multimodal method DIT-Cascade-L
    - 3.2x faster FPS compared to best unimodal method DINO

