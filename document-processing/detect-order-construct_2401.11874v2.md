# Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure

by Jiawei Wang, Kai Hu, Zhuoyao Zhong, Lei Sun, Qiang Huo 
https://arxiv.org/html/2401.11874v2

## Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
  - [2.1. Page Object Detection](#21-page-object-detection)
  - [2.2. Reading Order Prediction](#22-reading-order-prediction)
  - [2.3. Hierarchical Document Structure Reconstruction](#23-hierarchical-document-structure-reconstruction)
- [3.Problem](#3problem)
- [4. Methodology](#4-methodology)
  - [4.1. Overview](#41-overview)
  - [4.2. Detect Module](#42-detect-module)
  - [4.3. Order Module](#43-order-module)
- [5. Experiments](#5-experiments)
  - [5.1. Datasets and Evaluation Protocols](#51-datasets-and-evaluation-protocols)
  - [5.2. Implementation Details](#52-implementation-details)
  - [5.3. Comparisons with Prior Arts](#53-comparisons-with-prior-arts)
  - [5.4. Ablation Studies](#54-ablation-studies)
  - [5.5. Limitations of Our Approach](#55-limitations-of-our-approach)
- [6. Conclusion and Future Work](#6-conclusion-and-future-work)

## Abstract
**Document Structure Analysis: Hierarchical Document Structure (HDSA)**
- Importance of document structure analysis for understanding physical layout and logical structure of documents
- Application areas: information retrieval, document summarization, knowledge extraction
- Focus on HDSA to explore hierarchical relationships in structured documents using authoring software with hierarchical schemas
  - LaTeX, Microsoft Word, HTML

**Proposed Approach for HDSA:** tree construction based approach
- Addresses multiple subtasks concurrently: page object detection (Detect), reading order prediction of identified objects (Order), and construction of intended hierarchical structure (Construct)
- Demonstration of performance through an effective end-to-end solution

**Assessment:**
- Development of a comprehensive benchmark called Comp-HRDoc for evaluating the subtasks simultaneously
- Achievement of state-of-the-art performance on various datasets and benchmarks, including PubLayNet, DocLayNet, HRDoc, and Comp-HRDoc.

**Availability:** The Comp-HRDoc benchmark is publicly available at https://github.com/microsoft/CompHRDoc .

## 1. Introduction

**Document Structure Analysis (DSA)**
- Identifies fundamental components within a document: headings, paragraphs, lists, tables, figures
- Establishes logical relationships and structures of these components
- Results in structured representation of the document's physical layout and logical structure
- Enhances effectiveness and accessibility of information retrieval and processing

**Hierarchical Document Structure Analysis (HDSA)**
- Extracts and reconstructs inherent hierarchical structures within document layouts
- Gained significant attention due to the complexity of modern digital documents

**Previous Research in DSA**
- **Early research**: focused on physical layout analysis, logical structure analysis using knowledge-based [1], rule-based [2], model-based [3], grammar-based [4] methods
- Limitations: susceptibility to noise, ambiguity, difficulty handling complex document collections, absence of quantitative performance evaluations

**Deep Learning Approaches in DSA**
- Notable improvements in performance and robustness
- Primarily focus on specific sub-tasks: Page Object Detection, Reading Order Prediction, Table of Contents (TOC) Extraction
- Gap in research community for a comprehensive end-to-end system or benchmark addressing all aspects of document structure analysis concurrently.

**Hierarchical Document Structure Analysis**

**Recent Approaches**:
- **DocParser**: End-to-end system for parsing document renderings into hierarchical structures
  - Uses Mask R-CNN to detect document entities and predict relationships (parent of, followed by)
  - Limited by reliance on rule-based approach
- **HRDoc**: Encoder-decoder based DSPS to reconstruct the logical structure of documents
  - Employs multi-modal bidirectional encoder and structure-aware GRU decoder
  - Achieves significant performance improvements but requires provided reading order
  - Quadratic complexity with increasing text-lines, presenting challenges for longer documents

**Proposed Approach**:
- **Detect**: Identify document objects and assign logical roles
- **Order**: Establish reading order relationships between nodes
- **Construct**: Identify hierarchical relationships (e.g., Table of Contents) to build abstract tree structure

**End-to-End Solution**:
- **Hybrid method for Detect stage**: Combines top-down model with relation prediction model
  - Detects graphical objects, groups text-lines into regions, recognizes roles
- **Multi-modal transformer-based relation prediction models**: Tackle all three stages (reading order, TOC, structure)
  - Dependency parsing approach using a global self-attention mechanism
  - Structure-aware models for chain and tree structures

**Evaluation**:
- Comprehensive benchmark, Comp-HRDoc, to evaluate page object detection, reading order prediction, TOC extraction, and hierarchical structure reconstruction

## 2. Related Work

**Related Work on Document Structure Analysis**
* Numerous studies conducted since the 1980s on document structure analysis: physical layout analysis (PLA) and logical structure analysis (LSA)
* PLA focuses on identifying homogeneous regions of interest, or page objects; LSA assigns logical roles to these regions and determines relationships between them
* Early approaches based on heuristic rules or grammar analysis [11, 12]
* Recent research has focused on document layout analysis: PLA and logical role classification (page object detection) [8, 9, 13, 14]

**Page Object Detection**
* Task that incorporates both physical layout analysis and logical role classification in document structure analysis
* Main focus of recent research investigations

**Physical Layout Analysis (PLA)**
* Identifying homogeneous regions of interest or page objects within a document

**Logical Role Classification**
* Assigning logical roles to identified page objects
* Determining relationships between components within documents, such as reading order and organization of tables of contents. 

**Latest Advancements in Document Structure Analysis:**
- Page object detection: recent focus of research investigations
- Physical layout analysis (PLA)
- Logical role classification
- Reading order prediction
- Hierarchical document structure reconstruction.

### 2.1. Page Object Detection

**Page Object Detection (POD)**

**Approaches:**
- **Object detection-based methods:**
  - Leverage latest object detection frameworks for POD
  - Examples: R-CNN [18], Fast R-CNN [19], Mask R-CNN [6], YOLOv5 [24]
  - Enhance performance with multi-modal models, GNNs [32], and robust backbones [30, 34]
- **Semantic segmentation based methods:**
  - Use existing semantic segmentation frameworks for initial pixel-level segmentation
  - Merge pixels to form distinct page objects
  - Examples: FCN [41], Multi-modal FCN [13], Multi-scale, multi-task FCN [37]
  - Improve performance with label pyramids and deep watershed transformation [39]
- **Graph-based methods:**
  - Represent document pages as graphs, nodes correspond to primitive page objects, edges denote relationships between neighbors
  - Formulate POD as a graph labeling problem
  - Examples: CRF models [42], GAT [43], Multi-aspect GCNs [44]

**Challenges:**
- Detection of small-scale text regions remains a challenge for all approaches.

### 2.2. Reading Order Prediction

**Reading Order Prediction**

**Objective**: Determine appropriate reading sequence for documents
- Humans read left-to-right, top-to-bottom
- Complex documents may require more sophisticated methods

**Approaches to Reading Order Prediction**
1. **Rule-based sorting**
   * Topological sorting: based on x/y interval overlaps between text lines
      - Enables generation of reading order patterns for multi-column layouts [50]
   * Bidimensional relation rule: similar to topological rules, but also row-wise [51]
   * Argumentation-based approach: uses relationships between text blocks [52]
   * XY-Cut: efficient method for larger sizes and hierarchies [53, 54]
   * Effective in certain scenarios, but prone to failure with out-of-domain cases
2. **Machine learning-based sequence prediction**
   * Previous research has attempted to learn reading order patterns using various methods:
      - Probabilistic classifier within Bayesian framework [55]
      - Inductive logic programming (ILP) learning algorithm [56]
      - Deep learning models, such as graph convolutional encoder and pointer network decoder [57], or transformer-based architecture on spatial-text features [58]
   * Decoding speed of auto-regressive methods limited when applied to rich text documents
   * Quir'os et al. proposed new reading order decoding algorithms for handwritten documents, assuming a pairwise partial order at the element level [59]. However, this approach determines pair-wise spatial features without considering visual or textual information.

### 2.3. Hierarchical Document Structure Reconstruction

**Hierarchical Document Structure Reconstruction**

**Table of Contents (TOC) Extraction:**
- **Importance**: Recover document's logical structure and semantic information beyond character strings
- **Challenges**: Diversity of TOC styles and layouts
- **Approaches:**
  * Heuristic rules based on small datasets: Not effective in large-scale heterogeneous documents [60]
  * Adaptive selection of rules based on basic TOC styles [61]
    - Assumes existence of a Table of Contents page within the documents
  * Combination of TOC page detection and link-based reconstruction methods [62]
  * Hierarchy Extraction from Long Document (HELD) framework: Sequentially inserts each section heading into the TOC tree at the correct position using LSTM [63]
  * End-to-end model with multimodal tree decoder (MTD) for table of contents extraction [64]

**Overall Structure Reconstruction:**
- **Goal**: Represent document layout and logical structure in a hierarchical manner
- **Graph representation**: General but fails to capture hierarchical nature [65]
- **Formal grammars**: Useful for describing document structures, but multiple interpretations possible [66]
  * Stochastic grammars: Integrate multiple evidences and estimate most probable parse or interpretation [67]
    - May lack flexibility to model complex patterns and structures
- **Deep learning based methods:**
  * Form understanding task: Predict relationship between each pair of text fragments using an asymmetric parameter matrix [68]
    - High computational complexity in documents with a large number of text fragments
  * DocParser: Parse complete physical structure of documents but did not consider logical hierarchical structure [5]
- **HRDoc dataset and encoder-decoder-based hierarchical document structure parsing system (DSPS) proposed:**
  * Reconstructs hierarchical structure while taking into account the logical structure of a document [7]
    - Presumes that the reading order is provided
    - Directly predicts relationships between text-lines: Low representational ability and high computational cost.

## 3.Problem

**Hierarchical Document Structure Analysis**

**Objectives**:
- Reconstruct hierarchical structure tree H of a multi-page document D
- Consists of both page objects and hierarchical relationships

**Page Objects (Oi)**:
- Represent various page objects within document D
- Attributes: logical role category ci, bounding box coordinates bi, basic semantic units

**Hierarchical Relationships (Rij)**:
- Describe relationships between page object pairs Oi and Oj
- Represented by triplets (Oi, rij, Oj): subject page object, relation type, object page object
- Three types of relationships:
  1. Text Region Reading Order Relationship: main body text regions
  2. Graphical Region Relationship: captions, footnotes, tables or figures
  3. Table of Contents Relationship: section headings

**Hierarchy Tree H**:
- Consists of page objects and hierarchical relationships
- Can be used to extract various hierarchical relationships as needed
- Reading order sequence can be obtained by pre-order traversal on H

**Sub-Tasks**:
1. **Page Object Detection (Detect stage)**:
   - Identify individual page objects Oi within each page of document D
   - Assign a logical role to each detected page object (e.g., section headings, captions, footnotes)
2. **Reading Order Prediction (Order stage)**:
   - Determine the reading sequence of detected page objects based on their spatial arrangement within document D
   - Reading order represented as a permutation of indices of detected page objects
3. **Table of Contents Extraction (Construct stage)**:
   - Construct hierarchy tree that summarizes overall hierarchical structure H
   - Comprises list of section headings and their hierarchical levels

**Integration**:
- Integrating results from all three sub-tasks allows for effective reconstruction of the hierarchical document structure tree H.

## 4. Methodology
### 4.1. Overview

**Detect-Order-Construct Approach for Hierarchical Document Structure Analysis**

**Components:**
- **Detect stage**: Identifies individual page objects within the document rendering and assigns a logical role (page object detection)
- **Order stage**: Determines sequential order of page objects (reading order prediction)
- **Construct stage**: Extracts abstract hierarchy tree (table of contents extraction)

**Approach Overview:**
1. Detect: Identifies and roles page objects
2. Order: Predicts reading order
3. Construct: Extracts hierarchical document structure tree
4. Integrates outputs for complete reconstruction
5. Defines tasks as relation prediction problems
6. Uses multi-modal, transformer-based models
7. Relation prediction model approaches dependency parsing
8. Aligns with chain structure of reading order and tree structure of table of contents
9. Developed modules: Detect module, Order module, Construct module (elaborated in sections 4.2, 4.3, 4.4)

**Detect Module:**
- Uses CNN backbone network for detection
- Includes DINO based graphic page object detector
- Processes text lines and their contents
- Figure 3 illustrates the overall architecture

### 4.2. Detect Module

**Detect Module Components:**
- Shared visual backbone network: extracts multi-scale feature maps from input document images (ResNet-50 used in conference paper [10])
- Top-down graphical page object detection model: detects graphical objects like tables, figures, formulas (DINO [69] used)
- Bottom-up text region detection model: groups text-lines outside graphical objects based on reading order and identifies logical roles.

**Bottom-Up Text Region Detection Model:**
1. **Objective**: Group text-lines into distinct regions based on intra-region reading order, recognize their logical role.
2. **Components**: Multi-modal feature extraction module, multi-modal feature enhancement module, intra-region reading order relation prediction head, and logical role classification head (see Fig. 4 for schematic view).

**Multi-Modal Feature Extraction Module:**
1. **Visual Embedding**: Resize C4 and C5 to size of C3, concatenate along channel axis, feed into a convolutional layer to generate feature map Cfuse; extract feature maps using RoIAlign algorithm based on bounding box bti and obtain visual embedding Vti (Eq. 1).
2. **Text Embedding**: Serialize text-lines, tokenize, feed into pre-trained language model BERT [71], average embeddings of all tokens to obtain text embedding Tti, then apply a fully-connected layer with 1,024 nodes (Eq. 2).
3. **2D Positional Embedding**: Encode bounding box and size information as 2D Positional Embedding Bti using Layer Normalization and Multi-Layer Perceptron (MLP) (Eq. 3).
4. **Multi-Modal Representation**: Concatenate visual embedding Vti, text embeddings Tti, and 2D Positional Embedding Btito obtain multi-modal representation Uti (Eq. 4).

**Multi-Modal Feature Enhancement Module:**
1. **Transformer Encoder**: Model interactions between text-lines using self-attention mechanism; treat each text-line as a token and use its multi-modal representation Ut as input to the encoder (Eq. 5). Here, only a 1-layer Transformer encoder is used for saving computation.

#### 4.2.3. Intra-region Reading Order Relation Prediction Head

**Intra-region Reading Order Relation Prediction Head**

**Purpose**:
- Use a relation prediction head to predict intra-region reading order relationships between text-lines
- If text-line ti is the succeeding text-line of tj in the same text region, there exists an intra-region reading order relationship (ti→tj)
- Treat relation prediction as a dependency parsing task using softmax cross-entropy loss instead of binary classification
- Adopt spatial compatibility feature from [73] to model spatial interactions between text-lines for relation prediction

**Relation Prediction**:
- Calculate score sij to estimate how likely tj is the succeeding text-line of ti:
  - fij = FCq(Fti)◦FCk(Ftj) + MLP (rbti, btj)
  - sij = exp(fij)/Nexp(fij)
- Use a multi-class classifier to predict logical role label for each text-line and determine text region by plurality voting

**Relation Prediction Heads**:
- Employ an additional relation prediction head to identify the preceding text-line for each text-line
- Combine results from both relation prediction heads to obtain final results

**Text Region Grouping**:
- Use a Union-Find algorithm to group text-lines into text regions based on predicted intra-region reading order relationships
- Text region bounding box is the union bounding box of all its constituent text-lines

### 4.3. Order Module

**Order Module**

**Focus:** Determining reading sequence of graphical page objects and text regions in document D using multi-modal relation prediction model.

**Components:**
1. **Multi-modal Feature Extraction Module:** Fuses visual embedding and 2D positional embedding to obtain a multi-modal representation for each graphical object.
   - Uses Eqs. (1) and (3) from Section 4.2.1 to fuse embeddings
   - For text regions: proposes attention fusion model integrating features of text lines
2. **Multi-modal Feature Enhancement Module:** Improves multi-modal representations using a three-layer Transformer encoder and self-attention mechanism.
3. **Inter-region Reading Order Relationships:** Two categories: text region reading order relationships, graphical region reading order relationships between captions/footnotes and objects.
4. **Order Module Illustration:** Fig. 6 in the provided text.
5. **Multi-modal Feature Extraction for Text Regions:** Eqs. (10)-(12), attention fusion model integrating features of text lines, forming a multi-modal representation UOn.
6. **Region Type Embeddings:** Derived using Eq. (13) based on page object's logical role rOi.
7. **Final Representation:** Concatenate UOi and ROi to obtain ˆUOi (Eq. 14).
8. **Multi-modal Feature Enhancement for Page Objects:** Transformer Encoder processes multi-modal representations ˆUOi as tokens, FO=TransformerEncoder( ˆUO) yields improved embeddings.

#### 4.3.3. Inter-region Reading Order Relation Prediction Head

**Inter-region Reading Order Relation Prediction Head (Order Module)**
* Similar structure to inter-region reading order task of Detect module
* Multi-class classifier for relation type determination between page objects Oi and Oj
* Uses bilinear classifier and argmax function
* Input: fully-connected layers FCq(FOi) and FCk(FOj)
* Output: predicted relation type cij

**Inter-region Reading Order Relation Classification Head (Order Module)**
* Multi-class classifier for probability distribution across classes
* Determines relation type between page objects Oi and Oj
* Calculations: BiLinear(FCq(FOi), FCk(FOj)) and argmax(pij)
* Input: fully-connected layers FCq and FCk, output: predicted relation type cij

**Construct Module Goals**
* Generate tree structure for hierarchical table of contents based on detected section headings
* Extract multi-modal representation FS of each section heading from page objects' multi-modal representation Fo
* Input enhanced representations US of all section headings into transformer encoder with correct reading order
* Add positional encoding to convey reading order information
* Use tree-aware TOC relation prediction head for parent and sibling relationships

**TOC Relation Prediction Head (Construct Module)**
* Two types: parent-child and sibling relationships
* Parent-child relationship: when sec i is a parent node of seci in the TOC tree structure
* Sibling relationship: when sec j acts as left sibling of sec i in the TOC tree
* Multi-class classifier with same network structure for both types of relationships
* Calculations: fij and sp ij or ss ij using FCq and FCk layers, dot product operation ◦, and exp function.
* Parent and sibling relationships defined as dependencies between nodes.
* Softmax cross-entropy loss during training phase and serial decoding with tree structure constraint during testing phase.
* Tree insertion algorithm used to generate complete table of contents tree.

## 5. Experiments
### 5.1. Datasets and Evaluation Protocols

**Experiments and Evaluation Protocols**

**Proposed Approach Validation**:
- Experiments on two well-recognized large-scale document layout analysis benchmarks: PubLayNet [8] and DocLayNet [9]
- Demonstrated effectiveness of proposed Detect module

**HRDoc Benchmark and Subtasks**:
- High-quality public hierarchical document structure reconstruction benchmark
- Provides annotations and benchmarks for logical role classification and overall hierarchical structure reconstruction
- Importance of assessing each subtask involved in hierarchical document analysis:
  - **Page object detection**
  - Reading order prediction
  - Table of contents extraction
  - Hierarchical document structure reconstruction
- **Comp-HRDoc**: Comprehensive benchmark for hierarchical document structure analysis
- Replaced text-line-level logical role classification in HRDoc with more popular and significant subtask, page object detection

**PubLayNet and DocLayNet Datasets**:
- PubLayNet: Large-scale dataset from IBM, 5 types of page objects, COCO-style mean average precision (mAP) evaluation metric
- DocLayNet: Human-annotated dataset from IBM, 11 types of page objects, COCO-style mean average precision (mAP) evaluation metric

**HRDoc Dataset**:
- Human-annotated dataset for hierarchical document structure reconstruction
- Line-level annotations and cross-page relations
- Two parts: HRDoc-Simple (1,000 documents) and HRDoc-Hard (1,500 documents)
- Evaluation tasks: Semantic unit classification (logical role classification), hierarchical structure reconstruction
  - F1 score for each logical role as evaluation metric
  - Semantic-TEDS as evaluation metric for hierarchical structure reconstruction

**Comp-HRDoc Benchmark**:
- Comprehensive benchmark for hierarchical document structure analysis
- Encompasses tasks: page object detection, reading order prediction, table of contents extraction, and hierarchical structure reconstruction
- Built upon HRDoc-Hard dataset
- **Page object detection**: COCO-style segmentation-based mean average precision (mAP) evaluation metric
- **Reading order prediction**: Proposed reading edit distance score (REDS) to evaluate reading order groups independently
- Evaluation metrics for table of contents extraction and hierarchical structure reconstruction remain Semantic-TEDS.

### 5.2. Implementation Details

**Implementation Details**
- Approach implemented using **PyTorch v1.0** on a workstation with 8 Nvidia Tesla V100 GPUs (32 GB memory)
- In PubLayNet, lists are treated as entire objects containing multiple inconsistent labeled items
- Trained Detect stage of framework on **PubLayNet** and **DocLayNet** datasets for page object detection
  * Used three multi-scale feature maps {C3, C4, C5} from backbone network and DINO-based graphical page object detection model
  * Backbone network parameters initialized with a pretrained ResNet-50 model on ImageNet
  * Text embedding extractor parameters initialized with the pretrained BERT BASE model
  * Optimized using **AdamW** algorithm, batch size of 16, and trained for 12 epochs on PubLayNet and 24 epochs on DocLayNet
- For HRDoc and Comp-HRDoc:
  * Utilized four multi-scale feature maps {C2, C3, C4, C5} from backbone network
  * Used Mask2Former-based graphical page object detection model for identifying graphical objects
  * Reduced GPU memory requirements by choosing **ResNet-18** as CNN backbone network
  * Text embedding extractor parameters initialized with the pretrained BERT BASE model
  * Optimized using AdamW algorithm, batch size of 1, and trained for 20 epochs on HRDoc and Comp-HRDoc
  * Learning rate linearly decreases from initial learning rate set in optimizer to 0 after a warmup period (set to 2 epochs)

### 5.3. Comparisons with Prior Arts

**Comparison of Detect Module to Prior Arts**
* Effectiveness validated on DocLayNet and PubLayNet datasets
* Outperforms Mask R-CNN, Faster R-CNN, YOLOv5, DINO on DocLayNet
	+ Superior performance in document layout analysis
* Outperforms several state-of-the-art methods on PubLayNet
	+ Regardless of textual features used in bottom-up detection model
* Validated with HRDoc and Comp-HRDoc datasets
	+ Superior performance in semantic unit classification, hierarchical structure reconstruction
	+ Handles all tasks concurrently with significantly superior results.

**Performance Comparisons on DocLayNet (Table 1)**
- Mask R-CNN, Faster R-CNN, and YOLOv5 mAP percentages obtained from [9]
- Human Mask R-CNN: 84-89
- Faster R-CNN: 71.5
- YOLOv5: 76.8
- DINO: 83.2
- Our Approach: 81.0
	+ Significant improvement in mAP over YOLOv5
	+ Superior performance on challenging dataset.

**Performance Comparisons on PubLayNet (Table 2 and Table 3)**
- Vision-based and multimodal methods results obtained from various sources
- Faster R-CNN: 91.0 mAP for Text, 82.6 for List, 88.3 for Figure, 95.4 for Title
- Mask R-CNN: 91.6 mAP for Text, 84.0 for List, 88.6 for Table, 96.0 for Figure, 94.9 for Title
- Naik et al.: 94.3 mAP for Text, 88.7 for List, 94.3 for Table, 97.6 for Figure, 96.1 for Title
- Minouei et al.: 94.4 mAP for Text, 90.8 for List, 94.0 for Table, 97.4 for Figure, 96.6 for Title
- DiT-L: 94.4 mAP for Text, 89.3 for List, 96.0 for Table, 97.8 for Figure, 97.2 for Title
- SRRV: 95.8 mAP for Text, 90.1 for List, 95.0 for Table, 97.6 for Figure, 96.7 for Title
- DINO: 94.9 mAP for Text, 91.4 for List, 96.0 for Table, 98.0 for Figure, 97.3 for Title
- TRDLU: 95.8 mAP for Text, 92.1 for List, 97.6 for Table, 97.6 for Figure, 96.6 for Title
- UDoc: Vision+Text: 93.9 mAP, 88.5 for List, 93.7 for Table, 97.3 for Figure, 96.4 for Title
* Our Approach: 97.0 mAP for Text, 92.8 for List, 96.4 for Table, 98.1 for Figure, 97.4 for Title
	+ Superior performance in text and vision-based document layout analysis.

**Performance Comparisons on HRDoc (Table 4 and Table 5)**
- Semantic unit classification task: Our Proposed Method surpasses previous methods in the majority of categories, particularly Fstl(Firstline) and Footn (Footnote) classes.
	+ DSPS Encoder performance notably inferior to Sentence-BERT in Mail category but outperforms our method on HRDoc-Hard by nearly 5 percent.
* Hierarchical structure reconstruction: Our Proposed Tree Construction-based Method markedly outperforms the DSPS Encoder.
	+ On HRDoc-Hard, we exceed its performance by 16.63 percent and 15.77 percent in Micro-STEDS and Macro-STEDS, respectively.
	+ Similarly, on HRDoc-Simple, we surpass the DSPS Encoder by 13.61 percent and 13.36 percent in Micro-STEDS and Macro-STEDS, respectively.
* Our method evaluates performance based on predicted reading order sequence while DSPS Encoder takes advantage of ground-truth reading order.

**Performance Comparisons on Comp-HRDoc (Table 6)**
- Page Object Detection: Surpasses Mask2former by 14.52 percent in terms of segmentation-based mAP.
- Reading Order Prediction: Enhanced partial algorithm decodes both categories of reading order groups simultaneously and outperforms Lorenzo et al.'s approach by 15.78 percent in terms of previously defined REDS (text region reading order group).
- Table Of Contents Extraction: Exceeds Multimodal Tree Decoder (MTD) by 18.50 percent and 16.87 percent in Micro-STEDS and Macro-STEDS, respectively.
- Hierarchical Structure Reconstruction: Surpasses DSPS Encoder's results without ground truth support, achieving 14.68 percent and 13.94 percent improvement in Micro-STEDS and Macro-STEDS, respectively.
* Our method handles all tasks concurrently with significantly superior results compared to previous state-of-the-art methods designed for each task.

### 5.4. Ablation Studies

**Ablation Experiments Based on Comp-HRDoc:**

**Evaluating Effectiveness of Hybrid Strategy in Detect Module:**
- **Hybrid (V) Model**: Combines Mask2Former for graphical object detection and visual features for text region detection
  * Achieves comparable graphical page object detection results with Mask2Former-R50 but higher text region detection accuracy on Comp-HRDoc
  * Significantly improves small-scale text region detection performance (e.g., Page-footnote, Page-header, and Page-footer)
- **Hybrid (V+T) Model**: Leverages both visual features and text modality in the Detect module
  * Achieves much better performance in semantically sensitive categories like Author, Mail, Affiliate, leading to a 4.66% improvement in terms of segmentation-based mAP.

**Evaluating Effectiveness of Multimodality in Construct Module:**
- **Section Numbers**: Impact on table of contents extraction when removed from text content
  * Presence or absence significantly affects performance with respect to text modality
  * Highlights the relationship between text modality and section numbers in extracting tables of contents.

**Additional Ablation Experiments:**
- **Text vs. Image Modalities**: Comparative assessment of individual components within TOC Relation Prediction Head.
  * Robustness of image modality demonstrated through higher scores than text modality alone
  * Most favorable performance achieved when both modalities are used together.

### 5.5. Limitations of Our Approach

**Limitations of Proposed Approach**
- Outstanding performance in majority of tasks, but not without limitations
- Assumes accurately recognized section headers by previous stages
    - Recognition of section headings accounts for bottleneck in Construct module
    - Importance of section numbers for semantics and harnessing meaning of section headings

**Ablation Studies of Modalities in Construct Module:**
| Modality | Micro-STEDS | Macro-STEDS |
|---|---|---|
| Text | 0.6409 (✓) | 0.8341 (✓) |
| Image with section number | 0.8477 (✓) | 0.8528 (✓) |
| Without section number | - | 0.8436 |
- Lack of robustness for documents without section numbers

**Ablation Studies of Components in TOC Relation Prediction:**
| Method | Level | Table of Contents Extraction |
|---|---|---|
| Micro-STEDS | Document | 0.8605 (✓) |
| Macro-STEDS | Document | 0.8788 (✓) |
| Ours | - | 0.8436 (?) |

**Failure Examples:**
- Incorrect predictions indicated by red boxes
- Correct predictions shown in green boxes
- Difficulties faced by other state-of-the-art methods
- Focus of future work: finding solutions to these problems

## 6. Conclusion and Future Work

**Thorough Examination of Hierarchical Document Structure Analysis (HDSA)**
- Study examines various aspects of HDSA
- Proposes a tree construction based approach called Detect-Order-Construct
- Addresses multiple crucial subtasks in HDSA:
  - **Page object detection**
  - **Reading order prediction**
  - **Table of contents extraction**
  - **Hierarchical structure reconstruction**

**Effectiveness of Proposed Framework**
- Design an effective end-to-end solution
- Uniformly define tasks as relation prediction problems

**New Benchmark: Comp-HRDoc**
- Evaluates performance of different approaches on:
  - Page object detection
  - Reading order prediction
  - Table of contents extraction
  - Hierarchical structure reconstruction

**Performance Results**
- Proposed end-to-end system achieves state-of-the-art performance on:
  - PubLayNet and DocLayNet datasets
  - HRDoc dataset
  - Comp-HRDoc benchmark

**Future Research Directions**
- Broaden scope to encompass a wider range of scenarios (contracts, financial reports, handwritten documents)
- Address documents with graph-based logical structures for more general applications
- Find a comprehensive and universal document structure analysis solution.

