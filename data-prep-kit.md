# Data-Prep-Kit: getting your data ready for LLM application development

by David Wood, Boris Lublinsky, Alexy Roytman†, Shivdeep Singh, Abdulhamid Adebayo, Revital Eres†, Mohammad Nassar†, Hima Patel, Yousaf Shah, Constantin Adam, Petros Zerfos, Nirmit Desai, Daiki Tsuzuku, Takuya Goto, Michele Dolfi, Saptha Surendran, Paramesvaran Selvam, Sungeun An, Yuan Chi Chang, Dhiraj Joshi, Hajar Emami-Gohari, Xuan-Hong Dang, Yan Koyfman and Shahrokh Daijavad

https://arxiv.org/pdf/2409.18164
## Contents
- [Abstract](#abstract)
- [I. INTRODUCTION](#i-introduction)
- [II. RELATED WORK](#ii-related-work)
- [III. TOOLKIT DESIGN](#iii-toolkit-design)
  - [A. Data Access](#a-data-access)
  - [B. Transformation](#b-transformation)
  - [C. Runtime](#c-runtime)
- [IV. AVAILABLE TRANSFORMS](#iv-available-transforms)
- [VI. HOW TO BRING YOUR OWN TRANSFORM DPK](#vi-how-to-bring-your-own-transform-dpk)
- [VII. EXPERIMENTAL EVALUATION](#vii-experimental-evaluation)
- [VIII. CONCLUSION](#viii-conclusion)

## Abstract
- Importance of data preparation in Large Language Model (LLM) development
- Introduction of DPK: an open-source, extensible, and scalable toolkit for data preparation
- Architecture and design to enable users to prepare data on a local machine or cluster

**Features:**
- Highly scalable set of modules transform natural language and code data
- Extensibility for additional transforms using DPK's support for transform creation
- Modules can be used independently or pipelined to perform multiple operations

**Architecture:**
- Enables users to scale data preparation needs
- Compatibility with Ray, Spark, and KFP for large-scale processing

**Applications:**
- Improving performance of LLM models or fine-tuning Retrieval-Augmented Generation (RAG) models.

**Conclusion:**
- DPK is a valuable contribution to the AI community for easily preparing data to enhance LLM model development and generation.

## I. INTRODUCTION

**Data Prep Kit (DPK)**
- Introduced to address challenges in data preparation for Large Language Models (LLMs) applications
- Accessible at: https://github.com/IBM/data-prep-kit
- Designed to provide support for various data modalities and personas

**Goals of DPK:**
1. Offer a consistent approach to using data preparation modules for different data types
2. Support both proof-of-concept stage and production stage with scale flexibility
3. Usable by anyone without deep knowledge of distributed computing or frameworks like Ray and Spark
4. Automate capabilities through Kubeflow Pipelines (KFP) UI
5. Allow easy addition of new data preparation modules

**Features of DPK:**
- High-level overview: A toolkit offering out-of-the-box data preparation modules that can be connected to form pipelines tailored to specific needs
- Flexibility: Platform support from laptop to large Kubernetes clusters using Ray and Spark runtimes
- Extendibility: Data processing lib abstracting the details of Ray and Spark for adding new transforms with minimal skills required
- Automation: Scalable data preparation pipelines through Kubeflow Pipelines (KFP) no-code execution.

**Components:**
1. Out-of-the-box data preparation modules (transforms): Combine PDF2Parquet, exact-deduplication, document-quality check, document-chunking, and building document embeddings for RAG tasks.
2. Flexible computing options: Laptop to large Kubernetes clusters using Ray and Spark runtimes.
3. Novel data processing framework (data processing lib): Abstracts details of Ray and Spark for easy addition of new transforms without extensive skills required.
4. Automation via KubeFlow Pipelines (KFP) UI: Enables running data prep pipelines in a no-code mode through the UI, making it usable across various personas.
5. Customizable: Supports adding new transforms without requiring deep knowledge of Ray or Spark.

**Future Work:**
- Discuss related work in detail (Section II)
- Explore design of toolkit and available transforms (Section III & IV)
- Achieve automation via KFP (Section V)
- Add new transforms using DPK (Section VI)
- Present experimental results, including small to large scale data analysis for training LLMs using Granite models (Section VII).

## II. RELATED WORK

**Data Preparation Kit (DPK)**
- Open-source project for data processing and preparation of LLM applications
- Similar projects: BigCode [8], DataTrove [9], Dolma [10]
- Differences from similar projects:
  - Focuses on preparation of data in creating GenAI models, not just LLMs
  - Scalability via Ray and Spark frameworks
  - Automation via KFP
  - Wide range of transformation modules
  - Targets down-stream applications like Fine-tuning, Instruction-tuning, and RAG

**Comparison with Other Similar Projects:**

**Nvidia NeMo-curator**:
- Focuses on NLP only
- Leverages GPUs for parallelization using DASK and Nvidia's RAPIDS libraries
- DPK uses Ray and Spark for scalability, no need to use GPUs

**DataComp-LM (DCLM)**:
- Testbed for controlled dataset experiments to improve large language models
- Allows experimenting with data curation strategies like deduplication, filtering, and mixing at model scales
- Focuses on NLP only
- Uses Ray for scaling, but fewer transformation modules than DPK
- Does not target down-stream applications like RAG and fine-tuning

**Unstructured.io**:
- Full-featured ingestion engine for LLM applications
- Targets RAG, includes chunking and embedding modules
- No real scaling for ingestion on a cluster (no Ray or Spark)
- Does not have the same range of transformation modules as DPK

## III. TOOLKIT DESIGN

**DPK Architecture Components:**
- **Data Access**: Identifies, selects, reads, and writes data in a supported format. Supports checkpointing. Configurable via command line arguments. Independent of Transform and Runtime components.
- **Transformation**: Implements specific operations on the data, e.g., conversion or deduplication. Individually configurable using command line arguments. Can be executed in sequence to form pipelines.
- **Runtime**: Identifies execution environment for transforms and starts Transform Workers. Distributes work among workers and operates on identified data provided by Data Access component.

### A. Data Access
- Core element of the architecture that provides a general-purpose framework for data processing
- Supports local file system and S3-compatible storage through abstraction layer (DataAccess class)
- Configurable using command line arguments via DataAccessFactory
- Provides standardized APIs independent of actual storage type
- Current implementation supports local file system and S3 data access; easily extendable for user-specific storage types
- Enables checkpointing to determine unprocessed files during restarts.

**Architecture Overview:**
1. Data used for processing can be stored on various devices, including local or distributed networks and S3-compatible storage.
2. DPK supports an abstraction layer (DataAccess) that provides standardized APIs for data processing-oriented tasks.
3. The architecture includes Data Access, Transformation, and Runtime components to enable developers to quickly create new transforms and easily deploy them for data processing.

### B. Transformation

**Data Transformation**

**Purpose**: Manipulates arbitrary unstructured or structured data

**Key Features**:
1:1 transformation - single data object becomes a single transformed object (e.g., model score annotation)
1:N transformation - single data object becomes multiple data objects (e.g., splitting row data into smaller objects)
N:1 transformation - multiple data objects become a single data object (e.g., joining multiple rows into one larger object)
N:M transformation - any number of data objects can be converted to any other number (e.g., sorting data into specific data types)

**Methods in AbstractBinaryTransform**:
- `transform_binary(file_name, bytes_to_transform)`: Transforms the given byte array into 0 or more byte arrays based on file format determined by `file_name`. Returns list of transformed byte arrays, file extensions for writing, and optional metadata.
- `flush_binary()`: Supports stateful accumulation of data across transform method calls. Returns same data as `transform_binary()` method. Useful when aggregating small files into larger ones.

**Configuration**:
- Configured via a dictionary at creation time using Runtime component and command line arguments.
- Transforms may handle multiple data types by checking the format of bytes in `file_name`.

**AbstractTableTransform Class**:
- Simplifies transformation of Arrow tables (read from .parquet files) through methods like `transform(table, file_name)` and `flush()`.

**Additional Transform Classes**:
- Can be created to simplify processing specific file types.

**Transform Configuration**:
- Base class called `TransformConfiguration` defines the name of transform as reported by Runtime, class name of Transform implementation, and methods for defining and validating command line arguments used to instantiate the transform.

### C. Runtime

**DPK Runtime Components:**
- **Runtimes**: Establish transform environments, assign tasks, monitor progress
  - Pure Python
    * Runs transforms within a Python process
    * Supports multiprocessing
  - Ray
    * Runs transforms in Ray Actors
    * Uses local or remote Ray cluster
  - Spark
    * Runs transforms using either local or remote Spark cluster
- Flexible deployment: Local to Kubernetes clusters with thousands of nodes
- Testing simplified on Kind cluster for Kubernetes
- Transform-specific runtime support classes: Python, Ray, Spark
- Components: Launcher, Orchestrator, Workers
  - **Transform Launcher**: Entry point, configures components and initializes runtime (optional)
  - **Transform Orchestrator**: Establishes shared components and processes files using Data Access and Workers
    * Creates Data Processor instances for each file
    * Reads files, passes to transform, writes results back to storage, and updates statistics
  - **Data Processor**: Instantiates transform and data access, processes files and writes results
- Scalable architecture: Supports a wide range of use cases in LLM data preparation.

## IV. AVAILABLE TRANSFORMS

**DPK Transforms**

**Available Categories:**
- **Data Ingestion**: N/A
- **Universal**: N/A
- **Code**: Implemented in pure Python and Ray-based KFP for automation.
  - Compute execution parameters
  - Start a Ray cluster
  - Execute a Ray job
  - Stop/destroy the Ray cluster
- **Language**: Not specified in text

**Benefits of Automation with KFP:**
1. Scalability: Runs on Kubernetes, handling large datasets and complex workflows.
2. Modularity: Breaks down tasks into reusable shared components for easy pipeline building.
3. History and Reproducibility: Maintains a history of executions to ensure experiment repeatability.
4. Visualization: Provides UI for monitoring runs, visualizing results, and troubleshooting issues.

**DPK Super Pipeline:**
- A concept introduced in DPK for executing several transforms as one "super" pipeline with nested simple pipelines.

**Figure 3**: Simple pipeline execution (not shown)

**Figure 4**: Super pipeline for Code preprocessing

**Table 2**: DPK Transforms
| Category   | List of Current Transforms |
|---|---|
| Data Ingestion | N/A |
| Universal | N/A |
| Code         | Compute execution parameters, Start a Ray cluster, Execute a Ray job, Stop/destroy the Ray cluster |
| Language     | N/A                        |

## VI. HOW TO BRING YOUR OWN TRANSFORM DPK

**Data Prep Kit (DPK) and Hello World Example**

**DPK Overview**:
- Extensible library for creating custom transforms
- Transforms can be applied using one of the runtimes
- Illustrates steps to write a new transform: adding "hello" column to PyArrow Table objects

**PyArrow Transformation Specialization**
- Focuses on transforming PyArrow Table objects read from parquet files
- Adds a new column containing a "hello message" defined by command line arguments

**Transform Implementation**:
- **HelloTransform** class extends `AbstractTableTransform` and provides configuration through an initializer
- Implements a `transform()` method to add the new "hello" column
- Takes in-memory PyArrow Table and optional parquet file name
- Returns table and optional metadata

**Runtime Configuration**:
- **HelloTransformConfiguration** class defines transform implementation, command line options, and name
- Includes methods `add_input_params()` and `apply_input_params()` to configure command line arguments
- **PythonRuntimeConfiguration** class holds the **HelloTransformConfiguration** and includes a main function to run the transform on input data.

**Running the Transform**:
- To run the transform on parquet files in an "input" directory and place output in an "output" directory:
  - `% python hello_transform.py --data_local_config '{"input_folder": "input", "output_folder": "output"}' --who Universe --column_name hello`

## VII. EXPERIMENTAL EVALUATION

**Scalability of Data Processing Kit (DPK) Transforms**

**Performance on Single Node:**
- Investigated impact of logic complexity on performance
- Used a node with 16 CPU cores and 64GB RAM
- Observed influence of transform intricacy on throughput in Fig. 5

**Performance in Cluster Setting:**
- Evaluated scalability of DPK on a cluster of 40 nodes with:
  - 48 CPU cores
  - 384GB RAM
- Examined impact of three transform categories (C1, C2, and C3) on data processing throughput in Fig. 6
- Demonstrated relationship between complexity and handling larger volumes of data

**Effectiveness vs. Complexity:**
- Language identification (lang ID) in C3 has lower throughput compared to simple annotation transforms due to model inference requirements
- Proportional reduction in execution time with increased CPU cores, showing ability to handle distributed workloads and low-resource environments
- I/O bound transforms (C1) have least impact on scalability but moderate influence on complex file manipulation transforms (C2), most substantial impact on model inference transforms (C3)

**Data Processing Metrics:**
| Transform Name | Input Data Size | Percentage of data filtered | Compute Time (minutes) | Number of CPU Cores |
|---|---|---|---|---|
| edup | 2TB | 16.14% | 38.15 min | *N/A* |
| f-dedup | 2TB | 24.3% | 1,511.65 min | *N/A* |
| edup | 332GB | 3.3% | 5.2 min | 320 |
| f-dedup | 332GB | 4.9% | 107.49 min | 320 |

## VIII. CONCLUSION

**DPK (Data Preparation Kit) for LLM Applications**

**Features**:
- Flexible: runs on different platforms
- Extensible: add new scalable modules without deep Ray and/or Spark expertise
- Out-of-the-box automation for existing and newly added modules

**Benefits**:
- Useful toolkit for users to prepare data
- Allows for easy customization or extension of the toolkit

**DPK Modules**:
- Can be used independently or in a pipelined fashion
- Automation enables scaling workload on clusters through KFP dashboard
- Same automation applies to any new modules added by users

**Use Cases**:
- DPK modules have been used with automation at scale for IBM Granite Models
- Expected to be valuable to the larger LLM data engineering community

**Future Plans**:
- Expansion of DPK capabilities:
  - Support for new data modalities
  - Additional scalable runtimes
  - New readily usable transforms

