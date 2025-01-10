# From Commands to Prompts: LLM-based Semantic File System for AIOS

Co-authors Zeru Shi and Kai Mei share equal contributions in this research article. (Source: <https://arxiv.org/html/2410.11843v1>)

## Contents
- [From Commands to Prompts:](#from-commands-to-prompts)
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 Semantic File System](#21-semantic-file-system)
  - [2.2 Semantic Parser](#22-semantic-parser)
  - [2.3 OS-related LLMs and Agents](#23-os-related-llms-and-agents)
- [3 Architecture of LSFS](#3-architecture-of-lsfs)
- [4 Implementation of LSFS](#4-implementation-of-lsfs)
  - [4.1 Basic Syscall of LSFS](#41-basic-syscall-of-lsfs)
  - [4.2 Supervisor](#42-supervisor)
  - [4.3 API of LSFS](#43-api-of-lsfs)
  - [4.4 LSFS Parser](#44-lsfs-parser)
  - [4.5 The Interaction between Modules](#45-the-interaction-between-modules)
- [5 Evaluation](#5-evaluation)
  - [5.1 RQ1: Effectiveness of LSFS Parser](#51-rq1-effectiveness-of-lsfs-parser)
  - [5.2 RQ2: Analysis of LSFS in Semantic File Management Tasks](#52-rq2-analysis-of-lsfs-in-semantic-file-management-tasks)
  - [5.3 RQ3: Analysis of LSFS in Non-Semantic File Management Tasks](#53-rq3-analysis-of-lsfs-in-non-semantic-file-management-tasks)
- [6 Conclusions](#6-conclusions)
- [Appendix A The Implementation Details of Syscall](#appendix-a-the-implementation-details-of-syscall)
- [Appendix B Details of keyword-based and semantic retrieval.](#appendix-b-details-of-keyword-based-and-semantic-retrieval)
- [Appendix C The Instruction of API](#appendix-c-the-instruction-of-api)
- [Appendix D Experiment Details of File Sharing](#appendix-d-experiment-details-of-file-sharing)
  - [D.1 The code cannot generate link](#d1-the-code-cannot-generate-link)
  - [D.2 The code can only generate local link](#d2-the-code-can-only-generate-local-link)
  - [D.3 The code can generate shareable link](#d3-the-code-can-generate-shareable-link)
- [Appendix E Further Discussions](#appendix-e-further-discussions)

## From Commands to Prompts:

**LLM-based Semantic File System for AIOS**

**Authors:**
- Zeru Shi
- Kai Mei (spadesuit)
- Mingyu Jin (spadesuit)
- Chaoji Zuo (spadesuit)
- Wenyue Hua (spadesuit)
- Wujiang Xu (spadesuit)
- Yujie Ren (S)
- Zirui Liu (diamondsuit)
- Mengnan Du (♢)
- Dong Deng (spadesuit)
- Yongfeng Zhang (spadesuit)
- Dalian University of Technology (spadesuit)
- Rutgers University (heartsuit)
- Purdue University (diamondsuit)
- New Jersey Institute of Technology (ddagger)
- EPFL (S)
- University of Minnesota

**Affiliations:**
- Dalian University of Technology: Dalian, China
- Rutgers University: New Brunswick, NJ, USA
- Purdue University: West Lafayette, IN, USA
- New Jersey Institute of Technology: Newark, NJ, USA
- EPFL: Lausanne, Switzerland

**Contributions:**
- Zeru Shi: LLM-based semantic file system for AIOS
- Kai Mei, Mingyu Jin, Chaoji Zuo, Wenyue Hua, Wujiang Xu, Yujie Ren, and Zirui Liu: Authors (no specific contributions mentioned)
- Dong Deng: Author (no specific contribution mentioned)
- Yongfeng Zhang: Author (no specific contribution mentioned)

**Footnotes:**
- 1: Multiple authors with the same affiliation are listed in the text.
- 2: No specific information is provided about the footnote.

## Abstract

**Proposed LLM-based Semantic File System (LSFS)**

**Background:**
- Large language models (LLMs) have shown potential in developing intelligent applications and systems
- Traditional file systems rely on manual navigation through precise commands, posing a usability limitation

**Limitations of Conventional Approaches:**
- Users or agents must navigate complex folder hierarchies
- Remember cryptic file names

**Proposed Solution: LSFS**
- LLM-based semantic file system for prompt-driven file management
- Incorporates LLMs to enable natural language interactions with files

**Macro-Level Functionalities:**
1. **Semantic File Retrieval**: retrieves files based on semantic meaning
2. **File Update Monitoring and Summarization**: tracks changes in files and provides summaries
3. **Semantic File Rollback**: allows users to revert files to previous states

**Micro-Level Implementation:**
- Construct semantic indexes for files
- Implement syscalls for CRUD, group by, join operations using vector databases

**Benefits:**
1. **Improved User Convenience**: simplifies file management tasks
2. **Diversity of Supported Functions**: enables a wide range of semantic file operations
3. **Accuracy and Efficiency**: enhances the precision and speed of file operations
4. **Intelligent Tasks**: facilitates content summarization and version comparison

**Open Source:**
- Code available at [https://github.com/agiresearch/AIOS-LSFS]()

## 1 Introduction

**LSFS: An LLM-Based Semantic File System**

**Background:**
- Integration of AI with operating systems for innovative applications [^30]
- Machine learning algorithms adjust resource allocation [^36]
- Prediction of future resource needs and pre-allocation [^25]
- Generative AI envisioned in Large Language Models (LLMs) based file systems [^12]

**Limitations of Traditional File Systems:**
- Index structures limit semantic meaning [^30]
  * Organize files based on attributes like name, size, timestamps
  * Lack capability to organize or retrieve files based on content similarity
- Complex operating system commands for user interactions [^28]
  * Precise recall of file names or locations required
  * Inefficient and time-consuming retrieval process

**Proposed Solution: LSFS - LLM-based Semantic File System**

**Components:**
1. Semantic-based index structure with vector database for file storage [^28]
   * Extract semantic features from file content
   * Generate embedding vectors
   * Leverages semantic information in file operations
2. Reusable syscall interfaces and APIs [^28]
3. Natural language input integration [^12]
4. Template system prompt for complex functions [^28]
5. Safety insurance mechanisms to avoid unintended operations [^28]

**Benefits:**
- Semantic file management [^28]
  * Organizes files based on content similarity
- User-friendly interface [^12]
  * Natural language input simplifies user interactions
- Enhanced efficiency [^25]
  * Prediction of future resource needs and pre-allocation
- Improved accuracy [^36]
  * Safety checks for irreversible operations and user verification before instruction execution.

## 2 Related Work

### 2.1 Semantic File System

**Semantic File Systems**

**Current File Storage and Retrieval**:
- Relies on an index structure maintained by the system
- File metadata points to the file's location on the disk
- Optimizing the index structure can enhance retrieval efficiency
- Current model is still dependent on keywords extracted from file content

**Semantic File Systems Proposal**:
- Semantic file systems introduce a layer that generates directories by extracting attributes from files
- Allows users to query file attributes through navigation
- Examples: 
    - **Semantic file system proposed by Gifford et al.** [^13]
    - **Semantic file management system called Sedar** proposed by Mahalingam et al. [^29]
    - Semantic information integrated into metadata by Hua et al. [^20], [^21]
    - Semantic vocabulary used for managing desktop data by Schandl et al. [^35]

**Proposed Semantic File System**:
- Based on the strong language understanding ability of LLMs (Liquid Learning Models)
- Integrates comprehensive semantic information across all aspects of the file system
    - Storage and file operations
    - Practical applications
- Enhances the system's ability to understand and manage files
- Significantly improves functionality beyond traditional systems and earlier semantic file systems

### 2.2 Semantic Parser

**Semantic Parsing Research**

**Developing machine-interpretable format from natural language:**
- Mooney et al. [^32]: Pioneered the field of semantic parsing with task-specific parsers
- Iyer et al. [^22]: Focused on parsing database commands
- Berant et al. [^1]: Proposed question-answer pair learning approach for efficiency
- Paraphrasing technique [^2] improvement by Berant et al.
- Poon et al. [^33]: Markov logic-based approach
- Wang et al. [^39]: Building parsers from scratch in new domains
- Ge et al. [^10]: Parse tree-based method for accurate semantic analysis
- Lin et al. [^27]: Integrated a semantic parser into an operating system, mapping bash commands to natural language

**Limitations of early approaches:**
- Difficulties in handling complex semantics and unseen natural language inputs

**Our LSFS**:
- Improved generation ability for understanding and processing diverse natural language inputs

### 2.3 OS-related LLMs and Agents

**LLM-based AI Agents in Operating Systems:**
* Enhance user experience, improve efficiency: Wu et al. [^43], MetaGPT [^18], CHATDEV [^34]
* Co-piloting users' interaction with computers: drawing charts, creating web pages (Wu et al.)
* Automate software development: assigning roles to various GPTs for seamless collaboration (MetaGPT)
* Framework for software development: enhancing cooperation among different roles (CHATDEV)

**LLM-based Agents in System-Level:**
* Enables fundamental services: agent scheduling, memory and storage management, tool management, security features, access control (LLM-based AIOS)
* Integration of LLMs into system level research.

## 3 Architecture of LSFS

**LSFS Architecture and Functionality**

**Overview**:
- LSFS is an additional layer on top of traditional file systems
- Constructs semantic index (embedding vectors) for advanced file management operations
- Provides system calls (syscalls) and APIs to support complex tasks

**Components**:
1. **LSFS** - constructs semantic index, enabling more advanced content-based file management
2. **Supervisor** - synchronizes changes between LSFS and traditional file system
3. **Syscalls** - basic operations: add, delete, modify, search
4. **APIs** - more advanced general-purpose file operations, built on syscalls and LLMs

**Architecture**:
- [Figure 2a](https://arxiv.org/html/2410.11843v1#S3.F2) outlines LSFS architecture:
  - Operates as an additional layer on top of traditional file systems
  - Builds semantic index for advanced file management operations
- [Figure 2b](https://arxiv.org/html/2410.11843v1#S3.F2) presents syscall structure in LSFS:
  - Atomic syscalls: basic file operations
  - Composite syscalls: complex functions built on atomic syscalls
  - LSFS APIs: facilitate more advanced tasks using one or more syscalls and LLMs.

## 4 Implementation of LSFS

**LSFS Implementation**

**Comparison of Key Functions:**
- [Table 1](https://arxiv.org/html/2410.11843v1#S4.T1)
  - New directory: `mkdir()` vs `create_or_get_file()`
  - Open file: `open()` vs `create_or_get_file()`
  - Read file: `read()` vs `create_or_get_file()`
  - Get file state and metadata: `stat()` vs `create_or_get_file()`
  - Delete directory: `rmdir()` vs `del_()`
  - Delete file: `unlink()` / `remove()` vs `del_()`
  - Write data: `write()` vs `insert()`
  - Overwrite data: `write()` vs `overwrite()`
  - Update access time: `utime()` vs `update_access_time()`
  - Automatic comparison: `compare_change()`
  - Generate link: `symlink()` / `link()` / `readlink()` vs `generate_link()`
  - Lock or unlock file: `flock()` vs `lock_file()` / `unlock_file()`
  - Rollback: `snapshot` + `rollback` vs `rollback()`
  - File translation: `file_translate()`
  - File group: `group_text()` / `group_se()`
  - Merge file: `cat` vs `join()`
  - Keyword retrieve: `grep` vs `keyword_retrieve()`
  - Semantic retrieve: `semantic_retrieve()`
  - Hybrid retrieval: `integrated_retrieve()`

**LSFS Architecture:**
1. **Basic Syscalls**: Introduced in LSFS and interact between LSFS syscalls and traditional file systems.
2. **Supervisor**: Interacts between LSFS syscalls and traditional file systems.
3. **APIs**: Built upon basic syscalls to achieve more complex functionalities.
4. **LSFS Parser**: Decodes natural language prompts into executable LSFS APIs.
5. **Modules Execution**: Achieves functionalities using different concrete prompts and various modules in the LSFS.

### 4.1 Basic Syscall of LSFS

**LSFS Syscalls**

**Atomic Syscalls:**
- **create\_or\_get\_file()**: integrates create, read, open functions; returns file metadata and essential information
- **add\_()**: writes new content to end of specified file
- **overwrite()**: overwrites original file with new content; generates new metadata as required
- **del\_()**: deletes files by name or path; supports keyword-based deletion
- **keywords\_retrieve()**: retrieves files containing a keyword in a directory; supports single and multi-condition matching
- **semantic\_retrieve()**: retrieves top-n highest semantic similarity files in a directory based on the query

**Composite Syscalls:**
- **create()**: creates files in bulk by importing folder path and all its files
- **lock\_file() / unlock\_file()**: locks/unlocks a file to change read-only state or permissions
- **group\_semantic()**: selects content in specified directory, retrieves files with high similarity, creates new directory, places matching files inside
- **group\_keywords()**: selects files containing retrieved keywords, creates new directory, places matching files inside
- **integrated\_retrieve()**: combines keyword search and semantic search to retrieve relevant files
- **file\_join()**: concatenates two files into one file; can create a new file or concatenate original directly.

### 4.2 Supervisor

**File Change Tracking with Supervisor in LSFS**

- Supervisor tracks file changes on disk and syncs updates to LSFS using syscalls
- Periodically scans specified directory for file changes/deletions
- Automatically synchronizes changes within LSFS upon detection
- Process lock mechanism ensures accurate syscall operations after Supervisor modifies internal LSFS status
- Generates status change reports, such as modification logs, when requested

### 4.3 API of LSFS

**LSFS APIs for Semantic File Management**

**Retrieve-Summary API:**
- Retrieves files based on user-specified conditions
- Provides concise summaries using LLMs
- Keyword, semantic, and integrated retrieval methods
  - keyword_retrieve() syscall
  - semantic_retrieve() syscall
  - integrated_retrieve() syscall
- Interaction interface for refining results
- Automatic summarization of files using LLMs

**Change-Summary API:**
- Modifies file contents and compares them before/after summization
- Supervisor module monitors file changes in traditional file system
- Locate target files using natural language
- Automatically generates summary of file changes through LLMs integration
- overwrite() and del_() syscalls used for implementation
- Version recorder stores metadata and contents of files for version control

**Rollback API:**
- Achieves version restoration using the version recorder from Change-Summary API
- Overwrite() and create\_or\_get\_file() syscalls employed
- Time-based rollback: parses rollback time from user input
- Version-based rollback: specifies number of versions backward to revert
- Traditional file systems offer snapshots for rollback, but LSFS's Rollback API introduces a more flexible approach

**Link API:**
- Enables creation of shareable file links
- Cloud database used for uploading files and generating shareable links
- Validity period can be passed as argument for secure and time-bound file sharing.

### 4.4 LSFS Parser

**LSFS Parser for Extracting Key Information from Natural Language Prompts**

**Overview**:
- LSFS parser designed to convert natural language commands into system prompt words
- Leverages large language models (LLMs) to detect and extract key parameters
- Outputs extracted keywords in comma-separated format, which can be directly converted into syscall parameters

**LSFS Parser Process**:
1. Natural language command is input alongside user's command
2. LLM automatically detects and extracts key parameters
3. Extracted keywords are outputted in comma-separated format
4. Simple regex applied to output to directly convert extracted keywords into syscall parameters
5. Seamless execution of API command ensues

**Benefits**:
- Efficiently maps natural language commands to exact system call parameters
- Enhances interactions between natural language prompts and LSFS

**Examples**:
1. [Figure 3](https://arxiv.org/html/2410.11843v1#S4.F3 "Figure 3 ‣ 4.4 LSFS Parser ‣ 4 Implementation of LSFS ‣ From Commands to Prompts: LLM-based Semantic File System for AIOS"): Illustration of using LLMs in extracting key information from natural language prompts
2. [Figure 4](https://arxiv.org/html/2410.11843v1#S4.F3 "Figure 3 ‣ 4.4 LSFS Parser ‣ 4 Implementation of LSFS ‣ From Commands to Prompts: LLM-based Semantic File System for AIOS"): Interactive examples showing how LSFS solves file management tasks step by step

### 4.5 The Interaction between Modules

**LSFS Parser Workflows:**
* **Retrieve-summary API**: upper section of [Figure 4](https://arxiv.org/html/2410.11843v1#S4.F4)
  + LSFS parser decodes prompts into API calls with name and arguments
  + Executes the API to check vector database for results
  + Provides user interface for result verification
  + LLM is used to summarize content after verification
* **Change-summary & Rollback APIs**: lower section of [Figure 4](https://arxiv.org/html/2410.11843v1#S4.F4)
  + LSFS parser decodes file information (name, location) for modification
  + LSFS modifies semantic changes in the vector database and files on disk
  + Supervisor ensures consistency between LSFS index and disk files
  + Summarization API generates detailed change log and stores pre-modification content in version recorder
  + Rollback API retrieves specific version from version recorder and synchronizes it in LSFS and disk files.

## 5 Evaluation

**Research Questions Regarding LSFS Performance**

1. **RQ1**: What is the success rate of the LSFS parser in interpreting natural language prompts to generate executable API calls?
2. **RQ2**: How does LSFS perform in semantic file management tasks, like semantic file retrieval and semantic rollback?
3. **RQ3**: Can LSFS still maintain good performance in non-semantic tasks, such as keyword-based file retrieval and file sharing?

### 5.1 RQ1: Effectiveness of LSFS Parser

**RQ1: Effectiveness of LSFS Parser**

**Assessment Methodology**:
- Evaluate accuracy of LSFS parser in translating user natural language prompt into executable LSFS API calls
- 30 different samples for each API on four LLMs: Gemmi-1.5-Flash, GPT-4o-mini, Qwen-2, and Gemma-2

**Results**:
- **Change-summary API and link API**: High accuracy (up to 100%) across all LLMs due to relatively simple semantic information in user prompts
- **Rollback API and retrieve-summary API**: Lower accuracy (85% on average) for more complex prompts, except Gemma-2
- Overall parsing accuracy: 90%

**Significance**:
- LSFS parser effectively translates natural language information into executable API calls
- Demonstrates reliability in diverse scenarios
- Translated command provided to users for approval before real execution, ensuring safety

### 5.2 RQ2: Analysis of LSFS in Semantic File Management Tasks

**Performance Analysis of LSFS vs Baseline in Semantic File Retrieval**
* Comparison of accuracy and execution time using Gemini-1.5-flash and GPT-4o-mini as LLM backbones
* Significant enhancement of both accuracy and efficiency with LSFS compared to traditional file system without LSFS
* As file number increases, retrieval accuracy drops when using LLM for retrieval without LSFS due to longer context leading to errors
* Using LSFS replaces reasoning process with keyword matching and semantic similarity matching, saving time and avoiding errors
* Case study: "Please search for the two papers most related to LLMs Uncertainty from folder named example" illustrates retrieval results, showing that LSFS delivers more accurate results compared to method without using LSFS.

**Scalability Analysis of Semantic File Rollback in LSFS**
* LSFS supports semantic file rollback enabling restoration of files to a particular version
* Experiment evaluates stability and efficiency of version rollback process by varying number of rollback versions and calculating corresponding rollback time
* Results show consistent time consumption during version rollbacks across LLM backbones (Gemmi-1.5-Flash, GPT-4o-mini, Qwen2)
* Rollback time does not increase exponentially with the number of versions rolled back but tends to plateau as each version is stored independently avoiding long rollback paths.

### 5.3 RQ3: Analysis of LSFS in Non-Semantic File Management Tasks

**Performance Analysis in Keyword-based File Retrieval:**
* **LSFS vs Traditional File Systems (TFS):**
  * Comparison of precision, recall, and F1-score for file retrieval by keywords
    - LSFS outperforms TFS search window and TFS-grep, only second to TFS-grep*
    - Built-in retrieval tool in TFS has high recall but inconsistent results due to fuzzy search feature
    - TFS-grep* achieves perfect results but requires manual adjustment of binary files and complex commands
  * LSFS allows natural language descriptions, eliminating need for complex commands
  * LSFS supports various types of text files, converting them into the system’s vector database for seamless retrieval operations.

**Performance Analysis in File Sharing:**
* **LSFS vs Baselines:**
  - Evaluation of success rate of generating sharable links, code generation rate, link generation rate, and link validness rate
    - LSFS achieves 100% link generation success rate, showing strong task fulfillment ability on file sharing.
  - Comparison between Gemini-1.5-flash, GPT-4o-mini, AutoGPT, Code Interpreter, and LSFS:
    - All methods can generate code but do not consistently generate valid links for shareable files.
    - In contrast, LSFS achieves 100% link generation success rate.

## 6 Conclusions

**Paper Outline:**
- Introduction of LLM-based Semantic File System (LSFS)
  - Offers advancement over traditional file systems by storing/managing files based on semantic information
  - Enhances system's ability to comprehend and utilize embedded semantics in file contents
- Introduction of reusable semantic syscalls and framework for natural language mapping into LSFS parameters
- Potential future applications:
  - Wider scope of user environments
  - Semantic file management in everyday computing
  - Simplifying user operations and creating a more intelligent, user-friendly operating system

**Key Takeaway:**
LSFS is a new semantic file system offering advancements over traditional ones. It stores/manages files based on semantic information, enabling better understanding of embedded semantics. The innovation includes reusable syscalls and natural language mapping framework. Future research can focus on wider applications in various user environments, integrating semantic management into daily computing for simpler, more intelligent OS systems.

## Appendix A The Implementation Details of Syscall

**LSFS Syscall Functions**

**create\_or\_get\_file():**
- Integrates file creation, reading, and opening functions
- Parameters: LSFS path, target directory name, target filename, import file (string or file path)
  - First two parameters are positional
  - Last two are default
- If target file does not exist: create imported file using specified directory name, filename, and import file
- Supports various text formats: PDF, DOCX, TXT, etc.

**add\_():**
- Facilitates appending content to a file
- Parameters: LSFS path, target directory name, target filename, import file content (string or file)
  - All are positional
- Imports file format: supports various formats

**overwrite():**
- Overwrites the entire contents of a file with the provided import file
- Parameters: LSFS path, target directory name, target filename, import file
  - All are positional

**del\_():**
- Deletes files and directories based on given parameters
- Parameters: LSFS path, target directory name, target filename, key text (optional)
  - First two are positional, last two are default
- If no last two provided: raises error
- Deletes specified file if filename given; searches for files containing key text in target directory and deletes them if found

**keywords\_retrieve():**
- Retrieves files within a specified directory that contain the given keyword
- Parameters: LSFS path, directory name, keyword, matching condition (optional)
  - First two are positional, last two are default
- If no directory name provided: searches entire system
- Returns list of file names and contents for matching files

**semantic\_retrieve():**
- Performs semantic similarity search within LSFS based on given query
- Parameters: LSFS path, target directory, search keyword, number of results to return
  - First two are positional, last one is default
- Returns list of file names and contents for top-scoring matches

**create():**
- Facilitates batch directory creation and bulk file reading from import file path
- Parameters: LSFS path, directory name, import file path

**lock\_file() / unlock\_file():**
- Handle file locking and unlocking for read-only access
- Parameters: LSFS pathname, directory name, filename

**group\_keywords():**
- Groups files with a common keyword and creates a new directory for them
- Parameters: LSFS name, keyword, new directory name, target directory, search criteria (optional)
  - First four are positional, last one is default

**group\_semantic():**
- Organizes files containing a common keyword into a new directory
- Parameters: LSFS path, keyword, new directory name, target directory, search criteria (optional)
  - First four are positional, last one is default

**integrated\_retrieve():**
- Combines semantic and keyword searches for comprehensive file retrieval
- Parameters: LSFS path, keyword, new directory name, search criteria (optional), target directory, additional search conditions (optional)
  - First five are positional, last three are default

## Appendix B Details of keyword-based and semantic retrieval.

**Keyword-based Retrieval (Single-condition)**
* Task: Find papers in computer vision category authored by Emily Zhang
* LLM w/o LSFS: Judge if input paper satisfies retrieve condition, summarize if yes
* LSFS: "Find papers in the computer-vision category authored by Emily Zhang. LLM input: You need to summary the content. The content is [file content]"
* Keyword-based Retrieval (Multi-condition)
* Task: Find papers from either Cambridge University or Columbia University
* LLM w/o LSFS: Judge if input paper satisfies retrieve condition, summarize if yes
* LSFS: "Find papers from either Cambridge University or Columbia University. LLM input: You need to summary the content. The content is [file content]"

**Semantic-based Retrieval (Single-condition)**
* Task: Locate the 3 papers showing highest correlation with reinforcement learning in LLM-training
* LLM w/o LSFS: Accept and remember input papers without outputting anything, find relevant ones later
* LSFS: "Locate the 3 papers showing the highest correlation with reinforcement learning in LLM-training. LLM input: You need to summary the content. The content is [file content]"

## Appendix C The Instruction of API

**Instruction of API in Section [4.3](https://arxiv.org/html/2410.11843v1#S4.SS3 "4.3 API of LSFS ‣ 4 Implementation of LSFS ‣ From Commands to Prompts: LLM-based Semantic File System for AIOS")**

**API Types**:
- **Change-Summary API**:
    - **Instruction**:
        - **With Directory**: Change the content of `/xxxx/xxxx.txt` to `old-file` under `llm-directory`.
        - **Without Directory**: Modify `/xxxx/xxxx.txt` to contain `change-file`.
    - **LLM Input**: Summarize differences between the two contents, before update is `[old file]`, after update is `[new file]`.
- **Rollback API**:
    - **Instruction**:
        - **By Date**: Revert the file named `syntax` to its version from 2023-6-15.
        - **By Version Number**: Rollback the `cnn` file to the state it was in 3 versions ago.
    - **LSFS Input**: N/A
- **Link API**:
    - **Instruction**:
        - **With Period of Validity**: Provide a link for `llm-base` that will be active for 3 months.
        - **Without Period of Validity**: Generate a link for `system-architecture`.
    - **LSFS Input**: N/A

## Appendix D Experiment Details of File Sharing

**LLM Execution Issues**:
- Examples display failures in generating shareable links
- Input given to LLM: "Write code to generate a shareable link for 'path'"

### D.1 The code cannot generate link

**Issue with Code Generation in Experiment:**
- LLMs may produce incorrect links or link addresses
- Example given: Algorithm [1](https://arxiv.org/html/2410.11843v1#alg1 "Algorithm 1") pseudo-code does not lead to the intended file
- Pseudocode for D.1 at (https://arxiv.org/html/2410.11843v1#A4.SS1) also fails to generate correct link
- Code snippet demonstrates the problem: Flask app for PDF file download with a hardcoded path, resulting in an incorrect link

**Conclusion:** The generated code may not correctly produce the desired link to the intended target file.

### D.2 The code can only generate local link

**Pseudo-code for File Sharing using LLM-based Semantic File System**
* **Algorithm 2**: Pseudo-code for file sharing ([D.2](https://arxiv.org/html/2410.11843v1#A4.SS2))
* **file_path**: Path to a specific file
* **shareable_directory**: Directory for shareable files
* **shared_file_path**: Path to shared version of the file
* **FileNotFoundError**: Error raised if the file does not exist
* **shutil.copy()**: Copies a file from one location to another
* **print(shareable_link)**: Prints the shareable link for the file
* **shareable_link**: URL of the shared file (local access only)

**File Sharing Process**
1. Check if the given file exists using `Path.exists()` method and raise an error if it doesn't.
2. Create a new directory for shareable files, or use an existing one, with `Path.mkdir(parents=True, exist_ok=True)`.
3. Copy the file to the shareable directory using `shutil.copy()`.
4. Generate the shareable link by combining the file path and a URL prefix.
5. Print the generated shareable link for the user.

### D.3 The code can generate shareable link

**Experiment Details of File Sharing: From Commands to Prompts**

**LLM-based Semantic File System for AIOS**

**Installing Dropbox SDK:**
1. Log in to the [App Console](https://arxiv.org/html/2410.11843v1#A4.SS3 "D.3 The code can generate shareable link ‣ Appendix D Experiment Details of File Sharing ‣ From Commands to Prompts: LLM-based Semantic File System for AIOS")
2. Create a new app and select appropriate permissions
3. Configure application permissions on demand
4. Create an access token using OAuth2

**File Generator:**
5. Import the private token of OAuth2: `accesstoken = 'Your token'`
6. Create a Dropbox client: `dpclient = Dropbox(accesstoken)`
7. Import file path: `path = 'xxx / xxxx.pdf'`
8. Use Dropbox to create a shared link: `link = dpclient.share(path)`
9. Get the link

**Challenges with LLMs:**
- Complex configuration steps for generating shareable links
- Users need to authorize the Dropbox app and obtain an access token
- Variety of platforms may lead to considerable user configuration time

**Introducing Link API:**
10. Simplified process: users only need to provide Google Drive credentials
11. Generate shareable links without extensive configuration.

## Appendix E Further Discussions

**File System Enhancements Based on LLMs (LSFS)**
* **Multi-modal, Multi-extension File Manipulation:**
  * Lack of dedicated syscall interfaces for non-text file types
  * Future work: leverage semantic information to optimize file management across various formats
* **Security and Privacy Enhancements:**
  * Robust protection measures against security threats (data leakage, tampering, unauthorized access)
  * Encryption techniques for secure data interactions and transmissions between LSFS and LLMs
* **Optimized Retrieval Strategies:**
  * Integration of advanced algorithms to enhance retrieval accuracy and effectiveness
* **More Instantiated APIs and syscalls:**
  * Deepening integration of traditional file system functions into LSFS
  * Exploring unexplored functionalities for future development
  * Providing users with more versatile and advanced capabilities through related API operations.

