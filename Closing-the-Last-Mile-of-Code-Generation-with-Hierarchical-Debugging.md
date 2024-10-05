# From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging

https://arxiv.org/html/2410.01215v1
by Yuling Shi, Songsong Wang, Chengcheng Wan, Xiaodong Gu

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Methodology](#3-methodology)
  - [3.2 Hierarchical Code Decomposition](#32-hierarchical-code-decomposition)
  - [3.3 Generating Test Cases for Subfunctions](#33-generating-test-cases-for-subfunctions)
  - [3.4 Debugging Subfunctions with LLM-simulated Execution](#34-debugging-subfunctions-with-llm-simulated-execution)
- [4 Experiments](#4-experiments)
  - [4.2 Main Results](#42-main-results)
  - [4.3 Ablation Study](#43-ablation-study)
  - [4.4 Debugging Different Types of Bugs](#44-debugging-different-types-of-bugs)
  - [4.5 Debugging Code with Varying Length](#45-debugging-code-with-varying-length)
  - [4.6 Impact of Debug Attempts](#46-impact-of-debug-attempts)
  - [4.7 Case Study](#47-case-study)
- [5 Conclusion](#5-conclusion)

## Abstract
**Multi-Granularity Debugger (MGDebugger): Closing the Last Mile of Code Generation with Hierarchical Debugging**

**Background:**
- Large language models have made significant strides in code generation, but pass rate is bottlenecked by subtle errors requiring human intervention
- Existing LLM-based debugging systems treat generated programs as monolithic units, failing to address bugs at multiple levels of granularity

**Introduction:**
- MGDebugger: a hierarchical code debugger that isolates, identifies, and resolves bugs at various levels of granularity
- Decomposes problematic code into a hierarchical tree structure of subfunctions
- Analyzes each subfunction and iteratively resolves bugs in a bottom-up manner

**Approach:**
- Proposed LLM-simulated Python executor to effectively test each subfunction
- Traces code execution and tracks important variable states to pinpoint errors accurately

**Results:**
- Achieves an 18.9% improvement in accuracy over seed generations in HumanEval
- Repair success rate of 97.6% in HumanEvalFix
- Effectively fixes bugs across different categories and difficulty levels, demonstrating robustness and effectiveness

## 1 Introduction

**Large Language Models (LLMs)**
- Significant advances in AI-assisted coding tasks: GPT-4 (OpenAI), LLaMA (Touvron et al.), DeepSeek-Coder (Zhu et al.)
- Trained on vast corpora of text and code, understand and generate code snippets for various programming tasks
- Proficient in tasks like code completion, bug detection, competitive programming challenges
- Generated code may contain critical errors requiring human intervention to pass tests

**Code Debugging Paradigm**
- New development paradigm: large models generate the code, humans fix it
- Numerous efforts made to debug LLM-generated code
- Most popular way: reuse LLM generator with test case execution feedback
- Increases pass rates but treats erroneous program as a single set of statements

**Multi-granularity Debugger (MGDebugger)**
- Novel debugging method for LLM-generated code
- Hierarchical, bottom-up strategy to systematically debug code
- Decomposes code into tree structure of sub-functions for semantic unit isolation
- Each sub-function debugged progressively from most granular to higher-level compositions
- Generates test cases derived from public test cases of main function and uses LLM-based execution simulator
- Precise and flexible error identification based on failed test cases
- Uncovers and rectifies bugs traditional methods may overlook

**Experimental Results:**
- Significantly outperforms existing debugging methods on HumanEval (Chen et al., 2021) and HumanEvalFix (Muennighoff et al., 2023) benchmarks
- Ablation studies confirm hierarchical debugging strategy's vital role
- Effective in handling diverse bug types and varying code lengths, demonstrating robustness and adaptability.

## 2 Related Work

**Related Work: Code Generation with LLMs**

**Advancements in Code Generation**:
- GPT4 (OpenAI, 2023), Codestral (Mistral AI team, 2024), DeepSeek-Coder (Zhu et al., 2024) have advanced code generation through:
  - Instruction tuning and RLHF with mixed code and natural language data

**Improving Code Generation**:
- Approaches focus on improving quality of generated code using planning algorithms
  - Transition from outlines to detailed implementations (Zhang et al., 2022; Yao et al., 2023; Zelikman et al., 2023; Zhou et al., 2023; Zheng et al., 2023)
- Sampling multiple programs and ranking to identify best one (Chen et al., 2023a; 2022; Ni et al., 2023)
- Leveraging multi-agent collaboration frameworks to enhance code generation quality (Zhang et al., 2024; Huang et al., 2023a; Dong et al., 2024)

**MGDebugger Approach**:
- Targets the post-generation phase, focusing on debugging and fixing errors that inevitably arise during code generation
- Repairing LLM-Generated Code: a critical aspect of software development (Just et al., 2014; Gupta et al., 2020; Yasunaga & Liang, 2021)
- Two main streams of research in repairing code generated by LLMs:
  1. Training models to repair code
     - (Huang et al., 2023b; Jiang et al., 2024; Ding et al., 2024; Zheng et al., 2024; Moon et al., 2024; Kumar et al., 2024)
  2. Providing external feedback to raw pretrained models to fix code
     - (Jiang et al., 2023; Chen et al., 2023b; Olausson et al., 2023; Zhong et al., 2024; Hu et al., 2024)
- MGDebugger does not require task-specific retraining but takes advantage of inherent capabilities of pretrained LLMs
- Falls under the category of work that leverages pretrained models to fix code by reasoning with external feedback

**Recent Debugging Techniques**:
- Use execution results from test cases to guide LLMs in code correction (Zhang et al., 2023; Olausson et al., 2023; Bouzenia et al., 2024; Lee et al., 2024; Xia & Zhang, 2023)
- Reflexion (Shinn et al., 2023): LLMs reflect on generated code and use memory buffer for iterative refinement
- Self-Debugging (Chen et al., 2023b): LLMs explain or dry run generated programs, known as rubber duck debugging
- LDB (Zhong et al., 2024): Segments programs into basic blocks and tracks variable values during runtime after each block to verify correctness against task description.

**MGDebugger's Hierarchical Approach**:
- Debugging from low-level errors to high-level flaws, ensuring a more systematic and accurate debugging process

## 3 Methodology

**MGDebugger Methodology Overview:**
- **MGDebugger**: A novel bottom-up hierarchical debugging method for repairing large language model (LLM)-generated code
- **Workflow of MGDebugger**: Illustrated in Figure 1; includes Hierarchical Code Decomposition, Subfunction Test Case Generation, LLM-Simulated Execution, and Bottom-up Debugging

**Hierarchical Code Decomposition:**
- Decomposes input buggy code into a hierarchical structure of subfunctions
- Enables systematic identification and resolution of bugs at various levels of granularity

**Subfunction Test Case Generation:**
- Generates test cases for each subfunction from public test cases of the main function
- Executes these test cases during debugging process

**LLM-Simulated Execution:**
- Simulates code execution for failed test cases using a large language model
- Monitors critical variables and state changes to pinpoint causes of errors

**Bottom-up Debugging:**
- Fixes identified bugs in each subfunction and updates it in the hierarchical structure
- Propagates changes to dependent functions through bottom-up debugging approach

**Benefits:**
- Tackles different types of bugs at various levels of abstraction
- Guarantees a cohesive and systematic debugging process throughout the entire code structure.

### 3.2 Hierarchical Code Decomposition

**Code Decomposition using Hierarchical Structure**

**Advantages:**
- Improves understanding of complex functions (Jain et al., 2023; Zelikman et al., 2023)
- Enables hierarchical debugging
- Facilitates isolated testing and debugging

**Process:**
1. Decompose complex function `f` into smaller subfunctions: `(f1,…,fn)`
2. Organize as a tree with main function `froot` at the top and children representing directly called subfunctions
3. Adhere to three principles:
   - Each subfunction represents minimal reusable unit of code
   - Higher-level functions call lower-level functions for complex functionality
   - Structure facilitates isolated testing and debugging

**Example:**
```figure
froot = TREE(froot, CHILD(froot))
```

**Principles:**
1. Minimal reusable unit of code with a specific purpose
2. Higher-level functions call lower-level functions for complex functionality
3. Structure facilitates isolated testing and debugging

### 3.3 Generating Test Cases for Subfunctions

**Subfunction Debugging Process**

**Overview:**
- Illustrated in Figure 2
- Initial test case generation for subfunctions
- Subsequent simulation of code execution
- Pinpointing errors accurately

**Subfunction Verification:**
- Generate test cases for each subfunction fi∈froot using automatic techniques (Wang et al., 2021; Schäfer et al., 2024; Liu et al., 2023a)
- Leverage provided public test cases 𝒯pub for main function to derive corresponding test cases for each subfunction
- LLM performs steps: analyze usage in main function and contribute to expected outputs, figure out input and output for each public test case
- Ensure generated test cases are reflective of intended functionality and contextually relevant within constraints provided by public test cases.

### 3.4 Debugging Subfunctions with LLM-simulated Execution

**Debugging Subfunctions with LLM-simulated Execution**

**Process Overview**:
- Use generated test cases to debug each subfunction
- Run on test case inputs, obtain results, compare against expected outcomes
- Fix failed test cases by correcting corresponding subfunction

**Debugging High-Level Functions**:
- Often unnecessary to track variable values in lower-level subfunctions
- External debugger can add overhead and complexity

**LLM-Simulated Code Executor**:
- Prompts LLM to act as a Python interpreter and track code execution
- Eliminates need for external debugger, offers more flexible and efficient solution
- Accurately identifies errors and their context

**Bottom-up Recursive Debugging Algorithm (MGDebugger)**:
1. **Input**: LLM-generated function `f`, public test cases `𝒯pub`
2. **Output**: Debugged function `f′`
3. **Function**:
   - If `f` has subfunctions, traverse hierarchy depth-first and recursively debug each one
   - Generate test cases for `f`, execute, and check results against expected outcomes
   - If correct: return `f`; if not, debug `f` based on test results
4. **Function Debugging**: Identify and fix bugs in `f` using information from the failed test case execution (`ℛf`)
5. **Return**: Corrected function `f′`
6. **Debugging Overall Workflow**: Call `MGDebugger` on main function `froot` with public test cases `𝒯pub`, recursively debug subfunctions, propagate changes

## 4 Experiments

**Code Generation and Debugging Experiments**
- Three state-of-the-art Language Models (LLMs) used for code generation: CodeQwen1.5 (7B), DeepSeek-Coder-V2-Lite (16B), and Codestral (22B)
- Three datasets used: HumanEval, MBPP, HumanEvalFix
- **Metrics:** Accuracy, Repair Success Rate (RSR)
- **Baselines:** Simple Feedback, Self-Edit, Self-Debugging (Expl.), Self-Debugging (Trace), LDB (Block/Line/Function), Reflexion, MGDebugger

**Results on HumanEval and MBPP Datasets:**
| Model    | Method  | Dataset   | Accuracy | Improvement | Repair Success Rate |
|-----------|---------|-----------|----------|-------------|---------------------|
| DeepSeek-Coder-V2-Lite | No-Debugging | HumanEval, MBPP | 76.8%, 84.1% | N/A, N/A | 67.2%, 71.2% |
| ... | Simple Feedback | Both | 82.3%, 85.4% | +5.5%, +9.2% | 69.4%, 74.0% |
| ... | Self-Edit | Both | 82.9%, 84.1% | +6.1%, +7.9% | 26.3%, 33.3% |
| ... | LDB (Block) | Both | 84.1%, 79.3% | +7.3%, +3.1% | 31.6%, 12.8% |
| ... | Self-Debugging (Expl.) | Both | 87.2%, 87.8% | +10.4%, +11.6% | 44.7%, 48.7% |
| ... | Self-Debugging (Trace) | Both | 86.0%, 84.8% | +9.2%, +8.6% | 39.5%, 35.9% |
| ... | Reflexion | Both | 90.9%, 87.8% | +14.1%, +11.6% | 60.5%, 60.5% |
| MGDebugger | No-Debugging | HumanEval, MBPP | N/A, N/A | N/A, N/A | N/A, N/A |
| MGDebugger | Simple Feedback | Both | 87.2%, 91.5% | N/A, +15.3% | N/A, +13.4% |
| MGDebugger | Self-Edit | Both | 85.4%, 86.0% | -1.0%, -1.9% | 38.5%, 42.5% |
| MGDebugger | LDB (Block) | Both | 79.3%, 83.5% | +3.1%, +7.9% | 12.8%, 32.5% |
| MGDebugger | Self-Debugging (Expl.) | Both | 87.8%, 89.6% | N/A, +14.0% | N/A, +57.5% |
| MGDebugger | Self-Debugging (Trace) | Both | 84.1%, 82.3% | +8.5%, -1.7% | 35.0%, 27.5% |
| MGDebugger | Reflexion | Both | 86.6%, 86.6% | N/A, N/A | 45.0%, 45.0% |
| CodeQwen1.5 | No-Debugging | HumanEval, MBPP | 76.2%, 75.6% | N/A, N/A | 67.4%, 65.4% |
| ... | Simple Feedback | Both | 85.4%, 88.4% | +9.2%, +12.8% | 38.5%, 52.5% |
| ... | Self-Edit | Both | 84.1%, 86.0% | +7.9%, +10.4% | 33.3%, 42.5% |
| ... | LDB (Block) | Both | 79.3%, 83.5% | +3.1%, +7.9% | 12.8%, 32.5% |
| ... | Self-Debugging (Expl.) | Both | 87.8%, 89.6% | N/A, +14.0% | N/A, +57.5% |
| ... | Self-Debugging (Trace) | Both | 84.8%, 82.3% | +8.6%, -2.1% | 35.9%, 27.5% |
| ... | Reflexion | Both | 87.8%, 86.6% | N/A, N/A | 48.7%, 45.0% |
| Codestral | No-Debugging | HumanEval, MBPP | 75.6%, 75.6% | N/A, N/A | 65.4%, 65.4% |
| ... | Simple Feedback | Both | 88.4%, 94.5% | +12.8%, +18.9% | 71.6%, 76.8% |
| ... | Self-Edit | Both | 86.0%, 86.0% | N/A, N/A | 42.5%, 30.0% |
| ... | LDB (Block) | Both | 83.5%, 79.5% | +7.9%, -3.1% | 32.5%, 30.0% |
| ... | Self-Debugging (Expl.) | Both | 89.6%, 94.5% | N/A, +14.0% | N/A, +39.0% |
| ... | Self-Debugging (Trace) | Both | 82.3%, 78.8% | -3.7%, -3.6% | 27.5%, 22.3% |
| ... | Reflexion | Both | 86.6%, 94.5% | N/A, +11.0% | N/A, +34.4% |

### 4.2 Main Results

**Key Findings:**
* **MGDebugger outperforms existing approaches**: consistently achieves higher accuracy improvements on HumanEval (+15.3% to +18.9%) and MBPP (+11.4% to +13.4%) compared to Self-Debugging (Expl.) and Reflexion.
* **Strong results across various models**: demonstrates adaptability to different LLM architectures, achieving high accuracy with DeepSeek-Coder-V2-Lite (16B) and Codestral (22B).
* **Impressive zero-shot performance**: operates in a zero-shot setting without task-specific retraining, demonstrating inherent debugging ability of larger LLMs.
* **Robustness across RSR**: achieves up to 41.1% RSR on MBPP with smaller models like CodeQwen1.5 (7B).

**Method Description:**
- MGDebugger is a highly effective and scalable debugging method for large language models (LLMs).
- It consistently outperforms existing debugging approaches across various models and datasets, including HumanEval and MBPP.
- The method demonstrates strong results with DeepSeek-Coder-V2-Lite (16B) and Codestral (22B), achieving an accuracy of 94.5% on the HumanEval dataset and remarkable debugging capabilities.
- MGDebugger operates in a zero-shot setting without task-specific retraining, illustrating its inherent debugging ability in larger LLMs.

### 4.3 Ablation Study

**Ablation Study Results for DeepSeek-Coder-V2-Lite Model**

**Table 2:** Ablation study results for DeepSeek-Coder-V2-Lite on HumanEval and MBPP datasets.

| Method | HumanEval Accuracy | Δ Accuracy | Repair Success Rate (RSR) | MBPP Accuracy | Δ Accuracy | RSR |
|---|---|---|---|---|---|---|
| MGDebugger | 94.5% | +17.7% | 76.3% | 80.0% | +12.8% | - |
| **w/o Hierarchical Debugging** | 89.0% | +12.2% | 52.6% | 78.2% | +11.0% | 33.5% |
| **w/o Simulated Execution** | 90.2% | +13.4% | 61.3% | 79.2% | +12.0% | 36.6% |
| **w/o Test Case Generation** | 90.9% | +14.1% | 60.5% | 79.2% | +12.0% | 36.6% |
| No-Debugging | 76.8% | N/A | 67.2% | - | - | - |

**Components and Their Impact on MGDebugger:**
- **Hierarchical Debugging Strategy**: The most impactful component, removing it drops the repair success rate significantly from 76.3% to 52.6% for HumanEval and from 39.0% to 33.5% for MBPP.
- **LLM-Simulated Execution** and **Test Case Generation for Subfunctions**: Facilitate debugging the decomposed code, resulting in substantial improvements in accuracy and repair success rates.

### 4.4 Debugging Different Types of Bugs

**Experiments on HumanEvalFix Dataset**
- Assessing versatility and effectiveness of MGDebugger using HumanEvalFix dataset
- Six distinct bug categories: value misuse, missing logic, excess logic, operator misuse, variable misuse, function misuse

**Results for Different Methods (RSRs)**
| Method 	| Missing Logic | Excess Logic | Operator | Variable | Function | Overall |
|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite | **85.4** | 80.7 | 78.3 | **86.4** | 87.5 | 85.4 |
| Self-Edit | 82.3 | 62.5 | 80.7 | 84.1 | 62.5 | 82.3 |
| LDB (Block) | 81.1 | **96.0** | 74.2 | 86.4 | 62.5 | 81.1 |
| LDB (Line) | **74.4** | 52.4 | 56.5 | 54.6 | 62.5 | 74.4 |
| LDB (Function) | 76.8 | 51.2 | **87.0** | 73.9 | 59.1 | 76.8 |
| Self-Debugging (Expl.) | 78.7 | 62.5 | 69.6 | 77.3 | 78.7 | 74.4 |
| Self-Debugging (Trace) | **79.3** | 75.0 | 80.6 | 69.6 | 70.5 | 73.2 |
| Reflexion | 81.5 | **100.0** | 80.6 | **91.3** | 75.0 | 81.7 |
| MGDebugger | **97.6** (using DeepSeek-Coder) | N/A | N/A | N/A | N/A | N/A |

**Conclusion:**
- MGDebugger outperforms other methods with significantly higher overall accuracies in HumanEvalFix dataset experiments.
- Achieves remarkable repair success rate of 97.6% using DeepSeek-Coder, with 100% success rates in all bug categories except for value misuse.
- Strong advantage in debugging bottom-level bugs like missing logic and excess logic compared to other methods due to hierarchical decomposition.

### 4.5 Debugging Code with Varying Length

**MGDebugger's Performance Compared to Other Methods**
- **Refer to Figure 3**: Repair success rate of different methods on code snippets of varying lengths using DeepSeek-Coder.
- MGDebugger outperforms other methods in long codes.
- **Figure 4**: Impact of debugging attempts on cumulative repair success rates for MGDebugger and other methods on HumanEvalFix with DeepSeek-Coder.
- MGDebugger improves with more debug attempts and achieves highest success rate.

**Assessing Versatility of MGDebugger in Debugging Code of Different Lengths**
- **Code length**: Correlates with complexity and debugging challenges.
- Three groups: short, medium, and long based on equal sample sizes from HumanEvalFix dataset.
- Results presented in Figure 4.
- As code length increases, most methods perform poorly due to increased complexity.
- MGDebugger consistently outperforms other methods across different code lengths, especially in longer and more complex snippets.
- Demonstrates scalability and robustness of MGDebugger for handling varying lengths and complexities.

**Additional Results on Other Datasets**:
- Available in Appendix D (not provided).
- Consistently outperforms other methods across different code lengths on those datasets as well.

### 4.6 Impact of Debug Attempts

**Impact of Debug Attempts**

**MGDebugger's Ability**:
- Improves over successive iterations during debugging attempts using LLMs
- Achieves highest cumulative **RSR score** among all methods on HumanEvalFix dataset and DeepSeat-Coder
- Outperforms other methods from the first attempt and continues to improve with great potential

**Comparison to Other Methods**:
- Most methods plateau after a few debug attempts
- MGDebugger and Reflexion continue to improve with more iterations
- Highlights MGDebugger's potential for **iterative and comprehensive debugging**
- Promising solution for complex and challenging code repair tasks.

### 4.7 Case Study

**MGDebugger vs Baseline Methods in Code Debugging**

**Key Findings**:
- MGDebugger successfully identifies and corrects bugs, unlike baseline methods that introduce new bugs
- This is demonstrated through debugging code snippets from HumanEvalFix dataset using DeepSeek-Coder as the backbone LLM
- Original buggy solution had incorrect computation logic for Collatz sequence (n = n × 2 + 1 instead of n = n × 3 + 1)
- Baseline methods corrected this, but missed last "1" in the Collatz sequence by filtering odd numbers first and appending number to results after updating n
- MGDebugger decomposed problem into distinct subfunctions: **sequence generation** and **odd number filtering**, ensuring comprehensive error correction
- MGDebugger's ability to handle complex problems more effectively and restructure code for clarity and correctness demonstrates its potential in improving LLM-generated code quality.

## 5 Conclusion

**Conclusion**

**MGDebugger**:
- Novel hierarchical code debugging framework
- Fixes bugs at multiple levels of granularity
- Decomposes complex code into a hierarchical structure
- Generates targeted test cases and employs LLM-simulated execution
- Identifies and fixes syntax errors to logical flaws in a bottom-up manner

**Performance**:
- Superior performance over existing methods, particularly in handling complex logical errors and longer code snipplets

**Future Work**:
1. **Handling more complex bugs and code structures**: Extending MGDebugger to handle multi-file projects and codebase with multiple dependencies.
2. **Exploring collaboration of hierarchical approaches**: Building on the foundation of Parsel (Zelikman et al., 2023) for end-to-end code generation and debugging systems.
3. **Integrating MGDebugger into self-training systems**: Correcting outputs from base models, then retraining the base models with the corrected data to potentially improve performance iteratively (Gulcehre et al., 2023).

