# LLM With Tools: A Survey

**Tool Integration for Large Language Models (LLMs)**

by Zhuocheng Shen
https://arxiv.org/abs/2409.18807

## Abstract

**Abstract**:
- Integration of tools with LLMs: novel approach to enhance efficiency and accuracy
- Discuss methodology, challenges, and developments in teaching LLMs to use external tools
- Push the boundaries of capabilities beyond existing knowledge bases

**Paradigm for Tool Integration**:
- Standardized framework guided by functions that map user instructions to actionable plans
- Emphasis on understanding **user intent**, **tool selection**, and dynamic plan adjustment

**Challenges in Tool Integration**:
- **Tool invocation timing**: when and how to invoke tools during processing
- **Selection accuracy**: ensuring the selected tool is appropriate for the given task
- Need for **robust reasoning processes**: handling exceptions, errors, or unexpected results

**Approaches to Address Challenges**:
- **Fine-tuning**: refining pretrained models on new data
- **In-context learning**: learning from examples in context
- Ensuring **diversity**: using a wide range of training examples and tools
- Augmenting datasets: generating synthetic data to supplement real data
- Improving **generalization**: leveraging transfer learning, regularization, and ensembling techniques

**Autonomous Tool Creation by LLMs**:
- Reproducing Chameleon's results on ScienceQA and analyzing the code structure
- Potential for LLMs to create their own tools: redefining their role from tool users to creators.

**Keywords**: Large Language Models, Tool Integration, Fine-tuning, In-Context Learning, Retrieval.

## I. INTRODUCTION

**Introduction:**
- Rapid development of AI technology and large language models becoming integral to life
- Large models have excellent natural language understanding and reasoning abilities, but face challenges in specific fields
- Performance unsatisfactory in scenarios requiring precision or real-time performance due to being probabilistic models
- Inspiration drawn from human use of tools as an extension of abilities
- Overcoming challenges through endowing large models with tool usage ability

**Framework for Teaching Large Models:**
- User instructions: model must understand user intentions and needs
  - Accurately capture real needs behind instructions
  - Ensure pertinence and effectiveness of subsequent plans
- Understanding tools: functionality, usage methods, applicable scenarios, limitations
  - Deeply understanding tools to determine when, where, and how to use them effectively
- Decomposing complex tasks into subtasks: breaking down tasks into smaller units
  - Adjust plans flexibly to adapt to situations and needs
- Reasoning ability: dynamically adjust plan during task execution

**Benefits:**
- Improve accuracy of task execution
- Expand functional boundaries of large models
- Contribute to AI technology development
- Impact human society progressively.

## II. CHALLENGES

**Challenges in Using Large Models**

**A. Identifying When to Call External Tools:**
- Crucial to identify when model cannot provide accurate answers or solutions: real-time data, complex calculations, special format documents.
- Efficiency and accuracy considerations: call external tools only when necessary.

**B. Tool Selection and Accuracy:**
- Importance of choosing appropriate tools for successful task completion.
- Increasing number of available tools makes maintaining accuracy more difficult.
- Performance differences between tools; understanding their advantages and limitations.
- Complex reasoning processes involve non-linear reasoning or multi-tool links, increasing call accuracy challenges.

**C. Method of Tool Call:**
- Understanding tool interface: parameters, incoming data types, values.
- Correct usage of APIs essential for effective calls and avoiding errors.

**D. Robustness of Reasoning Process:**
- Errors can lead to amplification of problems, affecting quality of results.
- Establishing a robust mechanism to detect and correct errors crucial.
- Challenges in complex inference chains or multi-step operations.

**E. Time Efficiency:**
- Optimizing time efficiency important for large models usage.
- Some tools have delays, especially with network requests or processing large amounts of data.

**F. Generalization Ability:**
- Achieving General Artificial Intelligence (AGI) an ultimate goal in AI field.
- Whether different tools can be called to solve complex reasoning problems is an open question.
- Rapidly developing large language models but achieving complete AGI still faces many challenges.
- Key factor for AGI achievement: model's ability to create or innovate based on existing tools for unfamiliar ones.

## III. RESEARCH STATUS

**Research Status: Using External Tools for Large Models**

**Current Research Trend**: Shift towards teaching large models to utilize tools effectively
- Studies explore allowing models to develop their own tools

**Paradigm of Tool Use**
- Given user instructions U, toolset T, and environment state space E:
  - `fintent(u)`: Maps user instruction to intent
  - `fplan(i, T)`: Generates a plan based on intent and available tools
  - `fexec(p, e)`: Executes the plan in the environment
  - `ffeedback(e')`: Creates feedback results from the updated environment state
  - `fperceive(r)`: Processes feedback and generates a summary
  - `fadjust(s, p)`: Adjusts the plan based on the summary
- Tool usage process:
  1. User issues instruction `u`
  2. System identifies user intent `i = fintent(u)`
  3. System generates a plan `p = fplan(i, T)`
  4. System executes the plan in environment `e`, resulting in new state `e' = fexec(p, e)`
  5. System generates feedback `r = ffeedback(e')`
  6. System processes feedback and generates a summary `s = fperceive(r)`
  7. System adjusts the plan based on the summary `p' = fadjust(s, p)`
- This process iterates until the task is completed

**Approaches for Selecting Tools in Large Models**:
- **Fine-tuning**: Inject knowledge into the model through fine-tuning to assimilate specific expertise related to the tool and its application
- **In-context learning**: Leverage the in-context learning capabilities of the model to infer how to utilize the tool based on provided examples or hints within its current context
- **Generating custom tools**: Enable the model itself to generate tailored tools for specific tasks

### B. Fine tuning

**Fine Tuning Practices**

**Dataset C\*: Structure and Simplification**
- Dataset comprised of sequence of samples (x1 to xj)
- Each sample includes: X\* = w1:i−1; E(ci;Ri); wi:n

**E(c;r)**
- Syntactical construct for tool invocation
- <TOOL > and </TOOL > as boundary markers
- "." as delimiter between command and outcome
- ac denotes tool name, ic required input, ren result post-execution
- w1:i−1 and wi:n not inherent part of tool invocation

**Training Model M with Dataset C\*\n**
- Standardized language format during fine-tuning phase
- Unaltered large-scale model trained on exhaustive samples
- Context and internal feedback mechanisms influence tool usage decisions

**Text Generation Algorithm with Tool Integration**
1. procedure INFER_WITH_TOOL(model, input nl, maxlen )
2. output ←[]
3. i ← 0
4. while i < maxlen and notEndOfText (output ) do
5. predicted token ← model .predict( input nl)
6. output .append( predicted token )
7. i ← i+1
8. if predicted token =<TOOL > then
9. toolquery ← extract query( output )
10. toolresponse ← call tool(toolquery)
11. output .append( <TOOL >)
12. output .append( toolresponse )
13. output .append( </TOOL >)
14. i ≤ i+length( toolresponse )+2
15. end if
16. end while
17. final text← extract tool responses( output )
18. return final text
19. end procedure
    - Model recognizes <TOOL > marker and summons external tool
    - Deciphers name, input, anticipated outcome for tool request
    - Integrates data from tool into continuing output sequence
    - Text generation completed with external tool results.

### C. Fine tuned dataset construction

**Fine Tuning Datasets for Model Efficacy**

**Principles of Fine Tuning Technology**:
- Adaptive calibration of models using annotated datasets to enhance efficacy
- Securing high-quality datasets is a primary challenge

**Methods for Dataset Acquisition**:
- Deriving from empirical human tool interaction:
  - Exceptional data integrity, congruity with human engagement patterns, relevance
  - Drawbacks: financial onus, restrictive nature of data aggregation
- Using advanced models as instructive archetypes and generating proprietary datasets via LLMs
  - Emblematic implementations: Toolformer, ToolCoder
  - Mitigates labor expenses, expands operational milieus

**Data Quality**:
- Emphasis on data quality is paramount
- Robust measures to augment data caliber

**Challenges in Data Diversity**:
- Ensuring the applications derived are both original and varied
- Traditional methods may result in overlapping or similar applications
- Research by GPT4Tools, ToolAlpaca, ApiBank: diverse datasets enhance model's applicational competencies
  - Ablation analyses show deep influence on enhancing generalization
  - Multimodal data merges image tips with tools to stretch instructional diversity

**Data Augmentation**:
- Complex real-world problem solving requires intricate reasoning sequences and multiple tool engagements
- Existing research scant consideration of model's capacity to generalize complex scenarios
- Skills-in-Context prompting enhances model's aptitude for multi-tool interactions
  - Multi-agent framework emulates real-world tool usage workflow: user agent, assistant agent, tool executor agent
  - Streamlines intricacy of actual API invocations and aligns with experimental approach in real world
- TaskBench interlinks tools based on dependencies, derives diverse subgraphs to guide creation of directives
  - ToolLLM randomly selects tools from identical categories or sets, forms APIs to generate directives

**Conclusion**:
- Fine tuning technology relies on high-quality datasets to enhance model efficacy
- Various methods for acquiring and augmenting datasets are explored

### D. Other issues during the fine-tune phase

**LLMs and Tool Learning:**
- Critical challenges: continuous evolution & preservation of skills (ensuring model's adaptation to new tools while retaining existing ones)
- Traditional fine-tuning techniques lead to catastrophic forgetting
- Mitigating this issue through experience replay

**Expandability of LLMs:**
- Initial methods: distinct token for each tool, learning its embedding (e.g., ToolkitGPT [24])
- Potential drawbacks: increased complexity and potential performance decline as tool inventory expands
- Devising an efficient strategy to handle vast arrays of tools while maintaining model performance

**Evaluation Metrics:**
- Focus on more comprehensive assessment of model performance upon integrating new tools
- Critical evaluation metrics include accuracy, number, and variety of input parameters (often overlooked in studies)
- Input parameters crucial for correct functioning of tool applications

**User Feedback:**
- Integrating user feedback into model training processes essential for ongoing refinement
- Enhancing adaptability and user satisfaction through reinforcement learning techniques (e.g., WebGPT [16])

### E. In context learning without retrieval

**In-Context Learning Without Retrieval**

**Advancements in Intelligent Systems**:
- Large language models (LLMs) have made significant strides in understanding and using external tools
- **In-context learning phase**: Model uses tool instruction documents or sample tasks for few-shot learning
- Notable examples: Hugginggpt [25] and Chameleon [26]

**Challenges in In-Context Learning**:
- As model complexities increase, more extensive and high-quality task demonstrations are required
- **Definitive writing methodology/template for effective task demonstrations**: Applies to form, substance, integration of tools and methodologies

**MultiTool-CoT Method**:
- Introduces a novel approach using chain-of-thought (CoT) prompting [28]
- Enables LLMs to invoke external tools sequentially within appropriate reasoning phases
- Steers the model to execute critical procedural steps and optimize tool usage for task resolution
- Fortifies versatility, adaptability, and problem-solving prowess of the LLM

**Limitations of MultiTool-CoT Method**:
- Constrained assortment of tools, often tailored to specific domains or datasets
- Can impede generalizability, scalability, applicability, and effectiveness for diverse tasks

### F. In context learning with retrieval

**Context Learning with Retrieval:**
- Researchers propose this method to address challenges associated with integration and utilization of multiple tools
- Defined as search conducted based on specific tasks or tools
- Example: Chatcot uses tasks retrieved for decomposition of tasks by language models (LLMs) like TaskMatrix.AI
- Two methods for retrieval: Modularity and Dense Retrieval Model

**Modularity:**
- Categorizes tools, searches for appropriate categories first, then LLM selects specific tools
- Examples: TaskMatrix.AI, Restgpt

**Dense Retrieval Model:**
- Adjusts embedding vectors to find response semantically resonating with query
- Uses text embedding for tool retrieval and integration into plans
- Benefits: Accommodates extensive array of APIs, harnesses LLM's context learning faculties, circumvents errors, enables content updates.

**Effective API Documentation:**
- Clear and precise name avoiding ambiguity
- Parameter list: includes input parameters, return value, parameter description, data type, default value
- API Description: provides more information about what the API does, how it works, inputs & outputs, potential errors or exceptions
- Usage Example: provides examples for API usage.

### G. Other issues during the in context learning phase

**Online Planning in Large Language Models (LLMs)**

**Challenges of In Context Learning**:
- LLMs preliminarily configured with defined action paths display inherent static qualities
- Rigidity can lead to delayed responses and failure due to unforeseen events

**Overcoming Rigidity**:
- **Online Planning**: Every decision considered in the broader context, allowing real-time adjustments based on feedback
  - Allows effective adaptation to dynamic dialogue environments
  - Minimal impact on efficiency, as shown by minimal increase in token consumption
- **Learner Module**: Addresses execution errors through identification and rectification
  - Enhances model's resilience and adaptability

**DFSDT Algorithm for Tree Search**:
- Allows models to explore decision trees more thoroughly
- Consider multiple possibilities at each layer, improving decision-making quality and accuracy in complex situations

### H. LLM creates its own tools

**Large Language Models (LLMs) and Tool Generation**

**Existing Tools for LLMs**:
- Utilized to assist LLMs in task processing

**Autonomous Tool Creation**:
- Recent research proposes innovative paradigm of allowing LLMs to autonomously create and use tools
- Potential demonstrated by "Creator" framework [14] and "Large language models as tool makers" [33]

**The Creator Framework**:
- **Creation stage**: Guides LLM to generate required tools through fixed demonstrations
- **Decision stage**: Utilizes demonstration method to guide LLM in generating code/methods for using new tools
- **Execution stage**: Responsible for executing created tools, requiring LLM to solve specific problems and transform abstract tools into practical operations
- **Recognition stage**: Makes necessary modifications and optimizations to ensure effective serving of the target task
- Proves that tools created by large language models have transferability in similar scenarios

**Tool Reuse Challenges**:
- "Large language models as tool makers" [33] proposes a solution with a **dispatcher module** that searches and filters out suitable tools from past tool caches
- Dispatcher module uses naive context learning method without retrieval function, which may become a significant challenge as the number of tools increases

**Potential Solution: Retrieval System**
- Build a retrieval system containing rich tool information
- When reusing tools is needed, the retrieval system can quickly locate the appropriate tool and provide it to the dispatcher module for selection
- Helps reduce limitation of context length and may improve accuracy and efficiency of tool reuse.

### I. Other methods

**Gear's Efficient Strategy for Tool Selection**

**Semantic Similarity Scoring:**
- Analyzes internal relationship between problem and tool descriptions
- Uses natural language processing technology (word embedding model or deep learning algorithm) to quantify semantic proximity
- Screens out irrelevant toolsets based on semantic similarity

**Pattern Similarity Scoring:**
- Generates preliminary answers based on question nature
- Compares these with output generated by each tool
- Evaluates whether tool's output is consistent in structure, content or format with preliminary answer
- Screens out tools that cannot meet expectations

**Comprehensive Scoring:**
- Combines semantic similarity score and pattern similarity score for each tool
- Ensures selection of most appropriate tool based on both relevance to question and accurate output

**Challenges:**
- High computational cost due to needing to call all tools at once in large-scale data sets
- Limited flexibility for complex scenarios involving multiple problems
- May not cover specific areas or professional issues fully.

## IV. REFLECTION AND OUTLOOK

**Reflections on Research Directions for Large Language Models (LLMs)**

**I. Reflection on Current Research Limitations**
- **Research on tool scheduling topology optimization**: Enabling LLMs to derive optimal tool scheduling topology to generate accurate responses given predefined tools and user queries is a worthwhile research direction.
- **Graph data structures and neural networks**: Studying how to establish bridges between graph data structures, graph neural networks, and large models for auxiliary decomposition and decision-making.
- **Time optimization of model pipelines**: Sequential methods may not be optimal or most suitable for practical use; designing algorithms that can shorten pipeline time without sacrificing accuracy is essential.
- **Continuous learning and optimization**: Adapting to a large number of constantly increasing and evolving tools requires continuous learning and optimization. Achieving plug-and-play without compromising model performance is crucial.
- **Error handling and recovery**: Learning to evaluate API reliability, summarize errors, and prevent cascading errors is an important research direction.
- **Tool usage**: Teaching large models how to effectively use tools remains unresolved; finding the most relevant tools for a given context is also worth exploring.

**II. Future Research Directions**
- **Pretraining tool augmented LLMs**: Instead of fine-tuning pretrained models or relying solely on tool calls, attempting to pretrain tools to enhance LLMs could be an effective approach.

## V. EXPERIMENTS ON CHAMELEON

**Chameleon Experiment using Azure OpenAI Service**

**A. Experiment Settings:**
- Utilized Azure OpenAI Service for accessing various GPT models securely
- Tested on 4241 examples from ScienceQA using CoT and Chameleon methods
- Used GPT-3.5 (gpt-35-turbo-16k-0613) in this experiment

**B. Experiment Results:**
- QA accuracy results presented in Tables V-C and V-D, similar to original with decimal point differences
- Chameleon state transition diagrams (Fig. 5) slightly different from reproduced ones (Fig. 6)
- Call scale diagrams identical (Fig. 7)

**C. Code Structure and Overall Process of Chameleon:**
- Defines a Solver class with various modules: image capturer, bing search, solution generator, etc.
- Proposed pseudocode for main execution process: Algorithm 2
- Loads data, obtains problem text, predicts required modules and their execution, compares predictions to answers, updates statistics, saves results, summarizes final statistical data.
- Builds prompts for Chameleon and Cot modes, processes images and texts using image capturer and text detector respectively.
- Calls search engines using Bing search module, generates answers based on predicted results using solution generator, generates final answers using answer generator.

