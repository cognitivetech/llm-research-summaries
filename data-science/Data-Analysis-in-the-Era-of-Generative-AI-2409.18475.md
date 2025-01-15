# Data Analysis in the Era of Generative AI

Jeevana Priya Inala, Chenglong Wang, Steven Drucker, Gonzalo Ramos, Victor Dibia, Nathalie Riche, Dave Brown, Dan Marshall, Jianfeng Gao
https://arxiv.org/pdf/2409.18475?

## Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Background](#2-background)
- [3. Case study: User experiences with AI-tools for visualization authoring](#3-case-study-user-experiences-with-ai-tools-for-visualization-authoring)
  - [3.1. Example task and the traditional experience](#31-example-task-and-the-traditional-experience)
  - [3.2. Conversational LLM interface (ChatGPT with GPT-4o and CodeInterpreter)](#32-conversational-llm-interface-chatgpt-with-gpt-4o-and-codeinterpreter)
  - [3.3. LLM-powered interactive data analysis tools](#33-llm-powered-interactive-data-analysis-tools)
  - [3.4. Remarks](#34-remarks)
- [4. Opportunities for AI systems in the data analysis domain](#4-opportunities-for-ai-systems-in-the-data-analysis-domain)
  - [4.1. Closing the skills gap: Empowering users in data analysis](#41-closing-the-skills-gap-empowering-users-in-data-analysis)
  - [4.2. Potential AI workflows for the different stages of data analysis process](#42-potential-ai-workflows-for-the-different-stages-of-data-analysis-process)
- [5. Human-driven design considerations for AI-based data analysis systems](#5-human-driven-design-considerations-for-ai-based-data-analysis-systems)
  - [5.1. Enhancing user-AI interaction: For natural intent communication](#51-enhancing-user-ai-interaction-for-natural-intent-communication)
  - [5.2. Facilitating trust and verification: Enhancing model output reliability for users](#52-facilitating-trust-and-verification-enhancing-model-output-reliability-for-users)
  - [5.3. Unified analysis experience: Streamlining data tools and workflows](#53-unified-analysis-experience-streamlining-data-tools-and-workflows)
- [6. Challenges in developing AI powered data analysis systems](#6-challenges-in-developing-ai-powered-data-analysis-systems)
  - [6.1. Ensuring reliability and trust](#61-ensuring-reliability-and-trust)
  - [6.2. System benchmarking and evaluation metrics](#62-system-benchmarking-and-evaluation-metrics)
  - [6.3. Need more advances in models/agents](#63-need-more-advances-in-modelsagents)
  - [6.4. Understanding user preferences and abilities](#64-understanding-user-preferences-and-abilities)
  - [6.5. Data Infrastructure](#65-data-infrastructure)
- [7. Conclusion](#7-conclusion)

## Abstract
**AI-Powered Data Analysis: A New Era of Opportunities**

**Exploring AI's Role in Data Analysis**:
- AI tools can reshape data analysis by enhancing various stages of the workflow
- The emergence of large language and multimodal models offers new opportunities to translate high-level user intentions into executable code, charts, and insights

**Human-Centered Design Principles**:
- Facilitate intuitive interactions with AI systems
- Build user trust through clear explanations and expectations
- Streamline the analysis workflow across multiple apps

**Research Challenges**:
- Enhancing model capabilities to better understand complex data and user needs
- Evaluating and benchmarking AI systems to ensure accuracy and efficiency
- Understanding end-user needs and designing interfaces that cater to their diverse requirements

## 1. Introduction

**Data-Driven Decisions: Democratizing Data Analysis**

**Introduction:**
- Importance of data analysis in various industries and daily life
- Current high cost of data analysis limits accessibility to the general population
- Potential benefits of democratizing data analysis

**Challenges of Data Analysis:**
1. Complex process requiring multiple steps:
   - Task formulation
   - Data collection
   - Exploratory analysis
   - Creating visual representations
   - Documentation as reports
2. Analysts need various skills and expertise:
   - Conceptual knowledge (data sensemaking, domain, statistics, visualization design)
   - Tool expertise and programming skills
3. Overhead switching between tools and managing branching analysis steps
4. Existing systems trade-off between flexibility and ease of use
5. Complexities in data analysis make it a unique challenge for AI models
6. Iterative and multimodal nature of data analysis requires user involvement
7. Importance of accuracy and reliability in AI systems for sensitive domains like healthcare or finance
8. Need for interfaces that facilitate effective human-AI collaboration
9. Research challenges ahead to implement AI-powered data analysis systems:
   - Enhancing existing models' capabilities
   - Addressing scarcity of training and evaluation data
   - Ensuring system reliability and stability
   - Conducting user research for alignment with users' cognitive abilities and practical needs.

## 2. Background

**Data Analysis Process**
- Figure 1 and Table 2 illustrate steps involved in gaining insights from data
- Iterative process: multiple steps, back and forth
- Starts with task formulation: identifying problem/question (e.g., renewable energy trend analysis)
- Involves operationalizing the task into sub-tasks (global trends, country ranking, comparison of trends)
- Collecting relevant data: querying databases, instrumenting apps, web scraping, cleaning, integration from multiple sources
- Data exploration phase: gaining familiarity with dataset, identifying potential patterns or relationships
  - Descriptive statistics (mean, median, count)
  - Visualizing variable relationships through trends and correlations
  - Formulating hypotheses to explore further
  - Transforming data and generating visualization artifacts for hypothesis support
- Validating insights: ensuring reliability and accuracy of findings through statistical analysis, domain knowledge, or external sources
- Refining hypotheses based on validated insights
- Final results: creating reports, dashboards, or presentations to communicate findings to stakeholders
  - Iteratively refining communication based on feedback received

**LLMs, LMMs, and AI Systems**
- Advancements in language models (LLMs) and multimodal models (LMMs)
- Foundation models: ChatGPT, GPT4, Claude, Llama, Phi-3-Vision, GitHub copilot, Dibia, Narayan et al.
- Capable of generating code, understanding algorithms, and solving interview/competition problems
- Generating data transformations and visualization artifacts
- Instruction-tuned to understand and converse with people in a chat environment (Ouyang et al.)
- Agents: interfaces around foundation models, embedding application-specific knowledge through prompts or external tools
  - Code interpreter: generates code, executes it in a sandbox environment, produces further text or code
- Multi-agent systems: series of agents collaborating and responding to user queries (Wu et al., Hong et al.)
- "AI systems": refers to systems powered by latest breakthroughs in LLMs, LMMs, and Generative AI.

## 3. Case study: User experiences with AI-tools for visualization authoring

**Case Study: User Experiences with AI-Tools for Visualization Authoring**

**Data Visualizations**:
- Prevalent in data analysis stages
- Used by data analysts to:
  - Assess data quality issues
  - Explore relations between data fields
  - Understand data trends and statistical attributes
  - Communicate insights from data to the audience

**Challenges of Visualization Authoring**:
- Analysts need to learn visualization tools
- Transform data into the right format for chart design
- AI-powered visualization authoring tools aim to reduce this "authoring barrier"

**Comparison of User Experiences**:
- Traditional programming tools
- Direct conversational interaction with LLMs via chat-based interface
- LLM-powered interactive data-analysis tools

**User Experience Comparison Factors**:
1. **Specifying the Initial Intent**:
2. **Consuming AI System's Output**:
3. **Editing**:
4. **Iterating**:
5. **Verifying AI System's Output**:
6. **Workflow in Data Analysis Context**

### 3.1. Example task and the traditional experience

**Task:** Investigate trend of renewable electricity percentage over time for top 5 CO2 emission countries

**Traditional Approach (Non-AI):**
1. **Data Transformation:**
   - Create new columns: "percentage of renewable electricity"
   - Perform operations like grouping, summing, filtering
   - Complex tasks may require pivoting or unpivoting
2. **Chart Authoring:**
   - Determine chart type and encodings to represent data trends appropriately
   - Create charts using libraries such as Seaborn, Matplotlib, PowerBI, Tableau
3. **Challenges:**
   - Steep learning curve for beginner data analysts and non-programming end-users.

### 3.2. Conversational LLM interface (ChatGPT with GPT-4o and CodeInterpreter)

**Conversational LLM Interface: ChatGPT with GPT-4o and CodeInterpreter**

**Benefits of Conversational LLMs**:
- Simplifies the process for users with varying experience levels
- Allows natural language commands to express intent
- Performs steps such as loading dataset, understanding subtasks, writing code, and rendering final chart
- Provides explanations and code snippets for user understanding

**Limitations of Conversational LLMs**:
- Requires multiple model invocations, leading to long wait times
- Increases likelihood of failure due to repeated invocations
- Difficult to recover from system getting stuck on a step
- Limited expressiveness in natural language intentions (e.g., color preferences)
- Inefficient workflow when using multiple apps with separate copilots

**Conversational Framework**:
- Allows users to edit and iterate through follow-up instructions
- Chat interface allows backtracking and editing from previous conversations
- Users can't track different iterations easily
- System may lose context upon backtracking, requiring recomputation
- Long wait times and unreliable user experiences due to non-deterministic underlying model

**Comparative Analysis**:
- ChatGPT vs Data Formulator: Figure 2 shows the user experience for providing inputs and output formats
- Conversational editing vs. dynamic intent-based UIs in DyNaVis: Figure 3(a) compares conversational editing with data threads in Data Formulator
- Trust and verification of AI outputs: Figure 4 contrasts ChatGPT's explanation feature with Data Formulator's approach.

### 3.3. LLM-powered interactive data analysis tools

**Data Formula-tor**
- **Multi-modal UI**: allows users to specify intent using a combination of field encodings and natural language instructions
- **LLM model**: generates necessary data transformations based on user inputs, creates Vega-Lite specification for chart rendering
- **Chart options**: users can read/edit code, view transformed data table

**DynaVis**
- **Multi-modal UI**: allows users to specify chart editing intents through natural language
- **Dynamic widget generation**: system generates widgets based on high-level intent, reduces latency in the AI process
- **Intent exploration**: users can experiment with various options and receive instant feedback

**Data Formulator2**
- **Iterative exploration**: allows users to modify existing visualizations through multi-modal UI
- **Data threads**: help manage analysis session by organizing user interactions
- **Candidate chart/transformation creation**: enables users to inspect multiple candidates, disambiguate specifications
- **Streamlined workflow**: combines data transformations and visualization authoring into a single tool

**UFO**
- **UI-focused agent**: tailored for applications on Windows OS using GPT-Vision
- **Seamless navigation and operation**: employs models to observe and analyze GUI/control information
- **Simplifies complex tasks**: streamlines the data analysis workflow.

### 3.4. Remarks

**Comparative Analysis of AI Tools for Visualization Authoring**

**Observations**:
- Design of tools impacts users' ability to create desired visualizations
- Critical for visualization authoring as well as other stages of data analysis

**Design Considerations**:
- Section 5 will elaborate on these design principles and explore strategies beyond those examined in the case study
- To alleviate user specification burdens, enhance trust and verification strategies, and facilitate workflows across multiple stages and tools

**Additional AI Opportunities**:
- Section 4 will investigate additional AI opportunities within the broader data analysis landscape, extending beyond visualization authoring.

## 4. Opportunities for AI systems in the data analysis domain

**Data Analysis Opportunities for AI Systems**

**Challenges of Data Analysis Workflows:**
- Complex reasoning tasks
- Iterative processes
- Diverse skills required: coding, domain knowledge, scientific methodologies

**Opportunities for AI Systems:**
1. **Streamlining and enhancing workflows:**
   - Automating repetitive tasks
   - Enhancing complex reasoning
2. **Identifying areas of impact:**
   - Dissecting the analysis process into sub-tasks
   - Evaluating intermediate outputs
3. **Establishing user requirements:**
   - Reliability and effectiveness of AI systems
4. **Literature review:**
   - Preliminary overview for future research
5. **Limitations:**
   - Not exhaustive opportunities or related works

**Goal:**
- To explore how AI can help navigate data analysis complexities.

### 4.1. Closing the skills gap: Empowering users in data analysis

**Closing the Skills Gap: Empowering Users in Data Analysis**

**Challenges:**
- Diverse skill set required for effective data analysis
- Non-experts face barriers to engagement due to lack of skills

**Opportunities with Low/No Code Experiences:**
1. Alleviating coding burden for non-programmers
   - LLMs generate code snippets for various sub-stages
2. Improving data analysis tools and platforms
   - Natural language interfaces
3. Offering domain knowledge support and statistical expertise
4. Automating data cleaning, transformation, querying, and visualization
5. Enhancing user experience with interactive decision support

**Examples of LLMs for Data Analysis:**
- LIDA: Goal-based data visualization generation (open-source)
- ChatGPT: Code interpreter API for generating plots from natural language queries
- Chat2Vis, ChartGPT: Generating code snippets for data analysis tasks
- DataFormulator: Allows users to specify visualization intent through drag and drop or natural language
6. Statistical Assistance
   - Helping users select appropriate statistical tests based on tasks and data
   - Avoiding common pitfalls in statistical analysis
7. Domain Knowledge Support
   - Automating domain-specific contextual understanding of the data and task
8. Tool Copilots
   - Lowering entry barrier for engaging in data analysis tasks with tools
   - Intelligent tutors guiding users to familiarize themselves with complex software interfaces
9. Extending AI assistance to a broader range of applications
10. Design Considerations: Enabling OS-Agent communication between apps for a more streamlined experience.

### 4.2. Potential AI workflows for the different stages of data analysis process

**Data Analysis Process Stages and AI Workflows**

**Potential AI workflows for different stages of data analysis:**

**1. Task Formulation**
- Refine high-level goals into specific tasks
  * Address cold-start problem with domain knowledge
    * LIDA's goal explorer module generates hypotheses (Dibia, 2023)
    * InsightPilot issues concrete analysis actions (Ma et al., 2023)
    * NL4DV Toolkit breaks down natural language queries into attributes, tasks, and visualizations (Narechania et al., 2020)
- Draw inspiration from existing examples for new tasks
  * Identifying similar datasets and questions (Bako et al., 2022)
  * Retrieval Augmented Generation (RAG) approaches use external knowledge (Lewis et al., 2020; Izacard et al., 2022; Peng et al., 2023; Lu et al., 2024)
  * Few-shot learning with similar examples (Poesia et al., 2022; Li et al., 2023b; Liu et al., 2021)

**Reducing Biased Data Analysis with AI Systems**
- **Multiverse analysis**: examining various analytical approaches to reduce risk of biased conclusions
- **Boba (Liu et al., 2020)**: early pre-LLM system for simplifying multiverse analysis in data science
- **Automated hypothesis exploration**: beneficial but poses significant challenges due to potential bias and exponential hypothesis space
- **Execution + Authoring**: generating code for data transformations and configuring charts for visualization
- **Validation, insight generation, and refinement**: inspecting analysis artifacts to validate assumptions, extract insights, and iterate hypotheses
- **Vision language models (LLMs)**: analyzing images, charts, and figures to reason about visualizations and generate insights
- **User verification strategies**: users may need to resolve ambiguities or misunderstandings in AI-generated output

**Data Discovery with AI Systems**
- **Assumption of accessible data**: not always the case as new analysis may require additional data
- **Dataset location**: AI systems can help users locate necessary datasets for deeper, more accurate analysis
- **LLMs for semantic search in data lakes**: extending beyond traditional keyword/tag-based search methods
- **Interacting with web APIs and search engines**: dynamically extracting real-world data to enhance reasoning and analysis.

**LLM-Based Approaches for Data Analysis**

**Enhanced capabilities**:
- Enhanced data extraction
- Knowledge synthesis
- Help with data cleaning and integrating multiple datasets
- Handling data formatting issues, resolving inconsistencies, merging datasets

**Extracting structured data from non-tabular formats**
- LLMs can analyze text data to extract sentiments and trends
- Recent works on converting unstructured data to structured data, particularly in medical domain
- Examples: HeyMarvin for qualitative data analysis

**Assisting in report generation and creativity**
- Facilitating the generation of dashboards and what-if analyses with minimal developer involvement
- Transforming static elements into interactive versions
- Tailoring content and format of reports to suit various audiences and devices
- Creating artistic and aesthetically pleasing charts and infographics using GenAI models like DALL-E and Stable Diffusion

**Challenges for AI systems in creating interactive designs**:
- Linking various forms of media (text, images, videos, animations)
- Employing diverse interaction techniques (details-on-demand, drag-and-drop functionality, scroll-based interactions, hover effects, responsive design, gamification)

## 5. Human-driven design considerations for AI-based data analysis systems

Error: Output length exceeds input length after 5 attempts.

### 5.1. Enhancing user-AI interaction: For natural intent communication

**Enhancing User-AI Interaction: Natural Intent Communication**

**Challenges of Relying on Natural Language Based Interfaces:**
- Users face challenges expressing complex intents due to limited expressiveness
- Lengthy or difficult natural language communication required for some tasks (e.g., altering legend placement)

**Potential Ways AI Systems Can Enable Natural Modes of User Communication:**
1. **Multi-modal inputs:**
   - Other modalities like mouse-based operations, pen and touch, GUI based interactions, audio, and gestures provide more powerful and easier ways to communicate intent
   - Examples: Data Formulator, DirectGPT, InternGPT, ONYX
2. **Direct manipulation:**
   - Users can directly manipulate visualizations or UI elements to convey their intent (sketching, inking)
   - Input-output examples and programming by example for data transformations
3. **Audio input:**
   - Another representation of chat that is more powerful with demonstrations or user manipulations
4. **Multimodal interactions**
   - Involve both the user and AI system taking the initiative as appropriate
5. **Personalized and dynamic UIs:**
   - Generative capabilities of language models enable customized interfaces on-the-fly for various tasks (e.g., DynaVis, Stylette)

### 5.2. Facilitating trust and verification: Enhancing model output reliability for users

**Facilitating Trust and Verification: Enhancing Model Output Reliability for Users**

**GenAI Models**:
- Can produce unintended outputs due to model hallucinations or ambiguous user input
- Require users to verify the accuracy of their outputs
- The verification process should not impose a significant cognitive load on users

**Co-Audit Tools**:
- Aim to assist users in checking AI-generated content
- Provide code explanations: natural language, or step-by-step natural language utterances
- Generate multiple possible charts/data for the same user input to disambiguate
- Inspect and verify calculated columns without viewing underlying code (e.g., ColDeco)
- Show detailed visual transformation process with interactive widgets to support error adjustments

**Multi-Modal and Interactive Outputs**:
- Format of output significantly influences user's understanding and verification ability
- A multimodal document interlaced with text and images is more readable than lengthy code or text
- Interactive charts and what-if analyses allow users to quickly experiment and verify outcomes

**Debugging Support for Users**:
- AI systems consisting of multiple models, agents, and components can be challenging for end users to debug
- Providing real-time visibility into the actions of these agents, along with support for checkpointing and interruptibility, would allow users to provide feedback and resume
- Tools like AutoGen Studio orchestrate multiple agents, providing a no-code interface for authoring, debugging, and deploying multi-agent workflows

### 5.3. Unified analysis experience: Streamlining data tools and workflows

**Streamlining Data Analysis Tools: Unified Analysis Experience**

**Current Data Analysis Workflow**:
- Multiple tools like Excel, PowerBI, Jupyter Notebooks, SQL servers are used for different aspects of data analysis
- Overhead in switching contexts, transferring data, and maintaining intent across platforms

**Unifying Capabilities**:
- Unifying multiple capabilities into a single tool can streamline the analysis process
- Enables unified AI system to seamlessly debug and reason across steps of the process
- Examples: **Data Formulator**, which combines data transformation with visualization authoring; **LIDA**, which integrates data summarization, goal exploration, chart authoring, and infographics generation

**Multi-agent Systems**:
- Allow creating AI systems composed of multiple agents, each specialized in a particular capability
- Enables the system to plan and collaborate on complex tasks
- Promising in producing complex AI tools in various domains like scientific reasoning and software engineering

**Blended Apps**:
- Modern applications are starting to create all-in-one workbenches
- Allows for automatic transitions between different AI/non-AI based tools
- Enhances user experience by integrating the context of existing non-AI systems

**OS Agents**:
- Automatically control and interact with multiple applications on the desktop
- Perform tasks across multiple applications based on natural language commands from the user
- Promising for enabling a unified analysis experience, especially for applications that cannot be controlled via APIs or co-pilots.

## 6. Challenges in developing AI powered data analysis systems

**Challenges in Developing AI-Powered Data Analysis Systems**
* **Design Considerations**: Enhancing human-AI experiences in data analysis tools (Figure 6)
* **Research Challenges**: Developing effective AI systems incorporating these principles
* **Challenges**:
  * **Multimodality, Planning, and Personalization**: New models needed to support these capabilities
  * **User Preferences**: Understanding users' preferences in data analysis
  * **Data Infrastructure**: Creating infrastructure for AI-driven suggestions
  * **Model Reliability**: Enhancing the reliability of existing AI models
  * **System Benchmarking and Evaluation Metrics**: Establishing robust metrics to measure system performance
* **Contextualization**: Problems faced in data analysis domain (not just LLM-based interactive systems)

### 6.1. Ensuring reliability and trust

**Improving Reliability of AI-Based Data Analysis Systems**

**Potential Issues**:
- Hallucinations (Tonmoy et al., 2024)
- Sensitivity to prompts (Sclar et al., 2023)
- Failure to follow instructions (Liu et al., 2024b)
- Lack of acknowledgment of uncertainty (Key et al., 2022)
- Biases (Gallegos et al., 2023)
- Susceptibility to table representation in prompts (Singha et al., 2023)

**Improving Reliability**:
- Ensuring output satisfies specification:
  - Using correct API
  - Syntax error free

- Grounding techniques:
  - RAG (Retrieval-Augmented Generation) and GPT-4 Assistants API
  - Use of tools like calculator and custom functions to reduce errors

- Verifying code accuracy:
  - Input-output examples provided by user or automatically generated but verified by the user
  - Self-repair and self-rank mechanisms for models to evaluate their own outputs

**Handling Failure Cases**:
- Provide fallback options or have model intelligently request additional information from user
- Mitigate disruptions caused by errors:
  - Incorporating static UIs or dynamically composing from static UI elements
  - Error identification and correction mechanisms

**Ensuring Stability and Integrity of Analysis**:
- Identify hallucinations and ambiguous user intents:
  - Sample multiple outputs and examine agreement to detect inconsistencies

- Predictability, Computability, and Stability (PCS) framework:
  - Evaluate trustworthiness of results by ensuring consistency when slight perturbations are applied

- Probe model to consider alternative analyses and reason over them to determine most meaningful analysis given user intents, task, and domain

### 6.2. System benchmarking and evaluation metrics

**Data Analysis Benchmarking and Evaluation Metrics**

**Desired Evaluation Benchmarks**:
- Comprehensive benchmark suite for data analysis models and systems
- Covers a broad range of data sources and tasks within the domain
- Includes low-level and high-level tasks spanning multiple steps

**Existing Benchmarks**:
- Focus on specific aspects of the data analysis pipeline (e.g., visualization authoring, data transformations)
- Often use different evaluation metrics and procedures
- Limited ability to test all aspects of AI systems in the domain
- Lack benchmarks for open-ended exploration and iterative reasoning

**Need for New Benchmarks**:
- Reveal limitations of current models in areas like multi-modal reasoning, planning, exploration
- Drive AI researchers to push boundaries in new directions
- Compile diverse data tables, sources, questions/goals, and tasks into a centralized location
- Use taxonomy of tasks categorized by complexity for humans and models

**Evaluation Metric Challenges**:
- Difficulty measuring correctness of generated outputs for complex artifacts like charts or UIs
- Supporting multi-modal forms of communication in evaluation
- Assessing partial correctness and orthogonal dimensions of performance

**Behavioral Evaluation**:
- Importance of evaluating user trust, not just output correctness
- Considering worst-case metrics and adversarial testing

### 6.3. Need more advances in models/agents

**Further Advancements in Models/Agents for Multi-Modal AI Systems in Data Analysis Domain**

**Need for Advancements:**
- To realize multi-modal, iterative, and trustworthy AI systems for data analysis domain

**Inference Cost vs. Efficiency:**
- GPT-4 offers powerful capabilities but comes with high inference costs
- Smaller models like Llama and Phi-3 may not consistently produce reliable outputs
- Balancing efficiency and accuracy needed through advancements in smaller language models

**Training Data:**
- Many latest models lack training on data analysis-specific tasks or languages (e.g., R, VBA scripts)
- Insufficient training data for UI interactions (e.g., Excel button clicks) crucial for generating personalized UIs
- Difficulty in finding high-level user intents to low-level data analysis tasks mapping

**Finetuning and Multi-Agent Systems:**
- Fine-tuning AI models on smaller datasets can help overcome lack of training data problem
  * Understanding table/data semantics, generating domain-specific code/actions
  * Finetuning multi-agent systems over entire data analysis workflow
- Scarcity of ground truth data for fine-tuning and automated ways to gather it using RLAIF (using AI feedback)

**Personalization and Continual Learning:**
- Enhancing user experience through learning preferences, memory storage, and adapting to users over time
- Leveraging data from other users and offering tailored recommendations
- Predicting next steps for users similar to current recommendation systems
- Self-evolving AI systems that can learn and create new APIs optimized for common queries

**Multi-Modal Reasoning:**
- Data analysis requires reasoning across multiple modalities: code, text, speech, gestures, images, tables
- Improvements in multi-modal models like GPT-4, Llava, Phi-3-Vision but challenges remain, such as understanding complex charts or interpreting user gestures (e.g., drawing an arrow between columns)
- Combining reasoning across multiple modalities to improve performance

**Planning and Exploration:**
- Data analysis not a linear process; requires planning, reasoning, and exploration
- LLM-based planning shown better results when collaborating with humans rather than functioning autonomously
- Need for further advancements in both LLM capabilities and human-AI collaboration for effective planning in data analysis tasks.

### 6.4. Understanding user preferences and abilities

**Understanding User Preferences and Abilities for AI Systems**

**Importance:**
- Tailor experiences to meet user preferences and abilities
- Improve interaction and usability of interactive AI systems

**Formative Studies:**
- Inform design process: Gu et al. (2024b), McNutt et al. (2023)
- Assess system performance: Wang et al. (2023a), Vaithilingam et al. (2024)

**User Interaction Preferences:**
- GUI + NL approach more effective than chat-based AI assistants: Data Formulator study
- Participants preferred AI-generated widows over direct actions: DynaVis tool study
- Frequent UI changes increase cognitive load: Vaithilingam et al. (2024)

**User Preferences on AI Suggestions and Outputs:**
- Varying levels of statistical expertise influence reactions to AI suggestions
  - Some find them helpful, others overly basic or requiring significant effort
- Users need both procedure-oriented artifacts and data artifacts for understanding analysis steps and confirming details: Gu et al. (2024b)
- Users want to understand and control context given to LLM model: McNutt et al. (2023)
  - Prefer linter-like assistants that highlight inappropriate usage

**Future Research Directions:**
- Deduce contextual preferences based on user history, past actions, and evolving tasks
- Personalize UI elements for individual users without overwhelming them with frequent changes
- Understand user intervention interfaces in dynamic multi-application environments
  - User's ability to understand/manage AI-context across multiple apps
- Trust and verification with output artifacts that might span multiple apps.

### 6.5. Data Infrastructure

**Challenges for AI Systems' Data Infrastructure:**
- **Ensuring availability of high-quality data tables**: for various domains to provide initiative analysis suggestions
- **Search engine infrastructure**: necessary for crawling, indexing, and ranking data tables on the internet
- **Domain experts' involvement**: creating APIs, real-time updates using online data, managing enterprise and proprietary data tables effectively
- **Evaluating and ranking data tables**: through crowd-sourcing or other mechanisms to enhance reliability and relevance of available resources
- **Data privacy, security**: significant research area; anonymizing and aggregating data is essential but beyond the scope of this work.

## 7. Conclusion

**Generative AI Tools for Data Analysis: Potential and Challenges**

**Introduction**:
- Explores generative AI's potential in unlocking insights from data
- Outlines tasks that AI systems can assist with, some with related works and inspiration for others

**Tasks that AI Systems Can Assist With**:
- Data cleaning and preprocessing
- Feature engineering and selection
- Model training and selection
- Interpretation of results

**Design Considerations**:
- Optimizing user interactions
- Enhancing workflows
- Building trust and reliability

**Design Enhancements for AI Systems**:
- Improving robustness in data analysis scenarios
- Developing benchmarks and evaluation metrics

**Model Advancements Needed**:
- Continual learning
- Multi-modal reasoning
- Planning

**Benchmarking Model Innovations**:
- Using multi-step, multi-modal interactive data analysis scenarios

**User Studies**:
- Discerning preferences of data analysis users
- Optimizing AI-driven tools for usability

**Conclusion**:
- Bridging the gap between complex data and actionable insights

