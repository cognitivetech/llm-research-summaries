# Agentic Information Retrieval

Weinan Zhang, Junwei Liao, Ning Li, and Kounianhua Du

https://arxiv.org/html/2410.09713

## Abstract

**Agentic Information Retrieval (Agentic IR)**

**Background:**
- Since 1970s: domain-specific architectures for information retrieval
- Improvements with modern IR systems and web search engines
- Core paradigm unchanged: filtering predefined candidate items
- Introduction of large language models (LLMs) in 2022 transforming information access
- New technical paradigm for Agentic IR

**Agentic IR Overview:**
- Expands scope of accessible tasks
- Leverages techniques to redefine information retrieval

**Applications:**
1. **Cutting-edge applications**: TBA
2. Central information entry point in future digital ecosystems

**Challenges:**
TBD (To Be Determined)

**Discussion:**
- Agentic IR shaped by capabilities of LLM agents
- Transforms how information is accessed
- New paradigm for information retrieval.

## 1 The Trends of IR

**Information Retrieval (IR)**
- Refers to tasks or techniques of finding information items matching user's needs from a large corpus
- Wide range of applications: web search, recommendation systems, online services

**Traditional IR Architecture**:
- Employs specialized architecture for retrieving, ranking, and selecting information items based on query
- Web search engines use inverted index system to maintain posting list of documents for each term
- Given a query, candidate documents are retrieved using the inverted index and ranked using a scoring function
- Top-ranked documents presented on SERP

**Personalized Recommender Systems**:
- Involve retrieval, pre-ranking (optional), ranking, and re-ranking stages to filter items and present top recommendations to user

**Limitations of Traditional IR**:
- Predefined architecture with fixed information flow
- Difficult to perform interactive or complex tasks
- User unable to manipulate information items during the IR process

**Agentic Information Retrieval (Agentic IR)**:
- Novel paradigm for next-generation IR techniques
- Differentiated aspects:
  - Task scope: agent takes actions to reach user's desired information state
  - Architecture: unified architecture employing AI agent across various scenarios
  - Key methods: prompt engineering, retrieval-augmented generation, fine-tuning with supervised and reinforcement learning, multi-agent systems

**Formal Presentation of Agentic IR**:
- Task formulation
- Architecture form
- Key methods

**Applications of Agentic IR**:
- Life assistant
- Business assistant
- Coding assistant

**Challenges in Agentic IR**:
- To be discussed in Section 4

**Conclusion**:
- Introducing the concept of agentic IR as a next-generation IR architecture for more complex tasks.

## 2 Agentic IR
### 2.1 Task Formulation

**Core Components**
- **s\*** = target information state (desired result)
- **x(s\*)** = user's instruction text
- **π(at|x(st))** = agent's policy function
- **st** = state at time t
- **at** = action at time t

**Process**
- User inputs target description
- Agent takes actions via policy
- Environment transitions: **p(st+1\|st,at)**
- Terminates at state **sT**
- Success measured by **r(s*,sT)**

**Objective**
- Maximize: **maxπ 𝔼s*[r(s\*,sT)]**
- Subject to state/action transitions from t=1 to T-1

### 2.2 Architecture

**Agent Policy (π)**
- Conditional on user's language instruction: x(st)𝑥subscript𝑠𝑡subscriptℎ𝑡\pi(a\_t|x(s\_t))
- Interacts with environment with single or multiple turns
- Results in information state: x(st)𝑥subscript𝑠𝑡

**Inner Architecture Modules**
- Memory: stored history and experiences
  * Log history, experience
  * Stored in disk
- Thought: information in context window of LLM

**External Tools**
- Function that cannot be replaced by neural net model
  * Web search engine, relational DB, real-time weather app, calculator, etc.

**Textual Description of Information State (x(st))**
- Depends on current state st, memory ht, and thought Tht, tool Tool: x(st) = g(st, ht, Mem, Tht, Tool)
  * Mem, Tht, Tool update memory, manipulate thoughts, call tools, respectively

**Composite Function (g)**
- Takes current state st and memory ht as raw input
- Outputs intermediate representation of state st for further processing by LLM

**Design Determinants**
- Specific design of g directly determines agent architecture along with used LLM.

**Framework Instantiation**
- Architecture built in a unified way using DAAG over three functions.
  * Previous study: Christianos et al. (2023)

### 2.3 Key Methods

**Improving Agentic Information Retrieval (IR)**

**Key Methods:**
- **Prompt engineering**: setting input to enable task performance (Liu et al., [2023](https://arxiv.org/html/2410.09713v1#bib.bib8))
  * Human-controllable way for hidden state
  * Chain-of-thought prompting
- **Retrieval-augmented generation (RAG)**: using demonstrations to refine actions and information states (Zhou et al., [2024](https://arxiv.org/html/2410.09713v1#bib.bib9))
  * Demonstrations on action level or thought level
- **Reflection**: learning from failures to update thoughts for better interactions (Shinn et al., [2024](https://arxiv.org/html/2410.09713v1#bib.bib10))

**Fine-tuning Methods:**
- **Supervised fine-tuning (SFT)**: adapting LLMs to agentic IR tasks using successful historic trajectories as training data (Liu et al., [2023](https://arxiv.org/html/2410.09713v1#bib.bib8))
  * Behavioral cloning imitation learning methods in RL
  * Does not directly optimize objective
- **Preference learning**: fine-tuning LLMs based on preference objective over a pair of outputs (Rafailov et al., [2024](https://arxiv.org/html/2410.09713v1#bib.bib11))
  * Similar to pairwise learning to rank techniques in traditional IR
- **Reinforcement fine-tuning (RFT)**: optimizing objective with reward signal from environment or human feedbacks (Schulman et al., [2017](https://arxiv.org/html/2410.09713v1#bib.bib13); Silver et al., [2018](https://arxiv.org/html/2410.09713v1#bib.bib14))
  * Requires larger computational resources for exploration and updates

**Advanced Methods:**
- **Complex reasoning**: performing task planning and complex reasoning before taking actions (OpenAI, [2014](https://arxiv.org/html/2410.09713v1#bib.bib16))
  * Strong reasoner for improving agent's performance
- **Reward modeling**: crucial to enable RFT or search-based decoding techniques (Uesato et al., [2022](https://arxiv.org/html/2410.09713v1#bib.bib18); Luo et al., [2024](https://arxiv.org/html/2410.09713v1#bib.bib19))
  * Outcome reward models and process reward models are essential modules for high-performance math agents
- **Multi-agent systems (MAS)**: containing multiple homogeneous or heterogeneous agents that manage to coordinate and achieve collective intelligence (Chen et al., [2023](https://arxiv.org/html/2410.09713v1#bib.bib20); Li et al., [2024a](https://arxiv.org/html/2410.09713v1#bib.bib21))

## 3 Application Scenarios and Case Studies

**Brief Discussion of Three Types of Applications: Life Assistant, Business Assistant, and Coding Assistant**

1. **Life Assistant**: Agent Information Retrieval (IR) functions as an autonomous assistant for users in this application.
2. **Business Assistant**: Similar to the life assistant, the IR acts as an autonomous assistant for users in a business setting.
3. **Coding Assistant**: The IR operates autonomously to assist users in coding tasks.
4. Traditional Information Retrieval: A non-autonomous tool that is used to call upon agent IR.

### 3.1 Life Assistant

**Life Assistants: Evolution and Agentic Information Retrieval (IR)**

**Background:**
- Voice-activated tools transformed into sophisticated systems
- Significant advancement in IR technologies

**Agentic IR:**
- Empowers life assistants to gather, deliver info & proactively support tasks
- Understands user needs, context, preferences
- Acts as active, autonomous agents adapting to lifestyle

**Applications:**
- Apple Intelligence: enhances user experience, seamlessly integrates with devices and services ([Apple](https://www.apple.com/apple-intelligence/))
- Google Assistant, Amazon Alexa, Oppo Breeno, Huawei Celia: operate across diverse platforms (Google [Assistant](https://assistant.google.com/), [Amazon Alexa](https://www.alexa.com/), [Oppo Breeno](https://consumer.huawei.com/en/emui/celia/),)

**Benefits:**
- Convenient control over digital and physical environments
- Proactive, contextual assistance

**Scenario: Jane's Daily Life with Agentic IR**
1. Anticipates needs & gathers info: traffic conditions, suggests earlier departure time
2. Modular design: memory (context), manipulate thought (processing preferences), tools (external sources)
3. Adaptation: refines understanding from explicit queries and passive contextual cues
4. Autonomous task execution: books a dinner reservation or sets reminders
5. Seamless integration across devices and services: unifies various applications, ensures alignment between physical environment and personal schedule.

**Agentic IR Characteristics:**
- Proactive information gathering & state transition
- Adaptive through contextual understanding & interactive refinement
- Autonomous task execution & final information states
- Seamless integration across devices and services.

### 3.2 Business Assistant

**Business Assistant for Enterprise Users:**
- Designed to support business knowledge and insights from various documents and data sources
- Uses agentic IR capabilities for intention recognition and response generation
- Addresses wide range of business queries: financial analysis, marketing strategies, decision making
- Four stages in the workflow: query understanding, document retrieval, information integration, response generation

**Query Understanding:**
- Attempts to understand user's intention from a business-related query
- Generates thoughts with CoT for complex queries and multi-step reasoning
- Leverages historical dialogues as memory to better understand context and intent

**Document Retrieval:**
- Retrieves relevant information from external and internal documents
- Utilizes tools like OCR, SQL for diverse document formats
- Semantic search capabilities ensure alignment with query intent

**Information Integration:**
- Combines and condenses scattered information before responding to the query
- Uses thoughts or tools to generate cohesive responses or complete tasks
- RAG framework by default in systems like Amazon Q Business for response generation

**Response Generation:**
- Generates a response in various formats: plain text, tables, visualized charts
- Completes tasks and returns action states
- Links answer back to source documents for transparency

**Application of Business Assistant:**
- Continuous evolution with advancements in agentic IR and increasing market demand
- Enhanced contextual understanding and multi-step reasoning for complex instructions
- Retrieves information from ever-updating sources in continuous business scenarios
- Security concerns include protection of internal enterprise data and ensuring safe responses.

### 3.3 Coding Assistant

**Interactive Programming Assistance and Automatic Program Synthesis**
- **Productivity and development efficiency improved** by:
  - **Copilot**: interactive environment for developers to gather information from open world and meet programming needs (GitHub)
  - **Agentic Information Retrieval**: systems designed to autonomously retrieve and provide relevant information based on developer queries and contextual needs

**Developer-Coding Assistant Interaction Process**
1. **Information Need Diagnosis**:
   - Developers' information need can be:
     - Conscious (explicitly input requirements)
     - Unconscious (automatically identified by the coding assistant)
   - Agentic IR offers timely and tailored knowledge assistance due to:
     - Memory module that remembers previous interactions, preferences, queries, debugging histories, and coding projects
2. **Knowledge Content Generation**:
   - After information need is identified, the coding assistant queries for corresponding knowledge content using an intelligent large language model (e.g., OpenAI CodeX)
   - Integrated with various coding tools (debuggers, compilers, linters) to provide reliable and non-parametric knowledge
   - Examples: generating codes, synthesizing code completions, test generation, compiler feedback
3. **Information State Update**:
   - Developer perceives the generated knowledge content and refines their work, leading to an updated information state
   - A new round of interaction is then activated, with developers gathering timely, tailored, and evolving information from the coding assistant
4. **Qualified Code or Project Accomplishment**:
   - Developers reach a final information state (sT) where they have accomplished a qualified code or project

## 4 Challenges

**Challenges in Agentic Interactive Reasoning (IR)**

**Data Acquisition:**
- Logged data from agent's interaction with environment
- Determined by user instructions, agent policy, and environment dynamics
- Exploration-exploitation tradeoff crucial for high-quality data collection
- Direct labeling of correct trajectories expensive and challenging

**Model Training:**
- Agent policy consists of DAG of functions: memory update, thought manipulation, tool use
- Effectively updating parameters of these functions and composite policy function highly challenging
- Recent attempts using RFT (Christianos et al., 2023) and action decomposition (Wen et al., 2024)

**Inference Cost:**
- Large parameter size and autoregressive nature increase GPU requirements and processing time for LLMs
- System optimization crucial for practical service deployment

**Safety:**
- Agent directly interacts with real environment, changing it and user's information states
- Important to guarantee safety across user journey
- Alignment techniques (Ji et al., 2023) helpful but not guaranteed
- Proposed "world model + verifier" framework (Dalrymple et al., 2024) can explore safety for agentic IR

**Interacting with Users:**
- Product form of agentic IR still under-explored due to differences from traditional IR in aspects like inference latency, data manipulation, and information state representation.

## 5 Conclusions
- Proposed new paradigm in IR: Agentic IR
- Differentiates from traditional IR by interacting with environment to reach user's target information state
- Agentic IR serves a wide task scope, uses unified agent architecture, and employs distinct key methods compared to traditional IR
- Challenges exist but expected development and promotion in upcoming years
