# Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?

by Ruisheng Cao, Fangyu Lei, Haoyuan Wu, Jixuan Chen, Yeqiao Fu, Hongcheng Gao, Xinzhuang Xiong, Hanchong Zhang, Yuchen Mao, Wenjing Hu, Tianbao Xie, Hongshen Xu, Danyang Zhang, Sida Wang Ruoxi Sun, Pengcheng Yin, Caiming Xiong, Ansong Ni, Qian Liu, Victor Zhong, Lu Chen, Kai Yu, Tao Yu 
https://arxiv.org/pdf/2407.10956

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Executable Computer Environment of Spider2-V](#2-executable-computer-environment-of-spider2-v)
  - [2.1 Task Definition Generally, an autonomous data agent is modeled as a partially observable Markov decision pro- cess (POMDP). Given the current obs](#21-task-definition-generally-an-autonomous-data-agent-is-modeled-as-a-partially-observable-markov-decision-pro--cess-pomdp-given-the-current-obs)
  - [2.2 Environment Setup ORlocalcloud emptyclouddatabaseAPIcalls datauploading(a)FileTransfer(b)ApplicationLaunch(c)RemoteAPICalls(e)PlaywrightAutomatio](#22-environment-setup-orlocalcloud-emptyclouddatabaseapicalls-datauploadingafiletransferbapplicationlaunchcremoteapicallseplaywrightautomatio)
  - [2.3 Task-specific Evaluation](#23-task-specific-evaluation)
- [3 Benchmark Construction](#3-benchmark-construction)
  - [3.2 Document Warehouse](#32-document-warehouse)
- [4 Experiments and Analysis](#4-experiments-and-analysis)
  - [4.2 Main Results](#42-main-results)
  - [4.3 Analysis](#43-analysis)
- [5 Related Work](#5-related-work)
- [6 Conclusion](#6-conclusion)
- [A. Relevant URLs](#a-relevant-urls)
- [B Checklist of All Professional Software in Spider2-V](#b-checklist-of-all-professional-software-in-spider2-v)
- [C Details of Document Warehouse](#c-details-of-document-warehouse)
- [D Details of Executable Environment in Spider2-V](#d-details-of-executable-environment-in-spider2-v)
- [E Format of Task Examples](#e-format-of-task-examples)
- [F Task Examples](#f-task-examples)
- [G Prompts for Multi-modal Agents](#g-prompts-for-multi-modal-agents)
  - [G.1 System Prompt](#g1-system-prompt)
  - [G.1.2 Action Space Prompt](#g12-action-space-prompt)
  - [G.1.3 Overall System Prompt](#g13-overall-system-prompt)
  - [G.2 Task Prompt](#g2-task-prompt)
  - [G.3 Example of Retrieved Context Augmented Task Prompt](#g3-example-of-retrieved-context-augmented-task-prompt)

## Abstract
**Background:**
- Data science and engineering workflows span various stages using tools like BigQuery, dbt, Airbyte
- Vision language models (VLMs) may automate these workflows with SQL queries, Python code, GUI operations
- Improves productivity of experts, democratizes large-scale data analysis

**Spider2-V:**
- First multimodal agent benchmark for professional tasks in authentic environments
- Includes 494 real-world tasks and 20 enterprise-level applications
- Tasks evaluate ability to write code, manage GUI in data software systems

**Evaluation:**
- Balances realism with simplicity through automatic configurations and careful metrics
- Provides comprehensive documents for enterprise data software systems
- Reveals existing LLM/VLM agents underperform in fine-grained, knowledge-intensive tasks (16.2%) and cloud-hosted workspaces (10.6%)

**Significance:**
- Paves way for autonomous multimodal agents to transform data science and engineering workflows

**Resources:**
- Code and data available at https://spider2-v.github.io

## 1 Introduction

**Spider2-V: The First Multimodal Agent Benchmark for Data Science and Engineering Workflows**

**Introduction:**
- Data science and engineering pipelines rely on professional software systems (e.g., BigQuery, dbt, Airbyte) for data processing
- Writing code and using GUI controls in these systems can be complex even for experienced data scientists and engineers
- Rapid advances in language models (LLMs) and vision language models (VLMs) offer potential for automating workflows

**Challenges:**
- Previous studies focused mainly on daily life data processing using code or API calls, neglecting other stages like data ingestion and integration
- Data scientists and engineers often need to navigate multiple professional applications, combining code writing with intensive GUI controls
- No benchmark integrates both code generation and GUI controls for professional data science and engineering workflows

**Proposed Benchmark: Spider2-V**
- Covers the entire data science and engineering workflow (data warehousing, ingestion/integration, transformation, analysis/visualization, orchestration)
- Involves 494 real-world tasks in a real-time executable computer environment and 20 professional enterprise data software systems
- Aims to evaluate multimodal agents' ability to perform tasks by writing code and managing GUI in these applications

**Benchmark Design:**
- Based on OS-WORLD [34], allowing agents to simulate human actions (typing code or clicking buttons) in a realistic setting
- Includes real-time image-style screenshots and text-style accessibility tree of professional data applications, enabling dynamic multi-round interaction
- Connected to the real-world Internet for authentic user accounts in software requiring them (e.g., Snowflake)
- Provides 170 automatic task setup configurations and 151 customized evaluation metrics developed by computer science authors

**Experiments:**
- Tested with state-of-the-art LLMs and VLMs, including GPT-4 series [21], Gemini-Pro-1.5 [26], Claude-3-Opus [2], QWen-Max [3], Mixtral-8x7B [11], and Llama-3-70B [20]
- Results showed that even top-tier VLM (GPT-4V) achieved only 14.0% success rate, with performance dropping to 1.2% for tasks exceeding 15 action steps
- Open-source LLMs performed less than 2% overall; agents struggled with action grounding and alignment between observation modalities

**Conclusion:**
- Existing LLMs or VLMs are still far from achieving full data workflow automation, but findings lay the groundwork for developing practical multimodal agents that can revolutionize the automation of data science and engineering workflows.

## 2 Executable Computer Environment of Spider2-V

**Spider2-V Executable Computer Environment**

**Introduction**:
- Real-time executable computer environment based on virtual machines (VMs)
- Adapted from OSW ORLD [34]

**Components**:
- Spider2-V: real-time executable environment
- Virtual machines (VMs): underlying technology to build the environment

**Development**:
- Based on existing technology (OSW ORLD)

### 2.1 Task Definition Generally, an autonomous data agent is modeled as a partially observable Markov decision pro- cess (POMDP). Given the current obs

**Autonomous Data Agent Modeling as POMDP**
- **POMDP**: Partially observable Markov decision process
- **Agent's Role**: Generates executable action based on current observation (ot) and accessibility tree (a11ytree)
  - Observation: natural language instruction, screenshot, or combination
  - Execution of action results in new state (st+1) and partial observation (ot+1)
- **Interaction Loop**: Repeats until termination (DONE or FAIL) or max number of steps reached

**Autonomous Data Agent's Components:**
- **Observation Space**: Natural language instruction, screenshot, or combination
  - Described as: current observation (ot)
- **Action Space**: Clicking pixels on the screen (CLICK) or writing code through keyboard (TYPE)
  - Execution results in new state (st+1) and partial observation (ot+1)
- **a11ytree**: Text-style representation of desktop environment
  - Describes status, position, and text content of each element (windows, buttons, input boxes)

**Interaction Loop:**
- Repeats until termination or max number of steps reached.

### 2.2 Environment Setup ORlocalcloud emptyclouddatabaseAPIcalls datauploading(a)FileTransfer(b)ApplicationLaunch(c)RemoteAPICalls(e)PlaywrightAutomatio

**Environment Setup**

**Preparing for Consistent Initial State:**
- Call a series of functions based on VM snapshot to reset environment
- Functions vary among tasks

**Five Common Operations:**
1. **File Transfer**:
   - Transfer files or project archives from local or cloud storage into the VM
2. **Application Launch**:
   - Open software on the desktop, e.g., Visual Studio Code and Chromium
3. **Remote API Calls**:
   - Invoke tool-specific API calls for professional applications to reset and configure cloud workspaces
4. **Script Execution**:
   - Execute a shell script in VM to set up the initial state, e.g., run Docker container for localhost webserver setup like Superset
5. **Playwright Automation**:
   - Run web browser simulation with Playwright, e.g., sign into an account or click specific buttons and redirect to target pages.

### 2.3 Task-specific Evaluation 

**Task-specific Evaluation Methods**
* Three generic methods for task evaluation: `file-based comparison`, `information-based validation`, and `execution-based verification`

**File-based Comparison (a)**
- Finds and copies target files from VM to host
- Uses file-type based metrics (e.g., .json, .csv) for comparison
- Ground truth may be updated over time, fetched from the internet during evaluation

**Information-based Validation (b)**
- Extracts desired information from computer
- Utilized to confirm correct configuration of settings or values
- Examples: checking the time schedule in Airbyte connection or retrieving specific data using APIs or Chromium Playwright

**Execution-based Verification (c)**
- Manually trigger target tasks and check status through logs
- Used to verify whether an expected goal is achieved after executing a Shell script
- Example: manually triggering Airflow DAGs and checking their eventual status

**Task-specific Functions**: written to retrieve desired results from open-ended state of computer and return success flag (0/1)

**Spider2-V**: contains 170 initial state configurations and 151 evaluation scripts.

## 3 Benchmark Construction

**Annotation Pipeline for Spider2-V Dataset**

**3.1 Overview:**
- Collect tutorials: gather source URLs from official websites (217 total)
- Learn tutorials: understand key steps in integrating dbt project into Airflow task (5 steps)
- Write instructions: create abstract and verbose versions of instructions, with modifications
- Write environment setup functions: initialize Airflow project and launch web server on VM
- Write task-specific evaluation functions: obtain results from open-ended states and assess completeness
- Cross-validate on VM: ensure correctness through real-world use case checks and evaluations

**3.1.1 Collect Tutorials:**
- Find tutorials from official websites for each professional tool (10 annotators)
- In total, 217 source URLs collected

**3.1.2 Learn Tutorials:**
- Select one tutorial and learn it in the VM environment
- Summarize key knowledge points from the tutorial

**3.1.3 Write Instructions:**
- Abstract: brief summary of how to integrate a dbt project into an Airflow task
- Verbose: detailed steps with modifications (replace "my_simple_dbt_project" with "jaffle-shop", add time schedule requirement)

**3.1.4 Write Environment Setup Functions:**
- Upload unfinished Airflow project into the VM
- Execute Shell script to launch Docker containers and start web server for Airflow
- Open relevant applications on desktop to simulate real user scenarios
- Use Playwright to auto-login to default Airflow account

**3.1.5 Write Task-Specific Evaluation Functions:**
- Manually run target Airflow DAG and verify final status is "success"
- Retrieve details of the target Airflow DAG using Airflow CLIs and compare with ground truth (dbt subtasks, status, schedule)

**3.1.6 Cross-Validate on VM:**
- Check if chosen task reflects a real-world use case
- Ensure verbose instruction accurately fulfills abstract instruction's requirements
- Test whether environment can be reset to the same state in different trials
- Evaluate robustness when following verbose instructions or modifying some steps
- Verify that evaluation score is 0 if deliberate mistakes are made (red-teaming)

### 3.2 Document Warehouse

**Document Warehouse for Spider2-V**

**Background:**
- Senior data scientists refer to official documentation of professional applications during complex tasks
- Document warehouse built to compensate for deficiencies in enterprise software usage

**Crawling and Preprocessing:**
- Recursively crawl web pages from root websites of professional applications
- Convert raw HTML into text, markdown, or simplified HTML formats
- Obtain 11,231 documents in total

**Dataset Statistics (Table 1):**
- Total tasks: 494 (100%)
  * Pure CLI: 28 (5.7%)
  * Pure GUI: 186 (37.7%)
  * CLI + GUI: 280 (56.7%)
- With authentic user account: 170 (34.4%) vs without: 324 (65.6%)
- Levels and average action steps, instruction length, number of used apps per task

**Task Categories:**
- Classified into 7 categories and 11 software sub-categories
- Most tasks involve CLI and GUI operations (280, 56.7%)
- Easy: 98 (19.8%), Medium: 310 (62.8%), Hard: 86 (17.4%)
- Multiple professional applications used in most tasks

**Comparison with Existing Benchmarks:**
- Real-time executable environment and multiple enterprise software integration
- Intensive GUI operations focused on data science and engineering tasks.

## 4 Experiments and Analysis

**Experiments and Analysis**

**Benchmark Comparison**:
- Table comparing Spider2-V benchmark to existing agent benchmarks:
  - Columns: Research Field, Executable Environment, Enterprise Service, GUI Support
  - Spider2-V statistics: 1034 tasks, text-to-SQL, data science

**Environment Settings**:
- **Agent Baselines**: Set of Mark (SoM), Execution Feedback (EF), Retrieval-Augmented Generation (RAG)
  - SoM: Heuristic methods to retrieve coordinates from a11ytree and enhance alignment
  - EF: Append execution feedback messages for failed actions
  - RAG: Leverage task instruction, BERT model, and LlamaIndex framework for document context generation

**Experimented LLMs and VLMs**:
- Open-source LLMs: Mixtral-8x7B, Llama-3-70B
- Closed-source VLMs: Qwen-Max, GeminiPro-1.5, Claude-3-Opus, GPT families (GPT-4o, GPT-4V5)
  - Open-source LLMs cannot process images; SoM is used instead for them
  - Remaining VLMs support vision input; aligned text and image are used as observation type

**Experiment Settings**:
- Temperature: 0.5, top_p: 0.9
- History trajectory window size: 3
- Maximum length of a11ytree: 5000 tokens
- Maximum output tokens per turn: 1500
- Require agent to complete tasks within 15 interaction turns and one hour.

### 4.2 Main Results

**Spider2-V Task Results Comparison**
* Table 3 shows success rates of different LLMs and VLMs on Spider2-V tasks: Data Warehousing (ware.), Transformation (trans.), Ingestion (inges.), Visualization (visual.), Orchestration (orche.), Proc., and IT Service Management (manag.).
* All results include execution feedback (EF) and retrieval-augmented generation (RAG) techniques.
* Existing data agents are far from satisfactory, with state-of-the-art VLMs achieving at best a 14% overall success rate. LLMs perform worse than VLMs since they cannot handle visual information.
* Closed-source models outperform open-source ones due to pre-training on high-quality data and support for longer contexts and both vision and text modalities.
* Data agents exhibit high variance, especially in the "data ingestion" and "data visualization" categories.
* Reasons for poor performance in these categories include minor errors leading to entire sequences being wasted due to GUI operations and difficulties locating cells in spreadsheets during traditional data processing tasks.

### 4.3 Analysis

**Factors Influencing Success Rates of Data Agents:**
- **Tasks with more inherent action steps are more difficult**:
  - As number of intrinsic action steps increases, average performance decreases significantly
  - Existing VLM-based data agents struggle to accomplish goals in extremely tough tasks
- **Tasks involving authentic user accounts are much more challenging**:
  - Data agents have a 10.6% success rate with tasks involving authentic user accounts
  - Cloud Web UIs have comprehensive functionalities or options, which may cause network response delays and server overload
- **Incorporating GUI operations leads to improved performances:**
  - Pure CLI tasks are more difficult than GUI tasks due to the need for extra actions to locate and switch panels
  - GUIs of professional applications simplify coding tasks, making them easier for agents
- **Providing step-by-step guidelines in task instructions results in performance gains**:
  - Detailed stepwise oracle tutorials eliminate the need for agents to reason and plan, improving overall performance
  - However, the low success rate with verbose instructions (16.2%) indicates that current VLMs still struggle to ground actions in real-world contexts

**Ablation Study Findings:**
- **Action space**: Pyautogui code slightly outperforms JSON dict due to the ability to generate Python code and improve efficiency
- **Observation types**: Screenshot leads to low performances, while a11ytree with precise coordinates significantly promotes agent capability in locating target pixels
- **Techniques**: All 3 techniques (SoM, EF, RAG) boost performance, emphasizing the significance of modal alignment when handling state observations

**Hyperparameters:**
- **Sampling temperature**: Achieved top performance with sampling temperature 0.5
- **History window size**: Performance increases with enlarging history window sizes (from 0 to 3), but interaction efficiency is a serious issue

## 5 Related Work

**Benchmarks for Data Science and Engineering**

**Related Work**:
- Proposed novel benchmarks to evaluate capabilities of LLM agents: data processing/analysis using Excel spreadsheets, common libraries (SQL, pandas), machine learning, software engineering projects.
- Focus on single stage within data pipeline, overlook other stages like data warehousing and orchestration.
- Neglect GUIs in enterprise software and data workflow combination of code programming and intensive GUI operations.

**Spider2-V**:
- First-of-its-kind multimodal agent benchmark for data science and engineering.
- Covers entire data workflow, integrates visual interfaces.
- Addresses deficiencies in domain knowledge of agents through large volume of documents for retrieval.

**Benchmarks for Multimodal Agents**:
- Existing works: web navigation, mobile device, computer desktop GUIs.
- Recent advanced benchmarks provide executable simulation environments with realistic scenarios.
- Few studies investigate multimodal agents in enterprise-level software manipulation and domain knowledge understanding.

**Spider2-V's Contribution**:
- Tests proficiency of agents in data science and engineering by incorporating 20 professional tools into a real-time computer environment.

## 6 Conclusion

**Spider2-V: A Data Science and Engineering Benchmark**

**Overview:**
- Proposed benchmark for data science and engineering applications
- Integrates enterprise professional tools across full data pipeline
- Supports GUI operations alongside code writing
- Contains 494 tasks, involves 20 professional tools
- Provides real-time executable computer environment

**Benchmark Challenges:**
- Latest VLM (GPT-4V) achieves only a 14.0% success rate
- Current multimodal agents still unable to automate data workflows fully

**Significance of Spider2-V:**
- Accessible benchmark for future research in AI and ML
- Laying the foundation for advancing data science and engineering capabilities.

## A. Relevant URLs

**Building a Data Pipeline: Spider2-V Overview**

**Pipeline Creation:**
- Built 3-step Dagster pipeline using dataset "game\_sales" for hourly updates and trend analysis (line chart)
- Upload GoogleSheet to BigQuery as "census" datasets named 'population'

**Transferring Data:**
- Help needed to set up source for transferring data from Faker to target database Snowflake

**Data Management:**
- Separate logic of model "customers" into two staged models: "stg\_customers" and "stg\_orders"
- Full data pipeline: orchestration, transformation, warehousing, ingestion, integration

**Resources:**
1. Task examples: publicly available in Github repository [Spider2-V](https://github.com/xlang-ai/Spider2-V) under Apache-2.0 license
   - SheetCopilot (SheetCopilot[16], GPL-3.0 license)
   - WorkArena (WorkArena[6], Apache-2.0 license)
   - Official tutorials or guides on professional applications: dbt, Airflow, Dagster, Superset
2. Environment code adapted from OSW ORLD (released under Apache-2.0 license)
3. Non-exhaustive list of artifacts used in Spider2-V includes SheetCopilot[16], WorkArena[6], and official tutorials or guides on professional applications.
4. Sandbox functions or free trials for enterprise applications: BigQuery, Snowflake, dbt-cloud, and ServiceNow.

**Project Website:**
- Built project website based on Nerfies template (free-to-use under Creative Commons Attribution-ShareAlike 4.0 International License)
- Provides a high-level overview of Spider2-V, leaderboard of the benchmark, and dynamic task demonstrations.

## B Checklist of All Professional Software in Spider2-V

**Spider2-V Professional Tools**

**List of Incorporated Tools:**
- *Table 6*: All professional tools included in Spider2-V benchmark

**Categories and Descriptions:**
- Each tool categorized with relevant information.

## C Details of Document Warehouse

**Document Warehouse: Crawling and Filtering Process**

**Crawling:**
- Used HTTrack to download HTML files from official documentation websites of software tools
- Limited to English documentation, matching version installed for testing environment
- Built directories recursively to retain website structure

**Filtering:**
1. Irrelevant content:
   - Judged based on paths and manually confirmed
   - Removed category-type pages with no software usage content
2. Invalid pages:
   - Filter out based on token counts (less than 100)
   - Excluded access failures, invalid links, or webpage redirections
3. Preprocessing HTML files:
   - Three formats: plain text, simplified HTML, Markdown
      * BeautifulSoup4 used to extract textual elements from HTML DOM tree
      * Simplified HTML obtained by removing non-textual sub-trees and filtering out unnecessary tags
      * Markdown format converted using markdownify tool
   - Token statistics of all formats in Table 9
4. Performances: Subset experiments on Spider2-V using different formats, pure text outperforms others (see Table 7).
5. Relevant URLs and statistics of software documentation websites provided in Table 8.

## D Details of Executable Environment in Spider2-V

**Overview**
- Spider2-V formalizes interaction with Ubuntu desktop as a partially observable Markov decision process (POMDP)
- Includes state space S, observation space O, action space A, state transition function T, and reward function R
- Agent predicts next action based on current observation ot and desired outcome
- Interaction loop continues until "DONE" or "FAIL" action is issued, ending task episode and computing reward r

**Executable Environment (Ubuntu Operating System)**
- Built upon virtual machines (VMs)
- Localhost environment state can be recovered using VM snapshot functionality
- Snapshot with task setup functions serves as initial state s0 for different tasks
- Core controller is responsible for grounding actions into the VM desktop and obtaining observations ot

**Action Space**
- Two types: pyautogui code and JSON dict
- Pyautogui code accepts arbitrary executable python code, especially code using "pyautogui" library for GUI/CLI interaction
- Encapsulated into a JSON dict to restrict output format
- Checklist of all 7 actions presented in Figure 9(b)

**Observation Space**
- Two alternatives: screenshot or text-format accessibility tree (a11ytree)
- Accessibility tree is a text-format abstraction describing desktop elements, obtained using pyatspi library and converted into XML format
- Aligned with screenshot through set-of-mark method, where bounding boxes are drawn around elements of interest and labeled with numeric indexes
- Set-of-mark (SoM) aligns accessibility tree and screenshot, illustrated in Figure 13

**Execution Feedback**
- Includes execution feedback as third observation type when some predicted actions may be parsed erroneously or fail to execute
- Helps inform the agent of any errors, preventing repeated incorrect actions.

## E Format of Task Examples

**Task Example Format**
- Each task instance is represented as a JSON dictionary with the following fields:
  - **id**: Globally unique identifier for the current task example
  - **instruction**: Task goal indicated by the instruction
  - **source**: List of referenced tutorial links to construct the task
  - **config**: List of dictionaries defining operations to initialize and reset the computer desktop
    - Each dictionary contains:
      - **function name (type key)**: Name of the environment setup function
      - **parameters**: Parameters indicating one environment setup function
  - **related_apps**: List of application names used in the current task
  - **tags**: List of tags denoting different categories
  - **evaluator**: Dictionary containing fields to evaluate the final results once the task is completed
    - **func**: Customized function (or metric) used to compare predicted result and expected golden result
    - **result**: Method to extract predicted result from final environment states
    - **expected**: Method to obtain the golden result for comparison
- Example: Figure 14 illustrates a simple task example configuration file.

## F Task Examples

**Spider2-V Task Examples:**

**Table 10: Real task examples from Spider2-V**

| Related App(s) | Instruction                                                                     | Screenshot                                               | After Initialization        |
|---------------|-------------------------------------------------------------------------------|-----------------------------------------------------------|-----------------------------|
| Dagster       | Integrate dbt project "jaffle_shop", add asset "customers"                    | -                                                      | Materialize asset in Dagster UI|
| BigQuery       | Save top 5 male names from 'names_2014' into new table 'top5_male_2014'         | -                                                      | N/A                        |
| Airflow        | Migrate Airflow DAG to Dagster based on requirements in README.md                | Launch Dagster webserver, start schedule                    | Test and succeed on Dagster UI   |
| Metabase       | Create stack bar chart from Products table, summarize price by Product Category and Created At - Quarter, then stack the visualized chart as a PNG file named "stack_chart.png" | -                                                      | N/A                        |
| Jupyter        | Complete #TODO sections in Logistic Regression code framework and run the code | Read framework code, complete TODOs, run code              | N/A                        |
| Excel          | Add new column "Profit" and calculate profit for each week by subtracting COGS from Sales in that column    | -                                                      | N/A                        |
| Superset       | Create rolling mean line chart for table flights to see trend of average cost per day with a 7-day rolling period, save as "rolling_mean"     | -                                                      | N/A                        |
| Airbyte        | Set up data transfer destination to local JSON file in Airbyte Local UI          | Target file path: /local/json_destination                   | N/A                        |
| dbt-cloud      | Set up connection to BigQuery GCP for project "test_connection"                | No need to configure repository, use provided credential file   | N/A                        |
| Airflow Docker  | Change consumer DAG schedule such that it triggers when producer files are updated | Launch run of job "sklearn_job", schedule it to run at every hour on weekdays | Test and succeed on Dagster UI   |
| Dagster        | Add features "Age" and "Fare" to Logistic Regression model, launch "sklearn_job" with hourly schedule    | N/A                                                      | N/A                        |
| Snowflake      | Get one database about worldwide addresses named 'WORLD-WIDE_ADDRESSES'          | -                                                      | N/A                        |
| ServiceNow     | Order 8 iPad mini with purple color and 256 storage                              | Go to hardware store, order iPad minis                     | N/A                        |
| BigQuery       | Load data from Google Drive Spider002 folder into Bigquery's 'data1' table of 'information' datasets          | -                                                      | N/A                        |
| Metabase PostgreSQL | Complete metabase login setup using information shown in setup.json                  | -                                                      | N/A                        |
| Dagster       | Run pipeline "hacker_news_pipeline" regularly to keep all assets up to date          | Name target job, schedule it to run every hour                | Test and succeed on Dagster UI   |
| dbt-cloud     | Install dbt-cloud-cli from GitHub and set up connection for project "analytics"    | Follow instructions in account profile page                      | N/A                        |

## G Prompts for Multi-modal Agents

**Multi-modal Agent Baseline**

**Complex Prompt Engineering**:
- System prompt, task prompt, and retrieved context augmented prompt are introduced

**System Prompt**: N/A (not provided)

**Task Prompt**: N/A (not provided)

**Retrieved Context Augmented Prompt**: This prompt is used to provide the agent with additional context to help it understand and complete a task. It can be generated by retrieving relevant information from previous interactions, data sources, or other means. 

**Complexity of Prompt Engineering**:
- Involves complex engineering due to the need for generating appropriate prompts that can effectively guide the agent's behavior and understanding.

### G.1 System Prompt

**System Prompt: Observation Space**

**Overview:**
- Entire system prompt consists of environment, observation space, action space, and general tips
- Different action/observation types have distinct prompts

**Observation Space Prompt: Screenshot Setting**
- After each step: image-style observation = screenshot
- Predict next action based on image
- Use for locating elements or checking status

**Observation Space Prompt: Accessibility Tree Setting**
- Text-style observation from AT-SPI library
- Describes desktop elements and text content
- Pruned into tabular format with useful information
- `TAG`, `NAME`, `POSITION`, `SIZE`, and `TEXT` columns provided
- Use for accurate location and text checking

**Observation Space Prompt: Screenshot + Accessibility Tree Setting**
- Combines image-style screenshot and text-style accessibility tree
- Useful for predicting next feasible actions
- Screenshot marked with indexes for ease of use

**Observation Space Prompt: SoM (Set-of-Mark) Setting**
- Combination of image-style screenshot and text-style accessibility tree
- Labeled screenshot with numeric indexes for easy location
- Use for simplifying predicted actions with `pyautogui.click()` function

### G.1.2 Action Space Prompt

**Action Space Prompt (Pyautogui Code)**
* Two choices: pyautogui code or JSON dict
* Pyautogui code uses `pyautogui` library for interactions
	+ Two types of actions: Python code block and pre-defined special actions
		- Python code block wrapped in 3 backticks, e.g., ```python
		- Three pre-defined special actions: [WAIT, FAIL, DONED] wrapped in 3 backticks
* Important notes:
	+ No use of `pyautogui.locateCenterOnScreen` or `screenshot` functions
	+ Time efficiency: return one line or multiple lines for continuous actions
	+ Delay between predictions if necessary
	+ Observe environment changes before follow-ups
	+ Code in isolation with no shared variables or functions
	+ Ensure feasible coordinates
* Precautions:
	+ Strictly follow the format of action dicts (action_type, note, parameters)
	+ Obey FORMAT and only choose actions from defined action space
	+ Use backticks for each action in responses

**Action Space Prompt (JSON Dict)**
* JSON dict describes action types and parameters
* Use case: move the mouse to a specified position (x, y) with required arguments x and y
* Precautions:
	+ Strictly follow the format of action dicts (action_type, note, parameters)
	+ Obey FORMAT and only choose actions from defined action space
	+ Use backticks for each action in responses.

### G.1.3 Overall System Prompt

**System Overview**
- Intelligent agent specialized in data science/engineering tasks using professional tools on Ubuntu OS
- Deep understanding of computer basics and data science/engineering knowledge
- Interact with environment through iterations: take action, receive observation, repeat until task is complete

**Environment Details**
- Use password "password" for sudo rights
- Screen size: ({screen_width}, {screen_height})
- Patience required due to time-consuming actions (code execution, web page loading)

**Tips for Completing Task**
1. Minimize steps and use applications opened as much as possible
2. Ensure critical actions succeed before predicting or proceeding
3. Focus on the right window or input panel when writing codes or texts
4. Break down complex tasks into smaller steps, complete them one by one
5. Do not repeat same actions without progress; try another approach if necessary
6. Only return actions defined in action spaces; never return anything else
7. Use keyboard or mouse to control the environment
8. Obtain new observations after each action is grounded (executed)
9. Repeat steps 1-3 until task completion
10. Follow user instructions carefully and communicate with the environment interactively.

### G.2 Task Prompt

**Task Prompt for Spider2-V:**
* There are two forms of task instruction: abstract and verbose
* Abstract instruction describes overall goal without solution (tests planning and grounding abilities)
* Verbose instruction provides detailed tutorial-like solution (primarily validates grounding ability)

**Example of Task Prompt for Abstract Instructions:**
* Build an airflow project connecting to a local PostgreSQL database
* Install Docker, Astro, and PostgreSQL
* Configure Docker and PostgreSQL to auto-start on boot
* Prevent typing sudo when using Docker

**Example of Task Prompt for Verbose Instructions:**
* Upload data from `xlang_gcs/google_ads/` in Google Cloud Storage (GCS) to a dataset named `my_google_ads`
	+ Click "Add" button next to the "Explorer" panel
	+ Click on the "Google Cloud Storage" panel
	+ Enter `account_history_data.csv` in the input box labeled "Google Cloud Storage"
	+ Set Dataset to `my_google_ads` and Table to `account_history_data`
	+ Mark the checkmark for Auto detect in Schema part
	+ Click on the blue "Create Table" button at the bottom
* Repeat steps above for `account_stats_data.csv` (in the same way as step 7-14)
* Proactively tackle task based on real-time environment interaction or follow detailed plan.

### G.3 Example of Retrieved Context Augmented Task Prompt

**Pure Text Format:**
- Retrieve relevant documentation from web to aid task completion
- Examples of different formats for retrieved context: Pure Text Format
- Documentation source: release-1-7-2.dagster.dagster-docs.io/integrations/dagstermill/using-notebooks-with-dagster.html
- Title: Using Jupyter notebooks with Papermill and Dagster
- Content:
  * Display notebook asset in Asset Graph
  * Click Notebook Asset to view sidebar with info
  * In Description section, press View Source Notebook button
  * Dagster renders and executes notebook when materialized
  * After successful execution, view executed notebook in UI using View button
- Improvements: Factor Iris dataset into its own asset
  * Use same source data for all notebooks analyzing Iris dataset
  * Materialize notebooks without fetching data each time

**Markdown Syntax Format:**
- Retrieve relevant documentation from web to enhance task performance
- Different formats of retrieved context: Markdown Syntax Format
- Documentation source: release-1-7-2.dagster.dagster-docs.io/integrations/dagstermill/using-notebooks-with-dagster.md
- Title: Using Jupyter notebooks with Papermill and Dagster
- Content:
  * Clicking asset displays sidebar with info
  * Press View Source Notebook button to render and execute notebook in UI
- Improvements: Factor Iris dataset into its own asset
  * Use same source data for all notebooks analyzing Iris dataset
  * Materialize notebooks without fetching data each time

**Simplified HTML Format:**
- Retrieve relevant documentation from web to facilitate task accomplishment
- Three examples of retrieved context formats: Simplified HTML Format
- Documentation source: release-1-7-2.dagster.dagster-docs.io/integrations/dagstermill/using-notebooks-with-dagster.html
- Title: Using Jupyter notebooks with Papermill and Dagster
- Content: Several plots of Iris dataset created upon execution, K-means analysis conducted using estimator and plot results show separability of one species from others but not the other two.

**Creating a Dagster Asset:**
- Create Dagster asset from Jupyter Notebook to integrate with data platform
- Use define_dagstermill_asset function for creation.

