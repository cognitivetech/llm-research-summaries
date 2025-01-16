# The application of GPT-4 in grading design university students’ assignment and providing feedback: An exploratory study

by Qian Huang, Thijs Willems, King Wang Poon
https://arxiv.org/abs/2409.17698

## Abstract
**Study Aims:**
- Investigate if GPT-4 can effectively grade design university assignments and provide useful feedback
- Design projects have subjective nature, leading to grading inconsistencies between raters
- Employ iterative research approach to develop a Custom GPT for reliable results

**Findings:**
1. **Inter-reliability**: Between GPT and human raters reached acceptable level by providing accurate prompts and iterating a Custom GPT
2. **Intra-reliability**: Consistency of GPT's scoring is between 0.65 and 0.78, indicating reliable results with adequate instructions
3. **Comparability and consistency:** A precondition for educational assessment; Custom GPT adheres to these rules
4. **Useful feedback**: Tested whether Custom GPT can provide students with constructive feedback
5. **Educator's role:** Reflecting on how educators can develop and iterate a Custom GPT as a complementary rater.

## 1. Introduction

**ChatGPT's Impact on Education**

**Revolutionizing Industries**:
- ChatGPT has revolutionized various industries, including education (Montenegro-Rueda et al., 2023)
- Potential for ChatGPT to transform educational settings has sparked significant debate and discussion (Arif, Munaf & Ul-Haque, 2023; Hwang & Chen, 2023)

**Embracing Technology in Education**:
- A growing number of educators recognize the necessity of embracing technology to enhance learning outcomes (Sim, 2023)
- Universities like Arizona State University have integrated ChatGPT into their teaching and research frameworks

**Guidelines for Ethical and Effective Use**:
- Organizations and educational institutions are beginning to establish guidelines for the ethical and effective use of generative AI in educational contexts (Halaweh, 2023)

**Advancements with GPT-4**:
- Significant leap forward in AI accuracy and utility in academic contexts (Chen, 2023; Nori et al., 2023)
- Custom GPT models like "Scholar GPT2" have been developed to enhance research productivity and tailor solutions to specific professional needs

**Limitations of GPT-4**:
- Struggles with reasoning tasks and consistently solving complex problems (Arkoudas, 2023)
- Prone to hallucination and fabrication in responses (Currie, 2023)
- Concerns regarding potential erosion of students' critical thinking and problem-solving skills due to over-reliance on AI (Luckin, 2017)

**Ethical Considerations**:
- Potential for AI systems, including ChatGPT, to perpetuate biases present in their training data (Weidinger et al., 2022)
- Continuous efforts to audit and refine AI models are essential to ensure fairness and inclusivity (Halaweh, 2023)

**Study Objective**:
- This study seeks to empirically study the extent to which a Custom GPT-4 model can assist teachers in grading subjective and open-ended assignments, such as design projects

## 2. Literature

**ChatGPT in Education**

**Applications of ChatGPT**:
- Provides personalized learning experiences
- Facilitates administrative tasks
- Reshapes traditional educational landscapes

**Educators can leverage ChatGPT**:
- Assist in creating and delivering content
- Generate lecture materials, interactive discussions, and tailor assessments to student needs
- Support innovative testing approaches like adaptive testing

**Transformative capabilities of ChatGPT in Education**:
- **Academic writing**: Error detection, writing improvement, content generation
- **Personalized feedback**: Generating personalized feedback for programming assignments
- **Preparing teaching material**: Designing teaching contents, materials, quizzes
- **Analyzing qualitative data**: Enhancing research capabilities in various fields

**Reliability of Educational Assessment**

**Inter-Rater Reliability**:
- Degree of consistency among multiple raters in their evaluations
- Measured using intraclass correlation coefficient (ICC)
  - ICC above 0.85 recommended for high-stakes assessments
  - ICC between 0.60 and 0.74 acceptable for normal assessments
  - ICC above 0.70 good, indicating strong consistency

**Intra-Rater Reliability**:
- Degree of consistency of a single rater over time
- Measured using intraclass correlation coefficient (ICC)
  - ICC values and interpretation similar to inter-rater reliability

**Importance of Ensuring Assessment Reliability**:
- Critical for valid and equitable evaluation of student performance
- Intra-rater and inter-rater reliability essential for maintaining consistency and fairness in assessments.

## 3. Research methods

**Study Background**
- Conducted in a design thinking course at a university in Singapore for first-year students
- Observed that engineering instructors and architecture instructors gave different advice, leading to student confusion on how to balance feedback
- Researchers used ChatGPT-4 to grade design assignments of 10 students (20 posters)
- Employed Design-Based Research (DBR) approach: multiple iterations, prototypes, and testing for continuous refinement

**Study Process**
- **Part I:** Comparison without role-play
  - 1st iteration: General GPT grades assignments
  - 2nd iteration: ChatGPT-4 with instructor rubrics
  - 3rd iteration: Custom GPT using good example as benchmark

- **Part II:** Comparison with role-play instructions
  - 1st iteration: General GPT for grading (no role-play)
  - 2nd iteration: ChatGPT-4 for grading with rubrics (role-play)
  - 3rd iteration: Custom GPT, added rubrics and a good example (role-play)

- **Part III:** Intra-reliability of GPT assessment
  - Checked if custom GPT evaluates assignments consistently over time

**Research Questions**
1. Can ChatGPT accurately grade students' design course assignments?
2. Under what conditions can a Custom ChatGPT act as a reliable grader to serve as a complementary rater?

**Assumptions**
- Assumption 1: Inter-reliability increases progressively in Part I without role-play
- Assumption 2: Inter-reliability increases progressively in Part II with role-play
- Assumption 3: Custom ChatGPT has an intra-reliability greater than 0.5 when scoring the same assignment at different times.

## 4. Findings

### 4.1 (Part I) Inter-Reliability between GPT and Instructors without Role-Playing

**1st Iteration:**
- General GPT process: Graded student assignments using three dimensions - Design goal, Site drawing, MacroAEIOU
- Compared to instructor reliability: Low inter-reliability between Architect Instructor and Engineer Instructor (0.167)
- Comparison with GPT: Inter-reliability was slightly higher with Engineer instructor (average scores of two instructors at 0.473), but still below 0.5

**2nd Iteration:**
- Custom GPT model built using rubrics provided by instructors
- Researchers used screenshots and custom-GPT for scoring assignments with feedback based on rubrics
- Improved inter-reliability between GPT and both instructors (Architect Instructor: 0.5803, Engineer Instructor: 0.5803)
- Inter-reliability with average scores of two instructors also improved (0.5672)

**3rd Iteration:**
- Revised instructions to Custom-GPT by giving it a good example for reference
- Improved inter-reliability between GPT and both instructors further: Architect Instructor (0.7652), Engineer Instructor (0.7652)
- Inter-reliability with average scores of two instructors reached 0.7285, meeting high-stakes exam requirements.

**Assumptions:**
- In iterations without role-playing (Part I), inter-reliability increases progressively.

### 4.2 (Part II) Inter-reliability between GPT and human raters with role-play

**First Iteration: General GPT with Role-play Process**
* Let General GPT grade students' assignments from architect instructor and engineer instructor perspectives
* Inter-reliability between Architect GPT and Architect Instructor: **0.1874**, poor reliability
* Inter-reliability between Engineer GPT and Engineer Instructor: **0.5423**, moderate reliability
* Inter-reliability between two GPts: **0.5407**, moderate reliability
* Comparison of ICC Reliability for Iteration 1 (Part II with role-playing) in Table 4

**Second Iteration: Custom GPT (with rubrics added, with role-playing)**
* Added rubrics used by both instructors and conducted role-play to score from their perspectives
* Inter-reliability between Architect GPT and Architect Instructor: **0.5150**, moderate reliability +**0.2068**
* Inter-reliability between Engineer GPT and Engineer Instructor: **0.7652**, high reliability -**0.3024**
* Inter-reliability between two GPts: **0.5407** - **0.0132**, moderate reliability
* Comparison of ICC Reliability for Iteration 2 (Part II with role-playing) in Table 5

**Third Iteration: Custom GPT (with a good example added, with role-play)**
* Included a good example and conducted scoring from both instructors' perspectives
* Inter-reliability between Architect GPT and Architect Instructor: **0.7199**, high reliability +**0.3257**
* Inter-reliability between Engineer GPT and Engineer Instructor: **0.5554**, moderate reliability +**0.3155**
* Inter-reliability between two GPts: **0.5536**, moderate reliability +**0.0261**
* Comparison of ICC Reliability for Iteration 3 (Part II with role-playing) in Table 6

**Discussion:**
- The results show that inter-reliability increases progressively as the GPts adapt to the nuanced expectations of human instructors.
- In each iteration with role-playing, the inter-reliability between Architecture GPT and Engineering GPT is above 0.5 while inter-reliability between human Architecture and Engineering instructors is only 0.167.
- Possible reasons for higher ICC results in general GPts compared to custom GPts:
  - Roleplaying adds complexity as the model simulates different instructional perspectives.
  - Custom GPt struggles to adapt its grading process when switching between these varied perspectives without a good example.

### 4.3 (Part III) Intra-reliability of GPT accessed in three diKerent times

**GPT Scoring Reliability: Intra-Reliability Study Results**

**Assumption 3 Verification**:
- Researchers checked GPT's scoring consistency across different times using General GPT
- Three comparisons were conducted on different days

**Project Scores Over Time (ICC Values)**:
| Project | 1st time | 2nd time | 3rd time | ICC with 1st | ICC between 2nd and 3rd | ICC between 1st and 3rd |
|---|---|---|---|---|---|---|
| Accessed on: May 21, 2024 | 89 | 92 | 89 | 0.6485 (ICC) | N/A | N/A |
| Accessed on: May 27, 2024 | 94 | 93 | N/A | 0.7336 (ICC between 1st and 2nd) | N/A | N/A |
| Accessed on: June 3, 2024 | 78 | 79 | N/A | 0.7750 (ICC between 2nd and 3rd) | N/A | 0.6983 (ICC between 1st and 3rd) |

**Assumption 3 Conclusion**:
- High intra-reliability of GPT is confirmed with an ICC greater than 0.5 for repeated scoring at different times.

### 4.4 Custom GPT’s role in providing feedback

**GPT's Role in Providing Timely Feedback for Design Students:**
* Contributes significantly to providing feedback on a large scale
* Instructors lack capacity for individualized, continuous feedback
* Grades assignments with personalized feedback from different perspectives (engineer and architect) (Screenshot 3)
* Synthesizes opinions of multiple instructors for balanced feedback (Screenshot 4)
* Compares students' work to renowned designers, providing suggestions for improvement (Screenshot 8)
* Provides round-the-clock personalized feedback and stimulates ideas
* Serves as a co-worker during instructors' grading process.

**GPT's Impact on Design Students:**
* Receives timely, continuous feedback from GPT
* Balances different perspectives in their work through feedback (Screenshot 3)
* Aspires to reach or exceed industry benchmarks by learning from renowned designers (Screenshot 8)
* Receives constructive criticism and suggestions for improvement.

## Discussion

**Study Findings on Using ChatGPT for Educational Grading:**

**Iterative Process:**
- Study demonstrates two lines of iteration on how GPT can effectively grade students' design assignments
- High inter-reliability and intra-reliability achieved with appropriate instruction

**Guidelines for Customizing GPT:**
1. Provide accurate and consistent prompts
2. Establish rubrics consistent with instructor's use
3. Give a clear example to clarify good project/assignment

**Implications of Findings:**
- ChatGPT can serve as a co-worker for instructors in educational settings
- Importance of structured rater training, clear rubrics, and regular feedback mechanisms
- Innovations in tools and methods for training and calibrating raters essential to maintain fairness

**Use of GPT Models:**
- Provide consistent, unbiased scoring guidelines
- Assist human raters in understanding and correcting biases
- Concerns regarding transparency, potential biases, and reduced critical engagement by human raters

**Custom-GPT's Impact on Learning:**
- Promotes personalized learning for students (Baidoo-Anu & Owusu Ansah, 2023)
- Aligns with insights from recent studies on language learning (Dai et al., 2023)

**Productivity Enhancement:**
- GPT can increase productivity for educators by providing detailed feedback and interacting in formative assessments.

