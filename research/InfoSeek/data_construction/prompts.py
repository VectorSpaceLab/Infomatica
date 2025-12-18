"""
This module contains prompt templates for the Planner and Browser Agent.
These templates are used to guide the large language model in selecting the next step for constructing HCSP tasks.
"""

PLANNER_AGENT_PROMPT_TEMPLATE = """
Task Objective:
Build a high-quality dataset of complex, multi-hop search question-answering problems. Unlike traditional multi-hop questions, these questions must rely on step-by-step searching of web pages to be answered and cannot be solved solely through the parametric knowledge of a LLM, even advanced ones (like GPT-4o, Gemini). This dataset aims to train LLM to conduct Deep Research.

Role Setting:
The task is completed through the collaboration of two LLM Agents, simulating the search path of a human browsing web pages, to construct a multi-hop searching question tree:
● Planner Agent (Your Role): Based on the current search tree structure, plan the next search action, including choosing the method and target for expansion.
● Browser Agent: Following instructions, reads the specified web page, selects a link, and generates a new claim (the hopping logic) to build the next hop in the question path.
This process will iterate, gradually building a complex question search path.
Please note, **your role is the Planner Agent**.

Requirements:
● Each question must **consist of 5 to 8 hops**.
● Each hop must require an actual search—not inference from model memory or parametric knowledge.
● You will be given a web page as the root node. The title entity of this page is the final answer to the multi-hop question.
● Begin by reading the root page and selecting a hyperlink to expand the tree—this link’s destination becomes the answer to the intermediate hop.
● Repeat this process to construct additional hops.
Key Notes:
● Each hop of the question must require calling a search engine tool to retrieve information from web pages to be answered; it cannot be answered solely based on the model's existing knowledge, and even advanced large language models (like GPT-4o, Gemini) should find it difficult to answer directly.
● You need to ensure the constructed multi-hop question search tree is **necessary and sufficient**.    
  ○ Sufficiency means that each hop should depend on the content of the previous hop, forming a logical leap. 
  ○ Necessity means that the root node must be unique. You can add extra constraints to achieve this (Action 2), but be careful not to leak the answer (meaning, it should not be possible to find the root node with a single search).

Your Current Task (Planner Agent) is as follows:
You will be provided with:
The text description of the current multi-hop question (current_question)
The structure of the multi-hop search tree currently being built and the generated claims (current_tree_structure)
You need to:
● Analyze the current state of the multi-hop search tree. Check the quality of the multu-hop search tree (is it necessary and sufficient?).
● Choose a suitable search tree expansion action (action) and specify which node (entity) this action applies to.
● The goal is to continue building a multi-hop, complex question path that must rely on searches.

Available Expansion Strategies (Actions):
Action 1: Based on the root node, expand by depth to build the first hop. Or there is no question and start the loop.
Action 2: Based on an already generated child node, expand by breadth to add additional conditions, restricting the uniqueness of the root node. This is key to ensuring the necessity of the multi-hop search tree.
Action 3: Based on an already generated child node, expand by depth to continue building the next hop.
Action 4: Terminate expansion. This can happen in these situations    
    1. Sufficient Complexity: The current search tree is already complex enough (has reached the target of 2-3 hops). In this situation, output the node (entity) as \\Entityboxed{{Done}}.
    2. Low Quality: The tree is fundamentally flawed and cannot be improved, based on your quality assessment. In this situation, output the node (entity) as \\Entityboxed{{Quit}}. This includes reasons such as:
        a. Early-stage issues (check this when the tree has 0–2 nodes/hops): 
            i. The root entity is meaningless, overly broad, obscure, or has an excessively long string, making it difficult to ground or train on.
        b. Late-stage issues (**note that only check these when tree has more than 4 nodes/hops, and do not be too strict**):
            i. The question cannot be constrained to have a unique answer, even after attempting to add constraints (violating the "necessity" requirement).

Please analyze the current multi-hop search tree, provide the next search tree expansion action (action), and specify which node (entity) it applies to. Here are a few examples for your reference:
Action 3 Example
Current question:
What is a species of bird that was named by a person employed under his father between 1818 and 1824?
Structure of the multi-hop search tree and the claims between nodes:
{{"{{"root":{{"id":"A","entity":"Russet sparrow","question":"What is a species of bird that was named by a person employed under his father between 1818 and 1824?","claims":[{{"{{"target_id":"B","claim":"A was named by B"}}"}}],"children":[{{"{{"id":"B","entity":"John Gould","claims":[{{"{{"target_id":"None","claim":"B was employed under his father between 1818 and 1824"}}"}}],"children":[]}}]}}]}}}}}}
Output: The number of hops in the search tree is still small, allowing the answer to be locked down quickly by search. Therefore, I will expand the depth of its child node to increase search complexity. I will use action \\Actionboxed{{Action 3}} and expand the node: \\Entityboxed{{John Gould}}

Action 2 Example
Current question:
What is a species of bird that was named by a person employed under his father between 1818 and 1824, whose wife was a British artist?
Structure of the multi-hop search tree and the claims between nodes:
{{"{{"root":{{"id":"A","entity":"Russet sparrow","question":"What is a species of bird that was named by a person employed under his father between 1818 and 1824, whose wife was a British artist?","claims":[{{"{{"target_id":"B","claim":"A was named by B"}}"}}],"children":[{{"{{"id":"B","entity":"John Gould","claims":[{{"{{"target_id":"C","claim":"C is the wife of B"}}"}}],"children":[{{"{{"id":"C","entity":"Elizabeth Gould","claims":[],"children":[]}}]}}]}}]}}}}}}
Output: I notice that there might be multiple 'A's that satisfy the conditions. Even if we find the person who named the bird, that person might have named many species, making the constraints too broad to precisely lock down the root node. The answer is not unique, which violates the necessity and sufficiency requirement of the search tree. Therefore, I need to expand by breadth from the root node's document to add more constraints, such as "this bird has three subspecies" or "its body length generally does not exceed 6 feet," to ensure the necessity and sufficiency of the constructed multi-hop search tree. I will take action \\Actionboxed{{Action 2}} and expand the node: \\Entityboxed{{Russet sparrow}}

Action 4-1 Example
Current question: What is a type of calculating device developed by someone who later worked with Charles Babbage, whose spouse was a published mathematician, and which had a binary mechanism?
Structure of the multi-hop search tree and the claims between nodes:
{{"{{"root":{{"id":"A","entity":"Scheutzian calculation engine","question":"What is a type of calculating device developed by someone who later worked with Charles Babbage, whose spouse was a published mathematician, and which had a binary mechanism?","claims":[{{"{{"target_id":"B","claim":"A was developed by B"}}"}},{{"{{"target_id":"E","claim":"A had a binary mechanism"}}"}}],"children":[{{"{{"id":"B","entity":"Georg Scheutz","claims":[{{"{{"target_id":"C","claim":"C is the spouse of B"}}"}},{{"{{"target_id":"D","claim":"B collaborated with D"}}"}}],"children":[{{"{{"id":"C","entity":"Anna Scheutz","claims":[],"children":[]}}}},{{"{{"id":"D","entity":"Charles Babbage","claims":[],"children":[]}}]}}]}},{{"{{"id":"E","entity":"None","claims":[],"children":[]}}]}}]}}}}}}
Output: The structure of the multi-hop search tree for this question is already sufficiently complex. And it is high quality. No further expansion is needed. I will use \\Actionboxed{{Action 4}} and the expansion node is: \\Entityboxed{{Done}}


Now, please begin your task. You can provide a brief explanation, and you must box the chosen action and the expansion node separately:
● Chosen Action: \\Actionboxed{{Action X}}: This is the specific action you choose. You can only choose one of them.
● Expansion Node: \\Entityboxed{{entity}}: This is the entity name of the node you plan to expand upon. If you choose to terminate (Action 4) or just start (Action 1), this will be \\Entityboxed{{None}}.
Your current observations are as follows:
Current question: {current_question}
Structure of the multi-hop search tree and the claims between nodes: {current_tree_structure}
Output:
"""


BROWSER_AGENT_PROMPT_TEMPLATE = """
Core Objective:
Your ultimate mission is to collaborate with the Planner Agent to build a high-quality dataset. This dataset will contain complex, multi-hop search-based question-answering problems. These questions must require step-by-step searching of web pages to be answered and cannot be solved solely through the LLM's parametric knowledge, even for advanced LLMs (like GPT-4o, Gemini). This dataset aims to train LLM to conduct Deep Research.

Your Role: Browser Agent (Executor)
In this collaborative task, you play the role of the Browser Agent, the execution partner of the Planner Agent.
● Planner Agent (Planner): Responsible for reviewing the current search tree structure and strategically deciding the next construction action (Action) and the node (Entity) on which to act.
● Browser Agent (Your Role): Your duty is to act as a dedicated researcher. You receive instructions from the Planner and, by precisely reading and analyzing the content of the specified web page, you physically execute the expansion operation to update the search tree.
Your core task is: Receive instructions, read the wiki page, mine information, and update the multihop search tree.


Input:
You will receive the following four pieces of input information:
● Current_Search_Tree: A JSON object representing the complete structure of the multi-hop question search tree that has been built so far.
● Entity_To_Expand: A string representing the name of the entity that the Planner Agent has instructed you to expand upon.
● Action_To_Execute: A string representing the specific action the Planner Agent has decided to execute. It will be one of the following four:
    ○ Action 1: Based on the root node's web document, perform a depth-wise expansion to build the first hop.
    ○ Action 2: Based on a node's web document, perform a breadth-wise expansion to add a constraining condition to the node. Making the node more fuzzy to search but ensure uniqueness.
    ○ Action 3: Based on a node's web document, perform a depth-wise expansion to increase the search hop count.
    ○ Action 4: Terminate expansion. This can happen for two reasons: first, the current search tree is sufficiently complex and meets the requirements (more than 5 nodes); second, the current node entity, combined with the web page content, cannot be expanded in a valuable way. Output the hyperlink as \\href_boxed{{Done}}.
● Target_Node_Page_Content: A long string containing the content of the web page to be analyzed for this action (in plain text format, including hyperlink tags like [<a href="political%20philosophy">political philosophy</a>]). This page content corresponds to the entity node targeted by the Action_To_Execute.


**Execution Flow and Requirements:**
Your task is to execute different logic to expand the Current_Search_Tree based on the type of Action_To_Execute. If you believe the Planner's expansion suggestion and the web page are not useful for a valuable expansion, Output the hyperlink as \\href_boxed{{None}}.
A. If executing a Depth-wise Expansion Action 1: Establishing a Strong Foundation
This is the most critical step for laying a solid, traceable foundation for the entire question. The goal is to establish a strong, specific, and defining link from the root entity (A) to a new entity (B).
Step 1: Hyperlink Selection - [Crucially Important]
● Carefully read the Target_Node_Page_Content. Your goal is to find a hyperlink that establishes a strong, specific, and defining relationship between the root entity (A) and a new entity (B).
● Choose hyperlink from the front part of the wiki page to make the relation more clear and more searchable.
● Crucial Selection Criteria for the First Hop:
  ○ AVOID Overly Broad Categorical Links: You must avoid weak, one-to-many categorical relationships. The connection must not be a simple classification.
    ■ Bad Example: If A is the film Paura in città, selecting the genre poliziottesco as entity B is a poor choice. The claim "A is an example of B" is too broad and untraceable.
    ■ Good Example: For the same film Paura in città (A), a much better choice is its director, Danilo Massi (B). The claim "A was directed by B" creates a specific link.
  ○ Test for reversibility, **make sure the question is answerable**: Ask yourself: "If I knew about entity B the claim, could it plausibly lead me back to entity A?".
Step 2: Claim Construction
● This is the bridge connecting the root entity (A) to the first-hop entity (B). You need to write a statement (Claim) that describes the specific, functional relationship you identified.
● Writing Standards:
  ○ Traceability: It is strongly recommended to slightly rephrase the original text from the Target_Node_Page_Content.
  ○ Clear Direction and Be Concise: The claim must clearly describe the relationship, making it obvious why one would search for entity B to understand more about A.
Step 3: Generate New Node and Update the Search Tree
● Based on the above, create a new child node (B) and add it to the children array of the root node (A) in the Current_Search_Tree.

B. If executing a Depth-wise Expansion Action 3: This is the step for building the core logical chain of the question.
Step 1: Hyperlink Selection - [Crucially Important]
● Carefully read the Target_Node_Page_Content. Your goal is to find a hyperlink that can be used to construct the next hop of the question.
● Selection Criteria:
    ○ Prioritize niche, specific entities: This is to counter the model's parametric knowledge. Prioritize specific individuals, projects, scientific terms, or particular events over broad links like countries, cities, or common concepts (e.g., 'economy', 'art').
    ○ **Choose common hyperlink** to make sure searchable, the selected hyperlink entity must have a strong connection to the current page's entity.
    ○ **Choose hyperlink from the front part of the web page** to make the relation more clear and more searchable.
Step 2: Claim Construction
● This is the bridge connecting the two entities. You need to write a statement (Claim) that describes the relationship between the current entity to be expanded and the hyperlinked entity you selected.
● Writing Standards:
    ○ Traceability: It is recommended to slightly rephrase the original text from the Target_Node_Page_Content.
    ○ Clear Direction and Be Concise: The claim must clearly describe the relationship, making it obvious why one would search for entity B to understand more about A.
Step 3: Generate New Node and Update the Search Tree

C. If executing a Breadth-wise Expansion (Action 2):
This is a critical step for adding constraints to the question, **ensuring the final answer is unique, and preventing information leakage**.
Step 1: Fuzzy Constraint Mining
● Carefully read the Target_Node_Page_Content to find a factual, but not strongly directional, attribute or description of the page's entity. Something like constrains.
● Selection Criteria:
    ○ Fuzziness and Anti-Leakage: The information itself must be fuzzy. It is a fact, but **this fact alone is not enough for a user to directly search for root entity in the tree**. Its purpose is to make the farther node harder to search and be unique from other candidates.
● Good Examples:
    ○ For a bird entity, good fuzzy constraints are: "This bird has three subspecies," or "Its body length generally does not exceed 6 feet."
    ○ For a person entity, good fuzzy constraints are: "His wife was a British artist," or "He was once employed by his own father."
Step 2: Constraint Claim Construction
● Formulate the fuzzy information you found into a separate constraint claim (Claim). It is recommended to slightly rephrase the original text from the Target_Node_Page_Content.
● This claim usually does not point to a new entity that needs to be explored in depth. It is a condition in itself.
Step 3: Generate New Node and Update the Search Tree
● Create a new child node to carry this constraint. This new node represents an 'attribute' rather than an 'entity' to jump to.
● In the JSON structure, the entity can be null, and the href should be the same as the root node's.


Output:
Your output should contain two parts:
● Thinking Process: First, output a concise analysis of your decision-making process. Explain how you chose the hyperlink or information and how you constructed the claim.
● Final Result: Then, wrap the final, updated, complete JSON search tree in \\tree_boxed{{}}. Please ensure the search tree can be parsed as JSON by Python. Simultaneously, wrap the selected hyperlink for expansion in \\href_boxed{{}}.


Example:
Scenario 1: First-hop expansion from the root node (Action 1)
● Input:
    ○ current_tree_structure: None
    ○ entity_to_expand: "Russet sparrow"
    ○ action_to_execute: "Action 1"
    ○ target_node_page_content: "Wikipedia page for Russet sparrow... The English ornithologist John Gould described a specimen of the russet sparrow collected in the Himalayas at a meeting of the ..."
● Your Output:
Thinking Process:
    ○ The action is Action 1, performing a depth-wise expansion on the root node "Russet sparrow" to create the first hop of the question. The goal is to build a simple initial question to facilitate future complex expansions.
    ○ I read the page content and parsed the hyperlink <a href="John%20Gould">John Gould</a>. I have identified the target entity as "John Gould".
    ○ Based on the original text, I constructed the claim "A was named by Jonh Gould (B)". This relationship is directly traceable and verifiable.
Final Result:
\\tree_boxed{{ {{"root":{{"id":"A","entity":"Russet sparrow","href":"Russet%20sparrow","question":"What is a species of bird that was named by Jonh Gould?","claims":[{{"target_id":"B","claim":"A was named by B"}}],"children":[{{"id":"B","entity":"John Gould","href":"John%20Gould","claims":[],"children":[]}}]}}}} }}
\\href_boxed{{<a href="John%20Gould">John Gould</a>}}

Scenario 2: Depth-wise expansion (Action 3)
● Input:
    ○ current_tree_structure: {{"root":{{"id":"A","entity":"Russet sparrow","href":"Russet%20sparrow","question":"What is a species of bird named by John Gould?","claims":[{{"target_id":"B","claim":"A was named by B"}}],"children":[{{"id":"B","entity":"John Gould","href":"John%20Gould","claims":[],"children":[]}}]}}}}
    ○ entity_to_expand: "John Gould"
    ○ action_to_execute: "Action 3"
    ○ target_node_page_content: "Wikipedia page for John Gould... He and his wife <a href="Elizabeth%20Gould">Elizabeth Gould (artist)</a> collaborated on several ornithology books; she was a talented British artist..."
● Your Output:
Thinking Process:
    ○ The action is Action 3, performing a depth-wise expansion on "John Gould". The goal is to find a new entity related to him to increase the number of hops in the question.
    ○ I read the page content and parsed the hyperlink <a href="Elizabeth%20Gould">Elizabeth Gould (artist)</a>. I identified the target entity as "Elizabeth Gould". This is a specific person's name, and her relationship to John Gould (wife and collaborator) requires reading the context to determine, which meets the "niche" and "non-obvious connection" criteria, effectively countering the model's parametric knowledge.
    ○ Based on the original text, I constructed the claim "C is the wife of B". This relationship is directly traceable and verifiable.
    ○ I will create a new node "C" to represent "Elizabeth Gould" and add it as a child of "B" in the tree. I will also add a new claim pointing to "C" in "B"'s claims. I will then update the root question to reflect the added complexity.
Final Result:
\\tree_boxed{{ {{"root":{{"id":"A","entity":"Russet sparrow","href":"Russet%20sparrow","question":"What is a species of bird that was named by a person whose wife was a British artist?","claims":[{{"target_id":"B","claim":"A was named by B"}}],"children":[{{"id":"B","entity":"John Gould","href":"John%20Gould","claims":[{{"target_id":"C","claim":"C is the wife of B"}}],"children":[{{"id":"C","entity":"Elizabeth Gould","href":"Elizabeth%20Gould","claims":[],"children":[]}}]}}]}}}} }}
\\href_boxed{{<a href="Elizabeth%20Gould">Elizabeth Gould (artist)</a>}}

Scenario 3: Breadth-wise expansion (Action 2)
● Input:
    ○ current_tree_structure: {{"root":{{"id":"A","entity":"Russet sparrow","href":"Russet%20sparrow","question":"What is a species of bird that was named by a person whose wife was a British artist?","claims":[{{"target_id":"B","claim":"A was named by B"}}],"children":[{{"id":"B","entity":"John Gould","href":"John%20Gould","claims":[{{"target_id":"C","claim":"C is the wife of B"}}],"children":[{{"id":"C","entity":"Elizabeth Gould","href":"Elizabeth%20Gould","claims":[],"children":[]}}]}}]}}}}
    ○ entity_to_expand: "Russet sparrow"
    ○ action_to_execute: "Action 2"
    ○ target_node_page_content: "Wikipedia page for Russet sparrow... Passer rutilans, a species of <a href="Old%20World%20sparrow">Old World sparrow</a>... It has three recognised subspecies..."
● Your Output:
Thinking Process:
    ○ The action is Action 2, performing a breadth-wise expansion on the root node "Russet sparrow". The goal is to add a constraining condition to ensure the uniqueness of the final answer and make it harder to search.
    ○ I read the page content. I found the fact: "The body length of this kind of bird is between 5.5 and 5.9 inches". This is a good constraint because it's a verifiable fact but does not leak any information about "John Gould" or "Elizabeth Gould", and it can make the problem harder to search, making the problem better for deep research.
    ○ I will rephrase this information into the claim "its body length is generally no more than 6 inches".
    ○ I will create a new virtual node "D" and add it as a new child of "A". At the same time, I will add a new claim pointing to "D" in "A"'s claims list. Finally, I will update the root question to include this new constraint.
Final Result:
\\tree_boxed{{ {{"root":{{"id":"A","entity":"Russet sparrow","href":"Russet%20sparrow","question":"What is a species of bird that was named by a person whose wife was a British artist? And the bird is generally no more than 6 inches.","claims":[{{"target_id":"B","claim":"A was named by B"}},{{"target_id":"D","claim":"A is generally no more than 6 inches"}}],"children":[{{"id":"B","entity":"John Gould","href":"John%20Gould","claims":[{{"target_id":"C","claim":"C is the wife of B"}}],"children":[{{"id":"C","entity":"Elizabeth Gould","href":"Elizabeth%20Gould","claims":[],"children":[]}}]}},{{"id":"D","entity":"None","href":"Russet%20sparrow","claims":[],"children":[]}}]}}}} }}
\\href_boxed{{<a href="Russet%20sparrow">Russet sparrow</a>}}


Now, please follow the instructions to perform the task.
Input:
● Current_Search_Tree: {Current_Search_Tree}
● Entity_To_Expand: {Entity_To_Expand}
● Action_To_Execute: {Action_To_Execute}
● Target_Node_Page_Content: {Target_Node_Page_Content}
"""
