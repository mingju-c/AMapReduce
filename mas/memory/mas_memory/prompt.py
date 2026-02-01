from dataclasses import dataclass

# ---------------------------------------------- Expmemory memory ----------------------------------------------
# Retrieve tasks based on task relevance
generative_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
generative_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''

# successful trajectory extraction
extract_true_traj_system_prompt = """You are an agent skilled at extracting key points.
Given a task and a successful execution trajectory, your job is to identify the critical steps needed to complete the task while filtering out less important steps."""
extract_true_traj_user_prompt = """
Note: 
- Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added.
- Even in a successful trajectory, there may be some incorrect steps. Pay attention to actions that correspond to "Nothing happens" observations, as these actions are likely incorrect. Filter out these actions for me.
- You need to ensure that each step is at the finest granularity.
- You should strictly follow the output format in the example.

## Example 1:
### Task
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.

### Trajectory
... The agent manages to find a bowl 1 on desk 2
Act 14: think: Now I find a bowl (1) on desk 2. Next, I need to take it.
Obs 14: OK.
Act 15: take bowl 1 from desk 2
Obs 15: You pick up the bowl 1 from the desk 2.
Act 16: think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1-2), shelf (1-6), or drawer (1-6). I can check one by one, starting with desk 1 since I already found a desklamp there.
Obs 16: OK.
Act 17: go to desk 1
Obs 17: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2.
Act 18: think: Now I find a desklamp (1) on desk 1. Next, I need to use it.
Obs 18: OK.
Act 19: use desklamp 1
Obs 19: You turn on the desklamp 1.
reward: True

### Output
Construct the overall approach: Find and take a bowl, then find and use a desklamp.
Search for the Bowl: Check all locations systematically in order until the bowl is found (e.g., drawers, desks, shelves, garbage can, laundry hamper, and any other possible places).
Find the Bowl: Locate the bowl on desk 2.
Take the Bowl: Pick up the bowl from desk 2.
Search for the Desklamp: Recall that a desklamp was found earlier on desk 1.
Go to Desk 1: Move to desk 1 where the desklamp is located.
Use the Desklamp: Turn on the desklamp.

Now it's your turn! 
## Here is the task:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

# Insights
finetune_insights_suffix = dict(full = """Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.
""", not_full = """""")

format_rules_operation_template = """<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

#
critique_compare_rules_system_prompt = """
You are an advanced reasoning agent deriving **structural batching insights** for MapReduce-style web tasks.
Your goal is to output **procedural rules about how to plan the task**, not the task content.
Good insights cover four aspects:
1) Task Structure Pattern (independence, parent-child/grouping, ordering, whether rows are atomic)
2) Input Template–Task Mapping (what fields are substituted per row, how schema/template affect independence and context size)
3) Batching Strategy Pattern (when to use per_atom/by_attr/open; chunk_size guidance; context reuse considerations)
4) Reliability / Ordering (when to preserve order; when grouping is mandatory; parallelism vs. correctness)
Always produce short, transferable rules (3–6 items max) in natural language; avoid placeholders/schemas/dicts.
"""

critique_compare_rules_user_prompt = """
## SUCCESS pattern_info
{task1_pattern}

## FAILED pattern_info
{task2_pattern}

## EXISTING HINTS:
{existing_rules}

By contrasting SUCCESS vs FAILED patterns and the list of existing hints, propose operations (ADD/EDIT/REMOVE/AGREE) to refine the hint set.
Output only operations using the format below:
""" + format_rules_operation_template

# all success instruction
critique_success_rules_system_prompt = """
You are an advanced reasoning agent that extracts **structural batching insights** from successful MapReduce pattern_info logs.
Focus on transferable rules about task decomposition, template-field mapping, batching strategy (per_atom/by_attr/open, chunk sizing), and ordering/parallelism.
Return concise natural-language rules (3–6) that teach how to plan the next task_matrix; do NOT restate content/schema as placeholders.
"""

critique_success_rules_user_prompt = """
## SUCCESS pattern_info LIST
{success_history}

## EXISTING HINTS:
{existing_rules}

Goal: Identify recurring batching/template/schema patterns that led to coverage/accuracy success. Keep hints general and concise.
Return operations (ADD/EDIT/REMOVE/AGREE) only, using the format:
""" + format_rules_operation_template

# merge rules
merge_rules_system_prompt = """You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from similar tasks. These insights may contain redundancy or overlap.

Your job is to **merge and consolidate similar insights**, and output a refined version that is **clear, actionable, and concise**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- The output should be easy to read and follow.

📝 Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

merge_rules_user_prompt = """
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} refined insights**.

As the summarizing agent, remove redundancies, combine similar ideas, and ensure clarity.

Your output:
"""

# MapReduce batching hint merge (overrides above definitions for self-evolution)
merge_rules_system_prompt = """You are an agent that merges overlapping batching insights for MapReduce tasks.
Keep each merged hint actionable for planning: task structure, template-to-row mapping, batching strategy (per_atom/by_attr/open and chunk sizing), and ordering/parallelism.
Remove content-level or placeholder-only hints. Output a numbered list of concise natural-language rules.
"""

merge_rules_user_prompt = """
## Current batching hints
{current_rules}

Please consolidate into no more than {limited_number} refined hints.
Start directly with the numbered list.
"""

# project insights according to agent's role
project_insights_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be given a specific agent **role** and a set of **general insights** that apply to all roles. 
Your task is to **adapt these general insights** into **personalized insights tailored to the given role**, helping the agent perform more effectively.
Make sure your output aligns with the role's background, responsibilities, and point of view.

NOTE - Your output should follow the below format:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_user_prompt: str = """
### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""

# project insights according to agent's role and trajectory
project_insights_with_traj_system_prompt: str = """
You are a strategy assistant. You will be given a trajectory that contains MapReduce tool inputs (task_matrix, template, json_schema, batch_strategy) and a list of historical hints.
Your job is to generate NEW, global, role-specific insights that identify where the current batching strategy can be improved and what direction to adjust next.
Base your insights on BOTH the current trajectory and the historical hints. Emphasize actionable adjustment levers (e.g., batch size, grouping logic, schema/template constraints, source reuse, verification flow). Avoid repeating the inputs verbatim or inventing facts.
Return no more than 3 insights total.

NOTE - Your output must strictly follow the format below:
1. Insight 1
2. Insight 2
3. Insight 3
"""

project_insights_with_traj_user_prompt: str = """
### Trajectory (MapReduce inputs and context)
{trajectory}

### Agent's Role:
{role}

### Historical Hints:
{insights}

### Your Output (Personalized, Global Insights for This Role):
"""



@dataclass
class ExpmemoryPrompt:
    generative_task_system_prompt = generative_task_system_prompt
    generative_task_user_prompt = generative_task_user_prompt
    extract_true_traj_system_prompt = extract_true_traj_system_prompt
    extract_true_traj_user_prompt = extract_true_traj_user_prompt
    finetune_insights_suffix = finetune_insights_suffix
    critique_compare_rules_system_prompt = critique_compare_rules_system_prompt
    critique_compare_rules_user_prompt = critique_compare_rules_user_prompt
    critique_success_rules_system_prompt = critique_success_rules_system_prompt
    critique_success_rules_user_prompt = critique_success_rules_user_prompt
    merge_rules_system_prompt = merge_rules_system_prompt
    merge_rules_user_prompt = merge_rules_user_prompt
    project_insights_system_prompt=project_insights_system_prompt
    project_insights_user_prompt=project_insights_user_prompt
    project_insights_with_traj_system_prompt=project_insights_with_traj_system_prompt
    project_insights_with_traj_user_prompt=project_insights_with_traj_user_prompt


ExpmemoryPrompts = ExpmemoryPrompt()
