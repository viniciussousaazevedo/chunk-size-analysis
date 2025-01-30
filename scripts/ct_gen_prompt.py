import json

def get_prompt(us_id, us_title, us_description, acceptance_criteria):
    with open("./data/ct_gen/us/template.json", "r") as template_file:
        template = json.load(template_file)

    return f"""
    Read the User Story below (surrounded by triple backticks):
```
    US ID: {us_id}
    US TITLE: {us_title}
    US DESCRIPTION: {us_description}
    ACCEPTANCE CRITERIA: {acceptance_criteria}
```
Now, follow the steps provided below:
1- Understand the purpose of the User Story.
    1.1- Determine if the User Story is about implementing a new feature, fixing a bug, or achieving some other goal;
    1.2- Use the title and description to make this determination.
2- Only after finishing step 1, try to understand the acceptance criteria. Try to divide it in blocks, find what each block is trying to verify, where you might have to go to test it, and what is probably the final results.
3- Read all the context provided by RAG to better understand the user story and acceptance criteria
4- Only after finishing step 3, create black box test cases based on the User Story concept and try to do for each acceptance criteria in the format of user interface tests.
    4.1- Ask yourself: “Does this criterion imply multiple scenarios?”. If the answer is yes, when needed, separate the multiple parts that need checking in one acceptance criteria into different tests.
    4.2- If you do not fully understand the concept of the User Story or an Acceptance Criteria, simply say “I don’t know how to proceed” instead of made-up data. 
    4.3- Your output of must be a single JSON file containing a list with test cases following the pattern below (surrounded by triple backticks). The JSON output file must follow the template below:
    4.4- Try to be the most specific possible when describing an "action needed" or "expected result" of a test. To do this, use the context you have added in terms of context.
```
    {template}
```
Here is a quick description about each field in this template:
    1. "ID": the ID generated for this specific test case. You are the one responsible for generating this ID in the pattern “CTXY”, where “X” is the sequential number of each test case and “Y” is the related US ID. Example: If a certain test case is the first in the generation order and it is about the US with the ID “PSM-01”, then the ID for this test case is “CT01PSM-01”;
    2. "title": the title for this test case, summarizing in a short single sentence its objective;
    3. "description": a long, multi-sentence description about what this test case is about, its objective and goals. The description must include the fields 'Action needed' and 'Expected output'.
    4. "acceptance_criteria_related": the number of the acceptance criteria which this test case is related to (in terms of list index. For example, if a certain test case is related to the first acceptance criteria in the list provided, then the value of this field should be "1").
    """