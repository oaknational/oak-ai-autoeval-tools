import openai
from openai import OpenAI
from typing import List, Literal, Type, Dict
from pydantic import BaseModel, Field, conint, ValidationError, create_model
import json
import re
from dotenv import load_dotenv
import os
from utils.db_scripts import get_prompts

def get_env_variable(name: str) -> str:
    load_dotenv()
    value = os.getenv(name)
    if value is None:
        raise OSError(f"Missing environment variable: {name}")
    return value

def format_category_title(title: str) -> str:
    """Format category title to lowercase with underscores."""
    return re.sub(r'\W+', '_', title.lower())

def get_custom_category_groups():
    prompts = get_prompts()

    # Filter prompts to show rows containing "mit" in the prompt_title
    prompts = prompts[prompts['prompt_title'].str.contains("mit", case=False, na=False)]

    # Filter prompts to show rows with output_format 'Score'
    prompts = prompts[prompts['output_format'] == 'Score']

    def custom_abbreviate_prompt_title(title: str) -> str:
        # Remove the "mit-" prefix
        title_without_prefix = title.replace("mit-", "")
        # Split the remaining text by '-'
        parts = title_without_prefix.split('-')
        
        if len(parts) < 3:
            return title_without_prefix  # Return as is if insufficient parts

        # Keep the first two parts
        main_part = "-".join(parts[:2])
        
        # Custom abbreviation logic: first letter of each remaining word, except the first two parts
        abbreviation = ''.join(word[0] for word in parts[2:])
        
        return f"{main_part}-{abbreviation}"

    # Apply the transformation function
    prompts['abbreviated_title'] = prompts['prompt_title'].apply(custom_abbreviate_prompt_title)

    # st.dataframe(prompts)

    category_groups = {}
    for _, row in prompts.iterrows():
        category_title = row['abbreviated_title']
        category_name = row['prompt_title']
        

        if category_title not in category_groups:
            category_groups[category_title] = {"title": category_name, "categories": []}

        category_groups[category_title]["categories"].append({
            "title": category_title,
            "llmDescription": row['prompt_objective']+' '+ row['rating_instruction'] +' '+ row['general_criteria_note'] +' '+ row['rating_criteria'],
        })

    # Convert the dictionary to a list format
    custom_category_groups = list(category_groups.values())
    return custom_category_groups


def generate_merged_eval_prompt(category_groups: List[dict], lesson_plan: str) -> str:
    category_groups_text = "\n".join(
        f"<category-group>\n'{group.get('title', '')}' contains the following categories:\n" +
        "".join(f"- {category.get('title', '')}: {category.get('llmDescription', '')}\n"
                for category in group.get('categories', [])) +
        f"Rating Criteria:\n- 5 {group.get('criteria5', '')}\n- 1 {group.get('criteria1', '')}\n"
        "</category-group>"
        for group in category_groups
    )
    return f"""
OBJECTIVE:

You are an educational expert examining a lesson plan document. Your task is to evaluate the lesson plan document across various categories.

LESSON PLAN:

{lesson_plan}

CATEGORY GROUPS:

{category_groups_text}

INSTRUCTION:

Your justification should be concise, precise, and directly support your rating.
"""

def correct_schema(schema: dict) -> dict:
    if "minimum" in schema:
        del schema["minimum"]
    if "maximum" in schema:
        del schema["maximum"]
    for key, value in schema.get("properties", {}).items():
        correct_schema(value)
    return {
        "name": "merged_eval_response",
        "schema": schema
    }

def merged_eval_lesson_plan(lesson_plan: str, merged_eval_category_groups: List[dict], MergedEvalScores: Type[BaseModel], llm: str = "gpt-4o", temp: float = 0.7) -> BaseModel:
    category_identifiers = [format_category_title(cat.get('title', '')) for group in merged_eval_category_groups for cat in group.get('categories', [])]

    if not category_identifiers:
        raise ValueError("No valid category titles found in merged_eval_groups")

    CategoriesLiteral = Literal[tuple(category_identifiers)]  # type: ignore

    MergedEvalResponse = create_model(
        'MergedEvalResponse',
        scores=(MergedEvalScores, Field(...)),
        justification=(str, Field(...)),
        categories=(List[CategoriesLiteral], Field(...)),
        __base__=BaseModel
    )

    schema = correct_schema(MergedEvalResponse.model_json_schema())
    prompt = generate_merged_eval_prompt(merged_eval_category_groups, lesson_plan)

    client = OpenAI(api_key=get_env_variable("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": "You are a content evaluation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            response_format={"type": "json_schema", "json_schema": schema}
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    raw_response = response.choices[0].message.content
    merged_eval_response = MergedEvalResponse.model_validate_json(raw_response)
  

    return response, merged_eval_response

def generate_custom_scores_model(merged_eval_category_groups: List[Dict]) -> Type[BaseModel]:
    fields = {}
    for group in merged_eval_category_groups:
        for category in group.get("categories", []):
            category_id = format_category_title(category.get("title", ""))
            description = category.get("llmDescription", "No description available")
            fields[category_id] = (conint(ge=1, le=5), Field(..., description=description))

    CustomScores = create_model("CustomScores", **fields)
    return CustomScores

