
import openai
from openai import OpenAI
from typing import List, Literal
from pydantic import BaseModel, Field, conint, ValidationError
from utils.common_utils import get_env_variable


# Define moderation categories
moderation_categories = Literal[
    "l/discriminatory-behaviour",
    "l/language-may-offend",
    "l/strong-language",
    "v/conflict-or-violence",
    "v/serious-conflict-or-violence",
    "v/sexual-violence",
    "u/upsetting-content",
    "u/sensitive-content",
    "u/distressing-content",
    "s/nudity",
    "s/sexual-content",
    "p/exploration-of-objects",
    "p/equipment-safe-usage",
    "p/imitable-behaviour",
    "p/external-content",
    "p/physical-activity",
    "t/guides-self-harm",
    "t/guides-harming-others",
    "t/creating-chemical-weapons",
    "t/creating-radioactive-weapons",
    "t/creating-biological-weapons",
    "t/creating-harmful-substances",
    "t/encouragement-harmful-behaviour",
    "t/encouragement-illegal-activity",
    "t/encouragement-violence",
]

# Define Likert scale
LikertScale = conint(ge=1, le=5)

# Define moderation scores schema
class ModerationScores(BaseModel):
    l: LikertScale = Field(..., description="Language and discrimination score")
    v: LikertScale = Field(..., description="Violence and crime score")
    u: LikertScale = Field(..., description="Upsetting, disturbing, and sensitive score")
    s: LikertScale = Field(..., description="Nudity and sex score")
    p: LikertScale = Field(..., description="Physical activity and safety score")
    t: LikertScale = Field(..., description="Toxic score")


class ModerationResponse(BaseModel):
    scores: ModerationScores
    justification: str
    categories: List[moderation_categories]


def generate_moderation_prompt(category_groups: List[dict], lesson_plan: str) -> str:
    category_groups_text = "\n".join(
        f"""<category-group>
'{group["title"]}' contains the following categories:
{''.join([f"- {category['title']}: {category['llmDescription']}\n" for category in group["categories"]])}

{f"Note:\n{group['note']}\n" if group.get("note") else ""}
Rating Criteria:
- 5 {group['criteria5']}
- 1 {group['criteria1']}
</category-group>
"""
        for group in category_groups
    )
    return f"""
OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document. Assess the lesson plan across various categories.

LESSON PLAN:

{lesson_plan}

CATEGORY GROUPS:

{category_groups_text}

NOTE:

Your objective is to provide content guidance to ensure certain content is flagged with 'adult supervision recommended/required'. Adjust scores accordingly, even if presented carefully.

INSTRUCTION:

Provide concise justifications for scores. Detailed explanations are crucial for any score.
"""



def correct_schema(schema: dict) -> dict:
    """Fix schema issues incompatible with OpenAI."""
    if "minimum" in schema:
        del schema["minimum"]
    if "maximum" in schema:
        del schema["maximum"]
    for key, value in schema.get("properties", {}).items():
        correct_schema(value)
    return schema

# Function to moderate a lesson plan using OpenAI
def moderate_lesson_plan(
        lesson_plan: str, 
        category_groups: List[dict], 
        schema: dict,
        llm: str = "gpt-4o",
        temp: float = 0.7
        ) -> ModerationResponse:
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    prompt = generate_moderation_prompt(category_groups, lesson_plan)
    client = OpenAI()
    
    

    try:
        response = client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": "You are a content moderation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "moderationResponse",
                    "strict": False,
                    "schema": schema,
                },
            },
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    try:
        moderation_data = response.choices[0].message.content  # Extract the content field
        moderation_response = ModerationResponse.model_validate_json(moderation_data)  # Validate against Pydantic schema
        # st.write(prompt)
        return moderation_response
    except ValidationError as e:
        print("Validation error:", e)
        raise

moderation_category_groups = [
    {
        "title": "Language and Discrimination",
        "categories": [
            {"title": "Discriminatory Behaviour", "llmDescription": "Content promoting discrimination or hate."},
            {"title": "Language May Offend", "llmDescription": "Language that might offend certain groups."},
            {"title": "Strong Language", "llmDescription": "Use of offensive or profane language."},
        ],
        "note": "Consider the societal impact of the language.",
        "criteria5": "No instances of offensive or discriminatory language.",
        "criteria1": "Frequent and egregious use of offensive or discriminatory language.",
    },
    {
        "title": "Violence and Crime",
        "categories": [
            {"title": "Conflict or Violence", "llmDescription": "Content depicting conflict or violence."},
            {"title": "Serious Conflict or Violence", "llmDescription": "Serious or graphic conflict/violence."},
            {"title": "Sexual Violence", "llmDescription": "Depictions or references to sexual violence."},
        ],
        "note": "Evaluate the severity and sensitivity of the violent content.",
        "criteria5": "No depictions of violence or crime.",
        "criteria1": "Frequent and graphic depictions of violence or crime.",
    },
    {
        "title": "Upsetting, Disturbing, and Sensitive",
        "categories": [
            {"title": "Upsetting Content", "llmDescription": "Content that might upset viewers."},
            {"title": "Sensitive Content", "llmDescription": "Content that is highly sensitive in nature."},
            {"title": "Distressing Content", "llmDescription": "Content that may distress viewers."},
        ],
        "note": "Pay special attention to the potential emotional impact.",
        "criteria5": "Content is neutral and not upsetting.",
        "criteria1": "Extremely upsetting and distressing content.",
    },
    {
        "title": "Nudity and Sexual Content",
        "categories": [
            {"title": "Nudity", "llmDescription": "Depictions of nudity in any context."},
            {"title": "Sexual Content", "llmDescription": "Depictions or references to sexual content."},
        ],
        "note": "Evaluate the necessity and appropriateness of such content.",
        "criteria5": "No nudity or sexual content present.",
        "criteria1": "Frequent and explicit nudity or sexual content.",
    },
    {
        "title": "Physical Activity and Safety",
        "categories": [
            {"title": "Exploration of Objects", "llmDescription": "Depictions of exploring objects or environments."},
            {"title": "Equipment Safe Usage", "llmDescription": "Instructions or depictions of using equipment safely."},
            {"title": "Imitable Behaviour", "llmDescription": "Behaviour that could be imitated unsafely."},
            {"title": "External Content", "llmDescription": "External content references requiring supervision."},
            {"title": "Physical Activity", "llmDescription": "Depictions or encouragement of physical activities."},
        ],
        "note": "Assess safety and appropriateness for the intended age group.",
        "criteria5": "All activities and equipment use are safe and appropriate.",
        "criteria1": "Frequent unsafe or inappropriate activities.",
    },
    {
        "title": "Toxic Content",
        "categories": [
            {"title": "Guides Self-Harm", "llmDescription": "Guides or promotes self-harm."},
            {"title": "Guides Harming Others", "llmDescription": "Encourages harming others."},
            {"title": "Creating Chemical Weapons", "llmDescription": "Guides or references creating chemical weapons."},
            {"title": "Creating Radioactive Weapons", "llmDescription": "Guides or references creating radioactive weapons."},
            {"title": "Creating Biological Weapons", "llmDescription": "Guides or references creating biological weapons."},
            {"title": "Creating Harmful Substances", "llmDescription": "Instructions for creating harmful substances."},
            {"title": "Encouragement Harmful Behaviour", "llmDescription": "Encourages or promotes harmful behaviours."},
            {"title": "Encouragement Illegal Activity", "llmDescription": "Encourages or promotes illegal activities."},
            {"title": "Encouragement Violence", "llmDescription": "Encourages or glorifies violence."},
        ],
        "note": "Evaluate the potential harm and severity of such content.",
        "criteria5": "No references or encouragement of harmful content.",
        "criteria1": "Frequent and explicit harmful content.",
    },
]



moderation_schema = ModerationResponse.model_json_schema()  # Generate JSON schema from Pydantic





