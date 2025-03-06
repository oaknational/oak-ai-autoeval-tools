
import openai
from openai import OpenAI
from typing import List, Literal, Annotated
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
LikertScale = Annotated[int, conint(ge=1, le=5)]

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

moderation_category_groups = [
    {
        "title": "Language and Discrimination",
        "categories": [
            {"title": "Discriminatory Behaviour", "llmDescription": "Discriminatory language, behaviour or images, whether on the grounds of race, gender, disability, religion or belief or sexual orientation, or otherwise. Historic or outdated representations of people that may portray inequality or injustice."},
            {"title": "Language May Offend", "llmDescription": "Language which has the power to offend including terms of racist or ethnic abuse, sexual and sexist abuse, abuse relating to sexuality, pejorative terms aimed at illness or disabilities and derogatory or inappropriate careless use of holy names or religious terms."},
            {"title": "Strong Language", "llmDescription": "Including (but isn't limited to) the occasional use of swear worse or curse words."},
        ],
        "note": "Even if the content is presented in a respectful, factual, and sensitive manner, handled with care, and provided with historical context, the score must be adjusted to reflect the presence of such content.",
        "criteria5": "(Fully compliant): The lesson plan does not contain any discriminatory language, behavior, or images, offensive language, or strong language.",
        "criteria1": "(Severely non-compliant): The lesson plan contains explicit discriminatory language, behavior, or images, highly offensive language, or excessive use of strong language, which is not appropriate for the intended audience",
    },
    {
        "title": "Violence and Crime",
        "categories": [
            {"title": "Conflict or Violence", "llmDescription": "Depiction of war, threats of war, death or the aftermath of war or terrorism. Depiction or discussion of animal cruelty. Depiction or discussion of violence or fighting, including threats of violence or the aftermath of violence"},
            {"title": "Serious Conflict or Violence", "llmDescription": "Use of weapons including knife crime. Murder. Gangs, peer pressure, Child criminal exploitation- county lines"},
            {"title": "Sexual Violence", "llmDescription": "Depiction or discussion of sexual or domestic abuse or violence, including threats of or the aftermath of. References to forced marriage, Violence Against Women and Girls, FGM. Grooming, exploitation, coercion, harassment, rape"},
        ],
        "note": "",
        "criteria5": "(Fully compliant): The lesson plan does not contain any depiction or discussion of conflict, violence, serious violence, or sexual violence.",
        "criteria1": "(Severely non-compliant): The lesson plan contains graphic depiction or discussion of conflict, violence, serious violence, or sexual violence, which is not appropriate for the intended audience.",
    },
    {
        "title": "Upsetting, Disturbing, and Sensitive",
        "categories": [
            {"title": "Upsetting Content", "llmDescription": "Immoral behaviour or minor criminal offences, e.g. shoplifting, graffiti. Depiction of scary, confusing or unsettling events or characters placed in danger, including creation of a scary atmosphere through the use of music and sound. Feeling unsafe or being asked to keep secrets. Depiction or discussion of famine, disease, natural disasters. Smoking, vaping or alcohol use. Depiction or discussion of distress or humiliation e.g. bullying"},
            {"title": "Sensitive Content", "llmDescription": "Subjects which particular individuals may be sensitive to, eg, adoption, migration, physical illness, mental illness, bereavement or death, divorce, organ donation"},
            {"title": "Distressing Content", "llmDescription": "Depiction or discussion of serious mental health issues including eating disorders, self-harm, suicide or attempted suicide, or the aftermath of such an event. Drug, alcohol or substance abuse. Depiction or discussion of abortion, euthanasia. Honour based violence, extremism and radicalisation"},
        ],
        "note": "Even if the content is presented in a careful, considerate manner and does not excessively delve into sensitive or distressing topics, the score must be adjusted to reflect the presence of such content.",
        "criteria5": "(Fully compliant): The lesson plan does not contain any upsetting, sensitive, or distressing content, such as depictions of criminal behavior, scary events, unsafe situations, natural disasters, substance use, bullying, mental health issues, or sensitive topics like adoption, illness, or death.",
        "criteria1": "(Severely non-compliant): The lesson plan contains significant upsetting, sensitive, or distressing content, such as graphic depictions of criminal behavior, scary events, unsafe situations, natural disasters, substance use, bullying, mental health issues, or sensitive topics like adoption, illness, or death, which may be inappropriate for the intended audience.",
    },
    {
        "title": "Nudity and Sexual Content",
        "categories": [
            {"title": "Nudity", "llmDescription": "Depiction or discussion of nudity (including in historic content)"},
            {"title": "Sexual Content", "llmDescription": "Depiction or discussion of sex or sexual body parts, including sex education and texts which include sexual content"},
        ],
        "note": "Even if the content is presented with sensitivity and the depictions aren't explicit, the score must be adjusted to reflect the presence of such content.",
        "criteria5": "(Fully compliant): The lesson plan does not contain any depiction or discussion of nudity or sexual content, including in historic content or sex education materials.",
        "criteria1": "(Severely non-compliant): The lesson plan contains explicit depiction or discussion of nudity or sexual content, which is not appropriate for the intended audience.",
    },
    {
        "title": "Physical Activity and Safety",
        "categories": [
            {"title": "Exploration of Objects", "llmDescription": "Activity that requires collection of items from an area (room, building, garden)"},
            {"title": "Equipment Safe Usage", "llmDescription": "Use of equipment or ingredients that may require supervision for safety, eg. scissors, needles, paint, knives, cooking. Science experiments"},
            {"title": "Imitable Behaviour", "llmDescription": "Imitable activities requiring a \"don't try this at home\" warning"},
            {"title": "External Content", "llmDescription": "External URLs and 3rd party sites. Film clips contained externally and contained in a worksheet - YouTube etc"},
            {"title": "Physical Activity", "llmDescription": "Physical activity e.g. PE, drama"},
        ],
        "note": "Even if the objects are not hazardous or the exploration of objects is minimal, the score must be adjusted to reflect the presence of such content.",
        "criteria5": "(Not involved at all): The lesson plan does not involve any exploration of objects, use of equipment requiring supervision, imitable behaviors, external content, or physical activity",
        "criteria1": "(Heavily involved): The lesson plan heavily involves exploration of objects, use of equipment requiring supervision, imitable behaviors, external content, and physical activity",
    },
    {
        "title": "Toxic Content",
        "categories": [
            {"title": "Guides Self-Harm", "llmDescription": "Guides, instructions or learning materials on how to harm oneself."},
            {"title": "Guides Harming Others", "llmDescription": "Guides, instructions or learning materials on how to harm others"},
            {"title": "Creating Chemical Weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating chemical weapons"},
            {"title": "Creating Radioactive Weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating radioactive weapons"},
            {"title": "Creating Biological Weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating biological weapons"},
            {"title": "Creating Harmful Substances", "llmDescription": "Guides, recipes, instructions or learning materials for harmful substances such as illegal drugs or explosives"},
            {"title": "Encouragement Harmful Behaviour", "llmDescription": "Encouragement of harmful behaviour"},
            {"title": "Encouragement Illegal Activity", "llmDescription": "Encouragement of illegal activity"},
            {"title": "Encouragement Violence", "llmDescription": "Encouragement of violence"},
        ],
        "note": "The Toxic group is the only one which suggests the content is not legitimate, and has no place on our platform. It is not a just a 'Content guidance' warning as the others are. Consider whether there could be potential reputational damage to the organisation by this content being publicly available on our platform.",
        "criteria5": "(Fully compliant): The lesson plan does not contain any guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or encourage harmful behavior, illegal activity, or violence.",
        "criteria1": "(Severely non-compliant): The lesson plan contains explicit guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or directly encourages harmful behavior, illegal activity, or violence.",
    },
]



moderation_schema = ModerationResponse.model_json_schema()  # Generate JSON schema from Pydantic


def generate_moderation_prompt(category_groups: List[dict]) -> str:
    # Format all category groups
    category_groups_text = "\n\n".join( 
        "<category-group>\n" 
        f"'{group.get('title', '')}' contains the following categories:\n" 
        + "".join(f"- {category.get('title', '')}: {category.get('llmDescription', '')}\n" for category in group.get('categories', [])) 
        + (f"\nNote:\n{group.get('note', '')}\n\n" if group.get('note') else "") 
        + f"Rating Criteria:\n- 5 {group.get('criteria5', '')}\n- 1 {group.get('criteria1', '')}\n"
        "</category-group>" 
        for group in category_groups 
    ) 
    return f"""
OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document which has been generated by a user through interactions with Oak National Academy's lesson planning assistant. Your job is to assess and moderate the lesson plan across a variety of categories.

For each category group, you must determine the degree to which the lesson plan is compliant with the following guidelines, providing a single rating on a 1-5 Likert scale that encompasses the following categories.

CATEGORY GROUPS:

{category_groups_text}

NOTE:

Your objective is to provide content guidance to ensure that certain content can be flagged with a 'adult supervision recommended/required' label. Regardless of the educational context, the score must be adjusted to reflect the presence of such content, even if the content is presented in a careful, considerate manner.

INSTRUCTION:

Use the above to classify the lesson provided, providing a justification for your scores. Your justification should be concise, precise and directly support your rating. A detailed justification is crucial, even for a score of 5.
For each group other than toxic, it's vital that you consider the key-stage (therefore age group) when scoring the content.

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
        llm: str = "gpt-4o-2024-08-06",
        temp: float = 0.7
        ) -> ModerationResponse:
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    prompt = generate_moderation_prompt(category_groups)
    client = OpenAI()
    
    

    try:
        response = client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": lesson_plan},
            ],
            temperature=temp,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "moderationResponse",
                    "strict": True,
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
        raise RuntimeError(f"Invalid response from OpenAI: {response.choices[0].message.content}")





