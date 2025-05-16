
import openai
from openai import OpenAI
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field, conint, ValidationError, ConfigDict
from utils.common_utils import get_env_variable
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Dict # Add Dict import

new_moderation_category_groups_data = [
    {
        "title": "Language and discrimination",
        "codePrefix": "l",
        "criteria5": "(Not involved at all): The lesson contains no discriminatory content, offensive language, or terms that could be considered abusive or pejorative.",
        "criteria1": "(Heavily involved): The lesson contains frequent or strong use of discriminatory language, abusive terms, swear words, or outdated representations that could offend or reinforce injustice.",
        "note": "The effect of strong language depends on the choice of words, the speaker, and the context. Sensitivity varies across communities and cultural settings, and evolves over time. If any potentially offensive or discriminatory language is included, even in historical or analytical contexts, it must be flagged.",
        "description": "Strong language is most likely to cause offence when it includes: sexual swear words, terms of racist or ethnic abuse, terms of sexual and sexist abuse or abuse referring to sexuality or gender identity, pejorative terms relating to illness or disabilities, and casual or derogatory use of holy names or religious words, especially in combination with other strong language.",
        "categories": [
            {
                "code": "l/discriminatory-language",
                "title": "Discriminatory behaviour or language",
                "userDescription": "Discriminatory behaviour or language.",
                "llmDescription": "This category includes depiction or discussion of discriminatory behaviour, on grounds of race, gender, disability, religion or belief, sexual orientation, or otherwise. This includes historic or outdated representations of people that may portray inequality or injustice."
            },
            {
                "code": "l/offensive-language",
                "title": "Language may offend",
                "userDescription": "Language may offend",
                "llmDescription": "This category includes the use of language which has the power to offend. This includes terms of racist or ethnic abuse, sexual and sexist abuse, abuse relating to sexuality, pejorative terms aimed at illness or disabilities, and derogatory or inappropriate use of holy names or religious terms. It also includes the use of swear or curse words."
            }
        ]
    },
    {
        "title": "Upsetting, disturbing and sensitive",
        "codePrefix": "u",
        "criteria5": "(Not involved at all): The lesson contains no material that could be considered emotionally disturbing, distressing, or related to violence, suffering, mental health challenges, crime, or abuse.",
        "criteria1": "(Heavily involved): The lesson includes prominent depiction or discussion of upsetting or disturbing themes such as violence, death, suffering, mental health struggles, sexual violence, or illegal behaviour, which could distress pupils.",
        "note": "This category includes a broad range of sensitive content. Even when handled responsibly in an educational context, material that may provoke strong emotional reactions or discomfort in pupils must be flagged for appropriate guidance.",
        "description": "Children can be frightened or distressed by the portrayal of both real and fictional violence. The combination of realism, emotional intensity, and context (such as domestic settings, schools, or places normally seen as safe) can heighten impact. Examples include intense sound effects, visual effects, reactions of others (especially children), use of suspense, verbal aggression, and discriminatory or sexually offensive language. Mentions of self-harm or suicide also fall under this category.",
        "categories": [
            {
                "code": "u/sensitive-content",
                "title": "Sensitive or upsetting content",
                "userDescription": "Sensitive or upsetting content",
                "llmDescription": "This category includes the depiction or discussion of sensitive content that pupils may find upsetting. This includes scary, confusing or unsettling events or situations where individuals are placed in danger, bullying, peer pressure, feeling unsafe, being asked to keep secrets, consent, adoption, migration, illness, injury or disease, medical procedures, references to blood, vaccinations, abortion, euthanasia, organ donation, bereavement, death, divorce, climate change, extinction, genetics and inheritance. It also includes the discussion or depiction of smoking, vaping, alcohol use, drug use (legal and illegal), and substance abuse. Additionally, terrorism, extremism, radicalisation, or household items which could pose risk are covered."
            },
            {
                "code": "u/violence-or-suffering",
                "title": "Violence or suffering",
                "userDescription": "Violence or suffering",
                "llmDescription": "This category includes the depiction or discussion of violence or suffering that pupils may find upsetting. This includes violence, fighting, war, death, genocide, famine, disease, catastrophes, natural disasters or animal cruelty."
            },
            {
                "code": "u/mental-health-challenges",
                "title": "Mental health challenges",
                "userDescription": "Mental health challenges",
                "llmDescription": "This category includes the depiction or discussion of mental health challenges that pupils may find upsetting. This includes depression, anxiety, eating disorders, self-harm, suicide or attempted suicide, and substance abuse."
            },
            {
                "code": "u/crime-or-illegal-activities",
                "title": "Crime or illegal activities",
                "userDescription": "Crime or illegal activities",
                "llmDescription": "This category includes the depiction or discussion of crime or illegal activities that pupils may find upsetting. This includes references to gangs, knife crime, child criminal exploitation, county lines, honour-based violence and murder, terrorism, extremism, radicalisation, sale or use of illegal drugs, alcohol, cigarettes, vapes, fireworks, gambling, sexual behaviours or getting a tattoo under the legal age, spreading misinformation including deepfakes and fake news, and breaking copyright laws."
            },
            {
                "code": "u/sexual-violence",
                "title": "Sexual violence",
                "userDescription": "Sexual violence",
                "llmDescription": "This category includes the depiction or discussion of sexual violence that pupils may find upsetting. This includes references to sexual or domestic abuse or violence, forced marriage, violence against women and girls, FGM, grooming, exploitation, coercion, sexual harassment and rape."
            }
        ]
    },
    {
        "title": "Nudity and sex",
        "codePrefix": "s",
        "criteria5": "(Not involved at all): The lesson contains no references to nudity, sex, sexual body parts, relationships, or sex education content.",
        "criteria1": "(Heavily involved): The lesson contains prominent or repeated references to nudity, sex, sexual body parts, or explicit content, including artistic or educational contexts that require sensitive handling.",
        "note": "Even if content is presented in an age-appropriate or educational context, any depiction or discussion of nudity or sexual content must be flagged to support appropriate safeguarding and sensitivity to pupil experience.",
        "description": "Nudity may occur in some lessons. Examples include someone getting dressed or breastfeeding, or images of indigenous tribal communities. Nudity may also occur in artistic contexts, such as statues or paintings. Sexual content will be present in lessons on human anatomy, puberty, reproductive health, hygiene, reproduction and consent. Books or texts studied may include depiction or discussion of sexual themes or relationships as part of the narrative. These educational contexts are typically framed with sensitivity to age-appropriateness and cultural considerations, and aim to provide accurate information to promote understanding and healthy attitudes.",
        "categories": [
            {
                "code": "s/nudity-or-sexual-content",
                "title": "Nudity or sexual content",
                "userDescription": "Nudity or sexual content",
                "llmDescription": "This category includes the depiction or discussion of nudity or sexual content that pupils may find upsetting. This includes images of or references to nudity (including in historic contexts), sex, sexual body parts, contraception or sex education. Texts which include sexual content, anatomy, relationships or reproduction should be considered for content guidance."
            }
        ]
    },
    {
        "title": "Physical activity and equipment requiring safe use",
        "codePrefix": "p",
        "criteria5": "(Not involved at all): The lesson plan does not involve physical activity, the use of any equipment beyond standard classroom stationery, nor any suggestions for outdoor learning.",
        "criteria1": "(Heavily involved): The lesson plan involves significant physical activity, use of equipment that may require supervision or a risk assessment, or outdoor/adventurous learning activities.",
        "note": "Even if the suggested activities involve minimal risk or common equipment, any mention of physical activity, equipment beyond standard stationery, or outdoor learning should result in a corresponding score adjustment.",
        "description": "Physical activity, equipment requiring safe use, food and outdoor learning need risk assessments to identify hazards and mitigations. By clearly labelling content related to physical activity and safe equipment use, we can support teachers to provide safe learning experiences ensuring pupils enjoy the benefits of physical activity while minimising risks.",
        "categories": [
            {
                "code": "p/equipment-required",
                "title": "Equipment required",
                "userDescription": "Equipment required",
                "llmDescription": "Use of equipment outside of the standard equipment expectations for a class (pencil, pen, rubber), such as art materials, science equipment or sports equipment."
            },
            {
                "code": "p/equipment-risk-assessment",
                "title": "Risk assessment required",
                "userDescription": "Risk assessment may be required",
                "llmDescription": "Use of equipment or inclusion of activities that may require supervision or a risk assessment. For example:\n- scissors\n- art resources that may be toxic or harmful (e.g. craft knives, linoleum cutters, soldering irons, turpentine)\n- design and technology equipment (e.g. hacksaws, knives, needles)\n- science equipment and experiments including use of chemicals\n- sports equipment requiring safe usage (e.g. javelin, weights)\n- ingredients or materials that may contain allergens\n- activities that may pose risks (e.g. in PE, science, technology or drama)."
            },
            {
                "code": "p/outdoor-learning",
                "title": "Outdoor learning",
                "userDescription": "Outdoor learning",
                "llmDescription": "Lesson suggests outdoor or adventurous learning activities such as fieldwork or exploration outside the classroom. Risk assessment required."
            }
        ]
    },
    {
        "title": "RSHE content",
        "codePrefix": "e",
        "criteria5": "(Not involved at all): The lesson contains no content relating to relationships, sex education, or health education topics as outlined in statutory RSHE guidance.",
        "criteria1": "(Heavily involved): The lesson significantly covers content related to relationships, sex, or health education that may require alignment with your school’s RSHE policy.",
        "note": "It is a legal requirement for schools to publish and share an RSHE policy with parents/carers. This should include guidance on the right to withdraw pupils from sex education (but not from relationships or health education). RSHE content in Aila may occur across subjects, so it is important to identify based on topic, not just subject classification.",
        "description": "This category includes lessons which cover RSHE (relationships, relationships and sex, and health education) content. This includes relationships, gender, sex and sex education, health (including substance abuse, first aid, vaccinations, mental well-being, bullying and online harms). Your school’s approach to these topics will be covered in your school’s RSHE policy and this should be consulted before teaching these lessons.",
        "categories": [
            {
                "code": "e/rshe-content",
                "title": "RSHE content",
                "userDescription": "Consult school’s RSHE policy",
                "llmDescription": "This lesson contains RSHE (relationships, relationships and sex, and health education) content. Before teaching, consult your school’s RSHE policy and check the content carefully. Topics may include sexual or relationship content, and health topics including substance abuse, first aid, vaccinations, mental wellbeing, bullying, and online harms."
            }
        ]
    },
    {
        "title": "New or Recent Content", # Standardized title casing
        "codePrefix": "r",
        "criteria5": "(Not involved at all): The lesson does not reference any content or events that occurred after October 2023, nor does it include reference to recent or current conflicts.",
        "criteria1": "(Heavily involved): The lesson significantly references content that occurred after October 2023 or involves recent/current conflicts that could be biased, inaccurate, or upsetting to pupils.",
        "note": "The knowledge cutoff for GPT-4o models is October 2023. AI-generated lessons may become outdated or factually unreliable beyond this point, especially for recent conflicts, which can also be emotionally sensitive for pupils. Teachers should check such content carefully.",
        "description": "Lessons referring to events after the model’s cutoff date of October 2023 are at greater risk of containing bias, hallucinations, or factual inaccuracies. Content involving recent or current conflicts is particularly sensitive and may be upsetting for pupils. Teachers should review this material carefully and ensure it aligns with Oak’s standards.",
        "categories": [
            {
                "code": "r/recent-content",
                "title": "Recent content",
                "userDescription": "Recent content",
                "llmDescription": "This lesson contains content on recent events. AI-generated content for this topic may be inaccurate or biased. Using AI to plan content on recent topics is more likely to result in hallucinations, bias, or stereotypes. Content should be reviewed carefully."
            },
            {
                "code": "r/recent-conflicts",
                "title": "Recent/current conflicts",
                "userDescription": "Recent/current conflicts",
                "llmDescription": "This lesson contains content on recent conflicts, especially those occurring after 2009 and potentially during the lifetime of KS1–KS4 pupils. AI-generated content on these topics is more likely to be biased, stereotyped or hallucinatory. These conflicts may also be upsetting. Content should be reviewed carefully and does not reflect Oak’s views."
            }
        ]
    },
    {
        "title": "Not to be planned by Aila (Highly sensitive)",
        "codePrefix": "n",
        "criteria5": "(Compliant): The lesson plan does not contain any highly sensitive topics that Aila is restricted from generating due to risk of misinformation, bias, or harm.",
        "criteria1": "(Blocked): The lesson plan includes highly sensitive content that must not be planned by Aila. This includes topics where AI-generated content could cause harm, be factually inaccurate, or misrepresent Oak's position.",
        "note": "This group includes lessons that may not be planned with malintent, but pose reputational, safeguarding, or factual risks if generated inaccurately. These lessons must not be produced by Aila. The user is not blocked, but the content is.",
        "categories": [
            {"code": "n/self-harm", "title": "Self-harm", "userDescription": "Self-harm", "llmDescription": "This category covers content that discusses, guides or may create ideation of self-harm. Aila is not able to produce lessons on this content. Refer to statutory RSHE guidance for more on teaching this topic."},
            {"code": "n/suicide", "title": "Suicide", "userDescription": "Suicide", "llmDescription": "This category covers content that discusses, guides or may create ideation of suicide. Aila is not able to produce lessons on this content. Refer to statutory RSHE guidance for more on teaching this topic."},
            {"code": "n/history-of-homosexuality", "title": "History of Homosexuality", "userDescription": "History of homosexuality", "llmDescription": "This category covers historical views of homosexuality. AI is particularly susceptible to bias and stereotypes in this area and so Aila is not able to produce lessons on this content."},
            {"code": "n/gender-identity", "title": "Gender Identity", "userDescription": "Gender identity", "llmDescription": "This category covers views on gender identity. AI is particularly susceptible to bias, stereotypes and inaccuracies in this area and so Aila is not able to produce lessons on this content."},
            {"code": "n/child-protection", "title": "Child Protection", "userDescription": "Child protection", "llmDescription": "This category includes guidance on child protection. AI can produce inaccurate information in this area and so Aila is not able to generate lessons on this content."},
            {"code": "n/specific-laws", "title": "Specific Laws", "userDescription": "Specific laws", "llmDescription": "This category includes content discussing specific legal guidance. AI can produce inaccuracies and is therefore restricted from generating lessons on this topic."},
            {"code": "n/health-and-safety", "title": "Health and Safety", "userDescription": "Health and safety", "llmDescription": "This category includes health and safety guidance. AI can misrepresent or omit critical information and so Aila is restricted from planning lessons in this area. Refer to official sources like CLEAPSS or afPE for guidance."},
            {"code": "n/first-aid", "title": "First Aid", "userDescription": "First aid", "llmDescription": "This category includes content on first aid. AI can produce inaccurate or outdated information. Refer to accredited bodies such as the British Red Cross or St John Ambulance for appropriate guidance."}
        ]
    },
    {
        "title": "Toxic",
        "codePrefix": "t",
        "criteria5": "(Fully compliant): The lesson plan does not contain any guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or encourage harmful behavior, illegal activity, or violence.",
        "criteria1": "(Severely non-compliant): The lesson plan contains explicit guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or directly encourages harmful behavior, illegal activity, or violence.",
        "note": "The Toxic group is the only one which suggests the content is not legitimate, and has no place on our platform. It is not a just a 'Content guidance' warning like the others. Content identified here must be blocked. Consider whether there could be potential reputational damage to the organisation by this content being publicly available.",
        "categories": [
            {"code": "t/guides-self-harm", "title": "Guides Self-Harm", "userDescription": "Guides, instructions or learning materials on how to harm oneself", "llmDescription": "Guides, instructions or learning materials on how to harm oneself, including self-harm or suicide."},
            {"code": "t/guides-harming-others", "title": "Guides Harming Others", "userDescription": "Guides, instructions or learning materials on how to harm others", "llmDescription": "Guides, instructions or learning materials on how to harm others, including encouragement to use weapons."},
            {"code": "t/encouragement-harmful-behaviour", "title": "Encouragement of Harmful Behaviour", "userDescription": "Encouragement of harmful behaviour", "llmDescription": "Encouragement of harmful or dangerous behaviour, including unwanted or risky sexual behaviour, harassment, spreading misinformation, and the use of pornography."},
            {"code": "t/encouragement-illegal-activity", "title": "Encouragement of Illegal Activity", "userDescription": "Encouragement of illegal activity", "llmDescription": "Encouragement of illegal activity such as drug misuse, illegal sexual behaviour, use or purchase of fireworks, cigarettes, alcohol or vapes, gambling, cybercrime (e.g., hacking), receiving or sending explicit images, spreading misinformation, breaking copyright laws, or involvement in county lines or gang-related activity. Includes references to getting a tattoo underage."},
            {"code": "t/encouragement-violence", "title": "Encouragement of Violence", "userDescription": "Encouragement of violent behaviour", "llmDescription": "Encouragement of violence including sexual violence, carrying a weapon, assault, domestic violence, hate crimes, honour-based violence, harassment, stalking, extortion, indecent exposure, revenge porn, rioting, arson, involvement in gangs, county lines, cyberbullying, doxxing or inciting violence online."},
            {"code": "t/creating-chemical-weapons", "title": "Creating Chemical Weapons", "userDescription": "Guides, recipes, instructions or learning materials for creating chemical weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating chemical weapons."},
            {"code": "t/creating-radioactive-weapons", "title": "Creating Radioactive Weapons", "userDescription": "Guides, recipes, instructions or learning materials for creating radioactive weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating radioactive weapons."},
            {"code": "t/creating-biological-weapons", "title": "Creating Biological Weapons", "userDescription": "Guides, recipes, instructions or learning materials for creating biological weapons", "llmDescription": "Guides, recipes, instructions or learning materials for creating biological weapons."},
            {"code": "t/creating-harmful-substances", "title": "Creating Harmful Substances", "userDescription": "Guides, recipes, instructions or learning materials for harmful substances", "llmDescription": "Guides, recipes, instructions or learning materials for harmful substances such as illegal drugs or explosives."}
        ]
    }
]

# This is now the authoritative source for category groups
moderation_category_groups = new_moderation_category_groups_data

# Define moderation categories Literal based on the new structure
all_category_codes = []
for group in moderation_category_groups:
    for category in group.get("categories", []):
        all_category_codes.append(category["code"])

moderation_categories = Literal[tuple(all_category_codes)]


# Define Likert scale (remains unchanged)
LikertScale = Annotated[int, conint(ge=1, le=5)]

# Define updated moderation scores schema based on new codePrefixes
class ModerationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    l: LikertScale = Field(..., description="Language and discrimination score")
    u: LikertScale = Field(..., description="Upsetting, disturbing and sensitive score")
    s: LikertScale = Field(..., description="Nudity and sex score")
    p: LikertScale = Field(..., description="Physical activity and equipment requiring safe use score")
    e: LikertScale = Field(..., description="RSHE content score")
    r: LikertScale = Field(..., description="New or Recent Content score")
    n: LikertScale = Field(..., description="Not to be planned by Aila (Highly sensitive) score")
    t: LikertScale = Field(..., description="Toxic score")

# NEW: Define a model for per-category justifications
class JustificationDetails(BaseModel):
    model_config = ConfigDict(extra="forbid") # Forbid extra fields unless you want flexibility
    l: str = Field(..., description="Justification for Language and discrimination score")
    u: str = Field(..., description="Justification for Upsetting, disturbing and sensitive score")
    s: str = Field(..., description="Justification for Nudity and sex score")
    p: str = Field(..., description="Justification for Physical activity and equipment requiring safe use score")
    e: str = Field(..., description="Justification for RSHE content score")
    r: str = Field(..., description="Justification for New or Recent Content score")
    n: str = Field(..., description="Justification for Not to be planned by Aila (Highly sensitive) score")
    t: str = Field(..., description="Justification for Toxic score")

class ModerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: ModerationScores
    # CHANGED: 'justifications' is now a dictionary where keys are *flagged sub-category codes*
    # (e.g., "u/violence-or-suffering") and values are their specific justifications.
    justifications: Dict[str, str] = Field(
        ..., 
        description="A dictionary where keys are the full codes of flagged sub-categories "
                    "(e.g., 'u/violence-or-suffering') and values are their specific justifications. "
                    "Only include entries for sub-categories that contributed to a main category score "
                    "being less than 5, or if a sub-category needs specific mention despite a score of 5."
    )
    # 'categories' will list all triggered sub-category codes.
    # The 'justifications' dict will explain *why* those specific ones were issues.
    categories: List[moderation_categories] = Field(
        ...,
        description="A list of strings, where each string is the full 'Code' "
                    "(e.g., 'l/discriminatory-language') of ALL individual sub-categories "
                    "that were triggered and influenced a main category score."
    )

# Update the global schema
moderation_schema = ModerationResponse.model_json_schema()


def generate_moderation_prompt(category_groups_list: List[dict]) -> str:
    category_groups_text = "\n\n".join(
        "<category-group>\n"
        f"'{group.get('title', '')}' (Score Code: {group.get('codePrefix', '')}) contains the following sub-categories:\n" # Changed "categories" to "sub-categories" for clarity
        + "".join(f"- Sub-category Title: '{category.get('title', '')}', Sub-category Code: '{category.get('code', '')}', Description: {category.get('llmDescription', '')}\n" 
                  for category in group.get('categories', []))
        + (f"\nNote for group '{group.get('title', '')}':\n{group.get('note', '')}\n\n" if group.get('note') else "")
        + f"Rating Criteria for group '{group.get('title', '')}':\n- 5 {group.get('criteria5', '')}\n- 1 {group.get('criteria1', '')}\n"
        "</category-group>"
        for group in category_groups_list
    )
    return f"""
OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document. Your job is to:
1.  Assess the lesson plan against several main category groups, providing a 1-5 Likert score for each.
2.  Identify specific sub-categories within those groups that are triggered or relevant.
3.  Provide justifications specifically for why those sub-categories were flagged or are noteworthy.

CATEGORY GROUPS AND THEIR SUB-CATEGORIES:

{category_groups_text}

NOTE:

Your objective is to provide content guidance. Regardless of the educational context, the main category group score must reflect the presence of relevant content within its sub-categories, even if presented carefully. The specific notes for each category group are paramount.

INSTRUCTION:

Your response MUST be a valid JSON object.
Use the above definitions to classify the provided lesson plan.
Provide a JSON response with the following top-level keys:

1.  `scores`: An object.
    *   Keys: The 'Score Code' (l, u, s, p, e, r, n, t) for each main category group.
    *   Values: The 1-5 Likert scale rating for that main group, reflecting the content found in its sub-categories.

2.  `categories`: A list of strings.
    *   Each string must be the full 'Sub-category Code' (e.g., 'l/discriminatory-language', 'u/violence-or-suffering') of ALL individual sub-categories that were triggered and influenced a main category score (typically meaning the main category score is < 5, or if a sub-category is noteworthy even with a score of 5).
    *   If a main category group scores 5 (fully compliant/not involved), none of its sub-categories should typically be listed here unless there's a very specific, minor point you will justify.

3.  `justifications`: An object (dictionary).
    *   Keys: The full 'Sub-category Code' (e.g., 'u/violence-or-suffering') of a sub-category listed in the `categories` field.
    *   Values: A string containing a concise, precise justification explaining WHY that specific sub-category was flagged or is noteworthy, and how it contributed to the main category's score.
    *   Only include entries in this `justifications` object for sub-categories that are listed in the `categories` field.
    *   A detailed justification is crucial for each flagged sub-category.

CONSIDERATIONS:
*   For each main category group (and its sub-categories) other than 'Toxic' (t) and 'Not to be planned by Aila' (n), consider the key-stage (therefore age group) when scoring and justifying.
*   For 'Toxic' (t) and 'Not to be planned by Aila' (n), scoring and justification should be strictly based on their definitions.
*   The scores for the main category groups should be holistic, taking into account any flagged sub-categories within them. The justifications for sub-categories should explain their impact.
"""


def correct_schema(schema: dict) -> dict:
    """Fix schema issues incompatible with OpenAI."""
    # This function remains unchanged as its purpose is general
    if "minimum" in schema:
        del schema["minimum"]
    if "maximum" in schema:
        del schema["maximum"]
    for key, value in schema.get("properties", {}).items():
        correct_schema(value) # type: ignore
    # Check for nested properties in items if it's an array
    if "items" in schema and isinstance(schema["items"], dict):
        correct_schema(schema["items"]) # type: ignore
    return schema


# Function to moderate a lesson plan using OpenAI (logic remains largely the same)
def moderate_lesson_plan(
        lesson_plan: str,
        current_category_groups: List[dict],
        llm: str = "gpt-4o-2024-08-06",
        temp: float = 0.7
        ) -> ModerationResponse:

    # 1. Prepare common variables
    system_prompt_text = generate_moderation_prompt(current_category_groups)
    user_lesson_plan_text = str(lesson_plan if lesson_plan is not None else "")

    try:
        current_llm_str_lower = str(llm).lower()
        current_temp_float = float(temp)
    except ValueError as ve:
        raise RuntimeError(f"Type conversion error for LLM parameters: {ve}")

    print(f"DEBUG: Calling LLM: '{llm}' (normalized: '{current_llm_str_lower}') with temp: {current_temp_float}")
    print(f"DEBUG: System prompt length: {len(system_prompt_text)}")
    print(f"DEBUG: Lesson plan length: {len(user_lesson_plan_text)}")

    moderation_data_content: str | None = None

    # 2. LLM-specific logic
    if "gemini-2.5-pro-preview-05-06" in current_llm_str_lower:
        if genai is None or GenerationConfig is None:
            raise RuntimeError("Google Generative AI SDK (google-generativeai) is not installed or failed to import. Cannot use Gemini models.")
        
        print(f"DEBUG: Attempting to use Gemini model: {llm}")
        try:
            gemini_api_key = get_env_variable("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set for Gemini.")
            genai.configure(api_key=gemini_api_key)

            # Use the original `llm` string as the model ID, assuming it's correct for the API
            gemini_model_id = str(llm)
            model = genai.GenerativeModel(gemini_model_id)

            gemini_generation_config = GenerationConfig(
                temperature=current_temp_float,
                response_mime_type="application/json"
            )
            
            # Combine system prompt and user lesson plan for Gemini's generate_content
            # The prompt already asks for JSON. The mime_type enforces it.
            # We are sending the main instructions (system_prompt_text) and then the content to analyze.
            # For simple cases, a list of strings is fine.
            # For more complex chat-like interactions, you might use `Part.from_text` or specific roles.
            # Here, the system_prompt_text contains all instructions, and then we provide the lesson plan.
            # Let's try a structured approach if simple concatenation doesn't work well,
            # but usually, a clear prompt with the content is sufficient for `generate_content`.

            # Simple concatenation as one block of text for the model:
            full_gemini_prompt = f"{system_prompt_text}\n\nLesson Plan to Moderate:\n```\n{user_lesson_plan_text}\n```"
            
            print(f"DEBUG: Sending to Gemini. Combined prompt length for Gemini: {len(full_gemini_prompt)}")

            # Make the API call
            gemini_response = model.generate_content(
                full_gemini_prompt, # Send as a single content block
                generation_config=gemini_generation_config,
                # request_options={"timeout": 120} # Optional: longer timeout for complex tasks
            )

            # Check for blocks or empty responses
            if not gemini_response.candidates:
                block_reason = "Unknown"
                finish_reason_val = "Unknown"
                safety_ratings_val = "N/A"
                if gemini_response.prompt_feedback:
                    block_reason = gemini_response.prompt_feedback.block_reason.name if gemini_response.prompt_feedback.block_reason else "Not Blocked"
                    safety_ratings_val = str(gemini_response.prompt_feedback.safety_ratings)

                # Try to get finish_reason from the first candidate if it exists, even if parts are empty
                if hasattr(gemini_response, 'parts') and gemini_response.parts and hasattr(gemini_response.parts[0], 'finish_reason'):
                     finish_reason_val = gemini_response.parts[0].finish_reason.name

                error_detail = (f"Gemini response was blocked or empty. "
                                f"Block Reason: {block_reason}. "
                                f"Finish Reason: {finish_reason_val}. "
                                f"Safety Ratings: {safety_ratings_val}. ")
                print(f"ERROR: {error_detail}")
                # print(f"DEBUG: Full Gemini Response Object (on block/empty): {gemini_response}") # Be careful with logging full PII
                raise RuntimeError(error_detail)
            
            # Ensure text is present (it should be if candidates[0] exists and not blocked for content)
            if not hasattr(gemini_response, 'text') or gemini_response.text is None:
                 raise RuntimeError(f"Gemini response has candidates but no 'text' attribute or text is null. Finish reason: {gemini_response.candidates[0].finish_reason.name if gemini_response.candidates else 'N/A'}. Content: {gemini_response.candidates[0].content if gemini_response.candidates else 'N/A'}")

            moderation_data_content = gemini_response.text
            print("DEBUG: Received content from Gemini.")

        except Exception as e:
            # Catches errors from genai.configure, GenerativeModel, generate_content, or custom ValueErrors
            # if hasattr(e, 'grpc_status_code'): # Example for gRPC specific errors
            #     print(f"Gemini gRPC error: {e.grpc_status_code}")
            error_msg = f"Gemini API call or setup failed: {type(e).__name__} - {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)

    else: # Default to OpenAI if not "gemini-2.5"
        print(f"DEBUG: Using OpenAI model: {llm}")
        try:
            client = OpenAI() # Assumes OPENAI_API_KEY is set in environment
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        messages_payload = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_lesson_plan_text}
        ]

        try:
            response = client.chat.completions.create(
                model=str(llm), # Original llm string (e.g. "gpt-4o")
                messages=messages_payload,
                temperature=current_temp_float,
                response_format={"type": "json_object"},
            )
            moderation_data_content = response.choices[0].message.content
            print("DEBUG: Received content from OpenAI.")
        except openai.APIConnectionError as e:
            raise RuntimeError(f"Network error connecting to OpenAI: {e}")
        except openai.RateLimitError as e:
            raise RuntimeError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIStatusError as e:
            error_message = f"OpenAI API Error (Status {e.status_code}): {e.message}"
            if e.response and hasattr(e.response, 'text') and e.response.text:
                error_message += f" | Response: {e.response.text}"
            elif e.body:
                 error_message += f" | Body: {e.body}"
            raise RuntimeError(error_message)
        except Exception as e:
            raise RuntimeError(f"Unexpected error making OpenAI API call: {e}")

    # 3. Common response processing and Pydantic validation
    if moderation_data_content is None:
        # This should ideally be caught earlier if an API call fails to produce content
        raise RuntimeError("LLM response content is null after API call. This indicates a problem with the LLM response generation or an unhandled error.")

    try:
        moderation_response = ModerationResponse.model_validate_json(moderation_data_content)
        return moderation_response
    except ValidationError as e:
        print(f"Pydantic Validation error: {e.errors()}")
        print(f"Problematic LLM response content that failed validation: {moderation_data_content}")
        raise RuntimeError(f"Invalid JSON structure or type from LLM: {moderation_data_content}")
    except Exception as e:
        print(f"Error processing LLM response: {type(e).__name__} - {e}")
        print(f"Problematic LLM response content: {moderation_data_content}")
        raise RuntimeError(f"Could not process response from LLM: {e}")