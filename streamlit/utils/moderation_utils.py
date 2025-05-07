import openai
from openai import OpenAI
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field, conint, ValidationError, ConfigDict
from utils.common_utils import get_env_variable
import openai
from openai import OpenAI # Ensure this is openai v1.0+
from typing import List, Literal, Annotated # Keep other imports as they are
from pydantic import BaseModel, Field, conint, ValidationError, ConfigDict
from utils.common_utils import get_env_variable
# Define new moderation_category_groups data
# This data is directly transcribed from the new JSON structure provided
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


class ModerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: ModerationScores
    justification: str
    categories: List[moderation_categories]


moderation_schema = ModerationResponse.model_json_schema()  # Generate JSON schema from Pydantic


def generate_moderation_prompt(category_groups_list: List[dict]) -> str: # Renamed arg for clarity
    # Format all category groups
    # This logic remains the same, but will now operate on the new category_groups_list structure
    category_groups_text = "\n\n".join( 
        "<category-group>\n" 
        f"'{group.get('title', '')}' (Score Code: {group.get('codePrefix', '')}) contains the following categories:\n" # Added Score Code for clarity to LLM
        + "".join(f"- {category.get('title', '')} (Code: {category.get('code', '')}): {category.get('llmDescription', '')}\n" for category in group.get('categories', [])) 
        + (f"\nNote:\n{group.get('note', '')}\n\n" if group.get('note') else "") 
        + f"Rating Criteria for '{group.get('title', '')}':\n- 5 {group.get('criteria5', '')}\n- 1 {group.get('criteria1', '')}\n"
        "</category-group>" 
        for group in category_groups_list
    ) 
    return f"""
OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document which has been generated by a user through interactions with Oak National Academy's lesson planning assistant. Your job is to assess and moderate the lesson plan across a variety of categories.

For each category group, you must determine the degree to which the lesson plan is compliant with the guidelines, providing a single rating on a 1-5 Likert scale for that group. The score code for each group is indicated (e.g., Score Code: l). Use these score codes as keys in your 'scores' JSON object.

Additionally, for any specific categories triggered within these groups, list their full codes (e.g., l/discriminatory-language) in the 'categories' array.

CATEGORY GROUPS:

{category_groups_text}

NOTE:

Your objective is to provide content guidance to ensure that certain content can be flagged appropriately. Regardless of the educational context, the score must be adjusted to reflect the presence of such content, even if the content is presented in a careful, considerate manner. The specific notes for each category group are paramount. Please also consider the age group of the pupils when scoring the content, as this may affect the sensitivity of the material.

INSTRUCTION:

Use the above to classify the lesson provided.
Provide a JSON response with:
1. `scores`: An object where keys are the 'Score Code' (l, u, s, p, e, r, n, t) for each category group, and values are the 1-5 Likert scale rating.
2. `justification`: A single string providing a concise, precise justification for ALL your scores. A detailed justification is crucial, even for a score of 5 for all groups.
3. `categories`: A list of strings, where each string is the full 'Code' (e.g., 'l/discriminatory-language', 'u/violence-or-suffering') of individual categories that were triggered. If no specific categories are triggered for a group that still receives a score other than 5 (or even 5 if nuance exists), explain in justification. If a group is rated 5 (fully compliant/not involved), its sub-categories typically shouldn't be listed unless there's a specific reason (explain in justification).

For each group other than 'Toxic' (t) and 'Not to be planned by Aila' (n), consider the key-stage (therefore age group) when scoring the content, alongside the specific guidelines for that group. For 'Toxic' (t) and 'Not to be planned by Aila' (n), the scoring should be strictly based on their definitions, as they represent content that is either illegitimate or must not be planned by Aila.
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
        llm: str = "gpt-4o-2024-08-06", # Default model
        temp: float = 0.7 # Default temperature
        ) -> ModerationResponse:

    # 1. Remove legacy API key setting (for openai library v1.0+)
    # openai.api_key = get_env_variable("OPENAI_API_KEY") # REMOVE THIS LINE

    # 2. Generate prompt
    prompt = generate_moderation_prompt(current_category_groups)

    # 3. Initialize OpenAI client (relies on OPENAI_API_KEY env var or pass api_key explicitly)
    try:
        # Ensure API key is available. If OPENAI_API_KEY env var is not set,
        # this will raise an error or you can pass it explicitly:
        # api_key_value = get_env_variable("OPENAI_API_KEY")
        # if not api_key_value:
        #     raise ValueError("OPENAI_API_KEY environment variable not set.")
        # client = OpenAI(api_key=api_key_value)
        client = OpenAI() # Assumes OPENAI_API_KEY is set in the environment
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    # 4. Debug prints for parameters going into the API call
    print(f"DEBUG: Calling OpenAI with model='{llm}' (type: {type(llm)})")
    print(f"DEBUG: Temperature: {temp} (type: {type(temp)})")
    print(f"DEBUG: Lesson plan length: {len(lesson_plan) if lesson_plan else 'None'}")
    print(f"DEBUG: Prompt length: {len(prompt) if prompt else 'None'}")

    # Ensure all inputs to the API are of the correct type
    try:
        llm_str = str(llm)
        temp_float = float(temp) # Ensure temperature is a float
        
        # Ensure messages content are strings
        system_content_str = str(prompt if prompt is not None else "")
        user_content_str = str(lesson_plan if lesson_plan is not None else "")

        messages_payload = [
            {"role": "system", "content": system_content_str},
            {"role": "user", "content": user_content_str}
        ]

    except ValueError as ve:
        raise RuntimeError(f"Type conversion error for API parameters: {ve}")


    # 5. Make the API call with robust error handling
    try:
        response = client.chat.completions.create(
            model=llm_str,
            messages=messages_payload,
            temperature=temp_float,
            response_format={"type": "json_object"},
        )
    except openai.APIConnectionError as e:
        # Handles network issues
        print(f"OpenAI API Connection Error: {e}")
        raise RuntimeError(f"Network error connecting to OpenAI: {e}")
    except openai.RateLimitError as e:
        # Handles rate limit errors (429)
        print(f"OpenAI Rate Limit Error: {e}")
        raise RuntimeError(f"OpenAI rate limit exceeded: {e}")
    except openai.APIStatusError as e:
        # Handles other API errors (4xx, 5xx)
        # This is where your 400 error would typically be caught more specifically.
        error_message = f"OpenAI API Error (Status {e.status_code}): {e.message}"
        if e.response and hasattr(e.response, 'text') and e.response.text:
            error_message += f" | Response: {e.response.text}"
        elif e.body: # Sometimes the body might contain the error details
             error_message += f" | Body: {e.body}"
        print(error_message) # Log the detailed error
        raise RuntimeError(error_message)
    except Exception as e:
        # Fallback for any other unexpected errors during the API call
        print(f"Unexpected error during OpenAI API call: {type(e).__name__} - {e}")
        # Potentially log parts of the payload if it's a very strange error
        # print(f"DEBUG: llm_str='{llm_str}', temp_float={temp_float}")
        # print(f"DEBUG: system_content_str (first 100 chars): {system_content_str[:100]}")
        # print(f"DEBUG: user_content_str (first 100 chars): {user_content_str[:100]}")
        raise RuntimeError(f"Unexpected error making OpenAI API call: {e}")

    # 6. Process the response
    try:
        moderation_data_content = response.choices[0].message.content
        if moderation_data_content is None:
            # This can happen if the model's response is empty or generation fails.
            # It might indicate an issue with the prompt or model behavior.
            print("OpenAI response content is null. Choice finish reason:", response.choices[0].finish_reason)
            raise RuntimeError("OpenAI response content is null. The model may have failed to generate a valid response.")
        
        moderation_response = ModerationResponse.model_validate_json(moderation_data_content)
        return moderation_response
    except ValidationError as e:
        print(f"Pydantic Validation error: {e.errors()}")
        print(f"Problematic OpenAI response content: {moderation_data_content}")
        raise RuntimeError(f"Invalid JSON structure or type from OpenAI: {moderation_data_content}")
    except AttributeError as e: # If response.choices[0].message.content is not found
        print(f"Error accessing response data: {e}. Full response object: {response}")
        raise RuntimeError(f"Could not access content from OpenAI response: {e}")
    except Exception as e:
        print(f"Error processing OpenAI response: {type(e).__name__} - {e}")
        # moderation_data_content might not be defined if the error is before its assignment
        # print(f"Problematic OpenAI response (if available): {response.choices[0].message.content if response and response.choices else 'N/A'}")
        raise RuntimeError(f"Could not process response from OpenAI: {e}")