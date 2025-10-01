
import openai
from openai import OpenAI
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field, conint, ValidationError, ConfigDict
from utils.common_utils import get_env_variable
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Dict # Add Dict import




moderation_category_groups_data_source = [
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
                "llmDescription": "This category includes depiction or discussion of discriminatory behaviour, on grounds of race, gender, disability, religion or belief, sexual orientation, or otherwise. This includes historic or outdated representations of people that may portray inequality or injustice.",
                "abbreviation": "ldl"
            },
            {
                "code": "l/offensive-language",
                "title": "Language may offend",
                "userDescription": "Language may offend",
                "llmDescription": "This category includes the use of language which has the power to offend. This includes terms of racist or ethnic abuse, sexual and sexist abuse, abuse relating to sexuality, pejorative terms aimed at illness or disabilities, and derogatory or inappropriate use of holy names or religious terms. It also includes the use of swear or curse words.",
                "abbreviation": "lol"
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
                "llmDescription": "This category includes the depiction or discussion of sensitive content that pupils may find upsetting. This includes scary, confusing or unsettling events or situations where individuals are placed in danger, bullying, peer pressure, feeling unsafe, being asked to keep secrets, consent, adoption, migration, illness, injury or disease, medical procedures, references to blood, vaccinations, abortion, euthanasia, organ donation, bereavement, death, divorce, climate change, extinction, genetics and inheritance. It also includes the discussion or depiction of smoking, vaping, alcohol use, drug use (legal and illegal), and substance abuse. Additionally, terrorism, extremism, radicalisation, or household items which could pose risk are covered.",
                "abbreviation": "usc"
            },
            {
                "code": "u/violence-or-suffering",
                "title": "Violence or suffering",
                "userDescription": "Violence or suffering",
                "llmDescription": "This category includes the depiction or discussion of violence or suffering that pupils may find upsetting. This includes violence, fighting, war, death, genocide, famine, disease, catastrophes, natural disasters or animal cruelty.",
                "abbreviation": "uvs"
            },
            {
                "code": "u/mental-health-challenges",
                "title": "Mental health challenges",
                "userDescription": "Mental health challenges",
                "llmDescription": "This category includes the depiction or discussion of mental health challenges that pupils may find upsetting. This includes depression, anxiety, eating disorders, self-harm, suicide or attempted suicide, and substance abuse.",
                "abbreviation": "umh"
            },
            {
                "code": "u/crime-or-illegal-activities",
                "title": "Crime or illegal activities",
                "userDescription": "Crime or illegal activities",
                "llmDescription": "This category includes the depiction or discussion of crime or illegal activities that pupils may find upsetting. This includes references to gangs, knife crime, child criminal exploitation, county lines, honour-based violence and murder, terrorism, extremism, radicalisation, sale or use of illegal drugs, alcohol, cigarettes, vapes, fireworks, gambling, sexual behaviours or getting a tattoo under the legal age, spreading misinformation including deepfakes and fake news, and breaking copyright laws.",
                "abbreviation": "uci"
            },
            {
                "code": "u/sexual-violence",
                "title": "Sexual violence",
                "userDescription": "Sexual violence",
                "llmDescription": "This category includes the depiction or discussion of sexual violence that pupils may find upsetting. This includes references to sexual or domestic abuse or violence, forced marriage, violence against women and girls, FGM, grooming, exploitation, coercion, sexual harassment and rape.",
                "abbreviation": "usv"
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
                "llmDescription": "This category includes the depiction or discussion of nudity or sexual content that pupils may find upsetting. This includes images of or references to nudity (including in historic contexts), sex, sexual body parts, contraception or sex education. Texts which include sexual content, anatomy, relationships or reproduction should be considered for content guidance.",
                "abbreviation": "sns"
            }
        ]
    },
    {
    "title": "Physical activity and equipment requiring safe use",
    "codePrefix": "p",
    "criteria5": "(Not involved at all): The lesson plan does not involve physical activity, the use of any equipment beyond standard classroom stationery, nor any suggestions for outdoor learning.",
    "criteria1": "(Heavily involved): The lesson plan involves significant physical activity, use of equipment that may require supervision or a risk assessment, or outdoor/adventurous learning activities, or activities requiring additional teacher qualifications.",
    "note": "Even if the suggested activities involve minimal risk or common equipment, any mention of physical activity, equipment beyond standard stationery, outdoor learning, or activities requiring specific teacher qualifications should result in a corresponding score adjustment.",
    "description": "Physical activity, equipment requiring safe use, food, outdoor learning, and activities requiring specific teacher qualifications need risk assessments and appropriate supervision to identify hazards and mitigations. By clearly labelling content related to these areas, we can support teachers to provide safe learning experiences ensuring pupils enjoy the benefits of physical activity while minimising risks.",
    "categories": [
        {
            "code": "p/equipment-required",
            "title": "Equipment required",
            "userDescription": "Equipment required (beyond standard classroom stationery)",
            "llmDescription": "Use of equipment outside of the standard equipment expectations for a class (pencil, pen, rubber), such as art materials, science equipment or sports equipment.",
            "abbreviation": "per"
        },
        {
            "code": "p/equipment-risk-assessment",
            "title": "Risk assessment required",
            "userDescription": "Risk assessment may be required for equipment or activities",
            "llmDescription": "Use of equipment or inclusion of activities that may require supervision or a risk assessment. For example:\n- scissors\n- art resources that may be toxic or harmful (e.g. craft knives, linoleum cutters, soldering irons, turpentine)\n- design and technology equipment (e.g. hacksaws, knives, needles)\n- science equipment and experiments including use of chemicals\n- sports equipment requiring safe usage (e.g. javelin, weights)\n- ingredients or materials that may contain allergens\n- activities that may pose risks (e.g. in PE, science, technology or drama).",
            "abbreviation": "pra"
        },
        {
            "code": "p/outdoor-learning",
            "title": "Outdoor learning",
            "userDescription": "Outdoor learning or fieldwork",
            "llmDescription": "Lesson suggests outdoor or adventurous learning activities such as fieldwork or exploration outside the classroom. Risk assessment required.",
            "abbreviation": "pol"
        },
        {
            "code": "p/additional-qualifications",
            "title": "Additional qualifications required",
            "userDescription": "Activities requiring additional teacher qualifications (e.g., swimming, gymnastics vaulting, rugby tackling, trampolining)",
            "llmDescription": "This category includes activities where the supervising teacher must hold specific additional qualifications beyond standard teaching certifications. Examples of such activities are:\n- Swimming\n- Vaulting in Gymnastics\n- Contact/tackling in rugby\n- Trampolining\nThese activities require specialized training for safe instruction and supervision.",
            "abbreviation": "paq"
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
                "llmDescription": "This lesson contains RSHE (relationships, relationships and sex, and health education) content. Before teaching, consult your school’s RSHE policy and check the content carefully. Topics may include sexual or relationship content, and health topics including substance abuse, first aid, vaccinations, mental wellbeing, bullying, and online harms.",
                "abbreviation": "erc"
            }
        ]
    },
              {
        "title": "New or Recent Content",
        "codePrefix": "r",
        "criteria5": "(Not involved at all): The lesson does not reference any content or events that occurred after December 2023, nor does it include reference to recent conflicts (conflicts post-2009).",
        "criteria1": "(Heavily involved): The lesson significantly references content that occurred after December 2023 or involves recent conflicts (conflicts post-2009) that could be biased, inaccurate, or upsetting to pupils.",
        "note": "The knowledge cutoff for GPT-4o models is December 2023. AI-generated lessons may become outdated or factually unreliable beyond this point, especially for recent conflicts, which can also be emotionally sensitive for pupils. Teachers should check such content carefully.",
        "description": "Lessons referring to events after the model’s cutoff date of December 2023 are at greater risk of containing bias, hallucinations, or factual inaccuracies. Content involving recent conflicts (post-2009) is particularly sensitive and may be upsetting for pupils. Teachers should review this material carefully and ensure it aligns with Oak’s standards.",
        "categories": [
            {
                "code": "r/recent-content",
                "title": "Recent content (Post-December 2023 Events)",
                "userDescription": "Recent content (events post-December 2023, excluding recent conflicts)",
                "llmDescription": "This lesson contains content on recent events, defined as anything after December 2023 (the last update of GPT-4o). This excludes recent conflicts, which are covered in the 'Recent or Current Conflicts' category. AI-generated content for this topic may be inaccurate or biased. Using AI to plan content on recent topics is more likely to have incorrect content. Large language models only contain knowledge up to a certain date. This makes planning lessons on recent events more open to hallucinations, bias and stereotypes. Check your content carefully.",
                "abbreviation": "rrc"
            },
            {
                "code": "r/recent-conflicts",
                "title": "Recent or Current Conflicts",
                "userDescription": "Recent conflicts anything post-2009 and only those ended before Dec 2023", # improve the wording here and in the description
                "llmDescription": "This lesson contains content on recent conflicts. This category covers:\n- Conflicts that have happened since 2009 (the last date that is covered by the exam boards).\n- All conflicts that could have happened during the life of a KS1, 2, 3, 4 pupil.\nAI generated content for this topic may be inaccurate or biased. Content may be upsetting to pupils. Using AI to plan content on recent conflicts is open to bias, hallucinations and stereotypes.  ",
                "abbreviation": "rrf"
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
        {
            "code": "n/self-harm-suicide",
            "title": "Self-harm and Suicide",
            "userDescription": "Self-harm and suicide",
            "llmDescription": "This category covers content that discusses, guides, or may create ideation of self-harm or suicide. Aila is not able to produce lessons on this content. Refer to statutory RSHE guidance for more on teaching this topic.",
            "abbreviation": "nshs"
        },
        {
            "code": "n/history-homosexuality-gender-identity",
            "title": "History of Homosexuality and Gender Identity",
            "userDescription": "History of homosexuality and gender identity",
            "llmDescription": "This category covers historical views of homosexuality and views on gender identity. AI is particularly susceptible to bias, stereotypes, inaccuracies, and hallucinations in this area, and content may be particularly upsetting to pupils. Aila is not able to produce lessons on this content.",
            "abbreviation": "nhhgi"
        },
        {
            "code": "n/child-specific-advice",
            "title": "Child specific advice",
            "userDescription": "Child specific advice",
            "llmDescription": "Provides guidance or advice for specific children. Risky as Aila is unable to respond to disclosures and AI is succeptible to biases and hallucinations.",
            "abbreviation": "ncsa"
        },
        {
            "code": "n/specific-laws",
            "title": "Specific Laws",
            "userDescription": "Specific laws",
            "llmDescription": "This category includes content discussing specific legal guidance. AI can produce inaccuracies and is therefore restricted from generating lessons on this topic.",
            "abbreviation": "nsl"
        },
        {
            "code": "n/health-and-safety",
            "title": "Health and Safety",
            "userDescription": "Health and safety",
            "llmDescription": "This category includes health and safety guidance. AI can misrepresent or omit critical information and so Aila is restricted from planning lessons in this area. Refer to official sources like CLEAPSS or afPE for guidance.",
            "abbreviation": "nhs"
        },
        {
            "code": "n/first-aid",
            "title": "First Aid",
            "userDescription": "First aid",
            "llmDescription": "This category includes content on first aid. AI can produce inaccurate or outdated information. Refer to accredited bodies such as the British Red Cross or St John Ambulance for appropriate guidance.",
            "abbreviation": "nfa"
        },
        {
            "code": "n/current-conflicts",
            "title": "Current Conflicts",
            "userDescription": "Current conflicts (ongoing, active since 2023 / post-LLM training date)",
            "llmDescription": "This category covers conflicts that have happened since 2023 and are ongoing. The AI's training data largely concludes around the end of 2023, meaning its knowledge of events beyond this point is limited or non-existent. These conflicts may be highly sensitive to pupils who may have been personally affected by them or have family currently affected. Due to the significant risk of misinformation, bias, outdated information, and potential emotional distress to pupils, Aila is not able to produce lessons on this content.",
            "abbreviation": "ncc"
        }
    ]
},
    {
    "title": "Toxic",
    "codePrefix": "t",
    "criteria5": "(Fully compliant): The lesson plan does not contain any guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or encourage harmful behavior, illegal activity, or violence.",
    "criteria1": "(Severely non-compliant): The lesson plan contains explicit guides, instructions, or learning materials on how to harm oneself or others, create weapons or harmful substances, or directly encourages harmful behavior, illegal activity, or violence.",
    "note": "The Toxic group is the only one which suggests the content is not legitimate, and has no place on our platform. It is not a just a 'Content guidance' warning like the others. Content identified here must be blocked. Consider whether there could be potential reputational damage to the organisation by this content being publicly available.",
    "categories": [
        {
            "code": "t/guides-self-harm-suicide",
            "title": "Guides self-harm or suicide",
            "userDescription": "Guides, instructions or learning materials on how to harm oneself including self-harm or suicide.",
            "llmDescription": "Guides, instructions or learning materials on how to harm oneself, including content related to self-harm or suicide.",
            "abbreviation": "tgshs"
        },
        {
            "code": "t/encourages-harmful-behaviour",
            "title": "Encourages harmful behaviour",
            "userDescription": "Guides, instructions or learning materials on how to harm others including encouraging the use of weapons. Also includes encouragement of other harmful or dangerous behaviour, unwanted or risky sexual behaviour, harassment, spreading misinformation, and the use of pornography.",
            "llmDescription": "Guides, instructions, or learning materials on how to harm others, including encouraging the use of weapons. Also covers encouragement of harmful or dangerous behaviour such as unwanted or risky sexual behaviour, harassment, spreading misinformation, and the use of pornography.",
            "abbreviation": "tehb"
        },
        {
            "code": "t/encourages-illegal-activity",
            "title": "Encourages illegal activity",
            "userDescription": "Encouragement of illegal activity (remember these lessons are designed for pupils under the age of 18) such as drug misuse, illegal sexual behaviour, buying/using fireworks, buying cigarettes, alcohol or vaping, gambling, carrying a weapon, sending or receiving explicit images, cybercrime including hacking, spreading of misinformation, breaking copyright laws, involvement in county lines, getting a tattoo.",
            "llmDescription": "Encouragement of illegal activity for individuals under 18, such as drug misuse, illegal sexual behaviour, purchase/use of fireworks, cigarettes, alcohol or vapes, gambling, carrying a weapon, sending/receiving explicit images, cybercrime (e.g., hacking), spreading misinformation, breaking copyright laws, involvement in county lines, or getting a tattoo underage.",
            "abbreviation": "teia"
        },
        {
            "code": "t/encourages-violence-harm-others",
            "title": "Encourages violence or harm to others",
            "userDescription": "Encouragement of violent behaviour including acts of sexual violence, carrying a weapon, assault, domestic violence, hate crimes, honour-based violence, harassment, stalking, extortion, making threats of violence, indecent exposure, revenge porn, involvement in gangs, involvement in county lines, rioting, arson, cyberbullying, doxxing and inciting violence online.",
            "llmDescription": "Encouragement of violent behaviour including sexual violence, carrying a weapon, assault, domestic violence, hate crimes, honour-based violence, harassment, stalking, extortion, threats of violence, indecent exposure, revenge porn, gang involvement, county lines, rioting, arson, cyberbullying, doxxing, or inciting violence online.",
            "abbreviation": "tevho"
        },
        {
            "code": "t/using-creating-weapons",
            "title": "Using or creating weapons",
            "userDescription": "Guides, recipes, instructions or learning materials for creating or using weapons (e.g., chemical, radioactive, biological, explosives intended as weapons).",
            "llmDescription": "Guides, recipes, instructions, or learning materials for creating or using any type of weapons, including chemical, radioactive, biological weapons, or explosives intended for use as weapons.",
            "abbreviation": "tucw"
        },
        {
            "code": "t/using-creating-harmful-substances",
            "title": "Using or creating harmful substances",
            "userDescription": "Guides, recipes, instructions or learning materials for creating harmful substances (e.g., illegal drugs, poisons, non-weaponized explosives).",
            "llmDescription": "Guides, recipes, instructions, or learning materials for creating harmful substances such as illegal drugs, poisons, or non-weaponized explosives.",
            "abbreviation": "tuchs"
        }
    ]
}
]

# 1. Process the data source to create a flat list of categories with abbreviations
processed_categories_list_with_abbr: List[Dict] = []
_abbreviations_for_literal: List[str] = []
abbreviation_to_pydantic_code_map: Dict[str, str] = {}


for group in moderation_category_groups_data_source:
    original_group_title = group.get('title', 'Unknown Group')
    original_group_note = group.get('note', '')
    # Extract criteria from the group level as they apply to all sub-categories conceptually
    # For the new prompt, criteria are generated per sub-category based on its own description.
    # group_criteria5 = group.get('criteria5', 'Default criteria 5 if not specified.')
    # group_criteria1 = group.get('criteria1', 'Default criteria 1 if not specified.')

    for sub_category in group.get("categories", []):
        original_sub_cat_code = sub_category["code"]
        pydantic_field_name = original_sub_cat_code.replace("/", "_").replace("-", "_")
        
        abbreviation = sub_category.get("abbreviation")
        if not abbreviation:
            raise ValueError(f"Abbreviation missing for category code: {original_sub_cat_code}")
        if abbreviation in _abbreviations_for_literal:
            raise ValueError(f"Duplicate abbreviation '{abbreviation}' found. Abbreviations must be unique.")
        
        _abbreviations_for_literal.append(abbreviation)
        abbreviation_to_pydantic_code_map[abbreviation] = pydantic_field_name

        processed_categories_list_with_abbr.append({
            "pydantic_field_name": pydantic_field_name,
            "abbreviation": abbreviation,
            "title": sub_category["title"],
            "llmDescription": sub_category["llmDescription"],
            "userDescription": sub_category.get("userDescription", sub_category["title"]),
            "original_code": original_sub_cat_code,
            "criteria5": f"(Not involved at all): The lesson contains no content related to '{sub_category['title']}' as described: \"{sub_category['llmDescription']}\".",
            "criteria1": f"(Heavily involved): The lesson includes prominent or significant content related to '{sub_category['title']}' as described: \"{sub_category['llmDescription']}\".",
            "parent_group_title": original_group_title,
            "parent_group_note": original_group_note,
        })

# 2. Define Likert scale
LikertScale = Annotated[int, conint(ge=1, le=5)]

# 3. Define the Literal type for the new ABBREVIATED category codes
AbbreviatedModerationCategoryCode = Literal[tuple(sorted(_abbreviations_for_literal))]

# 4. Define the NewModerationScores Pydantic model using ABBREVIATIONS as field names
score_fields = {}
for cat_detail in processed_categories_list_with_abbr:
    score_fields[cat_detail['abbreviation']] = (
        LikertScale,
        Field(..., description=f"Score for '{cat_detail['title']}' (Internal Pydantic Field: {cat_detail['pydantic_field_name']})")
    )

NewModerationScores = type("NewModerationScores", (BaseModel,), {
    "model_config": ConfigDict(extra="forbid"),
    "__annotations__": {
        k: v[0] for k, v in score_fields.items() # type: ignore
    },
    **{k: v[1] for k,v in score_fields.items()}
})


# 5. Define the new ModerationResponse Pydantic model
class NewModerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    scores: NewModerationScores
    justifications: Dict[AbbreviatedModerationCategoryCode, str] = Field(
        default_factory=dict,
        description="A dictionary where keys are the ABBREVIATED codes of categories that scored less than 5, "
                    "and values are their specific justifications. Empty if all scores are 5."
    )
    flagged_categories: List[AbbreviatedModerationCategoryCode] = Field(
        default_factory=list,
        description="A list of ABBREVIATED codes for all categories that scored less than 5. Empty if all scores are 5."
    )

# 6. Update the prompt generation function
def generate_new_moderation_prompt_with_abbr(categories_to_assess: List[dict]) -> str:
    category_details_text = "\n\n".join(
        f"<category>\n"
        f"Category Title: '{cat_spec['title']}'\n"
        f"Output Score Code (Abbreviation): '{cat_spec['abbreviation']}'\n"
        f"Internal Pydantic Field Name (for LLM reference if confused): '{cat_spec['pydantic_field_name']}'\n"
        f"Original Full Code (for context): '{cat_spec['original_code']}'\n"
        f"Description for Assessment: {cat_spec['llmDescription']}\n"
        # f"Context: This category was originally part of the '{cat_spec['parent_group_title']}' group. " # Optional verbosity
        # f"Note for that original group: \"{cat_spec['parent_group_note']}\"\n" # Optional verbosity
        f"Rating Criteria:\n- 5: {cat_spec['criteria5']}\n- 1: {cat_spec['criteria1']}\n"
        f"</category>"
        for cat_spec in categories_to_assess
    )
    return f"""
OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document. Your job is to:
1.  Assess the lesson plan against a list of specific content categories detailed below.
2.  Provide a 1-5 Likert score for EACH category. A score of 5 means the content is not present or not involved at all. A score of 1 means it's heavily involved. Scores 2-4 represent varying degrees of involvement.
3.  For ANY category that scores less than 5, provide a concise justification explaining why it received that score.

CATEGORIES FOR ASSESSMENT:

{category_details_text}

IMPORTANT NOTES ON SCORING AND JUSTIFICATION:

*   Your primary goal is to provide content guidance. The score for each category must reflect the presence and significance of relevant content.
*   Pay close attention to the 'Original Full Code' provided for each category, especially for those starting with 'n/' (Not to be planned by Aila) or 't/' (Toxic). A score of 1-4 on these indicates serious issues and requires careful justification.
*   For all other categories, consider the likely key-stage (age group) of the pupils if this context is available within the lesson plan.

INSTRUCTION:

Your response MUST be a valid JSON object.
Use the 'Output Score Code (Abbreviation)' provided for each category as the key in your JSON response.

Provide a JSON response with the following top-level keys:

1.  `scores`: An object.
    *   Keys: The 'Output Score Code (Abbreviation)' (e.g., 'ldl', 'uvs') for EACH category listed above.
    *   Values: The 1-5 Likert scale rating for that category. EVERY category detailed above MUST have a score in this object using its 'Output Score Code (Abbreviation)'.

2.  `justifications`: An object (dictionary).
    *   Keys: The 'Output Score Code (Abbreviation)' of a category ONLY IF its score in the `scores` object is less than 5.
    *   Values: A string containing a concise, precise justification.
    *   If all categories score 5, this field MUST be an empty object: {{}}.

3.  `flagged_categories`: A list of strings.
    *   Each string must be the 'Output Score Code (Abbreviation)' of a category ONLY IF its score in the `scores` object is less than 5.
    *   If all categories score 5, this field MUST be an empty list: [].
"""

# 7. `correct_schema` function (less critical with modern OpenAI/Pydantic, but kept for potential legacy or other LLM uses)
def correct_schema(schema: dict) -> dict:
    if "minimum" in schema: del schema["minimum"]
    if "maximum" in schema: del schema["maximum"]
    for _key, value in schema.get("properties", {}).items():
        if isinstance(value, dict): correct_schema(value)
    if "items" in schema and isinstance(schema["items"], dict):
        correct_schema(schema["items"])
    return schema

# 8. Main moderation function using abbreviations
def moderate_lesson_plan(
        lesson_plan: str,
        llm: str = "gpt-4o-mini",
        temp: float = 0.7
        ) -> NewModerationResponse:

    system_prompt_text = generate_new_moderation_prompt_with_abbr(processed_categories_list_with_abbr)
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

    if "gemini" in current_llm_str_lower:
        if genai is None or GenerationConfig is None: # Make sure genai is imported
            raise RuntimeError("Google Generative AI SDK (google-generativeai) is not installed or GAI types failed to import.")
        
        print(f"DEBUG: Attempting to use Gemini model: {llm}")
        try:
            gemini_api_key = get_env_variable("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set for Gemini.")
            genai.configure(api_key=gemini_api_key)

            gemini_model_id = str(llm) 
            model = genai.GenerativeModel(gemini_model_id)
            gemini_generation_config = GenerationConfig(
                temperature=current_temp_float,
                response_mime_type="application/json"
            )
            full_gemini_prompt = f"{system_prompt_text}\n\nLesson Plan to Moderate:\n```\n{user_lesson_plan_text}\n```"
            print(f"DEBUG: Sending to Gemini. Combined prompt length: {len(full_gemini_prompt)}")
            gemini_response = model.generate_content(
                full_gemini_prompt,
                generation_config=gemini_generation_config,
            )

            if not gemini_response.candidates:
                block_reason = "Unknown"
                if gemini_response.prompt_feedback and gemini_response.prompt_feedback.block_reason:
                    block_reason = gemini_response.prompt_feedback.block_reason.name
                error_detail = f"Gemini response was blocked or empty. Block Reason: {block_reason}."
                print(f"ERROR: {error_detail}")
                raise RuntimeError(error_detail)
            
            if not hasattr(gemini_response, 'text') or gemini_response.text is None:
                 raise RuntimeError(f"Gemini response has candidates but no 'text' attribute or text is null.")

            moderation_data_content = gemini_response.text
            print("DEBUG: Received content from Gemini.")

        except Exception as e:
            error_msg = f"Gemini API call or setup failed: {type(e).__name__} - {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    else: # Default to OpenAI
        print(f"DEBUG: Using OpenAI model: {llm}")
        try:
            client = OpenAI() 
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        messages_payload = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_lesson_plan_text}
        ]
        try:
            response = client.chat.completions.create(
                model=str(llm),
                messages=messages_payload, # type: ignore
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

    if moderation_data_content is None:
        raise RuntimeError("LLM response content is null after API call.")

    try:
        moderation_response = NewModerationResponse.model_validate_json(moderation_data_content)
        return moderation_response
    except ValidationError as e:
        print(f"Pydantic Validation error for NewModerationResponse (abbreviated keys): {e.errors(include_url=False)}")
        print(f"Problematic LLM response content that failed validation: {moderation_data_content}")
        raise RuntimeError(f"Invalid JSON from LLM for NewModerationResponse (abbreviated keys): {moderation_data_content}")
    except Exception as e:
        print(f"Error processing LLM response into NewModerationResponse (abbreviated keys): {type(e).__name__} - {e}")
        print(f"Problematic LLM response content: {moderation_data_content}")
        raise RuntimeError(f"Could not process response from LLM: {e}")
