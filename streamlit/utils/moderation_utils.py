
"""
Moderation utilities for content assessment and validation.

This module provides functions for:
- Loading moderation categories from JSON configuration
- Processing lesson plans through AI moderation
- Generating structured moderation responses
- Supporting both OpenAI and Google Gemini models
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Annotated, Optional, Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field, conint, ValidationError, ConfigDict

from utils.common_utils import get_env_variable




def load_moderation_categories() -> List[Dict[str, Any]]:
    """Load moderation categories from JSON configuration file.
    
    Returns:
        List of moderation category groups with their subcategories
        
    Raises:
        FileNotFoundError: If the categories JSON file is not found
        json.JSONDecodeError: If the JSON file is malformed
        ValueError: If required fields are missing from the categories
    """
    # Get the path to the categories JSON file
    current_dir = Path(__file__).parent
    categories_file = current_dir.parent / "data" / "moderation_categories.json"
    
    if not categories_file.exists():
        raise FileNotFoundError(f"Moderation categories file not found: {categories_file}")
    
    try:
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in categories file: {e}")
    
    # Validate the structure
    if not isinstance(categories, list):
        raise ValueError("Categories file must contain a list of categories")
    
    for i, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ValueError(f"Category {i} must be a dictionary")
        
        required_fields = ["code", "title", "llmDescription", "abbreviation", "criteria5", "criteria1"]
        for field in required_fields:
            if field not in category:
                raise ValueError(f"Missing required field '{field}' in category {i}")
    
    return categories


# Load categories from JSON file
try:
    moderation_category_groups_data_source = load_moderation_categories()
except (FileNotFoundError, json.JSONDecodeError) as e:
    # Fallback to empty list if loading fails
    print(f"Warning: Failed to load moderation categories: {e}")
    moderation_category_groups_data_source = []

def process_moderation_categories(categories_data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str], Dict[str, str]]:
    """Process moderation categories data into a flat list with abbreviations.
    
    Args:
        categories_data: List of categories from JSON configuration
        
    Returns:
        Tuple containing:
        - processed_categories_list_with_abbr: Flat list of processed categories
        - abbreviations_for_literal: List of abbreviations for type generation
        - abbreviation_to_pydantic_code_map: Mapping from abbreviations to pydantic field names
        
    Raises:
        ValueError: If required fields are missing or abbreviations are duplicated
    """
    processed_categories_list_with_abbr: List[Dict[str, Any]] = []
    abbreviations_for_literal: List[str] = []
    abbreviation_to_pydantic_code_map: Dict[str, str] = {}

    if not categories_data:
        return processed_categories_list_with_abbr, abbreviations_for_literal, abbreviation_to_pydantic_code_map

    for category in categories_data:
        original_code = category["code"]
        pydantic_field_name = original_code.replace("/", "_").replace("-", "_")
        
        abbreviation = category.get("abbreviation")
        if not abbreviation:
            raise ValueError(f"Abbreviation missing for category code: {original_code}")
        if abbreviation in abbreviations_for_literal:
            raise ValueError(f"Duplicate abbreviation '{abbreviation}' found. Abbreviations must be unique.")
        
        abbreviations_for_literal.append(abbreviation)
        abbreviation_to_pydantic_code_map[abbreviation] = pydantic_field_name

        # Create criteria dictionary with all 5 levels
        criteria_dict = {}
        for i in [5, 4, 3, 2, 1]:
            criteria_key = f"criteria{i}"
            if criteria_key in category and category[criteria_key]:
                criteria_dict[criteria_key] = category[criteria_key]
            else:
                # Fallback to generated criteria if not provided
                if i == 5:
                    criteria_dict[criteria_key] = f"(Not involved at all): The lesson contains no content related to '{category['title']}' as described: \"{category['llmDescription']}\"."
                elif i == 1:
                    criteria_dict[criteria_key] = f"(Heavily involved): The lesson includes prominent or significant content related to '{category['title']}' as described: \"{category['llmDescription']}\"."
                else:
                    # Generate intermediate criteria
                    involvement_level = ["heavily", "significantly", "moderately", "minimally"][i-1]
                    criteria_dict[criteria_key] = f"({involvement_level.title()} involved): The lesson contains {involvement_level} related content to '{category['title']}' as described: \"{category['llmDescription']}\"."

        processed_categories_list_with_abbr.append({
            "pydantic_field_name": pydantic_field_name,
            "abbreviation": abbreviation,
            "title": category["title"],
            "llmDescription": category["llmDescription"],
            "userDescription": category.get("userDescription", category["title"]),
            "original_code": original_code,
            "parent_group_title": "General",  # No groups in new format
            "parent_group_note": "",  # No group notes in new format
            **criteria_dict  # Include all criteria levels
        })

    return processed_categories_list_with_abbr, abbreviations_for_literal, abbreviation_to_pydantic_code_map


# Process the loaded categories
try:
    processed_categories_list_with_abbr, _abbreviations_for_literal, abbreviation_to_pydantic_code_map = process_moderation_categories(
        moderation_category_groups_data_source
    )
except ValueError as e:
    print(f"Error processing moderation categories: {e}")
    processed_categories_list_with_abbr = []
    _abbreviations_for_literal = []
    abbreviation_to_pydantic_code_map = {}

# Define Likert scale constraint
LikertScale = Annotated[int, conint(ge=1, le=5)]

# Define the Literal type for abbreviated category codes
# Use a fallback to avoid empty Literal types which cause Pydantic errors
if _abbreviations_for_literal:
    AbbreviatedModerationCategoryCode = Literal[tuple(sorted(_abbreviations_for_literal))]
else:
    # Fallback to a dummy value to avoid empty Literal
    AbbreviatedModerationCategoryCode = Literal["dummy"]

def create_moderation_scores_model(categories: List[Dict[str, Any]]) -> type[BaseModel]:
    """Create a dynamic Pydantic model for moderation scores.
    
    Args:
        categories: List of processed category details
        
    Returns:
        Dynamic Pydantic model class for moderation scores
    """
    if not categories:
        # Return empty model if no categories
        class EmptyModerationScores(BaseModel):
            model_config = ConfigDict(extra="forbid")
        return EmptyModerationScores
    
    score_fields = {}
    for cat_detail in categories:
        abbreviation = cat_detail['abbreviation']
        title = cat_detail['title']
        pydantic_field_name = cat_detail['pydantic_field_name']
        
        score_fields[abbreviation] = (
            LikertScale,
            Field(..., description=f"Score for '{title}' (Internal Pydantic Field: {pydantic_field_name})")
        )

    return type("NewModerationScores", (BaseModel,), {
        "model_config": ConfigDict(extra="forbid"),
        "__annotations__": {
            k: v[0] for k, v in score_fields.items() # type: ignore
        },
        **{k: v[1] for k, v in score_fields.items()}
    })

def create_moderation_response_model(scores_model: type[BaseModel], selected_categories: List[str]) -> type[BaseModel]:
    """Create a dynamic Pydantic model for moderation response with only selected categories.
    
    Args:
        scores_model: The dynamic scores model for selected categories
        selected_categories: List of selected category abbreviations
        
    Returns:
        Dynamic Pydantic model class for moderation response
    """
    # Create a Literal type for selected categories only
    if selected_categories:
        SelectedCategoryCode = Literal[tuple(sorted(selected_categories))]
    else:
        SelectedCategoryCode = Literal["dummy"]  # Fallback
    
    return type("DynamicModerationResponse", (BaseModel,), {
        "model_config": ConfigDict(extra="forbid", populate_by_name=True),
        "__annotations__": {
            "scores": scores_model,
            "justifications": Dict[SelectedCategoryCode, str],
            "flagged_categories": List[SelectedCategoryCode]
        },
        "scores": Field(..., description="Moderation scores for selected categories"),
        "justifications": Field(
            default_factory=dict,
            description="A dictionary where keys are the ABBREVIATED codes of categories that scored less than 5, "
                       "and values are their specific justifications. Empty if all scores are 5."
        ),
        "flagged_categories": Field(
            default_factory=list,
            description="A list of ABBREVIATED codes for all categories that scored less than 5. Empty if all scores are 5."
        )
    })

# Create the moderation scores model
NewModerationScores = create_moderation_scores_model(processed_categories_list_with_abbr)


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
        (
            f"<category>\n"
            f"Category Title: '{cat_spec['title']}'\n"
            f"Output Score Code (Abbreviation): '{cat_spec['abbreviation']}'\n"
            # f"Internal Pydantic Field Name (for LLM reference if confused): '{cat_spec['pydantic_field_name']}'\n"
            f"Original Full Code (for context): '{cat_spec['original_code']}'\n"
            f"Description for Assessment: {cat_spec['llmDescription']}\n"
            "Rating Criteria:\n" +
            "\n".join(
                f"- {i}: {cat_spec[f'criteria{i}']}"
                for i in [5, 4, 3, 2, 1]
                if f'criteria{i}' in cat_spec and cat_spec[f'criteria{i}']
            ) +
            "\n</category>"
        )
        for cat_spec in categories_to_assess
    )
    return f"""OBJECTIVE:

You are a content moderation supervisor examining a lesson plan document. Your job is to:
1.  Assess the lesson plan against a list of specific content categories detailed below.
2.  Provide a 1-5 Likert score for EACH category. A score of 5 means the content is not present or not involved at all. A score of 1 means it's heavily involved. Scores 2-4 represent varying degrees of involvement.
3.  For ANY category that scores less than 5, provide a concise justification explaining why it received that score.

CATEGORIES FOR ASSESSMENT:

{category_details_text}

IMPORTANT NOTES ON SCORING AND JUSTIFICATION:

*   Your primary goal is to provide content guidance. The score for each category must reflect the presence and significance of relevant content.
*   Pay close attention to the 'Original Full Code' provided for each category, especially for those starting with 'n/' (Not to be planned by Aila) or 't/' (Toxic). A score of 1-4 on these indicates serious issues and requires careful justification.
*   For all other categories, when determining a score, consider the Key Stage (age group) of the pupils if Key Stage is specified within the lesson plan. For example, "Students will analyze Wilfred Owen's poem 'Dulce et Decorum Est', focusing on the graphic imagery of gas attacks and the psychological trauma of World War I soldiers.", would be suitable for KS4 pupils but not for KS2 pupils.

INSTRUCTION:

Your response MUST be a valid JSON object.
Use the 'Output Score Code (Abbreviation)' provided for each category as the key in your JSON response. All categories listed above MUST be included in your response, even if they score 5. Any category that scores less than 5 must have a justification provided in the 'justifications' object, using the same 'Output Score Code (Abbreviation)' as the key. If all categories score 5, the 'justifications' object must be empty: {{}}.

Provide a JSON response with the following top-level keys:

1.  `scores`: An object.
    *   Keys: The 'Output Score Code (Abbreviation)' (e.g., 'l1, 'u2') for EACH category listed above.
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

def moderate_lesson_plan(
        lesson_plan: str,
        llm: str = "gpt-4o-mini",
    temp: float = 0.7,
    selected_categories: Optional[Dict[str, bool]] = None
) -> Any:  # Return type is dynamic based on selected categories
    """Moderate a lesson plan using AI models.
    
    Args:
        lesson_plan: The lesson plan content to moderate
        llm: The language model to use for moderation (default: "gpt-4o-mini")
        temp: Temperature setting for the model (default: 0.7)
        selected_categories: Optional dict of category abbreviations to include (default: all categories)
        
    Returns:
        NewModerationResponse containing scores, justifications, and flagged categories
        
    Raises:
        RuntimeError: If moderation fails due to API errors, validation errors, or other issues
        ValueError: If input parameters are invalid
    """

    # Validate inputs
    if not lesson_plan or not lesson_plan.strip():
        raise ValueError("Lesson plan content cannot be empty")
    
    if not processed_categories_list_with_abbr:
        raise RuntimeError("No moderation categories available. Check categories configuration.")
    
    # Filter categories based on selection
    categories_to_use = processed_categories_list_with_abbr
    if selected_categories:
        # Filter to only include selected categories
        categories_to_use = [
            cat for cat in processed_categories_list_with_abbr 
            if selected_categories.get(cat.get('abbreviation', ''), True)
        ]
        
        if not categories_to_use:
            raise ValueError("No categories selected for moderation")
    
    # Create dynamic model for only selected categories
    dynamic_scores_model = create_moderation_scores_model(categories_to_use)
    
    # Get list of selected category abbreviations
    selected_category_abbreviations = [cat['abbreviation'] for cat in categories_to_use]
    
    # Create dynamic response model
    dynamic_response_model = create_moderation_response_model(dynamic_scores_model, selected_category_abbreviations)
    
    # Generate system prompt with filtered categories
    system_prompt_text = generate_new_moderation_prompt_with_abbr(categories_to_use)
    user_lesson_plan_text = str(lesson_plan).strip()

    # Validate and convert parameters
    try:
        current_llm_str_lower = str(llm).lower()
        current_temp_float = float(temp)
        
        if not (0.0 <= current_temp_float <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
            
    except ValueError as ve:
        raise ValueError(f"Invalid parameter: {ve}")

    print(f"DEBUG: Calling LLM: '{llm}' (normalized: '{current_llm_str_lower}') with temp: {current_temp_float}")
    print(f"DEBUG: System prompt length: {len(system_prompt_text)}")
    print(f"DEBUG: Lesson plan length: {len(user_lesson_plan_text)}")

    moderation_data_content: Optional[str] = None

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
                 raise RuntimeError("Gemini response has candidates but no 'text' attribute or text is null.")

            moderation_data_content = gemini_response.text
            print("DEBUG: Received content from Gemini.")

        except Exception as e:
            error_msg = f"Gemini API call or setup failed: {type(e).__name__} - {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    else: # Default to OpenAI or Azure OpenAI
        # Check if Azure OpenAI should be used
        if "azure" in current_llm_str_lower:
            print(f"DEBUG: Using Azure OpenAI model: {llm}")
            try:
                api_key = get_env_variable("AZURE_OPENAI_API_KEY")
                endpoint = get_env_variable("AZURE_OPENAI_ENDPOINT")
                api_version = get_env_variable("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                deployment_name = get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME")

                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                model_to_use = deployment_name
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")
        else:
            print(f"DEBUG: Using OpenAI model: {llm}")
            try:
                client = OpenAI()
                model_to_use = str(llm)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        messages_payload = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_lesson_plan_text}
        ]
        try:
            response = client.chat.completions.create(
                model=model_to_use,
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
        moderation_response = dynamic_response_model.model_validate_json(moderation_data_content)
        return moderation_response
    except ValidationError as e:
        print(f"Pydantic Validation error for dynamic moderation response: {e.errors(include_url=False)}")
        print(f"Problematic LLM response content that failed validation: {moderation_data_content}")
        raise RuntimeError(f"Invalid JSON from LLM for dynamic moderation response: {moderation_data_content}")
    except Exception as e:
        print(f"Error processing LLM response into dynamic moderation response: {type(e).__name__} - {e}")
        print(f"Problematic LLM response content: {moderation_data_content}")
        raise RuntimeError(f"Could not process response from LLM: {e}")
