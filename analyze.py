import json
import os
from typing import Any, Dict, List
from litellm import completion
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please create a .env file with your API key.")


class TravelItinerary(BaseModel):
    """Schema for travel itinerary response"""
    destination: str = Field(description="The destination name")
    price_range: str = Field(description="Expected price range (e.g., '$', '$$', '$$$')")
    ideal_visit_times: List[str] = Field(description="Best times to visit (e.g., seasons or months)")
    top_attractions: List[str] = Field(description="List of top attractions to visit")


def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # Create a prompt requesting structured travel information
    prompt = f"""Generate a travel itinerary for {destination}.

Return your response as a JSON object with the following structure:
- destination: the destination name
- price_range: expected price range (use $, $$, or $$$)
- ideal_visit_times: array of best times to visit (seasons or months)
- top_attractions: array of top attractions to visit

Respond with ONLY valid JSON, no additional text."""

    # Make the LiteLLM API call with structured JSON response
    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful travel advisor. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        api_key=api_key,
        response_format={"type": "json_object"}
    )

    # Extract the response content
    response_text = response.choices[0].message.content

    # Parse the JSON response
    data = json.loads(response_text)

    # Validate against the Pydantic schema
    try:
        validated_itinerary = TravelItinerary(**data)
        # Return as dict
        return validated_itinerary.model_dump()
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise ValueError(f"LLM response did not match expected schema: {e}")

    return data
