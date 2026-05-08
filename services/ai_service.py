import os
import json
from google import genai
from dotenv import load_dotenv
from pathlib import Path

# Debug — print what path we are looking at
# env_path = Path(__file__).resolve().parent.parent / '.env'

# load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")


client = genai.Client(api_key=api_key)


async def generate_outfit(
    user_profile: dict,
    wardrobe: list,
    occasion: str,
    extra_context: str = ''
) -> list:
    
    # Build wardrobe description for AI
    wardrobe_desc = _build_wardrobe_description(wardrobe)
    
    # Build the prompt
    prompt = _build_prompt(
        user_profile=user_profile,
        wardrobe_desc=wardrobe_desc,
        occasion=occasion,
        extra_context=extra_context,
        wardrobe_items=wardrobe
    )
    
    try:
        response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
)
        suggestions = _parse_response(response.text, wardrobe)
        return suggestions
    except Exception as e:
        print(f"Gemini error: {e}")
        # Return mock suggestions if AI fails
        return _get_mock_suggestions(occasion, wardrobe)

def _build_wardrobe_description(wardrobe: list) -> str:
    if not wardrobe:
        return "User has no items in wardrobe yet."
    
    desc = []
    for item in wardrobe:
        tags = item.get('ai_tags', {})
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = {}
        
        item_desc = f"- {item['category']}"
        if item.get('color'):
            item_desc += f" ({item['color']})"
        if tags.get('type'):
            item_desc += f" - {tags['type']}"
        item_desc += f" [ID: {item['id'][:8]}]"
        desc.append(item_desc)
    
    return '\n'.join(desc)

def _build_prompt(
    user_profile: dict,
    wardrobe_desc: str,
    occasion: str,
    extra_context: str,
    wardrobe_items: list
) -> str:
    
    occasion_map = {
        'job_interview': 'Job Interview - professional, confident, trustworthy',
        'date_night': 'Date Night - attractive, stylish, put together',
        'wedding_guest': 'Wedding Guest - festive, elegant, celebratory',
        'party': 'Party - fun, expressive, eye-catching',
        'college_casual': 'College/Casual - relaxed, cool, comfortable',
        'festive_pooja': 'Festive/Pooja - traditional, graceful, culturally appropriate'
    }
    
    occasion_desc = occasion_map.get(occasion, occasion)
    has_wardrobe = len(wardrobe_items) > 0
    
    prompt = f"""You are EMILY, an expert Indian fashion stylist. Generate 6 complete outfit suggestions.

USER PROFILE:
- Gender: {user_profile.get('gender', 'not specified')}
- Body type: {user_profile.get('body_type', 'not specified')}
- Height: {user_profile.get('height_cm', 'not specified')} cm
- Skin tone: {user_profile.get('skin_tone', 'not specified')}
- Preferred fit: {user_profile.get('preferred_fit', 'regular')}
- Top size: {user_profile.get('top_size', 'M')}
- Bottom size: {user_profile.get('bottom_size', '32')}
- Shoe size: {user_profile.get('shoe_size', '8')}
- City: {user_profile.get('city', 'India')}

OCCASION: {occasion_desc}
EXTRA CONTEXT: {extra_context if extra_context else 'None'}

USER'S WARDROBE:
{wardrobe_desc}

RULES FOR 6 SUGGESTIONS:
- Suggestions 1 and 2: Use ONLY items from user's wardrobe. Style what they already own.
- Suggestions 3 and 4: Mix wardrobe items with new purchases. Keep one or two pieces from wardrobe, suggest complementary new items.
- Suggestions 5 and 6: Completely new outfit to buy. Fresh look entirely.

For each suggestion always include: top, bottom, footwear, and one accessory.
Consider Indian fashion context, skin tone for color recommendations, body type for fit advice.
For new purchase suggestions, specify exact item description so we can find it on Amazon/Flipkart.

Return ONLY a JSON array with exactly this structure, no other text:
[
  {{
    "suggestion_number": 1,
    "title": "Short catchy outfit name",
    "uses_wardrobe": true,
    "items": [
      {{
        "type": "top",
        "description": "What to wear",
        "from_wardrobe": true,
        "wardrobe_item_id": "first 8 chars of item ID or null",
        "buy_description": null,
        "reason": "Why this works for the occasion and skin tone"
      }},
      {{
        "type": "bottom",
        "description": "What to wear",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "buy_description": "Specific item to search and buy e.g. navy slim fit chino trouser",
        "reason": "Why this works"
      }},
      {{
        "type": "footwear",
        "description": "What to wear",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "buy_description": "Specific item to search and buy",
        "reason": "Why this works"
      }},
      {{
        "type": "accessory",
        "description": "What accessory to add",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "buy_description": "Specific accessory to buy",
        "reason": "Why this completes the look"
      }}
    ],
    "overall_reasoning": "One sentence on why this complete outfit works for the occasion"
  }}
]"""
    
    return prompt

def _parse_response(response_text: str, wardrobe: list) -> list:
    try:
        # Clean the response — remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        suggestions = json.loads(cleaned)
        return suggestions
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw response: {response_text}")
        return _get_mock_suggestions('casual', wardrobe)

def _get_mock_suggestions(occasion: str, wardrobe: list) -> list:
    # Fallback mock suggestions if AI fails
    return [
        {
            "suggestion_number": 1,
            "title": "Classic & Confident",
            "uses_wardrobe": True,
            "items": [
                {
                    "type": "top",
                    "description": "White formal shirt",
                    "from_wardrobe": len(wardrobe) > 0,
                    "wardrobe_item_id": wardrobe[0]['id'][:8] if wardrobe else None,
                    "buy_description": None if wardrobe else "White cotton formal shirt",
                    "reason": "Clean white works for any occasion and complements all skin tones"
                },
                {
                    "type": "bottom",
                    "description": "Navy blue slim trousers",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "buy_description": "Navy blue slim fit formal trousers",
                    "reason": "Navy is professional and pairs perfectly with white"
                },
                {
                    "type": "footwear",
                    "description": "Brown leather oxford shoes",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "buy_description": "Brown leather oxford formal shoes",
                    "reason": "Brown leather adds warmth and sophistication"
                },
                {
                    "type": "accessory",
                    "description": "Brown leather belt",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "buy_description": "Brown leather formal belt",
                    "reason": "Matches shoes and completes the professional look"
                }
            ],
            "overall_reasoning": "A timeless professional combination that works for any formal occasion"
        }
    ]