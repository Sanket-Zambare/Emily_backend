import os
import json
import httpx
import base64
from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


async def generate_outfit(
    user_profile: dict,
    wardrobe: list,
    occasion: str,
    extra_context: str = ''
) -> list:
    try:
        contents = await _build_multimodal_contents(
            user_profile=user_profile,
            wardrobe=wardrobe,
            occasion=occasion,
            extra_context=extra_context
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        suggestions = _parse_response(response.text, wardrobe)
        return suggestions
    except Exception as e:
        print(f"Gemini error: {e}")
        return _get_mock_suggestions(occasion, wardrobe)


async def _build_multimodal_contents(
    user_profile: dict,
    wardrobe: list,
    occasion: str,
    extra_context: str
) -> list:

    occasion_map = {
        'job_interview': 'Job Interview - professional, confident, trustworthy',
        'date_night': 'Date Night - attractive, stylish, put together',
        'wedding_guest': 'Wedding Guest - festive, elegant, celebratory',
        'party': 'Party - fun, expressive, eye-catching',
        'college_casual': 'College/Casual - relaxed, cool, comfortable',
        'festive_pooja': 'Festive/Pooja - traditional, graceful, culturally appropriate'
    }
    occasion_desc = occasion_map.get(occasion, occasion)

    parts = []

    profile_text = f"""You are EMILY, an expert Indian fashion stylist.
You will be shown photos of clothing items from the user's wardrobe. Study them carefully and suggest outfits.

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

USER'S WARDROBE ITEMS (photos follow):"""

    parts.append({"text": profile_text})

    # Add each wardrobe item — label + photo
    async with httpx.AsyncClient(timeout=10.0) as client_http:
        for i, item in enumerate(wardrobe):
            item_label = f"\nItem {i+1}: {item['category']}"
            if item.get('color'):
                item_label += f" - {item['color']}"
            item_label += f" (ID: {item['id']})"
            parts.append({"text": item_label})

            # Try to load the actual image
            if item.get('image_url'):
                try:
                    resp = await client_http.get(item['image_url'])
                    if resp.status_code == 200:
                        img_b64 = base64.b64encode(resp.content).decode()
                        content_type = resp.headers.get('content-type', 'image/jpeg')
                        parts.append({
                            "inline_data": {
                                "mime_type": content_type,
                                "data": img_b64
                            }
                        })
                except Exception as img_err:
                    print(f"Could not load image for item {item['id']}: {img_err}")

    # Final instruction
    instruction = """
Now generate 6 complete outfit suggestions using what you see in the photos above.

RULES:
- Suggestions 1 and 2: Use ONLY items from the wardrobe photos shown. Style what they already own beautifully.
- Suggestions 3 and 4: Mix 1-2 wardrobe items with complementary new purchases.
- Suggestions 5 and 6: Completely new outfit to buy. No wardrobe items.

For each suggestion include: top, bottom, footwear, and one accessory.
Consider Indian fashion context, skin tone for color recommendations, body type for fit.
For wardrobe items, use the FULL UUID shown as "ID: ..." in the label above the photo.
For new purchases, give specific item descriptions for Amazon/Flipkart search.

Return ONLY a JSON array, no markdown, no extra text:
[
  {
    "suggestion_number": 1,
    "title": "Short catchy outfit name",
    "uses_wardrobe": true,
    "items": [
      {
        "type": "top",
        "description": "What to wear",
        "from_wardrobe": true,
        "wardrobe_item_id": "FULL-UUID-HERE",
        "image_url": null,
        "buy_description": null,
        "reason": "Why this works for the occasion and user skin tone/body type"
      },
      {
        "type": "bottom",
        "description": "What to wear",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Navy slim fit formal trousers for men",
        "reason": "Why this works"
      },
      {
        "type": "footwear",
        "description": "Footwear description",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Brown leather oxford shoes",
        "reason": "Why this works"
      },
      {
        "type": "accessory",
        "description": "Accessory description",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Brown leather belt",
        "reason": "Why this completes the look"
      }
    ],
    "overall_reasoning": "One sentence why this complete outfit works"
  }
]"""

    parts.append({"text": instruction})

    # Convert to Gemini content format
    gemini_parts = []
    for part in parts:
        if "text" in part:
            gemini_parts.append(types.Part.from_text(text=part["text"]))
        elif "inline_data" in part:
            gemini_parts.append(types.Part.from_bytes(
                data=base64.b64decode(part["inline_data"]["data"]),
                mime_type=part["inline_data"]["mime_type"]
            ))

    return [types.Content(role="user", parts=gemini_parts)]


def _parse_response(response_text: str, wardrobe: list) -> list:
    try:
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
                    "wardrobe_item_id": wardrobe[0]['id'] if wardrobe else None,
                    "image_url": None,
                    "buy_description": None if wardrobe else "White cotton formal shirt",
                    "reason": "Clean white works for any occasion and complements all skin tones"
                },
                {
                    "type": "bottom",
                    "description": "Navy blue slim trousers",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "image_url": None,
                    "buy_description": "Navy blue slim fit formal trousers",
                    "reason": "Navy is professional and pairs perfectly with white"
                },
                {
                    "type": "footwear",
                    "description": "Brown leather oxford shoes",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "image_url": None,
                    "buy_description": "Brown leather oxford formal shoes",
                    "reason": "Brown leather adds warmth and sophistication"
                },
                {
                    "type": "accessory",
                    "description": "Brown leather belt",
                    "from_wardrobe": False,
                    "wardrobe_item_id": None,
                    "image_url": None,
                    "buy_description": "Brown leather formal belt",
                    "reason": "Matches shoes and completes the professional look"
                }
            ],
            "overall_reasoning": "A timeless professional combination that works for any formal occasion"
        }
    ]