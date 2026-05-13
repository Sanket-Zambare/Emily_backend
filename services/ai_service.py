import os
import json
import httpx
import base64
import asyncio
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

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents
            )
        )
        suggestions = _parse_response(response.text, wardrobe)
        return suggestions
    except Exception as e:
        print(f"Gemini error: {e}")
        return _get_mock_suggestions(occasion, wardrobe)


async def validate_clothing(image_base64: str, category: str) -> dict:
    try:
        image_bytes = base64.b64decode(image_base64)

        prompt = f"""You are a strict clothing validator for a fashion app.

The user is adding a clothing item to their wardrobe under category: {category}

CATEGORY MEANINGS:
- "top": anything worn on the upper body — shirts, tshirts, kurtas, blouses, tops, etc
- "bottom": anything worn on the lower body — pants, jeans, skirts, leggings, salwars, etc
- "full_outfit": a complete outfit — saree, lehenga set, jumpsuit, one-piece dress, co-ord set, salwar suit
- "outerwear": anything worn as outer layer — jackets, blazers, shrugs, cardigans, dupattas, stoles
- "footwear": anything worn on feet — shoes, sandals, heels, sneakers, juttis, kolhapuris
- "accessory": anything that accessorises — bags, jewellery, belts, scarves, sunglasses, watches

REJECT (return false) if:
- The main subject of the photo is NOT a clothing item
- Examples to reject: cupboard, laptop, bottle, book, food, furniture, wall, floor, face only, stationery
- The item is innerwear: bra, bikini, panty, underwear, lingerie, thong
- The item clearly does not match the selected category

ACCEPT (return true) if:
- The main subject IS a clothing item
- It reasonably fits the selected category
- User is wearing the item in the photo — trust them, they know their wardrobe

Be slightly lenient — if clothing is clearly the main subject, accept it.
Only reject if clearly wrong category or clearly not clothing or clearly innerwear.

Reply with ONLY this JSON, no other text:
{{"is_clothing": true, "reason": "brief reason"}}
or
{{"is_clothing": false, "reason": "brief reason"}}"""

        part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        text_part = types.Part.from_text(text=prompt)
        content = types.Content(role="user", parts=[text_part, part])

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[content]
            )
        )

        cleaned = response.text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        result = json.loads(cleaned.strip())
        print(f"Validation result: {result}")
        return result
    except Exception as e:
        print(f"Clothing validation error: {e}")
        return {"is_clothing": True, "reason": "validation failed, allowing upload"}


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

    profile_text = f"""You are EMILY, an expert Indian fashion stylist with deep knowledge of Indian fashion, culture, and body types.

You will be shown photos of clothing items from the user's wardrobe. Study them carefully.

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

    async with httpx.AsyncClient(timeout=10.0) as client_http:
        for i, item in enumerate(wardrobe):
            item_label = f"\nItem {i+1}: {item['category']}"
            if item.get('color'):
                item_label += f" - {item['color']}"
            item_label += f" (ID: {item['id']})"
            parts.append({"text": item_label})

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

    instruction = """
Now generate outfit suggestions using what you see in the photos above.

CATEGORY RULES — understand what each category means:
- "top": upper body clothing
- "bottom": lower body clothing  
- "full_outfit": complete outfit like saree, lehenga, jumpsuit, one-piece — do NOT suggest separate top or bottom with these, only footwear and accessories
- "outerwear": outer layer like jacket, blazer, dupatta, stole
- "footwear": shoes, sandals, heels etc
- "accessory": bags, jewellery, belts etc

STRICT OUTFIT RULES:
- ONLY combine items that genuinely look good together and suit the occasion
- If a wardrobe item is NOT suitable for this occasion, do NOT use it
- If two items clash in colour or style, do NOT combine them
- For "full_outfit" category items — only pair with footwear and accessories, never add separate top or bottom
- Suggestions 1 and 2: Use ONLY suitable wardrobe items. If fewer than 2 good combinations exist, say so honestly with empty items array
- Suggestions 3 and 4: Mix 1-2 suitable wardrobe items with new purchases
- Suggestions 5 and 6: Completely new outfit to buy

For each suggestion include relevant items based on category:
- Regular outfit: top + bottom + footwear + accessory
- Full outfit (saree/lehenga/jumpsuit): full_outfit item + footwear + accessory only

Consider Indian fashion deeply — colour theory for Indian skin tones, Indian occasion appropriateness, body type flattery.
For wardrobe items use the FULL UUID from "ID: ..." label.
For new purchases give specific searchable descriptions for Amazon/Flipkart India.

Return ONLY a JSON array, no markdown, no extra text:
[
  {
    "suggestion_number": 1,
    "title": "Short catchy outfit name",
    "uses_wardrobe": true,
    "occasion_score": 85,
    "items": [
      {
        "type": "top",
        "description": "What to wear",
        "from_wardrobe": true,
        "wardrobe_item_id": "FULL-UUID-HERE",
        "image_url": null,
        "buy_description": null,
        "reason": "Why this works for the occasion and user profile"
      },
      {
        "type": "bottom",
        "description": "What to wear",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Navy slim fit formal trousers for women",
        "reason": "Why this works"
      },
      {
        "type": "footwear",
        "description": "Footwear description",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Nude block heels for women",
        "reason": "Why this works"
      },
      {
        "type": "accessory",
        "description": "Accessory description",
        "from_wardrobe": false,
        "wardrobe_item_id": null,
        "image_url": null,
        "buy_description": "Gold jhumka earrings",
        "reason": "Why this completes the look"
      }
    ],
    "overall_reasoning": "One sentence why this complete outfit works for this user and occasion"
  }
]"""

    parts.append({"text": instruction})

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
    wardrobe_top_id = wardrobe[0]['id'] if len(wardrobe) > 0 else None
    wardrobe_bottom_id = wardrobe[1]['id'] if len(wardrobe) > 1 else None

    return [
        {
            "suggestion_number": 1,
            "title": "Classic & Confident",
            "uses_wardrobe": True,
            "occasion_score": 85,
            "items": [
                {"type": "top", "description": "Your wardrobe top", "from_wardrobe": True, "wardrobe_item_id": wardrobe_top_id, "image_url": None, "buy_description": None, "reason": "Clean and versatile"},
                {"type": "bottom", "description": "Navy blue slim trousers", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Navy blue slim fit formal trousers", "reason": "Professional pairing"},
                {"type": "footwear", "description": "Brown leather oxford shoes", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Brown leather oxford formal shoes", "reason": "Sophisticated finish"},
                {"type": "accessory", "description": "Brown leather belt", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Brown leather formal belt", "reason": "Completes the look"}
            ],
            "overall_reasoning": "A timeless professional combination"
        },
        {
            "suggestion_number": 2,
            "title": "Smart Casual",
            "uses_wardrobe": True,
            "occasion_score": 78,
            "items": [
                {"type": "top", "description": "Your wardrobe item", "from_wardrobe": True, "wardrobe_item_id": wardrobe_bottom_id or wardrobe_top_id, "image_url": None, "buy_description": None, "reason": "Versatile piece"},
                {"type": "bottom", "description": "Chinos in beige", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Beige slim fit chinos", "reason": "Relaxed yet smart"},
                {"type": "footwear", "description": "White sneakers", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "White canvas sneakers", "reason": "Casual and clean"},
                {"type": "accessory", "description": "Watch", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Minimalist analog watch", "reason": "Subtle style accent"}
            ],
            "overall_reasoning": "Effortlessly smart for everyday occasions"
        },
        {
            "suggestion_number": 3,
            "title": "Weekend Vibes",
            "uses_wardrobe": False,
            "occasion_score": 72,
            "items": [
                {"type": "top", "description": "Olive green shirt", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Olive green casual cotton shirt", "reason": "Earthy tone suits Indian skin tones"},
                {"type": "bottom", "description": "Dark jeans", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Dark blue slim fit jeans", "reason": "Classic pairing"},
                {"type": "footwear", "description": "Loafers", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Brown suede loafers", "reason": "Smart casual footwear"},
                {"type": "accessory", "description": "Sunglasses", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Aviator sunglasses", "reason": "Adds personality"}
            ],
            "overall_reasoning": "Perfect relaxed look for casual outings"
        },
        {
            "suggestion_number": 4,
            "title": "Evening Ready",
            "uses_wardrobe": False,
            "occasion_score": 80,
            "items": [
                {"type": "top", "description": "Black slim fit shirt", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Black slim fit casual shirt", "reason": "Black is universally flattering"},
                {"type": "bottom", "description": "Grey trousers", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Grey slim fit trousers", "reason": "Sophisticated contrast"},
                {"type": "footwear", "description": "Black derby shoes", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Black leather derby shoes", "reason": "Sharp and polished"},
                {"type": "accessory", "description": "Silver watch", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Silver dial dress watch", "reason": "Elegant finishing touch"}
            ],
            "overall_reasoning": "Sleek evening look that turns heads"
        },
        {
            "suggestion_number": 5,
            "title": "Bold Statement",
            "uses_wardrobe": False,
            "occasion_score": 70,
            "items": [
                {"type": "top", "description": "Printed kurta", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Block print cotton kurta for women", "reason": "Indian aesthetic with modern edge"},
                {"type": "bottom", "description": "White palazzo pants", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "White cotton palazzo pants for women", "reason": "Breezy and comfortable"},
                {"type": "footwear", "description": "Kolhapuri chappals", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Kolhapuri leather sandals for women", "reason": "Authentic Indian touch"},
                {"type": "accessory", "description": "Oxidised earrings", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Oxidised silver jhumka earrings", "reason": "Traditional elegance"}
            ],
            "overall_reasoning": "Celebrate Indian fashion with this bold ethnic look"
        },
        {
            "suggestion_number": 6,
            "title": "Fresh & Modern",
            "uses_wardrobe": False,
            "occasion_score": 75,
            "items": [
                {"type": "top", "description": "Pastel blue linen shirt", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Pastel blue linen casual shirt for women", "reason": "Cool tone flatters warm skin"},
                {"type": "bottom", "description": "White linen trousers", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "White linen straight fit trousers for women", "reason": "Breathable summer combo"},
                {"type": "footwear", "description": "Tan sandals", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Tan leather casual sandals for women", "reason": "Relaxed and stylish"},
                {"type": "accessory", "description": "Woven bracelet", "from_wardrobe": False, "wardrobe_item_id": None, "image_url": None, "buy_description": "Woven thread bracelet set for women", "reason": "Casual boho accent"}
            ],
            "overall_reasoning": "Light and breezy for warm Indian weather"
        }
    ]