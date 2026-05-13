from fastapi import APIRouter, Request
from services.ai_service import generate_outfit, validate_clothing
from services.supabase_service import get_user_profile, get_wardrobe_items

router = APIRouter()


@router.post("/generate")
async def generate_outfit_route(request: Request):
    try:
        body = await request.json()
        request_id = body.get('request_id')
        user_id = body.get('user_id')

        if not request_id or not user_id:
            return {"error": "Missing request_id or user_id"}

        user_profile = await get_user_profile(user_id)
        wardrobe = await get_wardrobe_items(user_id)

        from services.supabase_service import get_outfit_request
        outfit_request = await get_outfit_request(request_id)
        occasion = outfit_request.get('occasion_slug', 'college_casual')
        extra_context = outfit_request.get('extra_context', '')

        suggestions = await generate_outfit(
            user_profile=user_profile,
            wardrobe=wardrobe,
            occasion=occasion,
            extra_context=extra_context
        )

        return {"suggestions": suggestions}

    except Exception as e:
        print(f"Outfit generation error: {e}")
        return {"error": str(e)}


@router.post("/validate-item")
async def validate_wardrobe_item(request: Request):
    try:
        body = await request.json()
        image_base64 = body.get('image_base64', '')
        category = body.get('category', 'top')

        if not image_base64:
            return {"is_clothing": True, "reason": "no image provided"}

        result = await validate_clothing(image_base64, category)
        return result

    except Exception as e:
        print(f"Validation error: {e}")
        return {"is_clothing": True, "reason": "error, allowing upload"}