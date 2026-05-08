from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.ai_service import generate_outfit
from services.supabase_service import get_user_profile, get_wardrobe_items, save_suggestion

router = APIRouter()

class OutfitRequest(BaseModel):
    request_id: str
    user_id: str

@router.post("/generate")
async def generate_outfit_endpoint(request: OutfitRequest):
    try:
        # Step 1 — Get user profile from Supabase
        user_profile = await get_user_profile(request.user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        # Step 2 — Get user's wardrobe from Supabase
        wardrobe = await get_wardrobe_items(request.user_id)

        # Step 3 — Get the outfit request details
        from services.supabase_service import get_outfit_request
        outfit_request = await get_outfit_request(request.request_id)
        if not outfit_request:
            raise HTTPException(status_code=404, detail="Request not found")

        # Step 4 — Generate outfit using Gemini AI
        suggestions = await generate_outfit(
            user_profile=user_profile,
            wardrobe=wardrobe,
            occasion=outfit_request['occasion_slug'],
            extra_context=outfit_request.get('extra_context', '')
        )

        # Step 5 — Save suggestions to database
        saved = await save_suggestion(
            request_id=request.request_id,
            suggestions=suggestions
        )

        return {
            "success": True,
            "suggestions": suggestions,
            "suggestion_id": saved['id']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))