import os
from supabase import create_client, Client
# from dotenv import load_dotenv
# from pathlib import Path

# # Load .env explicitly
# env_path = Path(__file__).resolve().parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)

def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    return create_client(url, key)

async def get_user_profile(user_id: str):
    try:
        supabase = get_supabase()
        response = supabase.table('users').select(
            'id, gender, body_type, preferred_fit, height_cm, '
            'skin_tone, top_size, bottom_size, shoe_size, city'
        ).eq('id', user_id).single().execute()
        return response.data
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

async def get_wardrobe_items(user_id: str):
    try:
        supabase = get_supabase()
        response = supabase.table('wardrobe_items').select(
            'id, category, color, fabric, image_url, ai_tags'
        ).eq('user_id', user_id).eq('is_available', True).execute()
        return response.data or []
    except Exception as e:
        print(f"Error fetching wardrobe: {e}")
        return []

async def get_outfit_request(request_id: str):
    try:
        supabase = get_supabase()
        response = supabase.table('outfit_requests').select(
            '*'
        ).eq('id', request_id).single().execute()
        return response.data
    except Exception as e:
        print(f"Error fetching outfit request: {e}")
        return None

async def save_suggestion(request_id: str, suggestions: list):
    try:
        supabase = get_supabase()
        response = supabase.table('outfit_suggestions').insert({
            'request_id': request_id,
            'product_ids': [],
            'ai_reasoning': str(suggestions),
            'wardrobe_item_ids': []
        }).execute()
        return response.data[0]
    except Exception as e:
        print(f"Error saving suggestion: {e}")
        return {'id': 'temp_id'}