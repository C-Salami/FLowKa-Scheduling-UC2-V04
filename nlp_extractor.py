import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_INSTRUCTIONS = """
You are an assistant that interprets production scheduling commands.

Return ONLY a valid JSON object with:
- intent: "delay_order" | "swap_orders" | "unknown"
- order_id: string like "ORD-001"
- order_id_2: string (only if swap)
- days: float (optional)
- hours: float (optional)
- minutes: float (optional)

Rules:
- 'order 1' or 'Order one' → 'ORD-001'
- If delay/advance mentioned, extract time values (days/hours/minutes).
- If swap mentioned, extract both orders.
- If unclear, return intent = "unknown".
"""

def ai_extract_intent(text: str) -> dict:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_INSTRUCTIONS},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # try parse JSON from reply
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {"intent": "unknown", "_error": str(e), "_raw": str(response)}
