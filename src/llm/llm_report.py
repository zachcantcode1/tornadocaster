import os
from typing import List, Dict

class ForecastDiscussionGenerator:
    """
    Acts as an automated SPC Meteorologist. Takes in prediction coordinates 
    and feature importances from LightGBM to yield a professional meteorological report.
    """
    
    SYSTEM_PROMPT = (
        "You are an expert SPC (Storm Prediction Center) Meteorologist. "
        "Given the following coordinate map of peak severe weather threats "
        "and the associated driving atmospheric variables from a LightGBM model, "
        "write a highly technical, concise, mesoscale forecast discussion explaining "
        "why the threat exists in these specific areas."
    )
    
    def __init__(self, api_key: str = None):
        try:
            from openai import AsyncOpenAI
            # Use provided key or fallback to environment var, or fake it for test mode
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-mock-key-for-testing")
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            self.client = None

    def construct_prompt(self, 
                         peak_coords: List[tuple], 
                         driving_features: Dict[str, float]) -> str:
        """
        Creates the factual boundary prompt combining the coordinates
        and feature importance multipliers.
        """
        coords_str = ", ".join([f"({lat:.2f}, {lon:.2f})" for lat, lon in peak_coords])
        features_str = "\n".join([f"- {name}: {weight:.2f} importance" for name, weight in driving_features.items()])
        
        prompt = (
            f"Observed Threat Maxima Coordinates (Lat/Lon): {coords_str}\n\n"
            f"Primary Driving Variables:\n{features_str}\n\n"
            f"Please write the Mesoscale Discussion."
        )
        return prompt

    async def generate_discussion(self, peak_coords: List[tuple], driving_features: Dict[str, float]) -> str:
        """
        Requests the generation from the LLM. 
        Note: Mocked out in tests via monkeypatching to avoid real API costs.
        """
        prompt = self.construct_prompt(peak_coords, driving_features)
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temperature for more deterministic, factual tone
            max_tokens=500
        )
        return response.choices[0].message.content
