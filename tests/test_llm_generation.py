import asyncio
import pytest
from unittest.mock import AsyncMock
from src.llm.llm_report import ForecastDiscussionGenerator

def test_prompt_construction():
    generator = ForecastDiscussionGenerator("fake_key")
    coords = [(34.2, -86.5), (35.1, -87.1)]
    features = {
        "SBCAPE": 0.45,
        "0-1km_SRH": 0.30,
        "LCL_Height": 0.15
    }
    
    prompt = generator.construct_prompt(coords, features)
    
    # Assert correctness of prompt chaining
    assert "34.20, -86.50" in prompt
    assert "SBCAPE: 0.45" in prompt
    assert "0-1km_SRH" in prompt
    assert "LCL_Height" in prompt
    
    print("Constructed Prompt:")
    print(prompt)

@pytest.mark.asyncio
async def test_llm_generation(monkeypatch):
    generator = ForecastDiscussionGenerator("fake_key")
    
    # Mock the AsyncOpenAI client response to simulate OpenAI without network/costs
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(message=AsyncMock(content="MESOSCALE DISCUSSION.\n\nTHREAT: Tornadoes and severe hail.\n\nA volatile airmass is developing over (34.2, -86.5) driven primarily by extreme SBCAPE and enhanced low-level shear (0-1km SRH)."))
    ]
    
    # Monkeypatch the chat.completions.create method
    # Since AsyncOpenAI creates `client.chat.completions.create`, we must traverse down
    async def mock_create(*args, **kwargs):
        return mock_response
        
    monkeypatch.setattr(generator.client.chat.completions, "create", mock_create)
    
    report = await generator.generate_discussion([(34.2, -86.5)], {"SBCAPE": 0.45})
    
    print("\nMock LLM Report Generated:")
    print(report)
    
    assert "MESOSCALE DISCUSSION" in report
    assert "SBCAPE" in report
    
    print("\nStep 4 AI/LLM Integration: Prompting and Generator orchestration correctly established.")

if __name__ == "__main__":
    test_prompt_construction()
    asyncio.run(test_llm_generation())
