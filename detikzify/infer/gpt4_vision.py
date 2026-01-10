"""
GPT-4 Vision Backend for TikZ Generation

Uses OpenAI's GPT-4 Vision API to generate TikZ code from images.
Provides higher quality outputs for complex diagrams compared to DeTikZify.
"""

import os
import base64
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger("gpt4-vision")

# TikZ generation prompt
TIKZ_PROMPT = """You are an expert LaTeX/TikZ programmer. Analyze this image and generate TikZ code that recreates it as accurately as possible.

Requirements:
1. Output ONLY the TikZ code, no explanations.
2. Start with \\documentclass and end with \\end{document}.
3. Include all necessary packages (tikz, amsmath, etc.).
4. Be precise with coordinates and proportions.
5. Use appropriate TikZ libraries for the diagram type.
6. Include labels, annotations, and text exactly as shown.

Generate the complete, compilable LaTeX document:"""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def generate_tikz_with_gpt4(image: Image.Image, api_key: str = None) -> str:
    """
    Generate TikZ code from an image using GPT-4 Vision.
    
    Args:
        image: PIL Image to convert to TikZ
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Generated TikZ code as a string
    
    Raises:
        ValueError: If no API key is provided
        Exception: If API call fails
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for GPT-4 Vision. Install with: pip install httpx")
    
    # Convert image to base64
    base64_image = image_to_base64(image)
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",  # GPT-4 with vision
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": TIKZ_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.2  # Lower temperature for more deterministic code generation
    }
    
    logger.info("Calling GPT-4 Vision API...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"GPT-4 API error: {error_msg}")
            raise Exception(f"GPT-4 API error ({response.status_code}): {error_msg}")
        
        result = response.json()
        tikz_code = result["choices"][0]["message"]["content"]
        
        # Clean up the response (remove markdown code blocks if present)
        if tikz_code.startswith("```"):
            lines = tikz_code.split("\n")
            # Remove first and last lines if they are code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            tikz_code = "\n".join(lines)
        
        logger.info("GPT-4 Vision generation complete.")
        return tikz_code.strip()
