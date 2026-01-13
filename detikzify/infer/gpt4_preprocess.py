"""
GPT-4 Vision Pre-Processing for DeTikZify

Analyzes images with GPT-4 Vision to provide structural descriptions
that guide the DeTikZify model for better TikZ generation.
"""

import os
import base64
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger("gpt4-preprocess")

# Analysis prompt - focuses on description, NOT generating TikZ
ANALYSIS_PROMPT = """Analyze this diagram image and provide a structured description for TikZ code generation.

Describe the following:

1. **Diagram Type**: What kind of diagram is this? (physics, geometry, circuit, flowchart, etc.)

2. **Canvas Layout**: 
   - Approximate aspect ratio
   - Key coordinate regions (what's at top, bottom, left, right, center)

3. **Main Elements** (list each with position):
   - Shapes (rectangles, circles, triangles, lines, curves)
   - Their approximate positions and sizes relative to each other
   - Colors and fill patterns

4. **Annotations and Labels**:
   - All text/math labels (θ, b, x, y, etc.)
   - Where each label appears relative to other elements

5. **Connections and Relationships**:
   - Lines, arrows, or arcs connecting elements
   - Angle indicators, dimension markers

6. **Critical Details**:
   - Any angles (approximate degrees)
   - Proportions (e.g., "rod length is about 1.5x container height")
   - Special visual effects (shading, gradients, transparency)

Be precise and use directional language (top-left, center-right, etc.).
Format as a structured list. Do NOT generate any code."""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def analyze_image_with_gpt4(image: Image.Image, api_key: str = None) -> str:
    """
    Analyze an image with GPT-4 Vision and return a structural description.
    
    Args:
        image: PIL Image to analyze
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Structural description of the image for TikZ generation
    
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
        "model": "gpt-4o-mini",  # Use mini for cost efficiency - analysis doesn't need full model
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ANALYSIS_PROMPT
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
        "max_tokens": 1000,  # Description doesn't need as many tokens as code
        "temperature": 0.3  # Slightly higher for more natural descriptions
    }
    
    logger.info("Calling GPT-4 Vision for image analysis...")
    print(">>> GPT-4 Pre-processing: Analyzing image structure...", flush=True)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
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
        description = result["choices"][0]["message"]["content"]
        
        logger.info("GPT-4 Vision analysis complete.")
        print(f">>> GPT-4 Analysis Result:\n{description[:500]}...", flush=True)
        
        return description.strip()


def format_context_prompt(description: str) -> str:
    """
    Format the GPT-4 description into a context prompt for DeTikZify.
    
    The DeTikZify model expects a text prompt that provides context.
    This formats the structural description appropriately.
    """
    return f"""Generate TikZ code for the following diagram:

{description}

Create accurate, compilable TikZ code matching this description:"""
