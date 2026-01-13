"""
GPT-4 Vision Pre-Processing for DeTikZify

Generates a TikZ code skeleton that DeTikZify can continue/refine.
"""

import os
import base64
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger("gpt4-preprocess")

# Generate a TikZ SKELETON, not full code - DeTikZify will complete it
SKELETON_PROMPT = """Analyze this diagram and generate the BEGINNING of TikZ code that sets up the structure.

Generate ONLY:
1. \\documentclass and \\usepackage statements
2. \\begin{document} and \\begin{tikzpicture}
3. Style definitions (\\tikzset)
4. Key dimension definitions (\\def)
5. Comments describing what each section should contain

STOP after setting up the structure. Do NOT complete the drawing.
The code should compile but be incomplete - another model will finish it.

Example output format:
```
\\documentclass[border=10pt]{standalone}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta, calc}

\\begin{document}
\\begin{tikzpicture}
    % Styles
    \\tikzset{...}
    
    % Dimensions
    \\def\\containerWidth{8}
    ...
    
    % TODO: Draw container
    % TODO: Draw water
    % TODO: Draw rod at angle
    % TODO: Add labels
```

Generate the skeleton now:"""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def analyze_image_with_gpt4(image: Image.Image, api_key: str = None) -> str:
    """
    Generate a TikZ skeleton from an image using GPT-4 Vision.
    
    Returns partial TikZ code that DeTikZify can continue.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required. Install: pip install httpx")
    
    base64_image = image_to_base64(image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SKELETON_PROMPT},
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
        "max_tokens": 800,
        "temperature": 0.2
    }
    
    logger.info("GPT-4: Generating TikZ skeleton...")
    print(">>> GPT-4: Generating TikZ skeleton...", flush=True)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"GPT-4 API error: {response.text}")
            raise Exception(f"GPT-4 API error: {response.text}")
        
        result = response.json()
        skeleton = result["choices"][0]["message"]["content"]
        
        # Clean up markdown code blocks
        if "```" in skeleton:
            lines = skeleton.split("\n")
            cleaned = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code or line.strip().startswith("\\"):
                    cleaned.append(line)
            skeleton = "\n".join(cleaned)
        
        logger.info(f"GPT-4 skeleton generated ({len(skeleton)} chars)")
        print(f">>> GPT-4 Skeleton:\n{skeleton[:300]}...", flush=True)
        
        return skeleton.strip()


def format_context_prompt(skeleton: str) -> str:
    """
    The skeleton IS the prompt - DeTikZify will continue generating from it.
    """
    return skeleton
