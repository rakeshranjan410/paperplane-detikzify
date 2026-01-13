import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import logging

from .model import load
from .infer.generate import DetikzifyPipeline


import sys

# Setup logging with forced flush for EC2 visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # Override any existing logging config
)
# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
logger = logging.getLogger("detikzify-api")

# Global variables
pipeline = None
MODEL_NAME = "nllg/detikzify-v2.5-8b"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading model...")
    # MPS Memory Fix logic should be in environment variables (start_server.sh)
    
    # Load model
    # We use similar logic to what we debugged: force float16, no auto device map
    try:
        model, processor = load(
            model_name_or_path=MODEL_NAME,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        logger.info(f"Moving model to {device}...")
        model.to(device)
        logger.info(f"Model loaded on {device} with dtype {model.dtype}")
        
        pipeline = DetikzifyPipeline(
            model=model, 
            processor=processor,
            temperature=0.8,
            top_p=0.95,
            fast_metric=False,  # Use SelfSim for perceptual similarity (better quality)
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
        
    yield
    # Clean up if needed
    pipeline = None

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateResponse(BaseModel):
    tikz: str
    backend: str = "detikzify"

@app.post("/generate", response_model=GenerateResponse)
async def generate_tikz(
    file: UploadFile = File(...),
    backend: str = "detikzify"  # Options: "detikzify", "gpt4", or "hybrid"
):
    """
    Generate TikZ code from an image.
    
    Args:
        file: Image file to convert
        backend: Which backend to use
            - "detikzify": Open-source DeTikZify model (default, free, lower quality for complex images)
            - "gpt4": GPT-4 Vision (requires OPENAI_API_KEY, higher quality, ~$0.02 per image)
            - "hybrid": GPT-4 analyzes image → DeTikZify generates TikZ with context (~$0.01 per image, best quality)
    """
    # IMMEDIATE diagnostic - this MUST print if request reaches endpoint
    print("=" * 50, flush=True)
    print(">>> ENDPOINT HIT: /generate", flush=True)
    print(f">>> Backend: {backend}", flush=True)
    print(f">>> File: {file.filename}", flush=True)
    print("=" * 50, flush=True)
    try:
        contents = await file.read()
        logger.info(f"Received image: {len(contents)} bytes, backend={backend}")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"Image processed. Size: {image.size}")
        
        if backend == "gpt4":
            # Use GPT-4 Vision for direct TikZ generation
            from .infer.gpt4_vision import generate_tikz_with_gpt4
            
            logger.info("Using GPT-4 Vision backend...")
            tikz_code = await generate_tikz_with_gpt4(image)
            
            if tikz_code:
                logger.info("GPT-4 Vision generation completed successfully.")
                return GenerateResponse(tikz=tikz_code, backend="gpt4")
            else:
                raise Exception("GPT-4 failed to generate TikZ code.")
        
        elif backend == "hybrid":
            # HYBRID: GPT-4 analyzes → DeTikZify generates with context
            from .infer.gpt4_preprocess import analyze_image_with_gpt4, format_context_prompt
            
            if not pipeline:
                raise HTTPException(status_code=503, detail="DeTikZify model not loaded")
            
            logger.info("Using HYBRID backend: GPT-4 analysis + DeTikZify generation...")
            
            # Step 1: Get structural description from GPT-4
            print(">>> Step 1: GPT-4 Vision analyzing image structure...", flush=True)
            description = await analyze_image_with_gpt4(image)
            context_prompt = format_context_prompt(description)
            
            print(f">>> Step 2: DeTikZify generating TikZ with context...", flush=True)
            logger.info(f"GPT-4 description: {description[:200]}...")
            
            # Step 2: Use DeTikZify with the context prompt
            best_code = None
            best_score = float("-inf")
            
            # Pass the context as 'text' parameter - this guides the model
            for score, tikz_doc in pipeline.simulate(
                image=image, 
                text=context_prompt,  # <-- Key difference: contextual guidance
                expansions=None, 
                timeout=180
            ):
                logger.info(f"Generated candidate with score: {score}")
                if score > best_score:
                    best_score = score
                    best_code = tikz_doc.code
                    
            if best_code:
                logger.info("HYBRID backend completed successfully.")
                return GenerateResponse(tikz=best_code, backend="hybrid")
            else:
                raise Exception("HYBRID backend failed to generate valid TikZ code.")
        
        else:
            # Use DeTikZify (default)
            if not pipeline:
                raise HTTPException(status_code=503, detail="DeTikZify model not loaded")
            
            logger.info("Using DeTikZify backend (MCTS, timeout=180s)...")
            best_code = None
            best_score = float("-inf")
            
            for score, tikz_doc in pipeline.simulate(image=image, expansions=None, timeout=180):
                logger.info(f"Generated candidate with score: {score}")
                if score > best_score:
                    best_score = score
                    best_code = tikz_doc.code
                    
            if best_code:
                logger.info("DeTikZify completed successfully.")
                return GenerateResponse(tikz=best_code, backend="detikzify")
            else:
                raise Exception("DeTikZify failed to generate valid TikZ code.")
                
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

