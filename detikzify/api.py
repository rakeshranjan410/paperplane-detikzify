import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import logging

from .model import load
from .infer.generate import DetikzifyPipeline


# Setup logging
logging.basicConfig(level=logging.INFO)
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

@app.post("/generate", response_model=GenerateResponse)
async def generate_tikz(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        logger.info(f"Received image: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"Image processed. Size: {image.size}. Starting inference...")
        
        # Run inference using time-based MCTS (per whitepaper recommendation)
        # Let MCTS explore as much as possible within the time budget.
        # More time = more simulations = better quality.
        
        logger.info("Starting MCTS simulation (timeout=180s)...")
        best_code = None
        best_score = float("-inf")
        
        for score, tikz_doc in pipeline.simulate(image=image, expansions=None, timeout=180):
            logger.info(f"Generated candidate with score: {score}")
            if score > best_score:
                best_score = score
                best_code = tikz_doc.code
                
        if best_code:
            logger.info("Inference completed successfully. Returning best response.")
            return GenerateResponse(tikz=best_code)
        else:
             raise Exception("Failed to generate any valid TikZ code.")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
