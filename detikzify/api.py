import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import logging

from .model import load
from .infer.generate import DetikzifyPipeline
from torch import float16

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
            torch_dtype=float16,
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
            top_p=0.95
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
        
        # Run inference
        # Run inference using simple sampling (much faster than MCTS)
        # Using MCTS on a single GPU can still be slow due to multiple expansions.
        # For a lean service, we prioritize speed.
        
        logger.info("Starting inference (Sampling)...")
        
        # Generate with sampling
        # We can enable beam search in gen_kwargs if needed, but sample is standard.
        tikz_doc = pipeline.sample(image=image)
        
        if tikz_doc and tikz_doc.code:
             logger.info("Inference completed successfully.")
             return GenerateResponse(tikz=tikz_doc.code)
        else:
             raise Exception("Failed to generate any valid TikZ code.")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}
