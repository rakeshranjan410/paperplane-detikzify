"""
SelfSim: Self-Assessed Perceptual Similarity Metric

Based on the DeTikZify whitepaper:
"SelfSim computes the reward as the perceptual similarity between the input image 
and the compiled output figure. We encode both images into embedding vectors using 
DeTikZify's vision encoder and calculate SelfSim as their cosine similarity."
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Any


class SelfSim:
    """
    Self-Assessed Perceptual Similarity metric.
    Uses the model's own vision encoder to compute similarity between images.
    """
    
    fast = False  # This is NOT a fast metric (requires compilation + rendering)
    
    def __init__(self, model, processor):
        """
        Initialize SelfSim with the DeTikZify model and processor.
        
        Args:
            model: The DeTikZify model (with vision encoder)
            processor: The processor for image preprocessing
        """
        self.model = model
        self.processor = processor
        self.img1_embedding = None
        self.img2_embedding = None
        
    def _get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get vision encoder embedding for an image."""
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(
            device=self.model.device, 
            dtype=self.model.dtype
        )
        
        # Get vision encoder output
        with torch.inference_mode():
            # Access the vision encoder from the model
            # DeTikZify models typically have vision_model or vision_tower
            if hasattr(self.model, 'vision_model'):
                vision_outputs = self.model.vision_model(pixel_values)
                # Use pooled output or mean of last hidden state
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    embedding = vision_outputs.pooler_output
                else:
                    embedding = vision_outputs.last_hidden_state.mean(dim=1)
            elif hasattr(self.model, 'vision_tower'):
                vision_outputs = self.model.vision_tower(pixel_values)
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    embedding = vision_outputs.pooler_output
                else:
                    embedding = vision_outputs.last_hidden_state.mean(dim=1)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                vision_outputs = self.model.model.vision_model(pixel_values)
                if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                    embedding = vision_outputs.pooler_output
                else:
                    embedding = vision_outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback: try to use get_image_features if available
                raise AttributeError(
                    f"Could not find vision encoder in model. "
                    f"Available attributes: {dir(self.model)}"
                )
        
        return embedding.squeeze()
    
    def update(self, img1: Optional[Image.Image] = None, img2: Optional[Image.Image] = None):
        """
        Update the metric with new images.
        
        Args:
            img1: First image (typically the rendered TikZ output)
            img2: Second image (typically the input/reference image)
        """
        if img1 is not None:
            self.img1_embedding = self._get_embedding(img1)
        if img2 is not None:
            self.img2_embedding = self._get_embedding(img2)
    
    def compute(self) -> float:
        """
        Compute cosine similarity between the two image embeddings.
        
        Returns:
            Similarity score in range [-1, 1]
        """
        if self.img1_embedding is None or self.img2_embedding is None:
            raise ValueError("Both images must be set before computing similarity")
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            self.img1_embedding.unsqueeze(0),
            self.img2_embedding.unsqueeze(0)
        )
        
        return similarity.item()
    
    def reset(self):
        """Reset the stored embeddings."""
        self.img1_embedding = None
        self.img2_embedding = None
