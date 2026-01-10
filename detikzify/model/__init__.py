# from datasets import DownloadManager
from safetensors.torch import load_file
from transformers.utils.hub import has_file
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizerFast,
    is_timm_available,
)
from transformers.utils.hub import is_remote_url

from .configuration_detikzify import *
from .modeling_detikzify import *
from .processing_detikzify import *
from .adapter import load as load_adapter

if is_timm_available():
    from .v1 import models as v1_models, load as load_v1

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        AutoModelForVision2Seq.register(DetikzifyConfig, DetikzifyForConditionalGeneration)
        AutoProcessor.register(DetikzifyConfig, DetikzifyProcessor)
        
        from transformers import LlamaTokenizer, LlamaTokenizerFast
        AutoTokenizer.register(DetikzifyConfig, slow_tokenizer_class=LlamaTokenizer, fast_tokenizer_class=LlamaTokenizerFast)
    except ValueError:
        pass # already registered

def load(model_name_or_path, modality_projector=None, is_v1=False, **kwargs):
    # backwards compatibility with v1 models
    if is_timm_available() and (is_v1 or model_name_or_path in v1_models): # type: ignore
        model, tokenizer, image_processor = load_v1( # type: ignore
            model_name_or_path=model_name_or_path,
            modality_projector=modality_projector,
            **kwargs
        )
        return model, DetikzifyProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_len=model.config.num_patches,
            image_token=tokenizer.convert_ids_to_tokens(model.config.patch_token_id)
        )

    register()
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
    except OSError:
        # Fallback for models missing preprocessor_config.json (like nllg/detikzify-ds-1.3b)
        from transformers import AutoTokenizer, SiglipImageProcessor
        
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        # DeTikZify uses 420x420 images
        image_processor.size = {"height": 420, "width": 420}
        
        # Use LlamaTokenizerFast directly to avoid AutoTokenizer mapping issues with custom config
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name_or_path)
        processor = DetikzifyProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)

    if modality_projector is not None:
        if is_remote_url(modality_projector):
            # modality_projector = DownloadManager().download(modality_projector)
            raise NotImplementedError("Remote modality_projector download not supported in lean mode")
        model.load_state_dict(
            state_dict=load_file(
                filename=modality_projector, # type: ignore
                device=str(model.device)
            ),
            strict=False
        )

    if has_file(model_name_or_path, "adapter/model.safetensors"):
        model, processor = load_adapter(model=model, processor=processor)

    return model, processor
