from functools import cached_property
from re import sub
from types import SimpleNamespace as Namespace
from queue import Empty
from typing import Optional, Union, Any, Dict, List, Tuple, Generator
Numeric = Union[int, float]
from math import sqrt
from time import time
from collections import deque
from multiprocessing.pool import ThreadPool

import torch
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteriaList
from PIL import Image

from ..util import (
    ExplicitAbort,
    StreamerList,
    TokenStreamer,
    cache_cast,
    expand,
    load,
    unwrap_processor as unwrap,
)
from .tikz import TikzDocument
from ..mcts.montecarlo import MonteCarlo
from ..mcts.node import Node

# Try importing ImageSim, if it fails, we fall back to fast metric
try:
    from ..evaluate.imagesim import ImageSim
except ImportError:
    ImageSim = None

class WideNode(Node):
    def __init__(self, state):
        super().__init__(state)
        self.is_widen_node = True

class DetikzifyGenerator:
    def __init__(
        self,
        model,
        processor,
        image: Optional[Image.Image],
        text: Optional[str] = None,
        metric: Optional[Any] = None,
        compile_timeout: Optional[int] = 60,
        mcts_timeout: Optional[int] = None,
        streamer: Optional[BaseStreamer] = None,
        control: Optional[ExplicitAbort] = None,
        exploration: float = 0.6, # exploration coefficient
        strict: bool = False, # if True, treat recoverable errors same as fatal errors when computing scores
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        self.metric = metric
        self.image = image
        self.text = text
        self.streamer = streamer
        self.control = control or ExplicitAbort()
        self.gen_kwargs = gen_kwargs

        self.compile_timeout = compile_timeout
        self.mcts_timeout = mcts_timeout
        self.fast_metric = metric is None or getattr(metric, "fast", False)

        # https://stackoverflow.com/a/68550238
        self.decode = cache_cast(lambda token_ids: tuple(token_ids.tolist()))(self.decode)
        
        # MCTS initialization
        self.thread = ThreadPool(processes=1)
        self.failed_rollouts = dict()
        
        # Determine newline_id
        dummy_decode = self.processor.decode([100]) # just check something
        # Ideally we find the newline token from the tokenizer.
        # For Llama, it's usually 13 (<0x0A>). Let's search or assume.
        # Actually in `newlineinfo` below we iterate vocab.
        # We need a newline_id for the rollout streamer logic.
        # Let's find it safely.
        tokenizer = unwrap(self.processor).tokenizer
        self.tokenizer = tokenizer # alias
        
        # Rough heuristic for newline_id
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

        self.montecarlo = MonteCarlo(
            Node(self.tokenizer.decode(
                token_ids=(
                    self.processor(images=self.image, text=self.text, return_tensors="pt")
                    .input_ids.to(model.device).squeeze()
                ),
                skip_special_tokens=True # Verify if we should skip special tokens here? 
                # Original code used: self.tokenizer.text.decode(...)
                # We use processor.decode or tokenizer.decode
            )),
            mins_timeout=None
        )
        self.montecarlo.root_node.token_ids = (
            self.processor(images=self.image, text=self.text, return_tensors="pt")
            .input_ids.to(model.device).squeeze()
        )
        self.montecarlo.child_finder = self.child_finder
        
        # Normalize score function
        self.norm = lambda x: x # Identity by default

    def generate(self, input_ids: torch.Tensor, streamer: Optional[BaseStreamer] = None, **gen_kwargs) -> torch.Tensor:
        streamers = StreamerList(filter(bool, [streamer, self.streamer]))
        # Check input_ids device
        input_ids = input_ids.to(self.model.device)
        
        # Prepare inputs similarly to original generate
        # But here we might be extending an existing sequence
        
        import logging
        from time import time as get_time
        logger = logging.getLogger("detikzify-generate")
        
        try:
            with torch.inference_mode():
                start_gen = get_time()
                logger.info(f"Starting model.generate() with {len(input_ids)} input tokens...")
                
                # We need to handle image inputs again if we are starting fresh, 
                # but valid MCTS rollouts continue from input_ids.
                # However, Detikzify/Llama-Vision needs pixel_values passed if they are not cached.
                # The simplest way is to pass pixel_values again.
                
                inputs = self.processor(images=self.image, text=self.text, return_tensors="pt")
                # We only need pixel_values from this if we are generation
                pixel_values = inputs.pixel_values.to(device=self.model.device, dtype=self.model.dtype)
                
                # Check for EOS
                if input_ids[-1] == self.tokenizer.eos_token_id:
                     streamers.end()
                     return input_ids

                output = self.model.generate(
                    input_ids=input_ids.unsqueeze(0),
                    pixel_values=pixel_values, # Pass image features
                    streamer=streamers,
                    **self.gen_kwargs,
                    **gen_kwargs
                ).squeeze()
                
                elapsed = get_time() - start_gen
                logger.info(f"model.generate() completed in {elapsed:.1f}s, output {len(output)} tokens")
                return output
        except Exception:
            # traceback.print_exc()
            raise

    def rollout(self, input_ids: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        rollout_control, streamer = ExplicitAbort(), TokenStreamer()
        async_result = self.thread.apply_async(
            func=self.generate,
            args=[input_ids],
            kwds=dict(
                stopping_criteria=StoppingCriteriaList([rollout_control]),
                streamer=streamer,
            )
        )

        try:
            prev, line = input_ids, list()
            for token in streamer:
                line.append(token)
                if token == self.newline_id:
                    prev = torch.cat((prev, torch.tensor(line, device=prev.device)))
                    line.clear()
                    yield prev
            # Yield remaining
            if line:
                 prev = torch.cat((prev, torch.tensor(line, device=prev.device)))
                 yield prev
                 
        except GeneratorExit:
            rollout_control.abort()
            async_result.wait()

    def decode(self, token_ids: torch.Tensor) -> TikzDocument:
        return TikzDocument(
            timeout=self.compile_timeout,
            code=self.processor.decode(
                token_ids=token_ids,
                skip_special_tokens=True
            )
        )

    def score(self, image: Image.Image) -> Numeric:
        if self.metric:
            self.metric.update(img1=image, img2=self.image)
            score = self.metric.compute()
            self.metric.reset()
            return score
        return 0 # Should not happen if fast_metric is checked

    def sample(self):
        # Prepare input_ids
        inputs = self.processor(
            images=self.image,
            text=self.text,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.model.device).squeeze()
        
        # generating
        output_ids = self.generate(input_ids=input_ids)
        
        # Output includes input_ids, so we might want to slice off the input if we want pure generation
        # The original code sliced: token_ids[len(root_node):]
        # input_ids length is len(root_node)
        # generated_ids = output_ids[len(input_ids):] # Don't slice for decode usually if start is prompt
        
        return self.decode(output_ids)

    def simulate(self, expansions: Optional[Numeric] = 1) -> Generator[TikzDocument, None, None]:
        """
        Run the simulations. Returns all rollouts (successful or unsuccessful)
        in descending order (best rollouts first) of their score.
        """
        import logging
        logger = logging.getLogger("detikzify-mcts")
        
        start_time = time()
        simulation_count = 0
        logger.info(f"Starting MCTS simulation (expansions={expansions}, timeout={self.mcts_timeout}s)")
        
        while expansions is None or (expansions:=expansions-1) >= 0:
            simulation_count += 1
            logger.info(f"Running simulation #{simulation_count}...")
            
            self.montecarlo.simulate()
            
            try:
                if self.montecarlo.solution:
                    result = self.montecarlo.solution.pop()
                    logger.info(f"Simulation #{simulation_count} yielded candidate with score: {result[0]}")
                    yield result
            except IndexError:
                logger.debug(f"Simulation #{simulation_count} produced no new solution")
                pass
            
            elapsed = time() - start_time
            if self.mcts_timeout is not None and elapsed > self.mcts_timeout:
                logger.info(f"Timeout reached after {elapsed:.1f}s and {simulation_count} simulations")
                return
        
        logger.info(f"Completed {simulation_count} simulations")

    def child_finder(self, node: WideNode, montecarlo: MonteCarlo):
        new_nodes = list()
        # Ensure we are on key device
        if not isinstance(node.token_ids, torch.Tensor):
             # Should be tensor
             pass
             
        for new_state in (rollout:=self.rollout(node.token_ids)):
            # convert state (tokens) to string/hashable for state key? 
            # In original: state was the string representation.
            # Here Node init took string.
            # But wide node needs token_ids.
            
            # Let's simplify: 
            # We assume node.token_ids is available.
            
            # Hack: WideNode init with token_ids
            new_node = WideNode(self.decode(new_state).code) # State is string
            new_node.token_ids = new_state
            
            if new_node.state in self.failed_rollouts:
                new_nodes.extend(self.failed_rollouts[new_node.state])
                rollout.close()
                break
            new_nodes.append(new_node)

        if node.is_widen_node:
            node.visits += 1
            node, new_nodes = self.merge(node.parent, new_nodes) # type: ignore

        tikz = self.decode(new_nodes[-1].token_ids)
        skip_idx = round(sqrt(len(new_nodes)))

        if tikz.has_content:
            for new_node in new_nodes[:skip_idx]:
                node.add_child(node:=new_node)
        else:
            if tikz.errors:
                 error_idx = max(min(tikz.errors), 1) - 1 - getattr(node, 'depth', 0)
                 # Safe index
                 error_idx = max(0, min(error_idx, len(new_nodes)-1))
                 for new_node in new_nodes[:min(error_idx, skip_idx)]:
                    node.add_child(node:=new_node)
                 self.failed_rollouts[new_nodes[error_idx].state] = new_nodes[error_idx:]

        if self.fast_metric:
            # 1 if has content effectively, -1 if error
            score = int(tikz.has_content) - int(tikz.compiled_with_errors)
        else:
            # Using SelfSim or other metric
            if tikz.has_content:
                try:
                    import logging
                    logger = logging.getLogger("detikzify-mcts")
                    logger.info("Attempting to rasterize TikZ for SelfSim scoring...")
                    rasterized = tikz.rasterize()
                    if rasterized is not None:
                        score = self.score(rasterized)
                        logger.info(f"SelfSim score: {score:.3f}")
                    else:
                        logger.warning("Rasterization returned None, using fast_metric fallback")
                        score = 1  # Compiled successfully, give it a positive score
                except Exception as e:
                    import logging
                    logger = logging.getLogger("detikzify-mcts")
                    logger.warning(f"Rasterization/scoring failed: {e}. Using fallback score.")
                    score = 1  # Compiled successfully, give it a positive score
            else:
                score = -1

        # node.update_win_value(self.norm(score) if tikz.has_content and not self.fast_metric else score)
        # Simplified scoring update
        node.update_win_value(score)
        
        if self.montecarlo.solution is None:
             self.montecarlo.solution = []
        self.montecarlo.solution.append((score, tikz))

    def merge(self, node: WideNode, nodes_to_merge: List[WideNode]) -> Tuple[WideNode, List[WideNode]]:
        for merge_node in nodes_to_merge:
            for child in node.children:
                if child.state == merge_node.state:
                    node, nodes_to_merge = child, nodes_to_merge[1:]
                    break
            else:
                break
        return node, nodes_to_merge


class DetikzifyPipeline:
    def __init__(
        self,
        model,
        processor,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        compile_timeout: int = 60,
        fast_metric: bool = True, # Default to fast (no extra deps)
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        
        metric = None
        if not fast_metric:
            # Try to import and initialize SelfSim
            try:
                from ..evaluate.imagesim import SelfSim
                metric = SelfSim(model=model, processor=processor)
            except ImportError:
                import logging
                logging.warning("SelfSim not available, falling back to fast_metric")
                metric = None
             
        self.gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=unwrap(processor).tokenizer.model_max_length,
            do_sample=True,
            compile_timeout=compile_timeout,
            metric=metric, # None means fast_metric=True mostly
            **gen_kwargs
        )

    def load(self, image: Union[Image.Image, str], preprocess: bool = True):
        image = load(image)
        if preprocess:
            return expand(image, max(image.size), do_trim=True) # trim vs do_trim
        return image

    def sample(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        text: Optional[str] = None,
        preprocess: bool = True,
        **gen_kwargs,
    ) -> TikzDocument:
        
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            image=self.load(image, preprocess=preprocess) if image is not None else None,
            text=text,
            **self.gen_kwargs,
            **gen_kwargs
        )

        return generator.sample()

    def simulate(
        self,
        image: Union[Image.Image, str],
        preprocess: bool = True,
        expansions: Optional[Numeric] = None,
        timeout: Optional[int] = None,
        **gen_kwargs,
    ) -> Generator[TikzDocument, None, None]:
    
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            mcts_timeout=timeout or None,
            image=self.load(image, preprocess=preprocess) if image is not None else None,
            **self.gen_kwargs,
            **gen_kwargs
        )

        yield from generator.simulate(expansions or None)

    def __call__(self, *args, **kwargs) -> TikzDocument:
        return self.sample(*args, **kwargs)
