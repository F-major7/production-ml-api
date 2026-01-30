"""
Sentiment Analysis Model Wrapper
Supports multiple model versions for A/B testing

NOTE: Docker-compatible version with single-threaded PyTorch.
"""
from typing import Dict, Optional
from transformers import pipeline
import logging
import torch
import gc
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# CRITICAL: Set PyTorch to single-threaded BEFORE any torch operations
# This must happen before pipeline/model loading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Disable all backends that might cause Docker issues
if hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn.enabled = False
if hasattr(torch.backends, 'mkl'):
    torch.backends.mkl.is_available = lambda: False
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.is_available = lambda: False

# Also set via environment (backup for subprocess/forks)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running inference (single worker only)
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torch_infer")

# Lock to ensure only one inference at a time (prevent concurrent model access)
_inference_lock = threading.Lock()


class SentimentModel:
    """
    Sentiment analysis model using HuggingFace Transformers.
    Supports multiple versions for A/B testing.
    Uses DistilBERT fine-tuned on SST-2 for sentiment classification.
    """
    _instances: Dict[str, 'SentimentModel'] = {}
    
    def __init__(self, version: str = "v1"):
        """
        Initialize model for specific version.
        
        Args:
            version: Model version identifier (v1, v2, etc.)
        """
        self.version = version
        self._pipeline = None
        self._initialize_model()
    
    @classmethod
    def get_model(cls, version: str = "v1") -> 'SentimentModel':
        """
        Get or create model instance for specified version.
        Implements singleton pattern per version.
        
        Args:
            version: Model version identifier
            
        Returns:
            SentimentModel instance for the version
        """
        if version not in cls._instances:
            logger.info(f"Creating new model instance for version: {version}")
            cls._instances[version] = cls(version)
        return cls._instances[version]
    
    def _initialize_model(self) -> None:
        """
        Load the sentiment analysis model.
        For Phase 4, both v1 and v2 use same DistilBERT (infrastructure focus).
        In production, these would be different models or configurations.
        """
        if self._pipeline is None:
            try:
                logger.info(f"Loading sentiment analysis model (version: {self.version})...")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch.nn.functional as F
                
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                
                # Load tokenizer with use_fast=False to avoid Rust tokenizer segfaults
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torchscript=False  # Disable torchscript
                )
                self._model.cpu()  # Explicitly move to CPU
                self._model.eval()  # Set to evaluation mode
                
                # Disable gradient computation globally for this model
                for param in self._model.parameters():
                    param.requires_grad = False
                
                # Store a dummy pipeline reference to maintain is_loaded compatibility
                self._pipeline = True  # Just a flag, we use _model and _tokenizer directly
                
                logger.info(f"Model {self.version} loaded successfully (direct mode)")
            except Exception as e:
                logger.error(f"Failed to load model {self.version}: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")
    
    def _run_inference(self, text: str) -> Dict[str, any]:
        """
        Internal method to run inference synchronously.
        Called from thread pool to avoid async/PyTorch conflicts.
        Uses lock to ensure only one inference at a time.
        Bypasses pipeline and runs model directly for Docker stability.
        """
        import torch.nn.functional as F
        
        logger.info(f"[INFERENCE] Waiting for lock, text length: {len(text)}")
        
        with _inference_lock:
            logger.info("[INFERENCE] Lock acquired, starting inference")
            try:
                # Tokenize
                logger.info("[INFERENCE] Tokenizing...")
                inputs = self._tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                logger.info(f"[INFERENCE] Tokenized: {inputs.keys()}")
                
                # Move inputs to CPU explicitly
                inputs = {k: v.cpu() for k, v in inputs.items()}
                logger.info(f"[INFERENCE] Inputs on device: {inputs['input_ids'].device}")
                
                # Run model with inference_mode (more efficient than no_grad)
                logger.info("[INFERENCE] Running model forward pass...")
                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    logits = outputs.logits.cpu()  # Ensure output is on CPU
                    logger.info(f"[INFERENCE] Model output logits shape: {logits.shape}")
                
                # Get prediction
                logger.info("[INFERENCE] Computing probabilities...")
                probs = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_class].item()
                
                # Map class to label (0=NEGATIVE, 1=POSITIVE for this model)
                label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
                
                gc.collect()
                
                output = {
                    "label": label,
                    "score": confidence
                }
                logger.info(f"[INFERENCE] Success: {output}")
                return output
            except Exception as e:
                logger.error(f"[INFERENCE] Exception: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"[INFERENCE] Traceback: {traceback.format_exc()}")
                raise
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for given text (synchronous version).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If text is invalid
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            return self._run_inference(text)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    async def predict_async(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for given text (async-safe version).
        Runs inference in thread pool to avoid uvloop/PyTorch conflicts.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If text is invalid
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            logger.info(f"Starting async prediction for text: {text[:50]}...")
            # Run inference in thread pool to avoid async event loop conflicts
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                _inference_executor,
                self._run_inference,
                text
            )
            logger.info(f"Prediction complete: {result}")
            return result
        except Exception as e:
            logger.error(f"Async prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self._pipeline is not None

