try:
    from .color import print_colored
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from source.color import print_colored

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, pipeline
import logging
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlmGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._check_gpu()


    def _check_gpu(self):
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB")
        else:
            self.device = 'cpu'
            logger.warning("No GPU available, using CPU")


    def is_loaded(self):
        return self.model is not None and self.tokenizer is not None

    def load_model(self):
        if self.is_loaded():
            logger.warning("Model is already loaded")
            return

        self._load_model_and_tokenizer()
        logger.info("Model loaded successfully")

    def unload_model(self):
        if not self.is_loaded():
            logger.warning("No model to unload")
            return

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory freed")

    def _load_model_and_tokenizer(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate(self, prompt, **kwargs):
        """Generate text from a prompt using the Hugging Face pipeline.

        Args:
            prompt (str): Input text prompt.
            kwargs: Additional generation parameters.

        Returns:
            str: Generated text.
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded - call load_model() first")

        try:
            generation_params = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "do_sample": True,
                "return_full_text": False,
            }
            generation_params.update(kwargs)

            text_gen_pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False
            )
            outputs = text_gen_pipe(prompt, **generation_params)
            return outputs[0]["generated_text"]

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()

    def get_memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent
    env_path = parent_dir / '.env'

    load_dotenv(env_path)

    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        print_colored("ERROR: MODEL_PATH not set in .env file", 'error')
        sys.exit(1)

    generator = LlmGenerator(model_path)

    try:
        print_colored("\nInitializing the model...", 'title')
        generator.load_model()
        print_colored(f"Initial GPU Memory used: {generator.get_memory_usage():.2f} MB", 'green')

        while True:
            try:
                print_colored("\nEnter your prompt (or 'quit' to exit): ", 'title')
                prompt = input().strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print_colored("\nExiting...", 'title')
                    break

                if not prompt:
                    print_colored("Please enter a valid prompt!", 'error')
                    continue

                print_colored("\nGenerating response...", 'title')
                response = generator.generate(prompt)

                print_colored("\nResponse:", 'green')
                print_colored(response, 'yellow')

                memory_usage = generator.get_memory_usage()
                print_colored(f"\nCurrent GPU Memory: {memory_usage:.2f} MB", 'green')

            except KeyboardInterrupt:
                print_colored("\nInterrupt received, exiting...", 'error')
                break
            except Exception as e:
                print_colored(f"\nError: {str(e)}", 'error')
                logger.error(f"Error processing prompt: {str(e)}")
                continue

    except Exception as e:
        print_colored(f"\nFatal error: {str(e)}", 'error')
        logger.error(f"Fatal error occurred: {str(e)}")

    finally:
        print_colored("\nCleaning up...", 'title')
        generator.unload_model()
        print_colored(f"Final GPU Memory: {generator.get_memory_usage():.2f} MB", 'green')