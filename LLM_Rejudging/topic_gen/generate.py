import base64
import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, load_prompt
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, ValidationError
from tqdm.contrib.concurrent import thread_map  # type: ignore[import-untyped]

from topic_gen import logger

load_dotenv()
PROMPT_DIR = resources.files("topic_gen.prompts")
PROMPTS = [Path(str(f)).stem for f in PROMPT_DIR.iterdir()]


def _encode_pdf_to_base64(file_path: str) -> Optional[str]:
    """Encodes a PDF file to a base64 string."""
    if not file_path.lower().endswith(".pdf"):
        logger.warning(f"Skipping unsupported file type: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None


class Generator:
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: Union[str, Path],
        output_class: Optional[type[BaseModel]] = None,
        config: Optional[RunnableConfig] = None,
        parse: bool = True,
        use_structured_output: bool = True,
    ):
        self.output_class = output_class
        self.config = config
        self.parse = parse
        self.use_structured_output = use_structured_output
        self.prompt_name = str(prompt)

        self.parser: Optional[PydanticOutputParser] = None
        self.chain = None

        # Check if structured output is requested and supported
        if self.output_class and parse and self.use_structured_output:
            if hasattr(llm, "with_structured_output"):
                try:
                    structured_llm = llm.with_structured_output(self.output_class)
                    self.chain = RunnableLambda(self._prepare_input_for_llm) | structured_llm
                except NotImplementedError:
                    logger.warning("LLM does not implement structured output. Falling back to PydanticOutputParser.")
                    self.use_structured_output = False
            else:
                logger.warning("LLM does not support structured output. Falling back to PydanticOutputParser.")
                self.use_structured_output = False

        # Fallback to string parsing if structured output is not used
        if self.chain is None:
            self.chain = (
                RunnableLambda(self._prepare_input_for_llm) | llm | StrOutputParser()
            )
            if self.output_class:
                self.parser = PydanticOutputParser(pydantic_object=self.output_class)
                if parse:
                    self.chain = self.chain | self.parser

        self.prompt_template = self._load_prompt(prompt)

    def _load_prompt(self, prompt: Union[str, Path]) -> BasePromptTemplate:
        """Loads and partially formats the prompt template."""
        logger.info(f"Prompt type: {type(prompt)}")
        prompt_path: Any = prompt
        # custom prompt path
        if os.path.exists(str(prompt)):
            logger.info(f"Generating using custom prompt from '{prompt}'")
        # default prompt
        elif str(prompt) in PROMPTS:
            prompt_path = PROMPT_DIR / f"{prompt}.yaml"
        else:
            raise FileNotFoundError(f"Prompt file '{prompt}' does not exist.")

        prompt_template = load_prompt(prompt_path)

        if self.output_class and self.parser and not self.use_structured_output:
            prompt_template = prompt_template.partial(
                format_instructions=self.parser.get_format_instructions()
            )
        elif "format_instructions" in prompt_template.input_variables:
            if self.use_structured_output:
                # No need for textual instructions when using structured output
                prompt_template = prompt_template.partial(format_instructions="")
            else:
                logger.warning(
                    "Prompt contains '{format_instructions}' but no output_class is defined. Removing it from the prompt or provide an output_class."
                )
                prompt_template = prompt_template.partial(format_instructions="")

        return prompt_template

    def _prepare_input_for_llm(self, kwargs: Dict[str, Any]) -> list[HumanMessage]:
        """Generate the prompt"""
        final_text_prompt = self.prompt_template.format(**kwargs)
        content: list[str | dict[Any, Any]] = [
            {"type": "text", "text": final_text_prompt}
        ]
        logger.info(f"Final text prompt:\n {final_text_prompt}")

        for key, value in kwargs.items():
            if (
                isinstance(value, str)
                and os.path.isdir(value)
                and os.path.exists(value)
            ):
                pdf_directory = value
                logger.info(f"Loading PDFs from '{pdf_directory}' for key '{key}'")
                for file_name in os.listdir(pdf_directory):
                    pdf_bytes = _encode_pdf_to_base64(
                        os.path.join(pdf_directory, file_name)
                    )
                    if pdf_bytes:
                        logger.info(f"Adding PDF content for '{file_name}'")
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_bytes}"
                                },
                            }
                        )

        return [HumanMessage(content=content)]

    def generate_one(
        self, item_id: Optional[str] = None, dry_run: Optional[bool] = False, **kwargs
    ):
        """
        Generate a single item.

        Args:
            item_id (str, optional): An identifier for the item being generated. This will be added to the output object if provided.
            dry_run (bool): If True, only logs the prompt without generating topics.
            **kwargs: Variables to be passed into the prompt template.
        """
        if dry_run:
            prompt_for_logging = self._prepare_input_for_llm(kwargs)
            logger.info(f"Dry run: Final prompt object:\n{prompt_for_logging}")
            return None

        try:
            result = self.chain.invoke(kwargs, config=self.config)

            # Add item id to result if applicable
            if self.output_class and isinstance(result, self.output_class) and item_id:
                result._id = item_id  # type: ignore[attr-defined]

            return result

        except ValidationError as e:
            logger.error(f"Output validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"An error occurred while running the chain: {e}")
            return None

    def _safe_invoke(self, input_data: Dict[str, Any]) -> Any:
        try:
            return self.chain.invoke(input_data, config=self.config)
        except Exception as e:
            return e

    def generate(
        self, item_ids: Optional[List[str]] = None, dry_run: bool = False, **kwargs
    ):
        # assert that the kwargs are lists
        for key, value in kwargs.items():
            if not isinstance(value, list):
                raise ValueError(f"Expected a list for '{key}', got {type(value)}")

        arg_names = kwargs.keys()
        value_lists = kwargs.values()

        batch_inputs: List[Dict[str, Any]] = [
            dict(zip(arg_names, item_group)) for item_group in zip(*value_lists)
        ]

        if dry_run:
            logger.info(
                f"Dry run: Preparing to generate for {len(batch_inputs)} items."
            )
            return None

        logger.info(f"Generating batch of {len(batch_inputs)} items...")
        try:
            # generate with progress bar
            max_workers = self.config.get("max_concurrency") if self.config else None
            results = thread_map(
                self._safe_invoke,
                batch_inputs,
                max_workers=max_workers,
                desc=f"Generating Topics ({self.prompt_name})",
            )

            # add item ids if applicable
            if item_ids and self.output_class:
                ret: list[Any] = []
                for item, _id in zip(results, item_ids):
                    if not isinstance(item, self.output_class):
                        continue
                    item._id = _id  # type: ignore[attr-defined]
                    ret.append(item)
                logger.info(f"Generated {len(ret)} items successfully.")
                return ret
            return results
        except Exception as e:
            logger.error(f"An error occurred during batch processing: {e}")
            return None
