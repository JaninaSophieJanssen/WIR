import logging
import os
import sys
from datetime import datetime

import click
from langchain.chat_models import init_chat_model

from topic_gen import logger
from topic_gen.generate import Generator
from topic_gen.models import Topics, iSearchTopic

CONTEXT_SETTINGS = dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
)


def parse_kwargs(ctx):
    """Parse additional options from click context and filter out known flags and options.

    Args:
        ctx (_type_): The click context with all arguments passed to the command.

    Returns:
        _type_: Only the additional keyword arguments that are not known options or flags.
    """
    known_options = [
        "model_name",
        "model_provider",
        "prompt",
        "number_of_topics",
        "output",
        "verbose",
        "v",
    ]
    known_flags = ["dry-run", "structured-output", "no-structured-output"]

    kwargs = [option.lstrip("-") for option in ctx.args]
    kwargs = [arg for arg in kwargs if arg not in known_flags]
    kwargs = {k: v for k, v in zip(kwargs[::2], kwargs[1::2]) if k not in known_options}
    return kwargs


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--model_name", default="gemini-2.5-flash")
@click.option("--model_provider", default="google_genai")
@click.option("--prompt", default="isearch-base")
@click.option("--number_of_topics", default=5, type=int)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Run without generating topics"
)
@click.option(
    "--structured-output/--no-structured-output",
    default=True,
    help="Use constraint generation (structured output) instead of textual prompt schemas",
)
@click.option("--output")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
@click.pass_context
def main_cli(
    ctx, model_name, model_provider, prompt, number_of_topics, dry_run, structured_output, output, verbose
):
    if verbose:
        logger.setLevel(logging.INFO)

    prompt_kwargs = parse_kwargs(ctx)

    llm = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        temperature=0,
    )
    generator = Generator(
        llm=llm,
        prompt=prompt,
        output_class=Topics[iSearchTopic],
        use_structured_output=structured_output
    )

    topics = generator.generate_one(
        dry_run=dry_run,
        number_of_topics=number_of_topics,
        **prompt_kwargs,
    )

    if topics:
        # prepare output directory
        if output:
            name = prompt_kwargs["name"].replace(" ", "-")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = f"{output}/{name}"
            os.makedirs(output_dir, exist_ok=True)
            with open(
                os.path.join(output_dir, f"topics-{prompt}-{timestamp}-cmd.txt"), "w"
            ) as f:
                f.write(" ".join(sys.argv) + "\n")

            topics.to_xml(os.path.join(output_dir, f"topics-{prompt}-{timestamp}.xml"))

        logger.info(f"Generated {len(topics.topics)} topics:")
        for topic in topics.topics:
            logger.info(topic.to_xml())


if __name__ == "__main__":
    try:
        main_cli()
    except BrokenPipeError:
        sys.stderr.write("BrokenPipe\n")
    except KeyboardInterrupt:
        sys.stderr.write("KeyboardInterrupt\n")
