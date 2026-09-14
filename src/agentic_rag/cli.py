"""Command-line interface.

Three verbs: build the index, ask a question, evaluate. Each is thin; the work
lives in the package so that the CLI, the UI and the tests all drive identical
code paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from agentic_rag import __version__
from agentic_rag.logging_config import configure_logging
from agentic_rag.settings import get_settings

app = typer.Typer(
    name="agentic-rag",
    help="A retrieval-augmented assistant with self-correction under a budget.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Configure logging before any subcommand runs."""
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print the resolved configuration, with secrets masked."""
    settings = get_settings()
    payload = json.loads(settings.model_dump_json())
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(help="File or directory to ingest.")],
    augment: Annotated[
        bool, typer.Option(help="Generate hypothetical questions per chunk.")
    ] = True,
    fixed_size: Annotated[
        bool, typer.Option(help="Use character-window chunking instead of semantic.")
    ] = False,
) -> None:
    """Parse, chunk and index a corpus."""
    from agentic_rag.ingestion.chunker import fixed_size_chunks
    from agentic_rag.ingestion.pipeline import IngestionPipeline
    from agentic_rag.llm.openai_client import OpenAIChatModel

    settings = get_settings()
    pipeline = IngestionPipeline(
        settings=settings,
        chunker=fixed_size_chunks if fixed_size else None,
        model=OpenAIChatModel(settings) if augment else None,
    )
    report = pipeline.run(source, augment=augment)
    typer.echo(report.summary())


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    show_trace: Annotated[
        bool, typer.Option(help="Print judge scores and the node path.")
    ] = False,
) -> None:
    """Answer a single question."""
    from agentic_rag.agent.graph import AgentRunner

    state = AgentRunner().ask(question)
    typer.echo(state.get("response", ""))

    if show_trace:
        typer.echo("", err=True)
        typer.echo(
            json.dumps(
                {
                    "run_id": state.get("run_id"),
                    "groundedness": state.get("groundedness"),
                    "precision": state.get("precision"),
                    "iterations": state.get("iteration", 0),
                    "exhausted": state.get("exhausted", False),
                    "path": state.get("history", []),
                },
                indent=2,
            ),
            err=True,
        )


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(help="JSON Lines evaluation set.")],
    out: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "evaluation/results/report.json"
    ),
    baseline: Annotated[
        Path | None, typer.Option(help="Committed baseline to compare against.")
    ] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] = "0",
    tolerance: Annotated[
        float, typer.Option(help="Allowed drift before a metric counts as regressed.")
    ] = 0.02,
) -> None:
    """Run the evaluation suite and optionally gate on a baseline.

    Scoring, aggregation and the gate come from the shared ``eval-harness``
    package. This command supplies only what is specific to this agent: the
    metrics it is judged on, and how one question becomes an observation.
    """
    from eval_harness import EvalCase, EvaluationRunner, RegressionCheck, load_cases
    from eval_harness.report import load_baseline

    from agentic_rag.agent.graph import AgentRunner
    from agentic_rag.eval_metrics import register_agent_metrics

    metrics = register_agent_metrics()
    runner = AgentRunner()

    def task(case: EvalCase) -> dict[str, object]:
        state = runner.ask(case.input)
        return {
            "question": case.input,
            "response": state.get("response", ""),
            "groundedness": state.get("groundedness"),
            "precision": state.get("precision"),
            "iteration": state.get("iteration", 0),
            "exhausted": state.get("exhausted", False),
        }

    report = EvaluationRunner(
        task,
        metrics=metrics,
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
    ).run(
        # The dataset keeps its documented "question" key rather than being
        # rewritten to suit the harness.
        load_cases(dataset, input_key="question"),
        dataset=dataset,
    )

    report.write(out)
    typer.echo(report.summary())
    typer.echo(f"\nWrote {out}")

    if baseline is not None:
        result = RegressionCheck(tolerance=tolerance).compare(
            report, load_baseline(baseline)
        )
        typer.echo("")
        typer.echo(result.report())
        if not result.passed:
            typer.echo(
                f"\n{len(result.regressions)} metric(s) regressed past "
                f"tolerance {tolerance}.",
                err=True,
            )
            sys.exit(1)
        typer.echo("\nNo regression against baseline.")


if __name__ == "__main__":
    app()
