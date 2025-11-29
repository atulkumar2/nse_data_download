"""Interactive Gemini agent answering questions over ingested PDF documents.

Refactored: no mutable module-level globals; uses logging instead of print.
"""

import argparse
import asyncio
import logging
import os
import traceback
from pathlib import Path

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from test_search_query import (
    init_search_resources,
    semantic_search,
)

# Constants
USER_ID = "default_user"
SESSION_NAME = "main_conversation"
DEFAULT_PERSIST_DIR = "./vector-data"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external deps).

    Reads KEY=VALUE pairs, ignores comments and blank lines.
    Does not overwrite existing environment variables.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        (
            "GOOGLE_API_KEY not set. Create a .env file with\n"
            "GOOGLE_API_KEY=your_key or export it in the shell."
        )
    )

logger = logging.getLogger("gemini_agent")


def _configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.debug("Logging configured (level=%s)", level.upper())

# Configuration
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

def setup_agent(
    persist_dirs: list[str] | None = None,
) -> tuple[Agent, InMemoryRunner, InMemorySessionService, str]:
    """Load vector collections (single or multiple) and construct runtime objects.

    If multiple directories are provided they are joined with commas and passed
    via PERSIST_DIR for downstream lazy initialization in search utilities.
    """
    if not persist_dirs:
        persist_dirs = [DEFAULT_PERSIST_DIR]
    display = ",".join(persist_dirs)
    logger.info("Loading ChromaDB collections (dirs=%s)...", display)
    combined = ",".join(persist_dirs)
    os.environ["PERSIST_DIR"] = combined
    init_search_resources(persist_dir=combined)
    logger.info("Vector collections ready (count=%d).", len(persist_dirs))

    agent = Agent(
        name="helpful_assistant",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config,
        ),
        description=(
            "A helpful assistant that can answer questions by searching through"
            " ingested PDF documents."
        ),
        instruction=(
            "You are a helpful assistant with access to a semantic_search tool:\n"
            "1. **semantic_search**: Use this to search through ingested"
            " PDF documents.\n"
            "This tool searches documents stored in a vector database. Use it"
            " when the user asks about information that might be in those PDFs.\n\n"
            "When answering questions:\n"
            "- Use semantic_search for information related to the ingested PDFs\n"
            "- If you cannot find relevant information, explain your knowledge is"
            "   limited to the ingested documents\n"
            "- Provide a citation of the document name used"
        ),
        tools=[semantic_search],
    )
    run = InMemoryRunner(agent=agent)
    # Align runner app name with root agent package to avoid ADK warning
    if hasattr(run, "app_name"):
        run.app_name = "agents"
        logger.debug("Runner app_name set to '%s'", run.app_name)
    session_service = InMemorySessionService()
    if hasattr(run, 'session_service'):
        run.session_service = session_service
    model_name = agent.name
    logger.info("Agent initialized and ready.")
    return agent, run, session_service, model_name


async def create_or_get_session(
    runner: InMemoryRunner,
    session_service: InMemorySessionService,
    root_agent: Agent,
    session_name: str = SESSION_NAME,
):
    """
    Create or retrieve a session for maintaining conversation context.

    Args:
        session_name: Name identifier for the session

    Returns:
        Session object

    Raises:
        RuntimeError: If session creation/retrieval fails
    """
    # Get app name from the Runner (fallback to agent name if not available)
    try:
        app_name = runner.app_name
    except AttributeError:
        app_name = root_agent.name

    # Always try to create a new session first (it will handle if it already exists)
    # If creation fails because it exists, then get it
    session = None
    try:
        # Try to create a new session
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
        if session is None:
            raise ValueError("Session creation returned None")
        logger.info("Created new session: %s", session_name)
    except Exception as create_error:
        # If creation fails (e.g., session already exists), try to get it
        try:
            session = await session_service.get_session(
                app_name=app_name, user_id=USER_ID, session_id=session_name
            )
            if session is None:
                raise ValueError("Session retrieval returned None") from create_error
            logger.info("Retrieved existing session: %s", session_name)
        except Exception as get_error:
            # If both fail, raise an error
            raise RuntimeError(
                (
                    "Failed to create or retrieve session '"
                    f"{session_name}'\nCreate error: {create_error}\n"
                    f"Get error: {get_error}"
                )
            ) from get_error

    # Validate session has required attributes
    if session is None:
        raise RuntimeError("Session is None after creation/retrieval")

    # Check for 'id' attribute (could vary by implementation)
    session_id = None
    if hasattr(session, 'id'):
        session_id = session.id
    # Fallback attribute names if present (ignore type check complaints)
    elif hasattr(session, 'session_id'):  # type: ignore[attr-defined]
        session_id = getattr(session, 'session_id')  # type: ignore[attr-defined]
    elif hasattr(session, 'sessionId'):  # type: ignore[attr-defined]
        session_id = getattr(session, 'sessionId')  # type: ignore[attr-defined]

    if session_id is None:
        # If no ID found, try to use the session_name as the ID
        logger.warning(
            "Session object has no 'id' attribute, using session_name as ID"
        )
        session_id = session_name
        # Try to set it if possible
        if hasattr(session, 'id'):
            session.id = session_id

    return session


async def run_query(
    runner_instance: InMemoryRunner,
    query: str,
    session_name: str,
    model_name: str,
):
    """
    Run a single query in the given session.

    Args:
        runner_instance: The InMemoryRunner instance
        query: Query string
        session_name: Session name/ID for maintaining context
    """
    logger.info("User query: %s", query)

    # Convert the query string to the ADK Content format
    query_content = types.Content(role="user", parts=[types.Part(text=query)])

    # Stream the agent's response asynchronously
    # Use session_name directly as the session_id
    async for event in runner_instance.run_async(
        user_id=USER_ID, session_id=session_name, new_message=query_content
    ):
        # Only print text parts; do not emit warnings for non-text parts.
        if event.content and event.content.parts:
            # Extract all text parts, ignore other kinds
            texts = [
                part.text for part in event.content.parts
                if hasattr(part, "text") and part.text
            ]
            for text in texts:
                if text != "None":
                    logger.info("%s > %s", model_name, text)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gemini agent over persisted Chroma collection",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="Single Chroma persistent directory (default: %(default)s)",
    )
    parser.add_argument(
        "--persist-dirs",
        nargs="+",
        help=(
            "Multiple persistent directories (space separated). "
            "Overrides --persist-dir."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: %(default)s)",
    )
    return parser


async def main(argv: list[str] | None = None):
    """Main function to run the interactive agent."""
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.log_level)
    # Determine directories precedence: --persist-dirs over --persist-dir
    dirs: list[str]
    if args.persist_dirs:
        dirs = args.persist_dirs
    else:
        dirs = [args.persist_dir]
    agent, run, session_service, model_name = setup_agent(persist_dirs=dirs)
    logger.info(
        "Ask questions about the PDF documents. Type 'exit' or 'quit' to quit."
    )

    # Create or get session once at the start to ensure it exists
    try:
        session = await create_or_get_session(
            run, session_service, agent, SESSION_NAME
        )
        if session is None:
            raise RuntimeError("Failed to create or retrieve session")
        logger.info("Session '%s' initialized.", SESSION_NAME)
    except Exception as e:
        logger.warning(
            "Session initialization issue: %s. Will attempt creation on first use.",
            e,
        )

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ("exit", "quit"):
            logger.info("Goodbye!")
            break

        if not query:
            continue

        try:
            # Pass session_name directly instead of session object
            await run_query(run, query, SESSION_NAME, model_name)
        except Exception as e:
            logger.error("Error handling query: %s", e)
            traceback.print_exc()


if __name__ == "__main__":
    import sys as _sys

    try:
        asyncio.run(main(_sys.argv[1:]))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.getLogger("gemini_agent").info("Interrupted by user. Exiting.")
