"""Ingest PDF files, generate embeddings, persist in ChromaDB, with logging."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Iterator

import chromadb  # type: ignore
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer  # type: ignore

DEFAULT_PDF_DIR = "./data/news_et_toi/202511"
DEFAULT_PERSIST_DIR = "./vector-data"
CHUNK_SIZE = 2048


def _init_vector_resources(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    model_name: str = "intfloat/multilingual-e5-base",
    collection_name: str = "pdf_chunks",
):
    """Initialize and return (model, collection) for vector storage."""
    if SentenceTransformer is None or chromadb is None:
        raise RuntimeError(
            "sentence_transformers/chromadb not available; install dependencies first."
        )
    os.makedirs(persist_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    return model, collection


def _read_pdfs_from_dir(directory: str) -> Iterator[str]:
    pdf_files = sorted(
        (f for f in os.listdir(directory) if f.lower().endswith(".pdf")),
        key=str.lower,
    )
    for pdf_file in pdf_files:
        yield os.path.join(directory, pdf_file)


def _extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if not page_text:
            continue
        for line in page_text.splitlines():
            stripped = line.strip()
            if (
                stripped.isupper()
                and len(stripped.split()) <= 10
                and any(c.isalpha() for c in stripped)
            ):
                all_text.append(f"# {stripped}")
            elif (
                len(stripped) > 2
                and (
                    sum(1 for c in stripped if c.isupper())
                    / max(1, len(stripped))
                ) > 0.5
                and len(stripped.split()) < 15
            ):
                all_text.append(f"**{stripped}**")
            else:
                all_text.append(stripped)
    return "\n".join(all_text)


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = 256,
) -> Iterator[str]:
    step = chunk_size - overlap
    i = 0
    while i < len(text):
        yield text[i : i + chunk_size]
        i += step


def _get_dir_size_mb(directory: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 * 1024)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time: use minutes if >= 60s else seconds.

    Examples:
        12.345  -> '12.35s'
        75.0    -> '1.25m'
    """
    if seconds >= 60:
        return f"{seconds/60:.2f}m"
    return f"{seconds:.2f}s"


def _ingest_pdfs(
    pdf_dir: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = 256,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or logging.getLogger("pdf_ingest")
    total_start = time.perf_counter()
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    log.info("PDF directory: %s", pdf_dir)
    log.info("Persist directory: %s", persist_dir)
    log.info("Chunk size: %d (overlap %d)", chunk_size, overlap)

    model, collection = _init_vector_resources(persist_dir=persist_dir)
    chunk_counter = 0
    for pdf_path in _read_pdfs_from_dir(pdf_dir):
        file_start = time.perf_counter()
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        log.info("Reading %s (%.2f MB)", pdf_path, pdf_size_mb)
        text = _extract_text_from_pdf(pdf_path)
        log.debug("Extracted %d chars", len(text))
        for idx, chunk in enumerate(_chunk_text(text, chunk_size, overlap=overlap)):
            embedding = model.encode(chunk).tolist()
            chunk_id = f"chunk_{chunk_counter}"
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": pdf_path, "chunk_index": idx}],
            )
            if idx % 10 == 0:
                log.debug(
                    "Stored chunk %s (chunk_index=%d, size=%d chars)",
                    chunk_id,
                    idx,
                    len(chunk),
                )
            chunk_counter += 1
        file_elapsed = time.perf_counter() - file_start
        total_elapsed = time.perf_counter() - total_start
        log.info(
            "Processed %s in %.2fs (total elapsed %s)",
            pdf_path,
            file_elapsed,
            _fmt_elapsed(total_elapsed),
        )
    log.info("Total chunks stored: %d", chunk_counter)
    log.info(
        "Persist directory size: %.2f MB", _get_dir_size_mb(persist_dir)
    )
    overall = time.perf_counter() - total_start
    log.info("Overall ingestion time: %s", _fmt_elapsed(overall))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="PDF directory")
    parser.add_argument(
        "--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist dir"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size (chars)"
    )
    parser.add_argument(
        "--overlap", type=int, default=256, help="Overlap between chunks"
    )
    parser.add_argument(
        "--log-file", default="pdf_ingest.log", help="Log file path"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-base",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--collection",
        default="pdf_chunks",
        help="Chroma collection name",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for PDF ingestion script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("pdf_ingest")
    logger.info("Starting ingestion")
    try:
        _ingest_pdfs(
            pdf_dir=args.pdf_dir,
            chunk_size=max(128, args.chunk_size),
            overlap=max(0, min(args.overlap, args.chunk_size // 2)),
            persist_dir=args.persist_dir,
            logger=logger,
        )
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Error: %s", e)
        return 2
    logger.info("Completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
