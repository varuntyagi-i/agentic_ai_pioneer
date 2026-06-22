"""Configuration shared by all agents."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "AttentionIsAllYouNeed.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
MEMORY_DB = BASE_DIR / "agent_memory.sqlite"
CONFIG_PATH = BASE_DIR / "config.yaml"

# Secrets live beside the assignment code. Existing shell variables take precedence.
load_dotenv(BASE_DIR / ".env")


class OpenAISettings(BaseModel):
    model: str
    embedding_model: str


class TavilySettings(BaseModel):
    search_depth: str = Field(pattern="^(basic|advanced)$")
    max_results: int = Field(gt=0, le=20)


class RAGSettings(BaseModel):
    top_k: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)


class ReflectionSettings(BaseModel):
    max_steps: int = Field(gt=0)


class AppSettings(BaseModel):
    openai: OpenAISettings
    tavily: TavilySettings
    rag: RAGSettings
    reflection: ReflectionSettings


def load_settings() -> AppSettings:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    settings = AppSettings.model_validate(raw_config)
    if settings.rag.chunk_overlap >= settings.rag.chunk_size:
        raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_size")
    return settings


SETTINGS = load_settings()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to the repository .env file or export it in the shell."
        )
    return value


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    require_env("OPENAI_API_KEY")
    return ChatOpenAI(
        model=SETTINGS.openai.model,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=SETTINGS.openai.embedding_model)
