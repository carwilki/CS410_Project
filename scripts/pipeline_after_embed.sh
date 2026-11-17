#!/bin/zsh
# Waits for the embedding job, then assembles the Faiss index (Pyserini layout). Search/judging happen in the TUI.
set -e
cd "$(dirname "$0")/.."
while pgrep -f "scripts.embed_corpus" >/dev/null; do sleep 60; done
n=$(ls artifacts/embed_work/chunk_*.npy | grep -vc tmp)
echo "$(date) embedding finished: $n chunk files"
[ "$n" -eq 941 ] || { echo "expected 941 chunks, got $n - rerun embed_corpus to resume"; exit 1; }
uv run python -m scripts.embed_corpus --assemble
echo "$(date) index ready"
