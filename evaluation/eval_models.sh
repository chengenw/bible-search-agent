#!/usr/bin/env bash
# eval_models.sh — Evaluate several models in sequence, then print the leaderboard.
#
#   ./evaluation/eval_models.sh                       # default MODELS below, full CSV
#   ./evaluation/eval_models.sh --limit 12 --runs 1   # any run_eval flags pass through
#   ./evaluation/eval_models.sh --csv data.csv --no-rag   # no-retrieval baseline
#   MODELS="openrouter/a openrouter/b" ./evaluation/eval_models.sh
#
# Any run_eval flag (--csv, --no-rag, --runs, --reeval, …) forwards to every
# model. Run once normally and once with --no-rag to populate the leaderboard's
# RAG-vs-baseline comparison.
#
# Edit the MODELS list below to change the roster. Questions already answered in
# evaluation/results/ are reused, not re-sent to the LLM — so re-running this
# after adding CSV questions only pays for the new ones (--reeval forces fresh).
# One model failing does not stop the rest.
set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
[ -x "$PY" ] || PY=python

MODELS="${MODELS:-
openrouter/deepseek/deepseek-v4-flash
openrouter/minimax/minimax-m3
openrouter/qwen/qwen3-235b-a22b-2507
openai/gpt-oss-20b
}"
# dead on OpenRouter (400 on every call), kept out of the default roster:
#   openrouter/qwen/qwen3.7-plus

# Ctrl-C: the running eval saves + reports, remaining models are skipped, and
# the leaderboard below STILL prints. (Without this trap, bash would kill the
# whole script on interrupt and the final leaderboard would never appear.)
STOP=0
trap 'STOP=1; echo; echo "⏸   interrupt — skipping remaining models"' INT

failed=""
for m in $MODELS; do
  [ "$STOP" -eq 1 ] && break
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "  MODEL: $m"
  echo "════════════════════════════════════════════════════════════"
  MODEL="$m" "$PY" -m evaluation.run_eval "$@" || failed="$failed $m"
done

# Scope the final leaderboard to THIS run only — the roster's models, the
# dataset evaluated, AND the translation — so leftover results for other
# datasets/versions never leak in. Defaults mirror run_eval (verses.csv, KJV),
# so the scoping holds even when --csv/--version are omitted. Bare model names
# mirror run_eval's MODEL_NAME: strip the provider path and OpenRouter routing
# suffixes (:free etc.) to match the names recorded in the result files.
names=""
for m in $MODELS; do
  n="${m##*/}"
  n="$(printf '%s' "$n" | sed -E 's/:(free|beta|extended|nitro|floor|online)$//')"
  names="$names,$n"
done

csv="verses.csv"; version="KJV"; prev=""
for a in "$@"; do
  case "$prev" in
    --csv)     csv="$a";;
    --version) version="$a";;
  esac
  case "$a" in
    --csv=*)     csv="${a#--csv=}";;
    --version=*) version="${a#--version=}";;
  esac
  prev="$a"
done
ds="${csv##*/}"; ds="${ds%.*}"

echo
"$PY" -m evaluation.leaderboard --models "${names#,}" --version "$version" --dataset "$ds"

if [ -n "$failed" ]; then
  echo
  echo "⚠️  failed:$failed"
  exit 1
fi
