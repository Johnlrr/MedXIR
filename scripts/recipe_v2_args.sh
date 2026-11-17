# Shared fuse_scores.py arguments of the corpus v2 recipe (0.2984 and later): source it, then add the variant flags.
#   source scripts/recipe_v2_args.sh; uv run python scripts/fuse_scores.py --tag ... $COMMON --doc-prior 0.3 ...
A=artifacts/submission
E() { echo "--extra $A/candidates-$1-v2-k500.npz:$2"; }
COMMON="--corpus v2 --bm25 artifacts/bm25-v2/vi-candidates-k500.npz --w-bm25 0.14 $(E 4b 0.09) $(E octen8b 0.09) \
$(E octen8b-prf 0.12) $(E qwen8b 0.08) $(E qwen8b-prf 0.09) $(E octen8b-question 0.035) $(E octen8b-passage 0.035) \
$(E qwen8b-question 0.025) $(E qwen8b-passage 0.025) $(E octen8b-terms 0.01) $(E qwen8b-terms 0.01) \
--bge-pool artifacts/bge-ens-v2/pool.parquet:0.25 --pool union --doc-beta 1.0 --context 1 \
--doc-rank $A/docrerank-v2ens-bge25/doc_rank.parquet"
