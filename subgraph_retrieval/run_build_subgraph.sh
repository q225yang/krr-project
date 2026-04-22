python build_subgraph.py \
  --conceptnet_csv /scratch/qyang129/conceptnet/conceptnet-assertions-5.7.0.csv \
  --anchors liquid particle flow area \
  --max_leaves_per_anchor 5 \
  --min_bridge_score 3.5 \
  --output_prefix liquid_snowflake