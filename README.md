# TrustMed-RL

<p align="center">
  <a href="https://huggingface.co/datasets/Lyra-stellAI/TrustMed-RL-Images"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Images-ffd21e?logo=huggingface&logoColor=black" alt="Hugging Face: TrustMed-RL Images"></a>
</p>

## Repository layout

| path | content |
|---|---|
| `data/sft/` | supervised fine-tuning trajectories (parquet, images embedded) |
| `data/rl/` | RL task pools `train.parquet` and `val.parquet` (`val_split` = A or B) and the case profiles they read |
| `data/test/` | the frozen test set, 2,500 cases |
| `src/` | the consultation environment, judges and trainer integration |

## Images

Every figure, panel crop and panel group of the cases in `data/`, with per-image annotations in
`image.parquet`: [Lyra-stellAI/TrustMed-RL-Images](https://huggingface.co/datasets/Lyra-stellAI/TrustMed-RL-Images).
