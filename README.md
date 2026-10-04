# TrustMed-RL

<p align="center">
  <a href="https://huggingface.co/datasets/Lyra-stellAI/TrustMed-RL-Images"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Images-ffd21e?logo=huggingface&logoColor=black" alt="Hugging Face: TrustMed-RL Images"></a>
  <a href="https://huggingface.co/datasets/Lyra-stellAI/trustmed-medical-retrieval-2026-07"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Retrieval%20corpus-ffd21e?logo=huggingface&logoColor=black" alt="Hugging Face: TrustMed retrieval corpus"></a>
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

## Retrieval corpus

The search tool retrieves from a frozen local corpus (snapshot 2026-07-30) with prebuilt BM25 and FAISS
indexes: [Lyra-stellAI/trustmed-medical-retrieval-2026-07](https://huggingface.co/datasets/Lyra-stellAI/trustmed-medical-retrieval-2026-07),
PubMed title and abstract records (NLM 2026 baseline with daily updates through 2026-07-30, MedCPT vectors)
and medical Wikipedia articles (CC BY-SA 4.0). StatPearls (CC BY-NC-ND 4.0) and the MedQA textbooks are
used under research-only terms and are not redistributed; the dataset card describes how to rebuild them.
