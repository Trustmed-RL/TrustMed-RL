# TrustMed-RL

<p align="center">
  <a href="https://huggingface.co/datasets/Trustmed-RL/TrustMed-RL-Images"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Images-85%2C581%20medical%20images-ffd21e?logo=huggingface&logoColor=black" alt="Images on Hugging Face: 85,581 medical images"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Trustmed-RL/trustmed-medical-retrieval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Retrieval%20corpus-29.2M%20PubMed%20records%20%2B%20Wikipedia-ffd21e?logo=huggingface&logoColor=black" alt="Retrieval corpus on Hugging Face: 29.2M PubMed records plus medical Wikipedia"></a>
</p>
<p align="center">
  <sub>85,581 medical images (7.5 GB), including 49,791 annotated medical images from the 8,453 case reports used for train, val, and test<br>
  29.2M-record PubMed + medical Wikipedia retrieval corpus with prebuilt BM25 and FAISS indexes<br>
  2,500-case frozen test set<br>
  23,161 SFT pairs and 4,438 RL training cases</sub>
</p>

Code and data for TrustMed-RL. The environment turns an open-access case report into a multi-turn
consultation: the doctor model takes a history from a simulated patient, orders tests, reads the figures
that come back, flags evidence that does not match what was ordered, searches the literature, and commits
to a diagnosis or abstains. Policies are trained with GiGPO / GRPO on Qwen3-VL-8B through verl-agent.

## Layout

```
data/
  sft/     SFT trajectories, 8 parquet shards, images embedded
  rl/      train.parquet, val.parquet, profiles.parquet
  test/    test.parquet, 2,500 cases
src/
  environment.py    consultation environment and rollout runner
  session.py        reset()/step() facade used by the trainer
  prompts.py        all prompt text
  rewards.py        scored trajectories -> GiGPO step records
  outcomes.py       outcome reward backfill
  curriculum.py     2D ScalingInter curriculum controller
  sft_export.py     trajectories -> SFT pairs
  judge/            LLM judge panel, ontology check, verdict fusion
  trainer/          verl-agent patches, curriculum sampler, env package
  tools/            offline advantage simulator, rollout probe
  taxonomy/         action-space taxonomy build
```

## Images

Figures, panel crops and panel groups for every case report in the corpus are on Hugging Face
(badge above); the 8,453 cases in `data/` are flagged `in_release`, with `case_splits` naming their tables.
`image.parquet` has one row per image: the panel annotations, the PMC caption, and the id of the parent
figure. `materialize.py` from that repo recreates the directory trees the environment reads:

```bash
python materialize.py --repo . --out images
export TRUSTMED_IMAGES_DIR=images/figures TRUSTMED_PANEL_DIR=images/panels
```

## Retrieval corpus

Search runs against a local index, given the live NCBI API rate limiting issue. The corpus (PubMed title and abstract records,
NLM 2026 baseline with daily updates through 2026-07-30, plus medical Wikipedia) and its BM25 and FAISS
indexes are stored on Hugging Face (badge above). StatPearls and the MedQA textbooks are used under research-only
terms and are not included; the dataset card provides instruction on how to rebuild them.
