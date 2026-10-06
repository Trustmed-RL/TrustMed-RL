# TrustMed-RL

<table align="center">
  <tr>
    <td align="center" width="50%">
      <a href="https://huggingface.co/datasets/Trustmed-RL/TrustMed-RL-Images"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Trustmed--RL%20Images-ffd21e?logo=huggingface&logoColor=black" alt="Trustmed-RL Images on Hugging Face"></a><br>
      <sub>85,581 medical images (7.5 GB), including 49,791 annotated medical images from the 8,453 case reports used for train, val, and test</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://huggingface.co/datasets/Trustmed-RL/trustmed-medical-retrieval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Trustmed--RL%20Medical%20Retrieval%20Corpus-ffd21e?logo=huggingface&logoColor=black" alt="Trustmed-RL Medical Retrieval Corpus on Hugging Face"></a><br>
      <sub>29.2M-record PubMed + medical Wikipedia retrieval corpus with prebuilt BM25 and FAISS indexes</sub>
    </td>
  </tr>
</table>
<p align="center">
  <sub>2,500-case test set, 23,161 SFT pairs and 4,438 RL training cases</sub>
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
(badge above); the 8,453 cases in `data/` are flagged `in_release`.
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
