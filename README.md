<div align="center">

# 🧠 How Good are Foundation Models in Longitudinal MRI Disease Progression Reasoning?

### Time-Aware Multi-View MRI Benchmark   [🔥 MICCAI 2026]

[Wafa Al Ghallabi](https://scholar.google.com/citations?user=m0ez8X8AAAAJ)¹ \*&nbsp; · &nbsp;
[Ritesh Thawkar](https://in.linkedin.com/in/ritesh-thawkar-b13192233)¹ \*&nbsp; · &nbsp;
[Sara Ghaboura](https://huggingface.co/SLMLAH)¹ \*&nbsp; · &nbsp;
[Omkar Thawakar](https://scholar.google.com/citations?user=flvl5YQAAAAJ)¹ &nbsp; · &nbsp;
[Numan Saeed](https://scholar.google.com/citations?user=VHRDcusAAAAJ)¹

[Dana Al Nuaimi](#)² &nbsp; · &nbsp;
[Ajnas Alkatheeri](#)³ &nbsp; · &nbsp;
[Salman Khan](https://salman-h-khan.github.io/)¹ &nbsp; · &nbsp;
[Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home)¹,⁴

¹ Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) &nbsp; · &nbsp;
² Department of Health Abu Dhabi &nbsp; · &nbsp;
³ Fatima College of Health Sciences &nbsp; · &nbsp;
⁴ Linköping University

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)](#)
[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue.svg)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-E7DAB7.svg)](#)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/wafaAlghallabi/Time-Aware-MRI?style=social)](https://github.com/wafaAlghallabi/Time-Aware-MRI/stargazers)

</div>

---

> **TL;DR.** We introduce the **Time-Aware Multi-View MRI Benchmark** — the first large-scale evaluation suite that probes vision-language models on *longitudinal*, *multi-view*, *clinically grounded* MRI reasoning. The benchmark covers **3,920 expert-verified QA pairs** from **890 patients** across **3,200+ timepoints** and **7 cohorts**, spanning **glioblastoma, brain metastases, neurodegeneration, and vestibular schwannoma**. We evaluate **17 VLMs** (closed- and open-source) and find that even state-of-the-art systems systematically fail on clinically critical change-direction recognition.

---

## 📢 Latest Updates

- 🔥🔥🔥 **[May 2026]** Paper **Early Accepted** at **MICCAI 2026** (top 9%) — see you in Strasbourg, France! 🇫🇷
- 🔥 **[May 2026]** Code and evaluation splits released — full dataset preprocessing scripts coming in **one week**.
- 📄 **[Coming soon]** arXiv preprint and project page.
- 🤗 **[Coming soon]** Benchmark on Hugging Face Datasets.

---

## ✨ Key Highlights

- 🧠 **First longitudinal multi-view MRI benchmark** for foundation models — unifies temporal reasoning, multi-view anatomical input, and structured localization guidance.
- 🏥 **Clinically grounded**: 7 expert cohorts spanning glioblastoma, brain metastases, neurodegeneration, and vestibular schwannoma — dual-radiologist verified.
- 📊 **3,920 expert-verified QA pairs** across 5 task families (open-ended, multiple-choice, and binary formats).
- 🤖 **17 VLMs evaluated** — including GPT-4o, GPT-5.2, o4-mini, Gemini-2.5/3 Pro & Flash, Qwen3-VL, Llama-4 Scout/Maverick, InternVL3.5, and MedGemma variants.
- 📐 **New TAC metric** — Time-Aware Composite jointly scores temporal consistency, change characterisation, and structural reasoning fidelity.
- 🔬 **Multi-view ablation** — agentic Resident-Attending protocol isolates the effect of multi-view vs axial-only input on spatial localization and temporal reasoning.

---

## 📋 Overview

Real-world radiology is **comparative and longitudinal**: radiologists assess disease progression by aligning current and prior scans across multiple anatomical views and sequences. Yet most medical VLM benchmarks remain confined to single-timepoint, single-view interpretation.

The **Time-Aware Multi-View MRI Benchmark** closes this gap with five complementary tasks:

| # | Task | Format | # QA Pairs | What it tests |
|---|---|---|---|---|
| 1 | **Temporal Reasoning** | Open-ended | 1,101 | Interval change identification across timepoints |
| 2 | **Disease Progression** | Open-ended | 942 | Trajectory and treatment-response prediction |
| 3 | **Structured Localization Guidance** | MCQ | 828 | Anatomical change regions + boundaries + features |
| 4 | **Temporal Sequence Ordering** | Binary | 487 | Chronological reconstruction of serial scans |
| 5 | **Change Localization Over Time** | MCQ | 562 | Maximal-change timepoints and locations |

<p align="center">
  <img src="assets/figure3_samples.png" alt="Representative QA samples" width="88%">
</p>

---

## 🏆 Benchmark Statistics

| Metric | Value |
|---|---|
| 🧑‍⚕️ Patients | 890 |
| 📅 Longitudinal timepoints | 3,200+ |
| 🩻 Cohorts | 7 |
| ❓ Expert-verified QA pairs | 3,920 |
| 🖼 Sequences | T1, T2, FLAIR, T1CE, DWI, ADC |
| 📐 Views per timepoint | Axial, Coronal, Sagittal (9–12 images) |
| ⏱ Inter-scan intervals | 4 months → 18+ months |
| 👩‍⚕️ Reviewers | 2 board-certified radiologists |
| ✅ Acceptance rate | 72% (dual-approved) |

---

## 📦 Dataset

The benchmark is built on **seven publicly available longitudinal MRI cohorts**, harmonised with a unified preprocessing pipeline (registration → multi-view extraction → sequence-specific intensity normalisation → quality control).

### 🩻 Source Cohorts

| Cohort | Pathology | Access |
|---|---|---|
| **Yale-Brain-Mets-Longitudinal** | Brain metastases | [TCIA](https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/) |
| **UCSF-ALPTDG** | Post-treatment diffuse glioma | [DOI 10.1148/ryai.230182](https://doi.org/10.1148/ryai.230182) |
| **UCSD-PTGBM** | Post-treatment glioblastoma (MGMT/IDH) | [TCIA](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/) |
| **LUMIERE** | Longitudinal glioblastoma + RANO | [Figshare](https://doi.org/10.6084/m9.figshare.c.5904905) |
| **OASIS-2** | Neurodegeneration (longitudinal) | [oasis-brains.org](https://www.oasis-brains.org/) |
| **ADNI** | Alzheimer's disease neuroimaging | [adni.loni.usc.edu](https://adni.loni.usc.edu/) |
| **Vestibular-Schwannoma-MC-RC** | Vestibular schwannoma follow-up | [TCIA](https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-mc-rc/) |

### 🔧 Preprocessing Scripts

> ⏳ **Coming in one week.** The complete preprocessing pipeline — including ANTs-based registration, sequence-specific percentile normalisation (T1CE/post-contrast: p₁–p₉₉.₅; T2/FLAIR: p₂–p₉₈ with adaptive ceiling), multi-view extraction, and the automated quality-control filter — will be released in [`preprocessing/`](preprocessing/) along with cohort-specific config files. Star ⭐ the repo to be notified.

### 📂 Evaluation Splits

Train / val / test splits used in the paper are provided under [`data/splits/`](data/splits/) as JSON manifests, supporting both zero-shot evaluation and future supervised adaptation studies.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/wafaAlghallabi/Time-Aware-MRI.git
cd Time-Aware-MRI

# 2. Create a conda environment
conda create -n time-aware-mri python=3.10 -y
conda activate time-aware-mri

# 3. Install dependencies
pip install -r requirements.txt
```

### 🔑 API Keys (for closed-source models)

```bash
export OPENAI_API_KEY="sk-..."          # GPT-4o, GPT-5.2, o4-mini
export GOOGLE_API_KEY="..."             # Gemini-2.5/3 Pro & Flash
export ANTHROPIC_API_KEY="..."          # (optional)
```

---

## 🚀 Quick Start

Run a single-model evaluation on the benchmark:

```bash
python evaluate.py \
  --model gpt-4o \
  --task temporal_reasoning \
  --data_dir ./data/splits/test \
  --output_dir ./results/gpt-4o
```

Run all 5 tasks for one model:

```bash
bash scripts/run_full_eval.sh gpt-4o
```

---

## 📊 Main Results — Table 1

**Performance of 17 VLMs on the Time-Aware Multi-View MRI Benchmark.** Higher TAC indicates stronger temporal consistency and reasoning fidelity. **Bold** = best per column.

| Model | Final Acc (%) | RS | TAC | TEDS | Trend F1 | Sign Acc | Coverage | Chronology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Closed-source** | | | | | | | | |
| o4-mini | 32.18 | 6.68 | 0.753 | 0.832 | 0.548 | 0.681 | 0.908 | 0.918 |
| GPT-4o | 32.00 | 6.26 | 0.731 | 0.807 | 0.546 | 0.654 | 0.877 | 0.917 |
| GPT-5.2 | 21.20 | 5.83 | 0.661 | 0.805 | 0.192 | 0.639 | **0.921** | 0.856 |
| Gemini-2.5-Flash | 23.57 | 5.83 | 0.692 | 0.780 | 0.477 | 0.596 | 0.875 | 0.825 |
| Gemini-2.5-Pro | 23.66 | 5.88 | 0.672 | 0.785 | 0.504 | 0.528 | 0.730 | 0.957 |
| Gemini-3-Flash | 22.30 | 5.17 | 0.577 | 0.764 | 0.216 | 0.470 | 0.575 | **1.000** |
| Gemini-3-Pro | 35.10 | 5.31 | 0.590 | 0.775 | 0.235 | 0.485 | 0.600 | 0.980 |
| **Open-source** | | | | | | | | |
| **InternVL3.5-Inst** | **35.15** | **6.68** | **0.800** | **0.870** | **0.631** | **0.740** | 0.903 | 0.951 |
| Qwen3-VL-Plus-Thinking | 28.38 | 6.54 | 0.733 | 0.812 | 0.558 | 0.659 | 0.835 | 0.830 |
| Qwen3-VL-235B-Thinking | 30.37 | 6.55 | 0.742 | 0.815 | 0.571 | 0.674 | 0.852 | 0.825 |
| Qwen3-VL-8B-Inst | 24.31 | 6.05 | 0.732 | 0.801 | 0.557 | 0.655 | 0.888 | 0.825 |
| Llama-4-Scout-17B-Inst | 28.47 | 5.78 | 0.708 | 0.810 | 0.485 | 0.601 | 0.860 | 0.870 |
| Llama-4-Maverick-17B-Inst | 26.84 | 5.85 | 0.690 | 0.779 | 0.505 | 0.574 | 0.846 | 0.661 |
| MedGemma-27B-IT | 19.13 | 5.04 | 0.602 | 0.696 | 0.280 | 0.523 | 0.936 | 0.645 |
| MedGemma-1.5-4B-IT | 21.80 | 4.81 | 0.587 | 0.706 | 0.262 | 0.472 | 0.873 | 0.749 |
| MedGemma-4B-IT | 23.50 | 4.58 | 0.572 | 0.717 | 0.245 | 0.421 | 0.809 | 0.854 |

### 🔁 Reproducing Table 1

```bash
# Run the full Table 1 evaluation across all 17 VLMs and 5 tasks
bash scripts/reproduce_table1.sh

# Or run model-by-model
python evaluate.py --model gpt-4o            --tasks all
python evaluate.py --model gemini-3-pro      --tasks all
python evaluate.py --model internvl3.5-inst  --tasks all
# ... etc.

# Aggregate and render the table
python scripts/aggregate_results.py --input results/ --output table1.md
```

Per-model evaluation scripts are organised under [`evaluation/`](evaluation/):

```
evaluation/
├── openai_model.py        # GPT-4o, GPT-5.2, o4-mini
├── gemini_model.py        # Gemini-2.5 / Gemini-3 Pro & Flash
├── internvl_model.py      # InternVL3.5
├── qwen_vl_model.py       # Qwen3-VL family
├── llama4_model.py        # Llama-4 Scout & Maverick
├── medgemma_model.py      # MedGemma variants
└── ...
```

---

## 🔬 Multi-View Configuration Analysis — Table 2

**Effect of multi-view (axial + coronal + sagittal) vs axial-only inputs**, evaluated on the UCSF-GBM subset (1,192 samples) under our agentic Resident-Attending protocol.

| Model | Axial-Only Acc (%) | Multi-View Acc (%) | Δ (pp) | Verdict |
|---|---:|---:|---:|:---:|
| **InternVL3.5-Inst** | 38.0 | **44.2** | **+6.2** | ⬆️ |
| Qwen3-VL-8B-Inst | 51.6 | 43.6 | −8.0 | ⬇️ |
| GPT-4o | 30.5 | 36.7 | +6.2 | ⬆️ |
| Gemini-2.5-Pro | 28.4 | 32.1 | +3.7 | ⬆️ |
| Gemini-2.5-Flash | 25.7 | 28.8 | +3.1 | ⬆️ |
| MedGemma-4B-IT | 27.5 | 21.7 | −5.8 | ⬇️ |

> 🔑 **Key finding.** Multi-view inputs **boost spatial localization** (peaks at 97.3% on progression localization) but **degrade temporal ordering in compact open-source models** (Qwen3-VL-8B: −8.0 pp; MedGemma-4B: −5.8 pp), suggesting information overload. Volumetric quantification stays below 16% across all models — a clear architectural deficiency that 2D multi-view input alone cannot bridge.

### 🔁 Reproducing Table 2

```bash
# Run the multi-view ablation with the Resident-Attending agentic workflow
bash scripts/reproduce_table2.sh

# Single-model multi-view run
python evaluate_multiview.py \
  --model internvl3.5-inst \
  --view_config multi    # or "axial"
  --subset ucsf-gbm \
  --output_dir results/multiview/internvl3.5
```

The agentic Resident-Attending workflow lives in [`evaluation/agentic_model.py`](evaluation/agentic_model.py).

---

## 🙏 Acknowledgments

This work was made possible by the openly released longitudinal MRI cohorts listed in the [Dataset](#-dataset) section. We thank the data contributors at **UCSF**, **UCSD**, **Yale**, **University Hospital Bern (LUMIERE)**, **OASIS**, **ADNI**, and the **Vestibular-Schwannoma-MC-RC consortium**. We also thank the two board-certified radiologists who provided dual review for every QA pair.

---

## 📝 Citation

If you find this benchmark useful in your research, please cite our work:

```bibtex
@inproceedings{alghallabi2026timeaware,
  title     = {How Good are Foundation Models in Longitudinal MRI Disease Progression Reasoning?},
  author    = {Al Ghallabi, Wafa and Thawkar, Ritesh and Ghaboura, Sara and
               Thawakar, Omkar and Saeed, Numan and Al Nuaimi, Dana and
               Alkatheeri, Ajnas and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer}
}
```

---

## 📧 Contact

For questions, issues, or collaborations, please open an [issue](https://github.com/wafaAlghallabi/Time-Aware-MRI/issues) or reach out to:

**Wafa Al Ghallabi** — `wafa.alghallabi@mbzuai.ac.ae`

---
