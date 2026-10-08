```markdown
# Blind Spot: Bangladeshi Water-Body Waste Recognition

An empirical probe of a small open-weight vision-language model (`Qwen/Qwen2-VL-2B-Instruct`)
on waste classification, comparing performance on the standard TrashNet benchmark against
real-world waste photographed in Bangladeshi water bodies.

This work was done for the **Fatima Institute of Technology (FIT) — Blind Spots of Frontier Models**
technical challenge.

---

## 1. The Gap

Frontier vision-language models are almost universally evaluated on clean, single-object,
studio-lit waste images — TrashNet, TACO, and their derivatives. These benchmarks share a
specific distribution:

- one object per image, centered, unoccluded
- Western consumer packaging (glass bottles, aluminium cans, cardboard)
- well-lit, neutral background
- a fixed six-class taxonomy

Real waste in Bangladeshi water bodies looks nothing like this. It is:

- **multi-object and occluded** — polythene bags tangled with water hyacinth, bottles half-submerged
- **environmentally degraded** — turbid water, mud, algae, low-light, motion blur
- **culturally specific** — fishing nets, religious offerings, medical waste, construction debris
- **categorically different** — the local taxonomy does not map cleanly onto TrashNet's
  `{cardboard, glass, metal, paper, plastic, trash}`

This is not a model-quality problem. It is a **benchmark coverage problem**: a geographic and
environmental blind spot that current evaluations cannot see, because they never sample from it.

**Why this matters to me personally.** I built the Waste Dataset 2025 by photographing waste in
Bangladeshi water bodies — collecting, annotating, and quality-controlling images across local
categories. Standard benchmarks do not resemble anything I saw while collecting that data.

---

## 2. Model Choice

| Property | Value |
|---|---|
| Model | `Qwen/Qwen2-VL-2B-Instruct` |
| Parameters | 2B (within the 0.6–6B range required) |
| Modality | Vision-language |
| Loading | 4-bit quantisation, runs on a free Colab T4 |
| Why this model | Open-weight, multimodal, small enough to load locally, and a reasonable proxy for how current frontier VLMs behave on vision-language classification |

A 2B VLM is not a frontier model, but it is trained on the same kind of web-scale image-text
data that larger VLMs use. If it fails on Bangladeshi water-body waste, that failure mode is
likely present — and simply hidden — in larger models.

---

## 3. Evaluation Setup

### Control: TrashNet

- Source: `garythung/trashnet` (GitHub) — 2,527 images across 6 classes
- Sample: 90 balanced images (15 per class), fixed seed 42
- Prompts: zero-shot English, few-shot English

### Out-of-distribution: Bangladeshi water-body waste

- Source: Waste Dataset 2025 (own collection)
- Categories: local taxonomy (e.g. `polythene_bag`, `plastic_bottle`, `organic_waste`)
- Prompts: zero-shot English, zero-shot Banglish (romanised Bangla)

### Decoding

- `do_sample=False` (deterministic)
- `max_new_tokens=32`

### Scoring

- Accuracy and macro-F1 per prompt condition
- Confusion matrices per condition
- Manual error tagging across six categories:
  - `correct`
  - `confused_with_standard_category`
  - `ignored_local_context`
  - `hallucinated_object`
  - `language_failure`
  - `generic_or_refusal`

---

## 4. Results

> Results below are from the current run. See `results/summary.csv` and `results/metrics.json`
> for the exact values.

| Set | Prompt | Accuracy | Macro-F1 |
|---|---|---|---|
| TrashNet (control) | zero-shot EN | _see results/summary.csv_ | _see results/summary.csv_ |
| TrashNet (control) | few-shot EN | _see results/summary.csv_ | _see results/summary.csv_ |
| Bangladesh water-body OOD | zero-shot EN | _see results/summary.csv_ | _see results/summary.csv_ |
| Bangladesh water-body OOD | zero-shot Banglish | _see results/summary.csv_ | _see results/summary.csv_ |

The central claim of this repository is the **delta** between the control and OOD rows.

---

## 5. Error Analysis

See `results/bangladesh_outputs.csv` for raw model strings and hand-tagged `error_type`.

Observed failure modes (fill in after manual review):

- water hyacinth predicted as `plastic`
- polythene bag in turbid water predicted as `trash` or `unknown`
- religious and medical waste items silently misclassified into standard TrashNet classes
- Banglish prompts produce English answers, indicating the language signal is ignored
- multi-object scenes collapse to a single guess

Confusion matrices: `results/cm_trashnet_en.png`, `results/cm_trashnet_fewshot_en.png`,
`results/cm_bangladesh_en.png`, `results/cm_bangladesh_banglish.png`.

---

## 6. Path Forward

The blind spot is not fixed by prompting harder. A real fix looks like:

**Data curation**
- Build a local waste taxonomy with native annotators, not translated TrashNet labels.
- Include hard negatives: water hyacinth vs. polythene, dead fish vs. plastic bottle,
  submerged rubble vs. glass.
- Capture low-light, turbid, motion-blurred, and multi-object scenes.
- Support multi-label instances (one photo may contain three categories).

**Fine-tuning**
- LoRA / QLoRA instruction tuning on local image-text pairs.
- Contrastive image-text training with hard negatives in the batch.
- Few-shot retrieval-augmented classification as a lighter-touch baseline.

**Architectural / training-time**
- Condition on geographic and environmental metadata (region, water type, time of day).
- Add calibrated uncertainty so the model abstains when it is out of distribution.
- Human-in-the-loop review for safety-critical deployment.

**Evaluation**
- Publish a Bangladeshi water-body waste benchmark. This is the artifact that would have
  caught the gap in the first place.

---

## 7. Limitations

- **Model scale.** A 2B open-weight VLM is a proxy for frontier behaviour, not a substitute.
  Larger models may show a smaller but still real drop.
- **Sample size.** The OOD set is small. All numbers are directional, not definitive.
- **Single run.** No repeated sampling, no seed sweep.
- **Manual tagging.** Error categories are subjective.
- **Prompt sensitivity.** Results depend on prompt wording; we did not sweep prompts beyond
  the two shown here.

---

## 8. Reproducibility

```
.
├── FIT_BlindSpot_Bangladesh_Waste_Qwen2VL.ipynb   # main notebook
├── results/
│   ├── README.md
│   ├── metrics.json
│   ├── summary.csv
│   ├── trashnet_outputs.csv
│   ├── bangladesh_outputs.csv
│   ├── cm_trashnet_en.png
│   ├── cm_trashnet_fewshot_en.png
│   ├── cm_bangladesh_en.png
│   └── cm_bangladesh_banglish.png
└── README.md
```

To reproduce:

1. Open the notebook in Google Colab with a T4 GPU.
2. Runtime → Run all.
3. Add your `HF_TOKEN` to Colab Secrets (🔑 sidebar) to enable the upload cell.
4. Point `BD_ROOT` at a folder containing `images/` and `labels.csv`.

Do **not** hardcode a Hugging Face token in the notebook. Load it from Colab Secrets.

---

## 9. Artifacts

All evaluation artifacts are published at:

**Hugging Face (public):**
https://huggingface.co/datasets/rezaurrahman/bangladesh-waste-blindspot

This repository serves as the public "Bucket" referenced in the prompt. Hugging Face
Buckets are a storage product distinct from dataset repos and are not universally
available via the public Python client; the equivalent public dataset repository is
used instead.

---

## 10. Citation

If you reference this work:

```bibtex
@misc{ratul2026bangladesh_waste_blindspot,
  author = {Rezaur Rahman Ratul},
  title  = {Blind Spot: Bangladeshi Water-Body Waste Recognition},
  year   = {2026},
  note   = {Technical challenge submission, Fatima Institute of Technology}
}
```

---

## 11. License

Code: MIT
Data: waste images are from the author's own collection; TrashNet images follow their
original license.
```
