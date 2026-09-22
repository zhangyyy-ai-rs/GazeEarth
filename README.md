# GazeEarth

Training-free inference for ultra-high-resolution remote-sensing images. The default configuration uses Qwen3-VL-8B-Instruct.

## Environment

Python 3.11 and a CUDA GPU are recommended.

```bash
conda create -n gazeearth python=3.11 -y
conda activate gazeearth
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps
python -m nltk.downloader wordnet omw-1.4
```

Run the commands below from the repository root. Set `model.path` in the YAML config to a local checkpoint directory if the model is already downloaded.

## Data

Place the datasets as follows to use the default paths in `configs/`:

```text
data/
├── LRS-GRO/
│   ├── test.json
│   └── images/
│       └── ...
├── XLRS-Bench/
│   ├── test.jsonl
│   └── images/
│       └── ...
└── MME-RealWorld-RS/
    ├── test.jsonl
    └── images/
        └── ...
```

Image paths in the annotations are relative to each `images/` directory. If your files are elsewhere, change `dataset.annotation_path` and `dataset.root` in the corresponding YAML config.

| Dataset | Annotation fields |
| --- | --- |
| LRS-GRO | `image_name`, `question`, `ground_truth`; `question_id` and `split` (`test`) are recommended |
| XLRS-Bench | `image_path` (or `image_paths`), `question`, `answer`, `multi-choice options` |
| MME-RealWorld-RS | `image_path`, `question`, `answer`, `options` |

Each annotation file is a JSON array or JSONL with one question per record. For multiple-choice questions, use an option list such as `["(A) ...", "(B) ..."]`; `answer` may contain one or more letters. Export image objects to files before evaluation. For MME-RealWorld, keep only the Remote Sensing subtask and rename the original fields (`Text`, `Ground truth`, `Answer choices`, `Image`) to the fields shown above; the raw annotation file is not directly accepted.

## Run

```bash
# Evaluate the full test sets.
python scripts/eval.py --config configs/xlrs_bench.yaml
python scripts/eval.py --config configs/lrs_gro.yaml
python scripts/eval.py --config configs/mme_realworld_rs.yaml

# Infer one question without benchmark annotations.
python scripts/infer.py --config configs/xlrs_bench.yaml --images path/to/image.jpg --question "What is shown?"

```

An evaluation writes `predictions.jsonl`, `traces.jsonl`, `results.json`, and `run_manifest.json` to the configured `output_dir` (or the directory passed with `--output`). Resume and restart use the same command and output directory:

```bash
python scripts/eval.py --config configs/xlrs_bench.yaml --output outputs/xlrs_qwen
python scripts/eval.py --config configs/xlrs_bench.yaml --output outputs/xlrs_qwen --resume
python scripts/eval.py --config configs/xlrs_bench.yaml --output outputs/xlrs_qwen --overwrite
```

`--resume` continues a matching run after checking its manifest and completed records. `--overwrite` backs up the existing output files and starts again. Do not use the two flags together.

## License

The code is released under Apache-2.0. Model weights and datasets retain their original licenses.
