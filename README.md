# Automatic Summarization of Medical Records with Hugging Face

Educational prototype that reads **simulated** medical records in JSON format, builds a structured text for each patient, and generates a summary with a pre-trained Hugging Face encoder-decoder model. The summaries of all patients are saved in a JSON file.

> **Warning:** the project uses simulated data only and is **not suitable for clinical decisions**. The generated summaries contain omissions and errors (see [Results and limitations](#results-and-limitations)).

## Folder contents

| File | Description |
|---|---|
| `Automatic_Summarization_Medical_Records.ipynb` | Notebook with the whole pipeline, the explanations and the outputs of the last run. |
| `hospital_records.json` | Simulated medical records (10 patients, 21 hospital stays). |
| `medical_record_summaries.json` | Output: one summary for each of the 10 patients. |
| `README.md` | This documentation. |

## Goal

1. Read a JSON file with the medical records.
2. Build, for each patient, a text with diagnosis, anamnesis, prognosis, tests, medications and admission and discharge dates.
3. Generate a one-paragraph summary with a pre-trained model.
4. Save the summaries in an output JSON file.

No training or fine-tuning is done: the models are used for **inference** only, on the **GPU (CUDA)**.

## Data

The dataset is downloaded from the public URL given in the notebook. Each patient has a list of hospital stays with the fields:

`admission_date`, `discharge_date`, `diagnosis`, `anamnesis`, `prognosis`, `medications` (list), `test_results` (list).

From the data exploration:

- 10 patients and 21 hospital stays in total (1 to 3 per patient);
- no missing values;
- some stays are not in chronological order in the file;
- the two stays of `Patient 9` overlap in dates. The dataset is simulated: the original dates are kept, with no corrections.

## How the pipeline works

1. **Installation and GPU.** `transformers`, `sentencepiece` and `torch` are installed, then the notebook checks that CUDA is available. If it is not, the notebook stops with an error, without falling back to the CPU.
2. **Loading and exploration.** The JSON is downloaded and converted into a Python list; stays and missing values are counted.
3. **Text building.** The stays of each patient are sorted by admission date. Each stay becomes a block with explicit labels (`Diagnosis:`, `Prognosis:`, ...). An empty field becomes `not recorded`.
4. **Tokenization.** The tokens of each text are counted with the tokenizer of each model, which counts differently. T5 requires the prefix `summarize: `.
5. **Long texts.** If the text exceeds the input limit, it is split into parts: one stay at a time is added and, when the limit would be exceeded, a new part is started. Only a single stay that is too long is split by words. The partial summaries are joined. No patient in the dataset exceeds the limit: the function was tested on artificially long texts.
6. **Generation.** The model is loaded on the GPU, the text is tokenized, the summary is generated and converted back to text. One model is loaded at a time and the GPU memory is freed at the end.
7. **Model comparison.** BART and T5 summarize the same three patients, with times measured after a warm-up generation.
8. **Hyperparameter experiment.** On `Patient 2` with BART, three configurations are compared (base, short, sampling).
9. **Final summaries and saving.** All patients are summarized with the chosen model and `medical_record_summaries.json` is written.
10. **Faithfulness check.** Original records and summaries are placed side by side to check for errors and omissions.

## Models and parameters

| Model | Prefix | Input limit used |
|---|---|---|
| `facebook/bart-large-cnn` | none | 1024 tokens |
| `t5-small` | `summarize: ` | 512 tokens (operational limit) |

Starting generation parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `min_length` / `max_length` | 30 / 160 | Minimum and maximum summary length, in tokens. |
| `num_beams` | 4 | Number of candidate sequences evaluated. |
| `no_repeat_ngram_size` | 3 | Avoids repeating groups of 3 tokens. |
| `length_penalty`, `early_stopping` | 1.0, `True` | Length control with beam search. |
| `do_sample` | `False` | Deterministic generation. |

Experiment configurations (BART, `Patient 2`):

- **base:** the parameters in the table;
- **short:** `min_length=10`, `max_length=60`, `num_beams=2`;
- **sampling:** `do_sample=True`, `temperature=0.8`, `top_p=0.9`, `num_beams=1`, with the seed fixed to 42.

BART is the model chosen for the prototype because it is trained for summarization. On the observed data it did not prove reliable, and **the choice is not a clinical validation**.

## Requirements and execution

- Python 3 and an **NVIDIA GPU with CUDA**.
- Internet connection to download the dataset and the model weights on the first run.
- Libraries installed by the first notebook cell: `transformers==4.57.6`, `sentencepiece`, `torch`.

To run the project:

1. Open `Automatic_Summarization_Medical_Records.ipynb`.
2. In Google Colab select **Runtime → Change runtime type → GPU**; locally, use an interpreter with a CUDA-enabled PyTorch build.
3. Run the cells in order from top to bottom.
4. At the end you get `medical_record_summaries.json`.

Note: the GPU check cell sets `TORCH_DISABLE_NATIVE_JIT=1` before importing `torch`. It avoids a compilation error met in WSL with T5, caused by the missing `python3.12-dev` package. The setting does not disable the GPU.

## Output format

`medical_record_summaries.json` contains a list with one element per patient:

```json
[
  {
    "patient_name": "Patient 1",
    "summary": "Hospitalization 1: Pneumonia. Discharge date: 2023-04-14. ..."
  }
]
```

## Execution status

The run took place on an NVIDIA GeForce RTX 4060 GPU (CUDA), in WSL (Ubuntu, Python 3.12), with PyTorch 2.14.1+cu130. The cells after the installation ran without errors and the JSON contains the 10 patients.

**Google Colab has not been verified:** times and results may change with hardware and library versions.

## Results and limitations

Average generation times on the three comparison patients (last run): **BART 0.85 s**, **T5 0.75 s**. These are few measurements, not a benchmark, and they vary on every run.

The pipeline works, but the summaries are not faithful to the records. Observed problems:

| Patient | Problem |
|---|---|
| 2 (BART) | Associates the discharge date of the second stay with the first (`2023-04-20` instead of `2023-01-27`). |
| 2 (T5) | Malformed date: `20,23-01-27`. |
| 3 | Omits the hypertension diagnosis. |
| 5 and 6 | Invent a fourth stay: the originals have three. |
| 8 | Omits the appendicitis diagnosis. |
| 9 | Invents the discharge date `2022-01-01` and omits the arthritis of the second episode. |
| Sampling (`Patient 2`) | Sentences not supported by the record, such as `Withdrawn from hospital`. |

General considerations:

- **Hallucinations and omissions:** a fluent text can contain invented dates, confused stays or missing diagnoses, even without sampling.
- **Language and domain:** data and summaries are in English. BART-CNN is specialized in news summarization and T5-small is a general-purpose model: neither is validated for medical terminology or for records in Italian.
- **Long texts:** splitting the text avoids truncation, but does not guarantee that the facts are preserved; joining partial summaries can lose connections between stays.
- **Privacy:** only simulated data is used. With real data, authorizations, data minimization, access control and GDPR compliance would be needed. In Colab the environment is still a cloud service, and the notebook outputs can also expose sensitive data.
- **Evaluation:** there are no reference summaries, so metrics such as ROUGE and a clinical accuracy percentage were not computed. The check is manual and limited to a few examples.

Any summary should be checked by a professional, comparing it with the original record.
