# Grammar Scoring Engine for Spoken Audio (SHL Hiring Assessment 2026)

Predicts a continuous grammar score (0-5) from 45-60 s spoken audio.

## Approach
- Whisper ASR transcripts (fillers/repetitions preserved) + ASR confidence features
- Acoustic/fluency features (librosa)
- Grammar-error-correction features + transformer text embeddings
- WavLM speech embeddings
- Ridge, SVR, LightGBM on different feature views
- NNLS blend of out-of-fold predictions, clipped to [0, 5]

## Results (769 training samples)
| Metric | Value |
|---|---|
| Training RMSE | 0.2686 |
| CV (OOF) RMSE | 0.5355 |
| CV (OOF) Pearson | 0.9039 |

## Files
- `notebook.ipynb`: full pipeline with report, visualizations and results
- `submission.csv`: predictions for the 216 test samples
