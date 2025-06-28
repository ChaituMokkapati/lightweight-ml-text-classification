# Lightweight ML Text Classification Benchmark

Reproducible CPU-only benchmark comparing five classical machine learning models for text classification (Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost) with a unified TF-IDF pipeline on SMS Spam and AG News.

**Paper:** *A Comparative Study of Classical Lightweight Machine Learning Models for Text Classification* (Computing, Springer Nature)

**Repository:** https://github.com/ChaituMokkapati/lightweight-ml-text-classification

## Authors

- Chaitanya Mokkapati — DVR & Dr. HS MIC College of Technology (AI & ML)
- Dr. Prasad Devarasetty — DVR & Dr. HS MIC College of Technology (CSE)
- Nagulmeera Syyed — DVR & Dr. HS MIC College of Technology (AI & ML)

## Repository structure

```
experiment/
  Lightweight_ML_Text_Classification.ipynb   # Main experiment notebook
  compute_stats.py                           # Wilcoxon tests from all_runs.csv
  generate_fig1_pipeline.py                  # Figure 1 generator
  generate_figures_from_csv.py               # Figures 2 & 4 from results CSV
manuscript/                                  # LaTeX source and figures
results2/content/results/                    # Published result CSVs/JSON
```

## Reproduce results

```bash
pip install scikit-learn xgboost pandas numpy matplotlib scipy
jupyter notebook experiment/Lightweight_ML_Text_Classification.ipynb
```

### Fixed configuration

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Test size | 0.2 (stratified) |
| Repeated runs | 3 |
| AG News subset | 20,000 (stratified) |
| TF-IDF max features | 20,000 |
| N-gram range | (1, 2) |
| min_df | 2 |

## Hardware (paper)

Intel Xeon @ 2.20 GHz, 2 cores, 12 GB RAM, Ubuntu 22.04, Python 3.12, CPU-only.

## Citation

```bibtex
@article{mokkapati2026lightweight,
  author  = {Mokkapati, Chaitanya and Devarasetty, Prasad and Syyed, Nagulmeera},
  title   = {A Comparative Study of Classical Lightweight Machine Learning Models for Text Classification},
  journal = {Computing},
  year    = {2026},
  note    = {Springer Nature}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Archival release

Version-tagged archival release (v1.1.0): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20781441.svg)](https://doi.org/10.5281/zenodo.20781441)

All versions: [10.5281/zenodo.20781407](https://doi.org/10.5281/zenodo.20781407)

## Overleaf bundle

`Mokkapati-Syyed-Devarasetty-comparative-study-lightweight-ml-text-classification-overleaf.zip`

Rebuild archives:

```bash
python experiment/build_release_archives.py
```
