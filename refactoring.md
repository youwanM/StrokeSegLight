
  
  
| Ancien Fichier                                          |            Statut            | Destination                      |
|:------------------------------------------------------- |:----------------------------:|:-------------------------------- |
| **Ancien dossier `./Distillation/`**                    |                              |                                  |
| `Distillation/models.py`                                |           Déplacé            | `Distiller/models.py`            |
| `Distillation/PrepareData.py`                           |    Renommé et restructuré    | `Distiller/prepare_data.py`      |
| `Distillation/prepareTest.py`                           |    Renommé et restructuré    | `Distiller/prepare_test.py`      |
| `Distillation/teach.py`                                 |         Restructuré          | `Distiller/teach.py`             |
| `Distillation/boxplot.py`                               |       Remplacé/intégré       | `Evaluation/plot_metrics.py`     |
| `Distillation/lesion_wise_evaluation.py`                |       Remplacé/intégré       | `Evaluation/evaluate_models.py`  |
| `Distillation/quantisation_equivalence.py`              |       Remplacé/intégré       | `Evaluation/test_equivalence.py` |
| `Distillation/evaluation/plotMetricsConfidenceBands.py` |       Remplacé/intégré       | `Evaluation/plot_metrics.py`     |
| `Distillation/evaluation/plotMetricsErrorBars.py`       |       Remplacé/intégré       | `Evaluation/plot_metrics.py`     |
| **Ancien dossier `./Quantisation/`**                    |                              |                                  |
| `Quantisation/build.cmd`                                |              —               |                                  |
| `Quantisation/convertONNXgui.py`                        |           Fusionné           | `Exporter/convert_onnx_gui.py`   |
| `Quantisation/convertONNX_KD_gui.py`                    |           Fusionné           | `Exporter/convert_onnx_gui.py`   |
| `Quantisation/convertONNX_LateFusion.py`                |           Fusionné           | `Exporter/convert_onnx_gui.py`   |
| `Quantisation/evaluate.py`                              |    Dédupliqué et intégré     | `Evaluation/`                    |
| `Quantisation/models.py`                                | Doublon supprimé, centralisé | `Distiller/models.py`            |
| **Ancien dossier `./RuntimeExp/`**                      |                              |                                  |
| `RuntimeExp/main.py`                                    |         Restructuré          | `Benchmark/analyze_benchmark.py` |
| `RuntimeExp/RuntimeExp.py`                              |         Restructuré          | `Benchmark/run_benchmark.py`     |