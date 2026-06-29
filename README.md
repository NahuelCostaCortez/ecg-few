# Análisis de ECGs con grandes modelos de lenguaje

Pipeline experimental del TFG para comparar CNN entrenadas con pocos ejemplos
frente a modelos de vision y lenguaje (VLM) usados con in-context learning
(ICL) en ECG de sospecha de Brugada.

El objetivo no es diagnosticar de forma autonoma. El sistema se plantea como
triaje investigacional: priorizar revision experta de trazados sospechosos y,
si procede, activar el circuito clinico que puede incluir recolocacion de V1/V2
y prueba farmacologica con bloqueantes del canal de sodio.

Ultima regeneracion local: 2026-06-28.

## Estado

| bloque | estado |
|---|---|
| Dataset sintetico QRS/ST | completo |
| Dataset real Brugada-HUCA procesado | completo |
| CNN sobre sintetico | completo |
| CNN sintetico -> HUCA | completo |
| Comparacion sintetico vs HUCA | completo |
| Domain adaptation sin etiquetas HUCA | completo |
| Auditoria CNN end-to-end | completa |
| VLM / ICL | protocolo completo; metricas numericas por ejecutar |

## Pregunta experimental

La pregunta principal es: con muy pocos ejemplos etiquetados, cuando merece la
pena entrenar un modelo visual especifico y cuando conviene usar esos mismos
ejemplos como demostraciones en contexto para un VLM?

La rama CNN ya esta ejecutada y sirve como precedente empirico: en sintetico
aprende, pero en HUCA real no logra un triaje robusto ni siquiera con adaptacion
de dominio. La rama VLM/ICL usa la misma imagen, los mismos pacientes, los
mismos presupuestos `k` y las mismas metricas; solo cambia como se usan los
ejemplos.

La ruta CNN mantiene una decision interpretable: aprende criterios
morfologicos y deriva Brugada mediante una regla explicita:

```text
ECG -> CNN multi-label -> RBBB, ST_ELEVATION, T_WAVE_INVERSION
    -> regla AND -> Brugada / Normal
```

![Pipeline general](assets/readme/tfg_pipeline_overview.svg)

## Regla Clinica

El detector produce tres probabilidades:

- `RBBB`
- `ST_ELEVATION`
- `T_WAVE_INVERSION`

La etiqueta final se deriva asi:

```text
if RBBB and ST_ELEVATION and T_WAVE_INVERSION:
    final = Brugada
else:
    final = Normal
```

La implementacion canonica vive en `src/ecg_few/findings.py`.

## Datasets

### Sintetico QRS/ST

Se genera con:

```bash
scripts/run/build_simulator_qrs_dataset.sh
```

| dato | valor |
|---|---:|
| pacientes | 100 |
| imagenes | 300 |
| Brugada derivado, tres condiciones presentes | 20 |
| normal o incompleto | 80 |
| leads por paciente | V1, V2, V3 |

Cada imagen sintetica tiene etiquetas honestas para las tres condiciones:
`label_rbbb`, `label_st_elevation` y `label_t_wave_inversion`.

### Brugada-HUCA Real

Se reconstruye desde WFDB con:

```bash
scripts/run/build_brugada_huca_dataset.sh
```

| dato | valor |
|---|---:|
| pacientes incluidos | 317 |
| imagenes | 951 |
| Brugada clinico | 116 |
| Normal clinico | 201 |
| pacientes excluidos | 46 |
| leads por paciente | V1, V2, V3 |

En HUCA real las columnas QRS/ST quedan vacias a proposito. Solo se conserva
`clinical_brugada`; las etiquetas reales por condicion no existen en este
dataset procesado.

## Protocolo LOOCV

El protocolo es patient-level leave-one-out cross-validation.

![Protocolo LOOCV](assets/readme/loocv_protocol.svg)

Para cada paciente:

1. Se deja el paciente como test.
2. Se seleccionan pacientes de contexto segun `k` y `seed`.
3. Se selecciona validacion para calibrar umbral cuando aplica.
4. La CNN predice las tres condiciones.
5. La regla AND produce `Brugada` o `Normal`.
6. Se compara contra la referencia patient-level.

Grid final:

```text
k = 2, 4, 8, 16, 32
seeds = 42, 123, 2026
```

Esto produce:

| dataset | pacientes por run | runs | predicciones patient-level |
|---|---:|---:|---:|
| sintetico | 100 | 15 | 1500 |
| HUCA | 317 | 15 | 4755 |

## CNN

Configuracion final:

```text
arquitectura: ResNet18
pesos: torchvision default
salidas: 3 logits multi-label
loss: BCEWithLogitsLoss
image_size: 224
epochs: 20
batch_size: 32
threshold_strategy: val_derived_balanced_accuracy
```

No queda selector de arquitectura ni ruta de CNN pequena: todos los experimentos
usan la misma ResNet18.

### CNN Sobre Sintetico

Comando:

```bash
RESUME=0 scripts/run/run_cnn_simulator_qrs_loocv.sh
```

| k | balanced accuracy | F1 | sensibilidad | especificidad | ROC AUC | AP |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.625 +/- 0.022 | 0.401 +/- 0.023 | 0.650 | 0.600 | 0.520 | 0.268 |
| 4 | 0.631 +/- 0.042 | 0.407 +/- 0.044 | 0.617 | 0.646 | 0.566 | 0.268 |
| 8 | 0.685 +/- 0.023 | 0.468 +/- 0.028 | 0.683 | 0.688 | 0.629 | 0.383 |
| 16 | 0.773 +/- 0.013 | 0.569 +/- 0.021 | 0.800 | 0.746 | 0.755 | 0.534 |
| 32 | 0.848 +/- 0.030 | 0.673 +/- 0.048 | 0.883 | 0.812 | 0.857 | 0.614 |

![CNN sintetico balanced accuracy](assets/readme/cnn_simulator_qrs_balanced_accuracy_by_k.png)

![Matriz sintetica k32](assets/readme/cnn_simulator_qrs_k32_confusion_matrix.png)

Lectura: en el mismo dominio generativo la mejora con `k` es clara. Esto valida
que el protocolo, la arquitectura y la regla derivada aprenden senales utiles
cuando no hay salto de dominio.

### CNN Sintetico -> HUCA

Comando:

```bash
RESUME=0 scripts/run/run_cnn_loocv.sh
```

Entrena con `data/simulator_qrs` y evalua sobre `data/brugada_huca`.
HUCA se usa como referencia clinica y para calibracion de umbral, no como fuente
de etiquetas QRS/ST inventadas.

| k | balanced accuracy | F1 | sensibilidad | especificidad | ROC AUC | AP |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.509 +/- 0.010 | 0.484 +/- 0.005 | 0.693 | 0.325 | 0.525 | 0.404 |
| 4 | 0.509 +/- 0.023 | 0.493 +/- 0.011 | 0.730 | 0.289 | 0.495 | 0.388 |
| 8 | 0.490 +/- 0.018 | 0.469 +/- 0.011 | 0.672 | 0.308 | 0.496 | 0.383 |
| 16 | 0.516 +/- 0.023 | 0.480 +/- 0.015 | 0.658 | 0.375 | 0.569 | 0.478 |
| 32 | 0.499 +/- 0.012 | 0.466 +/- 0.010 | 0.644 | 0.355 | 0.491 | 0.423 |

![CNN HUCA balanced accuracy](assets/readme/cnn_huca_balanced_accuracy_by_k.png)

![Matriz HUCA k16](assets/readme/cnn_huca_k16_confusion_matrix.png)

Lectura: el rendimiento real queda cerca de chance en balanced accuracy. La
sensibilidad es moderada, pero la especificidad baja indica sobredeteccion del
patron derivado en normales. El gap sintetico-real es el problema principal.

## Comparacion Sintetico vs HUCA

| k | BA sintetico | BA HUCA | F1 sintetico | F1 HUCA |
|---:|---:|---:|---:|---:|
| 2 | 0.625 | 0.509 | 0.401 | 0.484 |
| 4 | 0.631 | 0.509 | 0.407 | 0.493 |
| 8 | 0.685 | 0.490 | 0.468 | 0.469 |
| 16 | 0.773 | 0.516 | 0.569 | 0.480 |
| 32 | 0.848 | 0.499 | 0.673 | 0.466 |

![Comparacion BA](assets/readme/cnn_balanced_accuracy_sim_vs_real.png)

![Comparacion F1](assets/readme/cnn_f1_sim_vs_real.png)

El F1 no debe leerse aislado porque la prevalencia cambia: el sintetico tiene
20/100 positivos derivados, mientras HUCA tiene 116/317 positivos clinicos. La
balanced accuracy es la metrica principal de lectura.

## Adaptacion De Dominio

Objetivo: usar HUCA como dominio objetivo no etiquetado sin inventar etiquetas
QRS/ST.

```text
fuente etiquetada: data/simulator_qrs
objetivo no etiquetado: data/brugada_huca
salida: RBBB/ST_ELEVATION/T_WAVE_INVERSION -> regla AND -> Brugada/Normal
```

Metodos ejecutados:

- `ssl`: preentrenamiento SimCLR del encoder con HUCA sin etiquetas.
- `coral`: alineacion de covarianza de features.
- `mmd`: alineacion de distribucion con kernel RBF.
- `dann`: clasificador de dominio con gradient reversal.

Comandos:

```bash
METHOD=coral RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=mmd RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=dann RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=none SSL_PRETRAIN_EPOCHS=3 OUTPUT_ROOT=outputs/cnn_domain_adaptation/ssl REPORT_DIR=reports/loocv/cnn_domain_adaptation/ssl RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
```

Comparacion:

```bash
uv run --no-sync python scripts/eval/compare_cnn_domain_adaptation_reports.py
```

Resultados `k=16,32`:

| metodo | k | balanced accuracy | F1 | sensibilidad | especificidad |
|---|---:|---:|---:|---:|---:|
| baseline | 16 | 0.516 +/- 0.023 | 0.480 | 0.658 | 0.375 |
| ssl | 16 | 0.477 +/- 0.027 | 0.458 | 0.658 | 0.297 |
| coral | 16 | 0.524 +/- 0.008 | 0.481 | 0.641 | 0.408 |
| mmd | 16 | 0.516 +/- 0.019 | 0.474 | 0.635 | 0.396 |
| dann | 16 | 0.515 +/- 0.019 | 0.470 | 0.621 | 0.410 |
| baseline | 32 | 0.499 +/- 0.012 | 0.466 | 0.644 | 0.355 |
| ssl | 32 | 0.503 +/- 0.014 | 0.476 | 0.672 | 0.333 |
| coral | 32 | 0.528 +/- 0.013 | 0.480 | 0.632 | 0.423 |
| mmd | 32 | 0.549 +/- 0.029 | 0.498 | 0.649 | 0.448 |
| dann | 32 | 0.509 +/- 0.028 | 0.472 | 0.644 | 0.375 |

![Adaptacion dominio BA](assets/readme/cnn_domain_adaptation_balanced_accuracy_by_k.png)

![Adaptacion dominio F1](assets/readme/cnn_domain_adaptation_f1_by_k.png)

Lectura: la mejor configuracion HUCA de esta bateria es `mmd` con `k=32`
(`BA=0.549`). CORAL tambien mejora `k=32`. La adaptacion de dominio ayuda sobre
el baseline de `k=32`, sobre todo recuperando especificidad, pero el techo sigue
siendo modesto.

## Grad-CAM

El pipeline guarda paneles Grad-CAM por fold en `outputs/.../gradcam_panel.png`.
Sirven para inspeccionar si la CNN mira regiones compatibles con QRS, ST y
repolarizacion.

Ejemplo sintetico:

![Grad-CAM sintetico](assets/readme/cnn_simulator_qrs_gradcam_example.png)

Ejemplo HUCA:

![Grad-CAM HUCA](assets/readme/cnn_huca_gradcam_example.png)

## VLM / ICL

La ruta VLM/ICL es el segundo brazo de la comparacion. El diseno queda
especificado para Gemma 4 y MedGemma 1.5 con los mismos pacientes, el mismo
plan leave-one-out, los mismos presupuestos `k = 0, 2, 4, 8, 16, 32` y las
mismas metricas patient-level usadas por la CNN. Las celdas numericas se
rellenaran unicamente cuando existan inferencias auditadas por muestra.

Semanticas previstas:

```text
TASK=morphology:
  imagen ECG -> JSON con RBBB/ST_ELEVATION/T_WAVE_INVERSION -> regla AND

TASK=clinical:
  una derivacion fija HUCA por paciente -> JSON con clinical_brugada
```

Celda unica para que el tutor ejecute toda la evaluacion VLM sobre simulador y
HUCA real. Requiere tener ya levantado un servidor vLLM/OpenAI-compatible en
`VLM_API_BASE` que sirva los modelos indicados, o un router que acepte ambos
identificadores.

```bash
# 0) Ejecutar desde la raiz del repositorio.
set -eu
export MPLBACKEND=Agg
export VLM_API_BASE="${VLM_API_BASE:-http://your-host:8000/v1}"
export VLM_MODELS="${VLM_MODELS:-google/gemma-4-E4B-it,google/medgemma-4b-it}"
export VLM_RUNTIME="${VLM_RUNTIME:-remote_api}"
export K_VALUES="${K_VALUES:-0,2,4,8,16,32}"
export CONTROL_K_VALUES="${CONTROL_K_VALUES:-8,16,32}"
export CLINICAL_LEAD="${CLINICAL_LEAD:-V2}"
export SEEDS="${SEEDS:-42,123,2026}"
export CONDITIONS="${CONDITIONS:-zero_shot,normal,balanced,permuted,no_support_images}"

# 1) Validar plan VLM en el simulador QRS/ST, sin inferencia.
DATASET_ROOT="$PWD/data/simulator_qrs" \
CONTEXT_DATASET_ROOT="$PWD/data/simulator_qrs" \
REPORT_DIR="$PWD/reports/loocv/vlm_simulator_qrs" \
OUTPUT="$PWD/reports/loocv/vlm_simulator_qrs/vlm_setup_validation.json" \
scripts/run/validate_vlm_loocv.sh

# 2) Ejecutar inferencia VLM en el simulador QRS/ST.
DATASET_ROOT="$PWD/data/simulator_qrs" \
CONTEXT_DATASET_ROOT="$PWD/data/simulator_qrs" \
OUTPUT_ROOT="$PWD/outputs/vlm_simulator_qrs_loocv" \
REPORT_DIR="$PWD/reports/loocv/vlm_simulator_qrs" \
scripts/run/run_vlm_loocv.sh

# 3) Validar plan VLM en Brugada-HUCA real, usando el simulador como contexto
# etiquetado QRS/ST y HUCA solo como conjunto de evaluacion clinica.
DATASET_ROOT="$PWD/data/brugada_huca" \
CONTEXT_DATASET_ROOT="$PWD/data/simulator_qrs" \
REPORT_DIR="$PWD/reports/loocv/vlm" \
OUTPUT="$PWD/reports/loocv/vlm/vlm_setup_validation.json" \
scripts/run/validate_vlm_loocv.sh

# 4) Ejecutar inferencia VLM en Brugada-HUCA real.
DATASET_ROOT="$PWD/data/brugada_huca" \
CONTEXT_DATASET_ROOT="$PWD/data/simulator_qrs" \
OUTPUT_ROOT="$PWD/outputs/vlm_loocv" \
REPORT_DIR="$PWD/reports/loocv/vlm" \
scripts/run/run_vlm_loocv.sh

# 5) Validar ICL clinico real-context -> real-test.
# Esta rama usa K pacientes HUCA reales como demostraciones y fuerza que,
# para k>=2, el contexto incluya al menos un normal y un Brugada cuando existan.
# Todos los modelos ven la misma derivacion fija, por defecto V2.
TASK=clinical \
DATASET_ROOT="$PWD/data/brugada_huca" \
CONTEXT_DATASET_ROOT="" \
REPORT_DIR="$PWD/reports/loocv/vlm_real_context" \
OUTPUT="$PWD/reports/loocv/vlm_real_context/vlm_setup_validation.json" \
scripts/run/validate_vlm_loocv.sh

# 6) Ejecutar ICL clinico HUCA real-context -> HUCA real-test.
TASK=clinical \
DATASET_ROOT="$PWD/data/brugada_huca" \
CONTEXT_DATASET_ROOT="" \
OUTPUT_ROOT="$PWD/outputs/vlm_real_context_loocv" \
REPORT_DIR="$PWD/reports/loocv/vlm_real_context" \
scripts/run/run_vlm_loocv.sh

# 7) Comparar CNN frente a VLM en simulador, HUCA morfologico y HUCA clinico.
CNN_SUMMARY="$PWD/reports/loocv/cnn_simulator_qrs/cnn_summary_by_seed.csv" \
VLM_SUMMARY="$PWD/reports/loocv/vlm_simulator_qrs/vlm_summary_by_seed.csv" \
VLM_CONDITION=normal \
OUTPUT_DIR="$PWD/reports/loocv/comparison_vlm_simulator_qrs" \
scripts/run/build_loocv_comparison.sh

CNN_SUMMARY="$PWD/reports/loocv/cnn/cnn_summary_by_seed.csv" \
VLM_SUMMARY="$PWD/reports/loocv/vlm/vlm_summary_by_seed.csv" \
VLM_CONDITION=normal \
OUTPUT_DIR="$PWD/reports/loocv/comparison" \
scripts/run/build_loocv_comparison.sh

CNN_SUMMARY="$PWD/reports/loocv/cnn/cnn_summary_by_seed.csv" \
VLM_SUMMARY="$PWD/reports/loocv/vlm_real_context/vlm_summary_by_seed.csv" \
VLM_CONDITION=normal \
OUTPUT_DIR="$PWD/reports/loocv/comparison_vlm_real_context" \
scripts/run/build_loocv_comparison.sh

# 8) Auditar que los artefactos finales existen.
scripts/run/audit_loocv_results.sh
```

Prompts:

```text
prompts/system/qrs_huca.md
prompts/qrs/right_precordial_morphology.md
prompts/system/clinical_brugada_huca.md
prompts/clinical/brugada_patient.md
```

Condiciones previstas:

- zero-shot (`k=0`), sin demostraciones;
- ICL normal con pares imagen-etiqueta correctos;
- ICL balanceado cuando el fold lo permita;
- etiquetas permutadas para detectar dependencia espuria del prompt;
- demostraciones sin imagen de apoyo para comprobar uso visual real.

En `TASK=clinical`, cada demostracion es un paciente HUCA real representado por
una unica derivacion fija, por defecto `CLINICAL_LEAD=V2`, y una etiqueta
`clinical_brugada`. La consulta del paciente test usa esa misma derivacion, sin
enviar V1, V2 y V3 de golpe al VLM. La seleccion de
contexto es balanceada por diseno: para `k>=2` incluye normales y Brugada
siempre que el fold lo permita, y el paciente test nunca aparece en sus propios
ejemplos. Como la derivacion se fija globalmente, ambos modelos ven exactamente
las mismas imagenes.

El wrapper usa por defecto:

```text
K_VALUES=0,2,4,8,16,32
CONTROL_K_VALUES=8,16,32
SEEDS=42,123,2026
CONDITIONS=zero_shot,normal,balanced,permuted,no_support_images
```

Prueba de humo sin GPU ni API, generando todos los tipos de artefacto con
predicciones deterministas negativas:

```bash
VLM_RUNTIME=local_gpu \
DRY_RUN_PREDICTIONS=negative \
LIMIT_FOLDS=2 \
K_VALUES=0,2 \
CONTROL_K_VALUES=2 \
MODELS=fake/gemma4,fake/medgemma \
scripts/run/run_vlm_loocv.sh
```

Artefactos esperados tras ejecutar inferencia:

```text
outputs/vlm_loocv/<model>/<condition>/k<k>_seed<seed>/fold_predictions.csv
outputs/vlm_loocv/<model>/<condition>/k<k>_seed<seed>/fold_predictions.jsonl
outputs/vlm_loocv/<model>/<condition>/k<k>_seed<seed>/metrics.json
outputs/vlm_real_context_loocv/<model>/<condition>/k<k>_seed<seed>/fold_predictions.csv
reports/loocv/vlm/vlm_campaign_manifest.json
reports/loocv/vlm_real_context/vlm_campaign_manifest.json
reports/loocv/vlm/vlm_summary_by_seed.csv
reports/loocv/vlm/vlm_summary_by_k.csv
reports/loocv/vlm/vlm_summary_by_model_condition_k.csv
reports/loocv/vlm/confusion_by_model_condition_k/*.png
reports/loocv/vlm/balanced_accuracy_by_k.png
reports/loocv/vlm/f1_by_k.png
```

Si se ejecuta un unico modelo por servidor, basta con cambiar `VLM_MODELS` por
un solo identificador y repetir la celda para cada modelo; las salidas quedan
separadas por modelo, condicion, `k` y semilla.

El capitulo `thesis/thesis/chapters/09_vlm_icl.tex` contiene las tablas
reservadas para Gemma 4, MedGemma 1.5, controles multimodales y matriz de
decision CNN frente a VLM/ICL.

## Artefactos

Datasets:

```text
data/simulator_qrs/
data/brugada_huca/
```

Salidas CNN:

```text
outputs/cnn_simulator_qrs_loocv/
outputs/cnn_loocv/
outputs/cnn_domain_adaptation/
```

Reportes:

```text
reports/loocv/cnn_simulator_qrs/
reports/loocv/cnn/
reports/loocv/cnn_comparison/
reports/loocv/cnn_domain_adaptation/
reports/loocv/audit/
```

Figuras versionadas para este README:

```text
assets/readme/
```

## Reproduccion

Instalacion:

```bash
uv sync --extra dev --extra cnn
```

Si se reconstruye HUCA desde WFDB:

```bash
uv sync --extra dev --extra cnn --extra real-data
```

Reconstruir datasets:

```bash
scripts/run/build_simulator_qrs_dataset.sh
scripts/run/build_brugada_huca_dataset.sh
```

Relanzar CNNs:

```bash
RESUME=0 scripts/run/run_cnn_simulator_qrs_loocv.sh
RESUME=0 scripts/run/run_cnn_loocv.sh
```

Relanzar domain adaptation:

```bash
METHOD=coral RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=mmd RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=dann RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
METHOD=none SSL_PRETRAIN_EPOCHS=3 OUTPUT_ROOT=outputs/cnn_domain_adaptation/ssl REPORT_DIR=reports/loocv/cnn_domain_adaptation/ssl RESUME=0 scripts/run/run_cnn_domain_adaptation_loocv.sh
```

Comparaciones y auditoria:

```bash
uv run --no-sync python scripts/eval/compare_cnn_sim_real_reports.py
uv run --no-sync python scripts/eval/compare_cnn_domain_adaptation_reports.py
uv run --no-sync python scripts/eval/audit_loocv_results.py
```

Validacion de codigo:

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync python -m compileall -q src scripts
```

## Conclusiones

- En sintetico, la CNN multi-label aprende de forma consistente y mejora con
  mas contexto.
- En HUCA, el salto de dominio degrada claramente la balanced accuracy.
- La decision derivada es interpretable, pero la especificidad real sigue baja.
- La adaptacion de dominio no supervisada mejora el mejor punto `k=32`, con MMD
  como configuracion mas fuerte de esta bateria.
- Estos resultados justifican la comparacion VLM/ICL: la CNN establece que
  entrenar pesos con `k <= 32` no basta en HUCA.
- VLM/ICL queda documentado como segundo brazo experimental completo; sus
  metricas numericas dependen de ejecutar inferencias auditadas.
