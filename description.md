# <p align=center >Forecasting Emerging Research Areas</p>

### Two Main Components:

- **`Topic discovery:`** Trial descriptions will be emdedded using transformers. Embeddings wil be clustered with KMeans, KMedioid, DBScan giving latent research topics.
- **`Temporal forecasting:`** For each discovered topic, we track its frequency over time(aggregate TS columns somehow). This produces topic-specific time series reflecting the intensity of research interest. We will then train forecasting models to predict which topics will grow in the near future.

Success will be evaluated through testing: training models on data up to year N and predicting future trends then see if these trends actually realize. Metrics will include ranking correlation of predicted vs. actual growth, and classification accuracy for identifying “emerging” topics.

The that compliment this are:

- `study_first_submit_date`, `start_date`, `primary_completion_date`   → metadata
- `condition_browse_module` → disease areas.
- `intervention_browse_module` → drug classes, devices, etc.
- `study_type` and `overall_status` → trial design and outcomes.

Contributes by combining semantic embeddings with metadata to predict growth in research activity rather than just analyzing frequencies.

Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
