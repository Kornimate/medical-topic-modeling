# <p align="center">Anchored Topic Modeling for Interpretable Clinical Trial Landscapes</p>

## **Abstract**

The growth of clinical research has made it a bit more difficult to understand how the new medical topics come to light and evolve throughout the thousands of clinical trials. Many traditional topic modeling techniques capture latent themes but they lack interpretability, especially in some biomedical domains. This project's goal is to develop an **anchored topic modeling** framework that combines transformer-based text embeddings with structured metadata from the **Clinical Trials** dataset. By using domain-specific *anchors* like as MeSH terms and condition categories, we guide the topic formation towards medically more meaningful areas while preserving the ability of the model to create new themes. Our goal is to produce a more interpretable representation of the clinical research landscape, which enables the experts to explore how trials group around diseases and treatments. The project aims to overcome the gap between unsupervised text analysis and domain knowledge and understanding in medical NLP applications.

## **Contributions**

Our project contributes a **hybrid topic modeling pipeline** for large-scale biomedical text analysis. The contributions are:

1. **Anchored Topic Discovery:**  
   We implement topic modeling which is guided by domain-specific *anchors*, that are constructed from the processed features such as “condition_browse_module_clean”, “mesh_terms_clean” “condition_browse_module_clean”,” “brief_summary_clean”. Our aim is to test out these different features to see what would fit as features for the anchors in the best way possible. These anchors bias the cluster formation towards medically more meaningful categories.

2. **Hybrid Representation:**  
   We experiment with integrating semantic embeddings from transformer models (e.g., BioClinicalBERT or Sentence-BERT) with some of the features that we derived such as  keywords_clean. By providing a joint representation, it will capture both linguistic and ontological similarity.

3. **Interpretable Clustering:**  
   Unlike unsupervised clustering (e.g., KMeans only on embeddings), our anchored method will produce interpretable topics that domain experts can understand and validate.

4. **Evaluation Framework:**  
   We propose both intrinsic (topic coherence, cluster quality) to quantify interpretability and performance.

5. **Open and Reproducible Pipeline:**  
   The implementation is provided in a single `main.ipynb` notebook with documented preprocessing, embedding, and clustering steps for reproducibility and transparency. (process step after preprocessing is future work)

## **Dataset**

We use the **Clinical Trials dataset** from Hugging Face:  
[https://huggingface.co/datasets/louisbrulenaudet/clinical-trials](https://huggingface.co/datasets/louisbrulenaudet/clinical-trials)

### **Relevant Fields (used in this project)**
- `nct_id` — unique trial identifier (index / joins).
- `brief_title` — short title.
- `brief_summary`, `detailed_description` — primary textual inputs for embeddings.
- `eligibility_criteria` — optional textual signal, will be included in extended embeddings when available.
- `keywords` — used for enrichment/anchor support.
- `mesh_terms` — core anchor source.
- `condition_browse_module` — core anchor source.
- `intervention_browse_module` — anchor + feature.
- `conditions` — raw condition strings, normalization + anchor backfill.
- `interventions` — raw intervention strings, normalization + anchor backfill.
> We rely on the columns above. Other metadata features are **not** esential to our analysis and therefore omitted from the procedure.

### **Data Handling**

- The dataset is in arrow (pyarrow) format and can be accessed through the Hugging Face API.  
- We subsample for development before running on the full dataset. We can also process the whole data with batching.
- We process columns from the original dataset and map them into more easily processable formats.
- Missing values and long text fields will be cleaned and standardized.  
- Each record will be used for embeddings using a sentence-transformer model.  
- Structured terms are preprocessed to form anchor sets and will be mapped to embedding space.  

No extrenal datasets are used. All modeling and evaluation rely on this dataset. The dataset has only training set, so split will be necessary for model training but preprocess is necessary for all data.

## **Methods**

### **1. Text Embedding**
- Use transformer-based embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2` or domain-specific models like BioClinicalBERT). (this decision will be made after training and seeing the results)
- Each trial’s free text content will be encoded into an embedding vector representation.  
- We may use dimensionality reduction (PCA or UMAP) for visualization and clustering efficiency.

### **2. Anchor Construction**
- Extract relevant information and controlled terms from:  
  - `mesh_terms_clean`
  - `condition_browse_module_clean`
  - `intervention_browse_module_clean`
  - `brief_summary_clean`
- Convert all anchors and features into embedding vectors using the same transformer model and test different feature combinations to determine which configuration results in the most interpretable clusters.

### **3. Anchored Topic Modeling**
- Modify clustering to be *anchor-guided*:  
  - Anchors act as reference points influencing the similarity structure. (e.g. embeddings close to oncology-related anchors are softly pulled into oncology clusters)  
- Evaluate and experiment with different implementations such as:  
  - **Constrained BERTopic** with anchor guidance.  
  - **Post-processing of clusters** using nearest anchor similarity.  
  - **Hybrid embedding model** combining text and categorical metadata.
 - Experiment with different clustering algorithms:
 	- KMEANS
	- KMEDIOID
	- DBSCAN
    - HDBSCAN 

### **4. Evaluation**
- **Intrinsic metrics:**  
  - Topic coherence (UMass / c_v).  
  - Silhouette score and intra-cluster variance.  
- We also consider using extrinsic metrics

### **5. Visualization and Interpretation**
- Visualize discovered topics using UMAP or t-SNE projections.  
- Display representative keywords and anchor associations for each cluster.  
- Provide summaries showing how discovered topics align or deviate from established biomedical classifications.

## **Proposed Timeline**

- **Phase 1 – Data Familiarization and Preprocessing:**  
  Load the dataset, explore structure and quality, extract textual and metadata features, handle missing values.

- **Phase 2 – Embedding Generation:**  
  Generate transformer embeddings for trial descriptions and anchor terms, validate embedding quality through nearest-neighbor inspection.

- **Phase 3 – Anchored Clustering / Topic Modeling:**  
  Implement anchor-guided clustering or modified BERTopic, experiment with anchor weighting and features, and tune the number of topics.

- **Phase 4 – Evaluation and Refinement:**  
  Measure alignment with pre-known categories, observe interpretability, and visualize results.

- **Phase 5 – Documentation and Report Preparation:**  
  Finalize code and visualizations, update README, and prepare the final report and presentation.

## **Organization within the Team**

Each member contributes complementary skills to ensure balanced development:

- **Peeter Tarvas:** Data preprocessing and Anchor extraction and integration with topic modeling, report writing
- **Eliasz Piotr:** Data preprocessing, visualization and evaluation, report writing  
- **⁨Kornidesz Máté:** Data preprocessing and embedding pipeline implementation, report writing

### **Internal Milestones**
- Completion of embedding functions  
- Anchor extraction and validation
- First clustering results with anchors 
- Evaluation metrics and visual summaries
- Final report and README polishing before Milestone P3 submission

## Appendix
### Contents of repository:
 - `main.ipynb` : main notebook containing preprocessing and analysation pipeline
 - `README.md` : readme file containging the project description
 - `.gitignore` : gitignore file to avoid tracking unnecessary files
