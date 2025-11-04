### Link to GitHub Classroom for submission: https://github.com/au-nlp/project-milestone-p2-group-3

### Link to colab: https://colab.research.google.com/drive/1cUDtBFkbRPO70_0qv32JL3S1nn0XjT_k?usp=sharing

# medical-research-forecast
The repository is about an AI pipeline to model topics to interpret clinical trials data better based on the "https://huggingface.co/datasets/louisbrulenaudet/clinical-trials".


# <p align=center >Anchored Topic Modeling for Interpretable Clinical Trial Landscapes</p>

Help in unsupervised topic discovery for interpretability. Traditional clustering or topic modeling approaches may generate groups that are coherent in embedding space but difficult for domain experts to understand. The idea focuses on developing an anchored topic model for clinical trials that blends free-text embeddings with structured vocabularies (terms, intervention categories).

We would start by generating embeddings for trial descriptions using transformers. Embeddings wil be clustered with KMeans, KMedioid, DBScan giving latent research topics. Instead of relying only on unsupervised clustering, we introduce **anchors**: *predefined sets of terms from `mesh_terms` and `condition_browse_module`* that guide cluster formation. For instance, if MeSH terms indicate “Oncology,” “Immunotherapy,” or “Neurology,” the model biases embeddings to form topics aligned with these known domains, while allowing discovery of subtopics or emerging directions not yet well captured in controlled vocabularies.

We can experiment with approaches like:

- Constraining BERTopic with anchor terms.
- Post-processing clusters by mapping them to the nearest MeSH categories.
- Training a hybrid representation that integrates structured metadata with embeddings.

The contribution here would be a method for interpretable topic modeling in NLP that balances the richness of embeddings with the domain knowledge of concepts and categories. Evaluation could involve both intrinsic measures (topic coherence, clustering quality) and extrinsic measures (classification accuracy on therapeutic area labels).

This provides not only a structured map of the current research landscape but also a way to validate and enrich domain ontologies like MeSH by discovering “off-ontology” topics.

Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
