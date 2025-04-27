# Cozy AI Kitchen: Security Advisory Analysis and Clustering

## Overview
"Security Advisory Analysis and Clustering" is a demo featured in an episode of "The Cozy AI Kitchen" from the Microsoft Developer YouTube channel. This project highlights the powerful combination of traditional machine learning techniques and advanced AI models to analyze, cluster, and summarize security advisories. By processing advisories from sources like GitHub, generating embeddings for clustering, and producing actionable summaries, the project demonstrates how blending these approaches can streamline the identification of patterns, vulnerabilities, and mitigation strategies. This workflow showcases how security analysts can leverage both traditional ML algorithms and cutting-edge AI to enhance efficiency, scalability, and contextual insights in security advisory analysis.

## Project Structure
The workspace is organized as follows:

### Key Files and Directories
- **`.env`**: Contains environment variables, including API keys for Azure OpenAI services.
- **`input_folder/advisories.json`**: Raw dataset of security advisories, including metadata such as CVEs, CWEs, and descriptions.
- **`1_data_processing.ipynb`**: Prepares and cleans the raw dataset for further analysis.
- **`2_generate_embeddings.ipynb`**: Generates embeddings for advisories using Azure Open AI.
- **`3_clustering.ipynb`**: Notebook for clustering advisories based on embeddings.
- **`4_generate_summaries.ipynb`**: Uses Azure OpenAI to generate summaries for each cluster of advisories.
- **`output_folder/`**: Stores processed datasets, cluster summaries, and other outputs.

## Workflow
1. **Data Preparation**:
   - The public dataset is downloaded from `https://api.github.com/advisories` and saved locally, containing vulnerability details such as CVEs, CWEs, descriptions, and metadata.
   - The raw advisories dataset (`advisories.json`) is cleaned and preprocessed using `1_data_processing.ipynb`.
   - The cleaned dataset is saved as `output_folder/cleaned_advisories.json`.

2. **Embedding Generation**:
   - Embeddings are generated from advisory descriptions using Azure OpenAI (`2_generate_embeddings.ipynb`), transforming textual data into two dimensional numerical vectors suitable for clustering.
   - Outputs include `output_folder/dataset_with_2d_embeddings.json`.

3. **Clustering**:
   - Traditional unsupervised machine learning algorithms (e.g., K-Means) are applied to group advisories based on their embeddings (`3_clustering.ipynb`), identifying clusters of related vulnerabilities.
   - The optimal number of clusters is determined using techniques such as the **Elbow Method** and **Silhouette Analysis**, which evaluate clustering performance based on metrics like within-cluster variance and cluster cohesion.
   - The clustered dataset is saved as `output_folder/clustered_dataset.json`.

4. **Contextualization with LLM**:
   - Azure OpenAI's language models are utilized (`4_generate_summaries.ipynb`) to analyze each cluster, providing meaningful summaries and actionable insights. These summaries highlight common themes, vulnerabilities, and recommended mitigation strategies.
   - Summaries are saved in `output_folder/cluster_summaries.json`.

## Example Clusters
- **Cluster 0**: SurrealDB Denial-of-Service (DoS) vulnerabilities, highlighting resource exhaustion issues and suggesting specific configuration and upgrade actions.
- **Cluster 1**: Resource management and privilege escalation vulnerabilities across various software systems, emphasizing the importance of software upgrades, input validation, and access control measures.
- **Cluster 2**: Cross-Site Scripting (XSS) and XML External Entity (XXE) vulnerabilities, recommending software updates, input sanitization, and permission management.

## Benefits and Value
- **Efficiency**: Quickly identifies patterns and commonalities among numerous security advisories, significantly reducing manual analysis time.
- **Contextual Insights**: LLM-generated summaries provide actionable context, helping security analysts prioritize and address vulnerabilities effectively.
- **Scalability**: The automated workflow can handle large datasets, making it suitable for continuous monitoring and proactive security management.

## Conclusion
This project demonstrates the value of combining traditional machine learning clustering techniques with large language models to enhance security advisory analysis. By automating the identification and contextualization of vulnerabilities, organizations can rapidly respond to threats, improve their security posture, and efficiently allocate resources.

## Dependencies
- Python 3.x
- Required libraries: `openai`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `dotenv`
- Azure OpenAI API for generating summaries.

## Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the `.env` file with your Azure OpenAI API key and endpoint:
   ```plaintext
   OPENAI_API_KEY=<your_api_key>
   OPENAI_API_BASE=<your_api_base_url>
   ```

## Usage
1. Run `1_data_processing.ipynb` to clean and preprocess the dataset.
2. Execute `2_generate_embeddings.ipynb` to generate embeddings.
3. Use `3_clustering.ipynb` to cluster the advisories.
4. Run `4_generate_summaries.ipynb` to generate summaries for each cluster.

## Outputs
- **Processed Datasets**: Cleaned and clustered datasets for further analysis.
- **Cluster Summaries**: Detailed summaries of each cluster, including actionable recommendations.

## Future Enhancements
- Automate the workflow with a pipeline.
- Integrate additional data sources for advisories.
- Improve clustering and summarization techniques.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Azure OpenAI for providing the summarization capabilities.
- GitHub for the security advisories dataset.