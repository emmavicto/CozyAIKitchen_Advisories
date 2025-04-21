# AI Kitchen: Security Advisory Analysis and Clustering

## Overview
AI Kitchen is a project designed to analyze, cluster, and summarize security advisories using machine learning and natural language processing techniques. The project processes security advisories from sources like GitHub, generates embeddings for clustering, and produces actionable summaries for each cluster. This workflow aims to assist security analysts in identifying patterns, vulnerabilities, and mitigation strategies efficiently.

## Project Structure
The workspace is organized as follows:


### Key Files and Directories
- **`.env`**: Contains environment variables, including API keys for Azure OpenAI services.
- **`advisories.json`**: Raw dataset of security advisories, including metadata like CVEs, CWEs, and descriptions.
- **`clustering.ipynb`**: Notebook for clustering advisories based on embeddings.
- **`data_processing.ipynb`**: Prepares and cleans the raw dataset for further analysis.
- **`generate_embeddings.ipynb`**: Generates embeddings for advisories using machine learning models.
- **`generate_summaries.ipynb`**: Uses Azure OpenAI to generate summaries for each cluster of advisories.
- **`output_folder/`**: Stores processed datasets, cluster summaries, and other outputs.

## Workflow
1. **Data Preparation**:
   - The public dataset is downloaded from a `https://api.github.com/advisories` and saved locally, containing vulnerability details such as CVEs, CWEs, descriptions, and metadata.
   - The raw advisories dataset (`advisories.json`) is cleaned and preprocessed using `data_processing.ipynb`.
   - The cleaned dataset is saved as `output_folder/cleaned_advisories.json`.

2. **Embedding Generation**:
   - Embeddings are generated from advisory descriptions using Azure OpenAI (`generate_embeddings.ipynb`), transforming textual data into numerical vectors suitable for clustering.
   - `generate_embeddings.ipynb` generates vector embeddings for advisories using text features like summaries and descriptions.
   - Outputs include `dataset_with_2d_embeddings.json`.

3. **Clustering**:
  - Traditional unsupervised machine learning algorithms (e.g., K-Means) are applied to group advisories based on their embeddings (`clustering.ipynb`), identifying clusters of related vulnerabilities.
   - The optimal number of clusters is determined using techniques such as the **Elbow Method** and **Silhouette Analysis**, which evaluate clustering performance based on metrics like within-cluster variance and cluster cohesion.
   - The clustered dataset is saved as `output_folder/clustered_dataset.json`.

4. **Contextualization with LLM**:
   - Azure OpenAI's language models are utilized (`generate_summaries.ipynb`) to analyze each cluster, providing meaningful summaries and actionable insights. These summaries highlight common themes, vulnerabilities, and recommended mitigation strategies.
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
This project demonstrates the value of combining traditional machine learning clustering techniques with advanced language models to enhance security advisory analysis. By automating the identification and contextualization of vulnerabilities, organizations can rapidly respond to threats, improve security posture, and efficiently allocate resources.

## Benefits
- **Efficiency**: Reduces the time and effort required for manual vulnerability analysis.
- **Improved Security Posture**: Enables proactive identification and mitigation of vulnerabilities.
- **Actionable Recommendations**: Provides clear, concise, and actionable guidance for addressing identified security issues.

## Dependencies
- Python 3.x
- Required libraries: `openai`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `dotenv`
- Azure OpenAI API for generating summaries.

## Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Set up the .env file with your Azure OpenAI API key and endpoint
    </vscode_annotation> OPENAI_API_KEY=<your_api_key> OPENAI_API_BASE=<your_api_base_url>


## Usage
1. Run `data_processing.ipynb` to clean and preprocess the dataset.
2. Execute `generate_embeddings.ipynb` to generate embeddings.
3. Use `clustering.ipynb` to cluster the advisories.
4. Run `generate_summaries.ipynb` to generate summaries for each cluster.

## Outputs
- **Processed Datasets**: Cleaned and clustered datasets for further analysis.
- **Cluster Summaries**: Detailed summaries of each cluster, including actionable recommendations.

## Future Enhancements
- Automate the workflow with a pipeline.
- Integrate additional data sources for advisories.
- Improve clustering and summarization techniques.

## License
This dataset is licensed under the MIT License.

## Acknowledgments
- Azure OpenAI for providing the summarization capabilities.
- GitHub for the security advisories dataset.
