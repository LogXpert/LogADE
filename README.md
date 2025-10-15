# LogADE: Log Anomaly Detection and Explanation

LogADE is a semi-supervised log anomaly detection framework that integrates Graph Neural Networks (GNNs) and Large Language Models (LLMs) to provide accurate anomaly detection along with contextualized log explanations.

## Overview

LogADE leverages both graph neural networks (GNNs) and retrieval-augmented generation (RAG) to detect and explain anomalies in system logs. The system processes log data in two main stages:

## Architecture

1. **GNN Stage**: Multi-head attention DiGCN analyzes graph patterns
   - Loads log graphs with node/edge attributes
   - Trains enhanced DiGCN model with attention mechanisms
   - Identifies abnormal graph patterns and saves results

2. **RAG Stage**: LLM analyzes detected anomalies with contextual retrieval
   - Creates vector database from normal log entries
   - Retrieves similar logs for anomaly context
   - Uses LLM to provide detailed anomaly explanations

## Prerequisites

- **Python**: 3.9 or higher
- **GPU**: CUDA-compatible GPU (recommended for training)
- **LLM API**: Access to OpenAI GPT or Qwen API
- **Package Manager**: uv or pip

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd LogRAG/saner_logade

# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd LogRAG/saner_logade

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Configure API Keys**:
   ```bash
   # Edit config.yaml with your LLM API credentials
   nano config.yaml
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```



## Project Structure

```
saner_logade/
├── config.yaml              # Configuration file
├── main.py                  # Main execution script
├── RAG.py                   # RAG post-processor
├── dataloader.py            # Data loading and GNN models
├── digcn.py                 # Multi-head attention DiGCN
├── digcnconv.py             # DiGCN convolution operations
├── prompts.py               # LLM prompt templates
├── data/                    # Dataset directory
├── output/                  # Results and logs
└── requirements.txt
```

## Configuration

The `config.yaml` supports additional parameters for the enhanced implementation:

```yaml
# System configuration
device: cuda:0
output_dir: outputs
is_rag: True                 # Enable RAG post-processing
dataset_name: KUBE           # Kubernetes dataset
regenerate_graphs: false

# API configuration
api_key: your_api_key_here
api_base: your_api_base_here
llm_name: qwen-max           # Options: gpt-3.5-turbo, qwen-max

# Enhanced GNN Model configuration
model: DiGCN                 # Multi-head attention DiGCN
hidden_dim: 128
layers: 2
bias: false
aggregation: Mean

# Training configuration
batch_size: 32
epochs: 150
lr: 0.01
weight_decay: 0.0001
alpha: 1.0
beta: 0.0

# Enhanced RAG parameters
threshold: 0.8               # Similarity threshold (0.5-0.8)
topk: 5                     # Top-k retrieval results
prompt: PROMPT5             # Enhanced prompt template
persist_directory: ./output/ragdb-kube
```

## Key Features

- 🧠 **Multi-Head Attention DiGCN**: Enhanced graph neural network with attention mechanisms
- 🔍 **Integrated RAG Pipeline**: Seamless GNN + LLM integration for anomaly explanation
- ☸️ **Kubernetes Support**: Specialized support for K8s log analysis
- ⚙️ **Flexible Configuration**: YAML-based parameter tuning
- 🗃️ **Vector Database**: Efficient log retrieval with Chroma and Ollama

## Workflow

### Stage 1: GNN Anomaly Detection
1. Load Kubernetes log graphs with node/edge attributes
2. Train multi-head attention DiGCN model
3. Identify abnormal graph patterns
4. Save anomaly IDs and pod mappings

### Stage 2: RAG Post-Processing
1. Create vector database from normal log entries
2. Retrieve contextually similar logs for detected anomalies
3. Use LLM to generate detailed explanations
4. Output structured results with confidence scores

## Output Files

| File | Description |
|------|-------------|
| `gnn_abnormal_graph_ids.txt` | GNN detected anomaly IDs |
| `gnn_abnormal_pod_event.csv` | Pod and event template mappings |
| `anomaly_logs_detc_by_rag.csv` | Final RAG analysis results |
| `llm_answer.json` | Detailed LLM responses |
| `runtime.log` | Execution logs |

## Performance Tuning

### High Accuracy Settings
```yaml
threshold: 0.8
topk: 5
batch_size: 32
hidden_dim: 128
layers: 2
```

### Fast Processing Settings
```yaml
threshold: 0.5
topk: 3
batch_size: 64
hidden_dim: 64
layers: 1
```

## Troubleshooting

### Common Issues

**"No relevant docs were retrieved"**
```yaml
# Lower the similarity threshold
threshold: 0.5
```

**CUDA out of memory**
```yaml
# Use smaller batch size or CPU
batch_size: 16
device: cpu
```

**LLM API timeout**
- Verify API key and endpoint configuration
- Check network connectivity
- Consider using local models for offline processing

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: 8+ cores for efficient data loading
- **RAM**: 16GB+ for large datasets
- **Storage**: SSD recommended for vector database

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.




