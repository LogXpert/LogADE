# LogADE: Log Anomaly Detection and Explanation

LogADE is a semi-supervised log anomaly detection framework that integrates Graph Neural Networks (GNNs) and Large Language Models (LLMs) to provide accurate anomaly detection along with contextualized log explanations.

## Overview

LogADE leverages both graph neural networks (GNNs) and retrieval-augmented generation (RAG) to detect and explain anomalies in system logs. The system processes log data in two main stages:

1. **GNN Stage**: Multi-head attention DiGCN analyzes graph patterns
   - Loads log graphs with node/edge attributes
   - Trains enhanced MHDiGCN model with attention mechanisms
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
