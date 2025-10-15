import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm 
import logging
from RAG import RAGPostProcessor
import yaml
import torch
import torch.optim as optim

from dataloader import create_loaders

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)
    
api_key = configs['api_key']
os.environ["OPENAI_API_BASE"] = configs['api_base']
os.environ["OPENAI_API_KEY"] = api_key

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./output/runtime.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



def run_experiment():
    """Run GNN experiment based on config parameters"""
    from dataloader import DiGCN, MeanTrainer
    from types import SimpleNamespace
    
    # Get configuration parameters
    data_name = configs['dataset_name']
    data_seed = configs.get('data_seed', 1213)
    alpha = configs.get('gnn_alpha', 1.0)
    beta = configs.get('gnn_beta', 0.0)
    epochs = configs.get('gnn_epochs', 150)
    model_seed = configs.get('model_seed', 0)
    num_layers = configs.get('gnn_nlayer', 1)
    device_id = configs.get('device', 0)
    aggregation = configs.get('aggregation', 'Mean')
    bias = configs.get('gnn_bias', False)
    hidden_dim = configs.get('gnn_nhid', 64)
    lr = configs.get('gnn_lr', 0.1)
    weight_decay = configs.get('weight_decay', 1e-5)
    batch_size = configs.get('batch_size', 64)

    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(
        data_name=data_name, 
        batch_size=batch_size,  
        dense=False,
        data_seed=data_seed
    )

    # Set random seeds
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        
    # Step2: Train model
    model = DiGCN(nfeat=num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay) 
    
    if aggregation == "Mean":
        trainer = MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
        )
    
    epochinfo = []
    abnormal_graph_indices = []

    # Start training
    for epoch in range(epochs+1):
        print(f"\nEpoch {epoch:3d}")
        logger.info(f"Starting epoch {epoch}")
        
        svdd_loss = trainer.train(train_loader=train_loader)
        print(f"SVDD loss: {svdd_loss:.6f}")
        
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        print(f"ROC-AUC: {roc_auc:.6f}")
        logger.info(f"Epoch {epoch}: AP: {ap:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Save epoch information
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss
        epochinfo.append(TEMP)

    # Get best results and save abnormal graph IDs
    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]]) + 1
    best_epoch = epochinfo[best_svdd_idx]
    
    print(f"\nMin SVDD, at epoch {best_svdd_idx}, AP: {best_epoch.ap:.3f}, ROC-AUC: {best_epoch.roc_auc:.3f}")
    print(f"At the end, at epoch {epochs}, AP: {epochinfo[-1].ap:.3f}, ROC-AUC: {epochinfo[-1].roc_auc:.3f}")
    
    # Save abnormal graph IDs
    scores_np = best_epoch.dists.cpu().numpy() if hasattr(best_epoch.dists, 'cpu') else np.array(best_epoch.dists)
    threshold = np.mean(scores_np)
    abnormal_graph_indices = [i for i, s in enumerate(scores_np) if s > threshold]
    
    abnormal_id_path = os.path.join(output_dir, 'gnn_abnormal_graph_ids.txt')
    with open(abnormal_id_path, 'w') as f:
        for idx in abnormal_graph_indices:
            f.write(str(idx) + '\n')
    logger.info(f"Abnormal graph IDs saved to: {abnormal_id_path}")
    
    return abnormal_graph_indices

def main():
    logger.info(configs)    
    anomaly_log_structed_path = f"./data/{configs['dataset_name']}/anomaly_log_structured.csv"
    test_log_structed_path = f"./data/{configs['dataset_name']}/test_log_structured.csv"
    gnn_anomaly_log_structed_path = f"./data/{configs['dataset_name']}/gnn_anomaly_log_structured.csv"

    # Run GNN experiment
    abnormal_graph_indices = run_experiment()
    abnormal_id_path = os.path.join(output_dir, 'gnn_abnormal_graph_ids.txt')

    # Find PodUid and EventTemplate based on abnormal graph IDs
    anomaly_label_path = f"./data/{configs['dataset_name']}/anomaly_label.csv"
    poduids = []
    if os.path.exists(anomaly_label_path):
        df_label = pd.read_csv(anomaly_label_path)
        if 'PodUid' in df_label.columns:
            poduids = list(df_label['PodUid'].dropna().unique())
        else:
            logger.error(f"anomaly_label.csv missing PodUid column")
    else:
        logger.error(f"anomaly_label.csv not found: {anomaly_label_path}")


    raw_log_path = f"./data/{configs['dataset_name']}/Kubelet.log_structured.csv"
    event_rows = []
    if os.path.exists(raw_log_path):
        df_raw = pd.read_csv(raw_log_path)
        if 'PodUid' in df_raw.columns and 'EventTemplate' in df_raw.columns:
            for poduid in poduids:
                matched = df_raw[df_raw['PodUid'] == poduid]
                for _, row in matched.iterrows():
                    event_rows.append({'PodUid': poduid, 'EventTemplate': row['EventTemplate']})
        else:
            logger.error(f"Kubelet.log_structured.csv missing PodUid or EventTemplate columns")
    else:
        logger.error(f"Original log file not found: {raw_log_path}")

    # 4. Save results
    event_save_path = os.path.join(output_dir, 'gnn_abnormal_pod_event.csv')
    if event_rows:
        df_event = pd.DataFrame(event_rows)
        df_event.to_csv(event_save_path, index=False)
        logger.info(f"Abnormal PodUid and EventTemplate saved to: {event_save_path}")
    else:
        logger.warning(f"No EventTemplate found for abnormal PodUids, no CSV file generated.")

    # Call RAG after GNN execution is complete
    if configs['is_rag']:
        # RAG with normal logs
        RagPoster = RAGPostProcessor(configs, train_data_path=anomaly_log_structed_path, logger=logger)
        RagPoster.post_process(test_log_structed_path, test_log_structed_path)
        # RAG with GNN abnormal logs
        if os.path.exists(gnn_anomaly_log_structed_path):
            RagPosterGNN = RAGPostProcessor(configs, train_data_path=gnn_anomaly_log_structed_path, logger=logger)
            RagPosterGNN.post_process(test_log_structed_path, test_log_structed_path)
        else:
            logger.info(f"GNN abnormal log file not found: {gnn_anomaly_log_structed_path}")

if __name__ == "__main__":
    main()