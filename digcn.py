import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, MultiheadAttention

from digcnconv import DIGCNConv

class DiGCN(nn.Module):
    """
    Multi-Head Directed Graph Convolutional Network (DiGCN)
    
    This model processes directed graphs using specialized convolution operations
    that preserve directional information in the message passing, enhanced with
    multi-head attention mechanisms.
    
    Parameters:
    -----------
    nfeat : int
        Number of input node features
    nhid : int
        Number of hidden dimensions
    nlayer : int
        Number of DiGCN layers
    num_heads : int, optional
        Number of attention heads (default: 4)
    dropout : float, optional
        Dropout rate (default: 0.0)
    bias : bool, optional
        Whether to use bias in the convolution layers (default: False)
    """
    def __init__(self, nfeat, nhid, nlayer=2, num_heads=4, dropout=0.0, bias=False, **kwargs):
        super(DiGCN, self).__init__()
        
        self.dropout = dropout
        self.nlayer = nlayer
        self.num_heads = num_heads
        self.nhid = nhid
        
        # Ensure hidden dimension is divisible by number of heads
        assert nhid % num_heads == 0, f"Hidden dimension {nhid} must be divisible by number of heads {num_heads}"
        self.head_dim = nhid // num_heads
        
        # Create multiple layers based on nlayer parameter
        self.convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        # First layer transforms input features to hidden dimension
        self.convs.append(DIGCNConv(nfeat, nhid, bias=bias))
        self.attentions.append(MultiheadAttention(nhid, num_heads, dropout=dropout, batch_first=True))
        
        # Add additional layers if nlayer > 1
        for _ in range(1, nlayer):
            self.convs.append(DIGCNConv(nhid, nhid, bias=bias))
            self.attentions.append(MultiheadAttention(nhid, num_heads, dropout=dropout, batch_first=True))
            
        # Optional batch normalization after each layer
        self.bns = nn.ModuleList([BatchNorm1d(nhid) for _ in range(nlayer)])
        
        # Layer normalization for attention
        self.layer_norms = nn.ModuleList([nn.LayerNorm(nhid) for _ in range(nlayer)])
        
        self.score_layer = nn.Linear(nhid, 1)
        
    def reset_parameters(self):
        """Reset all learnable parameters of the model"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
        self.score_layer.reset_parameters()
        
    def forward(self, data):
        """
        Forward pass through the network with multi-head attention
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Input graph data object containing:
            - x: Node features
            - edge_index: Edge indices
            - edge_attr: Edge attributes
            - batch: Batch indices for nodes
            
        Returns:
        --------
        tuple
            Tuple containing (emb_list, score_list):
            - emb_list: List of node embeddings for each graph in the batch
            - score_list: List of anomaly scores for each graph in the batch
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Process through each layer with multi-head attention
        for i in range(self.nlayer):
            # Graph convolution
            conv_out = self.convs[i](x, edge_index, edge_attr)
            
            # Apply batch normalization
            conv_out = self.bns[i](conv_out)
            
            # Prepare input for multi-head attention
            # Group nodes by graph for attention computation
            graph_embeddings = []
            max_nodes = 0
            
            for g in range(data.num_graphs):
                graph_nodes = conv_out[data.batch == g]
                graph_embeddings.append(graph_nodes)
                max_nodes = max(max_nodes, graph_nodes.size(0))
            
            # Pad sequences and create attention mask
            padded_embeddings = []
            attention_masks = []
            
            for graph_emb in graph_embeddings:
                num_nodes = graph_emb.size(0)
                if num_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - num_nodes, self.nhid, 
                                        device=graph_emb.device, dtype=graph_emb.dtype)
                    padded_emb = torch.cat([graph_emb, padding], dim=0)
                else:
                    padded_emb = graph_emb
                
                padded_embeddings.append(padded_emb)
                
                mask = torch.ones(max_nodes, dtype=torch.bool, device=graph_emb.device)
                mask[:num_nodes] = False
                attention_masks.append(mask)
            
            batched_embeddings = torch.stack(padded_embeddings)  # [batch_size, max_nodes, nhid]
            batched_masks = torch.stack(attention_masks)  # [batch_size, max_nodes]
            
            # Apply multi-head attention
            attended_out, _ = self.attentions[i](
                batched_embeddings, batched_embeddings, batched_embeddings,
                key_padding_mask=batched_masks
            )
            
            # Combine convolution and attention outputs
            # Unpack attended output back to node-level representation
            attended_nodes = []
            for g, graph_emb in enumerate(graph_embeddings):
                num_nodes = graph_emb.size(0)
                attended_nodes.append(attended_out[g, :num_nodes])
            
            attended_x = torch.cat(attended_nodes, dim=0)
            
            # Residual connection and layer normalization
            x = self.layer_norms[i](conv_out + attended_x)
            
            if i < self.nlayer - 1:
                x = F.relu(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        emb_list = []
        score_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch == g]
            emb_list.append(emb)
            node_scores = self.score_layer(emb).squeeze(-1)  # shape: [num_nodes]
            score_list.append(node_scores)
        
        return emb_list, score_list


class DiGCN_IB_Sum(nn.Module):
    """
    Multi-Head DiGCN with Inception Block and Sum aggregation
    
    This model enhances DiGCN with inception blocks that process the graph
    at different granularities and combines the results, enhanced with
    multi-head attention mechanisms.
    
    Parameters:
    -----------
    nfeat : int
        Number of input node features
    nhid : int
        Number of hidden dimensions
    nlayer : int
        Number of DiGCN-IB layers (effectively multiplied by 3 due to inception blocks)
    num_heads : int, optional
        Number of attention heads (default: 4)
    dropout : float, optional
        Dropout rate (default: 0.1)
    bias : bool, optional
        Whether to use bias in the convolution layers (default: False)
    """
    def __init__(self, nfeat, nhid, nlayer=1, num_heads=4, dropout=0.1, bias=False, **kwargs):
        super(DiGCN_IB_Sum, self).__init__()
        
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.nhid = nhid
        self.nlayer = nlayer
        
        # Ensure hidden dimension is divisible by number of heads
        assert nhid % num_heads == 0, f"Hidden dimension {nhid} must be divisible by number of heads {num_heads}"
        
        # Create inception blocks based on nlayer parameter
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # First block transforms input features
        self.blocks.append(InceptionBlock(nfeat, nhid, bias=bias))
        self.attentions.append(MultiheadAttention(nhid, num_heads, dropout=dropout, batch_first=True))
        self.layer_norms.append(nn.LayerNorm(nhid))
        
        # Add additional blocks if nlayer > 1
        for _ in range(1, nlayer):
            self.blocks.append(InceptionBlock(nhid, nhid, bias=bias))
            self.attentions.append(MultiheadAttention(nhid, num_heads, dropout=dropout, batch_first=True))
            self.layer_norms.append(nn.LayerNorm(nhid))
            
    def reset_parameters(self):
        """Reset all learnable parameters of the model"""
        for block in self.blocks:
            block.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
        
    def forward(self, data):
        """
        Forward pass through the network with multi-head attention
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Input graph data object containing:
            - x: Node features
            - edge_index: Edge indices
            - edge_attr: Edge attributes
            - edge_index2: Second-order edge indices
            - edge_attr2: Second-order edge attributes
            - batch: Batch indices for nodes
            
        Returns:
        --------
        list
            List of node embeddings for each graph in the batch
        """
        x = data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr
        edge_index2, edge_attr2 = data.edge_index2, data.edge_attr2
        
        # Process through each inception block with attention
        for i in range(self.nlayer):
            # Get outputs from three branches
            x0, x1, x2 = self.blocks[i](x, edge_index, edge_attr, edge_index2, edge_attr2)
            
            # Apply dropout to each branch
            x0 = F.dropout(x0, p=self.dropout_rate, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
            x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
            
            # Sum the branches
            inception_out = x0 + x1 + x2
            
            # Prepare input for multi-head attention
            # Group nodes by graph for attention computation
            graph_embeddings = []
            max_nodes = 0
            
            for g in range(data.num_graphs):
                graph_nodes = inception_out[data.batch == g]
                graph_embeddings.append(graph_nodes)
                max_nodes = max(max_nodes, graph_nodes.size(0))
            
            # Pad sequences and create attention mask
            padded_embeddings = []
            attention_masks = []
            
            for graph_emb in graph_embeddings:
                num_nodes = graph_emb.size(0)
                if num_nodes < max_nodes:
                    # Pad with zeros
                    padding = torch.zeros(max_nodes - num_nodes, self.nhid, 
                                        device=graph_emb.device, dtype=graph_emb.dtype)
                    padded_emb = torch.cat([graph_emb, padding], dim=0)
                else:
                    padded_emb = graph_emb
                
                padded_embeddings.append(padded_emb)
                
                # Create attention mask (True for padding positions)
                mask = torch.ones(max_nodes, dtype=torch.bool, device=graph_emb.device)
                mask[:num_nodes] = False
                attention_masks.append(mask)
            
            # Stack for batch processing
            batched_embeddings = torch.stack(padded_embeddings)  # [batch_size, max_nodes, nhid]
            batched_masks = torch.stack(attention_masks)  # [batch_size, max_nodes]
            
            # Apply multi-head attention
            attended_out, _ = self.attentions[i](
                batched_embeddings, batched_embeddings, batched_embeddings,
                key_padding_mask=batched_masks
            )
            
            # Combine inception and attention outputs
            # Unpack attended output back to node-level representation
            attended_nodes = []
            for g, graph_emb in enumerate(graph_embeddings):
                num_nodes = graph_emb.size(0)
                attended_nodes.append(attended_out[g, :num_nodes])
            
            attended_x = torch.cat(attended_nodes, dim=0)
            
            # Residual connection and layer normalization
            x = self.layer_norms[i](inception_out + attended_x)
            
            # Apply dropout to combined output
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Organize embeddings by graph
        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch == g]
            emb_list.append(emb)
        
        return emb_list


class InceptionBlock(nn.Module):
    """
    Inception Block for DiGCN
    
    This block processes input features through three parallel paths:
    1. Linear transformation (identity path)
    2. First-order graph convolution
    3. Second-order graph convolution
    
    Parameters:
    -----------
    in_dim : int
        Input feature dimension
    out_dim : int
        Output feature dimension
    bias : bool, optional
        Whether to use bias in the layers (default: False)
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(InceptionBlock, self).__init__()
        
        # Three parallel processing paths
        self.ln = Linear(in_dim, out_dim, bias=bias)
        self.conv1 = DIGCNConv(in_dim, out_dim, bias=bias)
        self.conv2 = DIGCNConv(in_dim, out_dim, bias=bias)
        
    def reset_parameters(self):
        """Reset all learnable parameters of the block"""
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2):
        """
        Forward pass through the inception block
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices for first-order connections
        edge_attr : torch.Tensor
            Edge attributes for first-order connections
        edge_index2 : torch.Tensor
            Edge indices for second-order connections
        edge_attr2 : torch.Tensor
            Edge attributes for second-order connections
            
        Returns:
        --------
        tuple
            Three tensors corresponding to outputs from each branch
        """
        # Identity mapping through linear transformation
        x0 = self.ln(x)
        
        # First-order graph convolution
        x1 = self.conv1(x, edge_index, edge_attr)
        
        # Second-order graph convolution
        x2 = self.conv2(x, edge_index2, edge_attr2)
        
        return x0, x1, x2