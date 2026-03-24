import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict, deque
import random
import os

class KnowledgeGraph:
    """
    Inductive Knowledge Graph Loader
    能够根据 mode 动态加载 support set，并保证全局 ID 一致性。
    """
    def __init__(self, data_dir, mode='train', entity_dict=None, relation_dict=None):
        self.data_dir = data_dir
        self.mode = mode
        
        if entity_dict is None or relation_dict is None:
            print(f"Building vocabulary from all files in {data_dir}...")
            self.entity2id, self.relation2id = self._build_vocab()
        else:
            self.entity2id = entity_dict
            self.relation2id = relation_dict
            
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)
        
        self.adj_list = defaultdict(list)
        self.rev_adj_list = defaultdict(list)
        
        self._load_graph_structure(os.path.join(data_dir, 'train.txt'))
        
        if mode == 'test':
            support_path = os.path.join(data_dir, 'support.txt')
            if os.path.exists(support_path):
                print("    -> [Test Mode] Loaded Support Set into background graph.")
                self._load_graph_structure(support_path, known=False)
        
        target_file = 'train.txt' if mode == 'train' else f'{mode}.txt'
        self.samples = self._load_samples(os.path.join(data_dir, target_file))
        print(f"[{mode}] Graph Edges: {sum(len(v) for v in self.adj_list.values())} | Samples: {len(self.samples)}")

    def _build_vocab(self):
        ents = set()
        rels = set()
        files = ['train.txt', 'valid.txt', 'test.txt', 'support.txt']
        
        for fname in files:
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            ents.add(parts[0])
                            rels.add(parts[1])
                            ents.add(parts[2])
        
        entity2id = {e: i for i, e in enumerate(sorted(list(ents)))}
        relation2id = {r: i for i, r in enumerate(sorted(list(rels)))}
        return entity2id, relation2id

    def _load_graph_structure(self, path, known=True):
        if not os.path.exists(path): return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                
                h_str, r_str, t_str = parts[0], parts[1], parts[2]
                score = float(parts[3]) if len(parts) > 3 else 1.0
                
                h, r, t = self.entity2id[h_str], self.relation2id[r_str], self.entity2id[t_str]
                
                self.adj_list[h].append((r, t, score, known))
                self.rev_adj_list[t].append((r, h, score, known))

    def _load_samples(self, path):
        samples = []
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                h, r, t = self.entity2id[parts[0]], self.relation2id[parts[1]], self.entity2id[parts[2]]
                s = float(parts[3]) if len(parts) > 3 else 1.0
                samples.append((h, r, t, s))
        return samples

    def get_subgraph(self, root_node, max_hops=2, max_nodes=50, direction='backward'):
        nodes = {root_node: 0}
        edges = [] 
        
        queue = deque([(root_node, 0)])
        visited = {root_node}
        
        target_adj = self.rev_adj_list if direction == 'backward' else self.adj_list
        
        while queue:
            u, dist = queue.popleft()
            if dist >= max_hops: continue
            if len(nodes) >= max_nodes: break

            neighbors = target_adj[u]
            if len(neighbors) > 15:
                neighbors = random.sample(neighbors, 15)
            
            for r, v, s, known in neighbors:
                if v not in visited:
                    visited.add(v)
                    nodes[v] = dist + 1
                    queue.append((v, dist + 1))
                
                if v in nodes or u in nodes:
                    if direction == 'backward':
                        edges.append((v, r, u, s, known)) 
                    else:
                        edges.append((u, r, v, s, known)) 
        
        return nodes, edges

class ReasoningDataset(Dataset):
    def __init__(self, kg_graph: KnowledgeGraph, lre_hops=3, sfe_hops=2, max_subgraph_size=64, drop_rate=0.2):
        self.kg = kg_graph
        self.lre_hops = lre_hops
        self.sfe_hops = sfe_hops
        self.max_size = max_subgraph_size
        self.data = kg_graph.samples
        self.n_rels = kg_graph.n_relations
        
        # 【新增】：保存 DropEdge 的丢弃率
        self.drop_rate = drop_rate 

    def _apply_dropedge(self, edges):
        """
        动态 DropEdge 机制：仅在训练集中生效
        """
        if self.drop_rate <= 0.0 or self.kg.mode != 'train':
            return edges
            
        # 如果边数太少（比如少于 2 条），则不进行丢弃，防止图完全断开
        num_edges = len(edges)
        if num_edges <= 2:
            return edges
            
        # 计算保留的边数
        keep_num = max(1, int(num_edges * (1.0 - self.drop_rate)))
        
        # 随机采样保留下来的边
        kept_edges = random.sample(edges, keep_num)
        return kept_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, r, t, label_score = self.data[idx]
        
        # ==========================================
        # 1. SFE Subgraph (以 t 为根节点)
        # ==========================================
        sfe_nodes_dict, sfe_edges_raw = self.kg.get_subgraph(
            t, max_hops=self.sfe_hops, max_nodes=self.max_size, direction='backward'
        )
        
        # 【核心修改】：在构建双向边之前，应用 DropEdge 剔除部分原生边
        sfe_edges_raw = self._apply_dropedge(sfe_edges_raw)
        
        sfe_node_ids = list(sfe_nodes_dict.keys())
        sfe_dists = list(sfe_nodes_dict.values())
        sfe_map = {nid: i for i, nid in enumerate(sfe_node_ids)}
        
        sfe_edge_list = []
        for u, r_id, v, s, known in sfe_edges_raw:
            if u in sfe_map and v in sfe_map:
                u_idx, v_idx = sfe_map[u], sfe_map[v]
                sfe_edge_list.append([u_idx, r_id, v_idx, s, int(known)])
                sfe_edge_list.append([v_idx, r_id + self.n_rels, u_idx, s, int(known)])

        # ==========================================
        # 2. LRE Subgraph (以 h 为根节点)
        # ==========================================
        lre_nodes_dict, lre_edges_raw = self.kg.get_subgraph(
            h, max_hops=self.lre_hops, max_nodes=self.max_size, direction='forward'
        )
        
        # 【核心修改】：同样对 LRE 子图应用 DropEdge
        lre_edges_raw = self._apply_dropedge(lre_edges_raw)
        
        lre_node_ids = list(lre_nodes_dict.keys())
        lre_map = {nid: i for i, nid in enumerate(lre_node_ids)}
        
        lre_edge_list = []
        for u, r_id, v, s, known in lre_edges_raw:
            if u in lre_map and v in lre_map:
                u_idx, v_idx = lre_map[u], lre_map[v]
                lre_edge_list.append([u_idx, r_id, v_idx, s, int(known)])
                lre_edge_list.append([v_idx, r_id + self.n_rels, u_idx, s, int(known)])

        return {
            'h': h, 'r': r, 't': t, 'y': label_score,
            'sfe_nodes': sfe_node_ids, 'sfe_dists': sfe_dists, 'sfe_edges': sfe_edge_list,
            'lre_nodes': lre_node_ids, 'lre_edges': lre_edge_list
        }

def collate_fn(batch):
    h = torch.tensor([b['h'] for b in batch])
    r = torch.tensor([b['r'] for b in batch])
    t = torch.tensor([b['t'] for b in batch])
    y = torch.tensor([b['y'] for b in batch], dtype=torch.float)
    
    def pad_subgraph(batch_data, key_prefix):
        max_n = max([len(b[f'{key_prefix}_nodes']) for b in batch_data])
        max_e = max([len(b[f'{key_prefix}_edges']) for b in batch_data])
        
        max_n = max(max_n, 1) 
        max_e = max(max_e, 1)

        batch_size = len(batch_data)
        
        dists = torch.zeros(batch_size, max_n, dtype=torch.long)
        edge_index = torch.zeros(batch_size, 2, max_e, dtype=torch.long)
        edge_type = torch.zeros(batch_size, max_e, dtype=torch.long)
        edge_score = torch.zeros(batch_size, max_e, dtype=torch.float)
        edge_conf_mask = torch.zeros(batch_size, max_e, dtype=torch.bool)
        node_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
        edge_mask = torch.zeros(batch_size, max_e, dtype=torch.bool) 
        
        for i, b in enumerate(batch_data):
            n_count = len(b[f'{key_prefix}_nodes'])
            if n_count > 0:
                node_mask[i, :n_count] = True
                if f'{key_prefix}_dists' in b:
                    dists[i, :n_count] = torch.tensor(b[f'{key_prefix}_dists'])
            
            edges = b[f'{key_prefix}_edges'] 
            e_count = len(edges)
            if e_count > 0:
                edge_mask[i, :e_count] = True 
                edges_np = np.array(edges)
                edge_index[i, 0, :e_count] = torch.tensor(edges_np[:, 0]) 
                edge_index[i, 1, :e_count] = torch.tensor(edges_np[:, 2]) 
                edge_type[i, :e_count] = torch.tensor(edges_np[:, 1])
                edge_score[i, :e_count] = torch.tensor(edges_np[:, 3])
                if edges_np.shape[1] > 4:
                    edge_conf_mask[i, :e_count] = torch.tensor(edges_np[:, 4]).bool()
                else:
                    edge_conf_mask[i, :e_count] = True
                
        return {
            'dists': dists,
            'edge_index': edge_index, 
            'rels': edge_type,
            'scores': edge_score,
            'mask': node_mask,
            'edge_mask': edge_mask,
            'edge_conf_mask': edge_conf_mask
        }

    sfe_batch = pad_subgraph(batch, 'sfe')
    lre_batch = pad_subgraph(batch, 'lre')
    
    return h, r, t, y, lre_batch, sfe_batch
