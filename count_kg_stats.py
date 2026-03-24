import os

def count_kg_statistics(filepath):
    """
    读取文件并统计实体、关系和三元组的数量。
    假设文件格式为：头实体 \t 关系 \t 尾实体 (Tab或空格分隔)
    """
    entities = set()
    relations = set()
    triple_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 默认使用 split() 可以同时处理制表符 \t 和多余空格
                parts = line.strip().split() 
                
                # 确保是有效的三元组行 (至少包含头、关系、尾)
                if len(parts) >= 3:
                    # 假设格式通常为： head relation tail 或者 head tail relation
                    # 这里假设索引 0 是头实体，索引 1 是关系，索引 2 是尾实体
                    # 如果您的数据集格式不同，请调整这里的索引
                    head = parts[0]
                    relation = parts[1] 
                    tail = parts[2]
                    
                    entities.add(head)
                    entities.add(tail)
                    relations.add(relation)
                    triple_count += 1
                    
        return len(entities), len(relations), triple_count
    except Exception as e:
        return f"读取出错: {e}"

def main():
    base_dir = 'datasets' # 这是您截图中的根文件夹名称
    target_files = ['train.txt', 'valid.txt', 'test.txt', 'support.txt']

    if not os.path.exists(base_dir):
        print(f"找不到文件夹: '{base_dir}'。请确保脚本与 {base_dir} 文件夹在同一目录下。")
        return

    # 获取所有数据集子文件夹并排序
    dataset_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

    # 打印表头
    print(f"{'数据集 (Dataset)':<15} | {'文件 (File)':<12} | {'实体数量 (Entities)':<18} | {'关系数量 (Relations)':<18} | {'三元组数量 (Triples)'}")
    print("-" * 90)

    for dataset in dataset_folders:
        dataset_path = os.path.join(base_dir, dataset)
        
        for file_name in target_files:
            file_path = os.path.join(dataset_path, file_name)
            
            if os.path.exists(file_path):
                result = count_kg_statistics(file_path)
                
                if isinstance(result, tuple):
                    num_entities, num_relations, num_triples = result
                    print(f"{dataset:<15} | {file_name:<12} | {num_entities:<18} | {num_relations:<18} | {num_triples}")
                else:
                    print(f"{dataset:<15} | {file_name:<12} | {result}")
        print("-" * 90)

if __name__ == "__main__":
    main()