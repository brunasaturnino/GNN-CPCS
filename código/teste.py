import os
import json
import random
import time
import torch
import torch.nn.functional as F
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from gcn_model import GCN_Autoencoder

# --- Parâmetros ---
input_dim = 8
hidden_dim = 128
features_cols = ['pacotes', 'interval', 'tempo_ocioso', 'etapas',
                 'mov_mouse', 'repet_comando', 'rastros', 'pulos']

# Caminho para salvar o ip_lookup
IP_LOOKUP_PATH = "data/graph/ip_lookup.json"

# --- Função para carregar e combinar dados de 3 dias ---
def load_training_data(days):
    """
    Carrega os arquivos de features e adjacência para cada dia e os combina.
    As matrizes de adjacência são ajustadas para formar uma matriz block diagonal.
    """
    X_list = []
    adj_list = []
    total_nodes = 0

    for day in days:
        x_path = f'data/auto_embeddings/X_train_day_{day:02d}.pt'
        adj_path = f'data/matrizes/adj_matrix_day_{day:02d}.pt'

        
        if not os.path.exists(x_path) or not os.path.exists(adj_path):
            print(f"Aviso: Arquivos para o dia {day} não foram encontrados.")
            continue

        X = torch.load(x_path)
        A = torch.load(adj_path).coalesce()
        X_list.append(X)

        # Ajusta os índices da matriz A para posicionar os nós no bloco correto
        indices = A._indices().clone()
        values = A._values().clone()
        indices[0] += total_nodes
        indices[1] += total_nodes

        n_nodes = X.shape[0]
        size = total_nodes + n_nodes
        A_shifted = torch.sparse_coo_tensor(indices, values, size=(size, size)).coalesce()
        adj_list.append(A_shifted)
        total_nodes += n_nodes

    if not X_list:
        raise ValueError("Nenhum dado de treinamento foi carregado.")

    X_combined = torch.cat(X_list, dim=0)
    all_indices = []
    all_values = []
    for A in adj_list:
        A = A.coalesce()
        all_indices.append(A._indices())
        all_values.append(A._values())
    all_indices = torch.cat(all_indices, dim=1)
    all_values = torch.cat(all_values)
    adj_combined = torch.sparse_coo_tensor(all_indices, all_values, size=(total_nodes, total_nodes)).coalesce()

    return X_combined, adj_combined

# --- Função para sincronizar o ip_lookup com o grafo ---
def sync_ip_lookup_with_graph(base_features, adj, ip_lookup):
    """
    Garante que o tensor de features e a matriz de adjacência tenham número de linhas
    compatível com os índices presentes no ip_lookup.
    Para cada índice fora do intervalo atual, adiciona uma linha dummy (zeros)
    e expande a matriz de adjacência com zeros.
    """
    current_n = base_features.shape[0]
    if not ip_lookup:
        return base_features, adj
    max_idx = max(ip_lookup.values())
    if max_idx < current_n:
        return base_features, adj
    for i in range(current_n, max_idx + 1):
        dummy = torch.zeros((1, base_features.shape[1]), dtype=base_features.dtype)
        base_features = torch.cat([base_features, dummy], dim=0)
        new_size = i + 1
        indices = adj._indices().clone()
        values = adj._values().clone()
        adj = torch.sparse_coo_tensor(indices, values, size=(new_size, new_size)).coalesce()
        print(f"[SYNC] Expandindo grafo para {new_size} nós.")
    return base_features, adj

# --- Função de treinamento do modelo ---
def train_model(model, X, adj, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, z = model(X, adj)
        loss = F.mse_loss(x_hat, X)  # Exemplo de perda (MSE)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
    model.eval()

# --- Função para fine-tuning incremental ---
def fine_tune_model(model, X, adj, epochs=5, lr=0.001):
    """
    Realiza um fine-tuning incremental do modelo usando o grafo atualizado.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, _ = model(X, adj)
        loss = F.mse_loss(x_hat, X)
        loss.backward()
        optimizer.step()
        print(f"[FINE-TUNE] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    model.eval()

# --- Função para salvar o dicionário ip_lookup ---
def salvar_ip_lookup(ip_lookup, path=IP_LOOKUP_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ip_lookup, f)

# --- Função para expandir o grafo com base no IP ---
def expandir_grafo_com_ip(adj, x_features, base_features, ip_lookup, src_ip, dst_ip, conexoes=3):
    """
    Adiciona uma conexão na matriz de adjacência com base nos IPs.
    Mesmo que o src_ip já exista, verifica se há uma conexão com dst_ip.
    Se src_ip não existir, adiciona-o com as features fornecidas.
    Se dst_ip não existir, fixa-o em "192.168.1.2" e o adiciona com features aleatórias.
    """
    nova_x = base_features

    # Verifica se src_ip já existe
    if src_ip in ip_lookup:
        src_node_id = ip_lookup[src_ip]
        print(f"[✔] {src_ip} já existe com id {src_node_id}.")
    else:
        src_node_id = nova_x.shape[0]
        ip_lookup[src_ip] = src_node_id
        nova_x = torch.cat([nova_x, torch.tensor([x_features], dtype=torch.float32)], dim=0)
        print(f"[NOVO] Adicionando novo nó para src_ip {src_ip} com id {src_node_id}.")

    # Verifica se dst_ip existe; se não, fixa para "192.168.1.2"
    if dst_ip not in ip_lookup:
        print(f"[AVISO] dst_ip {dst_ip} não existe. Fixando dst_ip para 192.168.1.2.")
        dst_ip = "192.168.1.2"
        if dst_ip not in ip_lookup:
            dst_node_id = nova_x.shape[0]
            ip_lookup[dst_ip] = dst_node_id
            random_features = [random.random() for _ in features_cols]
            nova_x = torch.cat([nova_x, torch.tensor([random_features], dtype=torch.float32)], dim=0)
            print(f"[NOVO] Adicionando novo nó para dst_ip {dst_ip} com id {dst_node_id}.")
        else:
            dst_node_id = ip_lookup[dst_ip]
    else:
        dst_node_id = ip_lookup[dst_ip]

    # Atualiza a matriz de adjacência para adicionar a conexão se ela não existir
    indices = adj._indices().clone()
    values = adj._values().clone()
    connection_exists = False
    if indices.numel() > 0:
        for i in range(indices.shape[1]):
            if indices[0, i] == src_node_id and indices[1, i] == dst_node_id:
                connection_exists = True
                break

    if not connection_exists:
        new_edges = torch.tensor([[src_node_id, dst_node_id], [dst_node_id, src_node_id]], dtype=torch.long)
        new_values = torch.tensor([1.0, 1.0], dtype=torch.float32)
        if indices.numel() == 0:
            indices_final = new_edges
            values_final = new_values
        else:
            indices_final = torch.cat([indices, new_edges], dim=1)
            values_final = torch.cat([values, new_values])
        new_size = nova_x.shape[0]
        adj = torch.sparse_coo_tensor(indices_final, values_final, size=(new_size, new_size)).coalesce()
        print(f"[CONEXÃO] Adicionada conexão entre {src_ip} (id {src_node_id}) e {dst_ip} (id {dst_node_id}).")
    else:
        print(f"[CONEXÃO] Conexão entre {src_ip} (id {src_node_id}) e {dst_ip} (id {dst_node_id}) já existe.")

    return nova_x, adj, src_node_id

# --- Função para prever o cluster usando o embedding ---
def prever_cluster_com_embedding(x_features, src_ip, dst_ip):
    global base_features, adj, ip_lookup, model, kmeans
    nova_x, adj_exp, node_id = expandir_grafo_com_ip(adj, x_features, base_features, ip_lookup, src_ip, dst_ip)
    base_features = nova_x     # Atualiza o grafo global
    adj = adj_exp              # Atualiza a matriz de adjacência global
    with torch.no_grad():
        _, z_all = model(base_features, adj)
        z = z_all[node_id].unsqueeze(0)
    pred_cluster = kmeans.predict(z.cpu().numpy())[0]
    print(f"[DEBUG] IP {src_ip} (dst {dst_ip}) classificado como cluster {pred_cluster}.")
    return pred_cluster

# --- Função para tratar um ataque (ex.: tentativa de acesso via telnet) com fine-tuning ---
def handle_attack(src_ip):
    """
    Simula a chegada de um novo nó via ataque.
    Após atualizar o grafo com a nova conexão, realiza fine-tuning incremental na GCN
    e, em seguida, recalcula os embeddings para classificar o novo nó.
    """
    global base_features, adj, ip_lookup, model, kmeans
    dst_ip = "192.168.1.2"
    # Gera features aleatórias para o novo nó
    x_features = [random.random() for _ in features_cols]
    print(f"[ATACANDO] Recebido ataque do IP {src_ip}. Gerando features aleatórias e fixando dst_ip em {dst_ip}.")
    
    # Atualiza o grafo com a nova conexão
    nova_x, adj_exp, node_id = expandir_grafo_com_ip(adj, x_features, base_features, ip_lookup, src_ip, dst_ip)
    base_features = nova_x
    adj = adj_exp

    # Fine-tuning incremental com os dados atualizados
    fine_tune_model(model, base_features, adj, epochs=5, lr=0.001)
    
    # Recalcula os embeddings usando o modelo afinado
    with torch.no_grad():
        _, z_all = model(base_features, adj)
        z = z_all[node_id].unsqueeze(0)
    cluster = kmeans.predict(z.cpu().numpy())[0]
    
    # Salva o ip_lookup atualizado para persistência
    salvar_ip_lookup(ip_lookup)
    print(f"[ALERTA] Ataque do IP {src_ip} classificado no cluster {cluster}.")
    return cluster

# --- Fluxo Principal ---
if __name__ == "__main__":
    # Carrega ou inicializa o dicionário de IPs
    if os.path.exists(IP_LOOKUP_PATH):
        with open(IP_LOOKUP_PATH, "r") as f:
            ip_lookup = json.load(f)
    else:
        print("ip_lookup.json não encontrado. Inicializando dicionário vazio.")
        ip_lookup = {}

    # Define os dias de dados a serem carregados
    dias = [1, 2, 3]
    try:
        base_features, adj = load_training_data(dias)
        print(f"Dados de treinamento carregados: {base_features.shape[0]} nós.")
    except Exception as e:
        print("Erro ao carregar dados de treinamento:", e)
        exit(1)

    # Sincroniza o ip_lookup com o grafo para evitar inconsistências
    base_features, adj = sync_ip_lookup_with_graph(base_features, adj, ip_lookup)

    model = GCN_Autoencoder(input_dim, hidden_dim)

model = GCN_Autoencoder(input_dim, hidden_dim)
GCN_PATH = "data/graph/modelo_gcn_day_03.pt"
KMEANS_PATH = "data/cluster/modelo_kmeans.pkl"

# --- Carrega ou treina os modelos ---
if os.path.exists(GCN_PATH) and os.path.exists(KMEANS_PATH):
    model.load_state_dict(torch.load(GCN_PATH))
    model.eval()
    kmeans = joblib.load(KMEANS_PATH)
    print("[✔] Modelos carregados com sucesso: GCN e KMeans.")
else:
    print("[!] Modelos não encontrados. Treinando do zero...")
    train_model(model, base_features, adj, epochs=200, lr=0.01)
    with torch.no_grad():
        _, z_all = model(base_features, adj)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(z_all.cpu().numpy())
    torch.save(model.state_dict(), GCN_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    print("Modelos treinados e salvos.")

# 🟢 A partir daqui, SEMPRE roda a classificação, mesmo que o modelo tenha sido carregado

# --- Classifica os IPs do CSV ---
CSV_DATA = "data/honeypot_data.csv"
try:
    df = pd.read_csv(CSV_DATA)
except Exception as e:
    print("Erro ao ler o CSV:", e)
    exit(1)

num_linhas = min(6, len(df))
print(f"\nClassificando {num_linhas} linhas do dataset:")
for i, row in df.head(num_linhas).iterrows():
    try:
        src_ip = row["src_ip"]
        dst_ip = row["dst_ip"]
        x_features = [float(row[feat]) for feat in features_cols]
        cluster = prever_cluster_com_embedding(x_features, src_ip, dst_ip)
        print(f"Linha {i}: src_ip={src_ip}, dst_ip={dst_ip}, cluster={cluster}")
    except Exception as e:
        print(f"Erro na linha {i}: {e}")

# --- Simulação de ataques em tempo real ---
ataques = ["192.168.1.17", "10.0.0.101", "10.0.0.102", "10.0.0.100"]
print("\nMonitorando ataques em tempo real...")
for ip in ataques:
    time.sleep(1)
    handle_attack(ip)
