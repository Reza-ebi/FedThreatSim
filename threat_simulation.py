from client_node import ClientNode
from server_node import ServerNode
import numpy as np

clients = [ClientNode(i, is_malicious=(i==2)) for i in range(5)]  # node 2 is malicious

server = ServerNode(threshold=1.0)
updates = []

print("=== Training round started ===")
for c in clients:
    w = c.train()
    updates.append(w)
    print(f"[Node {c.node_id}] model update: {np.round(w[:4], 2)}...")

print("\n[Server] Aggregating models...")
global_model = server.aggregate_models(updates)

print(f"\n[Server] Global model (first weights): {np.round(global_model[:4], 2)}")i
