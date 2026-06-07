from py2neo import Graph
import networkx as nx
import neo4j

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
def create_graph():
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            node = graph.nodes.match("Digit", value=int(label)).first()
            if not node:
                graph.create_node("Digit", value=int(label))
            # Add relationships or properties as needed
def create_text_graph(texts):
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        # Process outputs and create nodes/relationships in the graph
        output_vector = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        node = graph.nodes.match("Text", content=text).first()
        if not node:
            graph.create_node("Text", content=text, vector=output_vector.tolist())
        
if __name__ == "__main__":
    create_graph()
    sample_texts = ["Hello world!", "Graph databases are powerful.", "BERT is a great model."]
    create_text_graph(sample_texts)
    

