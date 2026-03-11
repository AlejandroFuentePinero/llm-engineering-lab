# --- Paths
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))

# --- Imports
import os
from dotenv import load_dotenv
from huggingface_hub import login
from src.pricer.evaluation.evaluator import Tester
from src.pricer.data_prep.items import Item
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# --- Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.layer1(x))
        output2 = self.relu(self.layer2(output1))
        output3 = self.relu(self.layer3(output2))
        output4 = self.relu(self.layer4(output3))
        output5 = self.relu(self.layer5(output4))
        output6 = self.relu(self.layer6(output5))
        output7 = self.relu(self.layer7(output6))
        output8 = self.layer8(output7)
        return output8


def NN_benchmark(
    batch_size: int = 64,
    verbose: bool = True,
    epochs: int = 2,
    learning_rate: float = 0.001,
    evaluation: bool = True,
    username: str = "ed-donner",
    lite_mode: bool = False,
):

    # --- Auth
    load_dotenv(override=True)
    hf_token = os.environ["HF_TOKEN"]
    login(hf_token, add_to_git_credential=True)

    # --- Load data
    dataset = f"{username}/items_lite" if lite_mode else f"{username}/items_full"
    train, val, test = Item.from_hub(dataset)

    if verbose:
        print(
            f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
        )

    # --- Prepare response and features
    y = np.array([float(item.price) for item in train])
    documents = [item.summary for item in train]

    # --- Vectorise the summary into 5000 features
    np.random.seed(42)
    vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
    X = vectorizer.fit_transform(documents)

    # --- Prepare PyTorch data
    X_train_tensor = torch.FloatTensor(X.toarray())
    y_train_tensor = torch.FloatTensor(y).unsqueeze(1)

    # --- Train-Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.01, random_state=42
    )

    # --- Loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Initialize the model
    input_size = X_train_tensor.shape[1]
    model = NeuralNetwork(input_size)

    if verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params:,}")

    # --- Training the model
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    EPOCHS = epochs

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in tqdm(train_loader):
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_function(val_outputs, y_val)

        if verbose:
            print(
                f"Epoch [{epoch+1} / {EPOCHS}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}"
            )

    def neural_network(item):
        model.eval()
        with torch.no_grad():
            vector = vectorizer.transform([item.summary])
            vector = torch.FloatTensor(vector.toarray())
            result = model(vector)[0].item()
        return max(0, result)

    tester = None
    if evaluation:
        tester = Tester(neural_network, test)
        tester.run()

    predictions = [neural_network(item) for item in test]

    return {
        "model": model,
        "vectorizer": vectorizer,
        "predictor": neural_network,
        "predictions": predictions,
        "test_data": test,
        "tester": tester,
    }


if __name__ == "__main__":
    NN_benchmark()
