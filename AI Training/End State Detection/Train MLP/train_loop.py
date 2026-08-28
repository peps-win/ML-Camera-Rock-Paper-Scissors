import torch
import torch.nn as nn
from collections import OrderedDict
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import matplotlib.pyplot as plt
import numpy as np

model = nn.Sequential(OrderedDict([
    ('input', nn.Linear(63, 32)),
    ('activ1', nn.ReLU()),
    ('output', nn.Linear(32, 3))
]))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Open the CSV File
df = pd.read_csv("data.csv")
from dataclasses import fields
from shared.Data.models import hand
from shared.Data.fingerNames import joints # type: ignore

field_names = [f.name for f in fields(hand)]
print("fields(hand) count:", len(field_names))
print("joints count:", len(joints))
print("pinky_tip in field_names?", "pinky_tip" in field_names)
print("pinky_tip's id in joints?", joints)  # eyeball this list, look for whatever pinky_tip's id/name should be

# See which columns have NaNs
nan_cols = df.drop(columns=["label", "filename"]).isna().sum()
print(nan_cols[nan_cols > 0])
# Delete columns that have data that is not needed
X = df.drop(columns=["label", "filename"]).values
# Map the type to a number
label_map = {"rock": 0, "paper": 1, "scissors": 2}
y = df["label"].map(label_map).values

# Split my data into train and test for the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Convert data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

epochs = 1500
losses = []

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()          # clear old gradients
    outputs = model(X_train)       # forward pass -> raw logits
    loss = loss_fn(outputs, y_train)  # compare to true labels
    loss.backward()                # backpropagation -> compute gradients
    optimizer.step()               # update weights

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = torch.argmax(test_outputs, dim=1)   # pick the highest-scoring class
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
    torch.save(model.state_dict(), "rps_model.pth")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()