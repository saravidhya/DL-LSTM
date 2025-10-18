# DL-LSTM

# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.



## DESIGN STEPS
### STEP 1: 
Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.



### STEP 3: 
Define dataset and DataLoader for batching.



### STEP 4: 
Build a bidirectional LSTM model for sequence tagging.



### STEP 5: 
Train the model over multiple epochs, tracking loss



### STEP 6: 
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.





## PROGRAM
## Name: S VIDHIYA LAKSHMI
## REG.NO: 212223230238
```
# Model definition
class BiLSTMTagger(nn.Module):
   
  def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=64):
    super(BiLSTMTagger, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx["ENDPAD"])
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(hidden_dim * 2, tagset_size)


  def forward(self, input_ids):
    x = self.embedding(input_ids)
    x, _ = self.lstm(x)
    x = self.fc(x)
    return x
model = BiLSTMTagger(len(word2idx) + 1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])
optimizer =torch.optim.Adam(model.parameters(), lr=0.001)
# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses
def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != tag2idx["O"]:
                        true_tags.append(idx2tag[labels[i][j].item()])
                        pred_tags.append(idx2tag[preds[i][j].item()])
# Plot loss
history_df = pd.DataFrame({"loss": train_losses, "val_loss": val_losses})
history_df.plot(title="Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```
## OUTPUT
## Loss Vs Epoch Plot
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/678df63f-c406-4d33-8a52-10eea1033156" />

## Sample Text Prediction

<img width="413" height="508" alt="image" src="https://github.com/user-attachments/assets/815b279f-4f42-4065-a527-fe189d49e72e" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
