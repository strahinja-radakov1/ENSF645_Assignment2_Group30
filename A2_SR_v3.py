import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizer
import os
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.utils as vutils


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1️⃣ Data Preprocessing ---

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tokenizer for text
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 24  # Max token length

# Function to extract text from filenames
def extract_text_from_filename(filename):
    text = os.path.splitext(filename)[0]  # Remove extension
    text = text.replace('_', ' ')  # Replace underscores
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Custom dataset combining images and text
class MultimodalDataset(Dataset):
    def __init__(self, image_dir, tokenizer, max_len, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.classes = sorted(os.listdir(image_dir))
        self.label_map = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = []

        # Collect image paths and labels
        for class_name in self.classes:
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    self.samples.append((os.path.join(class_path, file), class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]
        label = self.label_map[class_name]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Process text from filename
        text = extract_text_from_filename(os.path.basename(image_path))
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- 2️⃣ Model Definition ---

# Image feature extractor (ResNet50)
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1") 
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten output (now 2048 features)


# Text feature extractor (DistilBERT)
class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # CLS Token

# Multimodal classifier
class MultimodalClassifier(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, num_classes):
        super(MultimodalClassifier, self).__init__()

        self.image_model = ImageFeatureExtractor()
        self.text_model = TextFeatureExtractor()

        # Combined feature size
        combined_dim = image_feature_dim + text_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)  # Extract image features
        text_features = self.text_model(input_ids, attention_mask)  # Extract text features

        combined_features = torch.cat((image_features, text_features), dim=1)  # Concatenate features
        output = self.classifier(combined_features)  # Final classification
        return output

# --- 3️⃣ Training & Evaluation ---

# Paths to dataset
TRAIN_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = "/work/TALC/enel645_2025w/garbage_data/CVPR_2024_dataset_Test"

# Load datasets
train_dataset = MultimodalDataset(TRAIN_PATH, tokenizer, max_len, transform=image_transform)
val_dataset = MultimodalDataset(VAL_PATH, tokenizer, max_len, transform=image_transform)
test_dataset = MultimodalDataset(TEST_PATH, tokenizer, max_len, transform=image_transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
image_feature_dim = 2048 
text_feature_dim = 768  # DistilBERT CLS token size
num_classes = len(train_dataset.classes)

model = MultimodalClassifier(image_feature_dim, text_feature_dim, num_classes).to(device)

# Parameter Count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

image_params = count_parameters(model.image_model)
text_params = count_parameters(model.text_model)
total_params = count_parameters(model)


# Training parameters
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 10
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Compute training accuracy
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train * 100  # Convert to percentage

    # --- VALIDATION ---
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val * 100  # Convert to percentage

    # Print epoch summary
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    # Save the best model based on validation loss
    if avg_val_loss < best_loss:
        torch.save(model.state_dict(), 'multimodal_model.pth')
        best_loss = avg_val_loss

print("Training Complete!")

# --- TEST ACCURACY CALCULATION ---
model.load_state_dict(torch.load('multimodal_model.pth'))  # Load best model
model.eval()

correct_test = 0
total_test = 0
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images, input_ids, attention_mask)

        _, preds = torch.max(outputs, 1)
        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)

        test_predictions.extend(preds.cpu().numpy())  # Store predictions for confusion matrix
        test_labels.extend(labels.cpu().numpy())

test_accuracy = correct_test / total_test * 100  # Convert to percentage
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Compute confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
class_names = sorted(train_dataset.classes)  # Ensure correct class order

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save image
plt.savefig("confusion_matrix.png")
plt.show()

# Print model parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")

# Function to plot weight histograms
def plot_weight_histograms(model):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:  # Only plot trainable weights
            plt.figure(figsize=(7, 5))
            plt.hist(param.data.cpu().numpy().flatten(), bins=50, alpha=0.7, color='b')
            plt.title(f"Weight Distribution: {name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid()
            plt.savefig(f"weight_histogram_{name.replace('.', '_')}.png")
            plt.show()

plot_weight_histograms(model)

# Visualize first convolutional layer filters
first_conv_weights = model.image_model.feature_extractor[0].weight.data.cpu()

# Normalize for visualization
first_conv_weights = (first_conv_weights - first_conv_weights.min()) / (first_conv_weights.max() - first_conv_weights.min())

# Plot filters
plt.figure(figsize=(8, 8))
grid = vutils.make_grid(first_conv_weights, normalize=True, nrow=8)
plt.imshow(grid.permute(1, 2, 0))
plt.title("First Convolutional Layer Filters")
plt.axis("off")
plt.savefig("convolutional_filters.png")
plt.show()

print(f"🔹 Image Model (ResNet18) Parameters: {image_params:,}")
print(f"🔹 Text Model (DistilBERT) Parameters: {text_params:,}")
print(f"🔹 Total Trainable Parameters: {total_params:,}")