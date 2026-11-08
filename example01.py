import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# m = models.mobilenet_v2(
#     weights=models.MobileNet_V2_Weights.DEFAULT
# )
# print(m.features)

def check_image(n_images):
    for i in range(len(full_dataset.displayname_list)):
        class_name = full_dataset.displayname_list[i]

        for j in range(n_images):
            image_name = full_dataset.dataname_list[i][j]
            image = full_dataset.load_image_as_numpy(image_name)

            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_title(f"{class_name}\n{image_name}")
            ax.axis("off")

            fig.savefig(f"./train_e/check/{class_name}_{j}.png", bbox_inches="tight")
            plt.close(fig)

# Step 1: Load and preprocess the dataset
class Dataset:
    def __init__(self,
                 data_dir = './train_e/train_e/',
                 labelname_path = './train_e/oidv7-class-descriptions-boxable.csv', 
                 dataname_path = './train_e/train-annotations-human-imagelabels-boxable.csv',
                 displayname_list = ['Cat', 'Dog'],
                 start_with = None,
                 transform = None):
        
        self.data_dir = data_dir
        self.labelname_path = labelname_path
        self.dataname_path = dataname_path
        self.displayname_list = displayname_list
        self.start_with = start_with
        self.transform = transform
        self.dataname_list_labeled = self.make_dataname_list()
        # self.dataname_list()

    def make_dataname_list(self):

        labelname_csv = np.loadtxt(self.labelname_path,
                                   dtype = {'names' : ('LabelName', 'DisplayName'), 
                                           'formats' : ('U16', 'U32')}, 
                                   delimiter = ',', 
                                   skiprows = 1
        )

        labelname_list = []
        for displayname in self.displayname_list:
            labelname_list.append(labelname_csv['LabelName'][labelname_csv['DisplayName'] == displayname][0])


        dataname_list = [] # jpg file name: [type][sample num]
        with open(self.dataname_path, 'r') as lines:

            for labelname in labelname_list:
                
                filenames = []
                # loop all lines
                lines.seek(0)
                for line in lines:
                    
                    str_image_name, str_source, str_labelname, str_confidence = line.strip().split(",")
                    
                    b_source = str_source == 'verification' or str_source == 'crowdsource-verification'
                    b_labelname = str_labelname == labelname
                    b_confidence = str_confidence == '1'
                    if (self.start_with is not None):
                        b_start_with = str_image_name.startswith(self.start_with)
                    
                    if b_source and b_labelname and b_confidence and b_start_with:
                        filenames.append(str_image_name)
            
                dataname_list.append(filenames)
        self.dataname_list = dataname_list

        dataname_list_labeled = []
        for label, filenames in enumerate(dataname_list):
            for filename in filenames:
                dataname_list_labeled.append((filename, label))

        return dataname_list_labeled

    def load_image_as_numpy(self, image_name):
        image_path = self.data_dir + image_name + '.jpg'
        with Image.open(image_path) as img:
            return np.array(img.convert('RGB'), dtype=np.uint8)
    
    def getnumclasses(self):
        return len(self.displayname_list)
    
    def __len__(self):
        return len(self.dataname_list_labeled)
    
    def __getitem__(self, index):  
        image_name, label = self.dataname_list_labeled[index]
        image = self.load_image_as_numpy(image_name)
        if (self.transform is not None):
            image = self.transform(image)
        # or one-hot encoding 
        # label = torch.nn.functional.one_hot(label, num_classes=4).float()
        # 2 -> [0., 0., 1., 0.]
        return image, label
        
W = 60
H = 60
transform_compose_A = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((H, W), antialias=True)
                                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_compose_B = transforms.Compose([
    transforms.ToTensor(),
    MobileNet_V2_Weights.DEFAULT.transforms()
])
    
full_dataset = Dataset(start_with='e', transform=transform_compose_B)
# print(len(full_dataset))

n = len(full_dataset)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(dataset=full_dataset, 
                                                        lengths=[n_train, n_val, n_test],
                                                        generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define the neural network with Xavier initialization

n_classes = full_dataset.getnumclasses()

# try fully connected
class NeuralNetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*W*H, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(10, n_classes)

        # Apply Xavier Initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten the input image
        x = torch.relu(self.fc1(x))  # First layer with ReLU
        x = torch.relu(self.fc2(x))  # Second layer with ReLU
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x

# try mobilenet_v2
class NeuralNetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ??
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convseq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.BatchNorm2d(num_features = out_channels),
                                     nn.ReLU6())
        self.conv = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm2d(num_features = out_channels)

    def forward(self, x):
        x_ = self.convseq(x)
        x_ = self.conv(x_)
        x_ = self.bn(x_)
        x_ = x_ + x
        return x_

class NeuralNetC(nn.Module):
    def __init__(self):
        super().__init__()
        self.convseq1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                                     nn.BatchNorm2d(num_features = 32),
                                     nn.ReLU6())
        self.residual = Residual(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.convseq2 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                                     nn.BatchNorm2d(num_features = 3),
                                     nn.ReLU6())
        self.convseq3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 10, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                                     nn.BatchNorm2d(num_features = 10),
                                     nn.ReLU6())
        self.pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.relu6 = nn.ReLU6()
        self.fc = nn.Linear(10, n_classes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.convseq1(x)
        x = self.residual(x)
        x = self.convseq2(x)
        
        x = self.convseq1(x)
        x = self.residual(x)
        x = self.convseq2(x)
        
        x = self.convseq1(x)
        x = self.residual(x)
        x = self.convseq2(x)

        x = self.convseq1(x)
        x = self.residual(x)
        x = self.convseq3(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu6(x)
        x = self.fc(x)
        return x


# Step 3: Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetC().to(device)

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Training loop
epochs = 5
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs} starts')
    model.train()
    train_loss_sum = 0.
    train_total = 0.
    batch_index = 0
    for images, labels in train_loader:
        
        print(f' Batch {batch_index}')
        batch_index += 1
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_total += batch_size

    # Val
    model.eval()
    val_loss_sum = 0.
    val_total = 0.
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_total += batch_size

            
    print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss_sum / train_total:.4f}")
    print(f"Epoch {epoch+1}/{epochs}, Val loss: {val_loss_sum / val_total:.4f}")

# Step 4: Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
