import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random




from torch.optim.optimizer import Optimizer

class TensorAdaptiveSGD(Optimizer):
    def __init__(self, params, lr, weight_decay=0.0, adapt_const=0.1):

        defaults = dict(lr=lr, weight_decay=weight_decay, adapt_const=adapt_const)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            adapt_const = group['adapt_const']

            for param in group['params']:
                if param.grad is None:
                    continue
                
                g = param.grad
                
                # TODO 1: Compute mean absolute gradient as a tensor
                mean_abs_g = g.abs().mean()
                

                # TODO 2: Compute adaptive learning rate as a tensor
                lr_t = torch.as_tensor(lr,device=param.device,dtype=param.dtype)
                lr_adaptive = lr_t / (1+adapt_const*mean_abs_g)
         

                # TODO 3: Update the parameter using weighted SGD
                upd = g + weight_decay*param
                param.add_(-lr_adaptive*upd)



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)



# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Load dataset
dataset = datasets.ImageFolder(
    root="images/",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(dataloader))

print("Batch image tensor shape:", images.shape)
print("Batch labels tensor shape:", labels.shape)

# Number of classes
num_classes = len(dataset.classes)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()


        # TODO: define 1×1 conv
        self.b1_c1 = nn.Conv2d(in_channels,32,kernel_size=1,padding=0,stride=1)
        self.b1 = nn.Sequential(
            self.b1_c1
        )
        
       
        # TODO: define 1×1 conv 
        self.b2_c1 = nn.Conv2d(64,32,kernel_size=1,padding=0,stride=1)
        # TODO: define 3×3 conv 
        self.b2_c2 = nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1)
        
        self.b2 = nn.Sequential(
            self.b2_c1,
            self.b2_c2
        )

       
        # TODO: define 1×1 conv
        self.b3_c1 = nn.Conv2d(64,32,kernel_size=1,padding=0,stride=1)
        # TODO: define 5×5 conv 
        self.b3_c2 = nn.Conv2d(32,16,kernel_size=5,padding=2,stride=1)
        
        self.b3 = nn.Sequential(
            self.b3_c1,
            self.b3_c2
        )

        # TODO: define maxpool 
        self.b4_p = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        # TODO: define 1×1 conv
        self.b4_c = nn.Conv2d(64,16,kernel_size=1,padding=0,stride=1) 
        
        self.b4 = nn.Sequential(
            self.b4_p,
            self.b4_c
        )

    
    def forward(self, x):
        # TODO: branch 1 forward
        x1 = self.b1(x)


        # TODO: branch 2 forward
        x2 = self.b2(x)


        # TODO: branch 3 forward
        x3 = self.b3(x)


        # TODO: branch 4 forward
        x4 = self.b4(x)


        # TODO: Final output from inception block , 
        # You can use torch.cat to concatenate along channel dimension
        # Syntax: torch.cat([tensor1, tensor2, ...], dim=channel_dimension)
        
        x = torch.cat([x1,x2,x3,x4],dim=1)


        return x


class MiniInceptionNet(nn.Module):
    def __init__(self):
        super(MiniInceptionNet, self).__init__()


        # TODO: Initial stem convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=8,padding=3,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        )
        

        # TODO: First Inception Block 
        self.inc = InceptionBlock(64)
        

        
        # TODO: Downsampling convolutional layer
        self.downconv = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1,stride=2),
            nn.ReLU()
        )

       

        # TODO: Global average pooling and final FC layer
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128,1)
        
        self.dummy()
        
    def dummy(self):
        x = torch.randn(1,3,224,224)
        print(x.shape)
        x = self.stem(x)
        print(x.shape)
        x = self.inc(x)
        print(x.shape)
        x = self.downconv(x)
        print(x.shape)
        x = self.gap(x)
        print(x.shape)
        x = self.fc(torch.flatten(x,1))
        print(x.shape)
        

    def forward(self, x):
        
        # TODO : Stem convolutional layer
        x = self.stem(x)

      
        # TODO : First Inception Block
        x = self.inc(x)

        # TODO: Downsampling convolutional layer
        x = self.downconv(x)

        # TODO: Global average pooling and final FC layer
        x = self.gap(x)
        x = self.fc(torch.flatten(x,1))

        return x
    


device = torch.device("cpu")

model = MiniInceptionNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = TensorAdaptiveSGD(model.parameters(), lr=0.1, weight_decay=0.01, adapt_const=0.5)

num_epochs = 10

print(f"\nTraining Custom ResNet with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Device: {device}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")