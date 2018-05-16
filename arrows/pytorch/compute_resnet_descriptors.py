import torch
import resnet
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms

weights_fp = "models/byte_200p/90001.pth.tar"
data_folder = "grey_chips/train"
batch_size = 16
num_classes = 46

#initialize
model = models.resnet50()
# replacing the last layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
#load the pretrained weights
weights = torch.load( weights_fp )['state_dict']
model.load_state_dict( weights )
# outputting the feature vector rather than the class 

model = nn.Sequential(*list(model.children())[:-1])
 
model.train( False )

model.cuda() # move the model to the GPU

transform = transforms.Compose([
        transforms.Resize(197),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_dataset = datasets.ImageFolder(data_folder, transform)

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

for inputs, labels in dataloader:
    inputs = Variable( inputs.cuda() )
    out = model( inputs )
    print( out.cpu().detach().numpy().shape )
    break
    
