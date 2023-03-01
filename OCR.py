#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Starter Code From Class:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import glob 
import itertools
from PIL import Image

cudnn.benchmark = True
plt.ion()   # interactive mode

#IMPORTED SOME STARTER CODE


# In[65]:


#Custom Dataset 
#https://github.com/Deepayan137/Adapting-OCR/blob/f86219f4fa8e9198850e343f13ad0d2cbee3f453/src/data/pickle_dataset.py
# and starter code


# Now we often need to create our own custom PyTorch dataset object.  While it's not
# necessary here, let's do it for fun and we will mimic the same structure as above,
# just to see what it looks like!
class MyImageFolderDataset(torch.utils.data.Dataset):
    """
    The input to this dataset is a directory name.  Each image has its label as part of its name.
        images_dir/
           
            ...
            
      We will support the following image file extensions:
        .jpg, 
    """
    def __init__(
        self,
        images_dir,
        image_transform,
    ):
        self.image_transform = image_transform
        
        # Get a list of all class names inside the images directory
        self.classes = self.get_class_names(images_dir)
        
        # Assign a unique label index to each class name
        self.class_labels = {name: idx for idx, name in enumerate(self.classes)}
        
        # Next, let's collect all image files underneath each class name directory 
        # as a single list of image files.  We need to correspond the class label
        # to each image.
        image_files, labels = self.get_image_filenames_with_labels(
            images_dir,
            self.classes,
            self.class_labels,
        )
        
        # This is a trick to avoid memory leaks over very large datasets.
        self.image_files = np.array(image_files)
        #self.labels = np.array(labels).astype("int")
        self.labels = np.array(labels)
        #self.labels = np.array(labels)
        # How many total images do we need to iterate in this entire dataset?
        self.num_images = len(self.image_files)
        #print(self.labels)
        #print(labels)
    def __len__(self):
        return self.num_images
        
    def get_class_names(self, images_dir):
        """
        Given a directory of images, underneath which we have directories of class
        names, collect all these class names and return as a list of strings.
        """
        #print(images_dir)
        #class_name_dirs = glob.glob(images_dir + "\\*_*.jpg")
        class_name_dirs = glob.glob(images_dir + "\\*")
        class_names = [name.replace(images_dir + "\\", "") for name in class_name_dirs]
        #class_names=[name.split("_")[0] for name in class_names]
        #class_names=glob.glob(images_dir + "*_*.jpgs")
        #print(class_names[0:5])
        #print(class_names)
        return sorted(class_names)  # sort just to keep consistency
    
    def get_image_filenames_with_labels(self, images_dir, class_names, class_labels):
        image_files = []
        labels = []
        #print(images_dir)
        #print(class_names[0:5])
        #print(class_labels[class_names[0]])
        #print(images_dir)
        
        supported_file_types = ["\\*.jpg"]
        
        for name in class_names:
            # Glob all (supported) image file names in this directory
            image_class_dir = os.path.join(images_dir, name)
            
            # Iterate through the supported file types.  For each, glob a list of 
            # all file names with that file extension.  Then combine the entire list
            # into one list using itertools.
            image_class_files = list(itertools.chain.from_iterable(
                [glob.glob(image_class_dir + file_type) for file_type in supported_file_types]
            ))
            #print(image_class_files[0:5])
            # Concatenate the image file names to the overall list and create a label for each
            image_files += image_class_files
            labels += [class_labels[name]] * len(image_class_files)
            #print("the image files are ")
            #print(image_files[0:5])
            #print("the labels are ")
            #print(labels[0:5])
            
        return image_files, labels
        
        #class_name_dirs = glob.glob(images_dir + "\\*_*.jpg")
        #return class_name_dirs, class_names
        
    def __getitem__(self, idx):  
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its ground truth label.  Sometimes
        # we get bad images so check for exceptions opening image files.
        # When this happens, there are various things we can do.  I will
        # choose to return None and handle this in the data collator.
        # You could also just randomly choose another example to return,
        # until success.  It's up to you.
        try:
            # Try to open the image file and convert it to RGB.  This makes
            # this cleaner and consistent assuming the same color type.
            # Even if it's gray-scale, it will replicate the color channels
            # 3 times.
            image = Image.open(self.image_files[idx]).convert('RGB')
            label = self.labels[idx]
            
            # Apply the image transform
            image = self.image_transform(image)
            #print(image)
            #print(label)
            return image, label
        except Exception as exc:  # <--- i know this isn't the best exception handling
            return None











# In[93]:




# For straightforward datasets, sometimes you can make do with built-in PyTorch dataset objects.
# We want to apply automated data augmentations, which will be different for the training
# and eval scenarios


#https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html#torchvision.transforms.GaussianBlur
#https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine
#https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
#https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=30,translate=[.3,.3],shear=30),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(.5,.5,.5),
        transforms.GaussianBlur(kernel_size=3,),
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = ''

#the class names here are just the word


# In[114]:


#Starter Code from class



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Now let's do the same as above to make sure we are getting the same outputs
image_datasets = {x: MyImageFolderDataset(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

#print(os.path.join(data_dir, x),data_transforms[x]for x in ['train', 'val'])
print(len(image_datasets['train']))
print(len(image_datasets['val']))



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


# In[95]:


def collate_fn(batch):
    # Filter failed images first
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    images = torch.stack([b[0] for b in batch])
    #labels = torch.LongTensor([b[1] for b in batch])
    #https://pytorch.org/docs/stable/generated/torch.as_tensor.html
    
    #labels=torch.as_tensor([b[1] for b in batch])
    
    
    #
    
    #code from https://github.com/Deepayan137/Adapting-OCR/blob/master/src/utils/utils.py encode function

    result = []
    maxsize=50
    # item = item.decode('utf-8', 'strict')
    for b in batch:
        newlist=[]
        for char in b[1]:
            #if char in self.dict:
                #index = self.dict[char]
                #self.dict is chars to ints in the alphabet
            #else:
                #index = 0
                
            #result.append(index)
            #https://www.digitalocean.com/community/tutorials/python-ord-chr
            newlist.append(ord(char))
            
        if len(newlist) < 50:
            newlist = newlist + [700] * (maxsize - len(newlist))
        elif len(newlist) > 50:
            newlist = newlist[:maxsize]
        #print(len(newlist))
        #print(newlist[0:5])
        result.append(newlist)
    #text = result
    #return (torch.IntTensor(text), torch.IntTensor(length))
    
    
    labels=result
    #flat_list = [item for sublist in labels for item in sublist]

    #print("images are ")
    #print(images)
    #print("labels are ")
    #print(torch.IntTensor(labels))
    #return images, torch.IntTensor(labels)
    return images, torch.LongTensor(labels)
    #torch.
    #return images, torch.stack(labels)


# In[96]:


def collate_fn2(batch):
    #print(batch)
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    #print(batch)
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    
    return images, labels


# In[97]:


#More Starter Code
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn2)
              for x in ['train', 'val']}



# In[98]:


#More Starter Code
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print(inputs.shape)
print(classes.shape)

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)
#print(out.shape)


# In[99]:


#More starter code 

# Now, let’s write a general function to train a model. 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #print("Input shape is ")
                #print(inputs.shape)
                #print("Label shape is ")
                #print(labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(type(outputs))
                    _, preds = torch.max(outputs, 1)
                    #print(type(labels))
                    #print(torch.max(labels))
                    #one_hot_labels = torch.nn.functional.one_hot(labels,701)
                    #print(outputs[0])
                    #print(labels)
                    loss = criterion(outputs,labels)
                    #loss = criterion(outputs[0], labels,input_lengths=200,target_lengths=50)
                    #CTCLoss()
                    #loss = criterion(outputs[0],one_hot_labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[100]:


class CNNLSTM(nn.Module):
    def __init__(self):  
        super().__init__()
        ourmodel3=nn.Sequential(
        nn.Conv2d(3, 224, kernel_size=3),
        nn.BatchNorm2d(224),
        nn.ReLU(inplace=True),
        nn.Conv2d(224, 224,kernel_size=1,stride=1),
        nn.MaxPool2d(kernel_size=2)
        )
        self.conv=ourmodel3
        ourmodel4=nn.LSTM(224 * 111 * 111,50,batch_first=True)
        #nn.Fold
        ourmodel5=nn.Linear(50,50)
        self.lstm=ourmodel4
        self.linear=ourmodel5
        
    def forward(self, inputs):
        #batch size, channels, height, width
        print(inputs.shape)
    


# In[108]:





ourmodelconv=nn.Conv2d(3,30,3)
#batch size, channels, height, width, 
ourmodellstm=nn.LSTM(ourmodelconv.out_channels,10,10,batch_first=True)
#pass the image to cnn to extract features and then pass them to an LSTM to generate the text
#ourmodel.add_module
#ourmodel=nn.Sequential()
#ourmodel=ourmodellstm(ourmodelconv)

ourmodel2=nn.Sequential(
    nn.Conv2d(3, 224, kernel_size=3),
    nn.BatchNorm2d(224),
    nn.ReLU(inplace=True),
    nn.Conv2d(224, 224,kernel_size=1,stride=1),
    nn.MaxPool2d(kernel_size=2),
   # nn.Flatten(),
    
    #nn.Unfold(kernel_size=111),
    #nn.LSTM(224 * 111 * 111,50,ba|tch_first=True),
    #nn.LSTM(224*111*111,50, batch_first=True),
    #nn.Fold
    #nn.Linear(224*111*111,2)
)
#ourmodel2.
ourmodel2.add_module(nn.Linear(224*111*111,2))
#ourmodel2.extend
#num_ftrs=ourmodel2.fc.in_features
#ourmodel2.fc=nn.Linear(num_ftrs,4)
#(N,L,H_in ),(D* num_layers, N, H_out), (D * num_layers, N, H_cell)
#N batch size, L sequence length, D is 1, H_in is input size, H_cell is hidden_size, H_out is hidden_size
#we want the model to take 4, 3, 224, 224, 
# then we want it to get 4,224, hidden1, hidden2
# after that we want it to get 4, 50
# 4 batches each with a sequence output of 50 exactly
print(ourmodel2)
#ourmodel2.add_module()
#ourmodel.add_module('conv{0}',ourmodelconv)
#ourmodel.add_module('lstm{0}',ourmodellstm)
#ourmodel = ourmodel.to(device)

#ourmodel.add_module()
print("reached here")
#Chatgpt code to figure out what the output shape is
input_tensor = torch.randn((4,3, 224, 224))
for name, layer in ourmodel2.named_children():
    input_tensor = layer(input_tensor)
    try:
        print(f'{name} output shape is : {input_tensor.shape}')
    except: 
        print("No shape here")
input_tensor2 = torch.randn((4,3, 224, 224))
output_tensor2=ourmodel2(input_tensor2)
print(output_tensor2[0].shape)
ourmodel2 =ourmodel2.to(device)


# In[110]:


#using class code


# Load a pretrained model and reset final fully connected layer for this particular classification problem.
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

# Move the model to the correct device (when we have access to a GPU)
model_ft = model_ft.to(device)


# In[112]:


# Let's set our loss function
criterion = nn.CrossEntropyLoss()
#criterion = nn.CTCLoss()
# Setup the optimizer to update the model parameters
ouroptimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(ouroptimizer, step_size=7, gamma=0.1)


# In[ ]:


# Train and evaluate.  
# It should take around 10 min on CPU. On GPU, it takes less than a minute.
model_ft = train_model(model_ft, criterion, ouroptimizer, exp_lr_scheduler,
                       num_epochs=10)


# In[64]:





# In[ ]:





# In[ ]:





# In[ ]:




