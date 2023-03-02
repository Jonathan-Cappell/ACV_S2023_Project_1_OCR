#!/usr/bin/env python
# coding: utf-8

# In[31]:


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
import math
from PIL import Image
import CRNN_ARCHITECTURE_OFFICIAL_PYTORCH.crnn as crnn
import CRNN_ARCHITECTURE_OFFICIAL_PYTORCH.utils as utils
from torch.autograd import Variable
#from warpctc_pytorch import CTCLoss
#https://stackoverflow.com/questions/65996797/how-to-refresh-a-python-import-in-a-jupyter-notebook-cell
from importlib import reload
reload(crnn)
reload(utils)
import CRNN_ARCHITECTURE_OFFICIAL_PYTORCH.crnn as crnn
import CRNN_ARCHITECTURE_OFFICIAL_PYTORCH.utils as utils

cudnn.benchmark = True
plt.ion()   # interactive mode

#IMPORTED SOME STARTER CODE


# In[32]:


#Custom Dataset 
#https://github.com/Deepayan137/Adapting-OCR/blob/f86219f4fa8e9198850e343f13ad0d2cbee3f453/src/data/pickle_dataset.py
# and starter code


# Now we often need to create our own custom PyTorch dataset object.  While it's not
# necessary here, let's do it for fun and we will mimic the same structure as above,
# just to see what it looks like!
class MyImageFolderDataset2(torch.utils.data.Dataset):
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
        #self.labels = np.array(labels)
        #self.ourstrLabelConverter=utils.strLabelConverter('012345678 9abcdefghijklmnopqrstuvwxyz-')
        #self.labels = self.ourstrLabelConverter.encode(labels)
        #self.labels = np.array(labels)
        self.labels=labels
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
        class_name_dirs = glob.glob(images_dir + "\\*_*.jpg")
        #class_name_dirs = glob.glob(images_dir + "\\*")
        class_names = [name.replace(images_dir + "\\", "") for name in class_name_dirs]
        class_names=[name.split("_")[0] for name in class_names]
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
        """
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
        
        """
        #print(images_dir)
        #print(class_names[0:5])
        #print(class_labels[0:5])
        image_files = glob.glob(images_dir + "\\*_*.jpg")
        labels=class_names
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
            #image= Image.open(self.image_files[idx]).convert
            label = self.labels[idx]
            #print("reached here\n")
            # Apply the image transform
            image = self.image_transform(image)
            #print(image)
            #print(label)
            return image, label
        except Exception as exc:  # <--- i know this isn't the best exception handling
            print(exc)
            return None











# In[33]:




# For straightforward datasets, sometimes you can make do with built-in PyTorch dataset objects.
# We want to apply automated data augmentations, which will be different for the training
# and eval scenarios


#https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html#torchvision.transforms.GaussianBlur
#https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html#torchvision.transforms.RandomAffine
#https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html#torchvision.transforms.ColorJitter
#https://pytorch.org/vision/stable/generated/torchvision.transforms.Grayscale.html#torchvision.transforms.Grayscale
#https://pytorch.org/vision/main/generated/torchvision.transforms.functional.rgb_to_grayscale.html 

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomAffine(degrees=30,translate=[.3,.3],shear=30),
        #transforms.RandomResizedCrop(224),
        transforms.ColorJitter(.5,.5,.5),
        transforms.GaussianBlur(kernel_size=3,),
        transforms.Grayscale(1),
        transforms.Resize((32,100),interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([.456],[.226])
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        #transforms.
        transforms.Grayscale(1),
        transforms.Resize((32,100),interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = ''

#the class names here are just the word


# In[34]:


def collate_fn3(batch):
    #print(batch)
    batch = list(filter(lambda x: x is not None, batch))
    #print(batch)
    images = torch.stack([b[0] for b in batch])
    labels= [b[1] for b in batch]
    
    
    
    return images, labels 
    


# In[35]:


#now we will try to implement the CRNN structure
#Starter Code from class

data_dir2='.\\whole_words'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Now let's do the same as above to make sure we are getting the same outputs
image_datasets2 = {x: MyImageFolderDataset2(os.path.join(data_dir2, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
print(image_datasets2)
#print(os.path.join(data_dir, x),data_transforms[x]for x in ['train', 'val'])
print(len(image_datasets2['train']))
print(len(image_datasets2['val']))



dataset_sizes2 = {x: len(image_datasets2[x]) for x in ['train', 'val']}
class_names2 = image_datasets2['train'].classes


# In[36]:


#More Starter Code
dataloaders = {x: torch.utils.data.DataLoader(image_datasets2[x], batch_size=4,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn3)
              for x in ['train', 'val']}



# In[37]:


#More Starter Code
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print(inputs.shape)
#print(classes.shape)
print(classes)
# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)
#print(out.shape)


# In[38]:


#make a custom train function that uses the other loss function (need to import it) 
#make the model?


# In[39]:



model_ft=crnn.CRNN(32, 1, 37, 256,)
print(model_ft)


# Let's set our loss function
#https://github.com/Deepayan137/Adapting-OCR/blob/f86219f4fa8e9198850e343f13ad0d2cbee3f453/src/criterions/ctc.py

criterion = nn.CTCLoss(reduction='mean', zero_infinity=True)
# Setup the optimizer to update the model parameters
ouroptimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(ouroptimizer, step_size=7, gamma=0.1)


# In[44]:


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
                #print(labels)
                inputs = inputs.to(device)
                
                #labels = labels.to(device)
                ourConverter=utils.strLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz-')
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
                    #label_lens=[len(b) for b in labels]
                    #print("The label_lens are ")
                    #print(label_lens)
                    #https://discuss.pytorch.org/t/stacking-a-list-of-tensors-whose-dimensions-are-unequal/31888
                    #torch.cat(data,dim=0)
                    labels2=torch.cat([ourConverter.encode(b)[0] for b in labels],dim=0)
                    label_lens=torch.stack([ourConverter.encode(b)[1] for b in labels])
                    #labels=torch.stack(b for b in labels)
                    #print("")
                    #print(labels)
                    #print("The targets for the loss function are: ")
                    #print(labels2)
                    #labels2=[b[0] for b in labels]
                    #print(torch.stack(labels2).shape)
                    #print(torch)
                    #labels=labels.to(device)
                    #print("The outputs are")
                    #print(outputs)
                    #print(outputs.shape)
                   
                    #https://github.com/meijieru/crnn.pytorch/blob/cdf07cc6d8dce0557e542e6cdc0558bd1ad66b53/train.py#L172
                    #preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                    #cost = criterion(preds, text, preds_size, length) / batch_size
                    #loss_avg.add(cost)

                    #_, preds = preds.max(2)
                    #preds = preds.squeeze(2)
                    #preds = preds.transpose(1, 0).contiguous().view(-1)
                    #sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                    prediction_lens=torch.IntTensor([outputs.size(0)]*4)
                    
                    #print("Prediction lengths is ")
                    #print(prediction_lens)
                    #chat gpt code to figure out what the prediction is
                    #for i in range(outputs.shape[0]):  # Iterate over the time steps
                    #    for j in range(outputs.shape[1]):  # Iterate over the batch size
                    #        predicted_outputs = outputs[i, j, :]  # Extract the predicted output for the current time step and batch index
                    #        # Do something with the predicted output, such as compute the argmax to get the predicted class
                    #        predicted_class = torch.argmax(predicted_outputs)
                    #        predicted_class=ourConverter.decode(predicted_class,prediction_lens)
                    #        print(f"Time step: {i}, Batch index: {j}, Predicted class: {predicted_class}")
                    #https://github.com/Deepayan137/Adapting-OCR/blob/f86219f4fa8e9198850e343f13ad0d2cbee3f453/src/criterions/ctc.py
                    
                    loss = criterion(outputs,labels2,prediction_lens,label_lens)
                    #removing /4.0
                    _, outputs = outputs.max(2)
                    #outputs = outputs.squeeze(2)
                    outputs = outputs.transpose(1, 0).contiguous().view(-1)
                    sim_outputs = ourConverter.decode(outputs.data, prediction_lens.data, raw=False)
                    #print(sim_outputs)
                    #print("Dtypes are: ")
                    #print(outputs.dtype)
                    #print(outputs.dtype)
                    #print(labels2.dtype)
                    #print(prediction_lens.dtype)
                    #print(label_lens.dtype)
                    #outputs.
                    #criterion()
                    #loss = criterion(outputs.to(torch.float32),labels2,prediction_lens,label_lens)/4.0
                    #loss = criterion(outputs,labels,prediction_lens,label_lens)/4.0
                    #loss = criterion(outputs[0], labels,input_lengths=200,target_lengths=50)
                    #CTCLoss()
                    #loss = criterion(outputs[0],one_hot_labels)
                    
                    
                    #print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if math.isnan(loss.item()):
                        print("outputs are ")
                        print(outputs)
                        print("inputs are ")
                        print(inputs)
                        print("labels are ")
                        print(labels)
                print(loss)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                print(running_loss)
                #running_corrects += torch.sum(preds == labels.data)
                #print(sim_outputs)
                #print(labels)
                for i in range(4):
                    running_corrects+= int(sim_outputs[0]==labels[0])
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes2[phase]
            epoch_acc = float(running_corrects) / dataset_sizes2[phase]

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


# In[ ]:





# In[ ]:





# In[45]:


#Chatgpt code to figure out what the output shape is
input_tensor = torch.randn((4,1, 32,100))
for name, layer in model_ft.named_children():
    try:
        input_tensor = layer(input_tensor)
        print(f'{name} output shape is : {input_tensor.shape}')
    except: 
        print("No shape here")


# In[ ]:





# In[46]:


# Train and evaluate.  
# It should take around 10 min on CPU. On GPU, it takes less than a minute.
model_ft = train_model(model_ft, criterion, ouroptimizer, exp_lr_scheduler,
                          num_epochs=10)


# In[ ]:





# In[ ]:




