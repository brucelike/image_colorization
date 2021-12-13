import torch
import torch.nn as nn
import basic_model
from unet_model import Unet
import colorize_data
from torch.utils.data import DataLoader
from config import num_epochs,train_folder,val_folder,batch_size, weight_decay, learing_rate
import argparse
import os
class Trainer:
    def __init__(self,model):
        # Define hparams here or load them from a config file
        if model=='basic':
            self.model=basic_model.Net()

        else:
            self.model=Unet()
        self.criterion=nn.L1Loss()
        self.num_epochs=num_epochs
        self.lr=learing_rate
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.use_gpu=torch.cuda.is_available()
        self.model_path=os.path.join('pretrained/',model, 'trained.pth')
    def train(self):
        # dataloaders
        train_dataset = colorize_data.ColorizeData(train_folder)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        # Model
        #model = self.model
        # Loss function to use
        criterion = self.criterion
        if self.use_gpu: 
            self.model=self.model.cuda()
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loss_avg = []
        # train loop
        self.model.train()
        for epoch in range(self.num_epochs):
              train_loss_avg.append(0)
              num_batches = 0
    
              for i, (input_gray,input_rgb) in enumerate(train_dataloader):
                   if self.use_gpu: input_gray, input_rgb = input_gray.cuda(), input_rgb.cuda()
        
                   output_rgb = self.model(input_gray) 
         
                 # loss caclulate
                   loss = criterion(output_rgb , input_rgb)
         
             # backpropagation
                   optimizer.zero_grad()
                   loss.backward()
         
             # one step of the optmizer (using the gradients from backpropagation)
                   optimizer.step()
         
                   train_loss_avg[-1] += loss.item()
                   num_batches += 1
         
                   train_loss_avg[-1] /= num_batches
                   print('Epoch [%d / %d---> %d]  loss error: %f' 
                         % (epoch+1, self.num_epochs,num_batches, train_loss_avg[-1]))
        torch.save(self.model.state_dict(), self.model_path) #save model
        print('model trained')

    def validate(self):
      self.model.eval()
      val_loss_avg, num_batches = 0, 0
      val_dataset = colorize_data.ColorizeData(val_folder)
      val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
      for i, (input_gray,input_rgb) in enumerate(val_dataloader):
        if self.use_gpu: 
            input_gray, input_rgb = input_gray.cuda(), input_rgb.cuda()
        with torch.no_grad():
        # to get the color  components
         output_rgb = self.model(input_gray)
         loss = self.criterion(output_rgb, input_rgb)
         val_loss_avg += loss.item()
         num_batches += 1
         val_loss_avg /= num_batches
         print('average loss: %f' % (val_loss_avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict')
    parser.add_argument('--model', default='basic',type=str, help='which model')
    args = parser.parse_args()
    #model=basic_model.Net()
    trainer=Trainer(args.model)
    trainer.train()
    # #trainer.validate()
#%%




