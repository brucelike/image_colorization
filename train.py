import torch
import torch.nn as nn
import basic_model
import colorize_data
from torch.utils.data import DataLoader
from config import num_epochs,train_folder,val_folder,batch_size, weight_decay, learing_rate

class Trainer:
    def __init__(self):      
        # Define hparams here or load them from a config file
        self.model=basic_model.Net()
        self.criterion=nn.MSELoss()
        self.num_epochs=num_epochs
        self.lr=learing_rate
        self.weight_decay=weight_decay
        self.batch_size=batch_size
    def train(self):
        # dataloaders
        train_dataset = colorize_data.ColorizeData(train_folder)
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        # Model
        #model = self.model
        # Loss function to use
        criterion = self.criterion
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loss_avg = []
        # train loop
        self.model.train()
        for epoch in range(num_epochs):
              train_loss_avg.append(0)
              num_batches = 0
    
              for i, (input_gray,input_rgb) in enumerate(train_dataloader):
        
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
                   print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
        torch.save(self.model.state_dict(), 'trained.pth')


    def validate(self):
      self.model.eval()
      val_loss_avg, num_batches = 0, 0
      val_dataset = colorize_data.ColorizeData(val_folder)
      val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
      for i, (input_gray,input_rgb) in enumerate(val_dataloader):
        with torch.no_grad():
        # to get the color  components
         output_rgb = self.model(input_gray)
         loss = self.criterion(output_rgb, input_rgb)
         val_loss_avg += loss.item()
         num_batches += 1
         val_loss_avg /= num_batches
         print('average loss: %f' % (val_loss_avg))

if __name__ == '__main__':
    #model=basic_model.Net()
    trainer=Trainer()
    trainer.train()
    trainer.validate()
