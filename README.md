# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Data preprare
-- download the dataset from https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing 

-- make new dir  *'/data/train'*  and  *'/data/val'* to current path

-- put first 3000 images into directory *'/data/train'* 

-- put remian images into directory *'/data/val'* 

## metrics
-- As the network is a regressor based approach, thus I select L1loss, other potential loss is MSELoss, you could set it on train.py

-- the performance is the average loss which can evalaute the pixel level percision 
## Parameter config
-- set your hyparameters in config.py e.g. train_folder, val_folder, epoch, batch size etc.

-- set the temprature parameter T in *config.py*

## Training
--run python train.py embeded the train and validate process

## inference 
--run `$ python inference.py --image 'Your test image path' to get the colorized image
