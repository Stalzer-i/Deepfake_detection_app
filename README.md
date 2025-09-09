# Deepfake_detection_app
Requirements :
(NOTE : these requirements are only needed if you are running the python scripts, the .exe can be run standalone on any windows machine)
1) CUDA (from nvidia)
2) Supported torch version for your CUDA
3) Supported torchvision for your torch version

How to use :
1) You can choose which CNN to train your sample in order to create your own model.
2) Once a model is trained, it can be saved to whichever whichever location you want to save on your local machine.
3) You can download pretrained models which I have originally trained through around 140K images from Kaggle ().
4) Once youre ready to make predictions, choose an image which you need to be classified :
   i) Choose the CNN model you need classified.
   ii) Choose which model you have trained/downloaded.
5) The model Prediction will be give, with an prediction confidence percentage.
