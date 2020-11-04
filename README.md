# 04-Lab-NeuralNetwork-ImageRecognition

## Methods

- We greyscale the image
- We downscale the image to 256x256
- We then say bibidibobiti boo and magic it works!

## The magic

- Each neuron takes as input a grey-scaled pixel
- Then it goes through the network and gives the output of either fish or not fish

## Curcumventing pitfalls

- Images are randomly augmented by being: Cropped, rotated, etc.
> This is to avoid the model focusing on parts which aren't the fish. 

- Images are also chosen at random to avoid first going through all the fish, then the cats, dogs,...

