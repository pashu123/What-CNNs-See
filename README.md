# What-CNN-s-See
Filter Visualizations, Heatmaps and Salience Maps

# Activation Visualization (activation_maximization.py)
Activation visualization show how successive convolution layers transform their input. <br>
<br>
We do this by visualizing intermediate activations, which display the feature maps that are 
output by various convolution and pooling layers in a network, given a certain input. The output
of these layer is called an Activation.
<br><br>
This shows us how an input is decomposed into the different filters learned by the network.
<br>
We are generating an input image that maximizes the filter output activations. Thus we are computing<br><br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{dActivationMaximizationLoss}{dinput}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />
<br>
and using that estimate to update the input.<br>
<br><br>
Activation Maximization loss simply outputs small values for large filter activations{we are minimizing losses during gradient descent iterations}. This allows us to understand what sort of input patterns activate a particular filter.
<br><br>
The best way to conceptualize what your cnn perceives is to visualize the Dense Layer Visualizations.


![15](https://user-images.githubusercontent.com/16246821/58276621-870f7300-7db5-11e9-86eb-23e346464585.png)
![21](https://user-images.githubusercontent.com/16246821/58276635-91317180-7db5-11e9-8815-8de4429c14ca.png)
![1](https://user-images.githubusercontent.com/16246821/58276588-77902a00-7db5-11e9-9486-604b2a6305dd.png)
![2](https://user-images.githubusercontent.com/16246821/58276593-79f28400-7db5-11e9-9dec-0aed7820e18a.png)
![3](https://user-images.githubusercontent.com/16246821/58276597-7c54de00-7db5-11e9-8950-d656d0c3fec9.png)


<p align="center">
  <h3>These were the activation map of some letters, while training Handwritten Alphabets </h3>
</p>
