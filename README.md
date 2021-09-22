## d-LOOK (deep-Look)

### What is d-LOOK

d-LOOK is an automated way to execute various supervised deep learning models including:
- VGG16
- VGG19
- MobileNetV2
- ResNet50
- ResNet50V2
- InceptionV3
- Xception
- InceptionResNetV2
- ResNet152V2
- DenseNet201
- NASNetLarge

See more about Keras Models: [Keras Applications](https://keras.io/api/applications/)


Fine tuning...

In `parameters_config.json` file, several variables can be configured:

	"epochs": ,
	"batch_size": ,
	"test_size": ,
	"dropout_keep_prob": ,
	"number_of_classes":,
	"dl_network":"",
	"activation_function": "",
	"activation_function_output": "",
	"loss_function": "",
	"optimizer": "Adam"

`epochs`: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation

`batch_size`: the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need

`test_size`: this parameter decides the size of the data that has to be split as the test dataset

`dropout_keep_prob`: the Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting

`number_of_classes`: the number of classes

`dl_network`: the Keras model (VGG16, ResNet50, etc.)

`activation_function`: the activation function is responsible for transforming the summed weighted input from the node into the activation of the node or output for that input

`activation_function_output`: the activation function of the output layer

`loss_function`: the loss function, a scalar value that we attempt to minimize during our training of the model. 

`optimizer`: Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses


##### Example Usage

###### _General cmd_
```shell
python train.py --d [dataset_path] --m [modelname (.model)] --c [config file]
```

###### _A local example cmd_
```shell
Example:  python train.py --dataset /home/antonis/repos/d-LOOK/test_dataet/4hours/classes --model testmodel.model -c parameters_config.json
```

After the execution ...

