## d-LOOK (deep-Look)

### What is d-LOOK

d-LOOK is an automated way to execute various supervised learning deep learning models including:
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


In `parameters_config.json` file, several variables can be configured

	"epochs": ,
	"batch_size": ,
	"test_size": ,
	"dropout_keep_prob": ,
	"number_of_classes":,
	"dl_network":"",
	"activation_function": "",
	"activation_function_output": "",
	"loss_function": ""



##### Example Usage

```shell
python train.py --dataset [dataset_path] --plot [plotname (.pdf)] --model [modelname (.model)]
```

```shell
Example:  python train.py --dataset /home/antonis/repos/d-LOOK/4hours/classes --model testmodel.model
```
