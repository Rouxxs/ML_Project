
# ML Project

This is the code for my final project from Machine Learning Course in University.

Our task is make models that can classify Movie genres with input is Movie Poster, Movie Title and User's Rating.


## Setup
To run the code you will need to download the dataset and movie ratings which can be download using the links below, and put it in the data/ folder (make sure to create it)

- Dataset: [Link](https://drive.google.com/uc?id=1rvmwYsRU9IykFPTlsckDNm4RHO1CxcJ3)

- Movie ratings: [Link](https://drive.google.com/uc?id=1rvmwYsRU9IykFPTlsckDNm4RHO1CxcJ3)

Or 
```bash
  gdown 1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD
  gdown 1rvmwYsRU9IykFPTlsckDNm4RHO1CxcJ3
```

You can also download all the model checkpoints that i have trained using this link: [Link](https://drive.google.com/uc?id=1ZtpYrYwulFK6S4PyuPpDAaubX_p8YzI4) (make sure to create a checkpoints folder and put it in that folder)


## Runing

To run this project using this comand

```bash
  python main.py 
```
Some flag that can be use:
```
--mode <single or combined> (default = combined) (select mode, single mean only use 1 type of data, combined mean using all 3 type of data, make sure to choose the --model and --type flag correctly)

--type <poster or ratings or title> (select type of input data, make sure to choose it to match your model)

--model <model name> (using this to select your model)

--lr <learning rate> (select learning rate)

--epochs <num of epoch> (default = 50)
```
Or you can using this Google Colab notebook: [Link](https://colab.research.google.com/drive/1ikhQSlS89NX1PCmXQc_8URpXFIUybFD1?usp=sharing)
## Model names
- Poster:
  - Resnet
  - MultiLabelVGG16
  - CustomModel
  - MultiLabelMobileNetV2
  - MultiLabelDenseNet
  - MultiLabelAlexNet
- Title:
  - BertMultiLabelClassifier
- Ratings:
  - RatingsTestModel
- Combined:
  - CombinedModel
  - CombinedModelWeight
