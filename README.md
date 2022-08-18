# AG_News-NLP-Topic-Classification

### Abstract
Text classification has been one of application areas under Natural Language Processing (NLP) domain where deep learning has achieved great progress and becomes dominant approach in past few years. As one of deep learning model structure, Recursive Neural Network (RNN) and its deviants such as Long Short-Term Memory (LSTM) and Gate Recurrent Units (GRU) have been proved to outperform other models thanks to its advantageous capability capture the sequential correlations. In this research, the AG’s news topic classification dataset that comprises of documents in four topic classes including World, Sports, Business, and Sci/Tech is used for training purposefully structured neural network model variants for classification. The primary objective of this research is to compare the performance of RNN, LSTM, and CNN on text classification based on differentiated model architecture and explore how these factors influence the model performance in terms of both accuracy and time.

### Introduction
Text classification has been one of focused areas under Natural Language Process (NLP) where deep neural network has achieved great success and outperformed traditional machine learning models as it leverages different approach for feature extraction and classifier principles (Otter, et al., 2020). As the variable-length text such as sentences, paragraphs and documents could be represented by vectors with fixed length, neural network models leverage this technique to extract features from text context and and then combine them with the different architectures of neural networks (Liu et al., 2016). The flow of continuous words is then treated as sequence of vectors, that could be fed into the neuron network model for it to learn the correlation. Sequence-based models construct sentence representations from word sequences by taking in account the relationship between successive words (Johnson and Zhang, 2015). Recursive Neural Network model and its deviants such as LSTM and GRU have been widely used for text classification task and achieved promising results.In this study, multiple models have been designed with differentiated architecture to experiment with both text representation and feature extraction layer such as LSTM, Simple RNN and CNN and explore how variants of these structures make impact on the model performance. The dataset employed is from AG News Class dataset, which has been purposefully manipulated and labelled. A snapshot of the dataset is
shown below
![image](https://user-images.githubusercontent.com/43327902/185475300-fb1bc019-f346-4883-baa2-772582d030eb.png)

### Literature review
Please refere to the final report in this repo

### Method

##### Data Collection & Exploration
The dataset used for this study is AG from TensorFlow, which comprises of more than 1 million news articles collected by ComeToMyHead from more than 2000 news sources. The AG's news topic classification dataset spreads across 4 largest classes including World, Sports, Business, and Sci/Tech. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

##### Data Preprocessing and Exploration
Data is pre-loaded from TensorFlow. Since there is no validation dataset from the original source, we split the training data and held 5% for validation, yielding 114,000 instances for training and 6,000 for validation. An exploratory data analysis was conducted to investigate the corpus size, token extraction and distribution.There are total 3,909,695 words in the corpus of 127,600 news articles. The tokens per document basis distribute left skewed with majority of documents having 20 to 40 tokens shown in Figure 2. 
![image](https://user-images.githubusercontent.com/43327902/185475548-09dc31fc-4d45-4cb5-b3bc-703f81861995.png)

##### Model Construction and Training
Given the differentiated purpose this study attempts to achieve, we set up four experiments with each experiment focusing on different study target and experimenting several models. Experiment A concentrates on encoding process by which the input data is generated for following feature extraction. Three variables including vocabulary size (1,000, 2,000, 3,000), customization of vocabulary (non-edited, edited) and output sequence length (default, arbitrary) have been considered to build up exhaustive combinations of encoder configuration, which yields 12 alternative encoders. To make the fair comparison, the feature extraction and classification parts are same across all models using one Bidirectional LSTM layer follow by DNN layers as shown in Figure 3.
![image](https://user-images.githubusercontent.com/43327902/185475629-c1c3ec3d-e86a-45b7-b3df-98effb0e9000.png)
![image](https://user-images.githubusercontent.com/43327902/185475663-41a043ac-54f5-453d-b2b6-341d5e69fc07.png)

Experiment B aims to test out regular RNN structure. The encoder is set same across all models, which is identified from Experiment A as best encoder. Five models are designed with various architectures based on Simple RNN and bidirectional training configuration as illustrated in Figure 4.
![image](https://user-images.githubusercontent.com/43327902/185475727-6c6d6814-72f2-4961-a619-8a34fb0410e3.png)
![image](https://user-images.githubusercontent.com/43327902/185475762-b7c9727a-0c9b-407c-842b-4fde5d70d4d7.png)

Experiment C moves focus to LSTM. Similar to Experiment B, same encoder is utilized across all models followed by various model architectures for feature extraction. What’s different is the functional layer is replaced with LSTM and configured by stacking more layers or applying bidirectional training. Details are shown in Figure 5.
![image](https://user-images.githubusercontent.com/43327902/185475816-74dfa9a7-9998-4b78-a917-9e22e99490f4.png)

For Experiment D, two CNN-architected models are tested out with same encoder configuration in Experiment B and C as shown in Figure 6.
![image](https://user-images.githubusercontent.com/43327902/185475903-a52e22ff-6964-42c3-8e92-8090e0403668.png)

##### Result
Experimental models are evaluated with performance metrics being traced, including accuracy and loss across train, validation and test set. Figure 7 illustrates the results from Experiment A, in which the primary focus is on experimenting with different encoders and their associated impact on model performance with all else set the same.
![image](https://user-images.githubusercontent.com/43327902/185475978-127d18e3-3d7b-490b-abd8-a414e35f9f4b.png)

Model 10 yields highest test accuracy 0.8862, slightly better than others. The observation of results yields findings in two perspectives, - Increasing vocabulary size from 1000 to 3000 has improve model performance by approximately 2%-4% without increasing the overfitting gap, indicating that adding more information into features can help better classify the document topic as shown in Figure 8. Figure

![image](https://user-images.githubusercontent.com/43327902/185476022-736d7f7a-7a73-4a28-974f-b28ab5eca4a5.png)

The model 3 from Experiment B return the best test accuracy 0.8838 as shown in Figure 9, slightly better than other models. Overall, the bidirectional training outperforms one directional training indicated by the performance surplus as Model 2 with 1 layer of bidirectional Simple RNN yield 0.8800 test accuracy better than 0.8762 of Model 1 with one layer Simple RNN, while Model 3 with stacked two layers of Simple RNN achieved 0.8838 outperforming Model 4 with 0.8751 accuracy score, which suggest that bidirectional training can enable the model to better capture the semantic context hence enhancing the classification accuracy.
![image](https://user-images.githubusercontent.com/43327902/185476108-72aac21b-ab3c-462a-bd0e-514bee804f73.png)

Additionally, adding more layers of RNN does not bring performance gain but oppositely cause overfitting problem as indicated by Model 5 performance in Figure 10. There is considerable discrepancy between training and validation accuracy after epoch 3 training. The time of training increase by 3 times as computation complexity increase with more parameters.
![image](https://user-images.githubusercontent.com/43327902/185476167-ac399efb-9a39-4580-90e9-8daa06104548.png)

The model 5 from Experiment C return the best test accuracy 0.8853 as shown in Figure 11, slightly better than other models. Consistent with Experiment B, model with bidirectional LSTM layers outperforms ones with single direction layers based on comparison of first four models. Additionally, increasing units in LSTM layers does not show significant performance gain as we can see the model 5 and achieve less than 0.1% test accuracy gain (0.8853) up from model 4 (0.8851).
![image](https://user-images.githubusercontent.com/43327902/185476215-4f107410-c3fc-4c7d-afbe-b0cfc68a5516.png)

If we conduct a horizontal comparison across Simple RNN, LSTM and CNN, surprisingly the model with 2 stacked CNN yielded highest test accuracy 0.8907 with shortest amount of training time 1,254s followed by model with 2 stacked bidirectional LSTM. However, the LSTM model takes longer to train than the other two models as shown in Figure 13. Figure
![image](https://user-images.githubusercontent.com/43327902/185476282-4376ccda-0057-46bb-b7e8-9c7f4f07e797.png)


### Conclusion
Overall, both CNN and RNN including LSTM delivers promising performance on text classification task in this study. Models trained on bidirectional structured layers across LSTM, Simple RNN outperforms ones with single directional layers. LSTM models on average take longer to train due to computational complexity driven by the size of weights. Mostly the model constructed with two layers of neural network can handle the task quite well while stacking more layers on top does not improve the performance significantly and computing cost effectively. Further investigation is needed to get better understanding on encoder construction and tuning of other hyperparameters such as optimizer and loss function.





