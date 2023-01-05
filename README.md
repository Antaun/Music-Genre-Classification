# Music-Genre-Classification
A system which classifies music into different genres using Machine Learning Techniques.

## Abstract
Music genre classification has been a difficult but promising task in the realm of Music Information Retrieval (MIR). Due to the elusive nature of audio musical data, the
effectiveness of any music genre classification system is dependent on its ability to extract usable and reliable features from audio signals. The performance of earlier timbral
feature-based methods for categorizing the genres of audio music was constrained. We
propose a method for extracting musical pattern characteristics from audio music using
the MFCC technique and classify songs based on their genres using a Convolutional
Neural Network (CNN), a model that is frequently used in tasks involving image information retrieval.

## Introduction
We’ve all listened to music through a music streaming app. However, what is the
logic behind the app’s creation of a personalised playlist for us? Classifying music
based on the genre is a fundamental component of a successful recommendation system. The classification of music genres is an important subject with numerous practical
applications. The ability to instantly classify songs in a playlist or library based on
their genre is an important functionality for any modern music streaming service/platform, as it allows users to listen to songs of their favourite genres rather than having
to search for songs of a particular genre separately. The need for precise meta-data for
database administration and search/storage purposes increases in direct proportion to
the daily growth in music production, particularly on online music streaming platforms
like Soundcloud, Spotify, and Apple Music.

In this paper, I propose a novel approach for automatically retrieving musical pattern
features from audio music the MFCC technique and classifying and predicting the genre
of audio music using a Convolutional Neural Network (CNN), a model widely used in
image information retrieval tasks.

## Methodology
The proposed methodology consists of two major phases:
    
**Feature Extraction:** The process of extracting features from data in order to use them for analysis is known as feature extraction. Each audio signal contains various audio features, but we must extract those that are relevant to the problem at hand. It has been shown that feeding the base model with features extracted from the audio signal rather than just the raw audio input yields noticeably better results.

We needed a way to represent song waveforms concisely for audio processing.
Existing music processing literature directed us to MFCCs as a method of representing waveforms in the time domain as a few frequency domain coefficients.
Since its inception, MFCC has been widely used for audio analysis. It is an efficient and highly informative feature set.

**Genre Prediction:** The second phase brings CNN into the picture for capturing
the extracted features and using the extracted information to classify songs based
on their genre.
Convolutional Neural Networks (CNNs) are neutral networks that are used to
handle multi-dimensional arrays like images. The sole difference between using
a CNN for binary classification and multi-classification problems is the amount
of output classes. An image classifier may be trained using a collection of animal
photos, for example.

## Dataset
The GTZAN dataset, which has been used in various other works in the field of music
genre classification, is the dataset used in our experiment. It includes 1000 30-second
song excerpts with a sampling rate of 22050 Hz and a 16-bit resolution. The songs are
divided into ten genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop,
Reggae, and Rock.

## Preprocessing the Data
Data preprocessing is required before we can train our model. If we want to run a model
on our data, we can’t include text in it. So, before we can run a model, we must first
pre-process this data. The LabelEncoder class is used to convert categorical text data
into numerical data which can be understood by the machine learning model. We also
use the StandardScaler class to standardise the features by eliminating the mean and
scaling to unit variance.

## Convolutional Neural Network (CNN)
Convolutional Neural Network (CNN) design has its roots in the study of biological brain
systems. The mechanism of connections revealed in cat visual neurons is in responsible
for distinguishing differences in the topological structure of items observed.

We used neural networks because they have been shown to be generally successful in
many machine learning problems, particularly image information retrieval tasks. Extensive experiments have demonstrated that CNN has a significant capacity for capturing
topological information in visual objects. Despite its success in vision research, CNN
has few applications in audio analysis.

Our model is made up of six convolution layers. We have used the ReLU activation
function, and cross-entropy loss. The first layer is a flatten layer which converts the
input data into a single linear vector, which is fed to the subsequent hidden layers. The
final layer is implemented with a softmax function which normalizes the outputs and
coverts them into a set of probability values for each of the 10 classes. This was implemented with TensorFlow and Keras.

## Results
The project’s goal was to propose a novel approach for developing a CNN model that
can classify songs based on their genre using extracted pattern features. Based on the
extracted features, this trained model will be able to predict the genre of a song. In terms
of time, efficiency, and the quality of the results obtained, this project was feasible.

The preliminary results showed a test accuracy of 64%. The CNN implementation
appeared to be overfit, as the accuracy for the training data was 96% versus only 64%
for the test data.

Following the initial results, modifications were made to the dataset, where each audio
file, which was 30 seconds long, was split into audio files of 3 seconds each. This
increases the size of the dataset exponentially, and the greater the amount of data used
to train a model, the better will be its performance and accuracy.

After training the model with the modifications mentioned above, the test accuracy for
this work improved to 94%. This is in contrast to the validation accuracy of the work
done by Nikki Pelchat and Craig M Gelowitz which was 67%.
