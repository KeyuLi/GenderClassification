# GenderClassification
a project of gender classification and a demo of real-time gender classification.
# Requirement
* Ubuntu or Windows
* Python3
* GPU
# Getting Started
To use the real-time gender classification, run ```realtime.py```
To get the gender classification based on photo, run ```photo.py --input_photo_path --output_photo_path```
To get the gender classification based on video, run ```video.py --input_video_path --output_video_path```
# Details
* The pre-trained model we used in ```realtime.py``` is based on AlexNet, and the model we used in ```photo.py```is based on ResNet, if you want to train your model, you can run the ```genderClassifyRes.py``` 
* In ```realtime.py``` and ```video.py```, we use the mtcnn to detect the face, it can get a great result in dynamic video. And in  
```photo.py```, we use the face detector in Opencv. You can replace it accord to the actual effect.
# Dataset
[You can Download CelebA dataset from here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
# Related Projects
[mtcnn](https://github.com/TropComplique/mtcnn-pytorch).
# Results
* The result of gender classification based on photo:
![image1](https://github.com/KeyuLi/GenderClassification/raw/master/result/02.jpg)
![image2](https://github.com/KeyuLi/GenderClassification/raw/master/result/03.jpg)
![image3](https://github.com/KeyuLi/GenderClassification/raw/master/result/08.jpg)

* The result of gender classification based on video:
![image4](https://github.com/KeyuLi/GenderClassification/raw/master/result/out.gif)

* The result of gender classification based on computer camera in real-time:
![image5](https://github.com/KeyuLi/GenderClassification/raw/master/result/test.gif)


