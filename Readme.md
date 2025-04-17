解压UNZIP文件夹，把其中三个文件夹放到主目录。

（1）oo/yolo里面有yolo对应的训练代码，如果要训练将数据集放到oo/datasets里，oo/yolo里的代码作用分别为：

YOLO.py 给定图片文件和权重，生成推理后的图片和标签。

SAHI.py 给定图片文件和权重，生成使用SAHI方法推理后的图片和标签。

YOLO_Video.py 给定视频文件和权重，生成推理后的图片和标签,是YOLO格式的，以及对应视频。

SAHI.py 给定视频文件和权重，生成使用SAHI方法推理后的的图片和标签，是YOLO格式的，以及对应视频。

metrics.py 给定真值和检测结果的YOLO格式txt文件夹，计算指标。

v11_train.py 用于训练模型。

spilt_data.py 用于分割数据集。

训练运行结果放在results文件夹里。

（2）GroundingDINO文件夹

Predict.py 给定图片文件和权重，生成推理后的图片和标签，是YOLO格式的。

Predict_Video.py 给定视频文件和权重，生成推理后的图片和标签，是YOLO格式的，及对应视频。

Predict_One.py 用于实现单张图片推理。

权重存放在weights文件夹，output文件夹和output_video文件夹分别图片推理和视频推理结果。

demo文件夹存放测试图片，与Predict_One.py进行测试。

（3）外部的demo文件夹先留着，暂时没用。