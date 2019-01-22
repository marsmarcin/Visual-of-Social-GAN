# Visual-of-Social-GAN
Show Visualization Result of Social GAN

1.
make sure you have install Social GAN source code from [Social GAN](https://github.com/agrimgupta92/sgan)

2.
download pretrained models from [models](https://www.dropbox.com/s/h8q5z4axfgzx9eb/models.zip?dl=0)
download datasets from [datasets](https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0)
then your folder is like this

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/01.png)

copy eth_8_model.pt in a new folder like '01'

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/03.png)

3.
download plot_model01.py put it in your folder like this

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/02.png)

4.
input 'python plot_model01.py --model_path models\sgan-models\01'

done!


![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/50.gif)


in version-2 use 'python plot_model02.py --model_path models\sgan-models\01'

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/002.gif)


in version-3 use 'python plot_model03.py --model_path models\sgan-models\01'

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/30000.gif)


you can change line 134 in v3 'interval=500' to see more clearly

![images](https://github.com/marsmarcin/Visual-of-Social-GAN/blob/master/img/4000.gif)


noted that dashed line is groundtruth the dotted line is the predicted line.
