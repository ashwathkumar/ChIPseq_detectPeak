# ChIPseq_detectPeak

The exercise is to see if we can use CNN to differentiate betweek true peaks and false peaks in ChIP-seq dataset. I have faced issues with peak callers where a region is called as a peak region but when I take a look at it on the genome browser, it does not really look like a peak. I wanted to experiment to see if I can use an automated pipeline to get rid of those regions. 

In this test, I used VGG16 pre trained weights as a starting point and added top layer for a FC and sigmoid classifier. I trained it on a simple image dataset that I created using `scripts/makeData.py` script. The peaks and not peaks regions were hand picked by me on various criteria for this exercise. I had 2000 training images and 800 validation images. Each set has equal number of peak and non peak images. You can see examples of the images in the `examples/example_images/` folder. 

There is also the trial run I performed on this dataset in a jupyter notebook format at `examples/trial_run.ipynb`. 

The stand alone training and prediction scripts are at `scripts/train.py` and `scripts/predict.py`. 