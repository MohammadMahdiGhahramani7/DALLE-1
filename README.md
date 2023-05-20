# DALLE-1
Full Implementation of DALLE-1 Text-to-Image generator: https://arxiv.org/abs/2102.12092

* Data:

I used `Flickr 8k` dataset, containing ~8k images, each captioned by 5 different sentences. So, when training the dVAE, you can assume there are ~8k image data at all, while when it comes to the transformer, it will be ~8k * 5 = ~40k data pairs. `utils` directory includes a very small subset of images taken from this dataset (`test` directory), along with `captions.txt` file that contains all captions for the dataset. `create_dataset_img_only.py`, as its name suggests, will create the dataset for the dVAE part, using `test` directory. `create_dataset_img_txt.py` also creates image-text pairs dataset for the transformer part, using both `test` directory and `captions.txt` file. `Example_Dataset_img_only.pkl` and `Example_Dataset.pkl` datasets that will be used to test `Firstpart` and `Secondpart` are created by calling aforementioned `.py` files on `test` directory and `captions.txt` file, respectively. The dataset and its corresponding captions are available in https://www.kaggle.com/datasets/adityajn105/flickr8k.


* FirstPart:

Following the paper flow, in the `FirstPart` directory, we design and train a dVAE to produce image tokens. According to the paper when training the transformer, we freeze this dVAE. That is why I considered dVAE training in a separate directory. `DallEdVAE.py` includes the full implementation of the aforementioned dVAE. `train_dVAE.py` also consists of the training procedure. Importing these files `Main.ipynb` trains the dVAE, capable of generating image tokens of length 32*32=1024. Note that, in this section we assume that the dataset contains images only and without their corresponding caption texts. To test the code, you might use `Example_Dataset_img_only.pkl` dataset.

* SecondPart:

Once the dVAE is trained, it is time to load the image-caption pairs dataset to train the transformer. `DalleDecoder.py` includes the full implementation of the decoder-only transformer, proposed in the paper. `train_transformer.py` also contain the training procedure for the transformer. Importing these files `Main.ipynb` trains the transformer. Note that, in this section, we assume that the dataset contains image-text pairs. Text processing step has also been included in the `Main.ipynb` file. To test the code, you might use `Example_Dataset.pkl` dataset. Although the paper uses BPE encoding for captions, I did not consider this encoding in the code. However, `BPE_Encoding` contains the full implementation of such encoding.

* Generation:

Once both the dVAE and the transformer are trained, we load them to do the generation. `Main.ipynb` file includes all steps required to do so. Note that the downloaded `vocab.pickle` file from `SecondPart` is used in generation to preprocess the input text.

**(I) Important note**: All code comments plus `Main.ipynb` files, have been written according to hyper-parameters proposed by the original paper. Hence, those are not necessarily pointing to the real configuration used in this repository. You might change these parameters when working on your customized dataset.

**(II) Important note**: The dVAE and the transformer, used in the paper, are both a specific kind of the general family of dVAE and transformers. Hence, if you are not familiar with the general structure and purpose of dVAEs and/or transformers, you are highly recommended to firstly review [General_dVAEs](https://github.com/MohammadMahdiGhahramani7/dVAE) and [Transformers](https://github.com/MohammadMahdiGhahramani7/MachineTranslation-Transformer).
