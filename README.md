# DALLE-1
Full Implementation of DALLE-1 Text-to-Image generator

* FirstPart:

Following the paper flow, in the `FirstPart` directory, we design and train a dVAE to produce image tokens. According to the paper when training the transformer, we freeze this dVAE. That is why I considered dVAE training in a separate directory. `DallEdVAE.py` includes the full implementation of the aforementioned dVAE. `train_dVAE.py` also consists of the training procedure. Importing these files Main.ipynb trains the dVAE, capable of generating image tokens of length 32*32=1024. Note that, in this section we assume that the dataset contains images only and without their corresponding caption texts.

* SecondPart:

Once the dVAE is trained, it is time to load the image-caption pair dataset to train the transformer. `DalleDecoder.py` includes the full implementation of the decoder-only transformer, proposed in the paper. `train_transformer.py` also contain the training procedure for the transformer. Importing these files `Main.ipynb` trains the transformer. Note that, in this section, we assume that the dataset contains image-text pairs. Text processing step has also been included in the `Main.ipynb` file.
