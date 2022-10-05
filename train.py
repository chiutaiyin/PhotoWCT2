import numpy as np
import tensorflow as tf
from utils.DatasetAPI import mscoco_dataset
import os, sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


mse = tf.keras.losses.MeanSquaredError()
def train(enc_dec, block, input_img):
    enc_feats = []
    skip_feats = [None]

    x = enc_dec.encoder(0, input_img)
    enc_feats.append(x)
    for l in range(1, block+1):
        x = enc_dec.encoder(l, x)
        enc_feats.append(x[0])
        skip_feats.append(x[1])
        x = x[0]

    dec_feats = []
    for l in reversed(range(block+1)):
        x = enc_dec.decoder(l, x, skip=skip_feats[l])
        dec_feats.append(x)

    loss_rec = mse(input_img, dec_feats[-1])

    if block == 0:
        return [loss_rec]

    loss_feat_rec = mse(enc_feats[-2], dec_feats[0])

    x = enc_dec.encoder(0, x)
    for l in range(1, block+1):
        x = enc_dec.encoder(l, x)
        x = x[0]
    loss_percept = mse(enc_feats[-1], x)

    return loss_rec, loss_percept, loss_feat_rec

        
parser = argparse.ArgumentParser()

'''
The VGG encoder is splitted into a series of blocks (enc0, enc1, enc2, enc3) from the input to the bottleneck.
The decoder is also splitted into (dec3, dec2, dec1, dec0) from the bottlenect to the output. 
encN and decN form a block pair. Blockwise training trains the four block pairs from N=0 to N=3.
However, sometimes the training of a certain pair, say N=2, may fail. 
We want to restart the training from N=2 with the former pairs restored from the saved checkpoints.
'''
parser.add_argument('--pair', '-p', type=int, choices=[0, 1, 2, 3], default=0, 
                    help="which pair of an encoder block and a decoder block to begin training from")

'''
By default, the numbers of epochs for the training of the four pairs are 10, 10, 10, and 10 for N=0, 1, 2, and 3, respectively.
If training does not start from N=0, say N=2, then the first two arguments will have no effects.
'''
parser.add_argument('--epochs', '-i', type=int, nargs=4, default=[10, 10, 10, 10],
                    help="the numbers of epochs for training of the four block pairs")

'''
Choose to reproduce the convN1 or reluN1 features of the pretrained VGG in the decoder.
By default, the relu features are chosen here.
'''
parser.add_argument('--feature', '-f', type=str, choices=['relu', 'conv'], default='relu')    
parser.add_argument("--encoder", '-e', type=str, default='./ckpts/ckpts-relu/encoder',
                    help="the path to the pretrained VGG encoder")
parser.add_argument("--saveto", '-s', type=str, default='./saved_decoder/relu/',
                    help="the path to the directory where to save the trained decoder blocks")    

# It is not necessary to use MS-COCO and the image preprocessing specified in the file at './utils/DatasetAPI.py'.
parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017', 
                    help="the path to the training dataset (MSCOCO)")

args = parser.parse_args()


if args.feature == 'relu':
    from utils.model_relu import VggDecoder, VggEncoder
else:
    from utils.model_conv import VggDecoder, VggEncoder
    
class VggEncDec(tf.keras.Model):
    def __init__(self, enc_path):
        super(VggEncDec, self).__init__()
        self.encoder = VggEncoder()
        self.decoder = VggDecoder()
        self.encoder.load_weights(enc_path)        


enc_dec = VggEncDec(args.encoder)
blocks = range(args.pair, 4)
ds = mscoco_dataset(args.dataset)

if args.pair != 0:
    for i in range(0, args.pair):
        ckpt = tf.train.Checkpoint(dec_block = enc_dec.decoder.btnecks[i])
        try:
            ckpt.restore(tf.train.latest_checkpoint(os.path.join(args.saveto, str(i)))).assert_existing_objects_matched()
        except:
            print(f"The decoder block {i} has to be trained before training the decoder block {args.pair}.")
            quit()


for block in blocks:
    lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, dec_block=enc_dec.decoder.btnecks[block])
    manager = tf.train.CheckpointManager(ckpt, f"{os.path.join(args.saveto, str(block))}", max_to_keep=1)
    
    # warm up
    train(enc_dec, block, tf.random.normal([1, 256, 256,3])) 
    weights = enc_dec.decoder.btnecks[block].trainable_weights
    for epoch in range(args.epochs[block]):
        for i, imgs in enumerate(ds):
            with tf.GradientTape() as tape:
                loss = train(enc_dec, block, imgs) 
                _loss = sum(loss)

            grads = tape.gradient(_loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            if (i+1) % 10 == 0:
                to_show = f"Epoch: {epoch+1}, iter: {i+1}, loss: {', '.join(map(lambda x: str(x.numpy()), loss))}"
                print(to_show)
        manager.save()

    if block == 3:
        enc_dec.decoder.save_weights(f"{os.path.join(args.saveto, 'decoder')}")
        

        

        
            