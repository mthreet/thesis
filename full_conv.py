import os
import numpy as np
from utils import load_dataset
import argparse
import pdb


def main(args):
    from keras.utils import plot_model
    # The directory containing wav files
    in_dir = '/home/mthret/class/thesis/data/mf'
    sr = 16000
    speaker_ids = ['8', '16']
    num_wavs = 50
    num_speakers = 2
    num_samples = 32000
    seg_len = 2048
    wav_data = load_dataset(in_dir, sr, speaker_ids, num_wavs, num_speakers,
                            num_samples)
    x, y = prep_data(wav_data, seg_len)

    conv_model = FullConvSep(x.shape[1:])
    plot_model(conv_model, 'model.png', show_shapes=True,
        show_layer_names=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Train or test
    # Initialize the model weights with provided weights
    if args.weights is not None:
        conv_model.load_weights(args.weights)
    # Train if testing is not specified
    if not args.testing:
        train_FullConvSep(model=conv_model, data=(x,y),
                          args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            exit('You must provide wieghts for the network')

        test_FullConvSep(conv_model, data=(x,y),
                         args=args, fs=sr)


def prep_data(wav_data, seg_len):
    num_examps, num_speakers, num_samples = wav_data.shape
    wav_data = wav_data[:, :, :num_samples - (num_samples % seg_len)]
    num_samples = wav_data.shape[2]
    sub_arrs = []
    for ex in range(num_examps):
        sub_arrs.append(np.split(wav_data[ex], int(num_samples / seg_len),
                                 axis=1))
    arrs = np.array([item for sublist in sub_arrs for item in sublist])
    x = np.expand_dims(arrs[:, -1, :], -1)
    y = np.expand_dims(arrs[:, 0, :], -1)
    return x, y


def FullConvSep(input_shape):
    from keras import layers, models

    x = layers.Input(shape=input_shape, name='input')
    conv1 = layers.Conv1D(filters=128, kernel_size=16,
                          padding='same', activation='tanh', name='conv1')(x)
    pool1 = layers.MaxPooling1D(2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=128, kernel_size=16,
                          padding='same', activation='tanh', name='conv2')(
        pool1)
    pool2 = layers.MaxPooling1D(2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=128, kernel_size=16,
                          padding='same', activation='tanh', name='conv3')(
        pool2)
    up1 = layers.UpSampling1D(2)(conv3)
    conv4 = layers.Conv1D(filters=128, kernel_size=16,
                          padding='same', activation='tanh', name='conv4')(up1)
    up2 = layers.UpSampling1D(2)(conv4)
    conv5 = layers.Conv1D(filters=128, kernel_size=16,
                          padding='same', activation='tanh', name='conv5')(up2)
    out = layers.Conv1D(filters=1, kernel_size=16, padding='same',
                        activation='linear', name='full_conv')(conv5)

    model = models.Model(x, out)
    return model


def train_FullConvSep(model, data, args):
    from keras import callbacks, optimizers

    # Unpack the data
    x_train, y_train_sig = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size,
                               histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir +
                                           '/weights-{epoch:02d}.h5',
                                           monitor='mean_squared_error',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=fftLoss,
                  metrics={'full_conv': 'mse'})

    # Training without data augmentation:
    model.fit(x_train, y_train_sig,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=None,
              callbacks=[log, tb, checkpoint, lr_decay])

    # Save the model
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test_FullConvSep(model, data, args, fs):
    import matplotlib.pyplot as plt
    from librosa.output import write_wav as wavwrite

    x_test, y_test_sig = data
    print('-' * 30 + 'Begin: testing' + '-' * 30)
    x_recon = model.predict(x_test, batch_size=100)
    residual = y_test_sig - x_recon

    img_dir = os.path.join(args.save_dir, 'outputs/images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    wav_dir = os.path.join(args.save_dir, 'outputs/wavs')
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    wavwrite(os.path.join(wav_dir, 'comb_sig.wav'), x_test.flatten(), 44100)
    wavwrite(os.path.join(wav_dir, 'sep_sig.wav'), x_recon.flatten(), 44100)
    wavwrite(os.path.join(wav_dir, 'true_sig.wav'), y_test_sig.flatten(),
                 44100)
    for i in range(x_test.shape[0]):
        # wavwrite(os.path.join(wav_dir, 'comb_sig_' + str(i) + '.wav'),
        #          x_test[i], fs)
        # wavwrite(os.path.join(wav_dir, 'sep_sig_' + str(i) + '.wav'),
        #          x_recon[i], fs)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('Mixed Signal')
        plt.plot(x_test[i])
        plt.subplot(3, 1, 2)
        plt.title('Speaker 1')
        plt.plot(y_test_sig[i])
        plt.subplot(3, 1, 3)
        plt.title('Reconstructed Signal')
        plt.plot(x_recon[i])
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'test_' + str(i) + '.png'))
        plt.close()


def ReLUClip(x, max_value=100):
    from keras import backend as K
    return K.relu(x, max_value=max_value)


def LinClip(x, max_value=100):
    from keras import backend as K
    return K.clip(x, -max_value, max_value)


def pearsonCorrLoss(y_true, y_pred):
    from tensorflow import multiply
    from keras import backend as K
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(multiply(xm, ym))
    r_den = K.sqrt(multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def mseLoss(y_true, y_pred):
    from keras import backend as K
    return K.mean(K.square(y_pred - y_true))


def fftLoss(y_true, y_pred):
    from tensorflow import spectral
    from keras import backend as K
    return K.sum(K.abs(spectral.rfft(y_pred) - spectral.rfft(y_true)))


def wassLoss(y_true, y_pred):
    from keras import backend as K
    return 1 - K.exp(-K.square(y_pred - y_true))


def corrMSEfftLoss(y_true, y_pred):
    from keras import backend as K
    return 10 * pearsonCorrLoss(y_true, y_pred) + 10 * mseLoss(y_true,
                                                               y_pred) + 10 * fftLoss(
        y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=1.0, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    main(args)
