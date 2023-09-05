nb_cls = 2               # number of classes
inp_key_p = "input_p"    # model input p
inp_key_xs = "input_xs"  # model input xs
tgt_key = "estimated"    # model target


import otbtf

def create_otbtf_dataset(p, xs, labels):
    return otbtf.DatasetFromPatchesImages(
        filenames_dict={
            "p": p,
            "xs": xs,
            "labels": labels
        }
    )


import tensorflow as tf

def dataset_preprocessing_fn(sample):
    return {
        inp_key_p: sample["p"],
        inp_key_xs: sample["xs"],
        tgt_key: tf.one_hot(
            tf.squeeze(tf.cast(sample["labels"], tf.int32), axis=-1),
            depth=nb_cls
        )
    }

def create_dataset(p, xs, labels, batch_size=8):
    otbtf_dataset = create_otbtf_dataset(p, xs, labels)
    return otbtf_dataset.get_tf_dataset(
        batch_size=batch_size,
        preprocessing_fn=dataset_preprocessing_fn,
        targets_keys=[tgt_key]
    )


def conv(inp, depth, name, strides=2):
    conv_op = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=3,
        strides=strides,
        activation="relu",
        padding="same",
        name=name
    )
    return conv_op(inp)


def tconv(inp, depth, name, activation="relu"):
    tconv_op = tf.keras.layers.Conv2DTranspose(
        filters=depth,
        kernel_size=3,
        strides=2,
        activation=activation,
        padding="same",
        name=name
    )
    return tconv_op(inp)


class FCNNModel(otbtf.ModelBase):

    def normalize_inputs(self, inputs):
        return {
            inp_key_p: tf.cast(inputs[inp_key_p], tf.float32) * 0.01,
            inp_key_xs: tf.cast(inputs[inp_key_xs], tf.float32) * 0.01
        }

    def get_outputs(self, normalized_inputs):
        norm_inp_p = normalized_inputs[inp_key_p]
        norm_inp_xs = normalized_inputs[inp_key_xs]

        cv_xs = conv(norm_inp_xs, 32, "convxs", 1)
        cv1 = conv(norm_inp_p, 16, "conv1")
        cv2 = conv(cv1, 32, "conv2") + cv_xs
        cv3 = conv(cv2, 64, "conv3")
        cv4 = conv(cv3, 64, "conv4")
        cv1t = tconv(cv4, 64, "conv1t") + cv3
        cv2t = tconv(cv1t, 32, "conv2t") + cv2
        cv3t = tconv(cv2t, 16, "conv3t") + cv1
        cv4t = tconv(cv3t, nb_cls, "softmax_layer", "softmax")

        argmax_op = otbtf.layers.Argmax(name="argmax_layer")

        return {tgt_key: cv4t, "estimated_labels": argmax_op(cv4t)}


def train(params, ds_train, ds_valid, ds_test=None):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = FCNNModel(dataset_element_spec=ds_train.element_spec)
        metrics = [
            tf.keras.metrics.Precision(class_id=1),
            tf.keras.metrics.Recall(class_id=1)
        ]
        model.compile(
            loss={tgt_key: tf.keras.losses.CategoricalCrossentropy()},
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            metrics={tgt_key: metrics}
        )
        model.summary()
        model.fit(
            ds_train,
            epochs=params.nb_epochs,
            validation_data=ds_valid,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(params.model_dir)]
        )

        if ds_test is not None:
            model.load_weights(params.model_dir)
            values = model.evaluate(ds_test, batch_size=params.batch_size)
            for metric_name, value in zip(model.metrics_names, values):
                print(f"{metric_name}: {value:.4f}")


import argparse

parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("--model_dir", required=True, help="model directory")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--nb_epochs", type=int, default=40)
params = parser.parse_args()
tf.get_logger().setLevel('ERROR')

ds_train = create_dataset(
    ["/data/output/train_p_patches.tif"],
    ["/data/output/train_xs_patches.tif"],
    ["/data/output/train_labels_patches.tif"],
)

ds_valid = create_dataset(
    ["/data/output/valid_p_patches.tif"],
    ["/data/output/valid_xs_patches.tif"],
    ["/data/output/valid_labels_patches.tif"],
)

ds_test = create_dataset(
    ["/data/output/test_p_patches.tif"],
    ["/data/output/test_xs_patches.tif"],
    ["/data/output/test_labels_patches.tif"],
)

train(params, ds_train, ds_valid, ds_test)