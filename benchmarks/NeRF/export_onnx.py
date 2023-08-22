# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import yaml

from tensorflow import keras
from tensorflow.keras import layers

import poprt

sys.path.append('../')
import helper


def compute_psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return -10.0 * np.log(mse) / np.log(10.0)


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def update_config(args):
    if args.config_yaml is None:
        raise ValueError(f"config_yaml is required.")

    print(
        f"Updating args with {args.config_yaml}, "
        f"all duplicated args will be overrided."
    )

    with open(args.config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    for k in cfg.keys():
        if not hasattr(args, k):
            raise ValueError(f"args has no attribute: {k}")
        elif type(getattr(args, k)) != type(cfg[k]):
            raise TypeError(
                f"{k} requires {type(getattr(args, k))}, "
                f"but receives {type(cfg[k])}"
            )
        setattr(args, k, cfg[k])
    return args


# From https://github.com/graphcore/examples/tree/master/vision/neural_image_fields/tensorflow2
# This strategy is used to minimise code
# differences in the GPU/CPU code path:
class NoStrategy:
    def __init__(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return

    def scope(self):
        return self


def cast_rays(origins, directions, t_vals):
    # r(t) = o + td
    rays = origins[..., None, :] + t_vals[..., None] * directions[..., None, :]

    return rays


@tf.function
def broadcast_t_vals(t_vals, origins, n_samples):
    return tf.broadcast_to(t_vals[None, ...], [tf.shape(origins)[0], n_samples])


def sample_along_rays(origins, directions, near, far, n_samples, randomized=False):
    # t in r(t) = o+td
    t_vals = tf.linspace(near, far, n_samples)
    t_vals = tf.cast(t_vals, dtype=origins.dtype)
    # TODO(lsy) implicit broadcast will cause compile error
    t_vals = broadcast_t_vals(t_vals, origins, n_samples)

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = tf.concat([mids, t_vals[..., -1:]], -1)
        lower = tf.concat([t_vals[..., :1], mids], -1)
        t_rand = tf.random.uniform(
            shape=[origins.shape[0], n_samples], dtype=t_vals.dtype
        )
        t_vals = lower + (upper - lower) * t_rand

    rays = cast_rays(origins, directions, t_vals)
    return (rays, t_vals)


@tf.function
def broadcast_u(u, cdf, n_samples):
    return tf.broadcast_to(u, tf.concat([tf.shape(cdf)[:-1], [n_samples]], axis=0))


def piecewise_constant_pdf(bins, weights, n_samples, randomized=False):
    eps = 1e-5
    weight_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)
    padding = tf.maximum(0.0, eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = tf.minimum(1.0, tf.cumsum(pdf[..., :-1], axis=-1))
    cdf = tf.concat(
        [
            tf.zeros(tf.concat([tf.shape(cdf)[:-1], [1]], axis=0)),
            cdf,
            tf.ones(tf.concat([tf.shape(cdf)[:-1], [1]], axis=0)),
        ],
        axis=-1,
    )

    if randomized:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [n_samples])
    else:
        # Match the behavior of random.uniform() by spanning [0, 1-eps].
        u = tf.linspace(0.0, 1.0 - np.finfo('float32').eps, n_samples)
        u = broadcast_u(u, cdf, n_samples)

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        x0 = tf.reduce_max(tf.where(mask, x[..., None], x[..., :1, None]), -2)
        x1 = tf.reduce_min(tf.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = (u - cdf_g0) / (cdf_g1 - cdf_g0)
    t = tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
    t = tf.clip_by_value(t, clip_value_min=0.0, clip_value_max=1.0)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    return tf.stop_gradient(samples)


def sample_pdf(bins, weights, origins, directions, t_vals, n_samples, randomized):
    t_samples = piecewise_constant_pdf(bins, weights, n_samples, randomized)
    t_vals = tf.sort(tf.concat([t_vals, t_samples], axis=-1), axis=-1)
    coords = cast_rays(origins, directions, t_vals)
    return t_vals, coords


def encode_position(x, min_deg, max_deg, fp16_sin=False):
    if min_deg == max_deg:
        return x

    scales = tf.constant([2**i for i in range(min_deg, max_deg)], dtype=x.dtype)

    xb = tf.reshape(
        (x[..., None, :] * scales[:, None]), tf.concat([tf.shape(x)[:-1], [-1]], axis=0)
    )
    if fp16_sin:
        xb = tf.concat([xb, xb + 0.5 * np.pi], axis=-1)
        four_feat = tf.cast(tf.sin(tf.cast(xb, dtype=tf.float16)), dtype=x.dtype)
    else:
        four_feat = tf.sin(tf.concat([xb, xb + 0.5 * np.pi], axis=-1))

    return tf.concat([x] + [four_feat], axis=-1)


def volumetric_rendering(rgb, sigma, t_vals, dirs, white_bkgd=False):
    dists = tf.concat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            tf.broadcast_to([1e10], shape=tf.shape(t_vals[..., :1])),
        ],
        axis=-1,
    )

    if dirs is not None:
        dists = dists * tf.linalg.norm(dirs[..., None, :], axis=-1)

    alpha = 1.0 - tf.exp(-sigma * dists)
    m_alpha = 1.0 - alpha[..., :-1] + 1e-10

    list_alpha = tf.split(
        m_alpha, num_or_size_splits=m_alpha.shape[-1], axis=len(m_alpha.shape) - 1
    )
    for i in range(len(list_alpha)):
        if i > 0:
            list_alpha[i] = list_alpha[i - 1] * list_alpha[i]
    concat_alpha = tf.concat(list_alpha, axis=-1)
    accum_prod = tf.concat(
        [tf.ones_like(alpha[..., :1], alpha.dtype), concat_alpha], axis=-1
    )
    weights = alpha * accum_prod
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if white_bkgd:
        acc = tf.reduce_sum(weights, axis=-1)
        rgb = rgb + (1.0 - acc[..., None])

    return rgb, weights


def get_mlp(
    depth, width, n_samples, pos_encode_dims, skip_layer, deg_view=0, fp16=False
):
    data_type = "float16" if fp16 else "float32"
    inputs = [
        keras.Input(shape=(n_samples, 2 * 3 * pos_encode_dims + 3), dtype=data_type)
    ]

    if deg_view != 0:
        inputs_viewdirs = keras.Input(shape=(2 * 3 * deg_view + 3), dtype=data_type)
        inputs = inputs + [inputs_viewdirs]

    samples = tf.reshape(inputs[0], [-1, inputs[0].shape[-1]])
    x = samples
    for i in range(depth):
        x = layers.Dense(units=width, activation="relu")(x)
        if i % skip_layer == 0 and i > 0:
            x = layers.concatenate([x, samples], axis=-1)

    sigma = layers.Dense(units=1, activation="relu")(x)
    sigma = tf.reshape(sigma, [-1, n_samples])
    sigma = tf.cast(sigma, tf.float32) if fp16 else sigma

    if deg_view != 0:
        bottleneck = layers.Dense(units=width, activation=None)(x)
        inputs_viewdirs = tf.tile(inputs[1][:, None, :], (1, n_samples, 1))
        inputs_viewdirs = tf.reshape(inputs_viewdirs, [-1, inputs_viewdirs.shape[-1]])
        inputs_viewdirs = tf.concat([bottleneck, inputs_viewdirs], axis=-1)
        x = layers.Dense(units=width // 2, activation="relu")(inputs_viewdirs)

    rgb = layers.Dense(units=3, activation="sigmoid")(x)
    rgb = tf.reshape(rgb, [-1, n_samples, 3])
    rgb = tf.cast(rgb, tf.float32) if fp16 else rgb

    return keras.Model(inputs=inputs, outputs=(rgb, sigma))


class NeRF(keras.Model):
    def __init__(self, cfg):
        super(NeRF, self).__init__()
        self._cfg = cfg

        pos_encode_dims = self._cfg.max_deg_point - self._cfg.min_deg_point
        self._mlp_coarse = get_mlp(
            depth=self._cfg.netdepth_coarse,
            width=self._cfg.netwidth_coarse,
            n_samples=self._cfg.n_coarse_samples,
            pos_encode_dims=pos_encode_dims,
            skip_layer=self._cfg.skip_layer,
            deg_view=self._cfg.deg_view,
            fp16=self._cfg.fp16,
        )

        if self._cfg.n_fine_samples > 0:
            self._mlp_fine = get_mlp(
                depth=self._cfg.netdepth_fine,
                width=self._cfg.netwidth_fine,
                n_samples=self._cfg.n_coarse_samples + self._cfg.n_fine_samples,
                pos_encode_dims=pos_encode_dims,
                skip_layer=self._cfg.skip_layer,
                deg_view=self._cfg.deg_view,
                fp16=self._cfg.fp16,
            )

    def call(self, inputs, training=False):
        if self._cfg.use_viewdirs:
            (origins, directions, viewdirs) = inputs
        else:
            (origins, directions) = inputs

        (samples, t_vals) = sample_along_rays(
            origins=origins,
            directions=directions,
            near=self._cfg.near,
            far=self._cfg.far,
            n_samples=self._cfg.n_coarse_samples,
            randomized=self._cfg.randomized,
        )

        samples_enc = encode_position(
            samples,
            min_deg=self._cfg.min_deg_point,
            max_deg=self._cfg.max_deg_point,
            fp16_sin=self._cfg.use_ipu,
        )

        if self._cfg.use_viewdirs:
            viewdirs_enc = encode_position(
                viewdirs,
                min_deg=0,
                max_deg=self._cfg.deg_view,
                fp16_sin=self._cfg.use_ipu,
            )
            (rgb, sigma) = self._mlp_coarse([samples_enc, viewdirs_enc])
        else:
            (rgb, sigma) = self._mlp_coarse(samples_enc)

        if self._cfg.fp16:
            t_vals = tf.cast(t_vals, tf.float32)
            origins = tf.cast(origins, tf.float32)
            directions = tf.cast(directions, tf.float32)
            viewdirs = tf.cast(viewdirs, tf.float32)

        rgb_coarse, weights = volumetric_rendering(
            rgb=rgb,
            sigma=sigma,
            t_vals=t_vals,
            dirs=viewdirs if self._cfg.use_viewdirs else None,
            white_bkgd=self._cfg.white_bkgd,
        )

        if self._cfg.n_fine_samples > 0:
            t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])

            t_vals, samples = sample_pdf(
                bins=t_vals_mid,
                weights=weights[..., 1:-1],
                origins=origins,
                directions=directions,
                t_vals=t_vals,
                n_samples=self._cfg.n_fine_samples,
                randomized=self._cfg.randomized,
            )

            samples_enc = encode_position(
                samples,
                min_deg=self._cfg.min_deg_point,
                max_deg=self._cfg.max_deg_point,
                fp16_sin=self._cfg.use_ipu,
            )

            if self._cfg.use_viewdirs:
                (rgb, sigma) = self._mlp_fine([samples_enc, viewdirs_enc])
            else:
                (rgb, sigma) = self._mlp_fine(samples_enc)

            rgb_fine, _ = volumetric_rendering(
                rgb=rgb,
                sigma=sigma,
                t_vals=t_vals,
                dirs=viewdirs if self._cfg.use_viewdirs else None,
                white_bkgd=self._cfg.white_bkgd,
            )

            if training:
                return tf.concat([rgb_coarse, rgb_fine], axis=0)
            else:
                return rgb_fine
        else:
            return rgb_coarse


def parse_args():
    parser = argparse.ArgumentParser("Neural Radiance Fields (NeRF) Generator")
    parser.add_argument(
        "--config_yaml", type=str, default=None, help="config_yaml used to test."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1312, help="number of rays used to test."
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16.")
    parser.add_argument("--psnr", action="store_true", help="compute psnr.")
    parser.add_argument(
        "--gif", action="store_true", help="save a gif for test results."
    )

    # data
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="blender",
        help="type of dataset(e.g. blender).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="nerf_synthetic/lego",
        help="input data directory.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=30,
        help="number of images used to test(<=0 for full size).",
    )
    parser.add_argument(
        "--factor", type=int, default=1, help="the downsample factor of images."
    )
    parser.add_argument(
        "--render_path",
        action="store_true",
        help="render generated path if set true(llff only).",
    )
    parser.add_argument(
        "--use_pixel_centers",
        action="store_true",
        help="generate rays through pixel center, False is the original way.",
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="using white color as default background(blender only).",
    )

    parser.add_argument(
        "--near", type=float, default=2.0, help="near clip of volumetric rendering."
    )
    parser.add_argument(
        "--far", type=float, default=6.0, help="far clip of volumentric rendering."
    )
    parser.add_argument(
        "--min_deg_point",
        type=int,
        default=0,
        help="min degree of positional encoding for points.",
    )
    parser.add_argument(
        "--max_deg_point",
        type=int,
        default=10,
        help="max degree of positional encoding for points.",
    )
    parser.add_argument(
        "--deg_view",
        type=int,
        default=4,
        help="degree of positional encoding for viewdirs.",
    )
    parser.add_argument(
        "--n_coarse_samples",
        type=int,
        default=64,
        help="number of samples on each ray for the coarse model",
    )
    parser.add_argument(
        "--n_fine_samples",
        type=int,
        default=128,
        help="number of samples on each ray for the fine model",
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use view directions as an input."
    )
    parser.add_argument(
        "--randomized", action="store_true", help="use randomized stratified sampling."
    )

    # MLP
    parser.add_argument(
        "--skip_layer",
        type=int,
        default=4,
        help="add a skip connection to the output vector of every.",
    )
    parser.add_argument(
        "--netdepth_coarse", type=int, default=8, help="depth of the coarse MLP."
    )
    parser.add_argument(
        "--netwidth_coarse", type=int, default=256, help="width of the coarse MLP."
    )
    parser.add_argument(
        "--netdepth_fine", type=int, default=8, help="depth of the fine MLP"
    )
    parser.add_argument(
        "--netwidth_fine", type=int, default=256, help="width of the fine MLP"
    )

    # IPU system arguments
    parser.add_argument(
        "--use_ipu", action="store_true", help="flag to enable IPU specific code paths."
    )
    parser.add_argument(
        "--use_half_partial",
        action="store_true",
        help="flag to use fp16 partials in conv/matmul.",
    )
    parser.add_argument(
        "--io_tiles", type=int, default=96, help="number of IPU io tiles."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args = update_config(args)

    if (args.use_ipu and args.use_half_partial) and (not args.fp16):
        raise ValueError("Only set use_half_partial to True with IPU fp16.")

    model = NeRF(args)
    input_data = [
        tf.random.normal([args.batch_size, 3]),
        tf.random.normal([args.batch_size, 3]),
        tf.random.normal([args.batch_size, 3]),
    ]
    model.build(
        input_shape=[(args.batch_size, 3), (args.batch_size, 3), (args.batch_size, 3)]
    )
    _ = model(input_data)
    # Export saved model
    tf.saved_model.save(model, "nerf_saved_model")
    # Export ONNX model
    os.system(
        "python -m tf2onnx.convert --saved-model nerf_saved_model --output NeRF.onnx"
    )

    print("Check precision between TF and ONNX:")
    # Generate input data
    input_info = {
        "input_1": ([args.batch_size, 3], np.float32),
        "input_2": ([args.batch_size, 3], np.float32),
        "input_3": ([args.batch_size, 3], np.float32),
    }
    input_data = helper.generate_data(input_info, 0.0, 0.2)
    # Run TF
    t1 = tf.convert_to_tensor(input_data["input_1"], tf.float32)
    t2 = tf.convert_to_tensor(input_data["input_2"], tf.float32)
    t3 = tf.convert_to_tensor(input_data["input_3"], tf.float32)
    tf_res = model([t1, t2, t3])
    tf_res = [tf_res.numpy()]

    # Run ONNXRUNTIME
    sess = poprt.backend.get_session("NeRF.onnx", 1, "onnxruntime")
    sess.load()
    inputs_info, outputs_info = sess.get_io_info()
    outputs_name = [o for o in outputs_info]
    ort_res = sess.run(outputs_name, input_data)
    helper.accuracy(tf_res, ort_res)
