import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np
from torchvision import utils


import click
import pickle
import re
import copy
import numpy as np
import torch
import dnnlib
from torch_utils import misc

from model import Generator, Discriminator


def _collect_tf_params(tf_net):
    # pylint: disable=protected-access
    tf_params = dict()
    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + '/', comp)
    recurse('', tf_net)
    return tf_params


class _TFNetworkStub(dnnlib.EasyDict):
    pass

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        
        return super().find_class(module, name)


def convert_modconv(vars, source_name, target_name, flip=False):
    weight = vars[source_name + "/weight"]
    mod_weight = vars[source_name + "/mod_weight"]
    mod_bias = vars[source_name + "/mod_bias"]
    noise = vars[source_name + "/noise_strength"]
    bias = vars[source_name + "/bias"]

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "noise.weight": np.array([noise]),
        "activate.bias": bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + ".conv.weight"] = torch.flip(
            dic_torch[target_name + ".conv.weight"], [3, 4]
        )

    return dic_torch


def convert_conv(vars, source_name, target_name, bias=True, start=0):
    weight = vars[source_name + "/weight"]

    dic = {"weight": weight.transpose((3, 2, 0, 1))}

    if bias:
        dic["bias"] = vars[source_name + "/bias"]

    dic_torch = {}

    dic_torch[target_name + f".{start}.weight"] = torch.from_numpy(dic["weight"])

    if bias:
        dic_torch[target_name + f".{start + 1}.bias"] = torch.from_numpy(dic["bias"])

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    weight = vars[source_name + "/weight"]
    mod_weight = vars[source_name + "/mod_weight"]
    mod_bias = vars[source_name + "/mod_bias"]
    bias = vars[source_name + "/bias"]

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "bias": bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    weight = vars[source_name + "/weight"]
    bias = vars[source_name + "/bias"]

    dic = {"weight": weight.transpose((1, 0)), "bias": bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        if k not in state_dict:
            raise KeyError(k + " is not found")

        if v.shape != state_dict[k].shape:
            raise ValueError(f"Shape mismatch: {v.shape} vs {state_dict[k].shape}")

        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, f"{size}x{size}/FromRGB", "convs.0"))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(vars, f"{reso}x{reso}/Conv0", f"convs.{conv_i}.conv1"),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Conv1_down", f"convs.{conv_i}.conv2", start=1
            ),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Skip", f"convs.{conv_i}.skip", start=1, bias=False
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, f"4x4/Conv", "final_conv"))
    update(statedict, convert_dense(vars, f"4x4/Dense0", "final_linear.0"))
    update(statedict, convert_dense(vars, f"Output", "final_linear.1"))

    return statedict


def fill_statedict(state_dict, vars, size, n_mlp):
    log_size = int(math.log(size, 2))
    
    for i in range(n_mlp):
        update(state_dict, convert_dense(vars, f"mapping/Dense{i}", f"style.{i + 1}"))

    update(
        state_dict,
        {
            "input.input": torch.from_numpy(
                vars["synthesis/4x4/Const/const"]
            )
        },
    )

    update(state_dict, convert_torgb(vars, "synthesis/4x4/ToRGB", "to_rgb1"))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(vars, f"synthesis/{reso}x{reso}/ToRGB", f"to_rgbs.{i}"),
        )

    update(state_dict, convert_modconv(vars, "synthesis/4x4/Conv", "conv1"))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                f"synthesis/{reso}x{reso}/Conv0_up",
                f"convs.{conv_i}",
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f"synthesis/{reso}x{reso}/Conv1", f"convs.{conv_i + 1}"
            ),
        )
        conv_i += 2

    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                f"noises.noise_{i}": torch.from_numpy(
                    vars[f"synthesis/noise{i}"]
                )
            },
        )

    return state_dict



def get_tf_params(tf_G):
    if tf_G.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    tf_params = _collect_tf_params(tf_G)
    
    return tf_params




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Tensorflow to pytorch model checkpoint converter"
    )
    parser.add_argument(
        "--gen", action="store_true", help="convert the generator weights"
    )
    parser.add_argument(
        "--disc", action="store_true", help="convert the discriminator weights"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor. config-f = 2, else = 1",
    )
    parser.add_argument("--path", metavar="PATH", help="path to the tensorflow weights")

    args = parser.parse_args()




    with open(args.path, "rb") as f:
        data = _LegacyUnpickler(f).load()
        if isinstance(data, tuple) and len(data) == 3 and all(isinstance(net, _TFNetworkStub) for net in data):
            tf_G, tf_D, tf_Gs = data
        
        generator = get_tf_params(tf_G)
        discriminator = get_tf_params(tf_D)
        g_ema = get_tf_params(tf_Gs)
            
        size = 512

        

    n_mlp = 8

    g = Generator(size, 512, n_mlp, channel_multiplier=args.channel_multiplier)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, g_ema, size, n_mlp)

    g.load_state_dict(state_dict)

    latent_avg = torch.from_numpy(g_ema["dlatent_avg"])

    ckpt = {"g_ema": state_dict, "latent_avg": latent_avg}

    if args.gen:
        g_train = Generator(size, 512, n_mlp, channel_multiplier=args.channel_multiplier)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator, size, n_mlp)
        ckpt["g"] = g_train_state

    if args.disc:
        disc = Discriminator(size, channel_multiplier=args.channel_multiplier)
        d_state = disc.state_dict()
        d_state = discriminator_fill_statedict(d_state, discriminator.vars, size)
        ckpt["d"] = d_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(ckpt, name + ".pt")

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g.to(device)

    z = np.random.RandomState(42).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g([torch.from_numpy(z).to(device)],truncation=0.5,truncation_latent=latent_avg.to(device),randomize_noise=False,)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False

    breakpoint()

    utils.save_image(img_pt, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1))
