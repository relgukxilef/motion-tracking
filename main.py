
import datetime, os
#import tensorflow as tf
from pygltflib import GLTF2
import numpy
import quaternion
import matplotlib.pyplot

file = "../Datasets/Mixamo/Catwalk Walk.glb"

bones = [
    "Hips", "RightUpLeg", "RightLeg", "LeftUpLeg", "LeftLeg", "Spine2", "Head",
    "RightHand", "LeftHand"
]

# Create a list of tensors of shape (time, bone, translation and orientation,)

# relative[animation][node, time]
# needs to contain every node to apply parent transformation

class transformation:
    def from_shape(shape):
        return transformation(
            numpy.zeros(shape + [3,], numpy.float32),
            numpy.zeros(shape + [4,], numpy.float32)
        )

    def __init__(self, translation, rotation):
        self.translation = numpy.array(translation)
        self.rotation = numpy.array(rotation)

    def apply(self, f, *other):
        return transformation(
            f(self.translation, *[i.translation for i in other]),
            f(self.rotation, *[i.rotation for i in other])
        )
    
    def __mul__(self, other):
        quat, other_quat = (
            quaternion.as_quat_array(t.rotation) for t in (self, other)
        )
        return transformation(
            self.translation + 
            quaternion.as_vector_part(
                quat * quaternion.from_vector_part(other.translation) * 
                quat.conj()
            ),
            quaternion.as_float_array(quat * other_quat)
        )

def get_float_array(gltf, blob, accessor_index):
    accessor = gltf.accessors[accessor_index]
    assert accessor.byteOffset in [0,]
    view = gltf.bufferViews[accessor.bufferView]
    return numpy.frombuffer(blob[
        view.byteOffset + accessor.byteOffset : 
        view.byteOffset + view.byteLength
    ], numpy.float32)

def load(file):
    gltf = GLTF2().load(file)

    blob = gltf.binary_blob()

    base = transformation.from_shape([len(gltf.nodes),])
    for i, node in enumerate(gltf.nodes):
        base.translation[i] = node.translation
        base.rotation[i] = node.rotation[3:] + node.rotation[:3]

    steps = gltf.accessors[0].count
    transforms = base.apply(
        lambda a: numpy.tile(a, (steps, len(gltf.nodes), 1,))
    )
    animation = gltf.animations[0]

    for channel in animation.channels:
        sampler = animation.samplers[channel.sampler]
        path = channel.target.path
        values = get_float_array(
            gltf, blob, sampler.output
        ).reshape((steps, 3 if path == "translation" else 4,))

        if path == "rotation":
            transforms.rotation[:, channel.target.node, :] = numpy.concatenate(
                [values[..., 3:], values[..., :3]], -1
            )
        else:
            transforms.translation[:, channel.target.node, :] = values

    stack = []
    for node_index in gltf.scenes[gltf.scene].nodes:
        #world_transform = (
        #    base.apply(lambda a: a[None, node_index, :]) * 
        #    transforms.apply(lambda a: a[:, node_index, :])
        #)
        #def apply(a, w):
        #    a[:, node_index, :] = w
        #transforms.apply(apply, world_transform)
        stack.append(node_index)

    while stack != []:
        node_index = stack.pop()
        node = gltf.nodes[node_index]
        for child_index in node.children:
            world_transform = (
                #base.apply(lambda a: a[None, child_index, :]) * 
                transforms.apply(lambda a: a[:, node_index, :]) * 
                transforms.apply(lambda a: a[:, child_index, :])
            )
            def apply(a, w):
                a[:, child_index, :] = w
            transforms.apply(apply, world_transform)
            stack.append(child_index)

    return transforms.apply(lambda v: v[::10, :, :])
    return transforms

t = load(file).translation

f = matplotlib.pyplot.figure()
p = f.add_subplot(111, projection="3d")
p.scatter(
    t[..., 0].flatten(), -t[..., 2].flatten(), t[..., 1].flatten()
)
p.set_xlim(-2, 2)
p.set_ylim(-2, 2)
p.set_zlim(-2, 2)
matplotlib.pyplot.show()
