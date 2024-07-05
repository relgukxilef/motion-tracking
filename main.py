
import datetime, os
import tensorflow as tf
from pygltflib import GLTF2
import numpy
import quaternion
import matplotlib.pyplot
import glob

pattern = "../Datasets/Mixamo/*.glb"
files = glob.glob(pattern)

bones = [
    "RightUpLeg", "RightLeg", "LeftUpLeg", "LeftLeg", "Spine2", 
    "Head", "RightHand", "LeftHand"
]

def rotate(q, v):
    return quaternion.as_vector_part(
        q * quaternion.from_vector_part(v) * q.conj()
    )

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
            self.translation + rotate(quat, other.translation),
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

    animation = gltf.animations[-1]
    steps = gltf.accessors[animation.samplers[0].input].count
    transforms = base.apply(
        lambda a: numpy.tile(a, (steps, len(gltf.nodes), 1,))
    )

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
        stack.append(node_index)

    while stack != []:
        node_index = stack.pop()
        node = gltf.nodes[node_index]
        for child_index in node.children:
            world_transform = (
                transforms.apply(lambda a: a[:, node_index, :]) * 
                transforms.apply(lambda a: a[:, child_index, :])
            )
            def apply(a, w):
                a[:, child_index, :] = w
            transforms.apply(apply, world_transform)
            stack.append(child_index)

    selected = transformation.from_shape([steps, len(bones)])
    for node_index, node in enumerate(gltf.nodes):
        name = node.name[len("mixamorig:"):]
        try:
            index = bones.index(name)
        except ValueError:
            continue
        def apply(a, v):
            a[:, index, :] = v[:, node_index, :]
        selected.apply(apply, transforms)

    return selected

def scatter(t):
    f = matplotlib.pyplot.figure()
    p = f.add_subplot(111, projection="3d")
    p.scatter(
        t[..., 0].flatten(), -t[..., 2].flatten(), t[..., 1].flatten()
    )
    s = 2
    p.set_xlim(-s, s)
    p.set_ylim(-s, s)
    p.set_zlim(-s, s)
    matplotlib.pyplot.show()

def simulate_sensors(transforms, sensor_offsets, north):
    sensor_bones = [
        "RightUpLeg", "RightLeg", "LeftUpLeg", "LeftLeg", "Spine2"
    ]
    sensors = transforms.apply(lambda a: a[:, :5, :])
    sensors = sensors * sensor_offsets
    quats = quaternion.as_quat_array(sensors.rotation)
    anuglar_velocity = numpy.moveaxis(numpy.apply_along_axis(
        quaternion.quaternion_time_series.angular_velocity,
        -2, quats, numpy.arange(0, quats.shape[-2]) / 60
    )[..., 1:-1, :, :], -2, -1)
    velocity = (
        sensors.translation[..., 1:, :, :] - 
        sensors.translation[..., :-1, :, :]
    ) * 60
    acceleration = (velocity[..., 1:, :, :] - velocity[..., :-1, :, :]) * 60
    inverse = numpy.invert(quats[..., 1:-1, :])
    acceleration = rotate(inverse, acceleration)
    heading = rotate(inverse, north)

    return sensors, anuglar_velocity, acceleration, heading

def random_cylinder(size):
    translation = numpy.random.uniform(
        [0, 0, .1], [0, 0.4, 0.1], (size, 1, 5, 3)
    )
    rotation = numpy.random.uniform(
        [0, 0, 0], [0, 2 * numpy.pi, 0], (size, 1, 5, 3)
    )
    quat = quaternion.from_rotation_vector(rotation)
    translation = rotate(quat, translation)
    return transformation(translation, quaternion.as_float_array(quat))


sensor_offsets = random_cylinder(2)
north = numpy.random.normal(size=(2, 1, 1, 3))

datasets = []

for file in files:
    transforms = load(file)

    sensors, anuglar_velocity, acceleration, heading = simulate_sensors(
        transforms, sensor_offsets, north
    )

    values = (
        (anuglar_velocity, acceleration, heading,),
        (
            sensors.translation[..., 1:-1, :, :], 
            sensors.rotation[..., 1:-1, :, :],
        ),
    )

    file_data = (
        tf.data.Dataset.from_tensor_slices(values).
        flat_map(
            lambda *x: 
            tf.data.Dataset.from_tensor_slices(x).
            window(120, 1, 1, True).
            flat_map(
                lambda *w: 
                tf.data.Dataset.zip(w).batch(120, True, tf.data.AUTOTUNE)
            )
        ).
        shuffle(1000)
    )
    datasets.append(file_data)

dataset = (
    tf.data.Dataset.sample_from_datasets(datasets).shuffle(128).
    batch(1, True, tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
)

# TODO: fit a model
e = next(iter(dataset))
print(e)

# batch shape: [batch, time, node, 3 + 3 + 3]

# TODO: save model
