.. _models_new:

Models and pre-trained weights - New
####################################

.. note::

    These are the new models docs, documenting the new multi-weight API.
    TODO: Once all is done, remove the "- New" part in the title above, and
    rename this file as models.rst


The ``torchvision.models`` subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection, video classification, and optical flow.

.. note ::
    Backward compatibility is guaranteed for loading a serialized 
    ``state_dict`` to the model created using old PyTorch version. 
    On the contrary, loading entire saved models or serialized 
    ``ScriptModules`` (seralized using older versions of PyTorch) 
    may not preserve the historic behaviour. Refer to the following 
    `documentation 
    <https://pytorch.org/docs/stable/notes/serialization.html#id6>`_   


Classification
==============

.. currentmodule:: torchvision.models

The following classification models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/alexnet
   models/convnext
   models/densenet
   models/efficientnet
   models/efficientnetv2
   models/googlenet
   models/mobilenetv2
   models/mobilenetv3
   models/regnet
   models/resnet
   models/resnext
   models/squeezenet
   models/swin_transformer
   models/vgg
   models/vision_transformer
   models/wide_resnet


Table of all available classification weights
---------------------------------------------

Accuracies are reported on ImageNet

.. include:: generated/classification_table.rst

Semantic Segmentation
=====================

.. currentmodule:: torchvision.models.segmentation

The following semantic segmentation models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/deeplabv3


Table of all available semantic segmentation weights
----------------------------------------------------

All models are evaluated on COCO val2017:

.. include:: generated/segmentation_table.rst



Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

.. currentmodule:: torchvision.models.detection

The following detection models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/fcos
   models/mask_rcnn
   models/retinanet

Table of all available detection weights
----------------------------------------

Box MAPs are reported on COCO

.. include:: generated/detection_table.rst


Video Classification
====================

.. currentmodule:: torchvision.models.video

The following video classification models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/video_resnet

Table of all available video classification weights
---------------------------------------------------

Accuracies are reported on Kinetics-400

.. include:: generated/video_table.rst
