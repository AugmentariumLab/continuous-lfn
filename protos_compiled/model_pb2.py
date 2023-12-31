# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/model.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12protos/model.proto\"\x8f\x0c\n\x05Model\x12\x1f\n\x07primary\x18\x01 \x01(\x0b\x32\x0e.Model.Network\x12!\n\tauxiliary\x18\x02 \x01(\x0b\x32\x0e.Model.Network\x1a\'\n\x06Tensor\x12\r\n\x05shape\x18\x01 \x03(\x05\x12\x0e\n\x06values\x18\x02 \x03(\x02\x1a\x98\x0b\n\x07Network\x12!\n\x03mlp\x18\x01 \x01(\x0b\x32\x12.Model.Network.MLPH\x00\x12\'\n\x06resnet\x18\x02 \x01(\x0b\x32\x15.Model.Network.ResNetH\x00\x12\'\n\x06subnet\x18\x03 \x01(\x0b\x32\x15.Model.Network.SubNetH\x00\x12/\n\nmultimodel\x18\x05 \x01(\x0b\x32\x19.Model.Network.MultimodelH\x00\x12\x31\n\x0bmultisubnet\x18\x06 \x01(\x0b\x32\x1a.Model.Network.MultisubnetH\x00\x12\x31\n\x0bmultiresnet\x18\x07 \x01(\x0b\x32\x1a.Model.Network.MultiresnetH\x00\x12\x31\n\nstate_dict\x18\x04 \x03(\x0b\x32\x1d.Model.Network.StateDictEntry\x1a?\n\x0eStateDictEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1c\n\x05value\x18\x02 \x01(\x0b\x32\r.Model.Tensor:\x02\x38\x01\x1ap\n\x03MLP\x12\x13\n\x0bin_features\x18\x01 \x01(\x05\x12\x14\n\x0cout_features\x18\x02 \x01(\x05\x12\x0e\n\x06layers\x18\x03 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x04 \x01(\x05\x12\x15\n\ruse_layernorm\x18\x05 \x01(\x08\x1a\x89\x01\n\x06ResNet\x12\x13\n\x0bin_features\x18\x01 \x01(\x05\x12\x14\n\x0cout_features\x18\x02 \x01(\x05\x12\x0e\n\x06layers\x18\x03 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x04 \x01(\x05\x12\x15\n\ruse_layernorm\x18\x05 \x01(\x08\x12\x14\n\x0coutput_every\x18\x06 \x01(\x05\x1a\x87\x01\n\x06SubNet\x12\x13\n\x0bin_features\x18\x01 \x01(\x05\x12\x14\n\x0cout_features\x18\x02 \x01(\x05\x12\x0e\n\x06layers\x18\x03 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x04 \x01(\x05\x12\x15\n\ruse_layernorm\x18\x05 \x01(\x08\x12\x12\n\ninterleave\x18\x06 \x01(\x08\x1a\xc9\x01\n\nMultimodel\x12\x12\n\nnum_models\x18\x01 \x01(\x05\x12\x13\n\x0bin_features\x18\x02 \x01(\x05\x12\x14\n\x0cout_features\x18\x03 \x01(\x05\x12\x0e\n\x06layers\x18\x04 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x05 \x01(\x05\x12\x18\n\x10selection_layers\x18\x06 \x01(\x05\x12!\n\x19selection_hidden_features\x18\x07 \x01(\x05\x12\x16\n\x0eselection_mode\x18\x08 \x01(\t\x1a\xca\x01\n\x0bMultisubnet\x12\x12\n\nnum_models\x18\x01 \x01(\x05\x12\x13\n\x0bin_features\x18\x02 \x01(\x05\x12\x14\n\x0cout_features\x18\x03 \x01(\x05\x12\x0e\n\x06layers\x18\x04 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x05 \x01(\x05\x12\x18\n\x10selection_layers\x18\x06 \x01(\x05\x12!\n\x19selection_hidden_features\x18\x07 \x01(\x05\x12\x16\n\x0eselection_mode\x18\x08 \x01(\t\x1a\xe0\x01\n\x0bMultiresnet\x12\x12\n\nnum_models\x18\x01 \x01(\x05\x12\x13\n\x0bin_features\x18\x02 \x01(\x05\x12\x14\n\x0cout_features\x18\x03 \x01(\x05\x12\x0e\n\x06layers\x18\x04 \x01(\x05\x12\x17\n\x0fhidden_features\x18\x05 \x01(\x05\x12\x18\n\x10selection_layers\x18\x06 \x01(\x05\x12!\n\x19selection_hidden_features\x18\x07 \x01(\x05\x12\x16\n\x0eselection_mode\x18\x08 \x01(\t\x12\x14\n\x0coutput_every\x18\t \x01(\x05\x42\t\n\x07networkb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.model_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MODEL_NETWORK_STATEDICTENTRY._options = None
  _MODEL_NETWORK_STATEDICTENTRY._serialized_options = b'8\001'
  _globals['_MODEL']._serialized_start=23
  _globals['_MODEL']._serialized_end=1574
  _globals['_MODEL_TENSOR']._serialized_start=100
  _globals['_MODEL_TENSOR']._serialized_end=139
  _globals['_MODEL_NETWORK']._serialized_start=142
  _globals['_MODEL_NETWORK']._serialized_end=1574
  _globals['_MODEL_NETWORK_STATEDICTENTRY']._serialized_start=472
  _globals['_MODEL_NETWORK_STATEDICTENTRY']._serialized_end=535
  _globals['_MODEL_NETWORK_MLP']._serialized_start=537
  _globals['_MODEL_NETWORK_MLP']._serialized_end=649
  _globals['_MODEL_NETWORK_RESNET']._serialized_start=652
  _globals['_MODEL_NETWORK_RESNET']._serialized_end=789
  _globals['_MODEL_NETWORK_SUBNET']._serialized_start=792
  _globals['_MODEL_NETWORK_SUBNET']._serialized_end=927
  _globals['_MODEL_NETWORK_MULTIMODEL']._serialized_start=930
  _globals['_MODEL_NETWORK_MULTIMODEL']._serialized_end=1131
  _globals['_MODEL_NETWORK_MULTISUBNET']._serialized_start=1134
  _globals['_MODEL_NETWORK_MULTISUBNET']._serialized_end=1336
  _globals['_MODEL_NETWORK_MULTIRESNET']._serialized_start=1339
  _globals['_MODEL_NETWORK_MULTIRESNET']._serialized_end=1563
# @@protoc_insertion_point(module_scope)
