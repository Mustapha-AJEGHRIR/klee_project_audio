??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ɑ

|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:@*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:@*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:@ *
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
: *
dtype0
}
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ?*
shared_nameconv3/kernel
v
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*'
_output_shapes
: ?*
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:?*
dtype0
}
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*
shared_nameconv4/kernel
v
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*'
_output_shapes
:?@*
dtype0
l

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv4/bias
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes
:@*
dtype0
{
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_namedense_1/kernel
t
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_output_shapes
:???*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	?@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
i
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D_random_generator
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
R
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
i
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
h

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
R
^	variables
_trainable_variables
`regularization_losses
a	keras_api
i
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f_random_generator
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
R
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
f
0
1
"2
#3
,4
-5
26
37
I8
J9
X10
Y11
g12
h13
f
0
1
"2
#3
,4
-5
26
37
I8
J9
X10
Y11
g12
h13
 
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

X0
Y1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
 
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_zero1_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_zero1_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_1027
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_1646
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_1698??	
?&
?
__inference__traced_save_1646
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@ : : ?:?:?@:@:???:?:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :-)
'
_output_shapes
: ?:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@:'	#
!
_output_shapes
:???:!


_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
?
?
>__inference_conv4_layer_call_and_return_conditional_losses_412

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHWY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@?=j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@?=w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_2_layer_call_and_return_conditional_losses_473

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_3_layer_call_and_return_conditional_losses_503

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
Z
>__inference_pool2_layer_call_and_return_conditional_losses_422

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?=:X T
0
_output_shapes
:?????????@?=
 
_user_specified_nameinputs
?
?
*__inference_sequential_1_layer_call_fn_892
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: ?
	unknown_4:	?$
	unknown_5:?@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?
?
$__inference_conv4_layer_call_fn_1351

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv4_layer_call_and_return_conditional_losses_412x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@?=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:????????????
 
_user_specified_nameinputs
?
[
?__inference_zero1_layer_call_and_return_conditional_losses_1262

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 e
PadPadinputsPad/paddings:output:0*
T0*1
_output_shapes
:???????????^
IdentityIdentityPad:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_2_layer_call_fn_1474

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_473a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
@
$__inference_zero1_layer_call_fn_1245

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_291?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_1440

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_dense_1_layer_call_fn_1449

inputs
unknown:???
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_455p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
?__inference_conv1_layer_call_and_return_conditional_losses_1282

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:?????????@??k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_1_layer_call_fn_1060

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: ?
	unknown_4:	?$
	unknown_5:?@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_529o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_1_layer_call_fn_1093

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: ?
	unknown_4:	?$
	unknown_5:?@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
[
?__inference_pool2_layer_call_and_return_conditional_losses_1382

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?=:X T
0
_output_shapes
:?????????@?=
 
_user_specified_nameinputs
?	
a
B__inference_dropout_3_layer_call_and_return_conditional_losses_596

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
Z
>__inference_pool2_layer_call_and_return_conditional_losses_315

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_1_layer_call_and_return_conditional_losses_455

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
[
?__inference_pool1_layer_call_and_return_conditional_losses_1317

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
@
$__inference_pool3_layer_call_fn_1387

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_327?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_conv2_layer_call_and_return_conditional_losses_372

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:????????? ??k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:????????? ??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????@??
 
_user_specified_nameinputs
?
@
$__inference_pool2_layer_call_fn_1367

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_315?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_1581

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_3_layer_call_fn_1535

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_596o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_1027
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: ?
	unknown_4:	?$
	unknown_5:?@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__wrapped_model_281o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?
a
E__inference_activation_2_layer_call_and_return_conditional_losses_496

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
&__inference_dense_3_layer_call_fn_1561

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_515o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
^
B__inference_flatten_1_layer_call_and_return_conditional_losses_443

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
E__inference_activation_1_layer_call_and_return_conditional_losses_466

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_3_layer_call_and_return_conditional_losses_1571

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
>__inference_conv3_layer_call_and_return_conditional_losses_395

inputs9
conv2d_readvariableop_resource: ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:????????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :????????? ?A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?A
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_1412

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_680w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_sequential_1_layer_call_fn_560
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: ?
	unknown_4:	?$
	unknown_5:?@
	unknown_6:@
	unknown_7:???
	unknown_8:	?
	unknown_9:	?@

unknown_10:@

unknown_11:@

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_sequential_1_layer_call_and_return_conditional_losses_529o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?
Z
>__inference_zero1_layer_call_and_return_conditional_losses_291

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
@
$__inference_zero1_layer_call_fn_1250

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_342j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_1469

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_conv4_layer_call_and_return_conditional_losses_1362

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHWY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????@?=j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????@?=w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:????????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_2_layer_call_and_return_conditional_losses_485

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
@
$__inference_pool2_layer_call_fn_1372

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_422h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?=:X T
0
_output_shapes
:?????????@?=
 
_user_specified_nameinputs
?	
a
B__inference_dropout_2_layer_call_and_return_conditional_losses_635

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_435

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?F
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_828

inputs#
	conv1_782:@
	conv1_784:@#
	conv2_787:@ 
	conv2_789: $
	conv3_793: ?
	conv3_795:	?$
	conv4_798:?@
	conv4_800:@ 
dense_1_807:???
dense_1_809:	?
dense_2_814:	?@
dense_2_816:@
dense_3_821:@
dense_3_823:
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?
zero1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_342?
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0	conv1_782	conv1_784*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv1_layer_call_and_return_conditional_losses_355?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	conv2_787	conv2_789*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? ??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv2_layer_call_and_return_conditional_losses_372?
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_382?
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0	conv3_793	conv3_795*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv3_layer_call_and_return_conditional_losses_395?
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0	conv4_798	conv4_800*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv4_layer_call_and_return_conditional_losses_412?
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_422?
pool3/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_428?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_680?
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_443?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_807dense_1_809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_455?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_466?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_814dense_2_816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_485?
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_2_layer_call_and_return_conditional_losses_496?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_596?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_821dense_3_823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_515?
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_3_layer_call_and_return_conditional_losses_526t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Z
>__inference_pool3_layer_call_and_return_conditional_losses_327

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Z
>__inference_zero1_layer_call_and_return_conditional_losses_342

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 e
PadPadinputsPad/paddings:output:0*
T0*1
_output_shapes
:???????????^
IdentityIdentityPad:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
[
?__inference_pool3_layer_call_and_return_conditional_losses_1397

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

a
B__inference_dropout_1_layer_call_and_return_conditional_losses_680

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_1525

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
G
+__inference_activation_2_layer_call_fn_1520

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_2_layer_call_and_return_conditional_losses_496`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1417

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
@
$__inference_pool1_layer_call_fn_1307

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_303?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
@
$__inference_pool1_layer_call_fn_1312

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_382i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:????????? ?A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? ??:Y U
1
_output_shapes
:????????? ??
 
_user_specified_nameinputs
?
a
C__inference_dropout_3_layer_call_and_return_conditional_losses_1540

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?[
?
__inference__wrapped_model_281
zero1_inputK
1sequential_1_conv1_conv2d_readvariableop_resource:@@
2sequential_1_conv1_biasadd_readvariableop_resource:@K
1sequential_1_conv2_conv2d_readvariableop_resource:@ @
2sequential_1_conv2_biasadd_readvariableop_resource: L
1sequential_1_conv3_conv2d_readvariableop_resource: ?A
2sequential_1_conv3_biasadd_readvariableop_resource:	?L
1sequential_1_conv4_conv2d_readvariableop_resource:?@@
2sequential_1_conv4_biasadd_readvariableop_resource:@H
3sequential_1_dense_1_matmul_readvariableop_resource:???C
4sequential_1_dense_1_biasadd_readvariableop_resource:	?F
3sequential_1_dense_2_matmul_readvariableop_resource:	?@B
4sequential_1_dense_2_biasadd_readvariableop_resource:@E
3sequential_1_dense_3_matmul_readvariableop_resource:@B
4sequential_1_dense_3_biasadd_readvariableop_resource:
identity??)sequential_1/conv1/BiasAdd/ReadVariableOp?(sequential_1/conv1/Conv2D/ReadVariableOp?)sequential_1/conv2/BiasAdd/ReadVariableOp?(sequential_1/conv2/Conv2D/ReadVariableOp?)sequential_1/conv3/BiasAdd/ReadVariableOp?(sequential_1/conv3/Conv2D/ReadVariableOp?)sequential_1/conv4/BiasAdd/ReadVariableOp?(sequential_1/conv4/Conv2D/ReadVariableOp?+sequential_1/dense_1/BiasAdd/ReadVariableOp?*sequential_1/dense_1/MatMul/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?*sequential_1/dense_2/MatMul/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?*sequential_1/dense_3/MatMul/ReadVariableOp?
sequential_1/zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 ?
sequential_1/zero1/PadPadzero1_input(sequential_1/zero1/Pad/paddings:output:0*
T0*1
_output_shapes
:????????????
(sequential_1/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_1/conv1/Conv2DConv2Dsequential_1/zero1/Pad:output:00sequential_1/conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW*
paddingVALID*
strides
?
)sequential_1/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/conv1/BiasAddBiasAdd"sequential_1/conv1/Conv2D:output:01sequential_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW?
sequential_1/conv1/ReluRelu#sequential_1/conv1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????@???
(sequential_1/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
sequential_1/conv2/Conv2DConv2D%sequential_1/conv1/Relu:activations:00sequential_1/conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW*
paddingVALID*
strides
?
)sequential_1/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
sequential_1/conv2/BiasAddBiasAdd"sequential_1/conv2/Conv2D:output:01sequential_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW?
sequential_1/conv2/ReluRelu#sequential_1/conv2/BiasAdd:output:0*
T0*1
_output_shapes
:????????? ???
sequential_1/pool1/MaxPoolMaxPool%sequential_1/conv2/Relu:activations:0*0
_output_shapes
:????????? ?A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
(sequential_1/conv3/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv3_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype0?
sequential_1/conv3/Conv2DConv2D#sequential_1/pool1/MaxPool:output:00sequential_1/conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW*
paddingVALID*
strides
?
)sequential_1/conv3/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/conv3/BiasAddBiasAdd"sequential_1/conv3/Conv2D:output:01sequential_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW?
sequential_1/conv3/ReluRelu#sequential_1/conv3/BiasAdd:output:0*
T0*1
_output_shapes
:?????????????
(sequential_1/conv4/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
sequential_1/conv4/Conv2DConv2D%sequential_1/conv3/Relu:activations:00sequential_1/conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW*
paddingVALID*
strides
?
)sequential_1/conv4/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/conv4/BiasAddBiasAdd"sequential_1/conv4/Conv2D:output:01sequential_1/conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW
sequential_1/conv4/ReluRelu#sequential_1/conv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?=?
sequential_1/pool2/MaxPoolMaxPool%sequential_1/conv4/Relu:activations:0*/
_output_shapes
:?????????@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
sequential_1/pool3/MaxPoolMaxPool#sequential_1/pool2/MaxPool:output:0*/
_output_shapes
:?????????@*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
sequential_1/dropout_1/IdentityIdentity#sequential_1/pool3/MaxPool:output:0*
T0*/
_output_shapes
:?????????@m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  ?
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_1/Identity:output:0%sequential_1/flatten_1/Const:output:0*
T0*)
_output_shapes
:????????????
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_1/activation_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
sequential_1/dropout_2/IdentityIdentity,sequential_1/activation_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
sequential_1/dense_2/MatMulMatMul(sequential_1/dropout_2/Identity:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@
sequential_1/activation_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
sequential_1/dropout_3/IdentityIdentity,sequential_1/activation_2/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_3/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!sequential_1/activation_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????z
IdentityIdentity+sequential_1/activation_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp*^sequential_1/conv1/BiasAdd/ReadVariableOp)^sequential_1/conv1/Conv2D/ReadVariableOp*^sequential_1/conv2/BiasAdd/ReadVariableOp)^sequential_1/conv2/Conv2D/ReadVariableOp*^sequential_1/conv3/BiasAdd/ReadVariableOp)^sequential_1/conv3/Conv2D/ReadVariableOp*^sequential_1/conv4/BiasAdd/ReadVariableOp)^sequential_1/conv4/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2V
)sequential_1/conv1/BiasAdd/ReadVariableOp)sequential_1/conv1/BiasAdd/ReadVariableOp2T
(sequential_1/conv1/Conv2D/ReadVariableOp(sequential_1/conv1/Conv2D/ReadVariableOp2V
)sequential_1/conv2/BiasAdd/ReadVariableOp)sequential_1/conv2/BiasAdd/ReadVariableOp2T
(sequential_1/conv2/Conv2D/ReadVariableOp(sequential_1/conv2/Conv2D/ReadVariableOp2V
)sequential_1/conv3/BiasAdd/ReadVariableOp)sequential_1/conv3/BiasAdd/ReadVariableOp2T
(sequential_1/conv3/Conv2D/ReadVariableOp(sequential_1/conv3/Conv2D/ReadVariableOp2V
)sequential_1/conv4/BiasAdd/ReadVariableOp)sequential_1/conv4/BiasAdd/ReadVariableOp2T
(sequential_1/conv4/Conv2D/ReadVariableOp(sequential_1/conv4/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?
D
(__inference_flatten_1_layer_call_fn_1434

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_443b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_1459

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1429

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
(__inference_dropout_2_layer_call_fn_1479

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_635p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_2_layer_call_fn_1505

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
?__inference_conv2_layer_call_and_return_conditional_losses_1302

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:????????? ??k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:????????? ??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????@??
 
_user_specified_nameinputs
?
[
?__inference_pool2_layer_call_and_return_conditional_losses_1377

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_3_layer_call_fn_1530

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_503`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?a
?

F__inference_sequential_1_layer_call_and_return_conditional_losses_1240

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@ 3
%conv2_biasadd_readvariableop_resource: ?
$conv3_conv2d_readvariableop_resource: ?4
%conv3_biasadd_readvariableop_resource:	??
$conv4_conv2d_readvariableop_resource:?@3
%conv4_biasadd_readvariableop_resource:@;
&dense_1_matmul_readvariableop_resource:???6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 q
	zero1/PadPadinputszero1/Pad/paddings:output:0*
T0*1
_output_shapes
:????????????
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv1/Conv2DConv2Dzero1/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHWf

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????@???
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW*
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHWf

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:????????? ???
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:????????? ?A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype0?
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW*
paddingVALID*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHWf

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:?????????????
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHWe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?=?
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:?????????@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
pool3/MaxPoolMaxPoolpool2/MaxPool:output:0*/
_output_shapes
:?????????@*
data_formatNCHW*
ksize
*
paddingVALID*
strides
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_1/dropout/MulMulpool3/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@]
dropout_1/dropout/ShapeShapepool3/MaxPool:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  ?
flatten_1/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*)
_output_shapes
:????????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_2/dropout/MulMulactivation_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????f
dropout_2/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@e
activation_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_3/dropout/MulMulactivation_2/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@f
dropout_3/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_3/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????k
activation_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentityactivation_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_3_layer_call_and_return_conditional_losses_1552

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?B
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_942
zero1_input#
	conv1_896:@
	conv1_898:@#
	conv2_901:@ 
	conv2_903: $
	conv3_907: ?
	conv3_909:	?$
	conv4_912:?@
	conv4_914:@ 
dense_1_921:???
dense_1_923:	?
dense_2_928:	?@
dense_2_930:@
dense_3_935:@
dense_3_937:
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
zero1/PartitionedCallPartitionedCallzero1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_342?
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0	conv1_896	conv1_898*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv1_layer_call_and_return_conditional_losses_355?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	conv2_901	conv2_903*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? ??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv2_layer_call_and_return_conditional_losses_372?
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_382?
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0	conv3_907	conv3_909*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv3_layer_call_and_return_conditional_losses_395?
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0	conv4_912	conv4_914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv4_layer_call_and_return_conditional_losses_412?
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_422?
pool3/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_428?
dropout_1/PartitionedCallPartitionedCallpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_435?
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_443?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_921dense_1_923*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_455?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_466?
dropout_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_473?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_928dense_2_930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_485?
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_2_layer_call_and_return_conditional_losses_496?
dropout_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_503?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_935dense_3_937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_515?
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_3_layer_call_and_return_conditional_losses_526t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?9
?
 __inference__traced_restore_1698
file_prefix7
assignvariableop_conv1_kernel:@+
assignvariableop_1_conv1_bias:@9
assignvariableop_2_conv2_kernel:@ +
assignvariableop_3_conv2_bias: :
assignvariableop_4_conv3_kernel: ?,
assignvariableop_5_conv3_bias:	?:
assignvariableop_6_conv4_kernel:?@+
assignvariableop_7_conv4_bias:@6
!assignvariableop_8_dense_1_kernel:???.
assignvariableop_9_dense_1_bias:	?5
"assignvariableop_10_dense_2_kernel:	?@.
 assignvariableop_11_dense_2_bias:@4
"assignvariableop_12_dense_3_kernel:@.
 assignvariableop_13_dense_3_bias:
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
Z
>__inference_pool1_layer_call_and_return_conditional_losses_382

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:????????? ?A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:????????? ?A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? ??:Y U
1
_output_shapes
:????????? ??
 
_user_specified_nameinputs
?
[
?__inference_zero1_layer_call_and_return_conditional_losses_1256

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_conv3_layer_call_fn_1331

inputs"
unknown: ?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv3_layer_call_and_return_conditional_losses_395y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :????????? ?A: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?A
 
_user_specified_nameinputs
?A
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_529

inputs#
	conv1_356:@
	conv1_358:@#
	conv2_373:@ 
	conv2_375: $
	conv3_396: ?
	conv3_398:	?$
	conv4_413:?@
	conv4_415:@ 
dense_1_456:???
dense_1_458:	?
dense_2_486:	?@
dense_2_488:@
dense_3_516:@
dense_3_518:
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
zero1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_342?
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0	conv1_356	conv1_358*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv1_layer_call_and_return_conditional_losses_355?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	conv2_373	conv2_375*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? ??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv2_layer_call_and_return_conditional_losses_372?
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_382?
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0	conv3_396	conv3_398*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv3_layer_call_and_return_conditional_losses_395?
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0	conv4_413	conv4_415*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv4_layer_call_and_return_conditional_losses_412?
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_422?
pool3/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_428?
dropout_1/PartitionedCallPartitionedCallpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_435?
flatten_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_443?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_456dense_1_458*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_455?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_466?
dropout_2/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_473?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_486dense_2_488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_485?
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_2_layer_call_and_return_conditional_losses_496?
dropout_3/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_503?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_516dense_3_518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_515?
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_3_layer_call_and_return_conditional_losses_526t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
[
?__inference_pool1_layer_call_and_return_conditional_losses_1322

inputs
identity?
MaxPoolMaxPoolinputs*0
_output_shapes
:????????? ?A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:????????? ?A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? ??:Y U
1
_output_shapes
:????????? ??
 
_user_specified_nameinputs
?
@
$__inference_pool3_layer_call_fn_1392

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_428h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@5:W S
/
_output_shapes
:?????????@5
 
_user_specified_nameinputs
?
G
+__inference_activation_1_layer_call_fn_1464

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_466a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_1407

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_435h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?F
?
E__inference_sequential_1_layer_call_and_return_conditional_losses_992
zero1_input#
	conv1_946:@
	conv1_948:@#
	conv2_951:@ 
	conv2_953: $
	conv3_957: ?
	conv3_959:	?$
	conv4_962:?@
	conv4_964:@ 
dense_1_971:???
dense_1_973:	?
dense_2_978:	?@
dense_2_980:@
dense_3_985:@
dense_3_987:
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?conv4/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?
zero1/PartitionedCallPartitionedCallzero1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_zero1_layer_call_and_return_conditional_losses_342?
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0	conv1_946	conv1_948*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv1_layer_call_and_return_conditional_losses_355?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0	conv2_951	conv2_953*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? ??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv2_layer_call_and_return_conditional_losses_372?
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:????????? ?A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool1_layer_call_and_return_conditional_losses_382?
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0	conv3_957	conv3_959*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv3_layer_call_and_return_conditional_losses_395?
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0	conv4_962	conv4_964*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@?=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv4_layer_call_and_return_conditional_losses_412?
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool2_layer_call_and_return_conditional_losses_422?
pool3/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_pool3_layer_call_and_return_conditional_losses_428?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_680?
flatten_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_1_layer_call_and_return_conditional_losses_443?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_971dense_1_973*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_455?
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_1_layer_call_and_return_conditional_losses_466?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_635?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_978dense_2_980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_485?
activation_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_2_layer_call_and_return_conditional_losses_496?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_3_layer_call_and_return_conditional_losses_596?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_985dense_3_987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_515?
activation_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_3_layer_call_and_return_conditional_losses_526t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:^ Z
1
_output_shapes
:???????????
%
_user_specified_namezero1_input
?
[
?__inference_pool3_layer_call_and_return_conditional_losses_1402

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@5:W S
/
_output_shapes
:?????????@5
 
_user_specified_nameinputs
?
?
?__inference_conv3_layer_call_and_return_conditional_losses_1342

inputs9
conv2d_readvariableop_resource: ?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:????????????k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :????????? ?A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?A
 
_user_specified_nameinputs
?
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_1484

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_activation_3_layer_call_and_return_conditional_losses_526

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Z
>__inference_pool1_layer_call_and_return_conditional_losses_303

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_1515

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_conv1_layer_call_fn_1271

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv1_layer_call_and_return_conditional_losses_355y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:?????????@??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Z
>__inference_pool3_layer_call_and_return_conditional_losses_428

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@5:W S
/
_output_shapes
:?????????@5
 
_user_specified_nameinputs
?	
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_1496

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
>__inference_conv1_layer_call_and_return_conditional_losses_355

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:?????????@??k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:?????????@??w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?J
?

F__inference_sequential_1_layer_call_and_return_conditional_losses_1156

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@ 3
%conv2_biasadd_readvariableop_resource: ?
$conv3_conv2d_readvariableop_resource: ?4
%conv3_biasadd_readvariableop_resource:	??
$conv4_conv2d_readvariableop_resource:?@3
%conv4_biasadd_readvariableop_resource:@;
&dense_1_matmul_readvariableop_resource:???6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?conv4/BiasAdd/ReadVariableOp?conv4/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 q
	zero1/PadPadinputszero1/Pad/paddings:output:0*
T0*1
_output_shapes
:????????????
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv1/Conv2DConv2Dzero1/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHW*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:?????????@??*
data_formatNCHWf

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:?????????@???
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHW*
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????? ??*
data_formatNCHWf

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:????????? ???
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:????????? ?A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype0?
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHW*
paddingVALID*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????*
data_formatNCHWf

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:?????????????
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHW*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@?=*
data_formatNCHWe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@?=?
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:?????????@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
?
pool3/MaxPoolMaxPoolpool2/MaxPool:output:0*/
_output_shapes
:?????????@*
data_formatNCHW*
ksize
*
paddingVALID*
strides
p
dropout_1/IdentityIdentitypool3/MaxPool:output:0*
T0*/
_output_shapes
:?????????@`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ?  ?
flatten_1/ReshapeReshapedropout_1/Identity:output:0flatten_1/Const:output:0*
T0*)
_output_shapes
:????????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????f
activation_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????r
dropout_2/IdentityIdentityactivation_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@e
activation_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@q
dropout_3/IdentityIdentityactivation_2/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_3/MatMulMatMuldropout_3/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????k
activation_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m
IdentityIdentityactivation_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:???????????: : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_3_layer_call_and_return_conditional_losses_515

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_conv2_layer_call_fn_1291

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? ??*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_conv2_layer_call_and_return_conditional_losses_372y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????? ??`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????@??: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????@??
 
_user_specified_nameinputs
?
G
+__inference_activation_3_layer_call_fn_1576

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_activation_3_layer_call_and_return_conditional_losses_526`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
zero1_input>
serving_default_zero1_input:0???????????@
activation_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:՛
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
"2
#3
,4
-5
26
37
I8
J9
X10
Y11
g12
h13"
trackable_list_wrapper
?
0
1
"2
#3
,4
-5
26
37
I8
J9
X10
Y11
g12
h13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@2conv1/kernel
:@2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@ 2conv2/kernel
: 2
conv2/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
(	variables
)trainable_variables
*regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% ?2conv3/kernel
:?2
conv3/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%?@2conv4/kernel
:@2
conv4/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
4	variables
5trainable_variables
6regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!???2dense_1/kernel
:?2dense_1/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
!:	?@2dense_2/kernel
:@2dense_2/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 :@2dense_3/kernel
:2dense_3/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
*__inference_sequential_1_layer_call_fn_560
+__inference_sequential_1_layer_call_fn_1060
+__inference_sequential_1_layer_call_fn_1093
*__inference_sequential_1_layer_call_fn_892?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1156
F__inference_sequential_1_layer_call_and_return_conditional_losses_1240
E__inference_sequential_1_layer_call_and_return_conditional_losses_942
E__inference_sequential_1_layer_call_and_return_conditional_losses_992?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_281zero1_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_zero1_layer_call_fn_1245
$__inference_zero1_layer_call_fn_1250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_zero1_layer_call_and_return_conditional_losses_1256
?__inference_zero1_layer_call_and_return_conditional_losses_1262?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_conv1_layer_call_fn_1271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_conv1_layer_call_and_return_conditional_losses_1282?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_conv2_layer_call_fn_1291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_conv2_layer_call_and_return_conditional_losses_1302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_pool1_layer_call_fn_1307
$__inference_pool1_layer_call_fn_1312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_pool1_layer_call_and_return_conditional_losses_1317
?__inference_pool1_layer_call_and_return_conditional_losses_1322?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_conv3_layer_call_fn_1331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_conv3_layer_call_and_return_conditional_losses_1342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_conv4_layer_call_fn_1351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_conv4_layer_call_and_return_conditional_losses_1362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_pool2_layer_call_fn_1367
$__inference_pool2_layer_call_fn_1372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_pool2_layer_call_and_return_conditional_losses_1377
?__inference_pool2_layer_call_and_return_conditional_losses_1382?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_pool3_layer_call_fn_1387
$__inference_pool3_layer_call_fn_1392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_pool3_layer_call_and_return_conditional_losses_1397
?__inference_pool3_layer_call_and_return_conditional_losses_1402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_1407
(__inference_dropout_1_layer_call_fn_1412?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_1417
C__inference_dropout_1_layer_call_and_return_conditional_losses_1429?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_flatten_1_layer_call_fn_1434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_1_layer_call_and_return_conditional_losses_1440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_1449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_1459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_1_layer_call_fn_1464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_1_layer_call_and_return_conditional_losses_1469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_2_layer_call_fn_1474
(__inference_dropout_2_layer_call_fn_1479?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_2_layer_call_and_return_conditional_losses_1484
C__inference_dropout_2_layer_call_and_return_conditional_losses_1496?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_1505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_1515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_2_layer_call_fn_1520?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_2_layer_call_and_return_conditional_losses_1525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_3_layer_call_fn_1530
(__inference_dropout_3_layer_call_fn_1535?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_3_layer_call_and_return_conditional_losses_1540
C__inference_dropout_3_layer_call_and_return_conditional_losses_1552?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dense_3_layer_call_fn_1561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_3_layer_call_and_return_conditional_losses_1571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_activation_3_layer_call_fn_1576?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_3_layer_call_and_return_conditional_losses_1581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_1027zero1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_281?"#,-23IJXYgh>?;
4?1
/?,
zero1_input???????????
? ";?8
6
activation_3&?#
activation_3??????????
F__inference_activation_1_layer_call_and_return_conditional_losses_1469Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
+__inference_activation_1_layer_call_fn_1464M0?-
&?#
!?
inputs??????????
? "????????????
F__inference_activation_2_layer_call_and_return_conditional_losses_1525X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
+__inference_activation_2_layer_call_fn_1520K/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_activation_3_layer_call_and_return_conditional_losses_1581X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
+__inference_activation_3_layer_call_fn_1576K/?,
%?"
 ?
inputs?????????
? "???????????
?__inference_conv1_layer_call_and_return_conditional_losses_1282p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0?????????@??
? ?
$__inference_conv1_layer_call_fn_1271c9?6
/?,
*?'
inputs???????????
? ""??????????@???
?__inference_conv2_layer_call_and_return_conditional_losses_1302p"#9?6
/?,
*?'
inputs?????????@??
? "/?,
%?"
0????????? ??
? ?
$__inference_conv2_layer_call_fn_1291c"#9?6
/?,
*?'
inputs?????????@??
? ""?????????? ???
?__inference_conv3_layer_call_and_return_conditional_losses_1342o,-8?5
.?+
)?&
inputs????????? ?A
? "/?,
%?"
0????????????
? ?
$__inference_conv3_layer_call_fn_1331b,-8?5
.?+
)?&
inputs????????? ?A
? ""??????????????
?__inference_conv4_layer_call_and_return_conditional_losses_1362o239?6
/?,
*?'
inputs????????????
? ".?+
$?!
0?????????@?=
? ?
$__inference_conv4_layer_call_fn_1351b239?6
/?,
*?'
inputs????????????
? "!??????????@?=?
A__inference_dense_1_layer_call_and_return_conditional_losses_1459_IJ1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? |
&__inference_dense_1_layer_call_fn_1449RIJ1?.
'?$
"?
inputs???????????
? "????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_1515]XY0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? z
&__inference_dense_2_layer_call_fn_1505PXY0?-
&?#
!?
inputs??????????
? "??????????@?
A__inference_dense_3_layer_call_and_return_conditional_losses_1571\gh/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? y
&__inference_dense_3_layer_call_fn_1561Ogh/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_1417l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_1429l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
(__inference_dropout_1_layer_call_fn_1407_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
(__inference_dropout_1_layer_call_fn_1412_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
C__inference_dropout_2_layer_call_and_return_conditional_losses_1484^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_2_layer_call_and_return_conditional_losses_1496^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_2_layer_call_fn_1474Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_2_layer_call_fn_1479Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_dropout_3_layer_call_and_return_conditional_losses_1540\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
C__inference_dropout_3_layer_call_and_return_conditional_losses_1552\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? {
(__inference_dropout_3_layer_call_fn_1530O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@{
(__inference_dropout_3_layer_call_fn_1535O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
C__inference_flatten_1_layer_call_and_return_conditional_losses_1440b7?4
-?*
(?%
inputs?????????@
? "'?$
?
0???????????
? ?
(__inference_flatten_1_layer_call_fn_1434U7?4
-?*
(?%
inputs?????????@
? "?????????????
?__inference_pool1_layer_call_and_return_conditional_losses_1317?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
?__inference_pool1_layer_call_and_return_conditional_losses_1322k9?6
/?,
*?'
inputs????????? ??
? ".?+
$?!
0????????? ?A
? ?
$__inference_pool1_layer_call_fn_1307?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_pool1_layer_call_fn_1312^9?6
/?,
*?'
inputs????????? ??
? "!?????????? ?A?
?__inference_pool2_layer_call_and_return_conditional_losses_1377?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
?__inference_pool2_layer_call_and_return_conditional_losses_1382i8?5
.?+
)?&
inputs?????????@?=
? "-?*
#? 
0?????????@5
? ?
$__inference_pool2_layer_call_fn_1367?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_pool2_layer_call_fn_1372\8?5
.?+
)?&
inputs?????????@?=
? " ??????????@5?
?__inference_pool3_layer_call_and_return_conditional_losses_1397?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
?__inference_pool3_layer_call_and_return_conditional_losses_1402h7?4
-?*
(?%
inputs?????????@5
? "-?*
#? 
0?????????@
? ?
$__inference_pool3_layer_call_fn_1387?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_pool3_layer_call_fn_1392[7?4
-?*
(?%
inputs?????????@5
? " ??????????@?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1156z"#,-23IJXYghA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1240z"#,-23IJXYghA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_1_layer_call_and_return_conditional_losses_942"#,-23IJXYghF?C
<?9
/?,
zero1_input???????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_1_layer_call_and_return_conditional_losses_992"#,-23IJXYghF?C
<?9
/?,
zero1_input???????????
p

 
? "%?"
?
0?????????
? ?
+__inference_sequential_1_layer_call_fn_1060m"#,-23IJXYghA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
+__inference_sequential_1_layer_call_fn_1093m"#,-23IJXYghA?>
7?4
*?'
inputs???????????
p

 
? "???????????
*__inference_sequential_1_layer_call_fn_560r"#,-23IJXYghF?C
<?9
/?,
zero1_input???????????
p 

 
? "???????????
*__inference_sequential_1_layer_call_fn_892r"#,-23IJXYghF?C
<?9
/?,
zero1_input???????????
p

 
? "???????????
"__inference_signature_wrapper_1027?"#,-23IJXYghM?J
? 
C?@
>
zero1_input/?,
zero1_input???????????";?8
6
activation_3&?#
activation_3??????????
?__inference_zero1_layer_call_and_return_conditional_losses_1256?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
?__inference_zero1_layer_call_and_return_conditional_losses_1262l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
$__inference_zero1_layer_call_fn_1245?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_zero1_layer_call_fn_1250_9?6
/?,
*?'
inputs???????????
? ""????????????