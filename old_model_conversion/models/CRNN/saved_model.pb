Ï#
³!!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Â!
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
shape: *
shared_nameconv3/kernel
v
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*'
_output_shapes
: *
dtype0
m

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv3/bias
f
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes	
:*
dtype0
}
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv4/kernel
v
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*'
_output_shapes
:@*
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
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0

lstm_1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

 *(
shared_namelstm_1/lstm_cell/kernel

+lstm_1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell/kernel* 
_output_shapes
:

 *
dtype0

!lstm_1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	( *2
shared_name#!lstm_1/lstm_cell/recurrent_kernel

5lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm_1/lstm_cell/recurrent_kernel*
_output_shapes
:	( *
dtype0

lstm_1/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namelstm_1/lstm_cell/bias
|
)lstm_1/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_1/lstm_cell/bias*
_output_shapes	
: *
dtype0

NoOpNoOp
8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ð7
valueÆ7BÃ7 B¼7
Æ
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
R
5	variables
6trainable_variables
7regularization_losses
8	keras_api
i
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=_random_generator
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api

Fcell
G
state_spec
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L_random_generator
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
R
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
R
[	variables
\trainable_variables
]regularization_losses
^	keras_api
^
0
1
2
 3
)4
*5
/6
07
_8
`9
a10
U11
V12
^
0
1
2
 3
)4
*5
/6
07
_8
`9
a10
U11
V12
 
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
­
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
%	variables
&trainable_variables
'regularization_losses
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
­
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
5	variables
6trainable_variables
7regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
 
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
«

state_size

_kernel
`recurrent_kernel
abias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
 

_0
`1
a2

_0
`1
a2
 
¿
states
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
 
 
 
 
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
 
 
 
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
 
 
 
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
[	variables
\trainable_variables
]regularization_losses
SQ
VARIABLE_VALUElstm_1/lstm_cell/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!lstm_1/lstm_cell/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElstm_1/lstm_cell/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
 
n
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

_0
`1
a2

_0
`1
a2
 
µ
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 

F0
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

serving_default_zero1_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿôÉ
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_zero1_inputconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biaslstm_1/lstm_cell/kernellstm_1/lstm_cell/bias!lstm_1/lstm_cell/recurrent_kerneldense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4240
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp+lstm_1/lstm_cell/kernel/Read/ReadVariableOp5lstm_1/lstm_cell/recurrent_kernel/Read/ReadVariableOp)lstm_1/lstm_cell/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_6606
ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense_1/kerneldense_1/biaslstm_1/lstm_cell/kernel!lstm_1/lstm_cell/recurrent_kernellstm_1/lstm_cell/bias*
Tin
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_6655£² 
¹
ø
__inference__wrapped_model_2566
zero1_inputK
1sequential_1_conv1_conv2d_readvariableop_resource:@@
2sequential_1_conv1_biasadd_readvariableop_resource:@K
1sequential_1_conv2_conv2d_readvariableop_resource:@ @
2sequential_1_conv2_biasadd_readvariableop_resource: L
1sequential_1_conv3_conv2d_readvariableop_resource: A
2sequential_1_conv3_biasadd_readvariableop_resource:	L
1sequential_1_conv4_conv2d_readvariableop_resource:@@
2sequential_1_conv4_biasadd_readvariableop_resource:@O
;sequential_1_lstm_1_lstm_cell_split_readvariableop_resource:

 L
=sequential_1_lstm_1_lstm_cell_split_1_readvariableop_resource:	 H
5sequential_1_lstm_1_lstm_cell_readvariableop_resource:	( F
3sequential_1_dense_1_matmul_readvariableop_resource:	B
4sequential_1_dense_1_biasadd_readvariableop_resource:
identity¢)sequential_1/conv1/BiasAdd/ReadVariableOp¢(sequential_1/conv1/Conv2D/ReadVariableOp¢)sequential_1/conv2/BiasAdd/ReadVariableOp¢(sequential_1/conv2/Conv2D/ReadVariableOp¢)sequential_1/conv3/BiasAdd/ReadVariableOp¢(sequential_1/conv3/Conv2D/ReadVariableOp¢)sequential_1/conv4/BiasAdd/ReadVariableOp¢(sequential_1/conv4/Conv2D/ReadVariableOp¢+sequential_1/dense_1/BiasAdd/ReadVariableOp¢*sequential_1/dense_1/MatMul/ReadVariableOp¢,sequential_1/lstm_1/lstm_cell/ReadVariableOp¢.sequential_1/lstm_1/lstm_cell/ReadVariableOp_1¢.sequential_1/lstm_1/lstm_cell/ReadVariableOp_2¢.sequential_1/lstm_1/lstm_cell/ReadVariableOp_3¢2sequential_1/lstm_1/lstm_cell/split/ReadVariableOp¢4sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOp¢sequential_1/lstm_1/while
sequential_1/zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 
sequential_1/zero1/PadPadzero1_input(sequential_1/zero1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ¢
(sequential_1/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ò
sequential_1/conv1/Conv2DConv2Dsequential_1/zero1/Pad:output:00sequential_1/conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW*
paddingVALID*
strides

)sequential_1/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ï
sequential_1/conv1/BiasAddBiasAdd"sequential_1/conv1/Conv2D:output:01sequential_1/conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW
sequential_1/conv1/ReluRelu#sequential_1/conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ¢
(sequential_1/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0ø
sequential_1/conv2/Conv2DConv2D%sequential_1/conv1/Relu:activations:00sequential_1/conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW*
paddingVALID*
strides

)sequential_1/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ï
sequential_1/conv2/BiasAddBiasAdd"sequential_1/conv2/Conv2D:output:01sequential_1/conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW
sequential_1/conv2/ReluRelu#sequential_1/conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅÑ
sequential_1/pool1/MaxPoolMaxPool%sequential_1/conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
£
(sequential_1/conv3/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv3_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0ö
sequential_1/conv3/Conv2DConv2D#sequential_1/pool1/MaxPool:output:00sequential_1/conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW*
paddingVALID*
strides

)sequential_1/conv3/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
sequential_1/conv3/BiasAddBiasAdd"sequential_1/conv3/Conv2D:output:01sequential_1/conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW
sequential_1/conv3/ReluRelu#sequential_1/conv3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?£
(sequential_1/conv4/Conv2D/ReadVariableOpReadVariableOp1sequential_1_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0÷
sequential_1/conv4/Conv2DConv2D%sequential_1/conv3/Relu:activations:00sequential_1/conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW*
paddingVALID*
strides

)sequential_1/conv4/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Î
sequential_1/conv4/BiasAddBiasAdd"sequential_1/conv4/Conv2D:output:01sequential_1/conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW
sequential_1/conv4/ReluRelu#sequential_1/conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=Ð
sequential_1/pool2/MaxPoolMaxPool%sequential_1/conv4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides

sequential_1/dropout_1/IdentityIdentity#sequential_1/pool2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5~
%sequential_1/permute_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
 sequential_1/permute_1/transpose	Transpose(sequential_1/dropout_1/Identity:output:0.sequential_1/permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@p
sequential_1/reshape_1/ShapeShape$sequential_1/permute_1/transpose:y:0*
T0*
_output_shapes
:t
*sequential_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_1/reshape_1/strided_sliceStridedSlice%sequential_1/reshape_1/Shape:output:03sequential_1/reshape_1/strided_slice/stack:output:05sequential_1/reshape_1/strided_slice/stack_1:output:05sequential_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :5q
&sequential_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
$sequential_1/reshape_1/Reshape/shapePack-sequential_1/reshape_1/strided_slice:output:0/sequential_1/reshape_1/Reshape/shape/1:output:0/sequential_1/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:µ
sequential_1/reshape_1/ReshapeReshape$sequential_1/permute_1/transpose:y:0-sequential_1/reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
p
sequential_1/lstm_1/ShapeShape'sequential_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
:q
'sequential_1/lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_1/lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_1/lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_1/lstm_1/strided_sliceStridedSlice"sequential_1/lstm_1/Shape:output:00sequential_1/lstm_1/strided_slice/stack:output:02sequential_1/lstm_1/strided_slice/stack_1:output:02sequential_1/lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
sequential_1/lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(
sequential_1/lstm_1/zeros/mulMul*sequential_1/lstm_1/strided_slice:output:0(sequential_1/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: c
 sequential_1/lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
sequential_1/lstm_1/zeros/LessLess!sequential_1/lstm_1/zeros/mul:z:0)sequential_1/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: d
"sequential_1/lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(¯
 sequential_1/lstm_1/zeros/packedPack*sequential_1/lstm_1/strided_slice:output:0+sequential_1/lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_1/lstm_1/zerosFill)sequential_1/lstm_1/zeros/packed:output:0(sequential_1/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!sequential_1/lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(
sequential_1/lstm_1/zeros_1/mulMul*sequential_1/lstm_1/strided_slice:output:0*sequential_1/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: e
"sequential_1/lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è
 sequential_1/lstm_1/zeros_1/LessLess#sequential_1/lstm_1/zeros_1/mul:z:0+sequential_1/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: f
$sequential_1/lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(³
"sequential_1/lstm_1/zeros_1/packedPack*sequential_1/lstm_1/strided_slice:output:0-sequential_1/lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_1/lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_1/lstm_1/zeros_1Fill+sequential_1/lstm_1/zeros_1/packed:output:0*sequential_1/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
"sequential_1/lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ·
sequential_1/lstm_1/transpose	Transpose'sequential_1/reshape_1/Reshape:output:0+sequential_1/lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
l
sequential_1/lstm_1/Shape_1Shape!sequential_1/lstm_1/transpose:y:0*
T0*
_output_shapes
:s
)sequential_1/lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_1/lstm_1/strided_slice_1StridedSlice$sequential_1/lstm_1/Shape_1:output:02sequential_1/lstm_1/strided_slice_1/stack:output:04sequential_1/lstm_1/strided_slice_1/stack_1:output:04sequential_1/lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_1/lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_1/lstm_1/TensorArrayV2TensorListReserve8sequential_1/lstm_1/TensorArrayV2/element_shape:output:0,sequential_1/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;sequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_1/transpose:y:0Rsequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_1/lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_1/lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
#sequential_1/lstm_1/strided_slice_2StridedSlice!sequential_1/lstm_1/transpose:y:02sequential_1/lstm_1/strided_slice_2/stack:output:04sequential_1/lstm_1/strided_slice_2/stack_1:output:04sequential_1/lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_masko
-sequential_1/lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :°
2sequential_1/lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp;sequential_1_lstm_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0ü
#sequential_1/lstm_1/lstm_cell/splitSplit6sequential_1/lstm_1/lstm_cell/split/split_dim:output:0:sequential_1/lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¼
$sequential_1/lstm_1/lstm_cell/MatMulMatMul,sequential_1/lstm_1/strided_slice_2:output:0,sequential_1/lstm_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¾
&sequential_1/lstm_1/lstm_cell/MatMul_1MatMul,sequential_1/lstm_1/strided_slice_2:output:0,sequential_1/lstm_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¾
&sequential_1/lstm_1/lstm_cell/MatMul_2MatMul,sequential_1/lstm_1/strided_slice_2:output:0,sequential_1/lstm_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¾
&sequential_1/lstm_1/lstm_cell/MatMul_3MatMul,sequential_1/lstm_1/strided_slice_2:output:0,sequential_1/lstm_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(q
/sequential_1/lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¯
4sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp=sequential_1_lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0î
%sequential_1/lstm_1/lstm_cell/split_1Split8sequential_1/lstm_1/lstm_cell/split_1/split_dim:output:0<sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splitÂ
%sequential_1/lstm_1/lstm_cell/BiasAddBiasAdd.sequential_1/lstm_1/lstm_cell/MatMul:product:0.sequential_1/lstm_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Æ
'sequential_1/lstm_1/lstm_cell/BiasAdd_1BiasAdd0sequential_1/lstm_1/lstm_cell/MatMul_1:product:0.sequential_1/lstm_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Æ
'sequential_1/lstm_1/lstm_cell/BiasAdd_2BiasAdd0sequential_1/lstm_1/lstm_cell/MatMul_2:product:0.sequential_1/lstm_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Æ
'sequential_1/lstm_1/lstm_cell/BiasAdd_3BiasAdd0sequential_1/lstm_1/lstm_cell/MatMul_3:product:0.sequential_1/lstm_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(£
,sequential_1/lstm_1/lstm_cell/ReadVariableOpReadVariableOp5sequential_1_lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0
1sequential_1/lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
3sequential_1/lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   
3sequential_1/lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
+sequential_1/lstm_1/lstm_cell/strided_sliceStridedSlice4sequential_1/lstm_1/lstm_cell/ReadVariableOp:value:0:sequential_1/lstm_1/lstm_cell/strided_slice/stack:output:0<sequential_1/lstm_1/lstm_cell/strided_slice/stack_1:output:0<sequential_1/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¼
&sequential_1/lstm_1/lstm_cell/MatMul_4MatMul"sequential_1/lstm_1/zeros:output:04sequential_1/lstm_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¾
!sequential_1/lstm_1/lstm_cell/addAddV2.sequential_1/lstm_1/lstm_cell/BiasAdd:output:00sequential_1/lstm_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#sequential_1/lstm_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%sequential_1/lstm_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
!sequential_1/lstm_1/lstm_cell/MulMul%sequential_1/lstm_1/lstm_cell/add:z:0,sequential_1/lstm_1/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(µ
#sequential_1/lstm_1/lstm_cell/Add_1AddV2%sequential_1/lstm_1/lstm_cell/Mul:z:0.sequential_1/lstm_1/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(z
5sequential_1/lstm_1/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ù
3sequential_1/lstm_1/lstm_cell/clip_by_value/MinimumMinimum'sequential_1/lstm_1/lstm_cell/Add_1:z:0>sequential_1/lstm_1/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
-sequential_1/lstm_1/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ù
+sequential_1/lstm_1/lstm_cell/clip_by_valueMaximum7sequential_1/lstm_1/lstm_cell/clip_by_value/Minimum:z:06sequential_1/lstm_1/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¥
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp5sequential_1_lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0
3sequential_1/lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   
5sequential_1/lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   
5sequential_1/lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/lstm_1/lstm_cell/strided_slice_1StridedSlice6sequential_1/lstm_1/lstm_cell/ReadVariableOp_1:value:0<sequential_1/lstm_1/lstm_cell/strided_slice_1/stack:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_1/stack_1:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¾
&sequential_1/lstm_1/lstm_cell/MatMul_5MatMul"sequential_1/lstm_1/zeros:output:06sequential_1/lstm_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
#sequential_1/lstm_1/lstm_cell/add_2AddV20sequential_1/lstm_1/lstm_cell/BiasAdd_1:output:00sequential_1/lstm_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
%sequential_1/lstm_1/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%sequential_1/lstm_1/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?µ
#sequential_1/lstm_1/lstm_cell/Mul_1Mul'sequential_1/lstm_1/lstm_cell/add_2:z:0.sequential_1/lstm_1/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(·
#sequential_1/lstm_1/lstm_cell/Add_3AddV2'sequential_1/lstm_1/lstm_cell/Mul_1:z:0.sequential_1/lstm_1/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(|
7sequential_1/lstm_1/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ý
5sequential_1/lstm_1/lstm_cell/clip_by_value_1/MinimumMinimum'sequential_1/lstm_1/lstm_cell/Add_3:z:0@sequential_1/lstm_1/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
/sequential_1/lstm_1/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ß
-sequential_1/lstm_1/lstm_cell/clip_by_value_1Maximum9sequential_1/lstm_1/lstm_cell/clip_by_value_1/Minimum:z:08sequential_1/lstm_1/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(µ
#sequential_1/lstm_1/lstm_cell/mul_2Mul1sequential_1/lstm_1/lstm_cell/clip_by_value_1:z:0$sequential_1/lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¥
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp5sequential_1_lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0
3sequential_1/lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   
5sequential_1/lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   
5sequential_1/lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/lstm_1/lstm_cell/strided_slice_2StridedSlice6sequential_1/lstm_1/lstm_cell/ReadVariableOp_2:value:0<sequential_1/lstm_1/lstm_cell/strided_slice_2/stack:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_2/stack_1:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¾
&sequential_1/lstm_1/lstm_cell/MatMul_6MatMul"sequential_1/lstm_1/zeros:output:06sequential_1/lstm_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
#sequential_1/lstm_1/lstm_cell/add_4AddV20sequential_1/lstm_1/lstm_cell/BiasAdd_2:output:00sequential_1/lstm_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"sequential_1/lstm_1/lstm_cell/TanhTanh'sequential_1/lstm_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(µ
#sequential_1/lstm_1/lstm_cell/mul_3Mul/sequential_1/lstm_1/lstm_cell/clip_by_value:z:0&sequential_1/lstm_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(°
#sequential_1/lstm_1/lstm_cell/add_5AddV2'sequential_1/lstm_1/lstm_cell/mul_2:z:0'sequential_1/lstm_1/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¥
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp5sequential_1_lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0
3sequential_1/lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   
5sequential_1/lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
5sequential_1/lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-sequential_1/lstm_1/lstm_cell/strided_slice_3StridedSlice6sequential_1/lstm_1/lstm_cell/ReadVariableOp_3:value:0<sequential_1/lstm_1/lstm_cell/strided_slice_3/stack:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_3/stack_1:output:0>sequential_1/lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¾
&sequential_1/lstm_1/lstm_cell/MatMul_7MatMul"sequential_1/lstm_1/zeros:output:06sequential_1/lstm_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
#sequential_1/lstm_1/lstm_cell/add_6AddV20sequential_1/lstm_1/lstm_cell/BiasAdd_3:output:00sequential_1/lstm_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
%sequential_1/lstm_1/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>j
%sequential_1/lstm_1/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?µ
#sequential_1/lstm_1/lstm_cell/Mul_4Mul'sequential_1/lstm_1/lstm_cell/add_6:z:0.sequential_1/lstm_1/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(·
#sequential_1/lstm_1/lstm_cell/Add_7AddV2'sequential_1/lstm_1/lstm_cell/Mul_4:z:0.sequential_1/lstm_1/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(|
7sequential_1/lstm_1/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ý
5sequential_1/lstm_1/lstm_cell/clip_by_value_2/MinimumMinimum'sequential_1/lstm_1/lstm_cell/Add_7:z:0@sequential_1/lstm_1/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
/sequential_1/lstm_1/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ß
-sequential_1/lstm_1/lstm_cell/clip_by_value_2Maximum9sequential_1/lstm_1/lstm_cell/clip_by_value_2/Minimum:z:08sequential_1/lstm_1/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
$sequential_1/lstm_1/lstm_cell/Tanh_1Tanh'sequential_1/lstm_1/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¹
#sequential_1/lstm_1/lstm_cell/mul_5Mul1sequential_1/lstm_1/lstm_cell/clip_by_value_2:z:0(sequential_1/lstm_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
1sequential_1/lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ô
#sequential_1/lstm_1/TensorArrayV2_1TensorListReserve:sequential_1/lstm_1/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_1/lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_1/lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_1/lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_1/lstm_1/whileWhile/sequential_1/lstm_1/while/loop_counter:output:05sequential_1/lstm_1/while/maximum_iterations:output:0!sequential_1/lstm_1/time:output:0,sequential_1/lstm_1/TensorArrayV2_1:handle:0"sequential_1/lstm_1/zeros:output:0$sequential_1/lstm_1/zeros_1:output:0,sequential_1/lstm_1/strided_slice_1:output:0Ksequential_1/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0;sequential_1_lstm_1_lstm_cell_split_readvariableop_resource=sequential_1_lstm_1_lstm_cell_split_1_readvariableop_resource5sequential_1_lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( */
body'R%
#sequential_1_lstm_1_while_body_2412*/
cond'R%
#sequential_1_lstm_1_while_cond_2411*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
Dsequential_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   þ
6sequential_1/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_1/while:output:3Msequential_1/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0|
)sequential_1/lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_1/lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_1/lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_1/lstm_1/strided_slice_3StridedSlice?sequential_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_1/strided_slice_3/stack:output:04sequential_1/lstm_1/strided_slice_3/stack_1:output:04sequential_1/lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_masky
$sequential_1/lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_1/lstm_1/transpose_1	Transpose?sequential_1/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(o
sequential_1/lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
*sequential_1/maxpooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :È
&sequential_1/maxpooling1d_1/ExpandDims
ExpandDims#sequential_1/lstm_1/transpose_1:y:03sequential_1/maxpooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(Ì
#sequential_1/maxpooling1d_1/MaxPoolMaxPool/sequential_1/maxpooling1d_1/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
©
#sequential_1/maxpooling1d_1/SqueezeSqueeze,sequential_1/maxpooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
squeeze_dims
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ±
sequential_1/flatten_1/ReshapeReshape,sequential_1/maxpooling1d_1/Squeeze:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0´
sequential_1/dense_1/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!sequential_1/activation_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+sequential_1/activation_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
NoOpNoOp*^sequential_1/conv1/BiasAdd/ReadVariableOp)^sequential_1/conv1/Conv2D/ReadVariableOp*^sequential_1/conv2/BiasAdd/ReadVariableOp)^sequential_1/conv2/Conv2D/ReadVariableOp*^sequential_1/conv3/BiasAdd/ReadVariableOp)^sequential_1/conv3/Conv2D/ReadVariableOp*^sequential_1/conv4/BiasAdd/ReadVariableOp)^sequential_1/conv4/Conv2D/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/lstm_1/lstm_cell/ReadVariableOp/^sequential_1/lstm_1/lstm_cell/ReadVariableOp_1/^sequential_1/lstm_1/lstm_cell/ReadVariableOp_2/^sequential_1/lstm_1/lstm_cell/ReadVariableOp_33^sequential_1/lstm_1/lstm_cell/split/ReadVariableOp5^sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOp^sequential_1/lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2V
)sequential_1/conv1/BiasAdd/ReadVariableOp)sequential_1/conv1/BiasAdd/ReadVariableOp2T
(sequential_1/conv1/Conv2D/ReadVariableOp(sequential_1/conv1/Conv2D/ReadVariableOp2V
)sequential_1/conv2/BiasAdd/ReadVariableOp)sequential_1/conv2/BiasAdd/ReadVariableOp2T
(sequential_1/conv2/Conv2D/ReadVariableOp(sequential_1/conv2/Conv2D/ReadVariableOp2V
)sequential_1/conv3/BiasAdd/ReadVariableOp)sequential_1/conv3/BiasAdd/ReadVariableOp2T
(sequential_1/conv3/Conv2D/ReadVariableOp(sequential_1/conv3/Conv2D/ReadVariableOp2V
)sequential_1/conv4/BiasAdd/ReadVariableOp)sequential_1/conv4/BiasAdd/ReadVariableOp2T
(sequential_1/conv4/Conv2D/ReadVariableOp(sequential_1/conv4/Conv2D/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/lstm_1/lstm_cell/ReadVariableOp,sequential_1/lstm_1/lstm_cell/ReadVariableOp2`
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_1.sequential_1/lstm_1/lstm_cell/ReadVariableOp_12`
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_2.sequential_1/lstm_1/lstm_cell/ReadVariableOp_22`
.sequential_1/lstm_1/lstm_cell/ReadVariableOp_3.sequential_1/lstm_1/lstm_cell/ReadVariableOp_32h
2sequential_1/lstm_1/lstm_cell/split/ReadVariableOp2sequential_1/lstm_1/lstm_cell/split/ReadVariableOp2l
4sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOp4sequential_1/lstm_1/lstm_cell/split_1/ReadVariableOp26
sequential_1/lstm_1/whilesequential_1/lstm_1/while:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
¾
D
(__inference_permute_1_layer_call_fn_5132

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_3217h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
Ü
É
#sequential_1_lstm_1_while_cond_2411D
@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counterJ
Fsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations)
%sequential_1_lstm_1_while_placeholder+
'sequential_1_lstm_1_while_placeholder_1+
'sequential_1_lstm_1_while_placeholder_2+
'sequential_1_lstm_1_while_placeholder_3F
Bsequential_1_lstm_1_while_less_sequential_1_lstm_1_strided_slice_1Z
Vsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_2411___redundant_placeholder0Z
Vsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_2411___redundant_placeholder1Z
Vsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_2411___redundant_placeholder2Z
Vsequential_1_lstm_1_while_sequential_1_lstm_1_while_cond_2411___redundant_placeholder3&
"sequential_1_lstm_1_while_identity
²
sequential_1/lstm_1/while/LessLess%sequential_1_lstm_1_while_placeholderBsequential_1_lstm_1_while_less_sequential_1_lstm_1_strided_slice_1*
T0*
_output_shapes
: s
"sequential_1/lstm_1/while/IdentityIdentity"sequential_1/lstm_1/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_1_lstm_1_while_identity+sequential_1/lstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
=
ñ
@__inference_lstm_1_layer_call_and_return_conditional_losses_2831

inputs"
lstm_cell_2749:

 
lstm_cell_2751:	 !
lstm_cell_2753:	( 
identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maská
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_2749lstm_cell_2751lstm_cell_2753*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2748n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_2749lstm_cell_2751lstm_cell_2753*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_2762*
condR
while_cond_2761*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Â

&__inference_dense_1_layer_call_fn_6312

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_3533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
"
Â
while_body_2762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_2786_0:

 %
while_lstm_cell_2788_0:	 )
while_lstm_cell_2790_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_2786:

 #
while_lstm_cell_2788:	 '
while_lstm_cell_2790:	( ¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_2786_0while_lstm_cell_2788_0while_lstm_cell_2790_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2748Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0".
while_lstm_cell_2786while_lstm_cell_2786_0".
while_lstm_cell_2788while_lstm_cell_2788_0".
while_lstm_cell_2790while_lstm_cell_2790_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
·å

F__inference_sequential_1_layer_call_and_return_conditional_losses_4953

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@ 3
%conv2_biasadd_readvariableop_resource: ?
$conv3_conv2d_readvariableop_resource: 4
%conv3_biasadd_readvariableop_resource:	?
$conv4_conv2d_readvariableop_resource:@3
%conv4_biasadd_readvariableop_resource:@B
.lstm_1_lstm_cell_split_readvariableop_resource:

 ?
0lstm_1_lstm_cell_split_1_readvariableop_resource:	 ;
(lstm_1_lstm_cell_readvariableop_resource:	( 9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv1/BiasAdd/ReadVariableOp¢conv1/Conv2D/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢conv2/Conv2D/ReadVariableOp¢conv3/BiasAdd/ReadVariableOp¢conv3/Conv2D/ReadVariableOp¢conv4/BiasAdd/ReadVariableOp¢conv4/Conv2D/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢lstm_1/lstm_cell/ReadVariableOp¢!lstm_1/lstm_cell/ReadVariableOp_1¢!lstm_1/lstm_cell/ReadVariableOp_2¢!lstm_1/lstm_cell/ReadVariableOp_3¢%lstm_1/lstm_cell/split/ReadVariableOp¢'lstm_1/lstm_cell/split_1/ReadVariableOp¢lstm_1/while
zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 q
	zero1/PadPadinputszero1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ë
conv1/Conv2DConv2Dzero1/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¨
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHWf

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ñ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW*
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¨
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHWf

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ·
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A*
data_formatNCHW*
ksize
*
paddingVALID*
strides

conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0Ï
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW*
paddingVALID*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¨
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHWf

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ð
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHWe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=¶
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMulpool2/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5]
dropout_1/dropout/ShapeShapepool2/MaxPool:output:0*
T0*
_output_shapes
:¨
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ì
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5q
permute_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
permute_1/transpose	Transposedropout_1/dropout/Mul_1:z:0!permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@V
reshape_1/ShapeShapepermute_1/transpose:y:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :5d
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ·
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapepermute_1/transpose:y:0 reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
V
lstm_1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èn
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: X
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èt
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_1/transpose	Transposereshape_1/Reshape:output:0lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskb
 lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp.lstm_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0Õ
lstm_1/lstm_cell/splitSplit)lstm_1/lstm_cell/split/split_dim:output:0-lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_1/lstm_cell/MatMulMatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_1MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_2MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_3MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
"lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0Ç
lstm_1/lstm_cell/split_1Split+lstm_1/lstm_cell/split_1/split_dim:output:0/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_1/lstm_cell/BiasAddBiasAdd!lstm_1/lstm_cell/MatMul:product:0!lstm_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_1BiasAdd#lstm_1/lstm_cell/MatMul_1:product:0!lstm_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_2BiasAdd#lstm_1/lstm_cell/MatMul_2:product:0!lstm_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_3BiasAdd#lstm_1/lstm_cell/MatMul_3:product:0!lstm_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/ReadVariableOpReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0u
$lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   w
&lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      À
lstm_1/lstm_cell/strided_sliceStridedSlice'lstm_1/lstm_cell/ReadVariableOp:value:0-lstm_1/lstm_cell/strided_slice/stack:output:0/lstm_1/lstm_cell/strided_slice/stack_1:output:0/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_4MatMullstm_1/zeros:output:0'lstm_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/addAddV2!lstm_1/lstm_cell/BiasAdd:output:0#lstm_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
lstm_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/MulMullstm_1/lstm_cell/add:z:0lstm_1/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_1AddV2lstm_1/lstm_cell/Mul:z:0!lstm_1/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&lstm_1/lstm_cell/clip_by_value/MinimumMinimumlstm_1/lstm_cell/Add_1:z:01lstm_1/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(e
 lstm_1/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
lstm_1/lstm_cell/clip_by_valueMaximum*lstm_1/lstm_cell/clip_by_value/Minimum:z:0)lstm_1/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   y
(lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   y
(lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_1StridedSlice)lstm_1/lstm_cell/ReadVariableOp_1:value:0/lstm_1/lstm_cell/strided_slice_1/stack:output:01lstm_1/lstm_cell/strided_slice_1/stack_1:output:01lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_5MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_2AddV2#lstm_1/lstm_cell/BiasAdd_1:output:0#lstm_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_1/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/Mul_1Mullstm_1/lstm_cell/add_2:z:0!lstm_1/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_3AddV2lstm_1/lstm_cell/Mul_1:z:0!lstm_1/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
*lstm_1/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
(lstm_1/lstm_cell/clip_by_value_1/MinimumMinimumlstm_1/lstm_cell/Add_3:z:03lstm_1/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
"lstm_1/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¸
 lstm_1/lstm_cell/clip_by_value_1Maximum,lstm_1/lstm_cell/clip_by_value_1/Minimum:z:0+lstm_1/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_2Mul$lstm_1/lstm_cell/clip_by_value_1:z:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   y
(lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   y
(lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_2StridedSlice)lstm_1/lstm_cell/ReadVariableOp_2:value:0/lstm_1/lstm_cell/strided_slice_2/stack:output:01lstm_1/lstm_cell/strided_slice_2/stack_1:output:01lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_6MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_4AddV2#lstm_1/lstm_cell/BiasAdd_2:output:0#lstm_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
lstm_1/lstm_cell/TanhTanhlstm_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_3Mul"lstm_1/lstm_cell/clip_by_value:z:0lstm_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_5AddV2lstm_1/lstm_cell/mul_2:z:0lstm_1/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   y
(lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_3StridedSlice)lstm_1/lstm_cell/ReadVariableOp_3:value:0/lstm_1/lstm_cell/strided_slice_3/stack:output:01lstm_1/lstm_cell/strided_slice_3/stack_1:output:01lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_7MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_6AddV2#lstm_1/lstm_cell/BiasAdd_3:output:0#lstm_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_1/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/Mul_4Mullstm_1/lstm_cell/add_6:z:0!lstm_1/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_7AddV2lstm_1/lstm_cell/Mul_4:z:0!lstm_1/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
*lstm_1/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
(lstm_1/lstm_cell/clip_by_value_2/MinimumMinimumlstm_1/lstm_cell/Add_7:z:03lstm_1/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
"lstm_1/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¸
 lstm_1/lstm_cell/clip_by_value_2Maximum,lstm_1/lstm_cell/clip_by_value_2/Minimum:z:0+lstm_1/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
lstm_1/lstm_cell/Tanh_1Tanhlstm_1/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_5Mul$lstm_1/lstm_cell/clip_by_value_2:z:0lstm_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Í
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Í
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_1_lstm_cell_split_readvariableop_resource0lstm_1_lstm_cell_split_1_readvariableop_resource(lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_1_while_body_4799*"
condR
lstm_1_while_cond_4798*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ×
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    _
maxpooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¡
maxpooling1d_1/ExpandDims
ExpandDimslstm_1/transpose_1:y:0&maxpooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(²
maxpooling1d_1/MaxPoolMaxPool"maxpooling1d_1/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides

maxpooling1d_1/SqueezeSqueezemaxpooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
squeeze_dims
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_1/ReshapeReshapemaxpooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityactivation_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^lstm_1/lstm_cell/ReadVariableOp"^lstm_1/lstm_cell/ReadVariableOp_1"^lstm_1/lstm_cell/ReadVariableOp_2"^lstm_1/lstm_cell/ReadVariableOp_3&^lstm_1/lstm_cell/split/ReadVariableOp(^lstm_1/lstm_cell/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm_1/lstm_cell/ReadVariableOplstm_1/lstm_cell/ReadVariableOp2F
!lstm_1/lstm_cell/ReadVariableOp_1!lstm_1/lstm_cell/ReadVariableOp_12F
!lstm_1/lstm_cell/ReadVariableOp_2!lstm_1/lstm_cell/ReadVariableOp_22F
!lstm_1/lstm_cell/ReadVariableOp_3!lstm_1/lstm_cell/ReadVariableOp_32N
%lstm_1/lstm_cell/split/ReadVariableOp%lstm_1/lstm_cell/split/ReadVariableOp2R
'lstm_1/lstm_cell/split_1/ReadVariableOp'lstm_1/lstm_cell/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
­x
	
while_body_5330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
¸
é
"__inference_signature_wrapper_4240
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: 
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:

 
	unknown_8:	 
	unknown_9:	( 

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_2566o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
¾
D
(__inference_dropout_1_layer_call_fn_5100

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3210h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
ë

$__inference_conv3_layer_call_fn_5044

inputs"
unknown: 
	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_3176y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ¥A: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A
 
_user_specified_nameinputs
è
ò
+__inference_sequential_1_layer_call_fn_3576
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: 
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:

 
	unknown_8:	 
	unknown_9:	( 

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_3547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
´
[
?__inference_pool2_layer_call_and_return_conditional_losses_3203

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=
 
_user_specified_nameinputs

[
?__inference_pool1_layer_call_and_return_conditional_losses_2588

inputs
identity¹
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý9
é
F__inference_sequential_1_layer_call_and_return_conditional_losses_4057

inputs$

conv1_4016:@

conv1_4018:@$

conv2_4021:@ 

conv2_4023: %

conv3_4027: 

conv3_4029:	%

conv4_4032:@

conv4_4034:@
lstm_1_4041:

 
lstm_1_4043:	 
lstm_1_4045:	( 
dense_1_4050:	
dense_1_4052:
identity¢conv1/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢conv3/StatefulPartitionedCall¢conv4/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall½
zero1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_3123
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0
conv1_4016
conv1_4018*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_3136
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
conv2_4021
conv2_4023*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_3153Ü
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_3163
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0
conv3_4027
conv3_4029*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_3176
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0
conv4_4032
conv4_4034*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_3193Û
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_3203ë
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3921ç
permute_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_3217Ü
reshape_1/PartitionedCallPartitionedCall"permute_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_1_4041lstm_1_4043lstm_1_4045*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3881ê
maxpooling1d_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513Ý
flatten_1/PartitionedCallPartitionedCall'maxpooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_4050dense_1_4052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_3533ã
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_3544t
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
¨
D
(__inference_flatten_1_layer_call_fn_6297

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
ë

$__inference_conv2_layer_call_fn_5004

inputs!
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_3153y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@òÇ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ
 
_user_specified_nameinputs
ï	
Å
lstm_1_while_cond_4469*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1@
<lstm_1_while_lstm_1_while_cond_4469___redundant_placeholder0@
<lstm_1_while_lstm_1_while_cond_4469___redundant_placeholder1@
<lstm_1_while_lstm_1_while_cond_4469___redundant_placeholder2@
<lstm_1_while_lstm_1_while_cond_4469___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
ç
ò
(__inference_lstm_cell_layer_call_fn_6349

inputs
states_0
states_1
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/1
±

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5122

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
î
_
C__inference_permute_1_layer_call_and_return_conditional_losses_5144

inputs
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             q
	transpose	Transposeinputstranspose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@]
IdentityIdentitytranspose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
ï	
Å
lstm_1_while_cond_4798*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3,
(lstm_1_while_less_lstm_1_strided_slice_1@
<lstm_1_while_lstm_1_while_cond_4798___redundant_placeholder0@
<lstm_1_while_lstm_1_while_cond_4798___redundant_placeholder1@
<lstm_1_while_lstm_1_while_cond_4798___redundant_placeholder2@
<lstm_1_while_lstm_1_while_cond_4798___redundant_placeholder3
lstm_1_while_identity
~
lstm_1/while/LessLesslstm_1_while_placeholder(lstm_1_while_less_lstm_1_strided_slice_1*
T0*
_output_shapes
: Y
lstm_1/while/IdentityIdentitylstm_1/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_1_while_identitylstm_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
­x
	
while_body_3357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
¼
@
$__inference_pool1_layer_call_fn_5025

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_3163i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ
 
_user_specified_nameinputs
£
@
$__inference_zero1_layer_call_fn_4958

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_2576
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ó
A__inference_dense_1_layer_call_and_return_conditional_losses_3533

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Ê
@__inference_lstm_1_layer_call_and_return_conditional_losses_3881

inputs;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3740*
condR
while_cond_3739*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
"
Â
while_body_3018
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*
while_lstm_cell_3042_0:

 %
while_lstm_cell_3044_0:	 )
while_lstm_cell_3046_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor(
while_lstm_cell_3042:

 #
while_lstm_cell_3044:	 '
while_lstm_cell_3046:	( ¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3042_0while_lstm_cell_3044_0while_lstm_cell_3046_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2951Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0".
while_lstm_cell_3042while_lstm_cell_3042_0".
while_lstm_cell_3044while_lstm_cell_3044_0".
while_lstm_cell_3046while_lstm_cell_3046_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
½
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_6303

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
«
¹
while_cond_3017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3017___redundant_placeholder02
.while_while_cond_3017___redundant_placeholder12
.while_while_cond_3017___redundant_placeholder22
.while_while_cond_3017___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
Í
d
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3108

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ØÝ

F__inference_sequential_1_layer_call_and_return_conditional_losses_4624

inputs>
$conv1_conv2d_readvariableop_resource:@3
%conv1_biasadd_readvariableop_resource:@>
$conv2_conv2d_readvariableop_resource:@ 3
%conv2_biasadd_readvariableop_resource: ?
$conv3_conv2d_readvariableop_resource: 4
%conv3_biasadd_readvariableop_resource:	?
$conv4_conv2d_readvariableop_resource:@3
%conv4_biasadd_readvariableop_resource:@B
.lstm_1_lstm_cell_split_readvariableop_resource:

 ?
0lstm_1_lstm_cell_split_1_readvariableop_resource:	 ;
(lstm_1_lstm_cell_readvariableop_resource:	( 9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv1/BiasAdd/ReadVariableOp¢conv1/Conv2D/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢conv2/Conv2D/ReadVariableOp¢conv3/BiasAdd/ReadVariableOp¢conv3/Conv2D/ReadVariableOp¢conv4/BiasAdd/ReadVariableOp¢conv4/Conv2D/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢lstm_1/lstm_cell/ReadVariableOp¢!lstm_1/lstm_cell/ReadVariableOp_1¢!lstm_1/lstm_cell/ReadVariableOp_2¢!lstm_1/lstm_cell/ReadVariableOp_3¢%lstm_1/lstm_cell/split/ReadVariableOp¢'lstm_1/lstm_cell/split_1/ReadVariableOp¢lstm_1/while
zero1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 q
	zero1/PadPadinputszero1/Pad/paddings:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ë
conv1/Conv2DConv2Dzero1/Pad:output:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW*
paddingVALID*
strides
~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¨
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHWf

conv1/ReluReluconv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ñ
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW*
paddingVALID*
strides
~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¨
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHWf

conv2/ReluReluconv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ·
pool1/MaxPoolMaxPoolconv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A*
data_formatNCHW*
ksize
*
paddingVALID*
strides

conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*'
_output_shapes
: *
dtype0Ï
conv3/Conv2DConv2Dpool1/MaxPool:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW*
paddingVALID*
strides

conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¨
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHWf

conv3/ReluReluconv3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ð
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW*
paddingVALID*
strides
~
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHWe

conv4/ReluReluconv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=¶
pool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
p
dropout_1/IdentityIdentitypool2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5q
permute_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
permute_1/transpose	Transposedropout_1/Identity:output:0!permute_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@V
reshape_1/ShapeShapepermute_1/transpose:y:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :5d
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ·
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapepermute_1/transpose:y:0 reshape_1/Reshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
V
lstm_1/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
lstm_1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èn
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: X
lstm_1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èt
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
lstm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_1/transpose	Transposereshape_1/Reshape:output:0lstm_1/transpose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maskb
 lstm_1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%lstm_1/lstm_cell/split/ReadVariableOpReadVariableOp.lstm_1_lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0Õ
lstm_1/lstm_cell/splitSplit)lstm_1/lstm_cell/split/split_dim:output:0-lstm_1/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_1/lstm_cell/MatMulMatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_1MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_2MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/MatMul_3MatMullstm_1/strided_slice_2:output:0lstm_1/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
"lstm_1/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
'lstm_1/lstm_cell/split_1/ReadVariableOpReadVariableOp0lstm_1_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0Ç
lstm_1/lstm_cell/split_1Split+lstm_1/lstm_cell/split_1/split_dim:output:0/lstm_1/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_1/lstm_cell/BiasAddBiasAdd!lstm_1/lstm_cell/MatMul:product:0!lstm_1/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_1BiasAdd#lstm_1/lstm_cell/MatMul_1:product:0!lstm_1/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_2BiasAdd#lstm_1/lstm_cell/MatMul_2:product:0!lstm_1/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/BiasAdd_3BiasAdd#lstm_1/lstm_cell/MatMul_3:product:0!lstm_1/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/ReadVariableOpReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0u
$lstm_1/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&lstm_1/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   w
&lstm_1/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      À
lstm_1/lstm_cell/strided_sliceStridedSlice'lstm_1/lstm_cell/ReadVariableOp:value:0-lstm_1/lstm_cell/strided_slice/stack:output:0/lstm_1/lstm_cell/strided_slice/stack_1:output:0/lstm_1/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_4MatMullstm_1/zeros:output:0'lstm_1/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/addAddV2!lstm_1/lstm_cell/BiasAdd:output:0#lstm_1/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
lstm_1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/MulMullstm_1/lstm_cell/add:z:0lstm_1/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_1AddV2lstm_1/lstm_cell/Mul:z:0!lstm_1/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
&lstm_1/lstm_cell/clip_by_value/MinimumMinimumlstm_1/lstm_cell/Add_1:z:01lstm_1/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(e
 lstm_1/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ²
lstm_1/lstm_cell/clip_by_valueMaximum*lstm_1/lstm_cell/clip_by_value/Minimum:z:0)lstm_1/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_1ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   y
(lstm_1/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   y
(lstm_1/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_1StridedSlice)lstm_1/lstm_cell/ReadVariableOp_1:value:0/lstm_1/lstm_cell/strided_slice_1/stack:output:01lstm_1/lstm_cell/strided_slice_1/stack_1:output:01lstm_1/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_5MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_2AddV2#lstm_1/lstm_cell/BiasAdd_1:output:0#lstm_1/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_1/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/Mul_1Mullstm_1/lstm_cell/add_2:z:0!lstm_1/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_3AddV2lstm_1/lstm_cell/Mul_1:z:0!lstm_1/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
*lstm_1/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
(lstm_1/lstm_cell/clip_by_value_1/MinimumMinimumlstm_1/lstm_cell/Add_3:z:03lstm_1/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
"lstm_1/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¸
 lstm_1/lstm_cell/clip_by_value_1Maximum,lstm_1/lstm_cell/clip_by_value_1/Minimum:z:0+lstm_1/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_2Mul$lstm_1/lstm_cell/clip_by_value_1:z:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_2ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   y
(lstm_1/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   y
(lstm_1/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_2StridedSlice)lstm_1/lstm_cell/ReadVariableOp_2:value:0/lstm_1/lstm_cell/strided_slice_2/stack:output:01lstm_1/lstm_cell/strided_slice_2/stack_1:output:01lstm_1/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_6MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_4AddV2#lstm_1/lstm_cell/BiasAdd_2:output:0#lstm_1/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
lstm_1/lstm_cell/TanhTanhlstm_1/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_3Mul"lstm_1/lstm_cell/clip_by_value:z:0lstm_1/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_5AddV2lstm_1/lstm_cell/mul_2:z:0lstm_1/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
!lstm_1/lstm_cell/ReadVariableOp_3ReadVariableOp(lstm_1_lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0w
&lstm_1/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   y
(lstm_1/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(lstm_1/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 lstm_1/lstm_cell/strided_slice_3StridedSlice)lstm_1/lstm_cell/ReadVariableOp_3:value:0/lstm_1/lstm_cell/strided_slice_3/stack:output:01lstm_1/lstm_cell/strided_slice_3/stack_1:output:01lstm_1/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_1/lstm_cell/MatMul_7MatMullstm_1/zeros:output:0)lstm_1/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/add_6AddV2#lstm_1/lstm_cell/BiasAdd_3:output:0#lstm_1/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_1/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>]
lstm_1/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/lstm_cell/Mul_4Mullstm_1/lstm_cell/add_6:z:0!lstm_1/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/Add_7AddV2lstm_1/lstm_cell/Mul_4:z:0!lstm_1/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(o
*lstm_1/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
(lstm_1/lstm_cell/clip_by_value_2/MinimumMinimumlstm_1/lstm_cell/Add_7:z:03lstm_1/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
"lstm_1/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¸
 lstm_1/lstm_cell/clip_by_value_2Maximum,lstm_1/lstm_cell/clip_by_value_2/Minimum:z:0+lstm_1/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
lstm_1/lstm_cell/Tanh_1Tanhlstm_1/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/lstm_cell/mul_5Mul$lstm_1/lstm_cell/clip_by_value_2:z:0lstm_1/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
$lstm_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Í
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_1/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Í
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0.lstm_1_lstm_cell_split_readvariableop_resource0lstm_1_lstm_cell_split_1_readvariableop_resource(lstm_1_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *"
bodyR
lstm_1_while_body_4470*"
condR
lstm_1_while_cond_4469*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ×
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStacklstm_1/while:output:3@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0o
lstm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(b
lstm_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    _
maxpooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¡
maxpooling1d_1/ExpandDims
ExpandDimslstm_1/transpose_1:y:0&maxpooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(²
maxpooling1d_1/MaxPoolMaxPool"maxpooling1d_1/ExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides

maxpooling1d_1/SqueezeSqueezemaxpooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
squeeze_dims
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_1/ReshapeReshapemaxpooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
activation_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityactivation_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^lstm_1/lstm_cell/ReadVariableOp"^lstm_1/lstm_cell/ReadVariableOp_1"^lstm_1/lstm_cell/ReadVariableOp_2"^lstm_1/lstm_cell/ReadVariableOp_3&^lstm_1/lstm_cell/split/ReadVariableOp(^lstm_1/lstm_cell/split_1/ReadVariableOp^lstm_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
lstm_1/lstm_cell/ReadVariableOplstm_1/lstm_cell/ReadVariableOp2F
!lstm_1/lstm_cell/ReadVariableOp_1!lstm_1/lstm_cell/ReadVariableOp_12F
!lstm_1/lstm_cell/ReadVariableOp_2!lstm_1/lstm_cell/ReadVariableOp_22F
!lstm_1/lstm_cell/ReadVariableOp_3!lstm_1/lstm_cell/ReadVariableOp_32N
%lstm_1/lstm_cell/split/ReadVariableOp%lstm_1/lstm_cell/split/ReadVariableOp2R
'lstm_1/lstm_cell/split_1/ReadVariableOp'lstm_1/lstm_cell/split_1/ReadVariableOp2
lstm_1/whilelstm_1/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
ê

$__inference_conv4_layer_call_fn_5064

inputs"
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_3193x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ£?: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?
 
_user_specified_nameinputs

Ì
@__inference_lstm_1_layer_call_and_return_conditional_losses_5471
inputs_0;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_5330*
condR
while_cond_5329*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
ß
d
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
 
_user_specified_nameinputs
ß
d
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6292

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
ksize
*
paddingVALID*
strides
q
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
squeeze_dims
\
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
 
_user_specified_nameinputs
Ù
Ê
@__inference_lstm_1_layer_call_and_return_conditional_losses_6001

inputs;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_5860*
condR
while_cond_5859*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
¼
ú
?__inference_conv3_layer_call_and_return_conditional_losses_5055

inputs9
conv2d_readvariableop_resource: .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ¥A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A
 
_user_specified_nameinputs

a
(__inference_dropout_1_layer_call_fn_5105

inputs
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3921w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@522
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
Ù
Ê
@__inference_lstm_1_layer_call_and_return_conditional_losses_6266

inputs;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_6125*
condR
while_cond_6124*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
ç
õ

lstm_1_while_body_4470*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0J
6lstm_1_while_lstm_cell_split_readvariableop_resource_0:

 G
8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	 C
0lstm_1_while_lstm_cell_readvariableop_resource_0:	( 
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorH
4lstm_1_while_lstm_cell_split_readvariableop_resource:

 E
6lstm_1_while_lstm_cell_split_1_readvariableop_resource:	 A
.lstm_1_while_lstm_cell_readvariableop_resource:	( ¢%lstm_1/while/lstm_cell/ReadVariableOp¢'lstm_1/while/lstm_cell/ReadVariableOp_1¢'lstm_1/while/lstm_cell/ReadVariableOp_2¢'lstm_1/while/lstm_cell/ReadVariableOp_3¢+lstm_1/while/lstm_cell/split/ReadVariableOp¢-lstm_1/while/lstm_cell/split_1/ReadVariableOp
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
&lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
+lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0ç
lstm_1/while/lstm_cell/splitSplit/lstm_1/while/lstm_cell/split/split_dim:output:03lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¹
lstm_1/while/lstm_cell/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
(lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : £
-lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ù
lstm_1/while/lstm_cell/split_1Split1lstm_1/while/lstm_cell/split_1/split_dim:output:05lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split­
lstm_1/while/lstm_cell/BiasAddBiasAdd'lstm_1/while/lstm_cell/MatMul:product:0'lstm_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_1BiasAdd)lstm_1/while/lstm_cell/MatMul_1:product:0'lstm_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_2BiasAdd)lstm_1/while/lstm_cell/MatMul_2:product:0'lstm_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_3BiasAdd)lstm_1/while/lstm_cell/MatMul_3:product:0'lstm_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
%lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0{
*lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   }
,lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_1/while/lstm_cell/strided_sliceStridedSlice-lstm_1/while/lstm_cell/ReadVariableOp:value:03lstm_1/while/lstm_cell/strided_slice/stack:output:05lstm_1/while/lstm_cell/strided_slice/stack_1:output:05lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¦
lstm_1/while/lstm_cell/MatMul_4MatMullstm_1_while_placeholder_2-lstm_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(©
lstm_1/while/lstm_cell/addAddV2'lstm_1/while/lstm_cell/BiasAdd:output:0)lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
lstm_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/while/lstm_cell/MulMullstm_1/while/lstm_cell/add:z:0%lstm_1/while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ( 
lstm_1/while/lstm_cell/Add_1AddV2lstm_1/while/lstm_cell/Mul:z:0'lstm_1/while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(s
.lstm_1/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
,lstm_1/while/lstm_cell/clip_by_value/MinimumMinimum lstm_1/while/lstm_cell/Add_1:z:07lstm_1/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
&lstm_1/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
$lstm_1/while/lstm_cell/clip_by_valueMaximum0lstm_1/while/lstm_cell/clip_by_value/Minimum:z:0/lstm_1/while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   
.lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   
.lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_1StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_1:value:05lstm_1/while/lstm_cell/strided_slice_1/stack:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_5MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_2AddV2)lstm_1/while/lstm_cell/BiasAdd_1:output:0)lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
lstm_1/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ? 
lstm_1/while/lstm_cell/Mul_1Mul lstm_1/while/lstm_cell/add_2:z:0'lstm_1/while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¢
lstm_1/while/lstm_cell/Add_3AddV2 lstm_1/while/lstm_cell/Mul_1:z:0'lstm_1/while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
0lstm_1/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
.lstm_1/while/lstm_cell/clip_by_value_1/MinimumMinimum lstm_1/while/lstm_cell/Add_3:z:09lstm_1/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
&lstm_1/while/lstm_cell/clip_by_value_1Maximum2lstm_1/while/lstm_cell/clip_by_value_1/Minimum:z:01lstm_1/while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/lstm_cell/mul_2Mul*lstm_1/while/lstm_cell/clip_by_value_1:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   
.lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   
.lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_2StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_2:value:05lstm_1/while/lstm_cell/strided_slice_2/stack:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_6MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_4AddV2)lstm_1/while/lstm_cell/BiasAdd_2:output:0)lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
lstm_1/while/lstm_cell/TanhTanh lstm_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ( 
lstm_1/while/lstm_cell/mul_3Mul(lstm_1/while/lstm_cell/clip_by_value:z:0lstm_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/lstm_cell/add_5AddV2 lstm_1/while/lstm_cell/mul_2:z:0 lstm_1/while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   
.lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_3StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_3:value:05lstm_1/while/lstm_cell/strided_slice_3/stack:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_7MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_6AddV2)lstm_1/while/lstm_cell/BiasAdd_3:output:0)lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
lstm_1/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ? 
lstm_1/while/lstm_cell/Mul_4Mul lstm_1/while/lstm_cell/add_6:z:0'lstm_1/while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¢
lstm_1/while/lstm_cell/Add_7AddV2 lstm_1/while/lstm_cell/Mul_4:z:0'lstm_1/while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
0lstm_1/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
.lstm_1/while/lstm_cell/clip_by_value_2/MinimumMinimum lstm_1/while/lstm_cell/Add_7:z:09lstm_1/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
&lstm_1/while/lstm_cell/clip_by_value_2Maximum2lstm_1/while/lstm_cell/clip_by_value_2/Minimum:z:01lstm_1/while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_1/while/lstm_cell/Tanh_1Tanh lstm_1/while/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¤
lstm_1/while/lstm_cell/mul_5Mul*lstm_1/while/lstm_cell/clip_by_value_2:z:0!lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Þ
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder lstm_1/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_4Identity lstm_1/while/lstm_cell/mul_5:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/Identity_5Identity lstm_1/while/lstm_cell/add_5:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(×
lstm_1/while/NoOpNoOp&^lstm_1/while/lstm_cell/ReadVariableOp(^lstm_1/while/lstm_cell/ReadVariableOp_1(^lstm_1/while/lstm_cell/ReadVariableOp_2(^lstm_1/while/lstm_cell/ReadVariableOp_3,^lstm_1/while/lstm_cell/split/ReadVariableOp.^lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"b
.lstm_1_while_lstm_cell_readvariableop_resource0lstm_1_while_lstm_cell_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_split_1_readvariableop_resource8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_1_while_lstm_cell_split_readvariableop_resource6lstm_1_while_lstm_cell_split_readvariableop_resource_0"Ä
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2N
%lstm_1/while/lstm_cell/ReadVariableOp%lstm_1/while/lstm_cell/ReadVariableOp2R
'lstm_1/while/lstm_cell/ReadVariableOp_1'lstm_1/while/lstm_cell/ReadVariableOp_12R
'lstm_1/while/lstm_cell/ReadVariableOp_2'lstm_1/while/lstm_cell/ReadVariableOp_22R
'lstm_1/while/lstm_cell/ReadVariableOp_3'lstm_1/while/lstm_cell/ReadVariableOp_32Z
+lstm_1/while/lstm_cell/split/ReadVariableOp+lstm_1/while/lstm_cell/split/ReadVariableOp2^
-lstm_1/while/lstm_cell/split_1/ReadVariableOp-lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
º
ø
?__inference_conv1_layer_call_and_return_conditional_losses_4995

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôÉ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
¤
G
+__inference_activation_1_layer_call_fn_6327

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_3544`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3921

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
ÿ
³
%__inference_lstm_1_layer_call_fn_5206

inputs
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3881s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
ù
[
?__inference_zero1_layer_call_and_return_conditional_losses_4975

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
:ÿÿÿÿÿÿÿÿÿôÉ^
IdentityIdentityPad:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôÉ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs

I
-__inference_maxpooling1d_1_layer_call_fn_6271

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3108v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
[
?__inference_zero1_layer_call_and_return_conditional_losses_2576

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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¹
while_cond_3739
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3739___redundant_placeholder02
.while_while_cond_3739___redundant_placeholder12
.while_while_cond_3739___redundant_placeholder22
.while_while_cond_3739___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
¾
@
$__inference_zero1_layer_call_fn_4963

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_3123j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôÉ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
Õ8
Å
F__inference_sequential_1_layer_call_and_return_conditional_losses_3547

inputs$

conv1_3137:@

conv1_3139:@$

conv2_3154:@ 

conv2_3156: %

conv3_3177: 

conv3_3179:	%

conv4_3194:@

conv4_3196:@
lstm_1_3499:

 
lstm_1_3501:	 
lstm_1_3503:	( 
dense_1_3534:	
dense_1_3536:
identity¢conv1/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢conv3/StatefulPartitionedCall¢conv4/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCall½
zero1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_3123
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0
conv1_3137
conv1_3139*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_3136
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
conv2_3154
conv2_3156*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_3153Ü
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_3163
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0
conv3_3177
conv3_3179*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_3176
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0
conv4_3194
conv4_3196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_3193Û
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_3203Û
dropout_1/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3210ß
permute_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_3217Ü
reshape_1/PartitionedCallPartitionedCall"permute_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_1_3499lstm_1_3501lstm_1_3503*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3498ê
maxpooling1d_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513Ý
flatten_1/PartitionedCallPartitionedCall'maxpooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_3534dense_1_3536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_3533ã
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_3544t
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
£
@
$__inference_pool2_layer_call_fn_5080

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_2600
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­x
	
while_body_6125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
ð

_
C__inference_reshape_1_layer_call_and_return_conditional_losses_5162

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :5Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@
 
_user_specified_nameinputs
î
_
C__inference_permute_1_layer_call_and_return_conditional_losses_3217

inputs
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             q
	transpose	Transposeinputstranspose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@]
IdentityIdentitytranspose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
¸
ù
?__inference_conv4_layer_call_and_return_conditional_losses_3193

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0²
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHWY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ£?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?
 
_user_specified_nameinputs
½
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
È	
ó
A__inference_dense_1_layer_call_and_return_conditional_losses_6322

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
[
?__inference_pool1_layer_call_and_return_conditional_losses_3163

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ
 
_user_specified_nameinputs
º
ø
?__inference_conv2_layer_call_and_return_conditional_losses_5015

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@òÇ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ
 
_user_specified_nameinputs
«
D
(__inference_permute_1_layer_call_fn_5127

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_2613
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­x
	
while_body_5860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
»J
¥
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2748

inputs

states
states_11
split_readvariableop_resource:

 .
split_1_readvariableop_resource:	 *
readvariableop_resource:	( 
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:

 *
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskd
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_namestates
«
¹
while_cond_5859
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_5859___redundant_placeholder02
.while_while_cond_5859___redundant_placeholder12
.while_while_cond_5859___redundant_placeholder22
.while_while_cond_5859___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
è
ò
+__inference_sequential_1_layer_call_fn_4117
zero1_input!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: 
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:

 
	unknown_8:	 
	unknown_9:	( 

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallzero1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_4057o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
Ù
Ê
@__inference_lstm_1_layer_call_and_return_conditional_losses_3498

inputs;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3357*
condR
while_cond_3356*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5ÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
ä8
Ê
F__inference_sequential_1_layer_call_and_return_conditional_losses_4162
zero1_input$

conv1_4121:@

conv1_4123:@$

conv2_4126:@ 

conv2_4128: %

conv3_4132: 

conv3_4134:	%

conv4_4137:@

conv4_4139:@
lstm_1_4146:

 
lstm_1_4148:	 
lstm_1_4150:	( 
dense_1_4155:	
dense_1_4157:
identity¢conv1/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢conv3/StatefulPartitionedCall¢conv4/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCallÂ
zero1/PartitionedCallPartitionedCallzero1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_3123
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0
conv1_4121
conv1_4123*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_3136
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
conv2_4126
conv2_4128*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_3153Ü
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_3163
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0
conv3_4132
conv3_4134*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_3176
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0
conv4_4137
conv4_4139*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_3193Û
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_3203Û
dropout_1/PartitionedCallPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3210ß
permute_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_3217Ü
reshape_1/PartitionedCallPartitionedCall"permute_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_1_4146lstm_1_4148lstm_1_4150*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3498ê
maxpooling1d_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513Ý
flatten_1/PartitionedCallPartitionedCall'maxpooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_4155dense_1_4157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_3533ã
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_3544t
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
¸
D
(__inference_reshape_1_layer_call_fn_5149

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@
 
_user_specified_nameinputs
¾6
¤
 __inference__traced_restore_6655
file_prefix7
assignvariableop_conv1_kernel:@+
assignvariableop_1_conv1_bias:@9
assignvariableop_2_conv2_kernel:@ +
assignvariableop_3_conv2_bias: :
assignvariableop_4_conv3_kernel: ,
assignvariableop_5_conv3_bias:	:
assignvariableop_6_conv4_kernel:@+
assignvariableop_7_conv4_bias:@4
!assignvariableop_8_dense_1_kernel:	-
assignvariableop_9_dense_1_bias:?
+assignvariableop_10_lstm_1_lstm_cell_kernel:

 H
5assignvariableop_11_lstm_1_lstm_cell_recurrent_kernel:	( 8
)assignvariableop_12_lstm_1_lstm_cell_bias:	 
identity_14¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¬
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ò
valueÈBÅB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp+assignvariableop_10_lstm_1_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_11AssignVariableOp5assignvariableop_11_lstm_1_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp)assignvariableop_12_lstm_1_lstm_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
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
£
@
$__inference_pool1_layer_call_fn_5020

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_2588
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

[
?__inference_pool2_layer_call_and_return_conditional_losses_2600

inputs
identity¹
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
%
Ö
__inference__traced_save_6606
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
'savev2_dense_1_bias_read_readvariableop6
2savev2_lstm_1_lstm_cell_kernel_read_readvariableop@
<savev2_lstm_1_lstm_cell_recurrent_kernel_read_readvariableop4
0savev2_lstm_1_lstm_cell_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ©
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ò
valueÈBÅB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B õ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop2savev2_lstm_1_lstm_cell_kernel_read_readvariableop<savev2_lstm_1_lstm_cell_recurrent_kernel_read_readvariableop0savev2_lstm_1_lstm_cell_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*«
_input_shapes
: :@:@:@ : : ::@:@:	::

 :	( : : 2(
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
: :!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:%	!

_output_shapes
:	: 


_output_shapes
::&"
 
_output_shapes
:

 :%!

_output_shapes
:	( :!

_output_shapes	
: :

_output_shapes
: 
«
¹
while_cond_5594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_5594___redundant_placeholder02
.while_while_cond_5594___redundant_placeholder12
.while_while_cond_5594___redundant_placeholder22
.while_while_cond_5594___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
«
¹
while_cond_2761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_2761___redundant_placeholder02
.while_while_cond_2761___redundant_placeholder12
.while_while_cond_2761___redundant_placeholder22
.while_while_cond_2761___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
=
ñ
@__inference_lstm_1_layer_call_and_return_conditional_losses_3087

inputs"
lstm_cell_3005:

 
lstm_cell_3007:	 !
lstm_cell_3009:	( 
identity¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_maská
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3005lstm_cell_3007lstm_cell_3009*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2951n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3005lstm_cell_3007lstm_cell_3009*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_3018*
condR
while_cond_3017*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(r
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º
ø
?__inference_conv2_layer_call_and_return_conditional_losses_3153

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ@òÇ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ
 
_user_specified_nameinputs
¼
ú
?__inference_conv3_layer_call_and_return_conditional_losses_3176

inputs9
conv2d_readvariableop_resource: .
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: *
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHW*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ ¥A: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A
 
_user_specified_nameinputs
»J
¥
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2951

inputs

states
states_11
split_readvariableop_resource:

 .
split_1_readvariableop_resource:	 *
readvariableop_resource:	( 
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:

 *
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskd
MatMul_4MatMulstatesstrided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_namestates
­x
	
while_body_5595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 

[
?__inference_pool2_layer_call_and_return_conditional_losses_5090

inputs
identity¹
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
õ

lstm_1_while_body_4799*
&lstm_1_while_lstm_1_while_loop_counter0
,lstm_1_while_lstm_1_while_maximum_iterations
lstm_1_while_placeholder
lstm_1_while_placeholder_1
lstm_1_while_placeholder_2
lstm_1_while_placeholder_3)
%lstm_1_while_lstm_1_strided_slice_1_0e
alstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0J
6lstm_1_while_lstm_cell_split_readvariableop_resource_0:

 G
8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	 C
0lstm_1_while_lstm_cell_readvariableop_resource_0:	( 
lstm_1_while_identity
lstm_1_while_identity_1
lstm_1_while_identity_2
lstm_1_while_identity_3
lstm_1_while_identity_4
lstm_1_while_identity_5'
#lstm_1_while_lstm_1_strided_slice_1c
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorH
4lstm_1_while_lstm_cell_split_readvariableop_resource:

 E
6lstm_1_while_lstm_cell_split_1_readvariableop_resource:	 A
.lstm_1_while_lstm_cell_readvariableop_resource:	( ¢%lstm_1/while/lstm_cell/ReadVariableOp¢'lstm_1/while/lstm_cell/ReadVariableOp_1¢'lstm_1/while/lstm_cell/ReadVariableOp_2¢'lstm_1/while/lstm_cell/ReadVariableOp_3¢+lstm_1/while/lstm_cell/split/ReadVariableOp¢-lstm_1/while/lstm_cell/split_1/ReadVariableOp
>lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
0lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0lstm_1_while_placeholderGlstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0h
&lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¤
+lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOp6lstm_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0ç
lstm_1/while/lstm_cell/splitSplit/lstm_1/while/lstm_cell/split/split_dim:output:03lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¹
lstm_1/while/lstm_cell/MatMulMatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_1MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_2MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(»
lstm_1/while/lstm_cell/MatMul_3MatMul7lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:0%lstm_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(j
(lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : £
-lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOp8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ù
lstm_1/while/lstm_cell/split_1Split1lstm_1/while/lstm_cell/split_1/split_dim:output:05lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split­
lstm_1/while/lstm_cell/BiasAddBiasAdd'lstm_1/while/lstm_cell/MatMul:product:0'lstm_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_1BiasAdd)lstm_1/while/lstm_cell/MatMul_1:product:0'lstm_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_2BiasAdd)lstm_1/while/lstm_cell/MatMul_2:product:0'lstm_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
 lstm_1/while/lstm_cell/BiasAdd_3BiasAdd)lstm_1/while/lstm_cell/MatMul_3:product:0'lstm_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
%lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0{
*lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   }
,lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_1/while/lstm_cell/strided_sliceStridedSlice-lstm_1/while/lstm_cell/ReadVariableOp:value:03lstm_1/while/lstm_cell/strided_slice/stack:output:05lstm_1/while/lstm_cell/strided_slice/stack_1:output:05lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¦
lstm_1/while/lstm_cell/MatMul_4MatMullstm_1_while_placeholder_2-lstm_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(©
lstm_1/while/lstm_cell/addAddV2'lstm_1/while/lstm_cell/BiasAdd:output:0)lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
lstm_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_1/while/lstm_cell/MulMullstm_1/while/lstm_cell/add:z:0%lstm_1/while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ( 
lstm_1/while/lstm_cell/Add_1AddV2lstm_1/while/lstm_cell/Mul:z:0'lstm_1/while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(s
.lstm_1/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ä
,lstm_1/while/lstm_cell/clip_by_value/MinimumMinimum lstm_1/while/lstm_cell/Add_1:z:07lstm_1/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
&lstm_1/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
$lstm_1/while/lstm_cell/clip_by_valueMaximum0lstm_1/while/lstm_cell/clip_by_value/Minimum:z:0/lstm_1/while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   
.lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   
.lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_1StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_1:value:05lstm_1/while/lstm_cell/strided_slice_1/stack:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_5MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_2AddV2)lstm_1/while/lstm_cell/BiasAdd_1:output:0)lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
lstm_1/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ? 
lstm_1/while/lstm_cell/Mul_1Mul lstm_1/while/lstm_cell/add_2:z:0'lstm_1/while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¢
lstm_1/while/lstm_cell/Add_3AddV2 lstm_1/while/lstm_cell/Mul_1:z:0'lstm_1/while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
0lstm_1/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
.lstm_1/while/lstm_cell/clip_by_value_1/MinimumMinimum lstm_1/while/lstm_cell/Add_3:z:09lstm_1/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
&lstm_1/while/lstm_cell/clip_by_value_1Maximum2lstm_1/while/lstm_cell/clip_by_value_1/Minimum:z:01lstm_1/while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/lstm_cell/mul_2Mul*lstm_1/while/lstm_cell/clip_by_value_1:z:0lstm_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   
.lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   
.lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_2StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_2:value:05lstm_1/while/lstm_cell/strided_slice_2/stack:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_6MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_4AddV2)lstm_1/while/lstm_cell/BiasAdd_2:output:0)lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
lstm_1/while/lstm_cell/TanhTanh lstm_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ( 
lstm_1/while/lstm_cell/mul_3Mul(lstm_1/while/lstm_cell/clip_by_value:z:0lstm_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/lstm_cell/add_5AddV2 lstm_1/while/lstm_cell/mul_2:z:0 lstm_1/while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
'lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp0lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0}
,lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   
.lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
&lstm_1/while/lstm_cell/strided_slice_3StridedSlice/lstm_1/while/lstm_cell/ReadVariableOp_3:value:05lstm_1/while/lstm_cell/strided_slice_3/stack:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:07lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask¨
lstm_1/while/lstm_cell/MatMul_7MatMullstm_1_while_placeholder_2/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(­
lstm_1/while/lstm_cell/add_6AddV2)lstm_1/while/lstm_cell/BiasAdd_3:output:0)lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
lstm_1/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>c
lstm_1/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ? 
lstm_1/while/lstm_cell/Mul_4Mul lstm_1/while/lstm_cell/add_6:z:0'lstm_1/while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¢
lstm_1/while/lstm_cell/Add_7AddV2 lstm_1/while/lstm_cell/Mul_4:z:0'lstm_1/while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(u
0lstm_1/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
.lstm_1/while/lstm_cell/clip_by_value_2/MinimumMinimum lstm_1/while/lstm_cell/Add_7:z:09lstm_1/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(m
(lstm_1/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
&lstm_1/while/lstm_cell/clip_by_value_2Maximum2lstm_1/while/lstm_cell/clip_by_value_2/Minimum:z:01lstm_1/while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_1/while/lstm_cell/Tanh_1Tanh lstm_1/while/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¤
lstm_1/while/lstm_cell/mul_5Mul*lstm_1/while/lstm_cell/clip_by_value_2:z:0!lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Þ
1lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_1_while_placeholder_1lstm_1_while_placeholder lstm_1/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_1/while/addAddV2lstm_1_while_placeholderlstm_1/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_1/while/add_1AddV2&lstm_1_while_lstm_1_while_loop_counterlstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_1/while/IdentityIdentitylstm_1/while/add_1:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_1Identity,lstm_1_while_lstm_1_while_maximum_iterations^lstm_1/while/NoOp*
T0*
_output_shapes
: n
lstm_1/while/Identity_2Identitylstm_1/while/add:z:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_3IdentityAlstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_1/while/NoOp*
T0*
_output_shapes
: 
lstm_1/while/Identity_4Identity lstm_1/while/lstm_cell/mul_5:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_1/while/Identity_5Identity lstm_1/while/lstm_cell/add_5:z:0^lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(×
lstm_1/while/NoOpNoOp&^lstm_1/while/lstm_cell/ReadVariableOp(^lstm_1/while/lstm_cell/ReadVariableOp_1(^lstm_1/while/lstm_cell/ReadVariableOp_2(^lstm_1/while/lstm_cell/ReadVariableOp_3,^lstm_1/while/lstm_cell/split/ReadVariableOp.^lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_1_while_identitylstm_1/while/Identity:output:0";
lstm_1_while_identity_1 lstm_1/while/Identity_1:output:0";
lstm_1_while_identity_2 lstm_1/while/Identity_2:output:0";
lstm_1_while_identity_3 lstm_1/while/Identity_3:output:0";
lstm_1_while_identity_4 lstm_1/while/Identity_4:output:0";
lstm_1_while_identity_5 lstm_1/while/Identity_5:output:0"L
#lstm_1_while_lstm_1_strided_slice_1%lstm_1_while_lstm_1_strided_slice_1_0"b
.lstm_1_while_lstm_cell_readvariableop_resource0lstm_1_while_lstm_cell_readvariableop_resource_0"r
6lstm_1_while_lstm_cell_split_1_readvariableop_resource8lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"n
4lstm_1_while_lstm_cell_split_readvariableop_resource6lstm_1_while_lstm_cell_split_readvariableop_resource_0"Ä
_lstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensoralstm_1_while_tensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2N
%lstm_1/while/lstm_cell/ReadVariableOp%lstm_1/while/lstm_cell/ReadVariableOp2R
'lstm_1/while/lstm_cell/ReadVariableOp_1'lstm_1/while/lstm_cell/ReadVariableOp_12R
'lstm_1/while/lstm_cell/ReadVariableOp_2'lstm_1/while/lstm_cell/ReadVariableOp_22R
'lstm_1/while/lstm_cell/ReadVariableOp_3'lstm_1/while/lstm_cell/ReadVariableOp_32Z
+lstm_1/while/lstm_cell/split/ReadVariableOp+lstm_1/while/lstm_cell/split/ReadVariableOp2^
-lstm_1/while/lstm_cell/split_1/ReadVariableOp-lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
ÉJ
§
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6455

inputs
states_0
states_11
split_readvariableop_resource:

 .
split_1_readvariableop_resource:	 *
readvariableop_resource:	( 
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:

 *
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/1
Ð¤
¼
#sequential_1_lstm_1_while_body_2412D
@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counterJ
Fsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations)
%sequential_1_lstm_1_while_placeholder+
'sequential_1_lstm_1_while_placeholder_1+
'sequential_1_lstm_1_while_placeholder_2+
'sequential_1_lstm_1_while_placeholder_3C
?sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1_0
{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0W
Csequential_1_lstm_1_while_lstm_cell_split_readvariableop_resource_0:

 T
Esequential_1_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0:	 P
=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0:	( &
"sequential_1_lstm_1_while_identity(
$sequential_1_lstm_1_while_identity_1(
$sequential_1_lstm_1_while_identity_2(
$sequential_1_lstm_1_while_identity_3(
$sequential_1_lstm_1_while_identity_4(
$sequential_1_lstm_1_while_identity_5A
=sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1}
ysequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensorU
Asequential_1_lstm_1_while_lstm_cell_split_readvariableop_resource:

 R
Csequential_1_lstm_1_while_lstm_cell_split_1_readvariableop_resource:	 N
;sequential_1_lstm_1_while_lstm_cell_readvariableop_resource:	( ¢2sequential_1/lstm_1/while/lstm_cell/ReadVariableOp¢4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_1¢4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_2¢4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_3¢8sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOp¢:sequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOp
Ksequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
=sequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_1_while_placeholderTsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0u
3sequential_1/lstm_1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¾
8sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOpReadVariableOpCsequential_1_lstm_1_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0
)sequential_1/lstm_1/while/lstm_cell/splitSplit<sequential_1/lstm_1/while/lstm_cell/split/split_dim:output:0@sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_splità
*sequential_1/lstm_1/while/lstm_cell/MatMulMatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:02sequential_1/lstm_1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(â
,sequential_1/lstm_1/while/lstm_cell/MatMul_1MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:02sequential_1/lstm_1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(â
,sequential_1/lstm_1/while/lstm_cell/MatMul_2MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:02sequential_1/lstm_1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(â
,sequential_1/lstm_1/while/lstm_cell/MatMul_3MatMulDsequential_1/lstm_1/while/TensorArrayV2Read/TensorListGetItem:item:02sequential_1/lstm_1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
5sequential_1/lstm_1/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ½
:sequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOpReadVariableOpEsequential_1_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0
+sequential_1/lstm_1/while/lstm_cell/split_1Split>sequential_1/lstm_1/while/lstm_cell/split_1/split_dim:output:0Bsequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splitÔ
+sequential_1/lstm_1/while/lstm_cell/BiasAddBiasAdd4sequential_1/lstm_1/while/lstm_cell/MatMul:product:04sequential_1/lstm_1/while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ø
-sequential_1/lstm_1/while/lstm_cell/BiasAdd_1BiasAdd6sequential_1/lstm_1/while/lstm_cell/MatMul_1:product:04sequential_1/lstm_1/while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ø
-sequential_1/lstm_1/while/lstm_cell/BiasAdd_2BiasAdd6sequential_1/lstm_1/while/lstm_cell/MatMul_2:product:04sequential_1/lstm_1/while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ø
-sequential_1/lstm_1/while/lstm_cell/BiasAdd_3BiasAdd6sequential_1/lstm_1/while/lstm_cell/MatMul_3:product:04sequential_1/lstm_1/while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(±
2sequential_1/lstm_1/while/lstm_cell/ReadVariableOpReadVariableOp=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0
7sequential_1/lstm_1/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9sequential_1/lstm_1/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   
9sequential_1/lstm_1/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
1sequential_1/lstm_1/while/lstm_cell/strided_sliceStridedSlice:sequential_1/lstm_1/while/lstm_cell/ReadVariableOp:value:0@sequential_1/lstm_1/while/lstm_cell/strided_slice/stack:output:0Bsequential_1/lstm_1/while/lstm_cell/strided_slice/stack_1:output:0Bsequential_1/lstm_1/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskÍ
,sequential_1/lstm_1/while/lstm_cell/MatMul_4MatMul'sequential_1_lstm_1_while_placeholder_2:sequential_1/lstm_1/while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ð
'sequential_1/lstm_1/while/lstm_cell/addAddV24sequential_1/lstm_1/while/lstm_cell/BiasAdd:output:06sequential_1/lstm_1/while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)sequential_1/lstm_1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+sequential_1/lstm_1/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Á
'sequential_1/lstm_1/while/lstm_cell/MulMul+sequential_1/lstm_1/while/lstm_cell/add:z:02sequential_1/lstm_1/while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ç
)sequential_1/lstm_1/while/lstm_cell/Add_1AddV2+sequential_1/lstm_1/while/lstm_cell/Mul:z:04sequential_1/lstm_1/while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
;sequential_1/lstm_1/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ë
9sequential_1/lstm_1/while/lstm_cell/clip_by_value/MinimumMinimum-sequential_1/lstm_1/while/lstm_cell/Add_1:z:0Dsequential_1/lstm_1/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x
3sequential_1/lstm_1/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ë
1sequential_1/lstm_1/while/lstm_cell/clip_by_valueMaximum=sequential_1/lstm_1/while/lstm_cell/clip_by_value/Minimum:z:0<sequential_1/lstm_1/while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(³
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_1ReadVariableOp=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0
9sequential_1/lstm_1/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   
;sequential_1/lstm_1/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   
;sequential_1/lstm_1/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
3sequential_1/lstm_1/while/lstm_cell/strided_slice_1StridedSlice<sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_1:value:0Bsequential_1/lstm_1/while/lstm_cell/strided_slice_1/stack:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_1/stack_1:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskÏ
,sequential_1/lstm_1/while/lstm_cell/MatMul_5MatMul'sequential_1_lstm_1_while_placeholder_2<sequential_1/lstm_1/while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ô
)sequential_1/lstm_1/while/lstm_cell/add_2AddV26sequential_1/lstm_1/while/lstm_cell/BiasAdd_1:output:06sequential_1/lstm_1/while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(p
+sequential_1/lstm_1/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+sequential_1/lstm_1/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ç
)sequential_1/lstm_1/while/lstm_cell/Mul_1Mul-sequential_1/lstm_1/while/lstm_cell/add_2:z:04sequential_1/lstm_1/while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(É
)sequential_1/lstm_1/while/lstm_cell/Add_3AddV2-sequential_1/lstm_1/while/lstm_cell/Mul_1:z:04sequential_1/lstm_1/while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
=sequential_1/lstm_1/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ï
;sequential_1/lstm_1/while/lstm_cell/clip_by_value_1/MinimumMinimum-sequential_1/lstm_1/while/lstm_cell/Add_3:z:0Fsequential_1/lstm_1/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(z
5sequential_1/lstm_1/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ñ
3sequential_1/lstm_1/while/lstm_cell/clip_by_value_1Maximum?sequential_1/lstm_1/while/lstm_cell/clip_by_value_1/Minimum:z:0>sequential_1/lstm_1/while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ä
)sequential_1/lstm_1/while/lstm_cell/mul_2Mul7sequential_1/lstm_1/while/lstm_cell/clip_by_value_1:z:0'sequential_1_lstm_1_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(³
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_2ReadVariableOp=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0
9sequential_1/lstm_1/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   
;sequential_1/lstm_1/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   
;sequential_1/lstm_1/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
3sequential_1/lstm_1/while/lstm_cell/strided_slice_2StridedSlice<sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_2:value:0Bsequential_1/lstm_1/while/lstm_cell/strided_slice_2/stack:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_2/stack_1:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskÏ
,sequential_1/lstm_1/while/lstm_cell/MatMul_6MatMul'sequential_1_lstm_1_while_placeholder_2<sequential_1/lstm_1/while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ô
)sequential_1/lstm_1/while/lstm_cell/add_4AddV26sequential_1/lstm_1/while/lstm_cell/BiasAdd_2:output:06sequential_1/lstm_1/while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
(sequential_1/lstm_1/while/lstm_cell/TanhTanh-sequential_1/lstm_1/while/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ç
)sequential_1/lstm_1/while/lstm_cell/mul_3Mul5sequential_1/lstm_1/while/lstm_cell/clip_by_value:z:0,sequential_1/lstm_1/while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
)sequential_1/lstm_1/while/lstm_cell/add_5AddV2-sequential_1/lstm_1/while/lstm_cell/mul_2:z:0-sequential_1/lstm_1/while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(³
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_3ReadVariableOp=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0
9sequential_1/lstm_1/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   
;sequential_1/lstm_1/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
;sequential_1/lstm_1/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ©
3sequential_1/lstm_1/while/lstm_cell/strided_slice_3StridedSlice<sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_3:value:0Bsequential_1/lstm_1/while/lstm_cell/strided_slice_3/stack:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_3/stack_1:output:0Dsequential_1/lstm_1/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskÏ
,sequential_1/lstm_1/while/lstm_cell/MatMul_7MatMul'sequential_1_lstm_1_while_placeholder_2<sequential_1/lstm_1/while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ô
)sequential_1/lstm_1/while/lstm_cell/add_6AddV26sequential_1/lstm_1/while/lstm_cell/BiasAdd_3:output:06sequential_1/lstm_1/while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(p
+sequential_1/lstm_1/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>p
+sequential_1/lstm_1/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?Ç
)sequential_1/lstm_1/while/lstm_cell/Mul_4Mul-sequential_1/lstm_1/while/lstm_cell/add_6:z:04sequential_1/lstm_1/while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(É
)sequential_1/lstm_1/while/lstm_cell/Add_7AddV2-sequential_1/lstm_1/while/lstm_cell/Mul_4:z:04sequential_1/lstm_1/while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
=sequential_1/lstm_1/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ï
;sequential_1/lstm_1/while/lstm_cell/clip_by_value_2/MinimumMinimum-sequential_1/lstm_1/while/lstm_cell/Add_7:z:0Fsequential_1/lstm_1/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(z
5sequential_1/lstm_1/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ñ
3sequential_1/lstm_1/while/lstm_cell/clip_by_value_2Maximum?sequential_1/lstm_1/while/lstm_cell/clip_by_value_2/Minimum:z:0>sequential_1/lstm_1/while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
*sequential_1/lstm_1/while/lstm_cell/Tanh_1Tanh-sequential_1/lstm_1/while/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ë
)sequential_1/lstm_1/while/lstm_cell/mul_5Mul7sequential_1/lstm_1/while/lstm_cell/clip_by_value_2:z:0.sequential_1/lstm_1/while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
>sequential_1/lstm_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_1_while_placeholder_1%sequential_1_lstm_1_while_placeholder-sequential_1/lstm_1/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_1/lstm_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_1/lstm_1/while/addAddV2%sequential_1_lstm_1_while_placeholder(sequential_1/lstm_1/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_1/lstm_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_1/lstm_1/while/add_1AddV2@sequential_1_lstm_1_while_sequential_1_lstm_1_while_loop_counter*sequential_1/lstm_1/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_1/lstm_1/while/IdentityIdentity#sequential_1/lstm_1/while/add_1:z:0^sequential_1/lstm_1/while/NoOp*
T0*
_output_shapes
: º
$sequential_1/lstm_1/while/Identity_1IdentityFsequential_1_lstm_1_while_sequential_1_lstm_1_while_maximum_iterations^sequential_1/lstm_1/while/NoOp*
T0*
_output_shapes
: 
$sequential_1/lstm_1/while/Identity_2Identity!sequential_1/lstm_1/while/add:z:0^sequential_1/lstm_1/while/NoOp*
T0*
_output_shapes
: Â
$sequential_1/lstm_1/while/Identity_3IdentityNsequential_1/lstm_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_1/while/NoOp*
T0*
_output_shapes
: ²
$sequential_1/lstm_1/while/Identity_4Identity-sequential_1/lstm_1/while/lstm_cell/mul_5:z:0^sequential_1/lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(²
$sequential_1/lstm_1/while/Identity_5Identity-sequential_1/lstm_1/while/lstm_cell/add_5:z:0^sequential_1/lstm_1/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(²
sequential_1/lstm_1/while/NoOpNoOp3^sequential_1/lstm_1/while/lstm_cell/ReadVariableOp5^sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_15^sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_25^sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_39^sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOp;^sequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_1_lstm_1_while_identity+sequential_1/lstm_1/while/Identity:output:0"U
$sequential_1_lstm_1_while_identity_1-sequential_1/lstm_1/while/Identity_1:output:0"U
$sequential_1_lstm_1_while_identity_2-sequential_1/lstm_1/while/Identity_2:output:0"U
$sequential_1_lstm_1_while_identity_3-sequential_1/lstm_1/while/Identity_3:output:0"U
$sequential_1_lstm_1_while_identity_4-sequential_1/lstm_1/while/Identity_4:output:0"U
$sequential_1_lstm_1_while_identity_5-sequential_1/lstm_1/while/Identity_5:output:0"|
;sequential_1_lstm_1_while_lstm_cell_readvariableop_resource=sequential_1_lstm_1_while_lstm_cell_readvariableop_resource_0"
Csequential_1_lstm_1_while_lstm_cell_split_1_readvariableop_resourceEsequential_1_lstm_1_while_lstm_cell_split_1_readvariableop_resource_0"
Asequential_1_lstm_1_while_lstm_cell_split_readvariableop_resourceCsequential_1_lstm_1_while_lstm_cell_split_readvariableop_resource_0"
=sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1?sequential_1_lstm_1_while_sequential_1_lstm_1_strided_slice_1_0"ø
ysequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_1_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2h
2sequential_1/lstm_1/while/lstm_cell/ReadVariableOp2sequential_1/lstm_1/while/lstm_cell/ReadVariableOp2l
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_14sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_12l
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_24sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_22l
4sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_34sequential_1/lstm_1/while/lstm_cell/ReadVariableOp_32t
8sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOp8sequential_1/lstm_1/while/lstm_cell/split/ReadVariableOp2x
:sequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOp:sequential_1/lstm_1/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
Ù
í
+__inference_sequential_1_layer_call_fn_4302

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: 
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:

 
	unknown_8:	 
	unknown_9:	( 

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_4057o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
«
¹
while_cond_6124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_6124___redundant_placeholder02
.while_while_cond_6124___redundant_placeholder12
.while_while_cond_6124___redundant_placeholder22
.while_while_cond_6124___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:
Û
_
C__inference_permute_1_layer_call_and_return_conditional_losses_2613

inputs
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¹
while_cond_3356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_3356___redundant_placeholder02
.while_while_cond_3356___redundant_placeholder12
.while_while_cond_3356___redundant_placeholder22
.while_while_cond_3356___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:

[
?__inference_pool1_layer_call_and_return_conditional_losses_5030

inputs
identity¹
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
data_formatNCHW*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ì
@__inference_lstm_1_layer_call_and_return_conditional_losses_5736
inputs_0;
'lstm_cell_split_readvariableop_resource:

 8
)lstm_cell_split_1_readvariableop_resource:	 4
!lstm_cell_readvariableop_resource:	( 
identity¢lstm_cell/ReadVariableOp¢lstm_cell/ReadVariableOp_1¢lstm_cell/ReadVariableOp_2¢lstm_cell/ReadVariableOp_3¢lstm_cell/split/ReadVariableOp¢ lstm_cell/split_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :èY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(O
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :(c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :(w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
shrink_axis_mask[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:

 *
dtype0À
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype0²
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?s
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/Add_1AddV2lstm_cell/Mul:z:0lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_3AddV2lstm_cell/Mul_1:z:0lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(y
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource*
_output_shapes
:	( *
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>V
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?y
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ({
lstm_cell/Add_7AddV2lstm_cell/Mul_4:z:0lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    £
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_5595*
condR
while_cond_5594*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ([
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
©
µ
%__inference_lstm_1_layer_call_fn_5184
inputs_0
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3087|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
­x
	
while_body_3740
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:

 @
1while_lstm_cell_split_1_readvariableop_resource_0:	 <
)while_lstm_cell_readvariableop_resource_0:	( 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:

 >
/while_lstm_cell_split_1_readvariableop_resource:	 :
'while_lstm_cell_readvariableop_resource:	( ¢while/lstm_cell/ReadVariableOp¢ while/lstm_cell/ReadVariableOp_1¢ while/lstm_cell/ReadVariableOp_2¢ while/lstm_cell/ReadVariableOp_3¢$while/lstm_cell/split/ReadVariableOp¢&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
element_dtype0a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:

 *
dtype0Ò
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_split¤
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype0Ä
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      »
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0while/lstm_cell/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_1AddV2while/lstm_cell/Mul:z:0 while/lstm_cell/Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_3AddV2while/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0*
_output_shapes
:	( *
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Å
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>\
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/Add_7AddV2while/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(n
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(f
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    µ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(k
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Â
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(v
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¦

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: 
º
ø
?__inference_conv1_layer_call_and_return_conditional_losses_3136

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0³
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*
data_formatNCHWZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôÉ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
¸
ù
?__inference_conv4_layer_call_and_return_conditional_losses_5075

inputs9
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0²
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHW*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*
data_formatNCHWY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ£?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?
 
_user_specified_nameinputs
ë

$__inference_conv1_layer_call_fn_4984

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_3136y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôÉ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
¸
I
-__inference_maxpooling1d_1_layer_call_fn_6276

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(
 
_user_specified_nameinputs
Ý
[
?__inference_zero1_layer_call_and_return_conditional_losses_4969

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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
[
?__inference_pool1_layer_call_and_return_conditional_losses_5035

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A*
data_formatNCHW*
ksize
*
paddingVALID*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ
 
_user_specified_nameinputs
ÿ
³
%__inference_lstm_1_layer_call_fn_5195

inputs
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3498s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ5
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5

 
_user_specified_nameinputs
ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3210

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
Ï
b
F__inference_activation_1_layer_call_and_return_conditional_losses_3544

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
í
+__inference_sequential_1_layer_call_fn_4271

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@ 
	unknown_2: $
	unknown_3: 
	unknown_4:	$
	unknown_5:@
	unknown_6:@
	unknown_7:

 
	unknown_8:	 
	unknown_9:	( 

unknown_10:	

unknown_11:
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_3547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
:
î
F__inference_sequential_1_layer_call_and_return_conditional_losses_4207
zero1_input$

conv1_4166:@

conv1_4168:@$

conv2_4171:@ 

conv2_4173: %

conv3_4177: 

conv3_4179:	%

conv4_4182:@

conv4_4184:@
lstm_1_4191:

 
lstm_1_4193:	 
lstm_1_4195:	( 
dense_1_4200:	
dense_1_4202:
identity¢conv1/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢conv3/StatefulPartitionedCall¢conv4/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢lstm_1/StatefulPartitionedCallÂ
zero1/PartitionedCallPartitionedCallzero1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_zero1_layer_call_and_return_conditional_losses_3123
conv1/StatefulPartitionedCallStatefulPartitionedCallzero1/PartitionedCall:output:0
conv1_4166
conv1_4168*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@òÇ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv1_layer_call_and_return_conditional_losses_3136
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0
conv2_4171
conv2_4173*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ðÅ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv2_layer_call_and_return_conditional_losses_3153Ü
pool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¥A* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_3163
conv3/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0
conv3_4177
conv3_4179*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv3_layer_call_and_return_conditional_losses_3176
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0
conv4_4182
conv4_4184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_conv4_layer_call_and_return_conditional_losses_3193Û
pool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_3203ë
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3921ç
permute_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_permute_1_layer_call_and_return_conditional_losses_3217Ü
reshape_1/PartitionedCallPartitionedCall"permute_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232
lstm_1/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0lstm_1_4191lstm_1_4193lstm_1_4195*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_3881ê
maxpooling1d_1/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_3513Ý
flatten_1/PartitionedCallPartitionedCall'maxpooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_3521
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_4200dense_1_4202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_3533ã
activation_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_3544t
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall^conv4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôÉ: : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:^ Z
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
%
_user_specified_namezero1_input
ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5110

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@5:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5
 
_user_specified_nameinputs
ð

_
C__inference_reshape_1_layer_call_and_return_conditional_losses_3232

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :5Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
]
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ5@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5@
 
_user_specified_nameinputs
ÉJ
§
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6544

inputs
states_0
states_11
split_readvariableop_resource:

 .
split_1_readvariableop_resource:	 *
readvariableop_resource:	( 
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:

 *
dtype0¢
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*@
_output_shapes.
,:	
(:	
(:	
(:	
(*
	num_splitZ
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:(:(:(:(*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    (   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ë
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskf
MatMul_4MatMulstates_0strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?U
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
Add_1AddV2Mul:z:0Const_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    (   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_3AddV2	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
mul_2Mulclip_by_value_1:z:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    P   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(I
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ([
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(i
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	( *
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    x   h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      õ
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:((*

begin_mask*
end_maskh
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(h
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?[
Mul_4Mul	add_6:z:0Const_4:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(]
Add_7AddV2	Mul_4:z:0Const_5:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(K
Tanh_1Tanh	add_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(_
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(X
IdentityIdentity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_1Identity	mul_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Z

Identity_2Identity	add_5:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(À
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/1
´
[
?__inference_pool2_layer_call_and_return_conditional_losses_5095

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5*
data_formatNCHW*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=
 
_user_specified_nameinputs
ç
ò
(__inference_lstm_cell_layer_call_fn_6366

inputs
states_0
states_1
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity

identity_1

identity_2¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lstm_cell_layer_call_and_return_conditional_losses_2951o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ
:ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
"
_user_specified_name
states/1
¸
@
$__inference_pool2_layer_call_fn_5085

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_3203h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@5"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡=
 
_user_specified_nameinputs
ù
[
?__inference_zero1_layer_call_and_return_conditional_losses_3123

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
:ÿÿÿÿÿÿÿÿÿôÉ^
IdentityIdentityPad:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôÉ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôÉ
 
_user_specified_nameinputs
Û
_
C__inference_permute_1_layer_call_and_return_conditional_losses_5138

inputs
identityg
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             
	transpose	Transposeinputstranspose/perm:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx
IdentityIdentitytranspose:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
µ
%__inference_lstm_1_layer_call_fn_5173
inputs_0
unknown:

 
	unknown_0:	 
	unknown_1:	( 
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_2831|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

"
_user_specified_name
inputs/0
Ï
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6332

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
d
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6284

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¹
while_cond_5329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_12
.while_while_cond_5329___redundant_placeholder02
.while_while_cond_5329___redundant_placeholder12
.while_while_cond_5329___redundant_placeholder22
.while_while_cond_5329___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ(:ÿÿÿÿÿÿÿÿÿ(: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
:"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Á
serving_default­
M
zero1_input>
serving_default_zero1_input:0ÿÿÿÿÿÿÿÿÿôÉ@
activation_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
¾
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
¾__call__
+¿&call_and_return_all_conditional_losses
À_default_save_signature"
_tf_keras_sequential
§
	variables
trainable_variables
regularization_losses
	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
½

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"
_tf_keras_layer
½

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"
_tf_keras_layer
§
%	variables
&trainable_variables
'regularization_losses
(	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
½

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"
_tf_keras_layer
½

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_layer
§
5	variables
6trainable_variables
7regularization_losses
8	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
¾
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=_random_generator
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
§
>	variables
?trainable_variables
@regularization_losses
A	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_tf_keras_layer
§
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ü
Fcell
G
state_spec
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L_random_generator
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
§
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
§
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
§
[	variables
\trainable_variables
]regularization_losses
^	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
~
0
1
2
 3
)4
*5
/6
07
_8
`9
a10
U11
V12"
trackable_list_wrapper
~
0
1
2
 3
)4
*5
/6
07
_8
`9
a10
U11
V12"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
¾__call__
À_default_save_signature
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
-
ßserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
&:$@2conv1/kernel
:@2
conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
	variables
trainable_variables
regularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
&:$@ 2conv2/kernel
: 2
conv2/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
!	variables
"trainable_variables
#regularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
%	variables
&trainable_variables
'regularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
':% 2conv3/kernel
:2
conv3/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
+	variables
,trainable_variables
-regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
':%@2conv4/kernel
:@2
conv4/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
1	variables
2trainable_variables
3regularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
5	variables
6trainable_variables
7regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object


state_size

_kernel
`recurrent_kernel
abias
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
 "
trackable_list_wrapper
Â
states
 non_trainable_variables
¡layers
¢metrics
 £layer_regularization_losses
¤layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_1/kernel
:2dense_1/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
[	variables
\trainable_variables
]regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
+:)

 2lstm_1/lstm_cell/kernel
4:2	( 2!lstm_1/lstm_cell/recurrent_kernel
$:" 2lstm_1/lstm_cell/bias
 "
trackable_list_wrapper

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
14"
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
5
_0
`1
a2"
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
F0"
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
ú2÷
+__inference_sequential_1_layer_call_fn_3576
+__inference_sequential_1_layer_call_fn_4271
+__inference_sequential_1_layer_call_fn_4302
+__inference_sequential_1_layer_call_fn_4117À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_1_layer_call_and_return_conditional_losses_4624
F__inference_sequential_1_layer_call_and_return_conditional_losses_4953
F__inference_sequential_1_layer_call_and_return_conditional_losses_4162
F__inference_sequential_1_layer_call_and_return_conditional_losses_4207À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÎBË
__inference__wrapped_model_2566zero1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
$__inference_zero1_layer_call_fn_4958
$__inference_zero1_layer_call_fn_4963¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
?__inference_zero1_layer_call_and_return_conditional_losses_4969
?__inference_zero1_layer_call_and_return_conditional_losses_4975¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_conv1_layer_call_fn_4984¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_conv1_layer_call_and_return_conditional_losses_4995¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_conv2_layer_call_fn_5004¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_conv2_layer_call_and_return_conditional_losses_5015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
$__inference_pool1_layer_call_fn_5020
$__inference_pool1_layer_call_fn_5025¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
?__inference_pool1_layer_call_and_return_conditional_losses_5030
?__inference_pool1_layer_call_and_return_conditional_losses_5035¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_conv3_layer_call_fn_5044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_conv3_layer_call_and_return_conditional_losses_5055¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_conv4_layer_call_fn_5064¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_conv4_layer_call_and_return_conditional_losses_5075¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
$__inference_pool2_layer_call_fn_5080
$__inference_pool2_layer_call_fn_5085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª2§
?__inference_pool2_layer_call_and_return_conditional_losses_5090
?__inference_pool2_layer_call_and_return_conditional_losses_5095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_dropout_1_layer_call_fn_5100
(__inference_dropout_1_layer_call_fn_5105´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_1_layer_call_and_return_conditional_losses_5110
C__inference_dropout_1_layer_call_and_return_conditional_losses_5122´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ü2ù
(__inference_permute_1_layer_call_fn_5127
(__inference_permute_1_layer_call_fn_5132¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
C__inference_permute_1_layer_call_and_return_conditional_losses_5138
C__inference_permute_1_layer_call_and_return_conditional_losses_5144¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_reshape_1_layer_call_fn_5149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_reshape_1_layer_call_and_return_conditional_losses_5162¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
%__inference_lstm_1_layer_call_fn_5173
%__inference_lstm_1_layer_call_fn_5184
%__inference_lstm_1_layer_call_fn_5195
%__inference_lstm_1_layer_call_fn_5206Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ã2à
@__inference_lstm_1_layer_call_and_return_conditional_losses_5471
@__inference_lstm_1_layer_call_and_return_conditional_losses_5736
@__inference_lstm_1_layer_call_and_return_conditional_losses_6001
@__inference_lstm_1_layer_call_and_return_conditional_losses_6266Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_maxpooling1d_1_layer_call_fn_6271
-__inference_maxpooling1d_1_layer_call_fn_6276¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¼2¹
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6284
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6292¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_1_layer_call_fn_6297¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_1_layer_call_and_return_conditional_losses_6303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_1_layer_call_fn_6312¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_6322¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_activation_1_layer_call_fn_6327¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_activation_1_layer_call_and_return_conditional_losses_6332¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
"__inference_signature_wrapper_4240zero1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_lstm_cell_layer_call_fn_6349
(__inference_lstm_cell_layer_call_fn_6366¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6455
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6544¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 °
__inference__wrapped_model_2566 )*/0_a`UV>¢;
4¢1
/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ
ª ";ª8
6
activation_1&#
activation_1ÿÿÿÿÿÿÿÿÿ¢
F__inference_activation_1_layer_call_and_return_conditional_losses_6332X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
+__inference_activation_1_layer_call_fn_6327K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
?__inference_conv1_layer_call_and_return_conditional_losses_4995p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@òÇ
 
$__inference_conv1_layer_call_fn_4984c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
ª ""ÿÿÿÿÿÿÿÿÿ@òÇ³
?__inference_conv2_layer_call_and_return_conditional_losses_5015p 9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@òÇ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ ðÅ
 
$__inference_conv2_layer_call_fn_5004c 9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@òÇ
ª ""ÿÿÿÿÿÿÿÿÿ ðÅ²
?__inference_conv3_layer_call_and_return_conditional_losses_5055o)*8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ ¥A
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ£?
 
$__inference_conv3_layer_call_fn_5044b)*8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ ¥A
ª ""ÿÿÿÿÿÿÿÿÿ£?²
?__inference_conv4_layer_call_and_return_conditional_losses_5075o/09¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ£?
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@¡=
 
$__inference_conv4_layer_call_fn_5064b/09¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ£?
ª "!ÿÿÿÿÿÿÿÿÿ@¡=¢
A__inference_dense_1_layer_call_and_return_conditional_losses_6322]UV0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
&__inference_dense_1_layer_call_fn_6312PUV0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
C__inference_dropout_1_layer_call_and_return_conditional_losses_5110l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@5
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@5
 ³
C__inference_dropout_1_layer_call_and_return_conditional_losses_5122l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@5
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@5
 
(__inference_dropout_1_layer_call_fn_5100_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@5
p 
ª " ÿÿÿÿÿÿÿÿÿ@5
(__inference_dropout_1_layer_call_fn_5105_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@5
p
ª " ÿÿÿÿÿÿÿÿÿ@5¤
C__inference_flatten_1_layer_call_and_return_conditional_losses_6303]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ(
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_flatten_1_layer_call_fn_6297P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿÐ
@__inference_lstm_1_layer_call_and_return_conditional_losses_5471_a`P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 Ð
@__inference_lstm_1_layer_call_and_return_conditional_losses_5736_a`P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
 ¶
@__inference_lstm_1_layer_call_and_return_conditional_losses_6001r_a`@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ5


 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ5(
 ¶
@__inference_lstm_1_layer_call_and_return_conditional_losses_6266r_a`@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ5


 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ5(
 §
%__inference_lstm_1_layer_call_fn_5173~_a`P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(§
%__inference_lstm_1_layer_call_fn_5184~_a`P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ


 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ(
%__inference_lstm_1_layer_call_fn_5195e_a`@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ5


 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ5(
%__inference_lstm_1_layer_call_fn_5206e_a`@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ5


 
p

 
ª "ÿÿÿÿÿÿÿÿÿ5(Æ
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6455þ_a`¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ

K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ(
"
states/1ÿÿÿÿÿÿÿÿÿ(
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ(
EB

0/1/0ÿÿÿÿÿÿÿÿÿ(

0/1/1ÿÿÿÿÿÿÿÿÿ(
 Æ
C__inference_lstm_cell_layer_call_and_return_conditional_losses_6544þ_a`¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ

K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ(
"
states/1ÿÿÿÿÿÿÿÿÿ(
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ(
EB

0/1/0ÿÿÿÿÿÿÿÿÿ(

0/1/1ÿÿÿÿÿÿÿÿÿ(
 
(__inference_lstm_cell_layer_call_fn_6349î_a`¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ

K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ(
"
states/1ÿÿÿÿÿÿÿÿÿ(
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ(
A>

1/0ÿÿÿÿÿÿÿÿÿ(

1/1ÿÿÿÿÿÿÿÿÿ(
(__inference_lstm_cell_layer_call_fn_6366î_a`¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ

K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ(
"
states/1ÿÿÿÿÿÿÿÿÿ(
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ(
A>

1/0ÿÿÿÿÿÿÿÿÿ(

1/1ÿÿÿÿÿÿÿÿÿ(Ñ
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6284E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
H__inference_maxpooling1d_1_layer_call_and_return_conditional_losses_6292`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ5(
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ(
 ¨
-__inference_maxpooling1d_1_layer_call_fn_6271wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-__inference_maxpooling1d_1_layer_call_fn_6276S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ5(
ª "ÿÿÿÿÿÿÿÿÿ(æ
C__inference_permute_1_layer_call_and_return_conditional_losses_5138R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
C__inference_permute_1_layer_call_and_return_conditional_losses_5144h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@5
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ5@
 ¾
(__inference_permute_1_layer_call_fn_5127R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
(__inference_permute_1_layer_call_fn_5132[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@5
ª " ÿÿÿÿÿÿÿÿÿ5@â
?__inference_pool1_layer_call_and_return_conditional_losses_5030R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
?__inference_pool1_layer_call_and_return_conditional_losses_5035k9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ ðÅ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ ¥A
 º
$__inference_pool1_layer_call_fn_5020R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$__inference_pool1_layer_call_fn_5025^9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ ðÅ
ª "!ÿÿÿÿÿÿÿÿÿ ¥Aâ
?__inference_pool2_layer_call_and_return_conditional_losses_5090R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
?__inference_pool2_layer_call_and_return_conditional_losses_5095i8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@¡=
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@5
 º
$__inference_pool2_layer_call_fn_5080R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$__inference_pool2_layer_call_fn_5085\8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@¡=
ª " ÿÿÿÿÿÿÿÿÿ@5¬
C__inference_reshape_1_layer_call_and_return_conditional_losses_5162e7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ5@
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ5

 
(__inference_reshape_1_layer_call_fn_5149X7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ5@
ª "ÿÿÿÿÿÿÿÿÿ5
È
F__inference_sequential_1_layer_call_and_return_conditional_losses_4162~ )*/0_a`UVF¢C
<¢9
/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
F__inference_sequential_1_layer_call_and_return_conditional_losses_4207~ )*/0_a`UVF¢C
<¢9
/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
F__inference_sequential_1_layer_call_and_return_conditional_losses_4624y )*/0_a`UVA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
F__inference_sequential_1_layer_call_and_return_conditional_losses_4953y )*/0_a`UVA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
+__inference_sequential_1_layer_call_fn_3576q )*/0_a`UVF¢C
<¢9
/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
+__inference_sequential_1_layer_call_fn_4117q )*/0_a`UVF¢C
<¢9
/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_1_layer_call_fn_4271l )*/0_a`UVA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_1_layer_call_fn_4302l )*/0_a`UVA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
"__inference_signature_wrapper_4240 )*/0_a`UVM¢J
¢ 
Cª@
>
zero1_input/,
zero1_inputÿÿÿÿÿÿÿÿÿôÉ";ª8
6
activation_1&#
activation_1ÿÿÿÿÿÿÿÿÿâ
?__inference_zero1_layer_call_and_return_conditional_losses_4969R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
?__inference_zero1_layer_call_and_return_conditional_losses_4975l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôÉ
 º
$__inference_zero1_layer_call_fn_4958R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$__inference_zero1_layer_call_fn_4963_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôÉ
ª ""ÿÿÿÿÿÿÿÿÿôÉ