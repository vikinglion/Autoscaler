Ь
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
z
dense_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_namedense_95/kernel
s
#dense_95/kernel/Read/ReadVariableOpReadVariableOpdense_95/kernel*
_output_shapes

:2*
dtype0
r
dense_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_95/bias
k
!dense_95/bias/Read/ReadVariableOpReadVariableOpdense_95/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
gru_23/gru_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namegru_23/gru_cell_23/kernel
?
-gru_23/gru_cell_23/kernel/Read/ReadVariableOpReadVariableOpgru_23/gru_cell_23/kernel*
_output_shapes
:	?*
dtype0
?
#gru_23/gru_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*4
shared_name%#gru_23/gru_cell_23/recurrent_kernel
?
7gru_23/gru_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp#gru_23/gru_cell_23/recurrent_kernel*
_output_shapes
:	2?*
dtype0
?
gru_23/gru_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_namegru_23/gru_cell_23/bias
?
+gru_23/gru_cell_23/bias/Read/ReadVariableOpReadVariableOpgru_23/gru_cell_23/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_95/kernel/m
?
*Adam/dense_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/m*
_output_shapes

:2*
dtype0
?
Adam/dense_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/m
y
(Adam/dense_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/m*
_output_shapes
:*
dtype0
?
 Adam/gru_23/gru_cell_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_23/gru_cell_23/kernel/m
?
4Adam/gru_23/gru_cell_23/kernel/m/Read/ReadVariableOpReadVariableOp Adam/gru_23/gru_cell_23/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/gru_23/gru_cell_23/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/gru_23/gru_cell_23/recurrent_kernel/m
?
>Adam/gru_23/gru_cell_23/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/gru_23/gru_cell_23/recurrent_kernel/m*
_output_shapes
:	2?*
dtype0
?
Adam/gru_23/gru_cell_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_23/gru_cell_23/bias/m
?
2Adam/gru_23/gru_cell_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_23/gru_cell_23/bias/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/dense_95/kernel/v
?
*Adam/dense_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/kernel/v*
_output_shapes

:2*
dtype0
?
Adam/dense_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_95/bias/v
y
(Adam/dense_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_95/bias/v*
_output_shapes
:*
dtype0
?
 Adam/gru_23/gru_cell_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/gru_23/gru_cell_23/kernel/v
?
4Adam/gru_23/gru_cell_23/kernel/v/Read/ReadVariableOpReadVariableOp Adam/gru_23/gru_cell_23/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/gru_23/gru_cell_23/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2?*;
shared_name,*Adam/gru_23/gru_cell_23/recurrent_kernel/v
?
>Adam/gru_23/gru_cell_23/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/gru_23/gru_cell_23/recurrent_kernel/v*
_output_shapes
:	2?*
dtype0
?
Adam/gru_23/gru_cell_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/gru_23/gru_cell_23/bias/v
?
2Adam/gru_23/gru_cell_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_23/gru_cell_23/bias/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
l
	cell


state_spec
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD
#
0
1
2
3
4
 
#
0
1
2
3
4
?
layer_metrics
non_trainable_variables
trainable_variables
layer_regularization_losses
 metrics
regularization_losses

!layers
	variables
 
~

kernel
recurrent_kernel
bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
 
 

0
1
2

0
1
2
?
&layer_metrics
'non_trainable_variables

(states
)metrics
*layer_regularization_losses
regularization_losses
trainable_variables

+layers
	variables
[Y
VARIABLE_VALUEdense_95/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_95/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
,layer_metrics
regularization_losses
-non_trainable_variables
.layer_regularization_losses
/metrics
trainable_variables

0layers
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEgru_23/gru_cell_23/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#gru_23/gru_cell_23/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEgru_23/gru_cell_23/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

10

0
1
 

0
1
2

0
1
2
?
2layer_metrics
"regularization_losses
3non_trainable_variables
4layer_regularization_losses
5metrics
#trainable_variables

6layers
$	variables
 
 
 
 
 

	0
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
~|
VARIABLE_VALUEAdam/dense_95/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_95/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_23/gru_cell_23/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_23/gru_cell_23/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_23/gru_cell_23/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_95/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_95/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/gru_23/gru_cell_23/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/gru_23/gru_cell_23/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/gru_23/gru_cell_23/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_gru_23_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_23_inputgru_23/gru_cell_23/biasgru_23/gru_cell_23/kernel#gru_23/gru_cell_23/recurrent_kerneldense_95/kerneldense_95/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1510004
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_95/kernel/Read/ReadVariableOp!dense_95/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-gru_23/gru_cell_23/kernel/Read/ReadVariableOp7gru_23/gru_cell_23/recurrent_kernel/Read/ReadVariableOp+gru_23/gru_cell_23/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_95/kernel/m/Read/ReadVariableOp(Adam/dense_95/bias/m/Read/ReadVariableOp4Adam/gru_23/gru_cell_23/kernel/m/Read/ReadVariableOp>Adam/gru_23/gru_cell_23/recurrent_kernel/m/Read/ReadVariableOp2Adam/gru_23/gru_cell_23/bias/m/Read/ReadVariableOp*Adam/dense_95/kernel/v/Read/ReadVariableOp(Adam/dense_95/bias/v/Read/ReadVariableOp4Adam/gru_23/gru_cell_23/kernel/v/Read/ReadVariableOp>Adam/gru_23/gru_cell_23/recurrent_kernel/v/Read/ReadVariableOp2Adam/gru_23/gru_cell_23/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1511260
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_95/kerneldense_95/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru_23/gru_cell_23/kernel#gru_23/gru_cell_23/recurrent_kernelgru_23/gru_cell_23/biastotalcountAdam/dense_95/kernel/mAdam/dense_95/bias/m Adam/gru_23/gru_cell_23/kernel/m*Adam/gru_23/gru_cell_23/recurrent_kernel/mAdam/gru_23/gru_cell_23/bias/mAdam/dense_95/kernel/vAdam/dense_95/bias/v Adam/gru_23/gru_cell_23/kernel/v*Adam/gru_23/gru_cell_23/recurrent_kernel/vAdam/gru_23/gru_cell_23/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1511336??
?	
?
gru_23_while_cond_1510072*
&gru_23_while_gru_23_while_loop_counter0
,gru_23_while_gru_23_while_maximum_iterations
gru_23_while_placeholder
gru_23_while_placeholder_1
gru_23_while_placeholder_2,
(gru_23_while_less_gru_23_strided_slice_1C
?gru_23_while_gru_23_while_cond_1510072___redundant_placeholder0C
?gru_23_while_gru_23_while_cond_1510072___redundant_placeholder1C
?gru_23_while_gru_23_while_cond_1510072___redundant_placeholder2C
?gru_23_while_gru_23_while_cond_1510072___redundant_placeholder3
gru_23_while_identity
?
gru_23/while/LessLessgru_23_while_placeholder(gru_23_while_less_gru_23_strided_slice_1*
T0*
_output_shapes
: 2
gru_23/while/Lessr
gru_23/while/IdentityIdentitygru_23/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_23/while/Identity"7
gru_23_while_identitygru_23/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?J
?
gru_23_while_body_1510238*
&gru_23_while_gru_23_while_loop_counter0
,gru_23_while_gru_23_while_maximum_iterations
gru_23_while_placeholder
gru_23_while_placeholder_1
gru_23_while_placeholder_2)
%gru_23_while_gru_23_strided_slice_1_0e
agru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_06
2gru_23_while_gru_cell_23_readvariableop_resource_0=
9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0?
;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0
gru_23_while_identity
gru_23_while_identity_1
gru_23_while_identity_2
gru_23_while_identity_3
gru_23_while_identity_4'
#gru_23_while_gru_23_strided_slice_1c
_gru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor4
0gru_23_while_gru_cell_23_readvariableop_resource;
7gru_23_while_gru_cell_23_matmul_readvariableop_resource=
9gru_23_while_gru_cell_23_matmul_1_readvariableop_resource??
>gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_0gru_23_while_placeholderGgru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_23/while/TensorArrayV2Read/TensorListGetItem?
'gru_23/while/gru_cell_23/ReadVariableOpReadVariableOp2gru_23_while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_23/while/gru_cell_23/ReadVariableOp?
 gru_23/while/gru_cell_23/unstackUnpack/gru_23/while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_23/while/gru_cell_23/unstack?
.gru_23/while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_23/while/gru_cell_23/MatMul/ReadVariableOp?
gru_23/while/gru_cell_23/MatMulMatMul7gru_23/while/TensorArrayV2Read/TensorListGetItem:item:06gru_23/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_23/while/gru_cell_23/MatMul?
 gru_23/while/gru_cell_23/BiasAddBiasAdd)gru_23/while/gru_cell_23/MatMul:product:0)gru_23/while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_23/while/gru_cell_23/BiasAdd?
gru_23/while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_23/while/gru_cell_23/Const?
(gru_23/while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_23/while/gru_cell_23/split/split_dim?
gru_23/while/gru_cell_23/splitSplit1gru_23/while/gru_cell_23/split/split_dim:output:0)gru_23/while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2 
gru_23/while/gru_cell_23/split?
0gru_23/while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp?
!gru_23/while/gru_cell_23/MatMul_1MatMulgru_23_while_placeholder_28gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_23/while/gru_cell_23/MatMul_1?
"gru_23/while/gru_cell_23/BiasAdd_1BiasAdd+gru_23/while/gru_cell_23/MatMul_1:product:0)gru_23/while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_23/while/gru_cell_23/BiasAdd_1?
 gru_23/while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2"
 gru_23/while/gru_cell_23/Const_1?
*gru_23/while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_23/while/gru_cell_23/split_1/split_dim?
 gru_23/while/gru_cell_23/split_1SplitV+gru_23/while/gru_cell_23/BiasAdd_1:output:0)gru_23/while/gru_cell_23/Const_1:output:03gru_23/while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2"
 gru_23/while/gru_cell_23/split_1?
gru_23/while/gru_cell_23/addAddV2'gru_23/while/gru_cell_23/split:output:0)gru_23/while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/add?
 gru_23/while/gru_cell_23/SigmoidSigmoid gru_23/while/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22"
 gru_23/while/gru_cell_23/Sigmoid?
gru_23/while/gru_cell_23/add_1AddV2'gru_23/while/gru_cell_23/split:output:1)gru_23/while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_1?
"gru_23/while/gru_cell_23/Sigmoid_1Sigmoid"gru_23/while/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22$
"gru_23/while/gru_cell_23/Sigmoid_1?
gru_23/while/gru_cell_23/mulMul&gru_23/while/gru_cell_23/Sigmoid_1:y:0)gru_23/while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/mul?
gru_23/while/gru_cell_23/add_2AddV2'gru_23/while/gru_cell_23/split:output:2 gru_23/while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_2?
gru_23/while/gru_cell_23/ReluRelu"gru_23/while/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/Relu?
gru_23/while/gru_cell_23/mul_1Mul$gru_23/while/gru_cell_23/Sigmoid:y:0gru_23_while_placeholder_2*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/mul_1?
gru_23/while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_23/while/gru_cell_23/sub/x?
gru_23/while/gru_cell_23/subSub'gru_23/while/gru_cell_23/sub/x:output:0$gru_23/while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/sub?
gru_23/while/gru_cell_23/mul_2Mul gru_23/while/gru_cell_23/sub:z:0+gru_23/while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/mul_2?
gru_23/while/gru_cell_23/add_3AddV2"gru_23/while/gru_cell_23/mul_1:z:0"gru_23/while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_3?
1gru_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_23_while_placeholder_1gru_23_while_placeholder"gru_23/while/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_23/while/TensorArrayV2Write/TensorListSetItemj
gru_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/while/add/y?
gru_23/while/addAddV2gru_23_while_placeholdergru_23/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_23/while/addn
gru_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/while/add_1/y?
gru_23/while/add_1AddV2&gru_23_while_gru_23_while_loop_countergru_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_23/while/add_1s
gru_23/while/IdentityIdentitygru_23/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_23/while/Identity?
gru_23/while/Identity_1Identity,gru_23_while_gru_23_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_23/while/Identity_1u
gru_23/while/Identity_2Identitygru_23/while/add:z:0*
T0*
_output_shapes
: 2
gru_23/while/Identity_2?
gru_23/while/Identity_3IdentityAgru_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_23/while/Identity_3?
gru_23/while/Identity_4Identity"gru_23/while/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
gru_23/while/Identity_4"L
#gru_23_while_gru_23_strided_slice_1%gru_23_while_gru_23_strided_slice_1_0"x
9gru_23_while_gru_cell_23_matmul_1_readvariableop_resource;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0"t
7gru_23_while_gru_cell_23_matmul_readvariableop_resource9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0"f
0gru_23_while_gru_cell_23_readvariableop_resource2gru_23_while_gru_cell_23_readvariableop_resource_0"7
gru_23_while_identitygru_23/while/Identity:output:0";
gru_23_while_identity_1 gru_23/while/Identity_1:output:0";
gru_23_while_identity_2 gru_23/while/Identity_2:output:0";
gru_23_while_identity_3 gru_23/while/Identity_3:output:0";
gru_23_while_identity_4 gru_23/while/Identity_4:output:0"?
_gru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensoragru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1510931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1510931___redundant_placeholder05
1while_while_cond_1510931___redundant_placeholder15
1while_while_cond_1510931___redundant_placeholder25
1while_while_cond_1510931___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?`
?
#__inference__traced_restore_1511336
file_prefix$
 assignvariableop_dense_95_kernel$
 assignvariableop_1_dense_95_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate0
,assignvariableop_7_gru_23_gru_cell_23_kernel:
6assignvariableop_8_gru_23_gru_cell_23_recurrent_kernel.
*assignvariableop_9_gru_23_gru_cell_23_bias
assignvariableop_10_total
assignvariableop_11_count.
*assignvariableop_12_adam_dense_95_kernel_m,
(assignvariableop_13_adam_dense_95_bias_m8
4assignvariableop_14_adam_gru_23_gru_cell_23_kernel_mB
>assignvariableop_15_adam_gru_23_gru_cell_23_recurrent_kernel_m6
2assignvariableop_16_adam_gru_23_gru_cell_23_bias_m.
*assignvariableop_17_adam_dense_95_kernel_v,
(assignvariableop_18_adam_dense_95_bias_v8
4assignvariableop_19_adam_gru_23_gru_cell_23_kernel_vB
>assignvariableop_20_adam_gru_23_gru_cell_23_recurrent_kernel_v6
2assignvariableop_21_adam_gru_23_gru_cell_23_bias_v
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_95_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_95_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_gru_23_gru_cell_23_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_gru_23_gru_cell_23_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_gru_23_gru_cell_23_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_dense_95_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_dense_95_bias_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_gru_23_gru_cell_23_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_gru_23_gru_cell_23_recurrent_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_gru_23_gru_cell_23_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_95_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_95_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_gru_23_gru_cell_23_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_gru_23_gru_cell_23_recurrent_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_gru_23_gru_cell_23_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
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
?j
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510334

inputs.
*gru_23_gru_cell_23_readvariableop_resource5
1gru_23_gru_cell_23_matmul_readvariableop_resource7
3gru_23_gru_cell_23_matmul_1_readvariableop_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity??gru_23/whileR
gru_23/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_23/Shape?
gru_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice/stack?
gru_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_23/strided_slice/stack_1?
gru_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_23/strided_slice/stack_2?
gru_23/strided_sliceStridedSlicegru_23/Shape:output:0#gru_23/strided_slice/stack:output:0%gru_23/strided_slice/stack_1:output:0%gru_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_23/strided_slicej
gru_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_23/zeros/mul/y?
gru_23/zeros/mulMulgru_23/strided_slice:output:0gru_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_23/zeros/mulm
gru_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_23/zeros/Less/y?
gru_23/zeros/LessLessgru_23/zeros/mul:z:0gru_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_23/zeros/Lessp
gru_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_23/zeros/packed/1?
gru_23/zeros/packedPackgru_23/strided_slice:output:0gru_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_23/zeros/packedm
gru_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_23/zeros/Const?
gru_23/zerosFillgru_23/zeros/packed:output:0gru_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
gru_23/zeros?
gru_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_23/transpose/perm?
gru_23/transpose	Transposeinputsgru_23/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_23/transposed
gru_23/Shape_1Shapegru_23/transpose:y:0*
T0*
_output_shapes
:2
gru_23/Shape_1?
gru_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice_1/stack?
gru_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_1/stack_1?
gru_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_1/stack_2?
gru_23/strided_slice_1StridedSlicegru_23/Shape_1:output:0%gru_23/strided_slice_1/stack:output:0'gru_23/strided_slice_1/stack_1:output:0'gru_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_23/strided_slice_1?
"gru_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_23/TensorArrayV2/element_shape?
gru_23/TensorArrayV2TensorListReserve+gru_23/TensorArrayV2/element_shape:output:0gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_23/TensorArrayV2?
<gru_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_23/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_23/transpose:y:0Egru_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_23/TensorArrayUnstack/TensorListFromTensor?
gru_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice_2/stack?
gru_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_2/stack_1?
gru_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_2/stack_2?
gru_23/strided_slice_2StridedSlicegru_23/transpose:y:0%gru_23/strided_slice_2/stack:output:0'gru_23/strided_slice_2/stack_1:output:0'gru_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_23/strided_slice_2?
!gru_23/gru_cell_23/ReadVariableOpReadVariableOp*gru_23_gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_23/gru_cell_23/ReadVariableOp?
gru_23/gru_cell_23/unstackUnpack)gru_23/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_23/gru_cell_23/unstack?
(gru_23/gru_cell_23/MatMul/ReadVariableOpReadVariableOp1gru_23_gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_23/gru_cell_23/MatMul/ReadVariableOp?
gru_23/gru_cell_23/MatMulMatMulgru_23/strided_slice_2:output:00gru_23/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/MatMul?
gru_23/gru_cell_23/BiasAddBiasAdd#gru_23/gru_cell_23/MatMul:product:0#gru_23/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/BiasAddv
gru_23/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/gru_cell_23/Const?
"gru_23/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_23/gru_cell_23/split/split_dim?
gru_23/gru_cell_23/splitSplit+gru_23/gru_cell_23/split/split_dim:output:0#gru_23/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_23/gru_cell_23/split?
*gru_23/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp3gru_23_gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*gru_23/gru_cell_23/MatMul_1/ReadVariableOp?
gru_23/gru_cell_23/MatMul_1MatMulgru_23/zeros:output:02gru_23/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/MatMul_1?
gru_23/gru_cell_23/BiasAdd_1BiasAdd%gru_23/gru_cell_23/MatMul_1:product:0#gru_23/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/BiasAdd_1?
gru_23/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_23/gru_cell_23/Const_1?
$gru_23/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_23/gru_cell_23/split_1/split_dim?
gru_23/gru_cell_23/split_1SplitV%gru_23/gru_cell_23/BiasAdd_1:output:0#gru_23/gru_cell_23/Const_1:output:0-gru_23/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_23/gru_cell_23/split_1?
gru_23/gru_cell_23/addAddV2!gru_23/gru_cell_23/split:output:0#gru_23/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add?
gru_23/gru_cell_23/SigmoidSigmoidgru_23/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Sigmoid?
gru_23/gru_cell_23/add_1AddV2!gru_23/gru_cell_23/split:output:1#gru_23/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_1?
gru_23/gru_cell_23/Sigmoid_1Sigmoidgru_23/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Sigmoid_1?
gru_23/gru_cell_23/mulMul gru_23/gru_cell_23/Sigmoid_1:y:0#gru_23/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul?
gru_23/gru_cell_23/add_2AddV2!gru_23/gru_cell_23/split:output:2gru_23/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_2?
gru_23/gru_cell_23/ReluRelugru_23/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Relu?
gru_23/gru_cell_23/mul_1Mulgru_23/gru_cell_23/Sigmoid:y:0gru_23/zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul_1y
gru_23/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_23/gru_cell_23/sub/x?
gru_23/gru_cell_23/subSub!gru_23/gru_cell_23/sub/x:output:0gru_23/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/sub?
gru_23/gru_cell_23/mul_2Mulgru_23/gru_cell_23/sub:z:0%gru_23/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul_2?
gru_23/gru_cell_23/add_3AddV2gru_23/gru_cell_23/mul_1:z:0gru_23/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_3?
$gru_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$gru_23/TensorArrayV2_1/element_shape?
gru_23/TensorArrayV2_1TensorListReserve-gru_23/TensorArrayV2_1/element_shape:output:0gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_23/TensorArrayV2_1\
gru_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_23/time?
gru_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_23/while/maximum_iterationsx
gru_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_23/while/loop_counter?
gru_23/whileWhile"gru_23/while/loop_counter:output:0(gru_23/while/maximum_iterations:output:0gru_23/time:output:0gru_23/TensorArrayV2_1:handle:0gru_23/zeros:output:0gru_23/strided_slice_1:output:0>gru_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_23_gru_cell_23_readvariableop_resource1gru_23_gru_cell_23_matmul_readvariableop_resource3gru_23_gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_23_while_body_1510238*%
condR
gru_23_while_cond_1510237*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
gru_23/while?
7gru_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7gru_23/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_23/TensorArrayV2Stack/TensorListStackTensorListStackgru_23/while:output:3@gru_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)gru_23/TensorArrayV2Stack/TensorListStack?
gru_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_23/strided_slice_3/stack?
gru_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_23/strided_slice_3/stack_1?
gru_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_3/stack_2?
gru_23/strided_slice_3StridedSlice2gru_23/TensorArrayV2Stack/TensorListStack:tensor:0%gru_23/strided_slice_3/stack:output:0'gru_23/strided_slice_3/stack_1:output:0'gru_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
gru_23/strided_slice_3?
gru_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_23/transpose_1/perm?
gru_23/transpose_1	Transpose2gru_23/TensorArrayV2Stack/TensorListStack:tensor:0 gru_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
gru_23/transpose_1t
gru_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_23/runtime?
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
dense_95/MatMul/ReadVariableOp?
dense_95/MatMulMatMulgru_23/strided_slice_3:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_95/MatMul?
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp?
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_95/BiasAdd|
IdentityIdentitydense_95/BiasAdd:output:0^gru_23/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2
gru_23/whilegru_23/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?j
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510169

inputs.
*gru_23_gru_cell_23_readvariableop_resource5
1gru_23_gru_cell_23_matmul_readvariableop_resource7
3gru_23_gru_cell_23_matmul_1_readvariableop_resource+
'dense_95_matmul_readvariableop_resource,
(dense_95_biasadd_readvariableop_resource
identity??gru_23/whileR
gru_23/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_23/Shape?
gru_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice/stack?
gru_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_23/strided_slice/stack_1?
gru_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_23/strided_slice/stack_2?
gru_23/strided_sliceStridedSlicegru_23/Shape:output:0#gru_23/strided_slice/stack:output:0%gru_23/strided_slice/stack_1:output:0%gru_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_23/strided_slicej
gru_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
gru_23/zeros/mul/y?
gru_23/zeros/mulMulgru_23/strided_slice:output:0gru_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_23/zeros/mulm
gru_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru_23/zeros/Less/y?
gru_23/zeros/LessLessgru_23/zeros/mul:z:0gru_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_23/zeros/Lessp
gru_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
gru_23/zeros/packed/1?
gru_23/zeros/packedPackgru_23/strided_slice:output:0gru_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_23/zeros/packedm
gru_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_23/zeros/Const?
gru_23/zerosFillgru_23/zeros/packed:output:0gru_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
gru_23/zeros?
gru_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_23/transpose/perm?
gru_23/transpose	Transposeinputsgru_23/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru_23/transposed
gru_23/Shape_1Shapegru_23/transpose:y:0*
T0*
_output_shapes
:2
gru_23/Shape_1?
gru_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice_1/stack?
gru_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_1/stack_1?
gru_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_1/stack_2?
gru_23/strided_slice_1StridedSlicegru_23/Shape_1:output:0%gru_23/strided_slice_1/stack:output:0'gru_23/strided_slice_1/stack_1:output:0'gru_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_23/strided_slice_1?
"gru_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_23/TensorArrayV2/element_shape?
gru_23/TensorArrayV2TensorListReserve+gru_23/TensorArrayV2/element_shape:output:0gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_23/TensorArrayV2?
<gru_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2>
<gru_23/TensorArrayUnstack/TensorListFromTensor/element_shape?
.gru_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_23/transpose:y:0Egru_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_23/TensorArrayUnstack/TensorListFromTensor?
gru_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_23/strided_slice_2/stack?
gru_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_2/stack_1?
gru_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_2/stack_2?
gru_23/strided_slice_2StridedSlicegru_23/transpose:y:0%gru_23/strided_slice_2/stack:output:0'gru_23/strided_slice_2/stack_1:output:0'gru_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru_23/strided_slice_2?
!gru_23/gru_cell_23/ReadVariableOpReadVariableOp*gru_23_gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_23/gru_cell_23/ReadVariableOp?
gru_23/gru_cell_23/unstackUnpack)gru_23/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_23/gru_cell_23/unstack?
(gru_23/gru_cell_23/MatMul/ReadVariableOpReadVariableOp1gru_23_gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(gru_23/gru_cell_23/MatMul/ReadVariableOp?
gru_23/gru_cell_23/MatMulMatMulgru_23/strided_slice_2:output:00gru_23/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/MatMul?
gru_23/gru_cell_23/BiasAddBiasAdd#gru_23/gru_cell_23/MatMul:product:0#gru_23/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/BiasAddv
gru_23/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/gru_cell_23/Const?
"gru_23/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"gru_23/gru_cell_23/split/split_dim?
gru_23/gru_cell_23/splitSplit+gru_23/gru_cell_23/split/split_dim:output:0#gru_23/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_23/gru_cell_23/split?
*gru_23/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp3gru_23_gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02,
*gru_23/gru_cell_23/MatMul_1/ReadVariableOp?
gru_23/gru_cell_23/MatMul_1MatMulgru_23/zeros:output:02gru_23/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/MatMul_1?
gru_23/gru_cell_23/BiasAdd_1BiasAdd%gru_23/gru_cell_23/MatMul_1:product:0#gru_23/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_23/gru_cell_23/BiasAdd_1?
gru_23/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_23/gru_cell_23/Const_1?
$gru_23/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$gru_23/gru_cell_23/split_1/split_dim?
gru_23/gru_cell_23/split_1SplitV%gru_23/gru_cell_23/BiasAdd_1:output:0#gru_23/gru_cell_23/Const_1:output:0-gru_23/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_23/gru_cell_23/split_1?
gru_23/gru_cell_23/addAddV2!gru_23/gru_cell_23/split:output:0#gru_23/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add?
gru_23/gru_cell_23/SigmoidSigmoidgru_23/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Sigmoid?
gru_23/gru_cell_23/add_1AddV2!gru_23/gru_cell_23/split:output:1#gru_23/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_1?
gru_23/gru_cell_23/Sigmoid_1Sigmoidgru_23/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Sigmoid_1?
gru_23/gru_cell_23/mulMul gru_23/gru_cell_23/Sigmoid_1:y:0#gru_23/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul?
gru_23/gru_cell_23/add_2AddV2!gru_23/gru_cell_23/split:output:2gru_23/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_2?
gru_23/gru_cell_23/ReluRelugru_23/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/Relu?
gru_23/gru_cell_23/mul_1Mulgru_23/gru_cell_23/Sigmoid:y:0gru_23/zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul_1y
gru_23/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_23/gru_cell_23/sub/x?
gru_23/gru_cell_23/subSub!gru_23/gru_cell_23/sub/x:output:0gru_23/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/sub?
gru_23/gru_cell_23/mul_2Mulgru_23/gru_cell_23/sub:z:0%gru_23/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/mul_2?
gru_23/gru_cell_23/add_3AddV2gru_23/gru_cell_23/mul_1:z:0gru_23/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/gru_cell_23/add_3?
$gru_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2&
$gru_23/TensorArrayV2_1/element_shape?
gru_23/TensorArrayV2_1TensorListReserve-gru_23/TensorArrayV2_1/element_shape:output:0gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_23/TensorArrayV2_1\
gru_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_23/time?
gru_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru_23/while/maximum_iterationsx
gru_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_23/while/loop_counter?
gru_23/whileWhile"gru_23/while/loop_counter:output:0(gru_23/while/maximum_iterations:output:0gru_23/time:output:0gru_23/TensorArrayV2_1:handle:0gru_23/zeros:output:0gru_23/strided_slice_1:output:0>gru_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0*gru_23_gru_cell_23_readvariableop_resource1gru_23_gru_cell_23_matmul_readvariableop_resource3gru_23_gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*%
bodyR
gru_23_while_body_1510073*%
condR
gru_23_while_cond_1510072*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
gru_23/while?
7gru_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   29
7gru_23/TensorArrayV2Stack/TensorListStack/element_shape?
)gru_23/TensorArrayV2Stack/TensorListStackTensorListStackgru_23/while:output:3@gru_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02+
)gru_23/TensorArrayV2Stack/TensorListStack?
gru_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru_23/strided_slice_3/stack?
gru_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_23/strided_slice_3/stack_1?
gru_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_23/strided_slice_3/stack_2?
gru_23/strided_slice_3StridedSlice2gru_23/TensorArrayV2Stack/TensorListStack:tensor:0%gru_23/strided_slice_3/stack:output:0'gru_23/strided_slice_3/stack_1:output:0'gru_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
gru_23/strided_slice_3?
gru_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_23/transpose_1/perm?
gru_23/transpose_1	Transpose2gru_23/TensorArrayV2Stack/TensorListStack:tensor:0 gru_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
gru_23/transpose_1t
gru_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_23/runtime?
dense_95/MatMul/ReadVariableOpReadVariableOp'dense_95_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
dense_95/MatMul/ReadVariableOp?
dense_95/MatMulMatMulgru_23/strided_slice_3:output:0&dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_95/MatMul?
dense_95/BiasAdd/ReadVariableOpReadVariableOp(dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_95/BiasAdd/ReadVariableOp?
dense_95/BiasAddBiasAdddense_95/MatMul:product:0'dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_95/BiasAdd|
IdentityIdentitydense_95/BiasAdd:output:0^gru_23/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2
gru_23/whilegru_23/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1511022

inputs'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1510932*
condR
while_cond_1510931*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511143

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:?????????22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????22
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?@
?
while_body_1509594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_gru_23_layer_call_fn_1510693
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15093952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510523
inputs_0'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1510433*
condR
while_cond_1510432*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
-__inference_gru_cell_23_layer_call_fn_1511171

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?@
?
while_body_1509753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510682
inputs_0'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1510592*
condR
while_cond_1510591*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?7
?

 __inference__traced_save_1511260
file_prefix.
*savev2_dense_95_kernel_read_readvariableop,
(savev2_dense_95_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_gru_23_gru_cell_23_kernel_read_readvariableopB
>savev2_gru_23_gru_cell_23_recurrent_kernel_read_readvariableop6
2savev2_gru_23_gru_cell_23_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_95_kernel_m_read_readvariableop3
/savev2_adam_dense_95_bias_m_read_readvariableop?
;savev2_adam_gru_23_gru_cell_23_kernel_m_read_readvariableopI
Esavev2_adam_gru_23_gru_cell_23_recurrent_kernel_m_read_readvariableop=
9savev2_adam_gru_23_gru_cell_23_bias_m_read_readvariableop5
1savev2_adam_dense_95_kernel_v_read_readvariableop3
/savev2_adam_dense_95_bias_v_read_readvariableop?
;savev2_adam_gru_23_gru_cell_23_kernel_v_read_readvariableopI
Esavev2_adam_gru_23_gru_cell_23_recurrent_kernel_v_read_readvariableop=
9savev2_adam_gru_23_gru_cell_23_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b8941c8f17944186accf860222ca21b4/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_95_kernel_read_readvariableop(savev2_dense_95_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_gru_23_gru_cell_23_kernel_read_readvariableop>savev2_gru_23_gru_cell_23_recurrent_kernel_read_readvariableop2savev2_gru_23_gru_cell_23_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_95_kernel_m_read_readvariableop/savev2_adam_dense_95_bias_m_read_readvariableop;savev2_adam_gru_23_gru_cell_23_kernel_m_read_readvariableopEsavev2_adam_gru_23_gru_cell_23_recurrent_kernel_m_read_readvariableop9savev2_adam_gru_23_gru_cell_23_bias_m_read_readvariableop1savev2_adam_dense_95_kernel_v_read_readvariableop/savev2_adam_dense_95_bias_v_read_readvariableop;savev2_adam_gru_23_gru_cell_23_kernel_v_read_readvariableopEsavev2_adam_gru_23_gru_cell_23_recurrent_kernel_v_read_readvariableop9savev2_adam_gru_23_gru_cell_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:: : : : : :	?:	2?:	?: : :2::	?:	2?:	?:2::	?:	2?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%	!

_output_shapes
:	2?:%
!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	2?:%!

_output_shapes
:	?:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	2?:%!

_output_shapes
:	?:

_output_shapes
: 
?
?
E__inference_dense_95_layer_call_and_return_conditional_losses_1509883

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1509032

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:?????????22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????22
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?<
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1509513

inputs
gru_cell_23_1509437
gru_cell_23_1509439
gru_cell_23_1509441
identity??#gru_cell_23/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_23_1509437gru_cell_23_1509439gru_cell_23_1509441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090722%
#gru_cell_23/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_23_1509437gru_cell_23_1509439gru_cell_23_1509441*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1509449*
condR
while_cond_1509448*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^gru_cell_23/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#gru_cell_23/StatefulPartitionedCall#gru_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1509593
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1509593___redundant_placeholder05
1while_while_cond_1509593___redundant_placeholder15
1while_while_cond_1509593___redundant_placeholder25
1while_while_cond_1509593___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509966

inputs
gru_23_1509953
gru_23_1509955
gru_23_1509957
dense_95_1509960
dense_95_1509962
identity?? dense_95/StatefulPartitionedCall?gru_23/StatefulPartitionedCall?
gru_23/StatefulPartitionedCallStatefulPartitionedCallinputsgru_23_1509953gru_23_1509955gru_23_1509957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15098432 
gru_23/StatefulPartitionedCall?
 dense_95/StatefulPartitionedCallStatefulPartitionedCall'gru_23/StatefulPartitionedCall:output:0dense_95_1509960dense_95_1509962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_15098832"
 dense_95/StatefulPartitionedCall?
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0!^dense_95/StatefulPartitionedCall^gru_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall2@
gru_23/StatefulPartitionedCallgru_23/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_1509752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1509752___redundant_placeholder05
1while_while_cond_1509752___redundant_placeholder15
1while_while_cond_1509752___redundant_placeholder25
1while_while_cond_1509752___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
%__inference_signature_wrapper_1510004
gru_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_15089602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?
?
'sequential_95_gru_23_while_cond_1508863F
Bsequential_95_gru_23_while_sequential_95_gru_23_while_loop_counterL
Hsequential_95_gru_23_while_sequential_95_gru_23_while_maximum_iterations*
&sequential_95_gru_23_while_placeholder,
(sequential_95_gru_23_while_placeholder_1,
(sequential_95_gru_23_while_placeholder_2H
Dsequential_95_gru_23_while_less_sequential_95_gru_23_strided_slice_1_
[sequential_95_gru_23_while_sequential_95_gru_23_while_cond_1508863___redundant_placeholder0_
[sequential_95_gru_23_while_sequential_95_gru_23_while_cond_1508863___redundant_placeholder1_
[sequential_95_gru_23_while_sequential_95_gru_23_while_cond_1508863___redundant_placeholder2_
[sequential_95_gru_23_while_sequential_95_gru_23_while_cond_1508863___redundant_placeholder3'
#sequential_95_gru_23_while_identity
?
sequential_95/gru_23/while/LessLess&sequential_95_gru_23_while_placeholderDsequential_95_gru_23_while_less_sequential_95_gru_23_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_95/gru_23/while/Less?
#sequential_95/gru_23/while/IdentityIdentity#sequential_95/gru_23/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_95/gru_23/while/Identity"S
#sequential_95_gru_23_while_identity,sequential_95/gru_23/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?J
?
gru_23_while_body_1510073*
&gru_23_while_gru_23_while_loop_counter0
,gru_23_while_gru_23_while_maximum_iterations
gru_23_while_placeholder
gru_23_while_placeholder_1
gru_23_while_placeholder_2)
%gru_23_while_gru_23_strided_slice_1_0e
agru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_06
2gru_23_while_gru_cell_23_readvariableop_resource_0=
9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0?
;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0
gru_23_while_identity
gru_23_while_identity_1
gru_23_while_identity_2
gru_23_while_identity_3
gru_23_while_identity_4'
#gru_23_while_gru_23_strided_slice_1c
_gru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor4
0gru_23_while_gru_cell_23_readvariableop_resource;
7gru_23_while_gru_cell_23_matmul_readvariableop_resource=
9gru_23_while_gru_cell_23_matmul_1_readvariableop_resource??
>gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape?
0gru_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemagru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_0gru_23_while_placeholderGgru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype022
0gru_23/while/TensorArrayV2Read/TensorListGetItem?
'gru_23/while/gru_cell_23/ReadVariableOpReadVariableOp2gru_23_while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'gru_23/while/gru_cell_23/ReadVariableOp?
 gru_23/while/gru_cell_23/unstackUnpack/gru_23/while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2"
 gru_23/while/gru_cell_23/unstack?
.gru_23/while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype020
.gru_23/while/gru_cell_23/MatMul/ReadVariableOp?
gru_23/while/gru_cell_23/MatMulMatMul7gru_23/while/TensorArrayV2Read/TensorListGetItem:item:06gru_23/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
gru_23/while/gru_cell_23/MatMul?
 gru_23/while/gru_cell_23/BiasAddBiasAdd)gru_23/while/gru_cell_23/MatMul:product:0)gru_23/while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2"
 gru_23/while/gru_cell_23/BiasAdd?
gru_23/while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
gru_23/while/gru_cell_23/Const?
(gru_23/while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(gru_23/while/gru_cell_23/split/split_dim?
gru_23/while/gru_cell_23/splitSplit1gru_23/while/gru_cell_23/split/split_dim:output:0)gru_23/while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2 
gru_23/while/gru_cell_23/split?
0gru_23/while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype022
0gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp?
!gru_23/while/gru_cell_23/MatMul_1MatMulgru_23_while_placeholder_28gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!gru_23/while/gru_cell_23/MatMul_1?
"gru_23/while/gru_cell_23/BiasAdd_1BiasAdd+gru_23/while/gru_cell_23/MatMul_1:product:0)gru_23/while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2$
"gru_23/while/gru_cell_23/BiasAdd_1?
 gru_23/while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2"
 gru_23/while/gru_cell_23/Const_1?
*gru_23/while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*gru_23/while/gru_cell_23/split_1/split_dim?
 gru_23/while/gru_cell_23/split_1SplitV+gru_23/while/gru_cell_23/BiasAdd_1:output:0)gru_23/while/gru_cell_23/Const_1:output:03gru_23/while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2"
 gru_23/while/gru_cell_23/split_1?
gru_23/while/gru_cell_23/addAddV2'gru_23/while/gru_cell_23/split:output:0)gru_23/while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/add?
 gru_23/while/gru_cell_23/SigmoidSigmoid gru_23/while/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22"
 gru_23/while/gru_cell_23/Sigmoid?
gru_23/while/gru_cell_23/add_1AddV2'gru_23/while/gru_cell_23/split:output:1)gru_23/while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_1?
"gru_23/while/gru_cell_23/Sigmoid_1Sigmoid"gru_23/while/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22$
"gru_23/while/gru_cell_23/Sigmoid_1?
gru_23/while/gru_cell_23/mulMul&gru_23/while/gru_cell_23/Sigmoid_1:y:0)gru_23/while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/mul?
gru_23/while/gru_cell_23/add_2AddV2'gru_23/while/gru_cell_23/split:output:2 gru_23/while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_2?
gru_23/while/gru_cell_23/ReluRelu"gru_23/while/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/Relu?
gru_23/while/gru_cell_23/mul_1Mul$gru_23/while/gru_cell_23/Sigmoid:y:0gru_23_while_placeholder_2*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/mul_1?
gru_23/while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru_23/while/gru_cell_23/sub/x?
gru_23/while/gru_cell_23/subSub'gru_23/while/gru_cell_23/sub/x:output:0$gru_23/while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_23/while/gru_cell_23/sub?
gru_23/while/gru_cell_23/mul_2Mul gru_23/while/gru_cell_23/sub:z:0+gru_23/while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/mul_2?
gru_23/while/gru_cell_23/add_3AddV2"gru_23/while/gru_cell_23/mul_1:z:0"gru_23/while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22 
gru_23/while/gru_cell_23/add_3?
1gru_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_23_while_placeholder_1gru_23_while_placeholder"gru_23/while/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype023
1gru_23/while/TensorArrayV2Write/TensorListSetItemj
gru_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/while/add/y?
gru_23/while/addAddV2gru_23_while_placeholdergru_23/while/add/y:output:0*
T0*
_output_shapes
: 2
gru_23/while/addn
gru_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru_23/while/add_1/y?
gru_23/while/add_1AddV2&gru_23_while_gru_23_while_loop_countergru_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru_23/while/add_1s
gru_23/while/IdentityIdentitygru_23/while/add_1:z:0*
T0*
_output_shapes
: 2
gru_23/while/Identity?
gru_23/while/Identity_1Identity,gru_23_while_gru_23_while_maximum_iterations*
T0*
_output_shapes
: 2
gru_23/while/Identity_1u
gru_23/while/Identity_2Identitygru_23/while/add:z:0*
T0*
_output_shapes
: 2
gru_23/while/Identity_2?
gru_23/while/Identity_3IdentityAgru_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru_23/while/Identity_3?
gru_23/while/Identity_4Identity"gru_23/while/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
gru_23/while/Identity_4"L
#gru_23_while_gru_23_strided_slice_1%gru_23_while_gru_23_strided_slice_1_0"x
9gru_23_while_gru_cell_23_matmul_1_readvariableop_resource;gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0"t
7gru_23_while_gru_cell_23_matmul_readvariableop_resource9gru_23_while_gru_cell_23_matmul_readvariableop_resource_0"f
0gru_23_while_gru_cell_23_readvariableop_resource2gru_23_while_gru_cell_23_readvariableop_resource_0"7
gru_23_while_identitygru_23/while/Identity:output:0";
gru_23_while_identity_1 gru_23/while/Identity_1:output:0";
gru_23_while_identity_2 gru_23/while/Identity_2:output:0";
gru_23_while_identity_3 gru_23/while/Identity_3:output:0";
gru_23_while_identity_4 gru_23/while/Identity_4:output:0"?
_gru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensoragru_23_while_tensorarrayv2read_tensorlistgetitem_gru_23_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_sequential_95_layer_call_fn_1510364

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_95_layer_call_and_return_conditional_losses_15099662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_95_layer_call_fn_1510349

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_95_layer_call_and_return_conditional_losses_15099352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
while_body_1509331
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_23_1509353_0
while_gru_cell_23_1509355_0
while_gru_cell_23_1509357_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_23_1509353
while_gru_cell_23_1509355
while_gru_cell_23_1509357??)while/gru_cell_23/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_23_1509353_0while_gru_cell_23_1509355_0while_gru_cell_23_1509357_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090322+
)while/gru_cell_23/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_23/StatefulPartitionedCall:output:1*^while/gru_cell_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4"8
while_gru_cell_23_1509353while_gru_cell_23_1509353_0"8
while_gru_cell_23_1509355while_gru_cell_23_1509355_0"8
while_gru_cell_23_1509357while_gru_cell_23_1509357_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2V
)while/gru_cell_23/StatefulPartitionedCall)while/gru_cell_23/StatefulPartitionedCall: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_gru_23_layer_call_fn_1511033

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15096842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
܅
?
"__inference__wrapped_model_1508960
gru_23_input<
8sequential_95_gru_23_gru_cell_23_readvariableop_resourceC
?sequential_95_gru_23_gru_cell_23_matmul_readvariableop_resourceE
Asequential_95_gru_23_gru_cell_23_matmul_1_readvariableop_resource9
5sequential_95_dense_95_matmul_readvariableop_resource:
6sequential_95_dense_95_biasadd_readvariableop_resource
identity??sequential_95/gru_23/whilet
sequential_95/gru_23/ShapeShapegru_23_input*
T0*
_output_shapes
:2
sequential_95/gru_23/Shape?
(sequential_95/gru_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_95/gru_23/strided_slice/stack?
*sequential_95/gru_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_95/gru_23/strided_slice/stack_1?
*sequential_95/gru_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_95/gru_23/strided_slice/stack_2?
"sequential_95/gru_23/strided_sliceStridedSlice#sequential_95/gru_23/Shape:output:01sequential_95/gru_23/strided_slice/stack:output:03sequential_95/gru_23/strided_slice/stack_1:output:03sequential_95/gru_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_95/gru_23/strided_slice?
 sequential_95/gru_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22"
 sequential_95/gru_23/zeros/mul/y?
sequential_95/gru_23/zeros/mulMul+sequential_95/gru_23/strided_slice:output:0)sequential_95/gru_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_95/gru_23/zeros/mul?
!sequential_95/gru_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_95/gru_23/zeros/Less/y?
sequential_95/gru_23/zeros/LessLess"sequential_95/gru_23/zeros/mul:z:0*sequential_95/gru_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_95/gru_23/zeros/Less?
#sequential_95/gru_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22%
#sequential_95/gru_23/zeros/packed/1?
!sequential_95/gru_23/zeros/packedPack+sequential_95/gru_23/strided_slice:output:0,sequential_95/gru_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_95/gru_23/zeros/packed?
 sequential_95/gru_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_95/gru_23/zeros/Const?
sequential_95/gru_23/zerosFill*sequential_95/gru_23/zeros/packed:output:0)sequential_95/gru_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
sequential_95/gru_23/zeros?
#sequential_95/gru_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_95/gru_23/transpose/perm?
sequential_95/gru_23/transpose	Transposegru_23_input,sequential_95/gru_23/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
sequential_95/gru_23/transpose?
sequential_95/gru_23/Shape_1Shape"sequential_95/gru_23/transpose:y:0*
T0*
_output_shapes
:2
sequential_95/gru_23/Shape_1?
*sequential_95/gru_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_95/gru_23/strided_slice_1/stack?
,sequential_95/gru_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_95/gru_23/strided_slice_1/stack_1?
,sequential_95/gru_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_95/gru_23/strided_slice_1/stack_2?
$sequential_95/gru_23/strided_slice_1StridedSlice%sequential_95/gru_23/Shape_1:output:03sequential_95/gru_23/strided_slice_1/stack:output:05sequential_95/gru_23/strided_slice_1/stack_1:output:05sequential_95/gru_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_95/gru_23/strided_slice_1?
0sequential_95/gru_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_95/gru_23/TensorArrayV2/element_shape?
"sequential_95/gru_23/TensorArrayV2TensorListReserve9sequential_95/gru_23/TensorArrayV2/element_shape:output:0-sequential_95/gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_95/gru_23/TensorArrayV2?
Jsequential_95/gru_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jsequential_95/gru_23/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_95/gru_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_95/gru_23/transpose:y:0Ssequential_95/gru_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_95/gru_23/TensorArrayUnstack/TensorListFromTensor?
*sequential_95/gru_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_95/gru_23/strided_slice_2/stack?
,sequential_95/gru_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_95/gru_23/strided_slice_2/stack_1?
,sequential_95/gru_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_95/gru_23/strided_slice_2/stack_2?
$sequential_95/gru_23/strided_slice_2StridedSlice"sequential_95/gru_23/transpose:y:03sequential_95/gru_23/strided_slice_2/stack:output:05sequential_95/gru_23/strided_slice_2/stack_1:output:05sequential_95/gru_23/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$sequential_95/gru_23/strided_slice_2?
/sequential_95/gru_23/gru_cell_23/ReadVariableOpReadVariableOp8sequential_95_gru_23_gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype021
/sequential_95/gru_23/gru_cell_23/ReadVariableOp?
(sequential_95/gru_23/gru_cell_23/unstackUnpack7sequential_95/gru_23/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2*
(sequential_95/gru_23/gru_cell_23/unstack?
6sequential_95/gru_23/gru_cell_23/MatMul/ReadVariableOpReadVariableOp?sequential_95_gru_23_gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6sequential_95/gru_23/gru_cell_23/MatMul/ReadVariableOp?
'sequential_95/gru_23/gru_cell_23/MatMulMatMul-sequential_95/gru_23/strided_slice_2:output:0>sequential_95/gru_23/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'sequential_95/gru_23/gru_cell_23/MatMul?
(sequential_95/gru_23/gru_cell_23/BiasAddBiasAdd1sequential_95/gru_23/gru_cell_23/MatMul:product:01sequential_95/gru_23/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2*
(sequential_95/gru_23/gru_cell_23/BiasAdd?
&sequential_95/gru_23/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_95/gru_23/gru_cell_23/Const?
0sequential_95/gru_23/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_95/gru_23/gru_cell_23/split/split_dim?
&sequential_95/gru_23/gru_cell_23/splitSplit9sequential_95/gru_23/gru_cell_23/split/split_dim:output:01sequential_95/gru_23/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2(
&sequential_95/gru_23/gru_cell_23/split?
8sequential_95/gru_23/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOpAsequential_95_gru_23_gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02:
8sequential_95/gru_23/gru_cell_23/MatMul_1/ReadVariableOp?
)sequential_95/gru_23/gru_cell_23/MatMul_1MatMul#sequential_95/gru_23/zeros:output:0@sequential_95/gru_23/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_95/gru_23/gru_cell_23/MatMul_1?
*sequential_95/gru_23/gru_cell_23/BiasAdd_1BiasAdd3sequential_95/gru_23/gru_cell_23/MatMul_1:product:01sequential_95/gru_23/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2,
*sequential_95/gru_23/gru_cell_23/BiasAdd_1?
(sequential_95/gru_23/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2*
(sequential_95/gru_23/gru_cell_23/Const_1?
2sequential_95/gru_23/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2sequential_95/gru_23/gru_cell_23/split_1/split_dim?
(sequential_95/gru_23/gru_cell_23/split_1SplitV3sequential_95/gru_23/gru_cell_23/BiasAdd_1:output:01sequential_95/gru_23/gru_cell_23/Const_1:output:0;sequential_95/gru_23/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2*
(sequential_95/gru_23/gru_cell_23/split_1?
$sequential_95/gru_23/gru_cell_23/addAddV2/sequential_95/gru_23/gru_cell_23/split:output:01sequential_95/gru_23/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22&
$sequential_95/gru_23/gru_cell_23/add?
(sequential_95/gru_23/gru_cell_23/SigmoidSigmoid(sequential_95/gru_23/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22*
(sequential_95/gru_23/gru_cell_23/Sigmoid?
&sequential_95/gru_23/gru_cell_23/add_1AddV2/sequential_95/gru_23/gru_cell_23/split:output:11sequential_95/gru_23/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22(
&sequential_95/gru_23/gru_cell_23/add_1?
*sequential_95/gru_23/gru_cell_23/Sigmoid_1Sigmoid*sequential_95/gru_23/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22,
*sequential_95/gru_23/gru_cell_23/Sigmoid_1?
$sequential_95/gru_23/gru_cell_23/mulMul.sequential_95/gru_23/gru_cell_23/Sigmoid_1:y:01sequential_95/gru_23/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22&
$sequential_95/gru_23/gru_cell_23/mul?
&sequential_95/gru_23/gru_cell_23/add_2AddV2/sequential_95/gru_23/gru_cell_23/split:output:2(sequential_95/gru_23/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_95/gru_23/gru_cell_23/add_2?
%sequential_95/gru_23/gru_cell_23/ReluRelu*sequential_95/gru_23/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22'
%sequential_95/gru_23/gru_cell_23/Relu?
&sequential_95/gru_23/gru_cell_23/mul_1Mul,sequential_95/gru_23/gru_cell_23/Sigmoid:y:0#sequential_95/gru_23/zeros:output:0*
T0*'
_output_shapes
:?????????22(
&sequential_95/gru_23/gru_cell_23/mul_1?
&sequential_95/gru_23/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2(
&sequential_95/gru_23/gru_cell_23/sub/x?
$sequential_95/gru_23/gru_cell_23/subSub/sequential_95/gru_23/gru_cell_23/sub/x:output:0,sequential_95/gru_23/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22&
$sequential_95/gru_23/gru_cell_23/sub?
&sequential_95/gru_23/gru_cell_23/mul_2Mul(sequential_95/gru_23/gru_cell_23/sub:z:03sequential_95/gru_23/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22(
&sequential_95/gru_23/gru_cell_23/mul_2?
&sequential_95/gru_23/gru_cell_23/add_3AddV2*sequential_95/gru_23/gru_cell_23/mul_1:z:0*sequential_95/gru_23/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22(
&sequential_95/gru_23/gru_cell_23/add_3?
2sequential_95/gru_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   24
2sequential_95/gru_23/TensorArrayV2_1/element_shape?
$sequential_95/gru_23/TensorArrayV2_1TensorListReserve;sequential_95/gru_23/TensorArrayV2_1/element_shape:output:0-sequential_95/gru_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_95/gru_23/TensorArrayV2_1x
sequential_95/gru_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_95/gru_23/time?
-sequential_95/gru_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_95/gru_23/while/maximum_iterations?
'sequential_95/gru_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_95/gru_23/while/loop_counter?
sequential_95/gru_23/whileWhile0sequential_95/gru_23/while/loop_counter:output:06sequential_95/gru_23/while/maximum_iterations:output:0"sequential_95/gru_23/time:output:0-sequential_95/gru_23/TensorArrayV2_1:handle:0#sequential_95/gru_23/zeros:output:0-sequential_95/gru_23/strided_slice_1:output:0Lsequential_95/gru_23/TensorArrayUnstack/TensorListFromTensor:output_handle:08sequential_95_gru_23_gru_cell_23_readvariableop_resource?sequential_95_gru_23_gru_cell_23_matmul_readvariableop_resourceAsequential_95_gru_23_gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*3
body+R)
'sequential_95_gru_23_while_body_1508864*3
cond+R)
'sequential_95_gru_23_while_cond_1508863*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
sequential_95/gru_23/while?
Esequential_95/gru_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2G
Esequential_95/gru_23/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_95/gru_23/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_95/gru_23/while:output:3Nsequential_95/gru_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype029
7sequential_95/gru_23/TensorArrayV2Stack/TensorListStack?
*sequential_95/gru_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_95/gru_23/strided_slice_3/stack?
,sequential_95/gru_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_95/gru_23/strided_slice_3/stack_1?
,sequential_95/gru_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_95/gru_23/strided_slice_3/stack_2?
$sequential_95/gru_23/strided_slice_3StridedSlice@sequential_95/gru_23/TensorArrayV2Stack/TensorListStack:tensor:03sequential_95/gru_23/strided_slice_3/stack:output:05sequential_95/gru_23/strided_slice_3/stack_1:output:05sequential_95/gru_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2&
$sequential_95/gru_23/strided_slice_3?
%sequential_95/gru_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_95/gru_23/transpose_1/perm?
 sequential_95/gru_23/transpose_1	Transpose@sequential_95/gru_23/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_95/gru_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_95/gru_23/transpose_1?
sequential_95/gru_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_95/gru_23/runtime?
,sequential_95/dense_95/MatMul/ReadVariableOpReadVariableOp5sequential_95_dense_95_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,sequential_95/dense_95/MatMul/ReadVariableOp?
sequential_95/dense_95/MatMulMatMul-sequential_95/gru_23/strided_slice_3:output:04sequential_95/dense_95/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_95/dense_95/MatMul?
-sequential_95/dense_95/BiasAdd/ReadVariableOpReadVariableOp6sequential_95_dense_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_95/dense_95/BiasAdd/ReadVariableOp?
sequential_95/dense_95/BiasAddBiasAdd'sequential_95/dense_95/MatMul:product:05sequential_95/dense_95/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_95/dense_95/BiasAdd?
IdentityIdentity'sequential_95/dense_95/BiasAdd:output:0^sequential_95/gru_23/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::28
sequential_95/gru_23/whilesequential_95/gru_23/while:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510863

inputs'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1510773*
condR
while_cond_1510772*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509916
gru_23_input
gru_23_1509903
gru_23_1509905
gru_23_1509907
dense_95_1509910
dense_95_1509912
identity?? dense_95/StatefulPartitionedCall?gru_23/StatefulPartitionedCall?
gru_23/StatefulPartitionedCallStatefulPartitionedCallgru_23_inputgru_23_1509903gru_23_1509905gru_23_1509907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15098432 
gru_23/StatefulPartitionedCall?
 dense_95/StatefulPartitionedCallStatefulPartitionedCall'gru_23/StatefulPartitionedCall:output:0dense_95_1509910dense_95_1509912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_15098832"
 dense_95/StatefulPartitionedCall?
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0!^dense_95/StatefulPartitionedCall^gru_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall2@
gru_23/StatefulPartitionedCallgru_23/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?	
?
-__inference_gru_cell_23_layer_call_fn_1511157

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?
?
while_cond_1510772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1510772___redundant_placeholder05
1while_while_cond_1510772___redundant_placeholder15
1while_while_cond_1510772___redundant_placeholder25
1while_while_cond_1510772___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1510591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1510591___redundant_placeholder05
1while_while_cond_1510591___redundant_placeholder15
1while_while_cond_1510591___redundant_placeholder25
1while_while_cond_1510591___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_sequential_95_layer_call_fn_1509948
gru_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_95_layer_call_and_return_conditional_losses_15099352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?@
?
while_body_1510773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?@
?
while_body_1510932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?	
?
gru_23_while_cond_1510237*
&gru_23_while_gru_23_while_loop_counter0
,gru_23_while_gru_23_while_maximum_iterations
gru_23_while_placeholder
gru_23_while_placeholder_1
gru_23_while_placeholder_2,
(gru_23_while_less_gru_23_strided_slice_1C
?gru_23_while_gru_23_while_cond_1510237___redundant_placeholder0C
?gru_23_while_gru_23_while_cond_1510237___redundant_placeholder1C
?gru_23_while_gru_23_while_cond_1510237___redundant_placeholder2C
?gru_23_while_gru_23_while_cond_1510237___redundant_placeholder3
gru_23_while_identity
?
gru_23/while/LessLessgru_23_while_placeholder(gru_23_while_less_gru_23_strided_slice_1*
T0*
_output_shapes
: 2
gru_23/while/Lessr
gru_23/while/IdentityIdentitygru_23/while/Less:z:0*
T0
*
_output_shapes
: 2
gru_23/while/Identity"7
gru_23_while_identitygru_23/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_1509330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1509330___redundant_placeholder05
1while_while_cond_1509330___redundant_placeholder15
1while_while_cond_1509330___redundant_placeholder25
1while_while_cond_1509330___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1509843

inputs'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1509753*
condR
while_cond_1509752*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1509684

inputs'
#gru_cell_23_readvariableop_resource.
*gru_cell_23_matmul_readvariableop_resource0
,gru_cell_23_matmul_1_readvariableop_resource
identity??whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
gru_cell_23/ReadVariableOpReadVariableOp#gru_cell_23_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell_23/ReadVariableOp?
gru_cell_23/unstackUnpack"gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell_23/unstack?
!gru_cell_23/MatMul/ReadVariableOpReadVariableOp*gru_cell_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!gru_cell_23/MatMul/ReadVariableOp?
gru_cell_23/MatMulMatMulstrided_slice_2:output:0)gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul?
gru_cell_23/BiasAddBiasAddgru_cell_23/MatMul:product:0gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAddh
gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell_23/Const?
gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split/split_dim?
gru_cell_23/splitSplit$gru_cell_23/split/split_dim:output:0gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split?
#gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02%
#gru_cell_23/MatMul_1/ReadVariableOp?
gru_cell_23/MatMul_1MatMulzeros:output:0+gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
gru_cell_23/MatMul_1?
gru_cell_23/BiasAdd_1BiasAddgru_cell_23/MatMul_1:product:0gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
gru_cell_23/BiasAdd_1
gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
gru_cell_23/Const_1?
gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_cell_23/split_1/split_dim?
gru_cell_23/split_1SplitVgru_cell_23/BiasAdd_1:output:0gru_cell_23/Const_1:output:0&gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
gru_cell_23/split_1?
gru_cell_23/addAddV2gru_cell_23/split:output:0gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add|
gru_cell_23/SigmoidSigmoidgru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid?
gru_cell_23/add_1AddV2gru_cell_23/split:output:1gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_1?
gru_cell_23/Sigmoid_1Sigmoidgru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Sigmoid_1?
gru_cell_23/mulMulgru_cell_23/Sigmoid_1:y:0gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul?
gru_cell_23/add_2AddV2gru_cell_23/split:output:2gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_2u
gru_cell_23/ReluRelugru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/Relu?
gru_cell_23/mul_1Mulgru_cell_23/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_1k
gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell_23/sub/x?
gru_cell_23/subSubgru_cell_23/sub/x:output:0gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/sub?
gru_cell_23/mul_2Mulgru_cell_23/sub:z:0gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/mul_2?
gru_cell_23/add_3AddV2gru_cell_23/mul_1:z:0gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
gru_cell_23/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_23_readvariableop_resource*gru_cell_23_matmul_readvariableop_resource,gru_cell_23_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1509594*
condR
while_cond_1509593*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimet
IdentityIdentitystrided_slice_3:output:0^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511103

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:?????????22
Relu^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????22
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????2
"
_user_specified_name
states/0
?<
?
C__inference_gru_23_layer_call_and_return_conditional_losses_1509395

inputs
gru_cell_23_1509319
gru_cell_23_1509321
gru_cell_23_1509323
identity??#gru_cell_23/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????22
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
#gru_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_23_1509319gru_cell_23_1509321gru_cell_23_1509323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090322%
#gru_cell_23/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_23_1509319gru_cell_23_1509321gru_cell_23_1509323*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????2: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_1509331*
condR
while_cond_1509330*8
output_shapes'
%: : : : :?????????2: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????2   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????2*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????2*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????22
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^gru_cell_23/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#gru_cell_23/StatefulPartitionedCall#gru_cell_23/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1509072

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1?y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
split?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:?????????22
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????22	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:?????????22
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????22
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:?????????22
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:?????????22
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:?????????22
Relu\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????22
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
subd
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:?????????22
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:?????????22
add_3]
IdentityIdentity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identitya

Identity_1Identity	add_3:z:0*
T0*'
_output_shapes
:?????????22

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????2::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????2
 
_user_specified_namestates
?^
?

'sequential_95_gru_23_while_body_1508864F
Bsequential_95_gru_23_while_sequential_95_gru_23_while_loop_counterL
Hsequential_95_gru_23_while_sequential_95_gru_23_while_maximum_iterations*
&sequential_95_gru_23_while_placeholder,
(sequential_95_gru_23_while_placeholder_1,
(sequential_95_gru_23_while_placeholder_2E
Asequential_95_gru_23_while_sequential_95_gru_23_strided_slice_1_0?
}sequential_95_gru_23_while_tensorarrayv2read_tensorlistgetitem_sequential_95_gru_23_tensorarrayunstack_tensorlistfromtensor_0D
@sequential_95_gru_23_while_gru_cell_23_readvariableop_resource_0K
Gsequential_95_gru_23_while_gru_cell_23_matmul_readvariableop_resource_0M
Isequential_95_gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0'
#sequential_95_gru_23_while_identity)
%sequential_95_gru_23_while_identity_1)
%sequential_95_gru_23_while_identity_2)
%sequential_95_gru_23_while_identity_3)
%sequential_95_gru_23_while_identity_4C
?sequential_95_gru_23_while_sequential_95_gru_23_strided_slice_1
{sequential_95_gru_23_while_tensorarrayv2read_tensorlistgetitem_sequential_95_gru_23_tensorarrayunstack_tensorlistfromtensorB
>sequential_95_gru_23_while_gru_cell_23_readvariableop_resourceI
Esequential_95_gru_23_while_gru_cell_23_matmul_readvariableop_resourceK
Gsequential_95_gru_23_while_gru_cell_23_matmul_1_readvariableop_resource??
Lsequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lsequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_95_gru_23_while_tensorarrayv2read_tensorlistgetitem_sequential_95_gru_23_tensorarrayunstack_tensorlistfromtensor_0&sequential_95_gru_23_while_placeholderUsequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>sequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItem?
5sequential_95/gru_23/while/gru_cell_23/ReadVariableOpReadVariableOp@sequential_95_gru_23_while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5sequential_95/gru_23/while/gru_cell_23/ReadVariableOp?
.sequential_95/gru_23/while/gru_cell_23/unstackUnpack=sequential_95/gru_23/while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num20
.sequential_95/gru_23/while/gru_cell_23/unstack?
<sequential_95/gru_23/while/gru_cell_23/MatMul/ReadVariableOpReadVariableOpGsequential_95_gru_23_while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<sequential_95/gru_23/while/gru_cell_23/MatMul/ReadVariableOp?
-sequential_95/gru_23/while/gru_cell_23/MatMulMatMulEsequential_95/gru_23/while/TensorArrayV2Read/TensorListGetItem:item:0Dsequential_95/gru_23/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-sequential_95/gru_23/while/gru_cell_23/MatMul?
.sequential_95/gru_23/while/gru_cell_23/BiasAddBiasAdd7sequential_95/gru_23/while/gru_cell_23/MatMul:product:07sequential_95/gru_23/while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????20
.sequential_95/gru_23/while/gru_cell_23/BiasAdd?
,sequential_95/gru_23/while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,sequential_95/gru_23/while/gru_cell_23/Const?
6sequential_95/gru_23/while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6sequential_95/gru_23/while/gru_cell_23/split/split_dim?
,sequential_95/gru_23/while/gru_cell_23/splitSplit?sequential_95/gru_23/while/gru_cell_23/split/split_dim:output:07sequential_95/gru_23/while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2.
,sequential_95/gru_23/while/gru_cell_23/split?
>sequential_95/gru_23/while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOpIsequential_95_gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02@
>sequential_95/gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp?
/sequential_95/gru_23/while/gru_cell_23/MatMul_1MatMul(sequential_95_gru_23_while_placeholder_2Fsequential_95/gru_23/while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_95/gru_23/while/gru_cell_23/MatMul_1?
0sequential_95/gru_23/while/gru_cell_23/BiasAdd_1BiasAdd9sequential_95/gru_23/while/gru_cell_23/MatMul_1:product:07sequential_95/gru_23/while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????22
0sequential_95/gru_23/while/gru_cell_23/BiasAdd_1?
.sequential_95/gru_23/while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????20
.sequential_95/gru_23/while/gru_cell_23/Const_1?
8sequential_95/gru_23/while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2:
8sequential_95/gru_23/while/gru_cell_23/split_1/split_dim?
.sequential_95/gru_23/while/gru_cell_23/split_1SplitV9sequential_95/gru_23/while/gru_cell_23/BiasAdd_1:output:07sequential_95/gru_23/while/gru_cell_23/Const_1:output:0Asequential_95/gru_23/while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split20
.sequential_95/gru_23/while/gru_cell_23/split_1?
*sequential_95/gru_23/while/gru_cell_23/addAddV25sequential_95/gru_23/while/gru_cell_23/split:output:07sequential_95/gru_23/while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22,
*sequential_95/gru_23/while/gru_cell_23/add?
.sequential_95/gru_23/while/gru_cell_23/SigmoidSigmoid.sequential_95/gru_23/while/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????220
.sequential_95/gru_23/while/gru_cell_23/Sigmoid?
,sequential_95/gru_23/while/gru_cell_23/add_1AddV25sequential_95/gru_23/while/gru_cell_23/split:output:17sequential_95/gru_23/while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22.
,sequential_95/gru_23/while/gru_cell_23/add_1?
0sequential_95/gru_23/while/gru_cell_23/Sigmoid_1Sigmoid0sequential_95/gru_23/while/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????222
0sequential_95/gru_23/while/gru_cell_23/Sigmoid_1?
*sequential_95/gru_23/while/gru_cell_23/mulMul4sequential_95/gru_23/while/gru_cell_23/Sigmoid_1:y:07sequential_95/gru_23/while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22,
*sequential_95/gru_23/while/gru_cell_23/mul?
,sequential_95/gru_23/while/gru_cell_23/add_2AddV25sequential_95/gru_23/while/gru_cell_23/split:output:2.sequential_95/gru_23/while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_95/gru_23/while/gru_cell_23/add_2?
+sequential_95/gru_23/while/gru_cell_23/ReluRelu0sequential_95/gru_23/while/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22-
+sequential_95/gru_23/while/gru_cell_23/Relu?
,sequential_95/gru_23/while/gru_cell_23/mul_1Mul2sequential_95/gru_23/while/gru_cell_23/Sigmoid:y:0(sequential_95_gru_23_while_placeholder_2*
T0*'
_output_shapes
:?????????22.
,sequential_95/gru_23/while/gru_cell_23/mul_1?
,sequential_95/gru_23/while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,sequential_95/gru_23/while/gru_cell_23/sub/x?
*sequential_95/gru_23/while/gru_cell_23/subSub5sequential_95/gru_23/while/gru_cell_23/sub/x:output:02sequential_95/gru_23/while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22,
*sequential_95/gru_23/while/gru_cell_23/sub?
,sequential_95/gru_23/while/gru_cell_23/mul_2Mul.sequential_95/gru_23/while/gru_cell_23/sub:z:09sequential_95/gru_23/while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22.
,sequential_95/gru_23/while/gru_cell_23/mul_2?
,sequential_95/gru_23/while/gru_cell_23/add_3AddV20sequential_95/gru_23/while/gru_cell_23/mul_1:z:00sequential_95/gru_23/while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22.
,sequential_95/gru_23/while/gru_cell_23/add_3?
?sequential_95/gru_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_95_gru_23_while_placeholder_1&sequential_95_gru_23_while_placeholder0sequential_95/gru_23/while/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02A
?sequential_95/gru_23/while/TensorArrayV2Write/TensorListSetItem?
 sequential_95/gru_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_95/gru_23/while/add/y?
sequential_95/gru_23/while/addAddV2&sequential_95_gru_23_while_placeholder)sequential_95/gru_23/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_95/gru_23/while/add?
"sequential_95/gru_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_95/gru_23/while/add_1/y?
 sequential_95/gru_23/while/add_1AddV2Bsequential_95_gru_23_while_sequential_95_gru_23_while_loop_counter+sequential_95/gru_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_95/gru_23/while/add_1?
#sequential_95/gru_23/while/IdentityIdentity$sequential_95/gru_23/while/add_1:z:0*
T0*
_output_shapes
: 2%
#sequential_95/gru_23/while/Identity?
%sequential_95/gru_23/while/Identity_1IdentityHsequential_95_gru_23_while_sequential_95_gru_23_while_maximum_iterations*
T0*
_output_shapes
: 2'
%sequential_95/gru_23/while/Identity_1?
%sequential_95/gru_23/while/Identity_2Identity"sequential_95/gru_23/while/add:z:0*
T0*
_output_shapes
: 2'
%sequential_95/gru_23/while/Identity_2?
%sequential_95/gru_23/while/Identity_3IdentityOsequential_95/gru_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2'
%sequential_95/gru_23/while/Identity_3?
%sequential_95/gru_23/while/Identity_4Identity0sequential_95/gru_23/while/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22'
%sequential_95/gru_23/while/Identity_4"?
Gsequential_95_gru_23_while_gru_cell_23_matmul_1_readvariableop_resourceIsequential_95_gru_23_while_gru_cell_23_matmul_1_readvariableop_resource_0"?
Esequential_95_gru_23_while_gru_cell_23_matmul_readvariableop_resourceGsequential_95_gru_23_while_gru_cell_23_matmul_readvariableop_resource_0"?
>sequential_95_gru_23_while_gru_cell_23_readvariableop_resource@sequential_95_gru_23_while_gru_cell_23_readvariableop_resource_0"S
#sequential_95_gru_23_while_identity,sequential_95/gru_23/while/Identity:output:0"W
%sequential_95_gru_23_while_identity_1.sequential_95/gru_23/while/Identity_1:output:0"W
%sequential_95_gru_23_while_identity_2.sequential_95/gru_23/while/Identity_2:output:0"W
%sequential_95_gru_23_while_identity_3.sequential_95/gru_23/while/Identity_3:output:0"W
%sequential_95_gru_23_while_identity_4.sequential_95/gru_23/while/Identity_4:output:0"?
?sequential_95_gru_23_while_sequential_95_gru_23_strided_slice_1Asequential_95_gru_23_while_sequential_95_gru_23_strided_slice_1_0"?
{sequential_95_gru_23_while_tensorarrayv2read_tensorlistgetitem_sequential_95_gru_23_tensorarrayunstack_tensorlistfromtensor}sequential_95_gru_23_while_tensorarrayv2read_tensorlistgetitem_sequential_95_gru_23_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509935

inputs
gru_23_1509922
gru_23_1509924
gru_23_1509926
dense_95_1509929
dense_95_1509931
identity?? dense_95/StatefulPartitionedCall?gru_23/StatefulPartitionedCall?
gru_23/StatefulPartitionedCallStatefulPartitionedCallinputsgru_23_1509922gru_23_1509924gru_23_1509926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15096842 
gru_23/StatefulPartitionedCall?
 dense_95/StatefulPartitionedCallStatefulPartitionedCall'gru_23/StatefulPartitionedCall:output:0dense_95_1509929dense_95_1509931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_15098832"
 dense_95/StatefulPartitionedCall?
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0!^dense_95/StatefulPartitionedCall^gru_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall2@
gru_23/StatefulPartitionedCallgru_23/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509900
gru_23_input
gru_23_1509866
gru_23_1509868
gru_23_1509870
dense_95_1509894
dense_95_1509896
identity?? dense_95/StatefulPartitionedCall?gru_23/StatefulPartitionedCall?
gru_23/StatefulPartitionedCallStatefulPartitionedCallgru_23_inputgru_23_1509866gru_23_1509868gru_23_1509870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15096842 
gru_23/StatefulPartitionedCall?
 dense_95/StatefulPartitionedCallStatefulPartitionedCall'gru_23/StatefulPartitionedCall:output:0dense_95_1509894dense_95_1509896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_15098832"
 dense_95/StatefulPartitionedCall?
IdentityIdentity)dense_95/StatefulPartitionedCall:output:0!^dense_95/StatefulPartitionedCall^gru_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::2D
 dense_95/StatefulPartitionedCall dense_95/StatefulPartitionedCall2@
gru_23/StatefulPartitionedCallgru_23/StatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?
?
(__inference_gru_23_layer_call_fn_1510704
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15095132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
E__inference_dense_95_layer_call_and_return_conditional_losses_1511054

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2:::O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?!
?
while_body_1509449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_23_1509471_0
while_gru_cell_23_1509473_0
while_gru_cell_23_1509475_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_23_1509471
while_gru_cell_23_1509473
while_gru_cell_23_1509475??)while/gru_cell_23/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/gru_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_23_1509471_0while_gru_cell_23_1509473_0while_gru_cell_23_1509475_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????2:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_15090722+
)while/gru_cell_23/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_23/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/gru_cell_23/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/gru_cell_23/StatefulPartitionedCall:output:1*^while/gru_cell_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22
while/Identity_4"8
while_gru_cell_23_1509471while_gru_cell_23_1509471_0"8
while_gru_cell_23_1509473while_gru_cell_23_1509473_0"8
while_gru_cell_23_1509475while_gru_cell_23_1509475_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::2V
)while/gru_cell_23/StatefulPartitionedCall)while/gru_cell_23/StatefulPartitionedCall: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?

*__inference_dense_95_layer_call_fn_1511063

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_95_layer_call_and_return_conditional_losses_15098832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
(__inference_gru_23_layer_call_fn_1511044

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_23_layer_call_and_return_conditional_losses_15098432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_1509448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1509448___redundant_placeholder05
1while_while_cond_1509448___redundant_placeholder15
1while_while_cond_1509448___redundant_placeholder25
1while_while_cond_1509448___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_1510433
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_1510432
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1510432___redundant_placeholder05
1while_while_cond_1510432___redundant_placeholder15
1while_while_cond_1510432___redundant_placeholder25
1while_while_cond_1510432___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????2: ::::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_sequential_95_layer_call_fn_1509979
gru_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallgru_23_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_95_layer_call_and_return_conditional_losses_15099662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????:::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:?????????
&
_user_specified_namegru_23_input
?@
?
while_body_1510592
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
+while_gru_cell_23_readvariableop_resource_06
2while_gru_cell_23_matmul_readvariableop_resource_08
4while_gru_cell_23_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
)while_gru_cell_23_readvariableop_resource4
0while_gru_cell_23_matmul_readvariableop_resource6
2while_gru_cell_23_matmul_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
 while/gru_cell_23/ReadVariableOpReadVariableOp+while_gru_cell_23_readvariableop_resource_0*
_output_shapes
:	?*
dtype02"
 while/gru_cell_23/ReadVariableOp?
while/gru_cell_23/unstackUnpack(while/gru_cell_23/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell_23/unstack?
'while/gru_cell_23/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_23_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/gru_cell_23/MatMul/ReadVariableOp?
while/gru_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul?
while/gru_cell_23/BiasAddBiasAdd"while/gru_cell_23/MatMul:product:0"while/gru_cell_23/unstack:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAddt
while/gru_cell_23/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell_23/Const?
!while/gru_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!while/gru_cell_23/split/split_dim?
while/gru_cell_23/splitSplit*while/gru_cell_23/split/split_dim:output:0"while/gru_cell_23/BiasAdd:output:0*
T0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split?
)while/gru_cell_23/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	2?*
dtype02+
)while/gru_cell_23/MatMul_1/ReadVariableOp?
while/gru_cell_23/MatMul_1MatMulwhile_placeholder_21while/gru_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/MatMul_1?
while/gru_cell_23/BiasAdd_1BiasAdd$while/gru_cell_23/MatMul_1:product:0"while/gru_cell_23/unstack:output:1*
T0*(
_output_shapes
:??????????2
while/gru_cell_23/BiasAdd_1?
while/gru_cell_23/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"2   2   ????2
while/gru_cell_23/Const_1?
#while/gru_cell_23/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#while/gru_cell_23/split_1/split_dim?
while/gru_cell_23/split_1SplitV$while/gru_cell_23/BiasAdd_1:output:0"while/gru_cell_23/Const_1:output:0,while/gru_cell_23/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:?????????2:?????????2:?????????2*
	num_split2
while/gru_cell_23/split_1?
while/gru_cell_23/addAddV2 while/gru_cell_23/split:output:0"while/gru_cell_23/split_1:output:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add?
while/gru_cell_23/SigmoidSigmoidwhile/gru_cell_23/add:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid?
while/gru_cell_23/add_1AddV2 while/gru_cell_23/split:output:1"while/gru_cell_23/split_1:output:1*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_1?
while/gru_cell_23/Sigmoid_1Sigmoidwhile/gru_cell_23/add_1:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Sigmoid_1?
while/gru_cell_23/mulMulwhile/gru_cell_23/Sigmoid_1:y:0"while/gru_cell_23/split_1:output:2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul?
while/gru_cell_23/add_2AddV2 while/gru_cell_23/split:output:2while/gru_cell_23/mul:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_2?
while/gru_cell_23/ReluReluwhile/gru_cell_23/add_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/Relu?
while/gru_cell_23/mul_1Mulwhile/gru_cell_23/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_1w
while/gru_cell_23/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell_23/sub/x?
while/gru_cell_23/subSub while/gru_cell_23/sub/x:output:0while/gru_cell_23/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/sub?
while/gru_cell_23/mul_2Mulwhile/gru_cell_23/sub:z:0$while/gru_cell_23/Relu:activations:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/mul_2?
while/gru_cell_23/add_3AddV2while/gru_cell_23/mul_1:z:0while/gru_cell_23/mul_2:z:0*
T0*'
_output_shapes
:?????????22
while/gru_cell_23/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_23/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/gru_cell_23/add_3:z:0*
T0*'
_output_shapes
:?????????22
while/Identity_4"j
2while_gru_cell_23_matmul_1_readvariableop_resource4while_gru_cell_23_matmul_1_readvariableop_resource_0"f
0while_gru_cell_23_matmul_readvariableop_resource2while_gru_cell_23_matmul_readvariableop_resource_0"X
)while_gru_cell_23_readvariableop_resource+while_gru_cell_23_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????2: : :::: 
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
:?????????2:

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
gru_23_input9
serving_default_gru_23_input:0?????????<
dense_950
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?!
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*E&call_and_return_all_conditional_losses
F__call__
G_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_95", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_95", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_23_input"}}, {"class_name": "GRU", "config": {"name": "gru_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_95", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_23_input"}}, {"class_name": "GRU", "config": {"name": "gru_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	cell


state_spec
regularization_losses
trainable_variables
	variables
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"?
_tf_keras_rnn_layer?
{"class_name": "GRU", "name": "gru_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 1]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 1]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_95", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>m?v@vAvBvCvD"
	optimizer
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
?
layer_metrics
non_trainable_variables
trainable_variables
layer_regularization_losses
 metrics
regularization_losses

!layers
	variables
F__call__
G_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
?

kernel
recurrent_kernel
bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
*M&call_and_return_all_conditional_losses
N__call__"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell_23", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
&layer_metrics
'non_trainable_variables

(states
)metrics
*layer_regularization_losses
regularization_losses
trainable_variables

+layers
	variables
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
!:22dense_95/kernel
:2dense_95/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
,layer_metrics
regularization_losses
-non_trainable_variables
.layer_regularization_losses
/metrics
trainable_variables

0layers
	variables
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	?2gru_23/gru_cell_23/kernel
6:4	2?2#gru_23/gru_cell_23/recurrent_kernel
*:(	?2gru_23/gru_cell_23/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?
2layer_metrics
"regularization_losses
3non_trainable_variables
4layer_regularization_losses
5metrics
#trainable_variables

6layers
$	variables
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
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
'
	0"
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
?
	7total
	8count
9	variables
:	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
&:$22Adam/dense_95/kernel/m
 :2Adam/dense_95/bias/m
1:/	?2 Adam/gru_23/gru_cell_23/kernel/m
;:9	2?2*Adam/gru_23/gru_cell_23/recurrent_kernel/m
/:-	?2Adam/gru_23/gru_cell_23/bias/m
&:$22Adam/dense_95/kernel/v
 :2Adam/dense_95/bias/v
1:/	?2 Adam/gru_23/gru_cell_23/kernel/v
;:9	2?2*Adam/gru_23/gru_cell_23/recurrent_kernel/v
/:-	?2Adam/gru_23/gru_cell_23/bias/v
?2?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509900
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510169
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510334
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509916?
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
/__inference_sequential_95_layer_call_fn_1510349
/__inference_sequential_95_layer_call_fn_1509948
/__inference_sequential_95_layer_call_fn_1509979
/__inference_sequential_95_layer_call_fn_1510364?
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
?2?
"__inference__wrapped_model_1508960?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
gru_23_input?????????
?2?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510682
C__inference_gru_23_layer_call_and_return_conditional_losses_1510523
C__inference_gru_23_layer_call_and_return_conditional_losses_1510863
C__inference_gru_23_layer_call_and_return_conditional_losses_1511022?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_gru_23_layer_call_fn_1510693
(__inference_gru_23_layer_call_fn_1510704
(__inference_gru_23_layer_call_fn_1511044
(__inference_gru_23_layer_call_fn_1511033?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_95_layer_call_and_return_conditional_losses_1511054?
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
*__inference_dense_95_layer_call_fn_1511063?
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
9B7
%__inference_signature_wrapper_1510004gru_23_input
?2?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511103
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511143?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
-__inference_gru_cell_23_layer_call_fn_1511171
-__inference_gru_cell_23_layer_call_fn_1511157?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

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
 ?
"__inference__wrapped_model_1508960w9?6
/?,
*?'
gru_23_input?????????
? "3?0
.
dense_95"?
dense_95??????????
E__inference_dense_95_layer_call_and_return_conditional_losses_1511054\/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? }
*__inference_dense_95_layer_call_fn_1511063O/?,
%?"
 ?
inputs?????????2
? "???????????
C__inference_gru_23_layer_call_and_return_conditional_losses_1510523}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????2
? ?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510682}O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????2
? ?
C__inference_gru_23_layer_call_and_return_conditional_losses_1510863m??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0?????????2
? ?
C__inference_gru_23_layer_call_and_return_conditional_losses_1511022m??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0?????????2
? ?
(__inference_gru_23_layer_call_fn_1510693pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????2?
(__inference_gru_23_layer_call_fn_1510704pO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????2?
(__inference_gru_23_layer_call_fn_1511033`??<
5?2
$?!
inputs?????????

 
p

 
? "??????????2?
(__inference_gru_23_layer_call_fn_1511044`??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????2?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511103?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????2
p
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
H__inference_gru_cell_23_layer_call_and_return_conditional_losses_1511143?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????2
p 
? "R?O
H?E
?
0/0?????????2
$?!
?
0/1/0?????????2
? ?
-__inference_gru_cell_23_layer_call_fn_1511157?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????2
p
? "D?A
?
0?????????2
"?
?
1/0?????????2?
-__inference_gru_cell_23_layer_call_fn_1511171?\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????2
p 
? "D?A
?
0?????????2
"?
?
1/0?????????2?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509900qA?>
7?4
*?'
gru_23_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1509916qA?>
7?4
*?'
gru_23_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510169k;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_95_layer_call_and_return_conditional_losses_1510334k;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_95_layer_call_fn_1509948dA?>
7?4
*?'
gru_23_input?????????
p

 
? "???????????
/__inference_sequential_95_layer_call_fn_1509979dA?>
7?4
*?'
gru_23_input?????????
p 

 
? "???????????
/__inference_sequential_95_layer_call_fn_1510349^;?8
1?.
$?!
inputs?????????
p

 
? "???????????
/__inference_sequential_95_layer_call_fn_1510364^;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_1510004?I?F
? 
??<
:
gru_23_input*?'
gru_23_input?????????"3?0
.
dense_95"?
dense_95?????????