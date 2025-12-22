## dev

### qwen3-0.6b-onnx

1. download onnx model

```sh
huggingface-cli download onnx-community/Qwen3-0.6B-ONNX --local-dir Qwen3-0.6B-ONNX
```

2. profile model

```sh
python scripts/summary_op.py <your onnx path>
```

```
=== Basic Model Information ===
IR version       : 7
Opset versions   : [14, 1]
Number of nodes  : 610
Graph inputs     : 59
Graph outputs    : 57
Initializers     : 312

=== OpType: Shape ===
Total nodes: 2
  Count: 1
    Inputs : (('batch_size', 'sequence_length'),)
    Outputs: ((2,),)
  Count: 1
    Inputs : (('batch_size', 'total_sequence_length'),)
    Outputs: ((2,),)

=== OpType: Constant ===
Total nodes: 7
  Count: 7
    Inputs : ()
    Outputs: ((),)

=== OpType: Gather ===
Total nodes: 3
  Count: 1
    Inputs : ((151936, 1024), ('batch_size', 'sequence_length'))
    Outputs: (('batch_size', 'sequence_length', 1024),)
  Count: 2
    Inputs : ((2,), ())
    Outputs: ((),)

=== OpType: Transpose ===
Total nodes: 1
  Count: 1
    Inputs : ((151936, 1024),)
    Outputs: (None,)

=== OpType: ReduceSum ===
Total nodes: 1
  Count: 1
    Inputs : (('batch_size', 'total_sequence_length'), ())
    Outputs: (('batch_size', 1),)

=== OpType: SimplifiedLayerNormalization ===
Total nodes: 57
  Count: 1
    Inputs : (('batch_size', 'sequence_length', 1024), (1024,))
    Outputs: (('batch_size', 'sequence_length', 1024),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length * num_attention_heads', 128), (128,))
    Outputs: (('batch_size', 'sequence_length * num_attention_heads', 128),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length * num_key_value_heads', 128), (128,))
    Outputs: (('batch_size', 'sequence_length * num_key_value_heads', 128),)

=== OpType: Unsqueeze ===
Total nodes: 1
  Count: 1
    Inputs : ((), ())
    Outputs: ((1,),)

=== OpType: Sub ===
Total nodes: 1
  Count: 1
    Inputs : (('batch_size', 1), ())
    Outputs: (('batch_size', 1),)

=== OpType: Cast ===
Total nodes: 2
  Count: 1
    Inputs : ((),)
    Outputs: ((),)
  Count: 1
    Inputs : (('batch_size', 1),)
    Outputs: (('batch_size', 1),)

=== OpType: MatMul ===
Total nodes: 197
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 1024), (1024, 2048))
    Outputs: (('batch_size', 'sequence_length', 2048),)
  Count: 56
    Inputs : (('batch_size', 'sequence_length', 1024), (1024, 1024))
    Outputs: (('batch_size', 'sequence_length', 1024),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 2048), (2048, 1024))
    Outputs: (('batch_size', 'sequence_length', 1024),)
  Count: 56
    Inputs : (('batch_size', 'sequence_length', 1024), (1024, 3072))
    Outputs: (('batch_size', 'sequence_length', 3072),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 3072), (3072, 1024))
    Outputs: (('batch_size', 'sequence_length', 1024),)
  Count: 1
    Inputs : (('batch_size', 'sequence_length', 1024), None)
    Outputs: (('batch_size', 'sequence_length', 151936),)

=== OpType: Concat ===
Total nodes: 1
  Count: 1
    Inputs : ((), (1,))
    Outputs: ((2,),)

=== OpType: Reshape ===
Total nodes: 113
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 2048), ())
    Outputs: (('batch_size', 'sequence_length * num_attention_heads', 128),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 1024), ())
    Outputs: (('batch_size', 'sequence_length * num_key_value_heads', 128),)
  Count: 1
    Inputs : (('batch_size', 'sequence_length'), (2,))
    Outputs: ((),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length * num_attention_heads', 128), ())
    Outputs: (('batch_size', 'sequence_length', 2048),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length * num_key_value_heads', 128), ())
    Outputs: (('batch_size', 'sequence_length', 1024),)

=== OpType: RotaryEmbedding ===
Total nodes: 56
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 2048), (), (40960, 64), (40960, 64))
    Outputs: (('batch_size', 'sequence_length', 2048),)
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 1024), (), (40960, 64), (40960, 64))
    Outputs: (('batch_size', 'sequence_length', 1024),)

=== OpType: GroupQueryAttention ===
Total nodes: 28
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 2048), ('batch_size', 'sequence_length', 1024), ('batch_size', 'sequence_length', 1024), ('batch_size', 8, 'past_sequence_length', 128), ('batch_size', 8, 'past_sequence_length', 128), ('batch_size', 1), (), None, None)
    Outputs: (('batch_size', 'sequence_length', 2048), ('batch_size', 8, 'total_sequence_length', 128), ('batch_size', 8, 'total_sequence_length', 128))

=== OpType: SkipSimplifiedLayerNormalization ===
Total nodes: 56
  Count: 55
    Inputs : (('batch_size', 'sequence_length', 1024), ('batch_size', 'sequence_length', 1024), (1024,))
    Outputs: (('batch_size', 'sequence_length', 1024), None, None, ('batch_size', 'sequence_length', 1024))
  Count: 1
    Inputs : (('batch_size', 'sequence_length', 1024), ('batch_size', 'sequence_length', 1024), (1024,))
    Outputs: (('batch_size', 'sequence_length', 1024),)

=== OpType: Sigmoid ===
Total nodes: 28
  Count: 28
    Inputs : (('batch_size', 'sequence_length', 3072),)
    Outputs: (('batch_size', 'sequence_length', 3072),)

=== OpType: Mul ===
Total nodes: 56
  Count: 56
    Inputs : (('batch_size', 'sequence_length', 3072), ('batch_size', 'sequence_length', 3072))
    Outputs: (('batch_size', 'sequence_length', 3072),)
```

### code

1. cpp
```sh
pip install pre-commit
pre-commit install
pre-commit run clang-format --all-files
```