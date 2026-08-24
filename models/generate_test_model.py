"""Generate a minimal deterministic test model for FoveaEdge.

Creates a tiny Conv -> ReLU -> GAP -> FC model in OpenVINO IR format.
This model exists only to test infrastructure — it is NOT a research model.

Usage:
    python models/generate_test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    """Simple sigmoid for ONNX export."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_test_model(
    output_dir: str | Path = "models/test_model",
    input_channels: int = 3,
    input_height: int = 32,
    input_width: int = 32,
    num_classes: int = 10,
    seed: int = 42,
) -> dict[str, str]:
    """Generate a minimal test model and export to OpenVINO IR.

    Architecture:
        Input (1, 3, 32, 32)
          -> Conv2d(3, 16, 3, padding=1)
          -> ReLU
          -> Global Average Pooling
          -> Fully Connected(16, 10)
          -> Output (1, 10)

    Args:
        output_dir: Directory to save model files.
        input_channels: Number of input channels.
        input_height: Input image height.
        input_width: Input image width.
        num_classes: Number of output classes.
        seed: Random seed for reproducibility.

    Returns:
        Dict with paths to generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # Generate weights
    conv_weights = np.random.randn(16, input_channels, 3, 3).astype(np.float32) * 0.1
    conv_bias = np.zeros(16, dtype=np.float32)
    fc_weights = np.random.randn(num_classes, 16).astype(np.float32) * 0.1
    fc_bias = np.zeros(num_classes, dtype=np.float32)

    # Save as numpy arrays for testing
    np.save(output_dir / "conv_weights.npy", conv_weights)
    np.save(output_dir / "conv_bias.npy", conv_bias)
    np.save(output_dir / "fc_weights.npy", fc_weights)
    np.save(output_dir / "fc_bias.npy", fc_bias)

    # Try to export using ONNX then OpenVINO conversion
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        # Build ONNX model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_channels, input_height, input_width])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_classes])

        # Conv weights initializer
        conv_w_init = numpy_helper.from_array(conv_weights, name="conv_weights")
        conv_b_init = numpy_helper.from_array(conv_bias, name="conv_bias")
        fc_w_init = numpy_helper.from_array(fc_weights.T, name="fc_weights")
        fc_b_init = numpy_helper.from_array(fc_bias, name="fc_bias")
        flatten_shape_init = numpy_helper.from_array(np.array([1, 16], dtype=np.int64), name="flatten_shape")

        # Nodes
        conv_node = helper.make_node("Conv", inputs=["input", "conv_weights", "conv_bias"],
                                     outputs=["conv_out"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"], name="relu")
        # Global Average Pooling
        gap_node = helper.make_node("GlobalAveragePool", inputs=["relu_out"], outputs=["gap_out"], name="gap")
        # Flatten: (1, 16, 1, 1) -> (1, 16)
        flatten_node = helper.make_node("Reshape", inputs=["gap_out", "flatten_shape"],
                                        outputs=["flat_out"], name="flatten")
        # FC: (1, 16) x (16, 10) -> (1, 10)
        fc_node = helper.make_node("MatMul", inputs=["flat_out", "fc_weights"], outputs=["matmul_out"], name="matmul")
        add_node = helper.make_node("Add", inputs=["matmul_out", "fc_bias"], outputs=["output"], name="add_bias")

        graph = helper.make_graph(
            [conv_node, relu_node, gap_node, flatten_node, fc_node, add_node],
            "test_model",
            [X],
            [Y],
            initializer=[conv_w_init, conv_b_init, fc_w_init, fc_b_init, flatten_shape_init],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        # Save ONNX
        onnx_path = str(output_dir / "test_model.onnx")
        onnx.save(model, onnx_path)

        # Convert to OpenVINO IR
        try:
            import openvino as ov
            from openvino.runtime import Core

            core = Core()
            ov_model = core.read_model(onnx_path)
            xml_path = str(output_dir / "test_model.xml")
            ov.save_model(ov_model, xml_path)

            return {
                "xml": xml_path,
                "bin": str(output_dir / "test_model.bin"),
                "onnx": onnx_path,
                "conv_weights": str(output_dir / "conv_weights.npy"),
                "fc_weights": str(output_dir / "fc_weights.npy"),
            }
        except Exception as e2:
            return {
                "onnx": onnx_path,
                "conv_weights": str(output_dir / "conv_weights.npy"),
                "fc_weights": str(output_dir / "fc_weights.npy"),
                "openvino_conversion_error": str(e2),
            }

    except ImportError:
        # No ONNX available — export using OpenVINO ops directly
        try:
            import openvino
            from openvino.runtime import Type
            from openvino.runtime import opset13 as ops

            # Input parameter
            param = ops.parameter(
                shape=[1, input_channels, input_height, input_width], name="input"
            )

            # Conv2d with explicit padding
            conv = ops.convolution(
                param,
                ops.constant(conv_weights),
                strides=[1, 1],
                pads_begin=[1, 1],
                pads_end=[1, 1],
                dilations=[1, 1],
            )

            # Add bias
            conv_bias_4d = ops.reshape(
                ops.constant(conv_bias),
                ops.constant(np.array([1, 16, 1, 1])),
            )
            conv_bias_node = ops.add(conv, conv_bias_4d)

            # ReLU
            relu = ops.relu(conv_bias_node)

            # Global Average Pooling using ops
            input_shape = ops.shape_of(relu)
            h = ops.convert(ops.gather(input_shape, ops.constant(np.array([2])), axis=0), Type.i32)
            w = ops.convert(ops.gather(input_shape, ops.constant(np.array([3])), axis=0), Type.i32)
            h_f = ops.convert(h, Type.f32)
            w_f = ops.convert(w, Type.f32)
            hw = ops.multiply(h_f, w_f)
            hw_4d = ops.reshape(hw, ops.constant(np.array([1, 1, 1, 1])))
            sum_pool = ops.reduce_sum(relu, axes=np.array([2, 3]), keepdims=True)
            gap = ops.divide(sum_pool, hw_4d)

            # Fully Connected
            gap_flat = ops.reshape(gap, ops.constant(np.array([1, 16])))
            fc = ops.matmul(gap_flat, ops.constant(fc_weights.T))
            output = ops.add(fc, ops.constant(fc_bias.reshape(1, num_classes)))
            output.set_friendly_name("output")

            model = ops.model([param], [output])
            model.get_ordered_ops()[-1].set_friendly_name("output")

            # Save as OpenVINO IR
            model_path = str(output_dir / "test_model")
            openvino.save_model(model, model_path + ".xml")

            return {
                "xml": model_path + ".xml",
                "bin": model_path + ".bin",
                "conv_weights": str(output_dir / "conv_weights.npy"),
                "fc_weights": str(output_dir / "fc_weights.npy"),
            }

        except Exception as e:
            print(f"OpenVINO model export failed: {e}", file=sys.stderr)
            print("Weights saved as .npy files for reference.", file=sys.stderr)
            return {
                "conv_weights": str(output_dir / "conv_weights.npy"),
                "fc_weights": str(output_dir / "fc_weights.npy"),
                "error": str(e),
            }


def main() -> int:
    """CLI entry point for test model generation."""
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "models/test_model"
    result = generate_test_model(output_dir)
    print("Generated test model files:")
    for key, path in result.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
