"""
Basic behavioral checks for TPC-C source codegen.

Run:
  /home/henry/miniforge3/bin/python /home/henry/tvm-src/test_tpc_codegen.py
"""

import tvm
from tvm.script import from_source
import textwrap


def _codegen(mod):
    f = tvm.get_global_func("tpcCodegen")
    return f(mod)


def _from_source(src: str):
    return from_source(textwrap.dedent(src))


def _assert_raises_tvmerror(fn, expected_substring: str):
    try:
        fn()
    except tvm.error.TVMError as err:
        msg = str(err)
        assert expected_substring in msg, msg
        return msg
    raise AssertionError("Expected TVMError, but function completed successfully")


def test_scalar_tensor_indexing_fails():
    # Scalar element accesses would otherwise fall back to illegal tensor[] syntax.
    mod = _from_source(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        class Module:
            @T.prim_func(private=True)
            def elemwise_add(A: T.handle("float32"), B: T.handle("float32"), C: T.handle("float32")):
                A_1 = T.decl_buffer((1024,), "float32", data=A)
                B_1 = T.decl_buffer((1024,), "float32", data=B)
                C_1 = T.decl_buffer((1024,), "float32", data=C)
                for i in range(1024):
                    C_1[i] = A_1[i] + B_1[i]
        """
    )
    print("=" * 60)
    print("Test 1: Scalar tensor indexing should fail")
    print("=" * 60)
    print("--- TIR ---")
    print(mod.script())
    print()

    err_msg = _assert_raises_tvmerror(lambda: _codegen(mod), "requires vector dtype float32x64")
    print("--- TPC-C Codegen Error ---")
    print(err_msg)
    print()


def test_vectorized_f32x64_uses_tpc_intrinsics():
    # Use ramp/slice form so BufferLoad/Store have vector dtype float32x64.
    mod = _from_source(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        class Module:
            @T.prim_func(private=True)
            def vec_add(A: T.handle("float32"), B: T.handle("float32"), C: T.handle("float32")):
                A_1 = T.decl_buffer((1024,), "float32", data=A)
                B_1 = T.decl_buffer((1024,), "float32", data=B)
                C_1 = T.decl_buffer((1024,), "float32", data=C)
                for i in range(16):
                    C_1[T.ramp(i * 64, 1, 64)] = A_1[T.ramp(i * 64, 1, 64)] + B_1[T.ramp(i * 64, 1, 64)]
        """
    )
    print("=" * 60)
    print("Test 2: Vectorized float32x64 emits TPC intrinsics")
    print("=" * 60)
    print("--- TIR ---")
    print(mod.script())
    print()

    code = _codegen(mod)
    print("--- TPC-C Output ---")
    print(code)
    print()
    assert "v_f32_ld_tnsr_b(" in code, code
    assert "v_f32_st_tnsr(" in code, code
    assert "v_f32_add_b(" in code, code
    assert "+(1*63)" not in code, code
    # Ensure we are not emitting illegal tensor[] fallback syntax for params.
    assert "A[" not in code, code
    assert "B[" not in code, code
    assert "C[" not in code, code


def test_vectorized_f32x64_mul_uses_tpc_intrinsics():
    mod = _from_source(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        class Module:
            @T.prim_func(private=True)
            def vec_mul(A: T.handle("float32"), B: T.handle("float32"), C: T.handle("float32")):
                A_1 = T.decl_buffer((1024,), "float32", data=A)
                B_1 = T.decl_buffer((1024,), "float32", data=B)
                C_1 = T.decl_buffer((1024,), "float32", data=C)
                for i in range(16):
                    C_1[T.ramp(i * 64, 1, 64)] = A_1[T.ramp(i * 64, 1, 64)] * B_1[T.ramp(i * 64, 1, 64)]
        """
    )
    print("=" * 60)
    print("Test 3: Vectorized float32x64 emits mul intrinsic")
    print("=" * 60)
    print("--- TIR ---")
    print(mod.script())
    print()

    code = _codegen(mod)
    print("--- TPC-C Output ---")
    print(code)
    print()
    assert "v_f32_mul_b(" in code, code
    assert "+(1*63)" not in code, code


def test_unsupported_vector_lane_fails():
    mod = _from_source(
        """
        # from tvm.script import ir as I
        # from tvm.script import tir as T

        @I.ir_module
        class Module:
            @T.prim_func(private=True)
            def vec_add_bad_lane(A: T.handle("float32"), B: T.handle("float32"), C: T.handle("float32")):
                A_1 = T.decl_buffer((1024,), "float32", data=A)
                B_1 = T.decl_buffer((1024,), "float32", data=B)
                C_1 = T.decl_buffer((1024,), "float32", data=C)
                for i in range(32):
                    C_1[T.ramp(i * 32, 1, 32)] = A_1[T.ramp(i * 32, 1, 32)] + B_1[T.ramp(i * 32, 1, 32)]
        """
    )
    print("=" * 60)
    print("Test 4: Unsupported vector lane should fail")
    print("=" * 60)
    print("--- TIR ---")
    print(mod.script())
    print()

    err_msg = _assert_raises_tvmerror(lambda: _codegen(mod), "requires vector dtype float32x64")
    print("--- TPC-C Codegen Error ---")
    print(err_msg)
    print()


if __name__ == "__main__":
    test_scalar_tensor_indexing_fails()
    test_vectorized_f32x64_uses_tpc_intrinsics()
    test_vectorized_f32x64_mul_uses_tpc_intrinsics()
    test_unsupported_vector_lane_fails()
    print("TPC codegen checks passed")
