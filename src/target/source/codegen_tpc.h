/*!
 * \file codegen_tpc.h
 * \brief Utility to generate TPC-C code for Habana Gaudi accelerators.
 *
 * TPC (Tensor Processing Core) is the programmable SIMD engine in Habana Gaudi.
 * TPC-C is a C-like language with vector intrinsics (e.g., float64 = 64xf32 SIMD).
 *
 * Key differences from standard C codegen:
 * - Function params use `tensor` type instead of pointers
 * - Buffer access uses TPC intrinsics (v_f32_ld_tnsr_b / v_f32_st_tnsr)
 * - Thread parallelism via index space (get_index_space_offset/size)
 * - SIMD vector types: float64 (64xf32), int64 (64xi32), etc.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_TPC_H_
#define TVM_TARGET_SOURCE_CODEGEN_TPC_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTPC final : public CodeGenC {
 public:
  CodeGenTPC();
  void Init(bool output_ssa);
  std::string Finish();

  /*!
   * \brief Print TPC function signature.
   *
   * TPC kernels use `void main(tensor input0, tensor input1, tensor output, ...)`
   * instead of C-style `void func(float* a, float* b, float* c)`.
   * Handle-type params become `tensor`, scalar params stay as-is.
   */
  void PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                              std::ostream& os) final;

  /*! \brief No prefix needed for TPC (unlike OpenCL's __kernel) */
  void PrintFuncPrefix(std::ostream& os) final;

  /*! \brief No extra attrs needed for now */
  void PrintExtraAttrs(const PrimFunc& f, std::ostream& os) final;

  /*!
   * \brief Inject TPC index space initialization before function body.
   *
   * Generates:
   *   const int5 index_space_start = get_index_space_offset();
   *   const int5 index_space_end = get_index_space_size() + index_space_start;
   */
  void PreFunctionBody(const PrimFunc& f) final;

  /*!
   * \brief Map TVM thread bindings to TPC index space.
   *
   * CUDA: threadIdx.x -> blockIdx.x * blockDim.x + threadIdx.x
   * TPC:  thread tag  -> get_index_space_offset()[dim]
   */
  void BindThreadIndex(const IterVar& iv) final;

  /*!
   * \brief Print TPC-C types.
   *
   * Key mappings:
   *   float32 scalar  -> float
   *   float32x64      -> float64  (TPC 64-element SIMD vector)
   *   int32 scalar    -> int
   *   int32x64        -> int64    (TPC 64-element SIMD int vector)
   *   handle          -> void*    (for non-tensor handles)
   */
  void PrintType(DataType t, std::ostream& os) final;
  // Bring the Type overload into scope (hidden by DataType override)
  using CodeGenSourceBase::PrintType;

  /*!
   * \brief Override BufferLoad for TPC tensor intrinsics.
   *
   * For tensor buffers with vector access:
   *   v_f32_ld_tnsr_b(coords, tensor_name)
   * For scalar/local buffers:
   *   falls back to base class behavior
   */
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;

  /*!
   * \brief Override BufferStore for TPC tensor intrinsics.
   *
   * For tensor buffers with vector store:
   *   v_f32_st_tnsr(coords, tensor_name, value)
   * For scalar/local buffers:
   *   falls back to base class behavior
   */
  void VisitStmt_(const BufferStoreNode* op) final;

  /*!
   * \brief Override vector binary lowering to TPC SIMD intrinsics.
   */
  void PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os) final;

  /*! \brief TPC storage scope handling (no shared memory concept) */
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;

 private:
  /*! \brief Whether scope is part of type (TPC: no) */
  bool IsScopePartOfType() const final { return false; }

  /*!
   * \brief Track which buffer variables are TPC tensor descriptors.
   * These are function parameters with handle type that should use
   * tensor intrinsics instead of pointer arithmetic.
   */
  std::unordered_set<const VarNode*> tensor_buffers_;

  /*!
   * \brief Map from thread tag to TPC index space dimension.
   * e.g., "threadIdx.x" -> 0, "threadIdx.y" -> 1, etc.
   */
  std::unordered_map<std::string, int> thread_tag_to_dim_;

  /*! \brief Whether index space variables have been emitted */
  bool index_space_emitted_{false};
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_TPC_H_
