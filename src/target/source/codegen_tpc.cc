/*!
 * \file codegen_tpc.cc
 * \brief TPC-C code generator for Habana Gaudi accelerators.
 *
 * Generates TPC-C kernel source code from TVM TIR.
 * Currently targets Gaudi2 with float32 support.
 */

#include "codegen_tpc.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <vector>

#include "../../tir/transform/ir_utils.h"

namespace tvm {
namespace codegen {

using namespace tir;

CodeGenTPC::CodeGenTPC() {
  // TPC-C doesn't use restrict keyword
  restrict_keyword_ = "";
}

void CodeGenTPC::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  // Reset TPC-specific state
  tensor_buffers_.clear();
  thread_tag_to_dim_.clear();
  index_space_emitted_ = false;
}

std::string CodeGenTPC::Finish() {
  // TPC-C kernel preamble: no special includes needed.
  // TPC-C has built-in types (float64, int5, tensor) and intrinsics.
  return CodeGenC::Finish();
}

// ---------------------------------------------------------------------------
// Function signature: void main(tensor input0, tensor input1, tensor output)
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintFuncPrefix(std::ostream& os) {
  // TPC-C kernels don't need a prefix like OpenCL's "__kernel"
}

void CodeGenTPC::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
  // No extra attributes for TPC kernels (unlike CUDA's __launch_bounds__)
}

void CodeGenTPC::PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                                        std::ostream& os) {
  PrintFuncPrefix(os);

  // TPC kernel entry point is always "void main"
  os << "void main(";

  for (size_t i = 0; i < func->params.size(); ++i) {
    tir::Var v = func->params[i];
    // auto type = GetType(v);
    // LOG(INFO) << type.is_scalar();
    // LOG(INFO) << "type: " << v->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>();
    if (i > 0) {
      os << ", ";
    }

    if (v.dtype().is_handle()) {
      // Handle-type parameters are TPC tensor descriptors
      os << "tensor " << AllocVarID(v.get());
      // Track this buffer as a TPC tensor for intrinsic-based access
      tensor_buffers_.insert(v.get());
    } else {
      // Scalar parameters (int, float, etc.) stay as-is
      PrintType(GetType(v), os);
      os << " " << AllocVarID(v.get());
    }
  }
  os << ")";

  // Register handle data types for buffer access resolution
  for (const auto& param : func->params) {
    if (auto* ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(param.get(), prim->dtype);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PreFunctionBody: inject TPC index space initialization
// ---------------------------------------------------------------------------

void CodeGenTPC::PreFunctionBody(const PrimFunc& f) {
  // Emit TPC index space boilerplate at the start of every kernel
  PrintIndent();
  stream << "const int5 index_space_start = get_index_space_offset();\n";
  PrintIndent();
  stream << "const int5 index_space_end = get_index_space_size() + index_space_start;\n";
  stream << "\n";
  index_space_emitted_ = true;
}

// ---------------------------------------------------------------------------
// Type printing: TPC-C has unique SIMD vector types
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintType(DataType t, std::ostream& os) {
  int lanes = t.lanes();

  if (t.is_handle()) {
    TVM_FFI_ICHECK(t.is_scalar()) << "TPC: do not support vector of handles";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  // TPC-C SIMD vector types for float32:
  //   64 lanes -> float64 (64 x float32, the native SIMD width)
  // For int32:
  //   64 lanes -> int64 (64 x int32)
  if (t.is_float()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "float";
        } else if (lanes == 64) {
          // TPC native SIMD vector: 64 x float32
          os << "float64";
        } else {
          LOG(FATAL) << "TPC: unsupported float32 vector width " << lanes
                     << " (only scalar and 64-lane supported)";
        }
        return;
      default:
        LOG(FATAL) << "TPC: unsupported float bit width " << t.bits()
                   << " (only float32 supported currently)";
    }
  }

  if (t.is_int()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "int";
        } else if (lanes == 64) {
          os << "int64";
        } else {
          LOG(FATAL) << "TPC: unsupported int32 vector width " << lanes;
        }
        return;
      case 16:
        if (lanes == 1) {
          os << "short";
        } else {
          LOG(FATAL) << "TPC: unsupported int16 vector width " << lanes;
        }
        return;
      case 8:
        if (lanes == 1) {
          os << "char";
        } else {
          LOG(FATAL) << "TPC: unsupported int8 vector width " << lanes;
        }
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported int bit width " << t.bits();
    }
  }

  if (t.is_uint()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "unsigned int";
        } else if (lanes == 64) {
          os << "uint64";
        } else {
          LOG(FATAL) << "TPC: unsupported uint32 vector width " << lanes;
        }
        return;
      case 16:
        os << "unsigned short";
        return;
      case 8:
        os << "unsigned char";
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported uint bit width " << t.bits();
    }
  }

  LOG(FATAL) << "TPC: unknown type " << t;
}

// ---------------------------------------------------------------------------
// Thread index binding: map TVM thread tags to TPC index space dimensions
// ---------------------------------------------------------------------------

void CodeGenTPC::BindThreadIndex(const IterVar& iv) {
  TVM_FFI_ICHECK(!var_idmap_.count(iv->var.get()));

  std::string tag = iv->thread_tag;
  int dim = -1;

  // Map TVM thread tags to TPC index space dimensions (0-4)
  // Support both CUDA-style tags and TPC-specific tags
  if (tag == "threadIdx.x" || tag == "tpc.index_space.0") {
    dim = 0;
  } else if (tag == "threadIdx.y" || tag == "tpc.index_space.1") {
    dim = 1;
  } else if (tag == "threadIdx.z" || tag == "tpc.index_space.2") {
    dim = 2;
  } else if (tag == "blockIdx.x" || tag == "tpc.index_space.3") {
    dim = 3;
  } else if (tag == "blockIdx.y" || tag == "tpc.index_space.4") {
    dim = 4;
  } else {
    LOG(FATAL) << "TPC: unknown thread tag " << tag
               << " (supported: threadIdx.x/y/z, blockIdx.x/y, tpc.index_space.0-4)";
  }

  // Bind the variable to index_space_start[dim]
  std::string expr = "index_space_start[" + std::to_string(dim) + "]";
  var_idmap_[iv->var.get()] = expr;
  thread_tag_to_dim_[tag] = dim;
}

// ---------------------------------------------------------------------------
// Buffer access: TPC tensor intrinsics
// ---------------------------------------------------------------------------

void CodeGenTPC::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "TPC: load from non-flat memory not supported";

  DataType value_dtype = op->dtype;
  Var buffer_var = op->buffer->data;
  bool is_tensor_buffer = tensor_buffers_.count(buffer_var.get());

  // Check if this buffer is a TPC tensor (function parameter with handle type)
  if (is_tensor_buffer && value_dtype.lanes() == 64 && value_dtype.is_float() &&
      value_dtype.bits() == 32) {
    // Vector load from TPC tensor: v_f32_ld_tnsr_b(coords, tensor_name)
    // For vectorized accesses, canonical TIR uses ramp(base, 1, lanes).
    // Emit the flat base index instead of the full vector expression.
    std::string idx;
    if (const auto* ramp = op->indices[0].as<RampNode>()) {
      const int64_t* ramp_lanes = as_const_int(ramp->lanes);
      TVM_FFI_ICHECK(ramp_lanes && *ramp_lanes == value_dtype.lanes())
          << "TPC: vector load lanes mismatch between index ramp and value dtype";
      TVM_FFI_ICHECK(is_one(ramp->stride))
          << "TPC: only contiguous vector load ramp with stride=1 is supported";
      idx = PrintExpr(ramp->base);
    } else {
      idx = PrintExpr(op->indices[0]);
    }
    std::string vid = GetVarID(buffer_var.get());

    // Generate: v_f32_ld_tnsr_b(coords, tensor_name)
    os << "v_f32_ld_tnsr_b(" << idx << ", " << vid << ")";
    return;
  }

  // Do not fall back to C pointer indexing for tensor handles.
  if (is_tensor_buffer) {
    LOG(FATAL) << "TPC: tensor load requires vector dtype float32x64, but got " << value_dtype;
  }

  // Fall back to base class for scalar access or local buffers
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTPC::VisitStmt_(const BufferStoreNode* op) {
  TVM_FFI_ICHECK_EQ(op->indices.size(), 1) << "TPC: store to non-flat memory not supported";

  DataType value_dtype = op->value.dtype();
  Var buffer_var = op->buffer->data;
  bool is_tensor_buffer = tensor_buffers_.count(buffer_var.get());

  // Check if this buffer is a TPC tensor for vector store
  if (is_tensor_buffer && value_dtype.lanes() == 64 && value_dtype.is_float() &&
      value_dtype.bits() == 32) {
    // Vector store to TPC tensor: v_f32_st_tnsr(coords, tensor_name, value)
    std::string value = PrintExpr(op->value);
    std::string idx;
    if (const auto* ramp = op->indices[0].as<RampNode>()) {
      const int64_t* ramp_lanes = as_const_int(ramp->lanes);
      TVM_FFI_ICHECK(ramp_lanes && *ramp_lanes == value_dtype.lanes())
          << "TPC: vector store lanes mismatch between index ramp and value dtype";
      TVM_FFI_ICHECK(is_one(ramp->stride))
          << "TPC: only contiguous vector store ramp with stride=1 is supported";
      idx = PrintExpr(ramp->base);
    } else {
      idx = PrintExpr(op->indices[0]);
    }
    std::string vid = GetVarID(buffer_var.get());

    PrintIndent();
    stream << "v_f32_st_tnsr(" << idx << ", " << vid << ", " << value << ");\n";
    return;
  }

  // Do not fall back to C pointer indexing for tensor handles.
  if (is_tensor_buffer) {
    LOG(FATAL) << "TPC: tensor store requires vector dtype float32x64, but got " << value_dtype;
  }

  // Fall back to base class for scalar access or local buffers
  CodeGenC::VisitStmt_(op);
}

void CodeGenTPC::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                  std::ostream& os) {
  if (t.is_float() && t.bits() == 32 && t.lanes() == 64) {
    std::string intrinsic;
    if (op == "+") {
      intrinsic = "v_f32_add_b";
    } else if (op == "-") {
      intrinsic = "v_f32_sub_b";
    } else if (op == "*") {
      intrinsic = "v_f32_mul_b";
    } else if (op == "/") {
      intrinsic = "v_f32_div_b";
    } else {
      LOG(FATAL) << "TPC: unsupported float32x64 vector op \"" << op << "\"";
    }
    os << intrinsic << "(" << PrintExpr(lhs) << ", " << PrintExpr(rhs) << ")";
    return;
  }

  LOG(FATAL) << "TPC: unsupported vector binary op \"" << op << "\" for dtype " << t;
}

// ---------------------------------------------------------------------------
// Storage scope: TPC has no shared memory concept
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintStorageScope(const std::string& scope, std::ostream& os) {
  // TPC has no storage scope qualifiers like CUDA's __shared__ or OpenCL's __local
  // All local variables are in register file or local memory automatically
}

// ---------------------------------------------------------------------------
// Build function registration
// ---------------------------------------------------------------------------

void TPCCodegen(ffi::PackedArgs args, ffi::Any* rv) {
  codegen::CodeGenTPC cg;
  cg.Init(false);

  IRModule mod = args[0].cast<IRModule>();

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenTPC: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    functions.Set(gvar, prim_func);
  }

  // Two-pass: declare all functions first, then define them
  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  *rv = cg.Finish();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tpcCodegen", TPCCodegen);
}

}  // namespace codegen
}  // namespace tvm
