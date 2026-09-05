// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-FileCopyrightText: 2026 isztld <https://isztld.com/>
// SPDX-License-Identifier: GPL-3.0

#pragma once

#include "common/Pcsx2Defs.h"
#include "arm64/ArmCompat.h"
#include "common/HashCombine.h"

#include "aarch64/constants-aarch64.h"
#include "aarch64/macro-assembler-aarch64.h"

#include <unordered_map>

#define RWRET vixl::aarch64::w0
#define RXRET vixl::aarch64::x0
#define RQRET vixl::aarch64::q0

#define RWARG1 vixl::aarch64::w0
#define RWARG2 vixl::aarch64::w1
#define RWARG3 vixl::aarch64::w2
#define RWARG4 vixl::aarch64::w3
#define RXARG1 vixl::aarch64::x0
#define RXARG2 vixl::aarch64::x1
#define RXARG3 vixl::aarch64::x2
#define RXARG4 vixl::aarch64::x3

#define RXVIXLSCRATCH vixl::aarch64::x16
#define RWVIXLSCRATCH vixl::aarch64::w16
#define RSCRATCHADDR vixl::aarch64::x17

#define RQSCRATCH vixl::aarch64::q30
#define RDSCRATCH vixl::aarch64::d30
#define RSSCRATCH vixl::aarch64::s30
#define RQSCRATCH2 vixl::aarch64::q31
#define RDSCRATCH2 vixl::aarch64::d31
#define RSSCRATCH2 vixl::aarch64::s31
#define RQSCRATCH3 vixl::aarch64::q29
#define RDSCRATCH3 vixl::aarch64::d29
#define RSSCRATCH3 vixl::aarch64::s29

#define RQSCRATCHI vixl::aarch64::VRegister(30, 128, 16)
#define RQSCRATCHF vixl::aarch64::VRegister(30, 128, 4)
#define RQSCRATCHD vixl::aarch64::VRegister(30, 128, 2)

#define RQSCRATCH2I vixl::aarch64::VRegister(31, 128, 16)
#define RQSCRATCH2F vixl::aarch64::VRegister(31, 128, 4)
#define RQSCRATCH2D vixl::aarch64::VRegister(31, 128, 2)

static inline s64 GetPCDisplacement(const void* current, const void* target)
{
	return static_cast<s64>((reinterpret_cast<ptrdiff_t>(target) - reinterpret_cast<ptrdiff_t>(current)) >> 2);
}

const vixl::aarch64::Register& armWRegister(int n);
const vixl::aarch64::Register& armXRegister(int n);
const vixl::aarch64::VRegister& armSRegister(int n);
const vixl::aarch64::VRegister& armDRegister(int n);
const vixl::aarch64::VRegister& armQRegister(int n);

class ArmConstantPool;

static const u32 SP_SCRATCH_OFFSET = 0;

extern thread_local vixl::aarch64::MacroAssembler* armAsm PCSX2_TLS_INITIAL_EXEC;
extern thread_local u8* armAsmPtr PCSX2_TLS_INITIAL_EXEC;
extern thread_local size_t armAsmCapacity PCSX2_TLS_INITIAL_EXEC;
extern thread_local ArmConstantPool* armConstantPool PCSX2_TLS_INITIAL_EXEC;

static __fi bool armHasBlock()
{
	return (armAsm != nullptr);
}

static __fi u8* armGetCurrentCodePointer()
{
	return static_cast<u8*>(armAsmPtr) + armAsm->GetCursorOffset();
}

__fi static u8* armGetAsmPtr()
{
	return armAsmPtr;
}

void armSetAsmPtr(void* ptr, size_t capacity, ArmConstantPool* pool);
void armAlignAsmPtr();
u8* armStartBlock();
u8* armEndBlock();

void armDisassembleAndDumpCode(const void* ptr, size_t size);
void armEmitJmp(const void* ptr, bool force_inline = false);
void armEmitCall(const void* ptr, bool force_inline = false);
// In-place patch: overwrite the 4-byte B at `code_address` with a branch to
// `target`. Used by the fastmem SIGSEGV backpatch to redirect a faulting Ldr/Str
// to its slow-path thunk (code_address is not tied to the current emit cursor).
// `code_address` must already hold a single 4-byte instruction; target must be
// within +/-128 MB (B imm26 range).
void armEmitJmpPtr(void* code_address, const void* target, bool flush_icache = true);

// Host code memory, the way this host allows it. On Apple silicon a code cache
// must be a MAP_JIT mapping (a plain RWX mmap is refused with EACCES), and every
// write into it must sit between pthread_jit_write_protect_np(0) and (1) on the
// writing thread -- a MAP_JIT page is execute-only until the thread says
// otherwise, and a write outside such a session is a SIGBUS. HostSys::Mmap adds
// MAP_JIT there and HostSys::Begin/EndCodeWrite are the session; on every other
// host the first is a plain mmap and the second is a no-op, so this is not an
// Apple branch, it is the one spelling that is right everywhere. armStartBlock/
// armEndBlock open and close the same session around the shared emitter; the
// scope is for emitters that own their own cursor (the EE and IOP recs).
u8* armJitMap(size_t size);
struct ArmCodeWriteScope
{
	ArmCodeWriteScope();
	~ArmCodeWriteScope();
	ArmCodeWriteScope(const ArmCodeWriteScope&) = delete;
	ArmCodeWriteScope& operator=(const ArmCodeWriteScope&) = delete;
};

// Can vixl-emitted code run on this host? A property of the host and the vixl
// build, answered once per process: emits `add x0, x0, 1; ret` into an
// armJitMap'd page under a write session and calls it. The stub must run from
// host code memory, never from vixl's own staging buffer -- that buffer is
// malloc'd on Darwin (VIXL_CODE_BUFFER_MALLOC) and SetExecutable() is
// VIXL_UNIMPLEMENTED there, so executing it is an instruction-fetch SIGBUS,
// which is how this test used to fail: by taking the process down instead of
// returning false.
bool armVixlSelfTest();

// Zero a page-aligned anonymous mapping in place, the way the block LUTs are
// cleared on every reset and cache wrap. It was madvise(MADV_DONTNEED), which
// discards the pages on Linux so the next read is zero -- and which is only a
// hint on Darwin: measured on macOS 15.3, the bytes are still there afterwards
// (MADV_FREE too). Every reset then left the EE and IOP LUTs full of pointers
// into a cache that had just been rewound, and the first frame after
// retro_reset dispatched into whatever the new blocks had overwritten: no
// crash, no ELF, a black screen for ever. A fresh anonymous MAP_FIXED mapping
// over the same range is zero-filled on every host and is not a hint.
void armZeroMapping(void* base, size_t size);
void armEmitCbnz(const vixl::aarch64::Register& reg, const void* ptr);
void armEmitCondBranch(vixl::aarch64::Condition cond, const void* ptr);
void armMoveAddressToReg(const vixl::aarch64::Register& reg, const void* addr);
void armLoadPtr(const vixl::aarch64::CPURegister& reg, const void* addr);
void armStorePtr(const vixl::aarch64::CPURegister& reg, const void* addr);
// C.72: emit an adrp into `scratch` and return a MemOperand with the page
// offset folded in (or a full materialization fallback). `size` = access size
// in bytes; the folded form is used only when the offset encodes for it.
vixl::aarch64::MemOperand armAbsMemOperand(const vixl::aarch64::Register& scratch, const void* addr, unsigned size);
void armBeginStackFrame(bool save_fpr);
void armEndStackFrame(bool save_fpr);
bool armIsCalleeSavedRegister(int reg);

vixl::aarch64::MemOperand armOffsetMemOperand(const vixl::aarch64::MemOperand& op, s64 offset);
void armGetMemOperandInRegister(const vixl::aarch64::Register& addr_reg,
	const vixl::aarch64::MemOperand& op, s64 extra_offset = 0);

void armLoadConstant128(const vixl::aarch64::VRegister& reg, const void* ptr);

// may clobber RSCRATCH/RSCRATCH2. they shouldn't be inputs.
void armEmitVTBL(const vixl::aarch64::VRegister& dst, const vixl::aarch64::VRegister& src1,
	const vixl::aarch64::VRegister& src2, const vixl::aarch64::VRegister& tbl);

//////////////////////////////////////////////////////////////////////////

class ArmConstantPool
{
public:
	void Init(void* ptr, u32 capacity);
	void Destroy();
	void Reset();

	u8* GetJumpTrampoline(const void* target);
	u8* GetLiteral(u64 value);
	u8* GetLiteral(const u128& value);
	u8* GetLiteral(const u8* bytes, size_t len);

	void EmitLoadLiteral(const vixl::aarch64::CPURegister& reg, const u8* literal) const;

private:
	__fi u32 GetRemainingCapacity() const { return m_capacity - m_used; }

	struct u128_hash
	{
		std::size_t operator()(const u128& v) const
		{
			std::size_t s = 0;
			HashCombine(s, v.lo, v.hi);
			return s;
		}
	};

	std::unordered_map<const void*, u32> m_jump_targets;
	std::unordered_map<u128, u32, u128_hash> m_literals;

	u8* m_base_ptr = nullptr;
	u32 m_capacity = 0;
	u32 m_used = 0;
};
