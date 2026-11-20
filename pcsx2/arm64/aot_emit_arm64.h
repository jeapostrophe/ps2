// SPDX-FileCopyrightText: lrps2 arm64 port
// SPDX-License-Identifier: LGPL-3.0+
//
// retro-trainer PLAN-handheld HH-37: how AOT-shaped code reaches a host address (aot_arm64.cpp
// says why it must): through the import table whose base the unit keeps in `imports` (the EE: x20;
// the IOP and microVU1: x28), in a form that depends on nothing but the slot and the offset. The
// EE's, the IOP's and the shared emitters' (AsmHelpers.cpp, armAotRoute) shaped forms are these.

#pragma once

#include "arm64/aot_arm64.h"
#include "aarch64/macro-assembler-aarch64.h"

#ifdef LRPS2_AOT_FORM
namespace aot
{
	namespace a64 = vixl::aarch64;

	// rd = rn + off, off below 16 MiB: at most two adds, no scratch
	inline void emitAddOffset(a64::MacroAssembler& m, const a64::Register& rd, const a64::Register& rn, u32 off)
	{
		AOT_CHECK(off < (1u << 24), "an import offset of %#x, beyond 16 MiB", off);
		const a64::Register* src = &rn;
		if (off & 0xfff000u)
		{
			m.Add(rd, *src, off & 0xfff000u);
			src = &rd;
		}
		if ((off & 0xfffu) || src != &rd)
			m.Add(rd, *src, off & 0xfffu);
	}

	// r = the address p: its import's entry, plus p's offset in the region
	inline void emitAddr(a64::MacroAssembler& m, const a64::Register& imports, const a64::Register& r, const void* p)
	{
		u32 off;
		const u32 slot = slotOf(p, &off);
		m.Ldr(r, a64::MemOperand(imports, slot * 8));
		if (off)
			emitAddOffset(m, r, r, off);
	}

	// r = a value import's value
	inline void emitValue(a64::MacroAssembler& m, const a64::Register& imports, const a64::Register& r, const char* name)
	{
		m.Ldr(r, a64::MemOperand(imports, slotOfValue(name) * 8));
	}

	// A call (link) or a jump to fn, which must be an import's start, through scratch
	inline void emitBranch(a64::MacroAssembler& m, const a64::Register& imports, const a64::Register& scratch, const void* fn, bool link)
	{
		u32 off;
		const u32 slot = slotOf(fn, &off);
		AOT_CHECK(off == 0, "a call or jump to %p, inside an import rather than at its start", fn);
		m.Ldr(scratch, a64::MemOperand(imports, slot * 8));
		if (link)
			m.Blr(scratch);
		else
			m.Br(scratch);
	}

	// An operand for `size` bytes at addr: scratch = its import's entry, the offset folded into the
	// operand when it encodes (size-aligned, at most 4095 units), added otherwise
	inline a64::MemOperand absMem(a64::MacroAssembler& m, const a64::Register& imports, const a64::Register& scratch, const void* addr, unsigned size)
	{
		u32 off;
		const u32 slot = slotOf(addr, &off);
		m.Ldr(scratch, a64::MemOperand(imports, slot * 8));
		if (size && (off % size) == 0 && off / size < 4096)
			return a64::MemOperand(scratch, off);
		emitAddOffset(m, scratch, scratch, off);
		return a64::MemOperand(scratch);
	}
} // namespace aot
#endif
