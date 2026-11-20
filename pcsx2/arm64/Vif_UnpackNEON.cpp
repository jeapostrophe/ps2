// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-FileCopyrightText: 2026 isztld <https://isztld.com/>
// SPDX-License-Identifier: GPL-3.0+
// Transplanted into lrps2 from isztldav/pcsx2 @ c89cb8a (C.19); near-verbatim
// except VifUnpackSSE_Init/Destroy, which use a local host code mapping
// instead of SysMemory.

#include "Vif_UnpackNEON.h"
#include "arm64/aot_arm64.h"
#include <algorithm>

#include "common/Console.h"
#include "common/General.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <sys/mman.h>

namespace a64 = vixl::aarch64;

// =====================================================================================================
//  VifUnpackSSE_Base Section
// =====================================================================================================
VifUnpackNEON_Base::VifUnpackNEON_Base()
	: usn(false)
	, doMask(false)
	, UnpkLoopIteration(0)
	, UnpkNoOfIterations(0)
	, IsAligned(0)
	, dstIndirect(a64::MemOperand(RXARG1))
	, srcIndirect(a64::MemOperand(RXARG2))
	, workReg(a64::q1)
	, destReg(a64::q0)
	, workGprW(a64::w4)
{
}

void VifUnpackNEON_Base::xMovDest() const
{
	if (!IsWriteProtectedOp())
	{
		if (IsUnmaskedOp())
			armAsm->Str(destReg, dstIndirect);
		else
			doMaskWrite(destReg);
	}
}

void VifUnpackNEON_Base::xShiftR(const vixl::aarch64::VRegister& regX, int n) const
{
	if (usn)
		armAsm->Ushr(regX.V4S(), regX.V4S(), n);
	else
		armAsm->Sshr(regX.V4S(), regX.V4S(), n);
}

void VifUnpackNEON_Base::xPMOVXX8(const vixl::aarch64::VRegister& regX) const
{
	// TODO(Stenzek): Check this
	armAsm->Ldr(regX.S(), srcIndirect);

	if (usn)
	{
		armAsm->Ushll(regX.V8H(), regX.V8B(), 0);
		armAsm->Ushll(regX.V4S(), regX.V4H(), 0);
	}
	else
	{
		armAsm->Sshll(regX.V8H(), regX.V8B(), 0);
		armAsm->Sshll(regX.V4S(), regX.V4H(), 0);
	}
}

void VifUnpackNEON_Base::xPMOVXX16(const vixl::aarch64::VRegister& regX) const
{
	armAsm->Ldr(regX.D(), srcIndirect);

	if (usn)
		armAsm->Ushll(regX.V4S(), regX.V4H(), 0);
	else
		armAsm->Sshll(regX.V4S(), regX.V4H(), 0);
}

void VifUnpackNEON_Base::xUPK_S_32() const
{
	if (UnpkLoopIteration == 0)
		armAsm->Ldr(workReg, srcIndirect);

	if (IsInputMasked())
		return;

	switch (UnpkLoopIteration)
	{
		case 0:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 0);
			break;
		case 1:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 1);
			break;
		case 2:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 2);
			break;
		case 3:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 3);
			break;
	}
}

void VifUnpackNEON_Base::xUPK_S_16() const
{
	if (UnpkLoopIteration == 0)
		xPMOVXX16(workReg);

	if (IsInputMasked())
		return;

	switch (UnpkLoopIteration)
	{
		case 0:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 0);
			break;
		case 1:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 1);
			break;
		case 2:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 2);
			break;
		case 3:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 3);
			break;
	}
}

void VifUnpackNEON_Base::xUPK_S_8() const
{
	if (UnpkLoopIteration == 0)
		xPMOVXX8(workReg);

	if (IsInputMasked())
		return;

	switch (UnpkLoopIteration)
	{
		case 0:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 0);
			break;
		case 1:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 1);
			break;
		case 2:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 2);
			break;
		case 3:
			armAsm->Dup(destReg.V4S(), workReg.V4S(), 3);
			break;
	}
}

// The V2 + V3 unpacks have freaky behaviour, the manual claims "indeterminate".
// After testing on the PS2, it's very much determinate in 99% of cases
// and games like Lemmings, And1 Streetball rely on this data to be like this!
// I have commented after each shuffle to show what data is going where - Ref

void VifUnpackNEON_Base::xUPK_V2_32() const
{
	if (UnpkLoopIteration == 0)
	{
		armAsm->Ldr(workReg, srcIndirect);

		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 0); //v1v0v1v0
		if (IsAligned)
			armAsm->Ins(destReg.V4S(), 3, a64::wzr); //zero last word - tested on ps2
	}
	else
	{
		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 1); //v3v2v3v2
		if (IsAligned)
			armAsm->Ins(destReg.V4S(), 3, a64::wzr); //zero last word - tested on ps2
	}
}

void VifUnpackNEON_Base::xUPK_V2_16() const
{
	if (UnpkLoopIteration == 0)
	{
		xPMOVXX16(workReg);

		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 0); //v1v0v1v0
	}
	else
	{
		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 1); //v3v2v3v2
	}
}

void VifUnpackNEON_Base::xUPK_V2_8() const
{
	if (UnpkLoopIteration == 0)
	{
		xPMOVXX8(workReg);

		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 0); //v1v0v1v0
	}
	else
	{
		if (IsInputMasked())
			return;

		armAsm->Dup(destReg.V2D(), workReg.V2D(), 1); //v3v2v3v2
	}
}

void VifUnpackNEON_Base::xUPK_V3_32() const
{
	if (IsInputMasked())
		return;

	armAsm->Ldr(destReg, srcIndirect);
	if (UnpkLoopIteration != IsAligned)
		armAsm->Ins(destReg.V4S(), 3, a64::wzr);
}

void VifUnpackNEON_Base::xUPK_V3_16() const
{
	if (IsInputMasked())
		return;

	xPMOVXX16(destReg);

	//With V3-16, it takes the first vector from the next position as the W vector
	//However - IF the end of this iteration of the unpack falls on a quadword boundary, W becomes 0
	//IsAligned is the position through the current QW in the vif packet
	//Iteration counts where we are in the packet.
	int result = (((UnpkLoopIteration / 4) + 1 + (4 - IsAligned)) & 0x3);

	if ((UnpkLoopIteration & 0x1) == 0 && result == 0)
		armAsm->Ins(destReg.V4S(), 3, a64::wzr); //zero last word on QW boundary if whole 32bit word is used - tested on ps2
}

void VifUnpackNEON_Base::xUPK_V3_8() const
{
	if (IsInputMasked())
		return;

	xPMOVXX8(destReg);
	if (UnpkLoopIteration != IsAligned)
		armAsm->Ins(destReg.V4S(), 3, a64::wzr);
}

void VifUnpackNEON_Base::xUPK_V4_32() const
{
	if (IsInputMasked())
		return;

	armAsm->Ldr(destReg.Q(), a64::MemOperand(srcIndirect));
}

void VifUnpackNEON_Base::xUPK_V4_16() const
{
	if (IsInputMasked())
		return;

	xPMOVXX16(destReg);
}

void VifUnpackNEON_Base::xUPK_V4_8() const
{
	if (IsInputMasked())
		return;

	xPMOVXX8(destReg);
}

void VifUnpackNEON_Base::xUPK_V4_5() const
{
	if (IsInputMasked())
		return;

	armAsm->Ldrh(workGprW, srcIndirect);
	armAsm->Lsl(workGprW, workGprW, 3); // ABG|R5.000
	armAsm->Dup(destReg.V4S(), workGprW); // x|x|x|R
	armAsm->Lsr(workGprW, workGprW, 8); // ABG
	armAsm->Lsl(workGprW, workGprW, 3); // AB|G5.000
	armAsm->Ins(destReg.V4S(), 1, workGprW); // x|x|G|R
	armAsm->Lsr(workGprW, workGprW, 8); // AB
	armAsm->Lsl(workGprW, workGprW, 3); // A|B5.000
	armAsm->Ins(destReg.V4S(), 2, workGprW); // x|B|G|R
	armAsm->Lsr(workGprW, workGprW, 8); // A
	armAsm->Lsl(workGprW, workGprW, 7); // A.0000000
	armAsm->Ins(destReg.V4S(), 3, workGprW); // A|B|G|R
	armAsm->Shl(destReg.V4S(), destReg.V4S(), 24); // can optimize to
	armAsm->Ushr(destReg.V4S(), destReg.V4S(), 24); // single AND...
}

void VifUnpackNEON_Base::xUnpack(int upknum) const
{
	switch (upknum)
	{
		case 0:
			xUPK_S_32();
			break;
		case 1:
			xUPK_S_16();
			break;
		case 2:
			xUPK_S_8();
			break;

		case 4:
			xUPK_V2_32();
			break;
		case 5:
			xUPK_V2_16();
			break;
		case 6:
			xUPK_V2_8();
			break;

		case 8:
			xUPK_V3_32();
			break;
		case 9:
			xUPK_V3_16();
			break;
		case 10:
			xUPK_V3_8();
			break;

		case 12:
			xUPK_V4_32();
			break;
		case 13:
			xUPK_V4_16();
			break;
		case 14:
			xUPK_V4_8();
			break;
		case 15:
			xUPK_V4_5();
			break;

		case 3:
		case 7:
		case 11:
			// TODO: Needs hardware testing.
			// Dynasty Warriors 5: Empire  - Player 2 chose a character menu.
			Console.Warning("Vpu/Vif: Invalid Unpack %d", upknum);
			break;
	}
}

// =====================================================================================================
//  VifUnpackSSE_Simple
// =====================================================================================================

VifUnpackNEON_Simple::VifUnpackNEON_Simple(bool usn_, bool domask_, int curCycle_)
{
	curCycle = curCycle_;
	usn = usn_;
	doMask = domask_;
	IsAligned = true;
}

void VifUnpackNEON_Simple::doMaskWrite(const vixl::aarch64::VRegister& regX) const
{
	armAsm->Ldr(a64::q7, dstIndirect);

	int offX = std::min(curCycle, 3);
	// HH-37: an AOT-shaped kernel reaches nVifMask off its third argument (x2), never at its address
	const bool shaped = aot::shaped() && !aot::redVifHostAddr();
	if (!shaped)
		armMoveAddressToReg(RXVIXLSCRATCH, nVifMask);
	const a64::Register& base = shaped ? RXARG3 : RXVIXLSCRATCH;
	armAsm->Ldr(a64::q29, a64::MemOperand(base, reinterpret_cast<const u8*>(nVifMask[0][offX]) - reinterpret_cast<const u8*>(nVifMask)));
	armAsm->Ldr(a64::q30, a64::MemOperand(base, reinterpret_cast<const u8*>(nVifMask[1][offX]) - reinterpret_cast<const u8*>(nVifMask)));
	armAsm->Ldr(a64::q31, a64::MemOperand(base, reinterpret_cast<const u8*>(nVifMask[2][offX]) - reinterpret_cast<const u8*>(nVifMask)));
	armAsm->And(regX.V16B(), regX.V16B(), a64::q29.V16B());
	armAsm->And(a64::q7.V16B(), a64::q7.V16B(), a64::q30.V16B());
	armAsm->Orr(regX.V16B(), regX.V16B(), a64::q31.V16B());
	armAsm->Orr(regX.V16B(), regX.V16B(), a64::q7.V16B());
	armAsm->Str(regX, dstIndirect);
}

// The kernel nVifUpk[index] at the assembler's pointer, its length in *len: null for an unpack
// type with no data (the C table runs those)
static u8* nVifGenOne(u32 index, u32* len)
{
	const int usn = (index >> 7) & 1, mask = (index >> 6) & 1, type = (index >> 2) & 15, curCycle = index & 3;
	if (nVifT[type] == 0)
		return nullptr;
	VifUnpackNEON_Simple vpugen(!!usn, !!mask, curCycle);
	u8* start = armStartBlock();
	vpugen.xUnpack(type);
	vpugen.xMovDest();
	armAsm->Ret();
	*len = static_cast<u32>(armEndBlock() - start);
	return start;
}

static u32 s_upkLen[(2 * 2 * 16) * 4]; // HH-37 Shape: each kernel's length, for the shape log

// ecx = dest, edx = src
static void nVifGen(int usn, int mask, int curCycle)
{
	for (int i = 0; i < 16; ++i)
	{
		const u32 index = ((usn * 2 * 16 + mask * 16 + i) * 4) + curCycle;
		nVifUpk[index] = (nVifCall)nVifGenOne(index, &s_upkLen[index]);
	}
}

// Code region for the 256 simple unpack kernels (measured ~30KB; 128KB gives
// slack). RWX like the other lrps2 arm64 code caches.
static u8* s_upkCode = nullptr;
static constexpr size_t kUpkCodeSize = 128 * 1024;

void VifUnpackSSE_Init()
{
	if (!s_upkCode)
	{
		s_upkCode = armJitMap(kUpkCodeSize, "VIF unpack"); // the writes are armStartBlock sessions
		if (!s_upkCode)
		{
			Console.Error("arm64 VIF unpack dynarec: no code memory; using the C reference path.");
			return;
		}
	}

	Console.WriteLn("Generating NEON-optimized unpacking functions for VIF interpreters...");

	armSetAsmPtr(s_upkCode, kUpkCodeSize, nullptr);

	for (int a = 0; a < 2; a++)
		for (int b = 0; b < 2; b++)
			for (int c = 0; c < 4; c++)
				nVifGen(a, b, c);
}

// ---- HH-37: the kernels in the AOT form (aot_arm64.cpp; the image's UnitVIFUnpack) ----

#ifdef LRPS2_AOT_FORM
static bool s_upkFromImage; // nVifUpk holds the image's kernels

// Shape, at admission (once the game is known, which the log's first line names): log each kernel
void vifAotLogKernels()
{
	if (!s_upkCode)
		return;
	for (u32 index = 0; index < std::size(nVifUpk); index++)
		if (nVifUpk[index])
		{
			aot::Block b;
			b.unit = aot::UnitVIFUnpack;
			b.pc = index;
			b.codeLen = s_upkLen[index];
			aot::logBlock(b, reinterpret_cast<const u8*>(nVifUpk[index]));
		}
}

// Load and Interp, once the image is open: every kernel from the image. One it lacks stops a Load;
// in Interp it stays null, and the C table runs its unpacks.
void vifAotInstallKernels()
{
	for (u32 index = 0; index < std::size(nVifUpk); index++)
	{
		if (nVifT[(index >> 2) & 15] == 0)
			continue;
		const std::vector<const aot::Block*>* v = aot::imageBlocks(aot::UnitVIFUnpack, index);
		if (!v || v->empty())
		{
			aot::miss(aot::UnitVIFUnpack, index);
			continue;
		}
		nVifUpk[index] = (nVifCall)v->front()->code;
		aot::countInstall(aot::UnitVIFUnpack);
	}
	s_upkFromImage = true;
}

void vifAotDropKernels()
{
	if (s_upkFromImage)
		memset(nVifUpk, 0, sizeof(nVifUpk));
	s_upkFromImage = false;
}

// Emit: re-generate every kernel into buf, checking each against the one the shape log has (a
// kernel the log lacks fails the image). LRPS2_AOT_CORRUPT=vifk:index emits that one as a bare
// return (the red receipt).
void vifAotEmitKernels(u8* buf, size_t cap)
{
	const aot::Block* byIndex[std::size(nVifUpk)] = {};
	for (const aot::Block& r : aot::emitRecords(aot::UnitVIFUnpack))
	{
		if (r.pc >= std::size(nVifUpk) || nVifT[(r.pc >> 2) & 15] == 0)
			aot::emitFailed(r, "it names no kernel");
		else
			byIndex[r.pc] = &r;
	}
	const u32 corrupt = aot::emitCorrupt(aot::UnitVIFUnpack);
	for (u32 index = 0; index < std::size(nVifUpk); index++)
	{
		if (nVifT[(index >> 2) & 15] == 0)
			continue;
		if (!byIndex[index])
		{
			aot::Block missing;
			missing.unit = aot::UnitVIFUnpack;
			missing.pc = index;
			aot::emitFailed(missing, "the shape log has no such kernel");
			continue;
		}
		armSetAsmPtr(buf, cap, nullptr);
		u32 len;
		u8* start;
		if (corrupt && index == corrupt)
		{
			start = armStartBlock();
			armAsm->Ret();
			len = static_cast<u32>(armEndBlock() - start);
			fprintf(stderr, "lrps2 AOT emit: VIF kernel %u emitted as a bare return, on purpose\n", index);
		}
		else
			start = nVifGenOne(index, &len);
		aot::emitted(*byIndex[index], start, len, {});
	}
}
#endif

void VifUnpackSSE_Destroy()
{
	if (s_upkCode)
	{
		armJitUnmap(s_upkCode, kUpkCodeSize, "VIF unpack");
		s_upkCode = nullptr;
	}
	memset(nVifUpk, 0, sizeof(nVifUpk));
}
