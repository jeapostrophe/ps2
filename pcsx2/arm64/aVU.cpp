// SPDX-FileCopyrightText: 2026 isztld <https://isztld.com/>
// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// ARM64 microVU recompiler — recompiler shell (Phase 7, task 7.2c).
//
// This is the ARM64 counterpart to pcsx2/x86/microVU.cpp. It holds the
// arch-neutral recompiler housekeeping: the microVU0/1 globals and providers,
// program/block-cache management (mVUinit/mVUreset/mVUclose/mVUclear + the
// mVUcreateProg/mVUcacheProg/mVUcmpProg/mVUsearchProg cache search) and the
// the VU0/1 recompiler providers.
//
// The dispatcher + helper-thunk codegen (mVUdispatcherAB/CD, mVUGenerateWaitMTVU/
// CopyPipelineState/CompareState — task 7.2d) lives further down, and the full
// block compiler (mVUcompile + entry points mVUblockFetch/mVUentryGet/
// mVUcompileJIT) is included from aVU_Compile.inl near the bottom. The microVU rec
// still stays *unselected* on ARM64 (VMManager pins CpuIntVU0/1) until 7.8, so the
// compiled block code is not executed yet — but it is fully emitted for real.

#include "arm64/aVU.h"
#include "arm64/aVU_IR.h"
#include "arm64/aVU_Misc.h" // arch-neutral macro layer (task 7.3)
#include "arm64/aot_arm64.h"

#include "VUmicro.h"
#include "Memory.h"
#include "Dmac.h"
#include "SaveState.h"
#include "Gif_Unit.h" // gifUnit + GIF_TRANS_XGKICK (XGKICK GIF transfer, aVU_Lower.inl)

#include "common/AlignedMalloc.h"
#include "common/General.h"
#include "common/Perf.h"

#include <algorithm>
#include <deque>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cctype>

// microVU rec contexts (x86 defines these in microVU.h; on ARM64 they live here,
// where microRegAlloc is a complete type so microVU's unique_ptr dtor compiles).
/* alignas must be at least the type's own: mVU_Globals inside is alignas(32)
 * for NEON ldp/stp of q-register pairs, so 16 under-aligns and clang rejects it. */
alignas(32) microVU microVU0;
alignas(32) microVU microVU1;

// Whole-program comparison on search (matches the x86 default — off). Cloned here
// rather than pulled from the x86-coupled microVU_Misc.h. doConstProp lives in aVU.h.
static constexpr bool doWholeProgCompare = false;

// recCall function-pointer types (x86: microVU.h). startFunct/startFunctXG point at
// the dispatcher code emitted by 7.2d; the provider Execute paths call through these.
typedef void (*mVUrecCall)(u32, u32);
typedef void (*mVUrecCallXG)(void);

// Program logging is compiled out by default (x86: microVU_Misc.h mVUlogProg).
#define mVUdumpProg(...) if (0) {}

// --- dispatcher + helper-thunk codegen (task 7.2d) ----------------------------
// These emit into the already-open armAsm (mVUgenerateDispatchers opens it).


static void mVUdispatcherAB(microVU& mVU);
static void mVUdispatcherCD(microVU& mVU);
static void mVUGenerateWaitMTVU(microVU& mVU);
static void mVUGenerateCopyPipelineState(microVU& mVU);
static void mVUGenerateCompareState(microVU& mVU);
static void mVUgenerateDispatchers(microVU& mVU);

// The C helper the waitMTVU thunk calls (x86: microVU_Misc.inl mVUwaitMTVU).
static void mVUwaitMTVU();

// The dispatcher (mVUdispatcherAB) calls these; defined later in this TU.
static void* mVUexecuteVU0(u32 startPC, u32 cycles);
static void* mVUexecuteVU1(u32 startPC, u32 cycles);
static void  mVUcleanUpVU0();
static void  mVUcleanUpVU1();

// Block compiler entry points (later Phase 7 tasks):
extern void* mVUblockFetch(microVU& mVU, u32 startPC, uptr pState);
static void* mVUentryGet(microVU& mVU, microBlockManager* block, u32 startPC, uptr pState);

// HH-37: microVU1's AOT form (its section, "microVU1's AOT form", says what it is).
thread_local u64* mvuReadMask;
static __fi bool mvuAot(const microVU& mVU) { return mVU.index == 1 && aot::unitShaped(aot::UnitVU1); }
static microBlock s_vu1AotBlock; // the block being compiled (mVUpBlock): its states, as literals
static a64::Label* vu1AotLit(const microRegInfo& st);
static void vu1AotEmitLink(microVU& mVU, const microRegInfo& st, int pcImm, bool jr);
static void* vu1AotFind(microVU& mVU, u32 pc, const microRegInfo* st);
static void vu1AotReset(microVU& mVU);
static void vu1AotClear();
static void vu1AotForget();

// Forward decls for the private cache helpers (mirrors microVU.cpp ordering).
static void mVUdeleteProg(microVU& mVU, microProgram*& prog);
static microProgram* mVUcreateProg(microVU& mVU, int startPC);
static void mVUcacheProg(microVU& mVU, microProgram& prog);

// Pass-1 analysis (task 7.3) — arch-neutral, operates on the IR, no emitter calls.
// Currently compiled but not yet driven (the per-op pass1 handlers + tables are
// task 7.5+); mVUanalyzeCompileCheck below odr-uses every entry point so the bodies
// are type-checked and codegen'd now.
#include "arm64/aVU_Analyze.inl"

// Pass-2 flag allocators (task 7.4/7.5) — the emit-coupled getFlagReg + the
// Status/Mac/Clip flag normalize/denormalize helpers. These DO make VIXL calls
// (the first emit-backend slice); mVUallocFlagCheck below odr-uses them so the
// bodies are compiled now even though their drivers (mVUsetupFlags/Branch/Lower)
// are later 7.5 slices.
#include "arm64/aVU_Alloc.inl"

// Operand/result clamp helpers (task 7.5) — VIXL min/max range + sign-overflow
// clamps used by every VU FMAC arithmetic op. Emit-coupled; mVUclampCheck below
// odr-uses them so the bodies compile ahead of the Upper/Lower opcode handlers.
#include "arm64/aVU_Clamp.inl"

// Misc emit helpers (task 7.5) — the VU address-transform helper mVUaddrFix.
// Emit-coupled; mVUmiscCheck below odr-uses it so the body compiles ahead of the
// Lower load/store handlers.
#include "arm64/aVU_Misc.inl"

// Upper (FMAC float-vector) opcode handlers (task 7.5a) — the full Upper ISA
// (ADD/SUB/MUL/MADD/MSUB + ACC, MAX/MINI, FTOI/ITOF, ABS, OPMULA/OPMSUB, CLIP,
// NOP) + mVUupdateFlags. Included BEFORE aVU_Tables.inl, which wires these
// handlers into mVU_UPPER_OPCODE / the FD_xx sub-tables.
#include "arm64/aVU_Upper.inl"

// Lower (integer/load-store/EFU/flag/branch) opcode handlers (task 7.5b) — the
// full Lower ISA (setBranchA, IADD…, LQ/SQ…, DIV/EFU…, FCxxx/FMxxx/FSxxx, RINIT…,
// XGKICK, B/BAL/IBxx/JR/JALR) + the XGKICK GIF-transfer helpers. Included BEFORE
// aVU_Tables.inl, which wires these into mVULOWER_OPCODE / the T3_xx sub-tables.
#include "arm64/aVU_Lower.inl"

// Macro mode native COP2 ALU emission (Phase 7.9 / M5) — setupMacroOp/endMacroOp +
// the REC_COP2_mVU0 dispatch (recVUMacroEmitMode0) reusing the Upper/Lower single-op
// emitters above. Included after them (their mVU_* handlers are static to this TU);
// the EE rec (aR5900.cpp) calls recVUMacroEmitMode0() across the TU boundary.
#include "arm64/aVU_Macro.inl"

// Opcode dispatch tables (Tables/Compile big-bang) — mVUopU dispatches the full
// Upper ISA, mVUopL the full Lower ISA. Included BEFORE aVU_Flags.inl because the
// flag read-scan (_mVUflagPass) calls mVUopU/mVUopL.
#include "arm64/aVU_Tables.inl"

// Status/Mac/Clip flag pipeline (task 7.5/7.6) — flag-instance analysis +
// mVUdivSet/mVUsetupFlags emit. The opcode-table-dependent flag read-scan
// (_mVUflagPass/mVUsetFlagInfo) is deferred to the Branch/Compile slice.
// mVUflagCheck below odr-uses the emit helpers so their VIXL bodies compile now.
#include "arm64/aVU_Flags.inl"

// Branch / program-exit emission (task 7.7) — the two program-exit emitters
// (mVUendProgram/mVUDTendProgram) + getLastFlagInst + the E/T-bit & lpState C
// thunks. They make NO opcode-table calls, so they compile standalone ahead of
// the Tables/Compile big-bang. mVUbranchCheck below odr-uses the emitters.
#include "arm64/aVU_Branch.inl"

//------------------------------------------------------------------
// Pass-1 pipeline / cycle / range helpers (task 7.3 part 2)
//------------------------------------------------------------------
// ARM64 clone of the *arch-neutral* helpers in pcsx2/x86/microVU_Compile.inl:
// program-range setup, per-instruction cycle/pipeline accounting, and the
// branch/E-bit/bad-op pass-1 bookkeeping. Like the analysis pass these operate
// purely on the IR (microOp/microIR/microRegInfo) + the program cache and make
// ZERO emitter calls, so they port near-verbatim onto the macro layer already in
// aVU_Misc.h. They are *driven* by the compile driver (mVUcompile) which is the
// emit-coupled task 7.4/7.5; until then mVUcompileHelpersCheck (below) odr-uses
// the non-inline ones so their bodies are compiled now.
//
// The emit-coupled neighbours in microVU_Compile.inl (doUpperOp/doLowerOp/
// doSwapOp/doIbit/mVUexecuteInstruction, mVUtestCycles, mVUDoDBit/mVUDoTBit,
// mvuPreloadRegisters, handleBadOp, mVUdebugPrintBlocks, and mVUcompile itself)
// are deliberately NOT ported here — they make VIXL/regAlloc calls and come over
// with the 7.4 compile driver.

// Used by mVUsetupRange — re-cache the program if the guest micro memory changed.
__fi void mVUcheckIsSame(mV)
{
	if (mVU.prog.isSame == -1)
	{
		mVU.prog.isSame = !memcmp((u8*)mVUcurProg.data, mVU.regs().Micro, mVU.microMemSize);
	}
	if (mVU.prog.isSame == 0)
	{
		mVUcacheProg(mVU, *mVU.prog.cur);
		mVU.prog.isSame = 1;
	}
}

// Sets up microProgram PC ranges based on whats been recompiled
void mVUsetupRange(microVU& mVU, s32 pc, bool isStartPC)
{
	std::deque<microRange>*& ranges = mVUcurProg.ranges;
	if (pc > (s64)mVU.microMemSize)
	{
		Console.Error("microVU%d: PC outside of VU memory PC=0x%04x", mVU.index, pc);
		pxFail("microVU: PC out of VU memory");
	}

	// The PC handling will prewrap the PC so we need to set the end PC to the end of the micro memory, but only if it wraps, no more.
	const s32 cur_pc = (!isStartPC && mVUrange.start > pc && pc == 0) ? mVU.microMemSize : pc;

	if (isStartPC) // Check if startPC is already within a block we've recompiled
	{
		std::deque<microRange>::const_iterator it(ranges->begin());
		for (; it != ranges->end(); ++it)
		{
			if ((cur_pc >= it[0].start) && (cur_pc <= it[0].end))
			{
				if (it[0].start != it[0].end)
				{
					microRange mRange = {it[0].start, it[0].end};
					ranges->erase(it);
					ranges->push_front(mRange);
					return; // new start PC is inside the range of another range
				}
			}
		}
	}
	else if (mVUrange.end >= cur_pc)
	{
		// existing range covers more area than current PC so no need to process it
		return;
	}

	if (doWholeProgCompare)
		mVUcheckIsSame(mVU);

	if (isStartPC)
	{
		microRange mRange = {cur_pc, -1};
		ranges->push_front(mRange);
		return;
	}

	if (mVUrange.start <= cur_pc)
	{
		mVUrange.end = cur_pc;
		s32 rStart = mVUrange.start;
		s32 rEnd = mVUrange.end;
		for (auto it = ranges->begin() + 1; it != ranges->end();)
		{
			if (((it->start >= rStart) && (it->start <= rEnd)) || ((it->end >= rStart) && (it->end <= rEnd))) // Starts after this prog but starts before the end of current prog
			{
				mVUrange.start = rStart = std::min(it->start, rStart); // Choose the earlier start
				mVUrange.end = rEnd = std::max(it->end, rEnd);
				it = ranges->erase(it);
			}
			else
				it++;
		}
	}
	else
	{
		mVUrange.end = mVU.microMemSize;
		DevCon.WriteLn(Color_Green, "microVU%d: Prog Range Wrap [%04x] [%04x] PC %x", mVU.index, mVUrange.start, mVUrange.end, cur_pc);
		microRange mRange = {0, cur_pc };
		ranges->push_front(mRange);
	}

	if(!doWholeProgCompare)
		mVUcacheProg(mVU, *mVU.prog.cur);
}

//------------------------------------------------------------------
// Warnings / Errors / Illegal Instructions
//------------------------------------------------------------------

// If 1st op in block is a bad opcode, then don't compile rest of block (Dawn of Mana Level 2)
__fi void mVUcheckBadOp(mV)
{

	// The BIOS writes upper and lower NOPs in reversed slots (bug)
	//So to prevent spamming we ignore these, however its possible the real VU will bomb out if
	//this happens, so we will bomb out without warning.
	if (mVUinfo.isBadOp && mVU.code != 0x8000033c)
	{

		mVUinfo.isEOB = true;
		DevCon.Warning("microVU Warning: Block contains an illegal opcode...");
	}
}

__ri void branchWarning(mV)
{
	incPC(-2);
	if (mVUup.eBit && mVUbranch)
	{
		incPC(2);
		DevCon.Warning("microVU%d Warning: Branch in E-bit delay slot! [%04x]", mVU.index, xPC);
		mVUlow.isNOP = true;
	}
	else
		incPC(2);

	if (mVUinfo.isBdelay && !mVUlow.evilBranch) // Check if VI Reg Written to on Branch Delay Slot Instruction
	{
		if (mVUlow.VI_write.reg && mVUlow.VI_write.used && !mVUlow.readFlags)
		{
			mVUlow.backupVI = true;
			mVUregs.viBackUp = mVUlow.VI_write.reg;
		}
	}
}

__fi void eBitPass1(mV, int& branch)
{
	if (mVUregs.blockType != 1)
	{
		branch = 1;
		mVUup.eBit = true;
	}
}

__ri void eBitWarning(mV)
{
	if (mVUpBlock->pState.blockType == 1)
		Console.Error("microVU%d Warning: Branch, E-bit, Branch! [%04x]",  mVU.index, xPC);
	if (mVUpBlock->pState.blockType == 2)
		DevCon.Warning("microVU%d Warning: Branch, Branch, Branch! [%04x]", mVU.index, xPC);
	incPC(2);
	if (curI & _Ebit_)
	{
		DevCon.Warning("microVU%d: E-bit in Branch delay slot! [%04x]", mVU.index, xPC);
		mVUregs.blockType = 1;
	}
	incPC(-2);
}

//------------------------------------------------------------------
// Cycles / Pipeline State
//------------------------------------------------------------------
__fi u8 optimizeReg(u8 rState) { return (rState == 1) ? 0 : rState; }
__fi u8 calcCycles(u8 reg, u8 x) { return ((reg > x) ? (reg - x) : 0); }
__fi u8 tCycles(u8 dest, u8 src) { return std::max(dest, src); }
__fi void incP(mV) { mVU.p ^= 1; }
__fi void incQ(mV) { mVU.q ^= 1; }

// Optimizes the End Pipeline State Removing Unnecessary Info
// If the cycles remaining is just '1', we don't have to transfer it to the next block
// because mVU automatically decrements this number at the start of its loop,
// so essentially '1' will be the same as '0'...
void mVUoptimizePipeState(mV)
{
	for (int i = 0; i < 32; i++)
	{
		mVUregs.VF[i].x = optimizeReg(mVUregs.VF[i].x);
		mVUregs.VF[i].y = optimizeReg(mVUregs.VF[i].y);
		mVUregs.VF[i].z = optimizeReg(mVUregs.VF[i].z);
		mVUregs.VF[i].w = optimizeReg(mVUregs.VF[i].w);
	}
	for (int i = 0; i < 16; i++)
	{
		mVUregs.VI[i] = optimizeReg(mVUregs.VI[i]);
	}
	if (mVUregs.q) { mVUregs.q = optimizeReg(mVUregs.q); if (!mVUregs.q) { incQ(mVU); } }
	if (mVUregs.p) { mVUregs.p = optimizeReg(mVUregs.p); if (!mVUregs.p) { incP(mVU); } }
	mVUregs.r = 0; // There are no stalls on the R-reg, so its Safe to discard info
}

void mVUincCycles(mV, int x)
{
	mVUcycles += x;
	// VF[0] is a constant value (0.0 0.0 0.0 1.0)
	for (int z = 31; z > 0; z--)
	{
		mVUregs.VF[z].x = calcCycles(mVUregs.VF[z].x, x);
		mVUregs.VF[z].y = calcCycles(mVUregs.VF[z].y, x);
		mVUregs.VF[z].z = calcCycles(mVUregs.VF[z].z, x);
		mVUregs.VF[z].w = calcCycles(mVUregs.VF[z].w, x);
	}
	// VI[0] is a constant value (0)
	for (int z = 15; z > 0; z--)
	{
		mVUregs.VI[z] = calcCycles(mVUregs.VI[z], x);
	}
	if (mVUregs.q)
	{
		if (mVUregs.q > 4)
		{
			mVUregs.q = calcCycles(mVUregs.q, x);
			if (mVUregs.q <= 4)
			{
				mVUinfo.doDivFlag = 1;
			}
		}
		else
		{
			mVUregs.q = calcCycles(mVUregs.q, x);
		}
		if (!mVUregs.q)
			incQ(mVU);
	}
	if (mVUregs.p)
	{
		mVUregs.p = calcCycles(mVUregs.p, x);
		if (!mVUregs.p || mVUregsTemp.p)
			incP(mVU);
	}
	if (mVUregs.xgkick)
	{
		mVUregs.xgkick = calcCycles(mVUregs.xgkick, x);
		if (!mVUregs.xgkick)
		{
			mVUinfo.doXGKICK = 1;
			mVUinfo.XGKICKPC = xPC;
		}
	}
	mVUregs.r = calcCycles(mVUregs.r, x);
}

// Helps check if upper/lower ops read/write to same regs...
void cmpVFregs(microVFreg& VFreg1, microVFreg& VFreg2, bool& xVar)
{
	if (VFreg1.reg == VFreg2.reg)
	{
		if ((VFreg1.x && VFreg2.x) || (VFreg1.y && VFreg2.y)
		 || (VFreg1.z && VFreg2.z) || (VFreg1.w && VFreg2.w))
		{
			xVar = 1;
		}
	}
}

void mVUsetCycles(mV)
{
	mVUincCycles(mVU, mVUstall);
	// If upper Op && lower Op write to same VF reg:
	if ((mVUregsTemp.VFreg[0] == mVUregsTemp.VFreg[1]) && mVUregsTemp.VFreg[0])
	{
		if (mVUregsTemp.r || mVUregsTemp.VI)
			mVUlow.noWriteVF = true;
		else
			mVUlow.isNOP = true; // If lower Op doesn't modify anything else, then make it a NOP
	}
	// If lower op reads a VF reg that upper Op writes to:
	if ((mVUlow.VF_read[0].reg || mVUlow.VF_read[1].reg) && mVUup.VF_write.reg)
	{
		cmpVFregs(mVUup.VF_write, mVUlow.VF_read[0], mVUinfo.swapOps);
		cmpVFregs(mVUup.VF_write, mVUlow.VF_read[1], mVUinfo.swapOps);
	}
	// If above case is true, and upper op reads a VF reg that lower Op Writes to:
	if (mVUinfo.swapOps && ((mVUup.VF_read[0].reg || mVUup.VF_read[1].reg) && mVUlow.VF_write.reg))
	{
		cmpVFregs(mVUlow.VF_write, mVUup.VF_read[0], mVUinfo.backupVF);
		cmpVFregs(mVUlow.VF_write, mVUup.VF_read[1], mVUinfo.backupVF);
	}

	mVUregs.VF[mVUregsTemp.VFreg[0]].x = tCycles(mVUregs.VF[mVUregsTemp.VFreg[0]].x, mVUregsTemp.VF[0].x);
	mVUregs.VF[mVUregsTemp.VFreg[0]].y = tCycles(mVUregs.VF[mVUregsTemp.VFreg[0]].y, mVUregsTemp.VF[0].y);
	mVUregs.VF[mVUregsTemp.VFreg[0]].z = tCycles(mVUregs.VF[mVUregsTemp.VFreg[0]].z, mVUregsTemp.VF[0].z);
	mVUregs.VF[mVUregsTemp.VFreg[0]].w = tCycles(mVUregs.VF[mVUregsTemp.VFreg[0]].w, mVUregsTemp.VF[0].w);

	mVUregs.VF[mVUregsTemp.VFreg[1]].x = tCycles(mVUregs.VF[mVUregsTemp.VFreg[1]].x, mVUregsTemp.VF[1].x);
	mVUregs.VF[mVUregsTemp.VFreg[1]].y = tCycles(mVUregs.VF[mVUregsTemp.VFreg[1]].y, mVUregsTemp.VF[1].y);
	mVUregs.VF[mVUregsTemp.VFreg[1]].z = tCycles(mVUregs.VF[mVUregsTemp.VFreg[1]].z, mVUregsTemp.VF[1].z);
	mVUregs.VF[mVUregsTemp.VFreg[1]].w = tCycles(mVUregs.VF[mVUregsTemp.VFreg[1]].w, mVUregsTemp.VF[1].w);

	mVUregs.VI[mVUregsTemp.VIreg] = tCycles(mVUregs.VI[mVUregsTemp.VIreg], mVUregsTemp.VI);
	mVUregs.q                     = tCycles(mVUregs.q,                     mVUregsTemp.q);
	mVUregs.p                     = tCycles(mVUregs.p,                     mVUregsTemp.p);
	mVUregs.r                     = tCycles(mVUregs.r,                     mVUregsTemp.r);
	mVUregs.xgkick                = tCycles(mVUregs.xgkick,                mVUregsTemp.xgkick);
}

//------------------------------------------------------------------
// First-pass initialization (emitter-free)
//------------------------------------------------------------------
// These set up the IR state at the start of mVUcompile's first pass. They make
// no emitter calls (memset/memcpy + block-manager add only), so they port
// verbatim from microVU_Compile.inl onto the cloned IR. mVUinitFirstPass takes
// the start-of-block host code pointer (x86 `x86Ptr` -> ARM64
// armGetCurrentCodePointer()) and stashes it in mVUblock.codeStart (renamed from
// x86ptrStart per the 7.2a struct rename). The emit-coupled compile driver that
// drives these (mVUcompile + pass-2) is the rest of task 7.4.

// This gets run at the start of every loop of mVU's first pass
__fi void startLoop(mV)
{
	if (curI & _Mbit_ && isVU0)
		DevCon.WriteLn(Color_Green, "microVU%d: M-bit set! PC = %x", getIndex, xPC);
	if (curI & _Dbit_)
		DevCon.WriteLn(Color_Green, "microVU%d: D-bit set! PC = %x", getIndex, xPC);
	if (curI & _Tbit_)
		DevCon.WriteLn(Color_Green, "microVU%d: T-bit set! PC = %x", getIndex, xPC);
	std::memset(&mVUinfo, 0, sizeof(mVUinfo));
	std::memset(&mVUregsTemp, 0, sizeof(mVUregsTemp));
}

// Initialize VI Constants (vi15 propagates through blocks)
__fi void mVUinitConstValues(microVU& mVU)
{
	for (int i = 0; i < 16; i++)
	{
		mVUconstReg[i].isValid  = 0;
		mVUconstReg[i].regValue = 0;
	}
	mVUconstReg[15].isValid = mVUregs.vi15v;
	mVUconstReg[15].regValue = mVUregs.vi15v ? mVUregs.vi15 : 0;
}

// Initialize Variables
__fi void mVUinitFirstPass(microVU& mVU, uptr pState, u8* thisPtr)
{
	mVUstartPC = iPC; // Block Start PC
	mVUbranch  = 0;   // Branch Type
	mVUcount   = 0;   // Number of instructions ran
	mVUcycles  = 0;   // Skips "M" phase, and starts counting cycles at "T" stage
	mVU.p      = 0;   // All blocks start at p index #0
	mVU.q      = 0;   // All blocks start at q index #0
	if ((uptr)&mVUregs != pState) // Loads up Pipeline State Info
	{
		memcpy((u8*)&mVUregs, (u8*)pState, sizeof(microRegInfo));
	}
	// (HH-37: never in the AOT form, where a compile touches no state the code runs on)
	if (((uptr)&mVU.prog.lpState != pState) && !mvuAot(mVU))
	{
		memcpy((u8*)&mVU.prog.lpState, (u8*)pState, sizeof(microRegInfo));
	}
	mVUblock.codeStart = thisPtr;
	if (mvuAot(mVU)) // HH-37: no block manager; the block is its key's, found by vu1AotFind
	{
		std::memcpy(&s_vu1AotBlock, &mVUblock, sizeof(microBlock));
		mVUpBlock = &s_vu1AotBlock;
	}
	else
		mVUpBlock = mVUblocks[mVUstartPC / 2]->add(mVU, &mVUblock); // Add this block to block manager
	mVUregs.needExactMatch = (mVUpBlock->pState.blockType) ? 7 : 0; // ToDo: Fix 1-Op block flag linking (MGS2:Demo/Sly Cooper)
	mVUregs.blockType = 0;
	mVUregs.viBackUp  = 0;
	mVUregs.flagInfo  = 0;
	mVUsFlagHack = CHECK_VU_FLAGHACK;
	mVUinitConstValues(mVU);
}

//------------------------------------------------------------------
// Micro VU - Main Functions
//------------------------------------------------------------------

// Each VU's code cache (mVUinit leases it, mVUclose hands it back).
static constexpr size_t kMvuRecSize = 0x4000000;

bool mVUavailable(int vuIndex)
{
	// HH-37: an image's VU1 runs from its stubs and blocks, with no code cache
	if (vuIndex && aot::imageOnly())
		return aot::unitShaped(aot::UnitVU1) && aot::admitted() && aot::imageBlocks(aot::UnitVU1, aot::VU1_STUB_PC);
	return (vuIndex ? microVU1 : microVU0).cache != nullptr;
}

// Only run this once per VU! ;)
void mVUinit(microVU& mVU, uint vuIndex)
{
	std::memset(&mVU.prog, 0, sizeof(mVU.prog));
	mVU.glob = mVUglob; // C.75: x27-reachable copies of the emit constants
	std::memcpy(mVU.sse4Min, sse4_minvals, sizeof(mVU.sse4Min));
	std::memcpy(mVU.sse4Max, sse4_maxvals, sizeof(mVU.sse4Max));

	mVU.index        =  vuIndex;
	mVU.cop2         =  0;
	mVU.vuMemSize    = (mVU.index ? 0x4000 : 0x1000);
	mVU.microMemSize = (mVU.index ? 0x4000 : 0x1000);
	mVU.progSize     = (mVU.index ? 0x4000 : 0x1000) / 4;
	mVU.mscalMemo    = (microMscalMemoA*)calloc(mVU.progSize / 2, sizeof(microMscalMemoA));
	mVU.progMemMask  =  mVU.progSize-1;
	// The libretro fork has no fixed HostMemoryMap code arena; give each VU a
	// dedicated host code cache (same 64 MB budget as upstream's
	// mVU0/1recSize), leased per VM and handed back by mVUclose -- a cache
	// kept for the process would hold 128 MB of a frontend's code memory
	// that the next core may need. The compile is already a session
	// (armStartBlock/armEndBlock, see compareState in aVU.h). With no code
	// memory to be had the cache stays null and the VU runs on its
	// interpreter (mVUavailable, VMManager::UpdateCPUImplementations).
	if (!mVU.cache)
		mVU.cache = armJitMap(kMvuRecSize, vuIndex ? "microVU1" : "microVU0");
	mVU.prog.codeReserveEnd = mVU.cache ? mVU.cache + kMvuRecSize : nullptr;
	mVU.prog.codeEnd = mVU.cache ? mVU.prog.codeReserveEnd - (mVUcacheSafeZone * _1mb) : nullptr;

	mVU.regAlloc.reset(new microRegAlloc(mVU.index));
}

// Resets Rec Data
void mVUreset(microVU& mVU, bool resetReserve)
{
	if (THREAD_VU1)
	{
		DevCon.Warning("mVU Reset");
		// If MTVU is toggled on during gameplay we need to flush the running VU1 program, else it gets in a mess
		if (VU0.VI[REG_VPU_STAT].UL & 0x100)
		{
			CpuVU1->Execute(vu1RunCycles);
		}
		VU0.VI[REG_VPU_STAT].UL &= ~0x100;
	}

	// Point the VIXL emitter at mVU.cache and emit the dispatchers + helper thunks
	// at the start of the cache (x86: x86Ptr = (u8*)(mVU.cache) + the five generate calls).
	// Also sets mVU.prog.codeStart / codePtr just past the emitted dispatchers.
	// HH-37: in the AOT form, the stubs in their shaped form (or the image's), and no blocks.
	if (mvuAot(mVU))
		vu1AotReset(mVU);
	else
		mVUgenerateDispatchers(mVU);

	mVU.regs().nextBlockCycles = 0;
	memset(&mVU.prog.lpState, 0, sizeof(mVU.prog.lpState));
	mVU.profiler.Reset(mVU.index);

	// Program Variables
	mVU.prog.cleared  =  1;
	mVU.prog.isSame   = -1;
	mVU.prog.cur      = NULL;
	mVU.prog.total    =  0;
	mVU.prog.curFrame =  0;

	// (codeStart / codePtr were set by mVUgenerateDispatchers, just past the
	// emitted dispatchers — the x86 rec does this with x86Ptr.)

	for (u32 i = 0; i < (mVU.progSize / 2); i++)
	{
		if (!mVU.prog.prog[i])
		{
			mVU.prog.prog[i] = new std::deque<microProgram*>();
			continue;
		}
		for (auto it = mVU.prog.prog[i]->begin(); it != mVU.prog.prog[i]->end(); ++it)
		{
			mVUdeleteProg(mVU, it[0]);
		}
		mVU.prog.prog[i]->clear();
		mVU.prog.quick[i].block = NULL;
		mVU.prog.quick[i].prog = NULL;
	}
}

// Free Allocated Resources
void mVUclose(microVU& mVU)
{
	// Delete Programs and Block Managers
	for (u32 i = 0; i < (mVU.progSize / 2); i++)
	{
		if (!mVU.prog.prog[i])
			continue;
		for (auto it = mVU.prog.prog[i]->begin(); it != mVU.prog.prog[i]->end(); ++it)
		{
			mVUdeleteProg(mVU, it[0]);
		}
		safe_delete(mVU.prog.prog[i]);
	}
	free(mVU.mscalMemo);
	mVU.mscalMemo = nullptr;
	if (mVU.index)
		vu1AotForget(); // HH-37: its blocks' code goes with the cache (or the image)

	// The cache goes with the VM that leased it (mVUinit).
	if (mVU.cache)
	{
		armJitUnmap(mVU.cache, kMvuRecSize, mVU.index ? "microVU1" : "microVU0");
		mVU.cache = nullptr;
		mVU.prog.codeStart = mVU.prog.codePtr = nullptr;
		mVU.prog.codeReserveEnd = mVU.prog.codeEnd = nullptr;
		// And every pointer mVUreset emitted into it: kept, each would point
		// into memory the frontend may already have lent to something else.
		mVU.startFunct = mVU.exitFunct = nullptr;
		mVU.startFunctXG = mVU.exitFunctXG = nullptr;
		mVU.compareStateF = mVU.waitMTVU = mVU.copyPLState = nullptr;
		mVU.resumePtrXG = nullptr;
		mVU.exactMulStub = nullptr;
	}
}

// Clears Block Data in specified range
void mVUclear(microVU& mVU, u32 addr, u32 size)
{
	if (mvuAot(mVU))
		vu1AotClear(); // HH-37: every block's words are checked again before it next runs
	if (!mVU.prog.cleared)
	{
		mVU.prog.cleared = 1; // Next execution searches/creates a new microprogram
		std::memset(&mVU.prog.lpState, 0, sizeof(mVU.prog.lpState)); // Clear pipeline state
		for (u32 i = 0; i < (mVU.progSize / 2); i++)
		{
			mVU.prog.quick[i].block = NULL; // Clear current quick-reference block
			mVU.prog.quick[i].prog = NULL; // Clear current quick-reference prog
		}
	}
}

//------------------------------------------------------------------
// Micro VU - Private Functions
//------------------------------------------------------------------

// Deletes a program
static void mVUdeleteProg(microVU& mVU, microProgram*& prog)
{
	for (u32 i = 0; i < (mVU.progSize / 2); i++)
	{
		safe_delete(prog->block[i]);
	}
	safe_delete(prog->ranges);
	safe_aligned_free(prog);
}

// Creates a new Micro Program
static microProgram* mVUcreateProg(microVU& mVU, int startPC)
{
	microProgram* prog = (microProgram*)_aligned_malloc(sizeof(microProgram), 64);
	memset(prog, 0, sizeof(microProgram));
	prog->idx = mVU.prog.total++;
	prog->ranges = new std::deque<microRange>();
	prog->startPC = startPC;
	if (doWholeProgCompare)
		mVUcacheProg(mVU, *prog); // Cache Micro Program
	double cacheSize = (double)((uptr)mVU.prog.codeEnd - (uptr)mVU.prog.codeStart);
	double cacheUsed = ((double)((uptr)mVU.prog.codePtr - (uptr)mVU.prog.codeStart)) / (double)_1mb;
	double cachePerc = ((double)((uptr)mVU.prog.codePtr - (uptr)mVU.prog.codeStart)) / cacheSize * 100;
	ConsoleColors c = mVU.index ? Color_Orange : Color_Magenta;
	DevCon.WriteLn(c, "microVU%d: Cached Prog = [%03d] [PC=%04x] [List=%02d] (Cache=%3.3f%%) [%3.1fmb]",
		mVU.index, prog->idx, startPC * 8, mVU.prog.prog[startPC]->size() + 1, cachePerc, cacheUsed);
	return prog;
}

// Caches Micro Program
static void mVUcacheProg(microVU& mVU, microProgram& prog)
{
	if (!doWholeProgCompare)
	{
		const microRange& range = (*mVU.prog.cur->ranges)[0];
		auto cmpOffset = [&](void* x) { return (u8*)x + range.start; };
		memcpy(cmpOffset(prog.data), cmpOffset(mVU.regs().Micro), (range.end - range.start));
	}
	else
	{
		if (!mVU.index)
			memcpy(prog.data, mVU.regs().Micro, 0x1000);
		else
			memcpy(prog.data, mVU.regs().Micro, 0x4000);
	}
	mVUdumpProg(mVU, prog);
}

// Generate Hash for partial program based on compiled ranges...
static u64 mVUrangesHash(microVU& mVU, microProgram& prog)
{
	union
	{
		u64 v64;
		u32 v32[2];
	} hash = {0};

	std::deque<microRange>::const_iterator it(prog.ranges->begin());
	for (; it != prog.ranges->end(); ++it)
	{
		if ((it[0].start < 0) || (it[0].end < 0))
		{
			DevCon.Error("microVU%d: Negative Range![%d][%d]", mVU.index, it[0].start, it[0].end);
		}
		for (int i = it[0].start / 4; i < it[0].end / 4; i++)
		{
			hash.v32[0] -= prog.data[i];
			hash.v32[1] ^= prog.data[i];
		}
	}
	return hash.v64;
}

// Prints the ratio of unique programs to total programs
[[maybe_unused]] static void mVUprintUniqueRatio(microVU& mVU)
{
	std::vector<u64> v;
	for (u32 pc = 0; pc < mProgSize / 2; pc++)
	{
		microProgramList* list = mVU.prog.prog[pc];
		if (!list)
			continue;
		for (auto it = list->begin(); it != list->end(); ++it)
		{
			v.push_back(mVUrangesHash(mVU, *it[0]));
		}
	}
	u32 total = v.size();
	std::sort(v.begin(), v.end());
	v.erase(std::unique(v.begin(), v.end()), v.end());
	if (!total)
		return;
	DevCon.WriteLn("%d / %d [%3.1f%%]", v.size(), total, 100. - (double)v.size() / (double)total * 100.);
}

// Compare Cached microProgram to mVU.regs().Micro
static bool mVUcmpProg(microVU& mVU, microProgram& prog)
{
	if (doWholeProgCompare)
	{
		if (memcmp((u8*)prog.data, mVU.regs().Micro, mVU.microMemSize))
			return false;
	}
	else
	{
		for (const auto& range : *prog.ranges)
		{
#if defined(PCSX2_DEVBUILD) || defined(_DEBUG)
			if ((range.start < 0) || (range.end < 0))
				DevCon.Error("microVU%d: Negative Range![%d][%d]", mVU.index, range.start, range.end);
#endif
			auto cmpOffset = [&](void* x) { return (u8*)x + range.start; };

			if (memcmp(cmpOffset(prog.data), cmpOffset(mVU.regs().Micro), (range.end - range.start)))
				return false;
		}
	}
	mVU.prog.cleared = 0;
	mVU.prog.cur = &prog;
	mVU.prog.isSame = doWholeProgCompare ? 1 : -1;
	return true;
}

// Searches for Cached Micro Program and sets prog.cur to it (returns entry-point to program)
template <int vuIndex>
void* mVUsearchProgUncached(u32 startPC, uptr pState)
{
	microVU& mVU = (vuIndex ? microVU1 : microVU0);
	microProgramQuick& quick = mVU.prog.quick[mVU.regs().start_pc / 8];
	microProgramList*  list  = mVU.prog.prog [mVU.regs().start_pc / 8];

	if (!quick.prog) // If null, we need to search for new program
	{
		for (auto it = list->begin(); it != list->end(); ++it)
		{
			bool b = mVUcmpProg(mVU, *it[0]);

			if (b)
			{
				quick.block = it[0]->block[startPC / 8];
				quick.prog  = it[0];
				list->erase(it);
				list->push_front(quick.prog);

				// Sanity check, in case for some reason the program compilation aborted half way through (JALR for example)
				if (quick.block == nullptr)
				{
					void* entryPoint = mVUblockFetch(mVU, startPC, pState);
					return entryPoint;
				}
				return mVUentryGet(mVU, quick.block, startPC, pState);
			}
		}

		// If cleared and program not found, make a new program instance
		mVU.prog.cleared = 0;
		mVU.prog.isSame  = 1;
		mVU.prog.cur     = mVUcreateProg(mVU, mVU.regs().start_pc/8);
		void* entryPoint = mVUblockFetch(mVU,  startPC, pState);
		quick.block      = mVU.prog.cur->block[startPC/8];
		quick.prog       = mVU.prog.cur;
		list->push_front(mVU.prog.cur);
		//mVUprintUniqueRatio(mVU);
		return entryPoint;
	}

	// If list.quick, then we've already found and recompiled the program ;)
	mVU.prog.isSame = -1;
	mVU.prog.cur = quick.prog;
	// Because the VU's can now run in sections and not whole programs at once
	// we need to set the current block so it gets the right program back
	quick.block = mVU.prog.cur->block[startPC / 8];

	// Sanity check, in case for some reason the program compilation aborted half way through
	if (quick.block == nullptr)
	{
		void* entryPoint = mVUblockFetch(mVU, startPC, pState);
		return entryPoint;
	}
	return mVUentryGet(mVU, quick.block, startPC, pState);
}

/* The invocation memo, arm64 twin: same keying as x86 after its trial
 * by fire - live quick pointer, section PC, and the query state
 * snapshotted before the search since lpState is a live variable the
 * search itself can overwrite. The compare uses the C++ state compare
 * rather than the emitted one, which also keeps Apple Silicon's
 * MAP_JIT write-protect sessions out of the picture entirely. The hit
 * replicates the quick path's side effects exactly. */
template <int vuIndex>
void* mVUsearchProg(u32 startPC, uptr pState)
{
	microVU& mVU = (vuIndex ? microVU1 : microVU0);
	const u32 idx = mVU.regs().start_pc / 8;
	microMscalMemoA* slot = &mVU.mscalMemo[idx];
	if (slot->prog && slot->prog == mVU.prog.quick[idx].prog
	 && slot->startPC == startPC
	 && mVU.compareState((microRegInfo*)pState, &slot->state) == 0)
	{
		mVU.prog.isSame = -1;
		mVU.prog.cur = slot->prog;
		mVU.prog.quick[idx].block = slot->prog->block[startPC / 8];
		return slot->entry;
	}
	{
		microRegInfo queryState;
		std::memcpy(&queryState, (void*)pState, sizeof(microRegInfo));
		void* entry = mVUsearchProgUncached<vuIndex>(startPC, pState);
		microProgram* p = mVU.prog.quick[idx].prog;
		if (p && mVU.regs().start_pc / 8 == idx)
		{
			slot->prog    = p;
			slot->entry   = entry;
			slot->startPC = startPC;
			std::memcpy(&slot->state, &queryState, sizeof(microRegInfo));
		}
		return entry;
	}
}

// Force both template instantiations to compile (the x86 rec instantiates these
// via mVUexecute<0/1>, which is task 7.2d; keep them compiled until then).
template void* mVUsearchProg<0>(u32 startPC, uptr pState);
template void* mVUsearchProg<1>(u32 startPC, uptr pState);

//------------------------------------------------------------------
// VU0 / VU1 recompiler providers
//------------------------------------------------------------------
// The x86 side replaced the BaseVUmicroCPU class hierarchy with a
// VUmicroCpu struct of function pointers; this is the ARM64 half of the
// same thing. Names and behaviour are unchanged, only the dispatch.

static void rec_vu0_shutdown(void)
{
	mVUclose(microVU0);
}

static void rec_vu1_shutdown(void)
{
	if (vu1Thread.IsOpen())
		vu1Thread.WaitVU();
	mVUclose(microVU1);
}

static void rec_vu0_reset(void)
{
	mVUreset(microVU0, true);
}

static void rec_vu1_reset(void)
{
	vu1Thread.WaitVU();
	vu1Thread.Get_MTVUChanges();
	mVUreset(microVU1, true);
}

static void rec_vu0_set_start_pc(u32 startPC)
{
	VU0.start_pc = startPC;
}

static void rec_vu1_set_start_pc(u32 startPC)
{
	VU1.start_pc = startPC;
}

static void rec_vu0_execute(u32 cycles)
{
	VU0.flags &= ~VUFLAG_MFLAGSET;

	if (!(VU0.VI[REG_VPU_STAT].UL & 1))
		return;
	VU0.VI[REG_TPC].UL <<= 3;

	((mVUrecCall)microVU0.startFunct)(VU0.VI[REG_TPC].UL, cycles);
	VU0.VI[REG_TPC].UL >>= 3;
	if (microVU0.regs().flags & 0x4)
	{
		microVU0.regs().flags &= ~0x4;
		hwIntcIrq(6);
	}
}

static void rec_vu1_execute(u32 cycles)
{
	if (!THREAD_VU1)
	{
		if (!(VU0.VI[REG_VPU_STAT].UL & 0x100))
			return;
	}

	VU1.VI[REG_TPC].UL <<= 3;
	// HH-37: the AOT form's dispatcher takes the import table too (it puts it in x28)
	if (mvuAot(microVU1))
		((void (*)(u32, u32, void**))microVU1.startFunct)(VU1.VI[REG_TPC].UL, cycles, aot::imports());
	else
		((mVUrecCall)microVU1.startFunct)(VU1.VI[REG_TPC].UL, cycles);
	VU1.VI[REG_TPC].UL >>= 3;
	if (microVU1.regs().flags & 0x4 && !THREAD_VU1)
	{
		microVU1.regs().flags &= ~0x4;
		hwIntcIrq(7);
	}
}

static void rec_vu0_clear(u32 addr, u32 size)
{
	mVUclear(microVU0, addr, size);
}

static void rec_vu1_clear(u32 addr, u32 size)
{
	mVUclear(microVU1, addr, size);
}

static void rec_vu1_resume_xgkick(void)
{
	if (!(VU0.VI[REG_VPU_STAT].UL & 0x100))
		return;
	if (mvuAot(microVU1))
		((void (*)(void**))microVU1.startFunctXG)(aot::imports());
	else
		((mVUrecCallXG)microVU1.startFunctXG)();
}

void vucpu_rec_vu0_reserve(void)
{
	mVUinit(microVU0, 0);
}

void vucpu_rec_vu1_reserve(void)
{
	mVUinit(microVU1, 1);
	vu1Thread.Open();
}

const struct VUmicroCpu vucpu_rec_vu0 =
{
	0, 0,
	rec_vu0_shutdown, rec_vu0_reset, rec_vu0_set_start_pc,
	rec_vu0_execute,  rec_vu0_clear, NULL
};

const struct VUmicroCpu vucpu_rec_vu1 =
{
	1, 0,
	rec_vu1_shutdown, rec_vu1_reset, rec_vu1_set_start_pc,
	rec_vu1_execute,  rec_vu1_clear, rec_vu1_resume_xgkick
};

bool SaveStateBase::vuJITFreeze()
{
	if (IsSaving())
		vu1Thread.WaitVU();

	Freeze(microVU0.prog.lpState);
	Freeze(microVU1.prog.lpState);
	return IsOkay();
}

//------------------------------------------------------------------
// Dispatcher / helper-thunk code generation (task 7.2d)
//------------------------------------------------------------------
// Shared exact-multiply tree stub (x86: mVUemitExactMulStub). The
// harness-proven paired Booth/CSA tree, transcribed for NEON. Memory
// ABI through mVU.exactMulBuf: in [0..3] fs, [4..7] ft (the raw operand
// words -- only their low halfwords are read, which the implicit bit at
// 23 does not reach, so mantissas and raw words are interchangeable
// here); [8..11] p01 and [12..15] p23 are the Umull-pair products,
// corrected in place. Saves and restores every vector
// register it touches, so call sites need no flush. NEON shortens
// the x86 stub: one per-lane Ushl with a count vector replaces each
// iteration's shift-shift-blend, Tbl is the mask lookup, two
// immediate Bics are the & 7, and Ext does the unpairing.
//------------------------------------------------------------------
static void emitAdd3(const a64::VRegister& x, const a64::VRegister& y, const a64::VRegister& z,
	const a64::VRegister& lo, const a64::VRegister& hi, const a64::VRegister& w1, const a64::VRegister& w2)
{
	armAsm->Eor(w1.V16B(), x.V16B(), y.V16B());        // u
	armAsm->And(w2.V16B(), x.V16B(), y.V16B());
	armAsm->And(hi.V16B(), w1.V16B(), z.V16B());       // u & z
	armAsm->Orr(hi.V16B(), hi.V16B(), w2.V16B());
	armAsm->Shl(hi.V8H(), hi.V8H(), 1);
	armAsm->Eor(lo.V16B(), w1.V16B(), z.V16B());
}

static void mVUemitExactMulStub(microVU& mVU)
{
	u32* buf = mVU.exactMulBuf;
	int i, k;
	const u32* cnts[4] = {&mVU.glob.tCnt0[0], &mVU.glob.tCnt1[0], &mVU.glob.tCnt2[0], &mVU.glob.tCnt3[0]};

	mVU.exactMulStub = armGetCurrentCodePointer();
	for (i = 0; i < 16; i++)
		mvuStrQ(mVU, &mVU.exactMulSave[i][0], a64::VRegister(i, 128));

	// v0 = aP, v1 = b16: load, narrow to 4H, duplicate, pre-shift high half
	mvuLdrQ(mVU, a64::v0, &buf[0]);
	armAsm->Xtn(a64::v0.V4H(), a64::v0.V4S());
	armAsm->Ins(a64::v0.V2D(), 1, a64::v0.V2D(), 0);
	mvuLdrQ(mVU, a64::v14, &mVU.glob.tPackCnt[0]);
	armAsm->Ushl(a64::v0.V8H(), a64::v0.V8H(), a64::v14.V8H());
	mvuLdrQ(mVU, a64::v1, &buf[4]);
	armAsm->Xtn(a64::v1.V4H(), a64::v1.V4S());
	armAsm->Ins(a64::v1.V2D(), 1, a64::v1.V2D(), 0);

	// LUT tables resident
	mvuLdrQ(mVU, a64::v9,  &mVU.glob.tLUTdbl[0]);
	mvuLdrQ(mVU, a64::v10, &mVU.glob.tLUTneg[0]);
	mvuLdrQ(mVU, a64::v11, &mVU.glob.tLUTany[0]);

	// booth loop; regs: v0 aP, v1 b16, v2-5 d0-3, v6-8 n1-3, v13 x, v14 t, v15 work
	for (k = 0; k < 4; k++)
	{
		const a64::VRegister dk = a64::VRegister(2 + k, 128);
		mvuLdrQ(mVU, a64::v15, cnts[k]);
		armAsm->Ushl(a64::v14.V8H(), a64::v1.V8H(), a64::v15.V8H());
		armAsm->Bic(a64::v14.V8H(), 0xf8, 0);
		armAsm->Bic(a64::v14.V8H(), 0xff, 8);          // t = shifted & 7
		armAsm->Shl(a64::v15.V8H(), a64::v14.V8H(), 8);
		armAsm->Orr(a64::v14.V16B(), a64::v14.V16B(), a64::v15.V16B()); // both bytes
		if (k) armAsm->Shl(a64::v13.V8H(), a64::v0.V8H(), k * 2);
		else   armAsm->Mov(a64::v13.V16B(), a64::v0.V16B());            // x = aP << 2k
		armAsm->Tbl(a64::v15.V16B(), a64::v9.V16B(), a64::v14.V16B());  // dbl
		armAsm->And(a64::v15.V16B(), a64::v15.V16B(), a64::v13.V16B());
		armAsm->Add(a64::v13.V8H(), a64::v13.V8H(), a64::v15.V8H());
		armAsm->Tbl(a64::v15.V16B(), a64::v10.V16B(), a64::v14.V16B()); // neg
		if (k >= 1)
		{
			const a64::VRegister nk = a64::VRegister(5 + k, 128);
			mvuLdrQ(mVU, dk, &mVU.glob.tOneP[0]);
			armAsm->Shl(dk.V8H(), dk.V8H(), k * 2);
			armAsm->And(nk.V16B(), a64::v15.V16B(), dk.V16B());        // n[k]
		}
		mvuLdrQ(mVU, dk, &mVU.glob.tOnesP[0]);
		if (k) armAsm->Shl(dk.V8H(), dk.V8H(), k * 2);
		armAsm->And(dk.V16B(), a64::v15.V16B(), dk.V16B());
		armAsm->Eor(a64::v13.V16B(), a64::v13.V16B(), dk.V16B());      // x ^= neg & (onesP<<2k)
		armAsm->Tbl(a64::v15.V16B(), a64::v11.V16B(), a64::v14.V16B()); // any
		armAsm->And(dk.V16B(), a64::v13.V16B(), a64::v15.V16B());       // d[k]
	}

	// unpair; v0, v1, v9-15 free now
	armAsm->Ext(a64::v9.V16B(),  a64::v2.V16B(), a64::v2.V16B(), 8);   // d4
	armAsm->Ext(a64::v10.V16B(), a64::v3.V16B(), a64::v3.V16B(), 8);   // d5
	armAsm->Ext(a64::v11.V16B(), a64::v4.V16B(), a64::v4.V16B(), 8);   // d6
	armAsm->Ext(a64::v12.V16B(), a64::v5.V16B(), a64::v5.V16B(), 8);   // d7
	armAsm->Ext(a64::v13.V16B(), a64::v6.V16B(), a64::v6.V16B(), 8);   // n5

	// b7 into v12 ; then d5&0x800 into v13 ; mask d4/d5
	mvuLdrQ(mVU, a64::v14, &mVU.glob.t0400[0]);
	armAsm->And(a64::v14.V16B(), a64::v10.V16B(), a64::v14.V16B());
	armAsm->Add(a64::v14.V8H(), a64::v14.V8H(), a64::v13.V8H());
	armAsm->Orr(a64::v12.V16B(), a64::v12.V16B(), a64::v14.V16B());
	mvuLdrQ(mVU, a64::v13, &mVU.glob.t0800[0]);
	armAsm->And(a64::v13.V16B(), a64::v10.V16B(), a64::v13.V16B());
	mvuLdrQ(mVU, a64::v14, &mVU.glob.tF800[0]);
	armAsm->And(a64::v9.V16B(), a64::v9.V16B(), a64::v14.V16B());
	mvuLdrQ(mVU, a64::v14, &mVU.glob.tF000[0]);
	armAsm->And(a64::v10.V16B(), a64::v10.V16B(), a64::v14.V16B());

	// t0 = add3(d1,d2,d3) -> lo v14, hi v15 (w: v1, v0)
	emitAdd3(a64::v3, a64::v4, a64::v5, a64::v14, a64::v15, a64::v1, a64::v0);
	// t1 = add3(d4m,d5m,d6) -> lo v3, hi v4 ; then |= n6 | (d5&0x800)
	emitAdd3(a64::v9, a64::v10, a64::v11, a64::v3, a64::v4, a64::v5, a64::v0);
	armAsm->Ext(a64::v5.V16B(), a64::v7.V16B(), a64::v7.V16B(), 8);    // n6
	armAsm->Orr(a64::v4.V16B(), a64::v4.V16B(), a64::v5.V16B());
	armAsm->Orr(a64::v4.V16B(), a64::v4.V16B(), a64::v13.V16B());
	// t2 = add3(d0,t0l,t0h) -> lo v9, hi v10
	emitAdd3(a64::v2, a64::v14, a64::v15, a64::v9, a64::v10, a64::v11, a64::v0);
	// t3 = add3(b7,t1l,t1h) -> lo v13, hi v14
	emitAdd3(a64::v12, a64::v3, a64::v4, a64::v13, a64::v14, a64::v15, a64::v0);
	// t4 = add3(t2h,t3l,t3h) -> lo v2, hi v3
	emitAdd3(a64::v10, a64::v13, a64::v14, a64::v2, a64::v3, a64::v4, a64::v0);
	// t5 = add3(t2l,t4l,t4h) -> lo v5, hi v10
	emitAdd3(a64::v9, a64::v2, a64::v3, a64::v5, a64::v10, a64::v11, a64::v0);
	// t5h += n7 ; sum ; tb
	armAsm->Ext(a64::v11.V16B(), a64::v8.V16B(), a64::v8.V16B(), 8);   // n7
	armAsm->Add(a64::v10.V8H(), a64::v10.V8H(), a64::v11.V8H());
	mvuLdrQ(mVU, a64::v11, &mVU.glob.t8000[0]);
	armAsm->And(a64::v5.V16B(), a64::v5.V16B(), a64::v11.V16B());
	armAsm->And(a64::v10.V16B(), a64::v10.V16B(), a64::v11.V16B());
	armAsm->Add(a64::v5.V8H(), a64::v5.V8H(), a64::v10.V8H());
	armAsm->Ushr(a64::v5.V8H(), a64::v5.V8H(), 15);                    // tb 4H lanes 0-3

	// correction: p -= ((tb ^ ((p >> 15) & 1)) << 15) per 64-bit lane
	armAsm->Uxtl(a64::v5.V4S(), a64::v5.V4H());                        // {b0,b1,b2,b3}
	armAsm->Uxtl (a64::v6.V2D(), a64::v5.V2S());                       // {b0,b1}
	armAsm->Uxtl2(a64::v7.V2D(), a64::v5.V4S());                       // {b2,b3}
	mvuLdrQ(mVU, a64::v1, &buf[8]);
	mvuLdrQ(mVU, a64::v2, &buf[12]);
	mvuLdrQ(mVU, a64::v8, &mVU.glob.t64one[0]);
	armAsm->Ushr(a64::v3.V2D(), a64::v1.V2D(), 15);
	armAsm->And(a64::v3.V16B(), a64::v3.V16B(), a64::v8.V16B());
	armAsm->Ushr(a64::v4.V2D(), a64::v2.V2D(), 15);
	armAsm->And(a64::v4.V16B(), a64::v4.V16B(), a64::v8.V16B());
	armAsm->Eor(a64::v6.V16B(), a64::v6.V16B(), a64::v3.V16B());
	armAsm->Eor(a64::v7.V16B(), a64::v7.V16B(), a64::v4.V16B());
	armAsm->Shl(a64::v6.V2D(), a64::v6.V2D(), 15);
	armAsm->Shl(a64::v7.V2D(), a64::v7.V2D(), 15);
	armAsm->Sub(a64::v1.V2D(), a64::v1.V2D(), a64::v6.V2D());
	armAsm->Sub(a64::v2.V2D(), a64::v2.V2D(), a64::v7.V2D());
	mvuStrQ(mVU, &buf[8], a64::v1);
	mvuStrQ(mVU, &buf[12], a64::v2);

	for (i = 0; i < 16; i++)
		mvuLdrQ(mVU, a64::VRegister(i, 128), &mVU.exactMulSave[i][0]);
	armAsm->Ret();
}

//------------------------------------------------------------------
// ARM64 port of microVU_Execute.inl 23-315. This pins the final ARM64 microVU
// ABI. All five generators below emit into the armAsm opened by
// mVUgenerateDispatchers (one MacroAssembler session for the whole dispatcher
// block, finalised once). microVU is still unselected on ARM64 (VMManager pins
// CpuIntVU0/1), so the emitted code is not executed yet — but it compiles for
// real and pins the register/entry contract every later Phase-7 task builds on.
//
// Entry contract (mVUdispatcherAB = mVU.startFunct), called from C as
//   ((void(*)(u32 startPC, u32 cycles))startFunct)(startPC, cycles):
//   * startPC -> w0 (RWARG1), cycles -> w1 (RWARG2);
//   * call mVUexecuteVU0/1(startPC, cycles); it returns the block entry in x0;
//   * load the VU FPCR (if it differs from the EE's), set RVUSTATE = &vuRegs[i],
//     load the PQ NEON reg + mac/clip flag instances + the 4 status-flag GPRs
//     (gprF0-3 = w23-w26);
//   * br x0 (into the compiled block); the block exits by jumping to
//     mVU.exitFunct, which restores the EE FPCR and runs mVUcleanUpVU0/1.

// Whether the dispatcher needs to swap FPCR on entry/exit (x86: mvuNeedsFPCRUpdate).
static bool mvuNeedsFPCRUpdate(microVU& mVU)
{
	// Always update on the VU1 (MTVU) thread.
	if (mVU.index == 1 && THREAD_VU1)
		return true;

	// Otherwise only when the VU's FPCR differs from the EE's.
	return EmuConfig.Cpu.FPUFPCR.bitmask !=
		(mVU.index == 0 ? EmuConfig.Cpu.VU0FPCR.bitmask : EmuConfig.Cpu.VU1FPCR.bitmask);
}

// Emit: load the u64 FPCR bitmask at `bitmaskPtr` and write it to the host FPCR.
// Uses x16/x17 scratch only (never the block-entry x0 the AB dispatcher holds).
static void mVUemitSetHostFPCR(const void* bitmaskPtr)
{
	armAsm->Ldr(RXVIXLSCRATCH, armAbsMemOperand(RSCRATCHADDR, bitmaskPtr, 8)); // x16 = bitmask
	armAsm->Msr(a64::FPCR, RXVIXLSCRATCH);
}

// Generates the code for entering/exiting recompiled blocks.
static void mVUdispatcherAB(microVU& mVU)
{
	mVU.startFunct = armGetCurrentCodePointer();

	// Save callee-saved GPRs (x19-x28, fp/lr) and the low halves of v8-v15.
	armBeginStackFrame(true);
	if (armAotRoute) // HH-37: x28 = the import table, the third argument
		armAsm->Mov(a64::x28, a64::x2);

	// Args (startPC, cycles) are already in w0/w1; mVUexecute returns entry in x0.
	armEmitCall(reinterpret_cast<const void*>(isVU1 ? mVUexecuteVU1 : mVUexecuteVU0));

	// Load VU's FPCR state.
	if (mvuNeedsFPCRUpdate(mVU))
		mVUemitSetHostFPCR(isVU1 ? &EmuConfig.Cpu.VU1FPCR.bitmask : &EmuConfig.Cpu.VU0FPCR.bitmask);

	// All VF/VI/flag loads address off RVUSTATE = &vuRegs[index]; mVU's own
	// fields (flag instances, cycles, branch state) address off RVUMVU = &mVU
	// (C.74). Both are callee-saved and survive helper C calls and block chains.
	armMoveAddressToReg(RVUSTATE, &::vuRegs[mVU.index]);
	armMoveAddressToReg(RVUMVU, &mVU);

	// Build the PQ latency register (v24). Final lane layout matches the x86 rec:
	//   VU0: [Q, pending_q, P, P]   VU1: [Q, pending_q, P, pending_p]
	// (NEON V4S lane0 = lowest 32 bits = X, same order as xmm.) v25 is a free
	// scratch (not in the VF pool v0-v23, not PQ, not VIXL scratch v31).
	const a64::VRegister vtmp = a64::VRegister(25, 128);
	armAsm->Ldr(mVU_xmmPQ.S(), a64::MemOperand(RVUSTATE, mVUoffVI(REG_Q)));      // lane0 = Q
	armAsm->Ldr(vtmp.S(),      a64::MemOperand(RVUSTATE, offsetof(VURegs, pending_q)));
	armAsm->Ins(mVU_xmmPQ.V4S(), 1, vtmp.V4S(), 0);                              // lane1 = pending_q
	armAsm->Ldr(vtmp.S(),      a64::MemOperand(RVUSTATE, mVUoffVI(REG_P)));      // vtmp = P
	armAsm->Ins(mVU_xmmPQ.V4S(), 2, vtmp.V4S(), 0);                             // lane2 = P
	if (isVU1)
	{
		armAsm->Ldr(vtmp.S(), a64::MemOperand(RVUSTATE, offsetof(VURegs, pending_p)));
		armAsm->Ins(mVU_xmmPQ.V4S(), 3, vtmp.V4S(), 0);                         // lane3 = pending_p
	}
	else
	{
		armAsm->Ins(mVU_xmmPQ.V4S(), 3, vtmp.V4S(), 0);                         // lane3 = P (vtmp still = P)
	}

	// Copy the mac/clip flag instances into the microVU's working storage.
	armAsm->Ldr(vtmp.Q(), a64::MemOperand(RVUSTATE, offsetof(VURegs, micro_macflags)));
	armAsm->Str(vtmp.Q(), mvuAbsMem(mVU, &mVU.macFlag[0], 16));
	armAsm->Ldr(vtmp.Q(), a64::MemOperand(RVUSTATE, offsetof(VURegs, micro_clipflags)));
	armAsm->Str(vtmp.Q(), mvuAbsMem(mVU, &mVU.clipFlag[0], 16));

	// Load the 4 status-flag instances into gprF0-gprF3 (w23-w26).
	for (int i = 0; i < 4; i++)
		armAsm->Ldr(armWRegister(mVU_F0 + i), a64::MemOperand(RVUSTATE, offsetof(VURegs, micro_statusflags) + i * 4));

	// Jump to the recompiled code block (entry still in x0).
	armAsm->Br(RXRET);

	// --- exit path: the block jumps here when done ---
	mVU.exitFunct = armGetCurrentCodePointer();

	// Restore the EE's FPCR state.
	if (mvuNeedsFPCRUpdate(mVU))
		mVUemitSetHostFPCR(&EmuConfig.Cpu.FPUFPCR.bitmask);

	armEmitCall(reinterpret_cast<const void*>(isVU1 ? mVUcleanUpVU1 : mVUcleanUpVU0));

	armEndStackFrame(true);
	armAsm->Ret();
}

// Generates the code for resuming/exiting xgkick.
static void mVUdispatcherCD(microVU& mVU)
{
	mVU.startFunctXG = armGetCurrentCodePointer();

	armBeginStackFrame(true);
	if (armAotRoute) // HH-37: x28 = the import table, the argument
		armAsm->Mov(a64::x28, a64::x0);

	// C.74: pin &mVU before anything below (or in the resumed block) uses it.
	armMoveAddressToReg(RVUMVU, &mVU);

	// Load VU's FPCR state.
	if (mvuNeedsFPCRUpdate(mVU))
		mVUemitSetHostFPCR(isVU1 ? &EmuConfig.Cpu.VU1FPCR.bitmask : &EmuConfig.Cpu.VU0FPCR.bitmask);

	// mVUrestoreRegs(): reload the PQ reg from its XGKICK backup slot.
	armAsm->Ldr(mVU_xmmPQ.Q(), mvuAbsMem(mVU, &mVU.vecBackup[mVU_xmmPQ.GetCode()][0], 16));

	armMoveAddressToReg(RVUSTATE, &::vuRegs[mVU.index]);
	for (int i = 0; i < 4; i++)
		armAsm->Ldr(armWRegister(mVU_F0 + i), a64::MemOperand(RVUSTATE, offsetof(VURegs, micro_statusflags) + i * 4));

	// Jump to the recompiled code position to resume xgkick.
	armAsm->Ldr(RSCRATCHADDR, armAbsMemOperand(RSCRATCHADDR, &mVU.resumePtrXG, 8));
	armAsm->Br(RSCRATCHADDR);

	// --- exit path ---
	mVU.exitFunctXG = armGetCurrentCodePointer();

	// Back up the status flags (the other regs were backed up on xgkick).
	armMoveAddressToReg(RVUSTATE, &::vuRegs[mVU.index]);
	for (int i = 0; i < 4; i++)
		armAsm->Str(armWRegister(mVU_F0 + i), a64::MemOperand(RVUSTATE, offsetof(VURegs, micro_statusflags) + i * 4));

	// Restore the EE's FPCR state.
	if (mvuNeedsFPCRUpdate(mVU))
		mVUemitSetHostFPCR(&EmuConfig.Cpu.FPUFPCR.bitmask);

	armEndStackFrame(true);
	armAsm->Ret();
}

// The C helper called by the waitMTVU thunk (x86: microVU_Misc.inl mVUwaitMTVU).
static void mVUwaitMTVU()
{
	if (IsDevBuild)
		DevCon.WriteLn("microVU: Waiting on VU1 thread to access VU1 regs!");
	vu1Thread.WaitVU();
}

// A transparent helper-thunk: the register allocator does not know this call
// happens, so it must preserve every caller-saved host reg that can hold live VU
// state — the VF/PQ NEON pool (v0-v24) and the caller-saved GPRs that can hold VI
// values (x0-x15). gprF0-3 (w23-w26) and callee-saved VI regs survive the inner C
// call automatically. Only reached under MTVU (VU1 thread).
static void mVUGenerateWaitMTVU(microVU& mVU)
{
	mVU.waitMTVU = armGetCurrentCodePointer();

	constexpr int kGprSave = 16; // x0..x15
	constexpr int kVecSave = 25; // v0..v24 (VF pool + PQ)
	constexpr int gprBytes = kGprSave * 8;
	constexpr int vecBytes = kVecSave * 16;
	constexpr int frame = gprBytes + ((vecBytes + 15) & ~15); // 16-aligned, 0x210
	// 16 extra bytes at [sp+frame] to save/restore lr (x30) around the C call.
	// blr clobbers x30 with the instruction after the call site; without this save,
	// ret returns to the restore sequence itself instead of the VU block that called
	// the thunk, creating a loop that advances sp by frame each iteration until it
	// reaches the thread stack guard (~41 iterations) and SIGBUSes.
	constexpr int frameTotal = frame + 16;

	armAsm->Sub(a64::sp, a64::sp, frameTotal);
	for (int i = 0; i < kGprSave; i += 2)
		armAsm->Stp(armXRegister(i), armXRegister(i + 1), a64::MemOperand(a64::sp, i * 8));
	int voff = gprBytes;
	for (int i = 0; i < kVecSave - 1; i += 2, voff += 32)
		armAsm->Stp(armQRegister(i), armQRegister(i + 1), a64::MemOperand(a64::sp, voff));
	armAsm->Str(armQRegister(kVecSave - 1), a64::MemOperand(a64::sp, voff));

	armAsm->Str(a64::x30, a64::MemOperand(a64::sp, frame));
	armEmitCall(reinterpret_cast<const void*>(mVUwaitMTVU));
	armAsm->Ldr(a64::x30, a64::MemOperand(a64::sp, frame));

	voff = gprBytes;
	for (int i = 0; i < kVecSave - 1; i += 2, voff += 32)
		armAsm->Ldp(armQRegister(i), armQRegister(i + 1), a64::MemOperand(a64::sp, voff));
	armAsm->Ldr(armQRegister(kVecSave - 1), a64::MemOperand(a64::sp, voff));
	for (int i = 0; i < kGprSave; i += 2)
		armAsm->Ldp(armXRegister(i), armXRegister(i + 1), a64::MemOperand(a64::sp, i * 8));
	armAsm->Add(a64::sp, a64::sp, frameTotal);
	armAsm->Ret();
}

// Copies the 96-byte pipeline state from [x0] (source pState, supplied by the
// caller) into mVU.prog.lpState (x86: mVUGenerateCopyPipelineState).
static void mVUGenerateCopyPipelineState(microVU& mVU)
{
	mVU.copyPLState = armGetCurrentCodePointer();

	armMoveAddressToReg(RXARG2, &mVU.prog.lpState); // x1 = dest
	for (int i = 0; i < 6; i++)
	{
		const a64::VRegister v = a64::VRegister(i, 128);
		armAsm->Ldr(v.Q(), a64::MemOperand(RXARG1, i * 16));
		armAsm->Str(v.Q(), a64::MemOperand(RXARG2, i * 16));
	}
	armAsm->Ret();
}

// Custom optimised block-search comparator: compares the two 96-byte
// microRegInfo blocks at x0 (lhs) and x1 (rhs); returns w0 = 0 iff fully equal
// (x86: mVUGenerateCompareState; the search treats 0 as a match).
static void mVUGenerateCompareState(microVU& mVU)
{
	mVU.compareStateF = armGetCurrentCodePointer();

	// 6 x 128-bit lanewise equality compares (0xffffffff per equal 32-bit lane),
	// AND-reduced into v0.
	auto cmp16 = [&](int off, int dst, int t0, int t1) {
		const a64::VRegister a = a64::VRegister(t0, 128), b = a64::VRegister(t1, 128);
		armAsm->Ldr(a.Q(), a64::MemOperand(RXARG1, off));
		armAsm->Ldr(b.Q(), a64::MemOperand(RXARG2, off));
		armAsm->Cmeq(a64::VRegister(dst, 128).V4S(), a.V4S(), b.V4S());
	};
	cmp16(0x00, 0, 0, 1);
	cmp16(0x10, 2, 2, 3);
	cmp16(0x20, 4, 4, 5);
	cmp16(0x30, 6, 6, 7);
	armAsm->And(a64::VRegister(0, 128).V16B(), a64::VRegister(0, 128).V16B(), a64::VRegister(2, 128).V16B());
	armAsm->And(a64::VRegister(4, 128).V16B(), a64::VRegister(4, 128).V16B(), a64::VRegister(6, 128).V16B());
	cmp16(0x40, 2, 2, 3);
	cmp16(0x50, 6, 6, 7);
	armAsm->And(a64::VRegister(2, 128).V16B(), a64::VRegister(2, 128).V16B(), a64::VRegister(6, 128).V16B());
	armAsm->And(a64::VRegister(0, 128).V16B(), a64::VRegister(0, 128).V16B(), a64::VRegister(4, 128).V16B());
	armAsm->And(a64::VRegister(0, 128).V16B(), a64::VRegister(0, 128).V16B(), a64::VRegister(2, 128).V16B());

	// v0 is all-ones iff every byte matched. Reduce: min byte == 0xff iff equal.
	armAsm->Uminv(a64::VRegister(1, 128).B(), a64::VRegister(0, 128).V16B());
	armAsm->Umov(RWRET, a64::VRegister(1, 128).V16B(), 0);
	armAsm->Eor(RWRET, RWRET, 0xff); // 0 iff equal
	armAsm->Ret();
}

// Emit the dispatchers + helper thunks at the start of the VU's code cache, then
// set codeStart/codePtr just past them (x86: x86Ptr = (u8*)(mVU.cache) + the five
// generate calls + xGetAlignedCallTarget in mVUreset). No constant pool: the
// dispatchers materialise addresses via adrp/mov and call C helpers via mov+blr.
static void mVUgenerateDispatchers(microVU& mVU)
{
	u8* const cacheEnd = mVU.prog.codeReserveEnd;
	armSetAsmPtr(mVU.cache, static_cast<size_t>(cacheEnd - mVU.cache), nullptr);
	armStartBlock();

	mVUdispatcherAB(mVU);
	mVUdispatcherCD(mVU);
	mVUemitExactMulStub(mVU);
	mVUGenerateWaitMTVU(mVU);
	mVUGenerateCopyPipelineState(mVU);
	mVUGenerateCompareState(mVU);

	std::memset(mVU.mscalMemo, 0, (mVU.progSize / 2) * sizeof(microMscalMemoA)); /* code region rebuilt: every memoized entry dies */
	u8* const end = armEndBlock();
	mVU.prog.codeStart = mVU.prog.codePtr =
		reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(end) + 15u) & ~uintptr_t(15));
}

//------------------------------------------------------------------
// Execution / cleanup (microVU_Execute.inl 318-381)
//------------------------------------------------------------------

// Executes for the given number of cycles: finds (compiling if needed) the
// microprogram for startPC and returns its host entry point.
template <int vuIndex>
static void* mVUexecute(u32 startPC, u32 cycles)
{
	microVU& mVU = (vuIndex ? microVU1 : microVU0);
	const u32 vuLimit = vuIndex ? 0x3ff8 : 0xff8;
	if (startPC > vuLimit + 7)
		DevCon.Warning("microVU%x Warning: startPC = 0x%x, cycles = 0x%x", vuIndex, startPC, cycles);

	mVU.cycles = cycles;
	mVU.totalCycles = cycles;

	const u32 spc = startPC & vuLimit;

	// HH-37: the AOT form finds a block by its pc, state and words alone (vu1AotFind); the
	// program search's one effect on what runs is that a found program is no longer cleared
	if (mvuAot(mVU))
	{
		mVU.prog.cleared = 0;
		return vu1AotFind(mVU, spc, &mVU.prog.lpState);
	}

	// Fast path: if the microprogram + block for this (start_pc, startPC, pipeline
	// state) is already compiled, return its host entry directly and skip the emit
	// session below. armStartBlock/armEndBlock allocate+finalize a VIXL
	// MacroAssembler, toggle W^X (Begin/EndCodeWrite), and flush the icache *on every
	// VU dispatch* -- pure overhead when nothing is emitted (x86 just bumps its global
	// emit cursor here, which is free). VU0-heavy titles (e.g. Ratchet & Clank)
	// dispatch VU constantly with the cache already warm, so this wrapping dominated
	// the EE thread (profiled at ~50% of CPU-thread samples in armStartBlock/
	// armEndBlock, vs ~0 actually compiling). mVUclear/mVUreset NULL every
	// quick[i].prog, so a non-NULL quick.prog means the cache is valid; the slow path
	// populates it on the first (cold) lookup. Mirrors mVUsearchProg's quick-reference
	// hit path (and mVUentryGet's hit branch) exactly, minus the compile.
	{
		microProgramQuick& quick = mVU.prog.quick[mVU.regs().start_pc / 8];
		if (quick.prog && !mVU.prog.cleared)
		{
			if (microBlockManager* mgr = quick.prog->block[spc / 8])
			{
				if (microBlock* b = mgr->search(mVU, (microRegInfo*)&mVU.prog.lpState))
				{
					mVU.prog.isSame = -1;
					mVU.prog.cur    = quick.prog;
					quick.block     = mgr;
					return b->codeStart;
				}
			}
		}
	}

	// Cold/compile path: open one MacroAssembler session at the program cursor for the
	// whole search/compile (the recursive mVUcompile/normBranch calls append into it);
	// armEndBlock finalises + flushes the icache so the freshly emitted code is safe to
	// branch into when the dispatcher does `br x0`. If the search still turns out to be
	// a hit, nothing is emitted and codePtr is unchanged (the fast path above usually
	// catches that case before paying for the session).
	armSetAsmPtr(mVU.prog.codePtr, mVU.prog.codeReserveEnd - mVU.prog.codePtr, nullptr);
	armStartBlock();
	void* const entry = mVUsearchProg<vuIndex>(spc, (uptr)&mVU.prog.lpState);
	mVU.prog.codePtr = armEndBlock();
	return entry;
}

// Post-execution bookkeeping: cache-limit reset + cycle accounting.
template <int vuIndex>
static void mVUcleanUp()
{
	microVU& mVU = (vuIndex ? microVU1 : microVU0);

	// The block compiler advanced mVU.prog.codePtr as it emitted; if it ran past
	// the cache limit, reset the program cache (x86 checks x86Ptr here).
	// (HH-37: an image's VU1 has no cache)
	if (mVU.cache && ((mVU.prog.codePtr < mVU.prog.codeStart) || (mVU.prog.codePtr >= mVU.prog.codeEnd)))
	{
		Console.WriteLn(vuIndex ? Color_Orange : Color_Magenta, "microVU%d: Program cache limit reached.", mVU.index);
		mVUreset(mVU, false);
	}

	mVU.cycles = mVU.totalCycles - std::max(0, mVU.cycles);
	mVU.regs().cycle += mVU.cycles;

	if (!vuIndex || !THREAD_VU1)
	{
		u32 cycles_passed = std::min(mVU.cycles, 3000) * EmuConfig.Speedhacks.EECycleSkip;
		if (cycles_passed > 0)
		{
			s64 vu0_offset = VU0.cycle - cpuRegs.cycle;
			cpuRegs.cycle += cycles_passed;

			// VU0 needs to stay in sync with the CPU otherwise things get messy.
			// So we need to adjust when VU1 skips cycles also.
			if (!vuIndex)
				VU0.cycle = cpuRegs.cycle + vu0_offset;
			else
				VU0.cycle += cycles_passed;
		}
	}
	mVU.profiler.Print();
}

// Caller wrappers (x86: microVU_Execute.inl 378-381) — the dispatcher branches
// to these by address.
static void* mVUexecuteVU0(u32 startPC, u32 cycles) { return mVUexecute<0>(startPC, cycles); }
static void* mVUexecuteVU1(u32 startPC, u32 cycles) { return mVUexecute<1>(startPC, cycles); }
static void  mVUcleanUpVU0() { mVUcleanUp<0>(); }
static void  mVUcleanUpVU1() { mVUcleanUp<1>(); }

//------------------------------------------------------------------
// Block compiler (Tables/Compile big-bang)
//------------------------------------------------------------------
// The full emit-coupled compile driver: the per-instruction executors + D/T-bit /
// cycle-test early exits + register preloader + debug/bad-op emitters, then
// mVUcompile itself and the block entry points (mVUentryGet/mVUblockFetch/
// mVUcompileJIT). Included here — after the arch-neutral pass-1 helpers
// (mVUsetupRange/mVUincCycles/...) and mVUsearchProg above, which it calls — so
// every callee is already declared. The branch drivers it mutually recurses with
// live in aVU_Branch.inl (included earlier, with forward decls of mVUcompile).
// This replaces the former mVUblockFetch/mVUentryGet pxFailRel stubs.
#include "arm64/aVU_Compile.inl"

//------------------------------------------------------------------
// Build-time validation
//------------------------------------------------------------------

// Layout sanity checks for the ported pipeline-state key / IR structs. These
// mirror the invariants the x86 microVU relies on (the 96-byte microRegInfo is
// compared as six 128-bit vectors by the generated compareStateF).
static_assert(sizeof(microRegInfo) == 96, "microRegInfo must stay 96 bytes (host pipeline-state compare)");
static_assert(alignof(microRegInfo) == 16, "microRegInfo must stay 16-byte aligned");
static_assert(alignof(microBlock) == 16, "microBlock must stay 16-byte aligned");

// Force the register allocator's emission paths to be compiled. microRegAlloc's
// emission members (allocReg/flushAll/...) are not reached by the cache-management
// code above (the block compiler/dispatcher call them — tasks 7.2d onward), so
// without an explicit use their VIXL bodies would never be instantiated and any
// emission error would go undetected. This function is never called.
[[maybe_unused]] static void mVUallocCompileCheck()
{
	microRegAlloc ra(0);
	ra.reset();
	(void)ra.allocReg(1, 2, 0xf);
	(void)ra.allocReg(0, 1, 0x4);
	(void)ra.allocReg(33, -1, 0x8);
	(void)ra.allocReg(32, 32, 0x6, false);
	(void)ra.allocGPR(1, 2, true);
	(void)ra.allocGPR(-1, 3);
	ra.moveVIToGPR(armWRegister(0), 1, true);
	ra.flushAll();
	ra.flushCallerSavedRegisters(true);
	ra.TDwritebackAll();
	(void)ra.checkVFClamp(0);
	(void)ra.hasRegVF(1);
	(void)ra.hasRegVI(1);
	(void)ra.getFreeXmmCount();
	(void)ra.getFreeGPRCount();
}

// Force the pass-1 analysis bodies (aVU_Analyze.inl) to be compiled. They are not
// reached by any live code yet (the per-op pass1 handlers that call them are task
// 7.5+), so without an explicit odr-use these inline functions would be parsed but
// their full instantiation/codegen — and any porting error — could go undetected.
// Never called: it dereferences mVU.prog.cur (null here), it exists only to compile.
[[maybe_unused]] static void mVUanalyzeCompileCheck()
{
	microVU& mVU = microVU0;
	mVUanalyzeFMAC1(mVU, 1, 2, 3);
	mVUanalyzeFMAC2(mVU, 1, 2);
	mVUanalyzeFMAC3(mVU, 1, 2, 3);
	mVUanalyzeFMAC4(mVU, 1, 2);
	mVUanalyzeIALU1(mVU, 1, 2, 3);
	mVUanalyzeIALU2(mVU, 1, 2);
	mVUanalyzeIADDI(mVU, 1, 2, (s16)3);
	mVUanalyzeMR32(mVU, 1, 2);
	mVUanalyzeFDIV(mVU, 1, 0, 2, 1, 3);
	mVUanalyzeEFU1(mVU, 1, 0, 3);
	mVUanalyzeEFU2(mVU, 1, 3);
	mVUanalyzeMFP(mVU, 1);
	mVUanalyzeMOVE(mVU, 1, 2);
	mVUanalyzeLQ(mVU, 1, 2, true);
	mVUanalyzeSQ(mVU, 1, 2, true);
	mVUanalyzeR1(mVU, 1, 0);
	mVUanalyzeR2(mVU, 1, true);
	mVUanalyzeSflag(mVU, 1);
	mVUanalyzeFSSET(mVU);
	mVUanalyzeMflag(mVU, 1, 2);
	mVUanalyzeCflag(mVU, 1);
	mVUanalyzeXGkick(mVU, 1, 2);
	mVUanalyzeCondBranch1(mVU, 1);
	mVUanalyzeCondBranch2(mVU, 1, 2);
	mVUanalyzeNormBranch(mVU, 1, true);
	mVUanalyzeJump(mVU, 1, 2, true);
}

// Force the pass-1 pipeline/cycle/range helpers (above) to be compiled. The
// non-inline ones (mVUsetupRange/mVUoptimizePipeState/mVUincCycles/cmpVFregs/
// mVUsetCycles) have external linkage so they codegen regardless, but their
// driver (mVUcompile) is task 7.4; until it lands this odr-use is the only thing
// proving the bodies + the inline helpers compile. Never called: it touches
// mVU.prog.cur (null here), it exists only to compile.
[[maybe_unused]] static void mVUcompileHelpersCheck()
{
	microVU& mVU = microVU0;
	mVUcheckIsSame(mVU);
	mVUsetupRange(mVU, 0, true);
	mVUcheckBadOp(mVU);
	branchWarning(mVU);
	int branch = 0;
	eBitPass1(mVU, branch);
	eBitWarning(mVU);
	mVUoptimizePipeState(mVU);
	mVUincCycles(mVU, 1);
	mVUsetCycles(mVU);
	microVFreg vfa{}, vfb{};
	bool xVar = false;
	cmpVFregs(vfa, vfb, xVar);
	(void)optimizeReg(0);
	(void)calcCycles(0, 0);
	(void)tCycles(0, 0);
	incP(mVU);
	incQ(mVU);
	startLoop(mVU);
	mVUinitConstValues(mVU);
	mVUinitFirstPass(mVU, (uptr)&mVU.prog.lpState, nullptr);
}

// Force the Pass-2 flag allocators (aVU_Alloc.inl) to be compiled. These are the
// first emit-backend helpers that actually call VIXL; their drivers
// (mVUsetupFlags/mVUdivSet, the Branch/Lower opcode handlers) are later 7.5
// slices, so without an explicit odr-use their VIXL bodies — and any emission
// error — would never be instantiated. Never called: it emits into no open
// assembler, it exists only to type-check + codegen the bodies.
[[maybe_unused]] static void mVUallocFlagCheck()
{
	microVU& mVU = microVU0;
	(void)getFlagReg(0);
	setBitSFLAG(gprT1, gprT2, 0x0f00, 0x0001);
	setBitFSEQ(gprT1, 0x0f00);
	mVUallocSFLAGa(gprT1, 1);
	mVUallocSFLAGb(gprT1, 1);
	mVUallocSFLAGc(gprT1, gprT2, 1);
	mVUallocSFLAGd(mVU, &mVU.divFlag, gprT1, gprT2, getFlagReg(0));
	mVUallocMFLAGa(mVU, gprT1, 1);
	mVUallocMFLAGb(mVU, gprT1, 1);
	mVUallocCFLAGa(mVU, gprT1, 1);
	mVUallocCFLAGb(mVU, gprT1, 1);

	const a64::VRegister t = a64::VRegister(0, 128);
	getPreg(mVU, t);       // uses mVUinfo.readP (mVU in scope)
	getQreg(t, 0);
	writeQreg(t, 1);
}

// Force the clamp helpers (aVU_Clamp.inl) to be compiled. Same rationale as
// mVUallocFlagCheck: they emit VIXL but their callers (the FMAC arithmetic
// opcode handlers) are a later 7.5 slice, so without an explicit odr-use their
// bodies — and any emission error — would never be instantiated. Never called.
[[maybe_unused]] static void mVUclampCheck()
{
	microVU& mVU = microVU0;
	const a64::VRegister reg = a64::VRegister(1, 128);
	const a64::VRegister tmp = a64::VRegister(2, 128);
	mVUclamp1(mVU, reg, tmp, 0xf);
	mVUclamp2(mVU, reg, tmp, 0x8);
	mVUclamp3(mVU, reg, tmp, 0xf);
	mVUclamp4(mVU, reg, tmp, 0x8);
}

// Force the misc emit helpers (aVU_Misc.inl) to be compiled. Never called.
[[maybe_unused]] static void mVUmiscCheck()
{
	microVU& mVU = microVU0;
	mVUaddrFix(mVU, a64::x9, a64::x10);
}

// (mVUcompileEmitCheck / mVUtablesCheck removed: the compile helpers + the
// mVUopU/mVUopL dispatchers now have live callers via mVUcompile, so their bodies
// codegen without an explicit odr-use.)

// Force the flag pipeline (aVU_Flags.inl) to be compiled. The analysis functions
// (findFlagInst/sortFlag/sortFullFlag/mVUstatusFlagOp/mVUsetFlags) have external
// linkage and codegen regardless, but the emit helpers (mVUdivSet/mVUsetupFlags)
// are __fi and have no live caller until the Branch/Compile slice — without this
// odr-use their VIXL bodies, and any emission error, would never be instantiated.
// Never called: it touches mVU.prog.cur (null here), it exists only to compile.
[[maybe_unused]] static void mVUflagCheck()
{
	microVU& mVU = microVU0;
	microFlagCycles mFC{};
	mVUdivSet(mVU);
	mVUsetupFlags(mVU, mFC);
	mVUsetFlags(mVU, mFC);
	mVUstatusFlagOp(mVU);
	(void)findFlagInst(mFC.xStatus, 0);
	int bFlag[4];
	(void)sortFlag(mFC.xStatus, bFlag, 0);
	sortFullFlag(mFC.xStatus, bFlag);
}

// Force the branch/program-exit emitters (aVU_Branch.inl) to be compiled. The two
// mVUendProgram/mVUDTendProgram emitters are the only non-static, never-yet-called
// emit functions in this slice; odr-use them so their VIXL bodies (and any
// emission error) are instantiated now. Never called.
[[maybe_unused]] static void mVUbranchCheck()
{
	microVU& mVU = microVU0;
	microFlagCycles mFC{};
	(void)getLastFlagInst(mVUregs, mFC.xStatus, 0, 0);
	mVUendProgram(mVU, &mFC, 1);
	mVUDTendProgram(mVU, &mFC, 1);
	mVUsetupBranch(mVU, mFC);
}

//------------------------------------------------------------------
// HH-37: microVU1's AOT form (aot_arm64.cpp says what the AOT form is)
//------------------------------------------------------------------
// In the AOT form (mvuAot: Shape, Emit and Load) a microVU1 block is a function of its start pc,
// its incoming pipeline state (all 96 bytes of microRegInfo), the micro-memory words its compile
// read (curI records them, mvuReadMask), this build and its settings (the image header's
// layout_hash and abi_hash), and of nothing else:
//   - One compile is one block. No successor is compiled inline or jumped to directly: every
//     block it can go to it leaves for through a link stub (vu1AotEmitLink): w0 = the successor's
//     pc, x1 = the address of a literal in this block holding the state it enters with; a
//     direct-mapped link cache (keyed by that literal's address and the pc, emptied by mVUclear)
//     gives the successor's code, and on a miss vu1AotLinkRT finds it (vu1AotFind). A JR/JALR's
//     link stub also sets start_pc, as mVUcompileJIT does.
//   - The states the stock form passes by heap pointer (the block's own, for an early exit's
//     copyPLState, and its end state) are literals at the block's end (vu1AotLit), reached by adr.
//   - Every host address goes through the import table in x28 (armAotRoute, AsmHelpers.h):
//     microVU1, the VU registers, settings, tables and helpers as regions; VU1's data and micro
//     memory as run-time regions; the dispatchers and stubs as value imports.
//   - Blocks are found by exact match of (pc, state), then by their words against micro memory
//     (vu1AotFind), never by the program search, whose answer depends on what was compiled
//     before; so Shape compiles, and Load installs, the same block for the same execution.
//   - The compile writes nothing a block runs on (mVUinitFirstPass leaves lpState alone) and
//     starts from zeroed IR state (setBranchA reads the IR of the op before a block's first).
// The dispatchers and helper stubs are one more record (VU1_STUB_PC), in their shaped form: the
// dispatcher takes the import table as its third argument (rec_vu1_execute).
// Shape logs each block it compiles; Emit re-compiles each record from its words into plain
// memory (vu1AotEmit); Load runs the image's, and a block it lacks stops the process.

namespace
{
	enum Vu1Stub { StubStart, StubExit, StubStartXG, StubExitXG, StubExactMul, StubWaitMTVU, StubCopyPL, StubCompare, StubCount };
	const char* const kVu1StubImport[StubCount] = {"vu1.start", "vu1.exit", "vu1.startXG", "vu1.exitXG", "vu1.exactMul",
		"vu1.waitMTVU", "vu1.copyPLState", "vu1.compareState"};
	u8** vu1StubField(microVU& mVU, int i)
	{
		u8** const f[StubCount] = {&mVU.startFunct, &mVU.exitFunct, &mVU.startFunctXG, &mVU.exitFunctXG, &mVU.exactMulStub,
			&mVU.waitMTVU, &mVU.copyPLState, &mVU.compareStateF};
		return f[i];
	}

	struct Vu1Entry
	{
		std::vector<u32> runs; // the words it was compiled from (the image format's runs)
		const u8* code;
	};
	struct Vu1Key
	{
		u32 pc;
		u32 st[aot::VU1_STATE_WORDS];
		bool operator==(const Vu1Key& o) const { return pc == o.pc && !std::memcmp(st, o.st, sizeof st); }
	};
	struct Vu1KeyHash
	{
		size_t operator()(const Vu1Key& k) const
		{
			u64 h = 1469598103934665603ULL ^ k.pc;
			for (u32 w : k.st)
				h = (h ^ w) * 1099511628211ULL;
			return (size_t)h;
		}
	};
	// The blocks of a key, and which of them matches micro memory now: valid while gen is the
	// current generation, which mVUclear advances.
	struct Vu1Node
	{
		std::vector<Vu1Entry> entries;
		u64 gen = 0;
		u32 valid = 0;
	};
	std::unordered_map<Vu1Key, Vu1Node, Vu1KeyHash> s_vu1Map;
	u64 s_vu1Gen = 1;

	// The link cache the link stubs read: slot ((lit >> 4) ^ (pc >> 3)) & (kLinkSlots - 1)
	// holds key lit | pc << 48 and the code it links to.
	struct Vu1LinkSlot
	{
		u64 key;
		const void* code;
	};
	static_assert(sizeof(Vu1LinkSlot) == 16, "the link stub indexes 16-byte slots");
	constexpr u32 kLinkSlots = 8192;
	Vu1LinkSlot* s_vu1Links;
	bool s_vu1LinksDirty;

	microProgram* s_vu1Prog; // a compile's program: its ranges only (no block manager is used)
	struct Vu1Lit
	{
		a64::Label label;
		microRegInfo st;
	};
	std::deque<Vu1Lit> s_vu1Lits; // the block being compiled's state literals
} // namespace

static a64::Label* vu1AotLit(const microRegInfo& st)
{
	for (Vu1Lit& l : s_vu1Lits)
		if (!std::memcmp(&l.st, &st, sizeof st))
			return &l.label;
	s_vu1Lits.emplace_back();
	std::memcpy(&s_vu1Lits.back().st, &st, sizeof st);
	return &s_vu1Lits.back().label;
}

// The block's literals, after its code: 16-aligned, each the 96 bytes of a state
static void vu1AotEmitLits()
{
	for (Vu1Lit& l : s_vu1Lits)
	{
		while (armAsm->GetCursorOffset() & 15)
		{
			a64::ExactAssemblyScope scope(armAsm, 4);
			armAsm->dc32(0);
		}
		armAsm->Bind(&l.label);
		a64::ExactAssemblyScope scope(armAsm, sizeof(microRegInfo));
		for (u32 w : l.st.full32)
			armAsm->dc32(w);
	}
	s_vu1Lits.clear();
}

static void* vu1AotLinkRT(u32 pc, const microRegInfo* st);

// A link to the block at pcImm (or, at -1, the pc in w0: a jump's, which also becomes start_pc)
// entered with state st. Every register but x19, x27, x28, the flag instances (w23-w26) and PQ
// (v24) is free here: the branch drivers have flushed the allocator (mVUsetupBranch), as the
// stock form's jump to a block's start requires.
static void vu1AotEmitLink(microVU& mVU, const microRegInfo& st, int pcImm, bool jr)
{
	using namespace a64;
	Label* lit = vu1AotLit(st);
	if (pcImm >= 0)
		armAsm->Mov(w0, (u32)pcImm);
	else
	{
		if (jr)
			armAsm->Str(w0, MemOperand(RVUSTATE, offsetof(VURegs, start_pc)));
		armAsm->And(w0, w0, mVU.microMemSize - 8);
	}
	armAsm->Adr(x1, lit);
	armAsm->Ldr(x2, MemOperand(x28, aot::slotOfValue("vu1.links") * 8));
	armAsm->Lsr(x3, x1, 4);
	armAsm->Eor(x3, x3, Operand(x0, LSR, 3));
	armAsm->And(x3, x3, kLinkSlots - 1);
	armAsm->Add(x2, x2, Operand(x3, LSL, 4));
	armAsm->Orr(x3, x1, Operand(x0, LSL, 48));
	armAsm->Ldp(x4, x5, MemOperand(x2));
	armAsm->Cmp(x4, x3);
	Label slow;
	armAsm->B(&slow, ne);
	armAsm->Br(x5);
	armAsm->Bind(&slow);
	armAsm->Str(mVU_xmmPQ.Q(), mvuAbsMem(mVU, &mVU.vecBackup[mVU_xmmPQ.GetCode()][0], 16));
	armEmitCall(reinterpret_cast<const void*>(&vu1AotLinkRT));
	armAsm->Ldr(mVU_xmmPQ.Q(), mvuAbsMem(mVU, &mVU.vecBackup[mVU_xmmPQ.GetCode()][0], 16));
	armAsm->Br(x0);
}

// The runs of the words a compile read (the mask), with their values in micro memory
static std::vector<u32> vu1AotRuns(const u64* mask, const u32* micro)
{
	std::vector<u32> runs;
	for (u32 i = 0; i < 4096;)
	{
		if (!((mask[i >> 6] >> (i & 63)) & 1))
		{
			i++;
			continue;
		}
		u32 n = 0;
		while (i + n < 4096 && ((mask[(i + n) >> 6] >> ((i + n) & 63)) & 1))
			n++;
		runs.push_back((i << 16) | n);
		runs.insert(runs.end(), micro + i, micro + i + n);
		i += n;
	}
	return runs;
}

static bool vu1AotRunsHold(const u32* runs, size_t n, const u32* micro)
{
	for (size_t i = 0; i < n;)
	{
		const u32 start = runs[i] >> 16, count = runs[i] & 0xffff;
		if (std::memcmp(micro + start, runs + i + 1, count * 4))
			return false;
		i += 1 + count;
	}
	return true;
}

// One block, into the open session at its cursor (Shape: the cache; Emit: plain memory)
static u8* vu1AotCompileBody(microVU& mVU, u32 pc, const microRegInfo& st, std::vector<u32>* runs)
{
	if (!s_vu1Prog)
	{
		s_vu1Prog = (microProgram*)_aligned_malloc(sizeof(microProgram), 64);
		std::memset(s_vu1Prog, 0, sizeof(microProgram));
		s_vu1Prog->ranges = new std::deque<microRange>();
	}
	s_vu1Prog->ranges->clear();
	mVU.prog.cur = s_vu1Prog;
	std::memset(&mVU.prog.IRinfo, 0, sizeof(mVU.prog.IRinfo));
	s_vu1Lits.clear();
	u64 mask[4096 / 64] = {};
	microRegInfo state;
	std::memcpy(&state, &st, sizeof state);
	mvuReadMask = mask;
	armAotRoute = true;
	u8* const start = armGetCurrentCodePointer();
	mVUcompile(mVU, pc, (uptr)&state);
	vu1AotEmitLits();
	armAotRoute = false;
	mvuReadMask = nullptr;
	*runs = vu1AotRuns(mask, (const u32*)mVU.regs().Micro);
	return start;
}

static void vu1AotLog(u32 pc, const u32* words, size_t nwords, u32 native, const u8* code, u32 len)
{
	aot::Block b;
	b.unit = aot::UnitVU1;
	b.pc = pc;
	b.native = native;
	b.words.assign(words, words + nwords);
	b.codeLen = len;
	aot::logBlock(b, code);
}

// Shape: compile the block for (pc, st) into the cache, log it, and give its code
static const u8* vu1AotCompile(microVU& mVU, u32 pc, const microRegInfo& st, std::vector<u32>* runs)
{
	armSetAsmPtr(mVU.prog.codePtr, mVU.prog.codeReserveEnd - mVU.prog.codePtr, nullptr);
	armStartBlock();
	u8* const code = vu1AotCompileBody(mVU, pc, st, runs);
	mVU.prog.codePtr = armEndBlock();
	std::vector<u32> words(st.full32, st.full32 + aot::VU1_STATE_WORDS);
	words.insert(words.end(), runs->begin(), runs->end());
	vu1AotLog(pc, words.data(), words.size(), aot::VU1_STATE_WORDS, code, (u32)(mVU.prog.codePtr - code));
	return code;
}

// The code of the block for (pc, st) whose words micro memory holds now: the node's, the
// image's (Load), or compiled (Shape). Load stops the process on a block the image lacks.
static void* vu1AotFind(microVU& mVU, u32 pc, const microRegInfo* st)
{
	Vu1Key k;
	k.pc = pc;
	std::memcpy(k.st, st, sizeof k.st);
	auto [it, fresh] = s_vu1Map.try_emplace(k);
	Vu1Node& n = it->second;
	if (n.gen == s_vu1Gen)
		return const_cast<u8*>(n.entries[n.valid].code);
	if (fresh && aot::imageOnly())
		if (const std::vector<const aot::Block*>* v = aot::imageBlocks(aot::UnitVU1, pc))
			for (const aot::Block* b : *v)
				if (b->native == aot::VU1_STATE_WORDS && !std::memcmp(b->words.data(), k.st, sizeof k.st))
				{
					n.entries.push_back({std::vector<u32>(b->words.begin() + aot::VU1_STATE_WORDS, b->words.end()), b->code});
					aot::countInstall(aot::UnitVU1);
				}
	const u32* micro = (const u32*)mVU.regs().Micro;
	for (u32 i = 0; i < n.entries.size(); i++)
		if (vu1AotRunsHold(n.entries[i].runs.data(), n.entries[i].runs.size(), micro))
		{
			n.gen = s_vu1Gen;
			n.valid = i;
			return const_cast<u8*>(n.entries[i].code);
		}
	if (aot::imageOnly())
	{
		aot::miss(aot::UnitVU1, pc); // Load: stops here
		std::abort();
	}
	std::vector<u32> runs;
	microRegInfo state; // (st may be a literal in the cache the compile writes to)
	std::memcpy(&state, k.st, sizeof state);
	const u8* code = vu1AotCompile(mVU, pc, state, &runs);
	n.entries.push_back({std::move(runs), code});
	n.gen = s_vu1Gen;
	n.valid = (u32)n.entries.size() - 1;
	return const_cast<u8*>(code);
}

// A link stub's miss (w0 the pc, x1 its state literal): find the block and fill the slot
static void* vu1AotLinkRT(u32 pc, const microRegInfo* st)
{
	void* code = vu1AotFind(microVU1, pc, st);
	const uintptr_t lit = reinterpret_cast<uintptr_t>(st);
	Vu1LinkSlot& slot = s_vu1Links[((lit >> 4) ^ (pc >> 3)) & (kLinkSlots - 1)];
	slot.key = lit | ((u64)pc << 48);
	slot.code = code;
	s_vu1LinksDirty = true;
	return code;
}

static void vu1AotClear()
{
	s_vu1Gen++;
	if (s_vu1LinksDirty && s_vu1Links)
		std::memset(s_vu1Links, 0, kLinkSlots * sizeof(Vu1LinkSlot));
	s_vu1LinksDirty = false;
}

static void vu1AotForget()
{
	s_vu1Map.clear();
	vu1AotClear();
}

// The stubs in their shaped form, at the open session's cursor: their entries' offsets in offs
static void vu1AotGenStubs(microVU& mVU, u32 offs[StubCount])
{
	armAotRoute = true;
	u8* const start = armGetCurrentCodePointer();
	mVUdispatcherAB(mVU);
	mVUdispatcherCD(mVU);
	mVUemitExactMulStub(mVU);
	mVUGenerateWaitMTVU(mVU);
	mVUGenerateCopyPipelineState(mVU);
	mVUGenerateCompareState(mVU);
	armAotRoute = false;
	for (int i = 0; i < StubCount; i++)
		offs[i] = (u32)(*vu1StubField(mVU, i) - start);
}

static void vu1AotSetImports(microVU& mVU)
{
	if (!s_vu1Links)
		s_vu1Links = (Vu1LinkSlot*)calloc(kLinkSlots, sizeof(Vu1LinkSlot));
	aot::setValueImport("vu1.links", s_vu1Links);
	aot::setValueImport("vu1.Mem", mVU.regs().Mem);
	aot::setValueImport("vu1.Micro", mVU.regs().Micro);
	for (int i = 0; i < StubCount; i++)
		aot::setValueImport(kVu1StubImport[i], *vu1StubField(mVU, i));
	armAotRedCall = aot::redVu1HostAddr() ? reinterpret_cast<const void*>(&mVU1clearlpStateJIT) : nullptr;
}

// mVUreset's part in the AOT form: no blocks; the stubs compiled (Shape, logged) or the image's
static void vu1AotReset(microVU& mVU)
{
	vu1AotForget();
	std::memset(mVU.mscalMemo, 0, (mVU.progSize / 2) * sizeof(microMscalMemoA));
	if (aot::imageOnly())
	{
		const std::vector<const aot::Block*>* v = aot::imageBlocks(aot::UnitVU1, aot::VU1_STUB_PC);
		const aot::Block* b = v ? (*v)[0] : nullptr;
		if (!b || b->words.size() != StubCount)
		{
			std::fprintf(stderr, "lrps2 AOT load: the image's VU1 stubs have %zu entries, this core %d: stopping\n", b ? b->words.size() : 0, (int)StubCount);
			std::fflush(stderr);
			std::_Exit(5);
		}
		for (int i = 0; i < StubCount; i++)
			*vu1StubField(mVU, i) = const_cast<u8*>(b->code) + b->words[i];
		mVU.prog.codeStart = mVU.prog.codePtr = nullptr;
	}
	else
	{
		armSetAsmPtr(mVU.cache, static_cast<size_t>(mVU.prog.codeReserveEnd - mVU.cache), nullptr);
		u8* const start = armStartBlock();
		u32 offs[StubCount];
		vu1AotGenStubs(mVU, offs);
		u8* const end = armEndBlock();
		vu1AotLog(aot::VU1_STUB_PC, offs, StubCount, StubCount, start, (u32)(end - start));
		mVU.prog.codeStart = mVU.prog.codePtr =
			reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(end) + 15u) & ~uintptr_t(15));
	}
	vu1AotSetImports(mVU);
}

// Emit: re-compile every logged VU1 record from its words, into plain memory, at one offset past
// the stubs (where nothing but the block's own words, state and this build decides its bytes).
// LRPS2_AOT_CORRUPT_VU1 names a pc whose records are emitted from a changed word (the first upper
// ADD made a SUB, or MUL an ADD): the red receipt for "the image's code is what runs".
void vu1AotEmit()
{
	microVU& mVU = microVU1;
	const std::vector<aot::Block>& recs = aot::emitRecords(aot::UnitVU1);
	if (recs.empty())
		return;
	constexpr size_t kBuf = 8u << 20;
	u8* const buf = static_cast<u8*>(mmap(nullptr, kBuf, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	static u32 micro[4096];
	alignas(16) static u8 mem[0x4000];
	u8* const savedMicro = mVU.regs().Micro;
	u8* const savedMem = mVU.regs().Mem;
	mVU.regs().Micro = reinterpret_cast<u8*>(micro);
	mVU.regs().Mem = mem;
	armAotDirect = true;

	armSetAsmPtr(buf, kBuf, nullptr);
	u8* const stubs = armStartBlock();
	u32 offs[StubCount];
	vu1AotGenStubs(mVU, offs);
	const u32 stubLen = (u32)(armEndBlock() - stubs);
	vu1AotSetImports(mVU); // the stubs' entries, here, and the run-time regions, the stand-ins above
	const size_t blockOff = (stubLen + 4095) & ~size_t(4095);

	u32 corruptPC;
	const bool corrupt = aot::emitCorruptVu1(&corruptPC);
	unsigned long corrupted = 0;
	for (const aot::Block& r : recs)
	{
		if (r.pc == aot::VU1_STUB_PC)
		{
			if (r.words.size() != StubCount || std::memcmp(r.words.data(), offs, sizeof offs))
				aot::emitFailed(r, "the stubs' entries are not where the shape run's were");
			else
				aot::emitted(r, stubs, stubLen, {});
			continue;
		}
		if (r.native != aot::VU1_STATE_WORDS || r.words.size() < aot::VU1_STATE_WORDS)
		{
			aot::emitFailed(r, "a VU1 record without its state");
			continue;
		}
		std::memset(micro, 0, sizeof micro);
		bool ok = true;
		for (size_t i = aot::VU1_STATE_WORDS; i < r.words.size() && ok;)
		{
			const u32 start = r.words[i] >> 16, count = r.words[i] & 0xffff;
			ok = count && start + count <= 4096 && i + 1 + count <= r.words.size();
			if (ok)
				std::memcpy(micro + start, &r.words[i + 1], count * 4);
			i += 1 + count;
		}
		if (!ok)
		{
			aot::emitFailed(r, "a VU1 record whose runs overrun micro memory or the record");
			continue;
		}
		const bool corruptThis = corrupt && r.pc == corruptPC;
		if (corruptThis)
		{
			for (u32 q = r.pc / 4 + 1; q < 4096; q += 2) // upper ops are the odd words
			{
				const u32 op = micro[q] & 0x3f;
				if ((micro[q] & 0x7e000000u) == 0 && (op == 0x28 || op == 0x2a))
				{
					micro[q] = (micro[q] & ~0x3fu) | (op == 0x28 ? 0x2c : 0x28);
					corrupted++;
					break;
				}
			}
		}
		microRegInfo st;
		std::memcpy(&st, r.words.data(), sizeof st);
		armSetAsmPtr(buf + blockOff, kBuf - blockOff, nullptr);
		armStartBlock();
		std::vector<u32> runs;
		u8* const code = vu1AotCompileBody(mVU, r.pc, st, &runs);
		const u32 len = (u32)(armEndBlock() - code);
		if (!corruptThis && (runs.size() != r.words.size() - aot::VU1_STATE_WORDS ||
								 std::memcmp(runs.data(), r.words.data() + aot::VU1_STATE_WORDS, runs.size() * 4)))
			aot::emitFailed(r, "it reads other micro words than the shape run's compile did");
		else
			aot::emitted(r, code, len, {});
	}
	if (corrupt)
		std::fprintf(stderr, "lrps2 AOT emit: %lu VU1 records at %04x re-emitted from a changed word, on purpose\n", corrupted, corruptPC);
	armAotDirect = false;
	mVU.regs().Micro = savedMicro;
	mVU.regs().Mem = savedMem;
	munmap(buf, kBuf);
}

#ifdef LRPS2_AOT_FORM
// Everything a shaped microVU1 block or stub reaches that is not off x19 or x27, in slot order.
void aot::vu1Imports(ImportAdder add)
{
#define AOT_FN(f) add(#f, reinterpret_cast<const void*>(&f), 1)
	AOT_FN(mVUexecuteVU1); AOT_FN(mVUcleanUpVU1); AOT_FN(mVU1clearlpStateJIT); AOT_FN(mVUTBit); AOT_FN(mVUEBit);
	AOT_FN(mVUwaitMTVU); AOT_FN(aVUexactDivC); AOT_FN(_vuXGKICKTransfermVU); AOT_FN(mVU_XGKICK_); AOT_FN(mVUbadOp1);
	AOT_FN(vu1AotLinkRT);
#undef AOT_FN
	add("microVU1", &microVU1, sizeof(microVU1));
	add("EmuConfig.Cpu", &EmuConfig.Cpu, sizeof(EmuConfig.Cpu));
	add("eeHw", eeHw, sizeof(eeHw));
	add("mVUmovemaskBit", mVUmovemaskBit, sizeof(mVUmovemaskBit));
	add("mVU_sse4_compvals", mVU_sse4_compvals, sizeof(mVU_sse4_compvals));
	add("sse4_minvals", sse4_minvals, sizeof(sse4_minvals));
	add("sse4_maxvals", sse4_maxvals, sizeof(sse4_maxvals));
	add("mVU_MIN_MAX_1", mVU_MIN_MAX_1, sizeof(mVU_MIN_MAX_1));
	add("mVU_MIN_MAX_2", mVU_MIN_MAX_2, sizeof(mVU_MIN_MAX_2));
	add("vu1.Mem", nullptr, 0x4000);
	add("vu1.Micro", nullptr, 0x4000);
	add("vu1.links", nullptr, 0);
	for (const char* name : kVu1StubImport)
		add(name, nullptr, 0);
}

// What microVU1's code addresses off x19 (VURegs), x27 (microVU) and its import regions, by name,
// offset and size: a layout that differs refuses the image.
u64 aot::vu1LayoutHash()
{
	u64 h = 1469598103934665603ULL;
	const auto mix = [&h](const char* name, size_t off, size_t size) {
		for (const char* c = name; *c; c++) { h ^= (u8)*c; h *= 1099511628211ULL; }
		const u64 w[2] = {off, size};
		for (size_t i = 0; i < sizeof w; i++) { h ^= ((const u8*)w)[i]; h *= 1099511628211ULL; }
	};
	const VURegs& r = vuRegs[1];
	const microVU& m = microVU1;
#define AOT_R(f) mix("VURegs." #f, (uintptr_t)&r.f - (uintptr_t)&r, sizeof(r.f))
#define AOT_V(f) mix("microVU." #f, (uintptr_t)&m.f - (uintptr_t)&m, sizeof(m.f))
	AOT_R(VF); AOT_R(VI); AOT_R(ACC); AOT_R(q); AOT_R(p); AOT_R(pending_q); AOT_R(pending_p); AOT_R(micro_macflags);
	AOT_R(micro_clipflags); AOT_R(micro_statusflags); AOT_R(flags); AOT_R(cycle); AOT_R(nextBlockCycles); AOT_R(start_pc);
	AOT_R(Mem); AOT_R(Micro); AOT_R(xgkickenable); AOT_R(xgkickcyclecount); AOT_R(xgkickaddr);
	AOT_V(statFlag); AOT_V(macFlag); AOT_V(clipFlag); AOT_V(vecCTemp); AOT_V(exactMulBuf); AOT_V(exactDivBuf);
	AOT_V(exactMulSave); AOT_V(vecBackup); AOT_V(glob); AOT_V(sse4Min); AOT_V(sse4Max); AOT_V(code); AOT_V(divFlag);
	AOT_V(VIbackup); AOT_V(VIxgkick); AOT_V(branch); AOT_V(badBranch); AOT_V(evilBranch); AOT_V(evilevilBranch);
	AOT_V(p); AOT_V(q); AOT_V(totalCycles); AOT_V(cycles); AOT_V(prog.lpState); AOT_V(resumePtrXG);
#undef AOT_R
#undef AOT_V
	mix("sizeof", sizeof(VURegs), sizeof(microVU));
	mix("microRegInfo", sizeof(microRegInfo), alignof(microRegInfo));
	mix("VIFregisters.top/itop", offsetof(VIFregisters, top), offsetof(VIFregisters, itop));
	mix("vif1Regs", (uintptr_t)&vif1Regs - (uintptr_t)eeHw, sizeof(VIFregisters));
	mix("EmuConfig.Cpu FPCRs", (uintptr_t)&EmuConfig.Cpu.VU1FPCR - (uintptr_t)&EmuConfig.Cpu,
		(uintptr_t)&EmuConfig.Cpu.FPUFPCR - (uintptr_t)&EmuConfig.Cpu);
	return h;
}

// The ABI of microVU1's code and stubs, and every setting its compile reads
const char* aot::vu1AbiText()
{
	static char text[512];
	snprintf(text, sizeof text,
		"vu1: dispatcher w0 pc w1 cycles x2 imports, xg x0 imports; x19 vuRegs[1]; x27 microVU1; x28 imports; "
		"link w0 pc x1 state literal (96, 16-aligned), %u slots of 16 keyed lit|pc<<48 at ((lit>>4)^(pc>>3)); "
		"PQ v24 saved at vecBackup; jr sets start_pc; exact state match; stubs %d; "
		"flaghack %d ovf %d sign %d extra %d exactmul %d exactdiv %d accaddsub %d addsubhack %d xgkickhack %d "
		"ibit %d vusync %d fullvu0sync %d mtvu %d fpcr %d",
		kLinkSlots, (int)StubCount, (int)CHECK_VU_FLAGHACK, (int)CHECK_VU_OVERFLOW(1), (int)CHECK_VU_SIGN_OVERFLOW(1),
		(int)CHECK_VU_EXTRA_OVERFLOW(1), (int)CHECK_VU_EXACTMUL, (int)CHECK_VU_EXACTDIV, (int)CHECK_VU_ACC_ADDSUB,
		(int)CHECK_VUADDSUBHACK, (int)CHECK_XGKICKHACK, (int)EmuConfig.Gamefixes.IbitHack, (int)EmuConfig.Gamefixes.VUSyncHack,
		(int)EmuConfig.Gamefixes.FullVU0SyncHack, (int)THREAD_VU1, (int)mvuNeedsFPCRUpdate(microVU1));
	return text;
}
#endif
