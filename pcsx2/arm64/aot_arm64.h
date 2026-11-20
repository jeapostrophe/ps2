// SPDX-FileCopyrightText: lrps2 arm64 port
// SPDX-License-Identifier: LGPL-3.0+
//
// retro-trainer PLAN-handheld HH-23/HH-37: the AOT form of the arm64 recompilers, which runs a
// game's blocks from an image compiled on the Mac, on a host that maps no executable memory.
// aot_arm64.cpp says what each mode does and holds the image format.

#pragma once

#include "common/Pcsx2Types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The AOT form exists for the Apple arm64 hosts, the Mac that makes images and the devices that
// run them; everywhere else the stubs below keep the stock recompilers.
#if defined(__APPLE__) && defined(__aarch64__)
#define LRPS2_AOT_FORM 1
#endif

namespace aot
{
	enum Mode
	{
		Off,    // the stock recompilers
		Shape,  // the JIT, emitting AOT-shaped blocks, each logged
		Emit,   // re-emit a shape log's blocks into an image, then exit
		Load,   // run from the image; a block it lacks stops the process
		Interp, // run from the image; a block it lacks runs on its recompiler's C twin
	};

	// The recompilers an image can hold blocks of, in the meta's `unit` field.
	enum Unit : u32
	{
		UnitEE = 0,
		UnitIOP = 1,
		UnitVU1 = 2,
		UnitVIF = 3,       // the VIF unpack dynarec's routines (Vif_Dynarec_arm64.cpp)
		UnitVIFUnpack = 4, // the VIF's 256 simple unpack kernels, nVifUpk (Vif_UnpackNEON.cpp)
		UnitCount,
	};

	// A logged block (Shape's log, Emit's input) and an image block (Load's): the guest address it
	// was compiled at, its key flags, the guest words it was compiled from (`words`; the first
	// `native` of them are what its code covers, the rest what decided where it ends), a mask of
	// the words whose memory access was kept off fastmem, its code, and the code offsets of its
	// fastmem sites, their stubs and the index in `words` of the access's instruction (triples).
	// A unit may give `words` a payload of its own (the image format's comment, aot_arm64.cpp):
	// microVU1's is its incoming pipeline state and the micro-memory words its compile read.
	struct Block
	{
		u32 unit = 0, pc = 0, flags = 0, native = 0;
		u32 mask[3] = {};
		std::vector<u32> words;
		u32 codeOff = 0, codeLen = 0;
		u64 hash = 0;
		std::vector<u32> sites; // site_off, stub_off, word index, ...
		const u8* code = nullptr; // Load: in the image
	};
	// microVU1's records: a block's `words` start with its incoming pipeline state (`native` of
	// them); the one record at VU1_STUB_PC is its dispatchers and helper stubs, whose `words` are
	// their entries' code offsets (aVU.cpp, Vu1Stub).
	constexpr u32 VU1_STATE_WORDS = 24, VU1_STUB_PC = 0xffffffffu;

#ifdef LRPS2_AOT_FORM
	// From retro_load_game, before the CPU thread reserves the caches: the mode the next VM runs
	// in (the LRPS2_AOT* environment, which wins, or the frontend's `rt_aot_image` option, which
	// runs that image in Interp).
	void configure(const char* rt_aot_image_option);
	Mode mode();
	// THE predicate of the AOT form: every recompiler that runs (Shape) or is loaded (Load,
	// Interp) emits or runs code that holds no host address and is never patched.
	inline bool shaped() { return mode() != Off; }
	// No code cache: the blocks are the image's.
	inline bool imageOnly() { return mode() == Load || mode() == Interp; }
	// Whether a recompiler's code cache may be mapped (armJitMap asks with its name): in an AOT
	// mode only Shape's shaped recompilers get one; every other runs its interpreter.
	bool jitAllowed(const char* what);
	// Whether this unit's recompiler runs in the AOT form (it is in the image).
	bool unitShaped(Unit u);

	// The import table, which a shaped block reaches through a register (EE: x20, IOP: x28), and the slot
	// of a thing it reaches, with `*off` its offset in the import's region.
	void** imports();
	using ImportAdder = void (*)(const char* name, const void* ptr, size_t size);
	// The slot of a value import (size 0: its table entry is set at run time, setValueImport).
	u32 slotOfValue(const char* name);
	// Each unit lists its imports once, in a fixed order (aot_arm64.cpp's buildImports).
	void eeImports(ImportAdder add);
	void iopImports(ImportAdder add);
	// aVU.cpp: microVU1's, after the EE's. A region with no pointer and a size is a region whose
	// start is set at run time (setValueImport), e.g. VU1's data memory.
	void vu1Imports(ImportAdder add);
	// The slot of what holds p: a region (static, or one whose start was set), else a value import
	// whose value is p (off 0).
	u32 slotOf(const void* p, u32* off);
	// recR5900_arm64.cpp, recR3000A_arm64.cpp, aVU.cpp (microVU1) and Vif_Dynarec_arm64.cpp (both
	// VIF units): the guest state each unit's code addresses off a base (the header's layout_hash
	// covers all) and the text of what its code and this C++ share (abi_hash, all).
	u64 eeLayoutHash();
	const char* eeAbiText();
	u64 iopLayoutHash();
	const char* iopAbiText();
	u64 vu1LayoutHash();
	const char* vu1AbiText();
	u64 vifLayoutHash();
	const char* vifAbiText();
	// An import whose VALUE is set at run time (a pointer the core maps, e.g. the EE's block LUT).
	void setValueImport(const char* name, void* value);

	// VMManager::UpdateCPUImplementations, once per VM: Emit emits and exits; Load and Interp open
	// the image and check it against this core and game (false: refused, the interpreters run).
	bool admit();
	bool admitted();
	// What the core did with the image, once per decision: true while one waits for the frontend
	// (retro_run hands it over on the frontend's thread).
	bool takeReport(int& admitted, std::string& detail);

	// Shape: log a compiled block (its code at `code`).
	void logBlock(const Block& b, const u8* code);
	// Load: the image's blocks at a unit's pc, in image order.
	const std::vector<const Block*>* imageBlocks(Unit u, u32 pc);
	// Load: where the image's code starts (null while no image is open).
	const u8* imageCodeBase();
	// Load: every image block's fastmem site (absolute code address) and its stub, sorted.
	struct Site { uintptr_t code, stub; u32 pc, block; }; // pc: the access's; block: its block's
	const std::vector<Site>& imageSites();
	// Load: a block the image lacks. Load stops the process; Interp counts it (the caller runs it).
	void miss(Unit u, u32 pc);
	void countInstall(Unit u);
	// A fastmem fault a shaped block's access took (resumed at its stub).
	void countFault(Unit u);
	// An unpack that wraps VU memory, which runs on the kernels (or, with a mode, on the C table)
	void countVifWrap(bool kernels);
	// LRPS2_AOT_RED=hostaddr (the Mac's tooling): one EE helper's address is materialized into
	// the code as the stock form does, so Emit's comparison must find every block that has it
	// (the red receipt for "a shaped block holds no host address").
	bool redHostAddr();
	// LRPS2_AOT_RED=iophostaddr: the same for one IOP datum (the IOP's red receipt).
	bool redIopHostAddr();
	// LRPS2_AOT_RED=vu1hostaddr: the same for one microVU1 helper (mVU1clearlpStateJIT).
	bool redVu1HostAddr();
	// LRPS2_AOT_RED=vifhostaddr: the VIF's code reaches MaskRow, MaskCol and nVifMask at their
	// addresses, as the stock form does (the VIF's red receipt).
	bool redVifHostAddr();

	// Emit: the unit's logged blocks, for its emitter to re-compile, and the unit's corrupt pc (a
	// red receipt: that block is emitted otherwise than its record says, its true key kept in the
	// meta; LRPS2_AOT_CORRUPT=pc names an EE block, iop:pc an IOP one, vif:pc a VIF routine and
	// vifk:index a VIF kernel; 0 when none of this unit's).
	const std::vector<Block>& emitRecords(Unit u);
	u32 emitCorrupt(Unit u);
	// LRPS2_AOT_CORRUPT_VU1=<pc>: every microVU1 record at that pc is emitted from a changed word.
	bool emitCorruptVu1(u32* pc);
	// An emitter's result for one record: its code as this process compiled it.
	void emitted(const Block& b, const u8* code, u32 len, const std::vector<u32>& sites);
	void emitFailed(const Block& b, const char* why);

	// The settings EE code depends on, in the header's ee_flags.
	constexpr u32 EE_FLAG_WAITLOOP = 1, EE_FLAG_FPU_SOFT = 2;
#else
	inline void configure(const char*) {}
	inline Mode mode() { return Off; }
	inline bool shaped() { return false; }
	inline bool imageOnly() { return false; }
	inline bool jitAllowed(const char*) { return true; }
	inline bool unitShaped(Unit) { return false; }
	inline void** imports() { return nullptr; }
	inline u32 slotOf(const void*, u32* off) { *off = 0; return 0; }
	inline bool admit() { return true; }
	inline bool admitted() { return false; }
	inline bool takeReport(int&, std::string&) { return false; }
	inline u32 slotOfValue(const char*) { return 0; }
	inline void setValueImport(const char*, void*) {}
	inline void logBlock(const Block&, const u8*) {}
	inline const std::vector<const Block*>* imageBlocks(Unit, u32) { return nullptr; }
	inline const u8* imageCodeBase() { return nullptr; }
	struct Site { uintptr_t code, stub; u32 pc, block; };
	inline const std::vector<Site>& imageSites() { static const std::vector<Site> none; return none; }
	inline void miss(Unit, u32) {}
	inline void countInstall(Unit) {}
	inline void countFault(Unit) {}
	inline void countVifWrap(bool) {}
	inline bool redHostAddr() { return false; }
	inline bool redIopHostAddr() { return false; }
	inline bool redVu1HostAddr() { return false; }
	inline bool emitCorruptVu1(u32*) { return false; }
	inline bool redVifHostAddr() { return false; }
	inline const std::vector<Block>& emitRecords(Unit) { static const std::vector<Block> none; return none; }
	inline u32 emitCorrupt(Unit) { return 0; }
	inline void emitted(const Block&, const u8*, u32, const std::vector<u32>&) {}
	inline void emitFailed(const Block&, const char*) {}
	constexpr u32 EE_FLAG_WAITLOOP = 1, EE_FLAG_FPU_SOFT = 2;
#endif
} // namespace aot
