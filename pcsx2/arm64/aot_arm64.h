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
		UnitCount,
	};

	// A logged block (Shape's log, Emit's input) and an image block (Load's): the guest address it
	// was compiled at, its key flags, the guest words it was compiled from (`words`; the first
	// `native` of them are what its code covers, the rest what decided where it ends), a mask of
	// the words whose memory access was kept off fastmem, its code, and the code offsets of its
	// fastmem sites, their stubs and the index in `words` of the access's instruction (triples).
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

	// The import table, which a shaped block reaches through a register (EE: x20), and the slot
	// of a thing it reaches, with `*off` its offset in the import's region.
	void** imports();
	using ImportAdder = void (*)(const char* name, const void* ptr, size_t size);
	// The slot of a value import (size 0: its table entry is set at run time, setValueImport).
	u32 slotOfValue(const char* name);
	// Each unit lists its imports once, in a fixed order (aot_arm64.cpp's buildImports).
	void eeImports(ImportAdder add);
	u32 slotOf(const void* p, u32* off);
	// recR5900_arm64.cpp: the guest state EE code addresses off a base (the header's layout_hash)
	// and the text of what its code and this C++ share (abi_hash).
	u64 eeLayoutHash();
	const char* eeAbiText();
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
	// Load: every image block's fastmem site (absolute code address) and its stub, sorted.
	struct Site { uintptr_t code, stub; u32 pc, block; }; // pc: the access's; block: its block's
	const std::vector<Site>& imageSites();
	// Load: a block the image lacks. Load stops the process; Interp counts it (the caller runs it).
	void miss(Unit u, u32 pc);
	void countInstall(Unit u);

	// Emit: the unit's logged blocks, for its emitter to re-compile, and the corrupt pc (a red
	// receipt: that block is emitted from a changed word, its true words kept in the meta).
	const std::vector<Block>& emitRecords(Unit u);
	u32 emitCorrupt();
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
	struct Site { uintptr_t code, stub; u32 pc, block; };
	inline const std::vector<Site>& imageSites() { static const std::vector<Site> none; return none; }
	inline void miss(Unit, u32) {}
	inline void countInstall(Unit) {}
	inline const std::vector<Block>& emitRecords(Unit) { static const std::vector<Block> none; return none; }
	inline u32 emitCorrupt() { return 0; }
	inline void emitted(const Block&, const u8*, u32, const std::vector<u32>&) {}
	inline void emitFailed(const Block&, const char*) {}
	constexpr u32 EE_FLAG_WAITLOOP = 1, EE_FLAG_FPU_SOFT = 2;
#endif
} // namespace aot
