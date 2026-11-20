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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The AOT form exists for the Apple arm64 hosts, the Mac that makes images and the devices that
// run them; everywhere else the stubs below keep the stock recompilers. AOT_TOOLING is the Mac's
// part (Shape, Emit, the red receipts, the twin's self-test, the VM report): a device core carries
// Load and Interp alone.
#if defined(__APPLE__) && defined(__aarch64__)
#define LRPS2_AOT_FORM 1
#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE
#define AOT_TOOLING 1
#endif
#endif

// An invariant of the AOT form, checked in every build (the tree's pxAssert is a no-op): a failed
// one stops the process, exit 70, saying which.
#define AOT_CHECK(cond, ...)                         \
	do                                               \
	{                                                \
		if (!(cond)) [[unlikely]]                    \
		{                                            \
			std::fprintf(stderr, "lrps2 AOT: ");     \
			std::fprintf(stderr, __VA_ARGS__);       \
			std::fprintf(stderr, "\n");              \
			std::fflush(stderr);                     \
			std::_Exit(70);                          \
		}                                            \
	} while (0)

namespace aot
{
	enum Mode
	{
		Off,    // the stock recompilers
		Shape,  // the JIT, emitting AOT-shaped blocks, each logged
		Emit,   // re-emit a shape log's blocks into an image, then exit
		Load,   // run from the image; a block it lacks stops the process
		Interp, // run from the image; a block it lacks runs on the EE's C twin, the IOP's
		        // interpreter or the VIF's C path; VU1 runs on its interpreter (unitShaped)
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
	// The EE's and the IOP's blocks: at most this many key words, which is also the mask's reach.
	constexpr u32 GUEST_WORDS_MAX = 96;
	static_assert(GUEST_WORDS_MAX <= 32 * 3, "Block::mask holds a bit per key word");
	// microVU1's records: a block's `words` start with its incoming pipeline state (`native` of
	// them), then runs of the micro words its compile read; the one record at VU1_STUB_PC is its
	// dispatchers and helper stubs, whose `words` are their entries' code offsets (aVU.cpp, Vu1Stub).
	constexpr u32 VU1_STATE_WORDS = 24, VU1_STUB_PC = 0xffffffffu, VU1_STUB_ENTRIES = 8, VU1_MICRO_WORDS = 4096;
	// Whether n words of runs (each a word (first index << 16 | count) and that many words) lie
	// inside micro memory and inside the n words
	inline bool vu1RunsValid(const u32* runs, size_t n)
	{
		for (size_t i = 0; i < n;)
		{
			const u32 start = runs[i] >> 16, count = runs[i] & 0xffff;
			if (!count || start + count > VU1_MICRO_WORDS || i + 1 + count > n)
				return false;
			i += 1 + count;
		}
		return true;
	}

	// An image block's fastmem site (absolute code address) and its stub. pc: the access's; block:
	// its block's.
	struct Site
	{
		uintptr_t code, stub;
		u32 pc, block;
	};
	// The settings EE code depends on, in the header's ee_flags.
	constexpr u32 EE_FLAG_WAITLOOP = 1, EE_FLAG_FPU_SOFT = 2;

	// The no-fastmem mask of a block at pc of n key words: bit i where offFastmem(pc + 4i).
	template <typename F>
	inline void faultMask(u32 mask[3], u32 pc, size_t n, F offFastmem)
	{
		mask[0] = mask[1] = mask[2] = 0;
		for (u32 i = 0; i < n && i < GUEST_WORDS_MAX; i++)
			if (offFastmem(pc + i * 4))
				mask[i >> 5] |= 1u << (i & 31);
	}

	// What a compile of a guest unit (the EE's, the IOP's) reads. Every guest word CompileBlock
	// reads goes through read(), which records it in a shaped compile: the words from the block's
	// pc up to the last one read are its key (Block::words), whatever decided its extent. In Emit
	// the record being re-compiled (`rec`) stands in for guest memory, and a read outside its words
	// sets `oob`, failing the record, rather than read memory the shape run's block never saw.
	struct CodeReader
	{
		u32 pc = 0;
		std::vector<u32> words;
		const Block* rec = nullptr;
		bool oob = false;
		void start(u32 at)
		{
			pc = at;
			words.clear();
		}
		template <typename Mem>
		u32 read(u32 a, bool shaped, Mem mem)
		{
			const u32 idx = (a - pc) >> 2;
			u32 w;
			if (rec)
			{
				if (a < pc || idx >= rec->words.size())
				{
					oob = true;
					return 0;
				}
				w = rec->words[idx];
			}
			else
				w = mem(a);
			if (shaped)
			{
				AOT_CHECK(a >= pc && (a & 3) == 0 && idx < GUEST_WORDS_MAX,
					"the compile at %08x read %08x, which is no key word of it", pc, a);
				if (idx >= words.size())
					words.resize(idx + 1, 0);
				words[idx] = w;
			}
			return w;
		}
	};

	// Emit's buffer, which every unit compiles into and none runs (plain memory)
	constexpr size_t EMIT_BUFFER_BYTES = 32u << 20;

#ifdef LRPS2_AOT_FORM
	namespace detail
	{
		extern Mode mode;
		extern void** imports;
	} // namespace detail

	// From retro_load_game, before the CPU thread reserves the caches: the mode the next VM runs
	// in (the LRPS2_AOT* environment, which wins, or the frontend's `rt_aot_image` option, which
	// runs that image in Interp).
	void configure(const char* rt_aot_image_option);
	inline Mode mode() { return detail::mode; }
	// THE predicate of the AOT form: every recompiler that runs (Shape) or is loaded (Load,
	// Interp) emits or runs code that holds no host address and is never patched.
	inline bool shaped() { return mode() != Off; }
	// No code cache: the blocks are the image's.
	inline bool imageOnly() { return mode() == Load || mode() == Interp; }
	// Whether a recompiler's code cache may be mapped (armJitMap asks with its name): in an AOT
	// mode only Shape's shaped recompilers get one; every other runs its interpreter.
	bool jitAllowed(const char* what);
	// Whether this unit's recompiler runs in the AOT form: every unit in Shape, Emit and Load; in
	// Interp all but microVU1, which has no C twin to miss to and runs on its interpreter.
	inline bool unitShaped(Unit u) { return mode() != Off && (u != UnitVU1 || mode() != Interp); }

	// The import table, which a shaped block reaches through a register (EE: x20, IOP and VU1:
	// x28), and the slot of a thing it reaches, with `*off` its offset in the import's region.
	inline void** imports() { return detail::imports; }
	using ImportAdder = void (*)(const char* name, const void* ptr, size_t size);
	// The slot of a value import (size 0: its table entry is set at run time, setValueImport).
	u32 slotOfValue(const char* name);
	// Each unit lists its imports once, in a fixed order (aot_arm64.cpp's unit table).
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
	// fnv64 of n bytes at p, continuing h (from FNV_BASIS): the one mix every hash of the AOT form uses.
	constexpr u64 FNV_BASIS = 1469598103934665603ULL;
	u64 fnv(u64 h, const void* p, size_t n);
	// A layout hash's entry: a member's name, offset and size.
	inline u64 fnvMember(u64 h, const char* name, size_t off, size_t size)
	{
		const u64 w[2] = {off, size};
		return fnv(fnv(h, name, std::char_traits<char>::length(name)), w, sizeof w);
	}

	// VMManager::UpdateCPUImplementations, once per VM: Emit emits and exits; Load and Interp open
	// the image and check it against this core, game and settings (false: refused, the
	// interpreters run).
	bool admit();
	bool admitted();
	// VMManager::LoadSettings, whenever the settings change in a VM (the game's fixes, an option):
	// Shape logs the settings its later blocks are compiled under; Load and Interp check the image
	// against them again, refusing one they no longer match (or admitting one they match again).
	// True when that decision changed, so the CPU providers must be chosen again.
	bool recheck();
	// Whether this unit's blocks come from an admitted image now (Load, Interp).
	inline bool fromImage(Unit u) { return imageOnly() && unitShaped(u) && admitted(); }
	// What the core did with the image, once per decision: true while one waits for the frontend
	// (retro_run hands it over on the frontend's thread).
	bool takeReport(int& admitted, std::string& detail);

	// Shape: log a compiled block (its code at `code`).
	void logBlock(const Block& b, const u8* code);
	// Load: the image's blocks at a unit's pc, in image order.
	const std::vector<const Block*>* imageBlocks(Unit u, u32 pc);
	// Load: where the image's code starts (null while no image is open).
	const u8* imageCodeBase();
	// Load: every image block's fastmem site, sorted by code address.
	const std::vector<Site>& imageSites();
	// Load: a block the image lacks. Load stops the process; Interp counts it (the caller runs it).
	void miss(Unit u, u32 pc);
	void countInstall(Unit u);
	// A fastmem fault a shaped block's access took (resumed at its stub).
	void countFault(Unit u);
	// An unpack that wraps VU memory, which runs on the kernels (or, with a mode, on the C table)
	void countVifWrap(bool kernels);
	// LRPS2_AOT_RED=<the unit's word> (the Mac's tooling): the unit's red receipt for "a shaped
	// block holds no host address", which materializes one of the unit's host addresses as the
	// stock form does (the VIF's: all three of its own), so Emit's comparison must find every block
	// that has it. hostaddr (EE: cpuBlockCycles), iophostaddr (IOP: ICFG), vu1hostaddr (VU1:
	// mVU1clearlpStateJIT), vifhostaddr (both VIF units: MaskRow, MaskCol, nVifMask).
	bool red(Unit u);

	// Emit: the unit's logged blocks, for its emitter to re-compile, and its corrupt pc (a red
	// receipt: that block is emitted otherwise than its record says, its true key kept in the
	// meta; LRPS2_AOT_CORRUPT=pc names an EE block, iop:pc an IOP one, vu1:pc microVU1's blocks at
	// that pc, vif:pc a VIF routine and vifk:index a VIF kernel; false when none of this unit's).
	const std::vector<Block>& emitRecords(Unit u);
	bool emitCorrupt(Unit u, u32* pc);
	// The buffer every unit's Emit compiles into (EMIT_BUFFER_BYTES, never unmapped: Emit exits).
	u8* emitBuffer();
	// An emitter's result for one record: its code as this process compiled it.
	void emitted(const Block& b, const u8* code, u32 len, const std::vector<u32>& sites);
	void emitFailed(const Block& b, const char* why);
	// Emit for a guest unit (EE, IOP): re-compile each logged block from its words (`compile`
	// compiles the block at pc into the emit buffer, reading through `reader`, and gives its key),
	// and hand its code to emitted. The corrupt block is compiled from a changed word (its first
	// ADDIU/ORI immediate one higher).
	void emitGuest(Unit u, CodeReader& reader, const Block& (*compile)(u32 pc));
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
	inline bool recheck() { return false; }
	inline bool fromImage(Unit) { return false; }
	inline bool takeReport(int&, std::string&) { return false; }
	inline u32 slotOfValue(const char*) { return 0; }
	inline void setValueImport(const char*, void*) {}
	inline void logBlock(const Block&, const u8*) {}
	inline const std::vector<const Block*>* imageBlocks(Unit, u32) { return nullptr; }
	inline const u8* imageCodeBase() { return nullptr; }
	inline const std::vector<Site>& imageSites() { static const std::vector<Site> none; return none; }
	inline void miss(Unit, u32) {}
	inline void countInstall(Unit) {}
	inline void countFault(Unit) {}
	inline void countVifWrap(bool) {}
	inline bool red(Unit) { return false; }
	inline u8* emitBuffer() { return nullptr; }
#endif
} // namespace aot
