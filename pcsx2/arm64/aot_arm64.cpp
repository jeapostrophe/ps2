// SPDX-FileCopyrightText: lrps2 arm64 port
// SPDX-License-Identifier: LGPL-3.0+
//
// retro-trainer PLAN-handheld HH-23/HH-37: the AOT form of the arm64 recompilers. aot::configure
// (libretro/main.cpp's retro_load_game: the LRPS2_AOT* environment, or the frontend's
// `rt_aot_image` option) chooses a mode, in force for the next VM:
//
// - Shape (the Mac's tooling): the JIT, emitting AOT-shaped blocks (each recompiler, wherever it
//   tests aot::shaped()). Such a block holds no host address: everything it calls or reads that
//   is not reached off a guest-register base reaches it through the import table (the EE: x20,
//   which the block's prologue takes from its argument); it chains to another block only through
//   the block lookup table, whose base is an import; and nothing ever patches it (a fastmem fault
//   resumes at the access's stub through PageFaultInfo::resume_pc). So its bytes are a function
//   of its guest address, the guest words it was compiled from, its key flags and this build, and
//   it runs anywhere. Every compiled block is logged (logBlock). Only the shaped recompilers get a
//   code cache (jitAllowed): the others, and microVU0 whatever the mode, run their interpreters,
//   so a Shape run and an image's run differ in nothing but where the shaped code lives. The
//   shaped recompilers are the EE's and microVU1's (aVU.cpp, "microVU1's AOT form").
// - Emit: re-compiles each distinct logged block from its logged words into plain memory, checks
//   its code against the JIT's (length and fnv64: the two processes load the core at different
//   slides, so a host address left in a block would differ), writes `image`.code and
//   `image`.meta, and exits.
// - Load: no code cache; a block is installed from the image where the JIT would compile one,
//   when the image holds a block for that address whose words guest memory holds now; a block
//   the image lacks stops the process.
// - Interp: Load, but a block the image lacks runs on its recompiler's C twin (the device's mode).
//   microVU1 has no C twin yet: in Interp, VU1 runs on its interpreter (unitShaped).
//
// admit(), from VMManager::UpdateCPUImplementations, opens the image in Load and Interp and
// refuses one made for another core build or another game (checkHeader): the interpreters run
// the game instead, and the frontend hears of it (takeReport).

#include "arm64/aot_arm64.h"

#ifdef LRPS2_AOT_FORM

#include "CDVD/CDVD.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mutex>
#include <TargetConditionals.h>
#include <unistd.h>
#include <unordered_map>

#if !TARGET_OS_IPHONE
// The Mac's tooling: Shape, Emit and the VM report. A device core carries Load and Interp alone.
#define AOT_TOOLING
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

void eeAotEmit(); // recR5900_arm64.cpp
void vu1AotEmit(); // aVU.cpp
u32 eeAotFastmemMode();
u32 eeAotFlags();

namespace aot
{
#define AOT_CHECK(cond, ...) do { if (!(cond)) { \
		fprintf(stderr, "lrps2 AOT: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
		fflush(stderr); _Exit(70); } } while (0)

	/*
	 * **The image format: this is its one home** (retro-trainer's cores/README.md, "AOT images",
	 * says how an image is made, keyed and shipped). Its meta, little-endian, is this header (the
	 * same 104 bytes, magic and version as mupen64plus_next's, flycast's and Dolphin's, so
	 * retro-trainer's tools read all four), then per block
	 *   u32 unit, pc, flags, native, nwords, mask[3], code_off, code_len, nsites, reserved,
	 *       words[nwords], sites[3 * nsites] (site_off and stub_off within the block's code, and
	 *       the index in words of the guest instruction whose access it is)
	 * and its code is each block's at its code_off (BlockAlign apart). What `words` holds is the
	 * unit's:
	 *   - UnitEE: the guest words from pc on (at most 96); `native` of them are the block's.
	 *   - UnitVU1, a block: its incoming pipeline state (microRegInfo, native = VU1_STATE_WORDS =
	 *     24 words), then the micro-memory words its compile read, as runs: a word
	 *     (first index << 16 | count) and that many words (indices are of 32-bit micro words,
	 *     below 4096). No mask, no sites.
	 *   - UnitVU1 at VU1_STUB_PC (one record): the dispatchers and helper stubs every VU1 block
	 *     calls or leaves through; `words` (native = nwords) are their entries' offsets in its code,
	 *     in aVU.cpp's Vu1Stub order.
	 * A load refuses an image unless each checked field is this core's (checkHeader names the
	 * first that differs); layout_hash and abi_hash cover microVU1's bases, settings and stubs.
	 */
	struct ImageHeader
	{
		u32 magic;        // MAGIC
		u32 version;      // VERSION, which fixes everything else here, the header's size too
		u32 import_count; // the import table's entries
		u32 nblocks;
		u8 core_uuid[16]; // the LC_UUID of the core build that may load it (Emit writes the one it
		                  // is told, LRPS2_AOT_CORE_UUID, or its own)
		u8 emit_uuid[16]; // the LC_UUID of the build that emitted it: not checked here;
		                  // retro-trainer's `cores.sh aot pin` requires the Mac build its lock pins
		u8 rom_md5[16];   // the game: a 128-bit fnv of the disc's serial and its boot ELF's CRC (a
		                  // block's words are checked where it is installed, but its timing and
		                  // neighbours are the game's, so another game is refused here)
		u64 imports_hash; // fnv64 of the import table's names and region sizes, in order
		u64 layout_hash;  // fnv64 of the guest-state members the code addresses off a base, by
		                  // name, offset and size (LAYOUT)
		u64 abi_hash;     // fnv64 of what the code and this core's C++ share (ABI_TEXT)
		u32 fastmem;      // the EE's fastmem mode at its first compile (shared readers: count_per_op)
		u32 ee_flags;     // the settings EE code depends on, EE_FLAG_* (count_per_op_denom_pot)
		u32 code_size;    // the bytes between rt_aot_code and rt_aot_code_end
		u32 reserved;     // padding to the header's 8-byte alignment: written 0, never read
	};
	static_assert(sizeof(ImageHeader) == 104, "the header's size is part of VERSION");
	constexpr u32 MAGIC = 0x31415452; // "RTA1"
	constexpr u32 VERSION = 3;
	constexpr u32 BlockAlign = 16;
	constexpr u32 BlockWords = 12; // a meta block record's fixed words
	constexpr u32 BLOCK_FLAGS_KNOWN = 0; // no block flag is defined yet: every block's is 0

	/**** configuration ****/

	static struct
	{
		Mode mode = Off;
		std::string image, log;
		u32 corrupt = 0;
		bool hasCorruptVu1 = false;
		u32 corruptVu1 = 0;
		bool hasCoreUuid = false;
		u8 coreUuid[16]{};
	} cfg;

	static struct
	{
		unsigned long compiled[UnitCount], installs[UnitCount], misses[UnitCount], faults[UnitCount], emitDiffers, emitSkipped;
	} stats;

	static bool parseHex16(const char* s, u8 out[16])
	{
		int n = 0;
		memset(out, 0, 16);
		for (; *s && n < 32; s++)
		{
			if (*s == '-')
				continue;
			const int v = *s >= '0' && *s <= '9' ? *s - '0' : *s >= 'a' && *s <= 'f' ? *s - 'a' + 10 : *s >= 'A' && *s <= 'F' ? *s - 'A' + 10 : -1;
			if (v < 0)
				return false;
			out[n / 2] |= (u8)(n & 1 ? v : v << 4);
			n++;
		}
		return n == 32 && !*s;
	}

	static void buildImports();
	static void releaseImage();
	enum Admission { Unknown, Admitted, Refused };
	static Admission admission = Unknown;

	void configure(const char* option)
	{
		const char* env = getenv("LRPS2_AOT");
		Mode m = Off;
		std::string image;
		if (env && *env)
		{
			if (!strcmp(env, "load"))
				m = Load;
			else if (!strcmp(env, "interp"))
				m = Interp;
#ifdef AOT_TOOLING
			else if (!strcmp(env, "shape"))
				m = Shape;
			else if (!strcmp(env, "emit"))
				m = Emit;
			else
				AOT_CHECK(false, "LRPS2_AOT=%s is none of load, interp, shape, emit", env);
#else
			else
				AOT_CHECK(false, "LRPS2_AOT=%s is neither load nor interp, the only ones a device core has", env);
#endif
			const char* img = getenv("LRPS2_AOT_IMAGE");
			image = img ? img : "";
		}
		else if (option && *option)
		{
			m = Interp;
			image = option;
		}
		releaseImage();
		admission = Unknown;
		cfg.mode = m;
		cfg.image = image;
		const char* log = getenv("LRPS2_AOT_LOG");
		cfg.log = (env && *env && log) ? log : "";
		const char* corrupt = getenv("LRPS2_AOT_CORRUPT");
		cfg.corrupt = (env && *env && corrupt && *corrupt) ? (u32)strtoul(corrupt, nullptr, 16) : 0;
		const char* corruptVu1 = getenv("LRPS2_AOT_CORRUPT_VU1");
		cfg.hasCorruptVu1 = env && *env && corruptVu1 && *corruptVu1;
		cfg.corruptVu1 = cfg.hasCorruptVu1 ? (u32)strtoul(corruptVu1, nullptr, 16) : 0;
		const char* uuid = getenv("LRPS2_AOT_CORE_UUID");
		cfg.hasCoreUuid = env && *env && uuid && *uuid;
		AOT_CHECK(!cfg.hasCoreUuid || parseHex16(uuid, cfg.coreUuid), "LRPS2_AOT_CORE_UUID=%s is not a uuid", uuid);
		if (m != Off)
		{
			buildImports();
			fprintf(stderr, "lrps2 AOT: mode %s%s%s\n", env && *env ? env : "interp (rt_aot_image)",
				cfg.image.empty() ? "" : ", image ", cfg.image.c_str());
		}
	}

	Mode mode() { return cfg.mode; }

	bool jitAllowed(const char* what)
	{
		if (cfg.mode == Off)
			return true;
		// Shape: the shaped recompilers and the vixl self-test that comes up with them; every other
		// recompiler (and microVU0, which the AOT form never runs) on its interpreter, as on a device
		if (cfg.mode == Shape)
			return !strcmp(what, "EE") || !strcmp(what, "microVU1") || !strcmp(what, "self-test");
		return false;
	}

	// microVU1 in Shape, Emit and Load; in Interp, which has no C twin for it to miss to, VU1 runs
	// on its interpreter
	bool unitShaped(Unit u) { return cfg.mode != Off && (u == UnitEE || cfg.mode != Interp); }

	/**** the import table ****/

	// A region import (start, size), a value import (size 0: its table entry is set at run time),
	// or a run-time region (`dynamic`: a size, and a start set at run time with its entry).
	struct Import { std::string name; uintptr_t start; size_t size; bool dynamic; };
	static std::vector<Import> importList;     // in slot order
	static std::vector<void*> importTable;
	static std::vector<u32> regionsByStart;     // slots of the static regions, by start
	static std::vector<u32> dynamicRegions;     // slots of the run-time regions
	static std::unordered_map<std::string, u32> slotByName;

	static void addImport(const char* name, const void* ptr, size_t size)
	{
		AOT_CHECK(!slotByName.count(name), "the import %s is listed twice", name);
		AOT_CHECK(size || !ptr, "the import %s has a pointer and no size (a value import has neither)", name);
		slotByName[name] = (u32)importList.size();
		importList.push_back({name, reinterpret_cast<uintptr_t>(ptr), size, size && !ptr});
		importTable.push_back(const_cast<void*>(ptr));
	}

	static void buildImports()
	{
		if (!importList.empty())
			return;
		eeImports(addImport);
		vu1Imports(addImport);
		for (u32 s = 0; s < importList.size(); s++)
			if (importList[s].dynamic)
				dynamicRegions.push_back(s);
			else if (importList[s].size)
				regionsByStart.push_back(s);
		std::sort(regionsByStart.begin(), regionsByStart.end(), [](u32 a, u32 b) { return importList[a].start < importList[b].start; });
		for (size_t i = 1; i < regionsByStart.size(); i++)
		{
			const Import& a = importList[regionsByStart[i - 1]];
			const Import& b = importList[regionsByStart[i]];
			AOT_CHECK(a.start + a.size <= b.start, "the imports %s and %s overlap", a.name.c_str(), b.name.c_str());
		}
		// an unsigned LDR's offset reaches 4095 slots
		AOT_CHECK(importTable.size() <= 4096, "%zu imports, more than an LDR off one register reaches", importTable.size());
	}

	void** imports() { return importTable.data(); }

	// What p is, for the message of a block that reaches something no import holds
	static std::string nameOf(const void* p)
	{
		Dl_info dl;
		char s[200];
		if (dladdr(p, &dl) && dl.dli_sname)
			snprintf(s, sizeof s, " (%s+%#lx)", dl.dli_sname, (unsigned long)((uintptr_t)p - (uintptr_t)dl.dli_saddr));
		else
			s[0] = 0;
		return s;
	}

	u32 slotOf(const void* p, u32* off)
	{
		const uintptr_t a = reinterpret_cast<uintptr_t>(p);
		auto it = std::upper_bound(regionsByStart.begin(), regionsByStart.end(), a, [](uintptr_t v, u32 s) { return v < importList[s].start; });
		if (it != regionsByStart.begin() && a < importList[*(it - 1)].start + importList[*(it - 1)].size)
		{
			*off = (u32)(a - importList[*(it - 1)].start);
			return *(it - 1);
		}
		for (u32 s : dynamicRegions)
			if (importList[s].start && a - importList[s].start < importList[s].size)
			{
				*off = (u32)(a - importList[s].start);
				return s;
			}
		for (u32 s = 0; s < importList.size(); s++)
			if (!importList[s].size && importTable[s] == p && p)
			{
				*off = 0;
				return s;
			}
		AOT_CHECK(false, "an AOT-shaped block reaches %p%s, which no import holds", p, nameOf(p).c_str());
		return 0;
	}

	u32 slotOfValue(const char* name)
	{
		auto it = slotByName.find(name);
		AOT_CHECK(it != slotByName.end() && (importList[it->second].size == 0 || importList[it->second].dynamic), "%s is no value import", name);
		return it->second;
	}

	void setValueImport(const char* name, void* value)
	{
		auto it = slotByName.find(name);
		AOT_CHECK(it != slotByName.end() && (importList[it->second].size == 0 || importList[it->second].dynamic), "%s is no value import", name);
		importTable[it->second] = value;
		if (importList[it->second].dynamic)
			importList[it->second].start = reinterpret_cast<uintptr_t>(value);
	}

	static u64 fnv(u64 h, const void* p, size_t n)
	{
		const u8* b = (const u8*)p;
		while (n--)
		{
			h ^= *b++;
			h *= 1099511628211ULL;
		}
		return h;
	}
	constexpr u64 FNV_BASIS = 1469598103934665603ULL;

	static u64 importsHash()
	{
		u64 h = FNV_BASIS;
		for (const Import& im : importList)
		{
			h = fnv(h, im.name.c_str(), im.name.size() + 1);
			const u64 sz = im.size;
			h = fnv(h, &sz, sizeof sz);
		}
		return h;
	}

	/**** the header ****/



	static bool selfUuid(u8 out[16])
	{
		Dl_info info;
		if (!dladdr((void*)&configure, &info) || info.dli_fbase == nullptr)
			return false;
		const mach_header_64* mh = (const mach_header_64*)info.dli_fbase;
		if (mh->magic != MH_MAGIC_64)
			return false;
		const load_command* lc = (const load_command*)(mh + 1);
		for (u32 i = 0; i < mh->ncmds; i++, lc = (const load_command*)((const u8*)lc + lc->cmdsize))
			if (lc->cmd == LC_UUID)
			{
				memcpy(out, ((const uuid_command*)lc)->uuid, 16);
				return true;
			}
		return false;
	}

	// The game: the disc's serial and boot path from its SYSTEM.CNF (cdvdPeekDiscIdentity, which
	// reads them with no other effect: the boot itself reads them later), taken once per VM, at
	// admission. A revision with the serial of another is told apart by the words every block is
	// installed under, not here.
	static std::string contentText;
	static bool contentDigest(u8 out[16])
	{
		if (contentText.empty())
			return false;
		const u64 a = fnv(FNV_BASIS, contentText.data(), contentText.size());
		const u64 b = fnv(0x6c62272e07bb0142ULL, contentText.data(), contentText.size());
		memcpy(out, &a, 8);
		memcpy(out + 8, &b, 8);
		return true;
	}
	static void readContent()
	{
		std::string serial, boot;
		contentText = cdvdPeekDiscIdentity(serial, boot) ? serial + "|" + boot : std::string();
	}

	enum { HEADER_NO_UUID = 1, HEADER_NO_DIGEST = 2 };

	static int makeHeader(ImageHeader& h, u32 nblocks, u32 codeSize, const u8* coreUuid)
	{
		memset(&h, 0, sizeof h);
		h.magic = MAGIC;
		h.version = VERSION;
		h.import_count = (u32)importTable.size();
		h.nblocks = nblocks;
		int missing = selfUuid(h.emit_uuid) ? 0 : HEADER_NO_UUID;
		memcpy(h.core_uuid, coreUuid != nullptr ? coreUuid : h.emit_uuid, 16);
		if (!contentDigest(h.rom_md5))
			missing |= HEADER_NO_DIGEST;
		h.imports_hash = importsHash();
		const u64 vl = vu1LayoutHash();
		h.layout_hash = fnv(eeLayoutHash(), &vl, sizeof vl);
		const char* abi = eeAbiText();
		const char* vabi = vu1AbiText();
		h.abi_hash = fnv(fnv(fnv(FNV_BASIS, abi, strlen(abi) + 1), vabi, strlen(vabi) + 1), &BlockAlign, sizeof BlockAlign);
		h.fastmem = eeAotFastmemMode();
		h.ee_flags = eeAotFlags();
		h.code_size = codeSize;
		return missing;
	}

	static std::string formatUuid(const u8 u[16])
	{
		char s[40];
		snprintf(s, sizeof s, "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
			u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
		return s;
	}

	// Why this core may not load an image with header h (the first field that differs, with both
	// values); empty when it may.
	static std::string checkHeader(const ImageHeader& h)
	{
		ImageHeader w;
		const int missing = makeHeader(w, h.nblocks, h.code_size, nullptr);
		char why[256];
#define AOT_FIELD(f, fmt) if (h.f != w.f) { snprintf(why, sizeof why, #f ": the image has " fmt ", this core " fmt, h.f, w.f); return why; }
#define AOT_BYTES(f, what) if (memcmp(h.f, w.f, 16)) { \
		return std::string(#f ": the image is for ") + what + " " + formatUuid(h.f) + ", this is " + formatUuid(w.f); }
		AOT_FIELD(magic, "%#x")
		AOT_FIELD(version, "%u")
		if (missing & HEADER_NO_UUID)
			return "core_uuid: this core has no LC_UUID to match";
		AOT_BYTES(core_uuid, "core")
		if (missing & HEADER_NO_DIGEST)
			return "rom_md5: this game has no serial or CRC to match";
		AOT_BYTES(rom_md5, "the game whose digest is")
		AOT_FIELD(import_count, "%u")
		AOT_FIELD(imports_hash, "%016llx")
		AOT_FIELD(layout_hash, "%016llx")
		AOT_FIELD(abi_hash, "%016llx")
		AOT_FIELD(fastmem, "%u")
		AOT_FIELD(ee_flags, "%#x")
#undef AOT_FIELD
#undef AOT_BYTES
		return "";
	}

	/**** the frontend's report ****/

	static std::mutex reportMutex;
	static bool reportPending;
	static int reportAdmitted;
	static std::string reportDetail;

	static void report(bool ok, const std::string& detail)
	{
		std::lock_guard<std::mutex> lock(reportMutex);
		reportPending = true;
		reportAdmitted = ok;
		reportDetail = detail;
	}

	bool takeReport(int& ok, std::string& detail)
	{
		std::lock_guard<std::mutex> lock(reportMutex);
		if (!reportPending)
			return false;
		reportPending = false;
		ok = reportAdmitted;
		detail = reportDetail;
		return true;
	}

	/**** the VM report (the Mac's tooling): what this process maps executable ****/

	// What this process maps executable: every executable region, those writable now (W+X), and
	// those whose maximum protection lets them be made writable (each named, with its image).
	static void vmReport(const char* when)
	{
#ifdef AOT_TOOLING
		mach_vm_address_t addr = 0;
		mach_vm_size_t size;
		u32 wx = 0, x = 0, maxw = 0, depth = 0;
		std::string named;
		for (;;)
		{
			vm_region_submap_info_data_64_t info;
			mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
			if (mach_vm_region_recurse(mach_task_self(), &addr, &size, &depth, (vm_region_recurse_info_t)&info, &count) != KERN_SUCCESS)
				break;
			if (info.is_submap)
			{
				depth++;
				continue;
			}
			if (info.protection & VM_PROT_EXECUTE)
			{
				x++;
				const bool w = (info.protection & VM_PROT_WRITE) != 0;
				const bool mw = (info.max_protection & VM_PROT_WRITE) != 0;
				wx += w;
				maxw += mw;
				if (w || mw)
				{
					Dl_info dl;
					char line[256];
					snprintf(line, sizeof line, "; %s %llx+%llx (%s)", w ? "W+X" : "max W", (unsigned long long)addr,
						(unsigned long long)size, dladdr((const void*)addr, &dl) && dl.dli_fname ? dl.dli_fname : "no image");
					named += line;
				}
			}
			addr += size;
		}
		fprintf(stderr, "lrps2 AOT VM (%s): %u executable regions, %u writable and executable, %u whose maximum protection allows writing%s\n",
			when, x, wx, maxw, named.c_str());
#else
		(void)when;
#endif
	}

	/**** Shape: the log ****/

#ifdef AOT_TOOLING
	static FILE* logFile;
	static std::string logPath;

	// The log's first line: this build's and game's header fields, which Emit refuses a log
	// without (a log from another build, other settings or another game)
	static std::string buildLine()
	{
		ImageHeader h;
		makeHeader(h, 0, 0, nullptr);
		char line[200];
		snprintf(line, sizeof line, "R %016llx %016llx %016llx %u %x %s\n", (unsigned long long)h.imports_hash,
			(unsigned long long)h.layout_hash, (unsigned long long)h.abi_hash, h.fastmem, h.ee_flags, formatUuid(h.rom_md5).c_str());
		return line;
	}
#endif

	void logBlock(const Block& b, const u8* code)
	{
		stats.compiled[b.unit]++;
#ifdef AOT_TOOLING
		if (cfg.mode != Shape || cfg.log.empty())
			return;
		if (logFile == nullptr || logPath != cfg.log)
		{
			if (logFile)
				fclose(logFile);
			logFile = fopen(cfg.log.c_str(), "w");
			AOT_CHECK(logFile != nullptr, "cannot write %s", cfg.log.c_str());
			logPath = cfg.log;
			fputs(buildLine().c_str(), logFile);
		}
		fprintf(logFile, "B %u %08x %x %u %zu %u %016llx %x %x %x %zu", b.unit, b.pc, b.flags, b.native, b.words.size(),
			b.codeLen, (unsigned long long)fnv(FNV_BASIS, code, b.codeLen), b.mask[0], b.mask[1], b.mask[2], b.sites.size() / 3);
		for (u32 w : b.words)
			fprintf(logFile, " %08x", w);
		for (u32 s : b.sites)
			fprintf(logFile, " %x", s);
		fputc('\n', logFile);
		fflush(logFile);
#else
		(void)code;
#endif
	}

	/**** Load and Interp: the image ****/

	static void* imageLib;
	static const u8* imageCode;
	static std::vector<Block> imageBlockList;
	static std::unordered_map<u64, std::vector<const Block*>> imageIndex;
	static std::vector<Site> imageSiteList;
	static std::string imagePath;

	static void releaseImage()
	{
		imageIndex.clear();
		imageBlockList.clear();
		imageSiteList.clear();
		imageCode = nullptr;
		imagePath.clear();
		if (imageLib != nullptr)
		{
			dlclose(imageLib);
			imageLib = nullptr;
		}
	}

	static u64 indexKey(u32 unit, u32 pc) { return ((u64)unit << 32) | pc; }

	static std::string openImage(const std::string& path)
	{
		releaseImage();
		std::string why;
		const u8* codeEnd;
		const u32 *meta, *metaEnd, *p;
		ImageHeader h;
		imageLib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
		if (imageLib == nullptr)
		{
			why = std::string("dlopen: ") + dlerror();
			goto refuse;
		}
		imageCode = (const u8*)dlsym(imageLib, "rt_aot_code");
		codeEnd = (const u8*)dlsym(imageLib, "rt_aot_code_end");
		meta = (const u32*)dlsym(imageLib, "rt_aot_meta");
		metaEnd = (const u32*)dlsym(imageLib, "rt_aot_meta_end");
		if (imageCode == nullptr || codeEnd == nullptr || meta == nullptr || metaEnd == nullptr || codeEnd < imageCode || metaEnd < meta)
		{
			why = "symbols: no rt_aot_code, rt_aot_code_end, rt_aot_meta or rt_aot_meta_end in order";
			goto refuse;
		}
		if ((size_t)(metaEnd - meta) * 4 < sizeof h)
		{
			why = "meta: shorter than a header";
			goto refuse;
		}
		memcpy(&h, meta, sizeof h);
		why = checkHeader(h);
		if (!why.empty())
			goto refuse;
		if (h.code_size != (size_t)(codeEnd - imageCode))
		{
			why = "code_size: the header says " + std::to_string(h.code_size) + " bytes, the image holds " + std::to_string(codeEnd - imageCode);
			goto refuse;
		}
		p = meta + sizeof h / 4;
		if (h.nblocks > (size_t)(metaEnd - p) / BlockWords)
		{
			why = "meta: " + std::to_string(h.nblocks) + " blocks cannot fit";
			goto refuse;
		}
		imageBlockList.reserve(h.nblocks);
		for (u32 i = 0; i < h.nblocks; i++)
		{
			const size_t left = (size_t)(metaEnd - p);
			if (left < BlockWords)
			{
				why = "meta: it ends inside block " + std::to_string(i);
				goto refuse;
			}
			Block b;
			b.unit = p[0]; b.pc = p[1]; b.flags = p[2]; b.native = p[3];
			const u32 nwords = p[4];
			b.mask[0] = p[5]; b.mask[1] = p[6]; b.mask[2] = p[7];
			b.codeOff = p[8]; b.codeLen = p[9];
			const u32 nsites = p[10];
			// p[11] reserved
			if (b.unit >= UnitCount)
			{
				why = "block_unit: block " + std::to_string(i) + " is for unit " + std::to_string(b.unit) + ", this core knows " + std::to_string((u32)UnitCount);
				goto refuse;
			}
			if (b.flags & ~BLOCK_FLAGS_KNOWN)
			{
				char k[160];
				snprintf(k, sizeof k, "block_flags: block %u (%08x) has flags %#x, this core knows %#x", i, b.pc, b.flags, BLOCK_FLAGS_KNOWN);
				why = k;
				goto refuse;
			}
			// each unit's payload (the format's comment): the EE's at most 96 guest words; microVU1's
			// a state and at most every micro word with a run header each, or its stubs' entries
			const bool vu1 = b.unit == UnitVU1, stub = vu1 && b.pc == VU1_STUB_PC;
			const u32 maxWords = !vu1 ? 96 : stub ? 64 : VU1_STATE_WORDS + 2 * 4096;
			const bool shapeOk = !vu1 || (stub ? b.native == nwords : b.native == VU1_STATE_WORDS) && nsites == 0;
			if (nwords == 0 || nwords > maxWords || b.native > nwords || !shapeOk || nsites > 256 || left < BlockWords + nwords + 3 * (size_t)nsites)
			{
				why = "meta: block " + std::to_string(i) + " has " + std::to_string(nwords) + " words or overruns the meta";
				goto refuse;
			}
			if (b.codeOff > h.code_size || b.codeLen > h.code_size - b.codeOff || (b.codeOff % BlockAlign))
			{
				why = "meta: block " + std::to_string(i) + " overruns the code";
				goto refuse;
			}
			b.words.assign(p + BlockWords, p + BlockWords + nwords);
			b.sites.assign(p + BlockWords + nwords, p + BlockWords + nwords + 3 * nsites);
			for (u32 s = 0; s < b.sites.size(); s++)
				if (s % 3 == 2 ? b.sites[s] >= nwords : b.sites[s] >= b.codeLen)
				{
					why = "meta: block " + std::to_string(i) + " has a fastmem site outside its code";
					goto refuse;
				}
			b.code = imageCode + b.codeOff;
			p += BlockWords + nwords + 3 * nsites;
			imageBlockList.push_back(std::move(b));
		}
		if (p != metaEnd)
		{
			why = "meta: " + std::to_string(metaEnd - p) + " words after the last block";
			goto refuse;
		}
		for (const Block& b : imageBlockList)
		{
			imageIndex[indexKey(b.unit, b.pc)].push_back(&b);
			for (size_t s = 0; s < b.sites.size(); s += 3)
				imageSiteList.push_back({reinterpret_cast<uintptr_t>(b.code) + b.sites[s],
					reinterpret_cast<uintptr_t>(b.code) + b.sites[s + 1], b.pc + b.sites[s + 2] * 4, b.pc});
		}
		std::sort(imageSiteList.begin(), imageSiteList.end(), [](const Site& a, const Site& b) { return a.code < b.code; });
		imagePath = path;
		fprintf(stderr, "lrps2 AOT load: %s: %u blocks, %u bytes of code, emitted by core %s at %p\n", path.c_str(), h.nblocks,
			h.code_size, formatUuid(h.emit_uuid).c_str(), imageCode);
		return std::string();
	refuse:
		releaseImage();
		return why.empty() ? "refused" : why;
	}

	const std::vector<const Block*>* imageBlocks(Unit u, u32 pc)
	{
		auto it = imageIndex.find(indexKey(u, pc));
		return it == imageIndex.end() ? nullptr : &it->second;
	}

	const std::vector<Site>& imageSites() { return imageSiteList; }

	void countInstall(Unit u) { stats.installs[u]++; }
	void countFault(Unit u) { stats.faults[u]++; }

	bool redHostAddr()
	{
#ifdef AOT_TOOLING
		static const bool red = [] { const char* e = getenv("LRPS2_AOT_RED"); return e && !strcmp(e, "hostaddr"); }();
		return red;
#else
		return false;
#endif
	}

	bool redVu1HostAddr()
	{
#ifdef AOT_TOOLING
		static const bool red = [] { const char* e = getenv("LRPS2_AOT_RED"); return e && !strcmp(e, "vu1hostaddr"); }();
		return red;
#else
		return false;
#endif
	}

	void miss(Unit u, u32 pc)
	{
		stats.misses[u]++;
		if (cfg.mode == Load)
		{
			fprintf(stderr, "lrps2 AOT load: unit %u has no image block at %08x for the words guest memory holds there: stopping\n", u, pc);
			fflush(stderr);
			_Exit(5);
		}
	}

	/**** Emit ****/

	static std::vector<Block> emitRecs[UnitCount];
	static std::vector<u8> emitCode;
	static std::vector<u32> emitMeta;
	static u32 emitBlocks;

	const std::vector<Block>& emitRecords(Unit u) { return emitRecs[u]; }
	u32 emitCorrupt() { return cfg.corrupt; }
	bool emitCorruptVu1(u32* pc)
	{
		*pc = cfg.corruptVu1;
		return cfg.hasCorruptVu1;
	}

	void emitted(const Block& b, const u8* code, u32 len, const std::vector<u32>& sites)
	{
		const u64 h = fnv(FNV_BASIS, code, len);
		const bool corrupt = b.unit == UnitEE ? cfg.corrupt != 0 && b.pc == cfg.corrupt
		                                      : cfg.hasCorruptVu1 && b.pc == cfg.corruptVu1;
		if (!corrupt && (len != b.codeLen || h != b.hash || sites != b.sites))
		{
			if (stats.emitDiffers++ < 10)
				fprintf(stderr, "lrps2 AOT emit: unit %u %08x re-emits as %u bytes %016llx (%zu sites), the JIT ran %u bytes %016llx (%zu sites)\n",
					b.unit, b.pc, len, (unsigned long long)h, sites.size() / 3, b.codeLen, (unsigned long long)b.hash, b.sites.size() / 3);
		}
		emitCode.resize((emitCode.size() + BlockAlign - 1) & ~(size_t)(BlockAlign - 1));
		const u32 off = (u32)emitCode.size();
		emitCode.insert(emitCode.end(), code, code + len);
		const u32 rec[BlockWords] = {b.unit, b.pc, b.flags, b.native, (u32)b.words.size(), b.mask[0], b.mask[1], b.mask[2],
			off, len, (u32)(sites.size() / 3), 0};
		emitMeta.insert(emitMeta.end(), rec, rec + BlockWords);
		emitMeta.insert(emitMeta.end(), b.words.begin(), b.words.end()); // the true words
		emitMeta.insert(emitMeta.end(), sites.begin(), sites.end());
		emitBlocks++;
	}

	void emitFailed(const Block& b, const char* why)
	{
		if (stats.emitSkipped++ < 10)
			fprintf(stderr, "lrps2 AOT emit: unit %u %08x: %s\n", b.unit, b.pc, why);
	}

#ifdef AOT_TOOLING
	static bool sameKey(const Block& a, const Block& b)
	{
		return a.unit == b.unit && a.pc == b.pc && a.flags == b.flags && a.native == b.native &&
		       !memcmp(a.mask, b.mask, sizeof a.mask) && a.words == b.words;
	}

	// Read the shape log, emit every distinct block through its unit's emitter, write the image,
	// and exit: 0 when it was written, 3 when a block re-emits otherwise than the JIT ran it (or
	// the JIT compiled one block two ways), 4 when a block could not be emitted or the log is
	// another build's; on 3 or 4 no image is left behind.
	[[noreturn]] static void emitAll()
	{
		AOT_CHECK(!cfg.log.empty() && !cfg.image.empty(), "emit needs LRPS2_AOT_LOG to read and LRPS2_AOT_IMAGE to write");
		FILE* f = fopen(cfg.log.c_str(), "r");
		AOT_CHECK(f != nullptr, "cannot read %s", cfg.log.c_str());
		static char line[1 << 20]; // a microVU1 record reads up to 4096 words
		bool buildOk = false;
		unsigned long nrecs = 0;
		while (fgets(line, sizeof line, f))
		{
			if (line[0] == 'R')
			{
				const std::string mine = buildLine();
				if (mine != line)
				{
					fprintf(stderr, "lrps2 AOT emit: the log's build line is %sthis build's is %s", line, mine.c_str());
					fclose(f);
					_Exit(4);
				}
				buildOk = true;
				continue;
			}
			Block b;
			unsigned long long hash;
			size_t nw, ns;
			int n;
			AOT_CHECK(sscanf(line, "B %u %x %x %u %zu %u %llx %x %x %x %zu%n", &b.unit, &b.pc, &b.flags, &b.native, &nw, &b.codeLen,
						  &hash, &b.mask[0], &b.mask[1], &b.mask[2], &ns, &n) == 11 && b.unit < UnitCount,
				"a bad record in %s: %.60s", cfg.log.c_str(), line);
			b.hash = hash;
			const char* s = line + n;
			for (size_t i = 0; i < nw + 3 * ns; i++)
			{
				unsigned w;
				int m;
				AOT_CHECK(sscanf(s, " %x%n", &w, &m) == 1, "a short record in %s: %.60s", cfg.log.c_str(), line);
				(i < nw ? b.words : b.sites).push_back(w);
				s += m;
			}
			emitRecs[b.unit].push_back(std::move(b));
			nrecs++;
		}
		fclose(f);
		if (!buildOk)
		{
			fprintf(stderr, "lrps2 AOT emit: %s has no build line\n", cfg.log.c_str());
			_Exit(4);
		}
		for (u32 u = 0; u < UnitCount; u++)
		{
			std::vector<Block>& v = emitRecs[u];
			std::stable_sort(v.begin(), v.end(), [](const Block& a, const Block& b) {
				if (a.pc != b.pc) return a.pc < b.pc;
				if (a.flags != b.flags) return a.flags < b.flags;
				if (a.native != b.native) return a.native < b.native;
				if (memcmp(a.mask, b.mask, sizeof a.mask)) return memcmp(a.mask, b.mask, sizeof a.mask) < 0;
				return a.words < b.words;
			});
			std::vector<Block> uniq;
			for (Block& b : v)
			{
				if (!uniq.empty() && sameKey(uniq.back(), b))
				{
					if (b.hash != uniq.back().hash || b.codeLen != uniq.back().codeLen || b.sites != uniq.back().sites)
						if (stats.emitDiffers++ < 10)
							fprintf(stderr, "lrps2 AOT emit: the JIT compiled unit %u %08x two ways\n", u, b.pc);
					continue;
				}
				uniq.push_back(std::move(b));
			}
			v = std::move(uniq);
		}
		eeAotEmit();
		vu1AotEmit();
		const int status = stats.emitDiffers ? 3 : stats.emitSkipped ? 4 : 0;
		fprintf(stderr, "lrps2 AOT emit: %lu records, %zu EE blocks, %zu VU1 blocks (and stubs), %u emitted (%zu bytes of code), %lu skipped, %lu differ from the JIT's\n",
			nrecs, emitRecs[UnitEE].size(), emitRecs[UnitVU1].size(), emitBlocks, emitCode.size(), stats.emitSkipped, stats.emitDiffers);
		const std::string codePath = cfg.image + ".code", metaPath = cfg.image + ".meta";
		unlink(codePath.c_str());
		unlink(metaPath.c_str());
		if (status != 0)
		{
			fprintf(stderr, "lrps2 AOT emit: no image written\n");
			fflush(stderr);
			_Exit(status);
		}
		ImageHeader h;
		AOT_CHECK(!makeHeader(h, emitBlocks, (u32)emitCode.size(), cfg.hasCoreUuid ? cfg.coreUuid : nullptr),
			"this core has no LC_UUID or the game no serial, so the image could name neither");
		fprintf(stderr, "lrps2 AOT emit: an image for core %s and the game %s (%s), emitted by core %s\n",
			formatUuid(h.core_uuid).c_str(), formatUuid(h.rom_md5).c_str(), contentText.c_str(), formatUuid(h.emit_uuid).c_str());
		f = fopen(codePath.c_str(), "wb");
		AOT_CHECK(f != nullptr && fwrite(emitCode.data(), 1, emitCode.size(), f) == emitCode.size(), "cannot write %s", codePath.c_str());
		fclose(f);
		f = fopen(metaPath.c_str(), "wb");
		AOT_CHECK(f != nullptr && fwrite(&h, sizeof h, 1, f) == 1 && fwrite(emitMeta.data(), 4, emitMeta.size(), f) == emitMeta.size(),
			"cannot write %s", metaPath.c_str());
		fclose(f);
		fflush(stderr);
		_Exit(0);
	}
#endif

	/**** admission ****/

	static void endOfRun()
	{
		if (cfg.mode == Off)
			return;
		fprintf(stderr, "lrps2 AOT %s: EE %lu compiled, %lu installed, %lu missed, %lu fastmem faults resumed at a stub; "
			"VU1 %lu compiled, %lu installed, %lu missed\n",
			cfg.mode == Shape ? "shape" : cfg.mode == Emit ? "emit" : cfg.mode == Load ? "load" : "interp",
			stats.compiled[UnitEE], stats.installs[UnitEE], stats.misses[UnitEE], stats.faults[UnitEE],
			stats.compiled[UnitVU1], stats.installs[UnitVU1], stats.misses[UnitVU1]);
		vmReport("end of run");
		fflush(stderr);
	}

	bool admitted() { return admission == Admitted; }

	bool admit()
	{
		static bool atExit;
		if (!atExit)
		{
			atexit(endOfRun);
			atExit = true;
		}
		if (cfg.mode == Off)
			return true;
		if (admission != Unknown)
			return admission == Admitted;
		readContent();
#ifdef AOT_TOOLING
		if (cfg.mode == Emit)
			emitAll();
#endif
		if (!imageOnly())
		{
			admission = Admitted;
			vmReport("shape");
			return true;
		}
		const std::string why = cfg.image.empty() ? std::string("no image is configured") : openImage(cfg.image);
		if (!why.empty())
		{
			fprintf(stderr, "lrps2 AOT: refused %s: %s; the interpreters run instead\n", cfg.image.c_str(), why.c_str());
			admission = Refused;
			report(false, why);
			return false;
		}
		admission = Admitted;
		vmReport("image loaded");
		report(true, imagePath + ": " + std::to_string(imageBlockList.size()) + " blocks");
		return true;
	}
} // namespace aot

#endif
