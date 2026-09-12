/* The arm64 JIT's code-memory decisions, held without any code memory.
 *
 * WHAT THIS TESTS
 *
 * pcsx2/arm64/ArmJitMemory.h: which source the recompilers take their code
 * memory from after the env-83 probe (armJitChooseSource), and how an
 * execute address becomes the address its bytes are written at
 * (ArmJitRegions). On an iOS station every cache is a dual-mapped lease from
 * the frontend -- read-execute at `rx`, read-write at `rw` -- and the two
 * wrong answers are both fatal there and invisible on a desktop, where a
 * self-mapped cache is written where it runs: writing at the execute address
 * faults on the first byte of the first block, and publishing the write
 * address faults on the first fetch. So the translation is pure and held
 * here, with made-up addresses: this program maps nothing, writes no code
 * and executes nothing.
 *
 * It includes the real header (a real-code harness): the table and the
 * decision under test are the ones the recompilers use.
 *
 * USAGE (from repo root)
 *
 *   sh tests/jitmem/build.sh      # builds and runs; exit 0 = every check held
 */
#include "pcsx2/arm64/ArmJitMemory.h"

#include <cstdio>

static int s_failures = 0;

#define CHECK(cond, ...) \
	do \
	{ \
		if (!(cond)) \
		{ \
			std::fprintf(stderr, "FAIL: "); \
			std::fprintf(stderr, __VA_ARGS__); \
			std::fprintf(stderr, "\n"); \
			s_failures++; \
		} \
	} while (0)

using AddResult = ArmJitRegions::AddResult;

static constexpr uintptr_t MB = 1u << 20;

static void a_lease_is_written_through_its_write_alias()
{
	ArmJitRegions r;
	const uintptr_t rx = 0x1'0000'0000, rw = 0x2'0000'4000; // same offset within a 4 KiB page
	CHECK(r.Add(rx, rw, 32 * MB) == AddResult::Ok, "a well-formed 32 MB lease was refused");
	CHECK(r.WriteAddress(rx) == rw,
		"the first byte of a lease is written at %#lx, not its write alias %#lx -- on the iPad the execute "
		"alias is read-execute only, so the first block's first byte would fault",
		(unsigned long)r.WriteAddress(rx), (unsigned long)rw);
	CHECK(r.WriteAddress(rx + 32 * MB - 4) == rw + 32 * MB - 4, "the last instruction of the lease does not translate by the lease's offset");
	CHECK(r.WriteAddress(rx + 32 * MB) == rx + 32 * MB, "the byte past the lease was translated as if it were inside it");
	CHECK(r.WriteAddress(rx - 4) == rx - 4, "the word before the lease was translated as if it were inside it");
	int dummy = 0;
	CHECK(r.Write(&dummy) == &dummy, "an address no lease holds (a self-mapped cache) must be written in place");
}

static void every_lease_keeps_its_own_offset()
{
	// Two caches from one frontend: nothing in the libretro header says their
	// write aliases sit at one fixed distance from their execute aliases.
	ArmJitRegions r;
	const uintptr_t ee_rx = 0x1'0000'0000, ee_rw = 0x3'0000'0000;
	const uintptr_t iop_rx = 0x1'4000'0000, iop_rw = 0x2'0000'0000;
	CHECK(r.Add(ee_rx, ee_rw, 32 * MB) == AddResult::Ok && r.Add(iop_rx, iop_rw, 16 * MB) == AddResult::Ok, "two leases were refused");
	CHECK(r.WriteAddress(ee_rx + 0x40) == ee_rw + 0x40, "the EE lease translates by another lease's offset");
	CHECK(r.WriteAddress(iop_rx + 0x40) == iop_rw + 0x40,
		"the IOP lease is written at %#lx, not %#lx -- the table applied one global rx-rw offset",
		(unsigned long)r.WriteAddress(iop_rx + 0x40), (unsigned long)(iop_rw + 0x40));
}

static void a_returned_lease_is_forgotten()
{
	ArmJitRegions r;
	const uintptr_t rx = 0x1'0000'0000, rw = 0x2'0000'0000;
	r.Add(rx, rw, MB);
	CHECK(!r.Remove(rx + 4096), "an address inside a lease removed it");
	CHECK(r.Remove(rx), "the lease's own execute base did not remove it");
	CHECK(r.WriteAddress(rx) == rx, "a returned lease still translates -- the next core's pages would be written through it");
	CHECK(!r.Remove(rx), "a lease was removed twice");
	CHECK(r.Add(rx, rw, MB) == AddResult::Ok, "the slot of a returned lease cannot be reused");
}

static void malformed_leases_are_refused()
{
	ArmJitRegions r;
	CHECK(r.Add(0x1'0000'0000, 0x2'0000'0000, 0) == AddResult::Empty, "a zero-size lease was accepted");
	CHECK(r.Add(0, 0x2'0000'0000, MB) == AddResult::Empty, "a null execute alias was accepted");
	CHECK(r.Add(0x1'0000'0000, 0x1'0000'0000, MB) == AddResult::SameAlias, "a lease whose two aliases are one address was accepted as a dual mapping");
	CHECK(r.Add(0x1'0000'0000, 0x2'0000'0800, MB) == AddResult::PageOffset,
		"aliases 0x800 apart within a 4 KiB page were accepted -- adrp arithmetic done while writing would not be what runs");
	CHECK(r.Add(0x1'0000'0000, 0x2'0000'0000, 2 * MB) == AddResult::Ok, "a good lease was refused");
	CHECK(r.Add(0x1'0010'0000, 0x3'0000'0000, MB) == AddResult::Overlap, "a lease overlapping a held one was accepted");
}

static void the_table_holds_kMaxRegions_and_no_more()
{
	ArmJitRegions r;
	for (size_t i = 0; i < ArmJitRegions::kMaxRegions; i++)
		CHECK(r.Add(0x1'0000'0000 + i * 64 * MB, 0x4'0000'0000 + i * 64 * MB, 64 * MB) == AddResult::Ok, "lease %zu of %zu was refused", i + 1, ArmJitRegions::kMaxRegions);
	CHECK(r.Add(0x9'0000'0000, 0xA'0000'0000, MB) == AddResult::Full, "a lease past kMaxRegions was accepted");
	const uintptr_t last = 0x1'0000'0000 + (ArmJitRegions::kMaxRegions - 1) * 64 * MB;
	CHECK(r.WriteAddress(last + 8) == 0x4'0000'0000 + (ArmJitRegions::kMaxRegions - 1) * 64 * MB + 8, "a full table stopped translating its last lease");
}

static void the_probe_decides_where_code_memory_comes_from()
{
	// (probe answered, mode, 74 answered, 74 capable, host may self-map)
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_DUAL_MAP, true, false, false) == ArmJitSource::Frontend,
		"a frontend offering dual-mapped memory on iOS must be used, whatever 74 says");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_DUAL_MAP, false, false, true) == ArmJitSource::Frontend,
		"a frontend offering dual-mapped memory on a desktop must be used too");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_UNAVAILABLE, true, true, true) == ArmJitSource::None,
		"UNAVAILABLE must mean no code memory -- self-mapping there is what the frontend just said not to do");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_UNAVAILABLE, true, false, false) == ArmJitSource::None,
		"an iPad with no debugger attached (UNAVAILABLE) must run the interpreters");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_RWX, true, true, true) == ArmJitSource::None, "RWX is not implemented here and must not be taken");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_WX_TOGGLE, true, true, true) == ArmJitSource::None, "WX_TOGGLE is not implemented here and must not be taken");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_UNRESTRICTED, true, true, true) == ArmJitSource::Self,
		"a desktop frontend (UNRESTRICTED, 74 yes) must leave the core on its own MAP_JIT caches");
	CHECK(armJitChooseSource(false, 0, false, false, true) == ArmJitSource::Self,
		"a frontend that knows neither call must get the behaviour the core always had");
	CHECK(armJitChooseSource(true, RETRO_EXEC_MEM_MODE_UNRESTRICTED, true, false, true) == ArmJitSource::None,
		"UNRESTRICTED with 74 answering no must not self-map");
	CHECK(armJitChooseSource(false, 0, false, false, false) == ArmJitSource::None,
		"on iOS a frontend without env 83 must never lead the core to map its own code pages");
}

int main()
{
	a_lease_is_written_through_its_write_alias();
	every_lease_keeps_its_own_offset();
	a_returned_lease_is_forgotten();
	malformed_leases_are_refused();
	the_table_holds_kMaxRegions_and_no_more();
	the_probe_decides_where_code_memory_comes_from();
	if (s_failures)
	{
		std::fprintf(stderr, "jitmem: %d check(s) failed\n", s_failures);
		return 1;
	}
	std::printf("jitmem: ok\n");
	return 0;
}
