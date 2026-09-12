/* Shared memory: the views really are one memory.
 *
 * WHAT THIS TESTS
 *
 * SysMainMemory's guest RAM is HostSys::CreateSharedMemory, mapped once by
 * VirtualMemoryManager (HostSys::MapSharedMemory) and then again, page by
 * page, into the 4 GB fastmem area (SharedMemoryMappingArea::Map). The
 * recompilers read and write guest RAM through the fastmem pages while the
 * interpreters, DMA and the GS read it through the main mapping, so the one
 * property everything rests on is that the views are the same physical pages.
 * A view that is a private copy boots, runs for a while, and then diverges
 * wherever a write went through one view and a read through the other.
 *
 * On Apple the implementation is a Mach memory entry (vm_allocate +
 * mach_make_memory_entry_64, every view a vm_map of the entry) rather than
 * shm_open, so this is what holds that shape to the property. It links the
 * real common/HostSys.cpp out of a built libcommon.a -- a real-code harness,
 * not a transcription. Plain read-write data memory only: nothing here is
 * ever mapped executable.
 *
 * USAGE (from repo root, after a cmake build of the core)
 *
 *   sh tests/hostmem/build.sh            # builds and runs
 *   LRPS2_BUILD=path/to/build sh tests/hostmem/build.sh
 *
 * Exit 0 = every check held; otherwise each failure is named.
 */
#include "common/General.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unistd.h>

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

static PageProtectionMode rw()
{
	PageProtectionMode m;
	m.m_read = true;
	m.m_write = true;
	m.m_exec = false;
	return m;
}

int main()
{
	const size_t page = static_cast<size_t>(getpagesize());
	const size_t size = 4 * page;

	void* shm = HostSys::CreateSharedMemory(HostSys::GetFileMappingName("lrps2_hostmem").c_str(), size);
	CHECK(shm, "CreateSharedMemory(%zu) returned no handle", size);
	if (!shm)
		return 1;

	u8* a = static_cast<u8*>(HostSys::MapSharedMemory(shm, 0, nullptr, size, rw()));
	u8* b = static_cast<u8*>(HostSys::MapSharedMemory(shm, 0, nullptr, size, rw()));
	CHECK(a && b && a != b, "two views of one shared memory: %p and %p", static_cast<void*>(a), static_cast<void*>(b));
	if (!a || !b)
		return 1;

	for (size_t i = 0; i < size; i++)
		a[i] = static_cast<u8>(i * 7 + 3);
	CHECK(std::memcmp(a, b, size) == 0,
		"a write through one view is not visible through the other -- the views are copies, not "
		"one memory, so fastmem and the interpreters would see different guest RAM");
	b[page + 5] = 0xA5;
	CHECK(a[page + 5] == 0xA5, "and not the other way round either (read %#x through the first view)", a[page + 5]);

	/* A placement request over something already mapped fails and leaves it
	 * alone -- VirtualMemoryManager walks candidate bases relying on that. */
	void* clash = HostSys::MapSharedMemory(shm, 0, a, size, rw());
	CHECK(clash == nullptr, "a placement request over the first view succeeded (%p) instead of failing", clash);
	CHECK(a[page + 5] == 0xA5 && a[0] == 3, "and the refused placement disturbed the view it was refused over");

	/* A view at an offset starts at that offset. */
	u8* off = static_cast<u8*>(HostSys::MapSharedMemory(shm, 2 * page, nullptr, page, rw()));
	CHECK(off && off[0] == a[2 * page] && off[page - 1] == a[3 * page - 1],
		"a view at offset %zu does not show the memory at that offset", 2 * page);

	/* The fastmem area: a PROT_NONE reservation whose pages are replaced, one
	 * at a time, by views of guest RAM. */
	std::unique_ptr<SharedMemoryMappingArea> area = SharedMemoryMappingArea::Create(4 * page);
	CHECK(area != nullptr, "SharedMemoryMappingArea::Create failed");
	if (area)
	{
		u8* p = area->Map(shm, page, area->PagePointer(3), page, rw());
		CHECK(p == area->PagePointer(3), "Map placed the page at %p, not at the area's page 3 (%p)",
			static_cast<void*>(p), static_cast<void*>(area->PagePointer(3)));
		if (p)
		{
			CHECK(p[5] == 0xA5, "the area's page shows %#x where guest RAM holds 0xA5 -- not the same pages", p[5]);
			p[6] = 0x5A;
			CHECK(a[page + 6] == 0x5A, "a write through the fastmem page is not visible in guest RAM");
			CHECK(area->Unmap(p, page), "Unmap of the area's page failed");
		}
	}

	HostSys::UnmapSharedMemory(off, page);
	HostSys::Munmap(off, page);
	HostSys::UnmapSharedMemory(b, size);
	HostSys::Munmap(b, size);
	HostSys::UnmapSharedMemory(a, size);
	HostSys::Munmap(a, size);
	HostSys::DestroySharedMemory(shm);

	if (s_failures)
	{
		std::fprintf(stderr, "hostmem: %d check(s) failed\n", s_failures);
		return 1;
	}
	std::printf("hostmem: ok (shared memory views alias; placement requests refuse without side effects)\n");
	return 0;
}
