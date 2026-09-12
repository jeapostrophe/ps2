/* Shared memory: the views really are one memory.
 *
 * WHAT THIS TESTS
 *
 * SysMainMemory's guest RAM is HostSys::CreateSharedMemory, mapped once by
 * VirtualMemoryManager (HostSys::MapSharedMemory) and then again, page by
 * page, into the 4 GB fastmem area (SharedMemoryMappingArea::Map). The
 * recompilers read and write guest RAM through the fastmem pages while the
 * interpreters, DMA and the GS read it through the main mapping, so the one
 * property everything rests on is that the views are the same physical pages
 * -- all of them, at the size SysMainMemory really asks for.
 *
 * On Apple the implementation is Mach memory entries (vm_allocate +
 * mach_make_memory_entry_64, every view a vm_map) rather than shm_open. One
 * entry names at most one VM object, and XNU builds anonymous memory out of
 * 128 MB objects, so at SysMainMemory's size the backing needs several
 * entries -- which is why this runs at HostMemoryMap::MainSize and checks
 * every page, rather than at a size where one entry would do: a first version
 * of it ran at 64 KiB and passed over code that could not map guest RAM.
 *
 * It links the real common/HostSys.cpp out of a built libcommon.a -- a
 * real-code harness, not a transcription. Plain read-write data memory only:
 * nothing here is ever mapped executable.
 *
 * USAGE (from repo root, after a cmake build of the core)
 *
 *   sh tests/hostmem/build.sh            # builds and runs
 *   LRPS2_BUILD=path/to/build sh tests/hostmem/build.sh
 *
 * Exit 0 = every check held; otherwise each failure is named.
 */
#include "common/General.h"
#include "pcsx2/Memory.h"

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

// A value per page that no other page shares.
static u32 stamp(size_t page_index)
{
	return static_cast<u32>(page_index * 2654435761u) ^ 0x5A5A0001u;
}

int main()
{
	const size_t page = static_cast<size_t>(getpagesize());
	const size_t size = HostMemoryMap::MainSize; // what SysMainMemory asks for
	const size_t pages = size / page;

	void* shm = HostSys::CreateSharedMemory(HostSys::GetFileMappingName("lrps2_hostmem").c_str(), size);
	CHECK(shm, "CreateSharedMemory(%#zx) -- SysMainMemory's size -- returned no handle, so no guest RAM", size);
	if (!shm)
		return 1;

	u8* a = static_cast<u8*>(HostSys::MapSharedMemory(shm, 0, nullptr, size, rw()));
	u8* b = static_cast<u8*>(HostSys::MapSharedMemory(shm, 0, nullptr, size, rw()));
	CHECK(a && b && a != b, "two views of one shared memory: %p and %p", static_cast<void*>(a), static_cast<void*>(b));
	if (!a || !b)
		return 1;

	// Every page, through one view, then back through the other.
	size_t first_bad = pages;
	for (size_t i = 0; i < pages; i++)
	{
		const u32 s = stamp(i);
		std::memcpy(a + i * page, &s, sizeof(s));
	}
	for (size_t i = 0; i < pages && first_bad == pages; i++)
	{
		u32 v;
		std::memcpy(&v, b + i * page, sizeof(v));
		if (v != stamp(i))
			first_bad = i;
	}
	CHECK(first_bad == pages,
		"page %zu of %zu (offset %#zx) written through one view is not what the other view reads -- "
		"the views are not one memory there, so fastmem and the interpreters would see different guest RAM",
		first_bad, pages, first_bad * page);
	b[size - 1] = 0xA5;
	CHECK(a[size - 1] == 0xA5, "the last byte written through the second view is %#x through the first", a[size - 1]);

	/* A placement request over something already mapped fails and leaves it
	 * alone -- VirtualMemoryManager walks candidate bases relying on that. */
	void* clash = HostSys::MapSharedMemory(shm, 0, a, size, rw());
	CHECK(clash == nullptr, "a placement request over the first view succeeded (%p) instead of failing", clash);
	u32 v0;
	std::memcpy(&v0, a, sizeof(v0));
	CHECK(v0 == stamp(0) && a[size - 1] == 0xA5, "and the refused placement disturbed the view it was refused over");

	/* A view at an offset starts at that offset -- here, the last two pages. */
	u8* off = static_cast<u8*>(HostSys::MapSharedMemory(shm, size - 2 * page, nullptr, 2 * page, rw()));
	CHECK(off && std::memcmp(off, a + size - 2 * page, 2 * page) == 0,
		"a view at offset %#zx does not show the memory at that offset", size - 2 * page);

	/* The fastmem area: a PROT_NONE reservation whose pages are replaced, one
	 * at a time, by views of guest RAM -- from anywhere in it. */
	std::unique_ptr<SharedMemoryMappingArea> area = SharedMemoryMappingArea::Create(4 * page);
	CHECK(area != nullptr, "SharedMemoryMappingArea::Create failed");
	if (area)
	{
		const size_t sources[] = {page, size / 2, size - page};
		for (size_t k = 0; k < 3; k++)
		{
			const size_t src = sources[k];
			u8* p = area->Map(shm, src, area->PagePointer(k + 1), page, rw());
			CHECK(p == area->PagePointer(k + 1), "Map of guest offset %#zx placed the page at %p, not at the area's page %zu (%p)",
				src, static_cast<void*>(p), k + 1, static_cast<void*>(area->PagePointer(k + 1)));
			if (!p)
				continue;
			CHECK(std::memcmp(p, a + src, page) == 0, "the area's page for guest offset %#zx is not guest RAM there", src);
			p[8] = static_cast<u8>(0x40 + k);
			CHECK(a[src + 8] == 0x40 + k, "a write through the fastmem page for guest offset %#zx is not visible in guest RAM", src);
			CHECK(area->Unmap(p, page), "Unmap of the area's page %zu failed", k + 1);
		}
	}

	if (off)
	{
		HostSys::UnmapSharedMemory(off, 2 * page);
		HostSys::Munmap(off, 2 * page);
	}
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
	std::printf("hostmem: ok (%zu MB of shared memory: every page aliases; placement requests refuse without side effects)\n", size >> 20);
	return 0;
}
