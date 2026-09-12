// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#if defined(__APPLE__)

#include "common/General.h"

#include <cstddef>
#include <memory>

/* GSAllocateWrappedMemory on Apple: `repeat` consecutive views of one
 * `size`-byte block, so a GS access that runs off the end wraps to the start.
 * No shm_open here -- HostSys::CreateSharedMemory's Mach memory entries (see
 * there for why), and every slot, the first included, is a view of them mapped
 * over one reservation (SharedMemoryMappingArea::Map: VM_FLAGS_FIXED |
 * VM_FLAGS_OVERWRITE). A first cut used the region the entry was made from as
 * slot 0 and mapped only the others; that aliases only because XNU happens to
 * share that region's VM object with the entry, which nothing checked.
 * tests/hostmem holds every slot against every other. */
static void* s_gs_shm = nullptr;
static std::unique_ptr<SharedMemoryMappingArea> s_gs_area;

void GSFreeWrappedMemory(void* ptr, size_t size, size_t repeat)
{
	(void)ptr;
	(void)size;
	(void)repeat;
	s_gs_area.reset(); // unmaps the reservation, every view with it
	if (s_gs_shm)
		HostSys::DestroySharedMemory(s_gs_shm);
	s_gs_shm = nullptr;
}

void* GSAllocateWrappedMemory(size_t size, size_t repeat)
{
	s_gs_shm = HostSys::CreateSharedMemory(HostSys::GetFileMappingName("pcsx2_gs").c_str(), size);
	if (!s_gs_shm)
		return nullptr;
	s_gs_area = SharedMemoryMappingArea::Create(size * repeat);
	if (!s_gs_area)
	{
		GSFreeWrappedMemory(nullptr, size, repeat);
		return nullptr;
	}

	PageProtectionMode rw;
	rw.m_read = true;
	rw.m_write = true;
	rw.m_exec = false;
	for (size_t i = 0; i < repeat; i++)
	{
		if (!s_gs_area->Map(s_gs_shm, 0, s_gs_area->OffsetPointer(i * size), size, rw))
		{
			GSFreeWrappedMemory(nullptr, size, repeat);
			return nullptr;
		}
	}
	return s_gs_area->BasePointer();
}

#endif
