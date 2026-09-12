// SPDX-FileCopyrightText: 2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0

#pragma once

// Where the arm64 recompilers' code memory comes from, and how the address a
// block runs at becomes the address its bytes are written at.
//
// Two shapes. SELF: the core maps its own caches (armJitMap -> HostSys::Mmap,
// MAP_JIT on macOS) and writes them in place, so the address written is the
// address executed. FRONTEND: the libretro frontend lends each cache through
// RETRO_ENVIRONMENT_EXEC_MEM_ALLOC as two mappings of the same pages -- `rx`,
// read-execute, and `rw`, read-write -- which is the only code memory an iOS
// 26 SPTM/TXM device allows (the frontend's debugger prepared it). There
// every code pointer the recompilers keep, branch to, compute a displacement
// from, or publish is the EXECUTE address, and only the bytes themselves go
// through the write alias. ArmJitRegions is the one place an execute address
// becomes a write address.
//
// Everything defined here is pure -- no mapping, no emitter, no environment
// call -- so tests/jitmem holds it on any host without mapping or executing
// anything. The one declaration, armJitSetSource, is how the libretro layer
// hands the decision and the frontend's two calls to the recompilers.

#include "libretro.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>

enum class ArmJitSource
{
	Self,     // map our own caches (MAP_JIT on macOS; plain mmap elsewhere)
	Frontend, // every cache is a dual-mapped lease from the frontend (env 83)
	None,     // no code memory at all: every recompiler runs its interpreter
};

// The env-83 probe (a size-0 RETRO_ENVIRONMENT_EXEC_MEM_ALLOC), reduced to a
// decision -- flycast's, so both cores read one frontend the same way:
//  - DUAL_MAP: lease every cache from the frontend.
//  - no answer, or UNRESTRICTED: the platform does not restrict executable
//    memory, so map our own -- unless GET_JIT_CAPABLE (74) says no, or this
//    host must never self-map (iOS: a self-mapped page there is one this
//    process may not branch into).
//  - anything else (UNAVAILABLE, or the RWX / WX_TOGGLE shapes this core
//    does not implement): no code memory; the recompilers stand down.
constexpr ArmJitSource armJitChooseSource(bool probe_answered, unsigned mode, bool capable_answered, bool capable,
	bool host_may_self_map)
{
	if (probe_answered && mode == RETRO_EXEC_MEM_MODE_DUAL_MAP)
		return ArmJitSource::Frontend;
	const bool unrestricted = !probe_answered || mode == RETRO_EXEC_MEM_MODE_UNRESTRICTED;
	if (unrestricted && host_may_self_map && (!capable_answered || capable))
		return ArmJitSource::Self;
	return ArmJitSource::None;
}

// Whether this host may map its own code pages. Not iOS: a self-mapped page
// there is one this process may not branch into (on a TXM device, touching
// it kills the process where nothing can catch it).
#if defined(__APPLE__) && TARGET_OS_IPHONE
inline constexpr bool kArmJitHostMaySelfMap = false;
#else
inline constexpr bool kArmJitHostMaySelfMap = true;
#endif

// The source before any retro_init() has decided one: what a frontend that
// answered neither call gets. Self on a desktop (the behaviour the core
// always had), None on iOS -- never a self-mapped cache there.
constexpr ArmJitSource armJitDefaultSource(bool host_may_self_map)
{
	return armJitChooseSource(false, 0, false, false, host_may_self_map);
}

// RETRO_ENVIRONMENT_EXEC_MEM_ALLOC / _FREE, as the libretro layer wraps them.
// Alloc returns true with both aliases of a dual-mapped lease, else false.
using ArmJitHostAlloc = bool (*)(size_t size, void** rx, void** rw);
using ArmJitHostFree = void (*)(void* rx);

// Where this load's code memory comes from; set at every retro_init(), before
// any cache is mapped (defined in AsmHelpers.cpp).
void armJitSetSource(ArmJitSource source, ArmJitHostAlloc alloc, ArmJitHostFree free);

// The dual-mapped leases the recompilers hold, as execute->write pairs. Each
// lease carries its own offset: the libretro header promises nothing about
// how two leases' aliases relate, so there is no single "rx - rw" to apply.
//
// Add/Remove run only where caches are reserved and released (a load and an
// unload), never while a block is being compiled or run. Find runs
// on every compile thread and inside the fastmem fault handler, so it is
// async-signal-safe: a fixed table, loads only, no locks, no allocation. A
// slot is published by storing its execute base last (release), and read by
// loading it first (acquire).
class ArmJitRegions
{
public:
	// The recompilers' caches at their most: EE, IOP, microVU0/1, VIF0/1,
	// VIF unpack, plus the transient self-test page.
	static constexpr size_t kMaxRegions = 8;
	// adrp works in 4 KiB pages. The two aliases must sit at the same offset
	// within one, or page arithmetic done while emitting (vixl binds labels
	// against the buffer it writes) would differ from what runs.
	static constexpr uintptr_t kAdrpPageMask = 0xFFF;

	enum class AddResult
	{
		Ok,
		Empty,      // a zero size, or a null alias
		SameAlias,  // exec == write: not a dual mapping
		PageOffset, // the aliases disagree within an adrp page
		Overlap,    // the execute range overlaps a region already held
		Full,       // kMaxRegions already held
	};

	AddResult Add(uintptr_t exec, uintptr_t write, size_t size)
	{
		if (!exec || !write || !size)
			return AddResult::Empty;
		if (exec == write)
			return AddResult::SameAlias;
		if ((exec ^ write) & kAdrpPageMask)
			return AddResult::PageOffset;
		Slot* free_slot = nullptr;
		for (Slot& s : m_slots)
		{
			const uintptr_t base = s.exec.load(std::memory_order_acquire);
			if (!base)
			{
				if (!free_slot)
					free_slot = &s;
				continue;
			}
			if (exec < base + s.size && base < exec + size)
				return AddResult::Overlap;
		}
		if (!free_slot)
			return AddResult::Full;
		free_slot->write = write;
		free_slot->size = size;
		free_slot->exec.store(exec, std::memory_order_release);
		return AddResult::Ok;
	}

	// Forget the region whose execute base is `exec`. False when no region
	// starts there -- an address inside one does not remove it.
	bool Remove(uintptr_t exec)
	{
		if (!exec)
			return false;
		for (Slot& s : m_slots)
		{
			if (s.exec.load(std::memory_order_acquire) == exec)
			{
				s.exec.store(0, std::memory_order_release);
				return true;
			}
		}
		return false;
	}

	// The address the byte executed at `exec` is written at -- the same
	// offset in the write alias of the region holding it -- or 0 when no
	// region does. What a miss means depends on the source:
	// armJitWriteAddress.
	uintptr_t Find(uintptr_t exec) const
	{
		for (const Slot& s : m_slots)
		{
			const uintptr_t base = s.exec.load(std::memory_order_acquire);
			if (base && exec - base < s.size)
				return s.write + (exec - base);
		}
		return 0;
	}

private:
	struct Slot
	{
		std::atomic<uintptr_t> exec{0};
		uintptr_t write = 0;
		size_t size = 0;
	};
	Slot m_slots[kMaxRegions];
};

// Where a write to the code byte executed at `exec` goes: through the write
// alias of the lease holding it; in place only when this load maps its own
// caches. Under any other source a code byte no lease holds is not code
// memory this load has, and the answer is 0 -- the caller stops the process
// with a message rather than write at an execute address (on the iPad a
// read-execute page: a fault with no explanation, or, if the pointer is
// stale, a write into whatever the frontend lent next).
inline uintptr_t armJitWriteAddress(const ArmJitRegions& regions, ArmJitSource source, uintptr_t exec)
{
	if (const uintptr_t write = regions.Find(exec))
		return write;
	return source == ArmJitSource::Self ? exec : 0;
}

// How a cache goes back. A lease the table held returns to the frontend --
// by what the table says, not by the source: a lease goes back to whoever
// lent it. A cache no lease holds is munmapped only when this load maps its
// own; under any other source it is pages this core does not own (under
// Frontend, the frontend's pool, which a munmap would take from it for
// good), so the caller refuses and stops the process with a message.
enum class ArmJitRelease
{
	ToFrontend,
	Munmap,
	Refuse,
};

constexpr ArmJitRelease armJitReleaseRule(bool was_leased, ArmJitSource source)
{
	if (was_leased)
		return ArmJitRelease::ToFrontend;
	return source == ArmJitSource::Self ? ArmJitRelease::Munmap : ArmJitRelease::Refuse;
}
