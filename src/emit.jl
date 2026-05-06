@enum SideEffectGroup PureALU Memory Sync

is_volatile(g::SideEffectGroup) = g != PureALU
needs_memory_clobber(g::SideEffectGroup) = g != PureALU
