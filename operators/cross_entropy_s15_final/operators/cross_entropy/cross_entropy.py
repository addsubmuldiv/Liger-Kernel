"""Stable two-pass CrossEntropy kernels for Ascend950; select the backend at launch."""

import tilelang
import tilelang.ascend.language as T

from tilelang.ascend.language import simd as S


@tilelang.jit
def baseline_cross_entropy_kernel(N, V, dtype, cores, tile, has_weight, smoothing, softcap, z_scale, need_grad):
    """Compute per-row loss and optionally its input gradient without changing logits."""
    blocks = tilelang.cdiv(V, tile)

    @T.prim_func
    def main(
        X: T.Tensor((N, V), dtype),
        Target: T.Tensor((N,), "int32"),
        W: T.Tensor((V,), "float32"),
        Loss: T.Tensor((N,), dtype),
        ZLoss: T.Tensor((N,), dtype),
        DX: T.Tensor((N, V), dtype),
        ignore_index: T.int32,
        denominator: T.float32,
        valid_count: T.float32,
        weight_sum: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((tile,), dtype)
            w = T.alloc_shared((tile,), "float32")
            dx = T.alloc_shared((tile,), dtype)
            target = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((8,), "float32")
            loss = T.alloc_shared((8,), dtype)
            zloss = T.alloc_shared((8,), dtype)
            for step in T.serial(tilelang.cdiv(N, cores)):
                row = step * cores + core
                if row < N:
                    T.copy(Target[row : row + 1], target[:1])
                    with T.SimtVF(threads=128):
                        for i in T.Parallel(8):
                            state[i] = T.if_then_else(i == 0, -T.infinity("float32"), T.float32(0))
                    for block in T.serial(blocks):
                        start = block * tile
                        valid = T.min(tile, V - start)
                        T.copy(X[row, start : start + valid], x[:valid], l2_cache_ctrl="NORMAL_PERS")
                        if has_weight:
                            T.copy(W[start : start + valid], w[:valid])
                        with T.SimtVF(threads=128):
                            a = T.alloc_fragment((tile,), "float32")
                            e = T.alloc_fragment((tile,), "float32")
                            wx = T.alloc_fragment((tile,), "float32")
                            xy = T.alloc_fragment((tile,), "float32")
                            wy = T.alloc_fragment((tile,), "float32")
                            m = T.alloc_fragment((1,), "float32")
                            d = T.alloc_fragment((1,), "float32")
                            sx = T.alloc_fragment((1,), "float32")
                            sy = T.alloc_fragment((1,), "float32")
                            sw = T.alloc_fragment((1,), "float32")
                            for i in T.Parallel(tile):
                                if i < valid:
                                    value = T.cast(x[i], "float32")
                                    if softcap > 0:
                                        scaled = value / softcap
                                        magnitude = T.if_then_else(scaled < 0, -scaled, scaled)
                                        decay = T.exp(-2 * magnitude)
                                        tanh_abs = (1 - decay) / (1 + decay)
                                        value = softcap * T.if_then_else(scaled < 0, -tanh_abs, tanh_abs)
                                    a[i] = value
                                    if has_weight:
                                        wx[i] = value * w[i]
                                        wy[i] = T.if_then_else(start + i == target[0], w[i], T.float32(0))
                                    else:
                                        wx[i] = value
                                        wy[i] = T.if_then_else(start + i == target[0], T.float32(1), T.float32(0))
                                    xy[i] = T.if_then_else(start + i == target[0], value, T.float32(0))
                                else:
                                    a[i] = -T.infinity("float32")
                                    wx[i] = T.float32(0)
                                    xy[i] = T.float32(0)
                                    wy[i] = T.float32(0)
                            T.reduce_max(a, m, dim=0)
                            for i in T.Parallel(tile):
                                e[i] = T.exp(a[i] - m[0])
                            T.reduce_sum(e, d, dim=0)
                            T.reduce_sum(wx, sx, dim=0)
                            T.reduce_sum(xy, sy, dim=0)
                            T.reduce_sum(wy, sw, dim=0)
                            for i in T.Parallel(1):
                                new_max = T.max(state[0], m[0])
                                state[1] = state[1] * T.exp(state[0] - new_max) + d[0] * T.exp(m[0] - new_max)
                                state[0] = new_max
                                state[2] = state[2] + sx[0]
                                state[3] = state[3] + sy[0]
                                state[4] = state[4] + sw[0]
                    with T.SimtVF(threads=128):
                        for i in T.Parallel(1):
                            lse = state[0] + T.log(state[1])
                            ce = (1 - smoothing) * state[4] * (lse - state[3])
                            ce = ce + smoothing / V * (lse * weight_sum - state[2])
                            z = z_scale * lse * lse / valid_count
                            loss[0] = T.cast(
                                T.if_then_else(target[0] == ignore_index, T.float32(0), ce / denominator + z), dtype
                            )
                            zloss[0] = T.cast(T.if_then_else(target[0] == ignore_index, T.float32(0), z), dtype)
                            state[5] = lse
                    T.copy(loss[:1], Loss[row : row + 1])
                    T.copy(zloss[:1], ZLoss[row : row + 1])
                    if need_grad:
                        for block in T.serial(blocks):
                            start = block * tile
                            valid = T.min(tile, V - start)
                            T.copy(X[row, start : start + valid], x[:valid], l2_cache_ctrl="NORMAL_PERS")
                            if has_weight:
                                T.copy(W[start : start + valid], w[:valid])
                            with T.SimtVF(threads=128):
                                for i in T.Parallel(tile):
                                    if i < valid:
                                        value = T.cast(x[i], "float32")
                                        chain = T.float32(1)
                                        if softcap > 0:
                                            scaled = value / softcap
                                            magnitude = T.if_then_else(scaled < 0, -scaled, scaled)
                                            decay = T.exp(-2 * magnitude)
                                            tanh_abs = (1 - decay) / (1 + decay)
                                            t = T.if_then_else(scaled < 0, -tanh_abs, tanh_abs)
                                            value = softcap * t
                                            chain = 1 - t * t
                                        # Avoid log/exp round trips when the true-class probability is near one.
                                        p = T.exp(value - state[0]) / state[1]
                                        wi = w[i] if has_weight else T.float32(1)
                                        g = p * ((1 - smoothing) * state[4] + smoothing / V * weight_sum)
                                        g = g - smoothing / V * wi
                                        g = g - T.if_then_else(
                                            start + i == target[0], (1 - smoothing) * state[4], T.float32(0)
                                        )
                                        g = g / denominator + 2 * z_scale * state[5] * p / valid_count
                                        dx[i] = T.cast(
                                            T.if_then_else(target[0] == ignore_index, T.float32(0), g * chain), dtype
                                        )
                            T.copy(dx[:valid], DX[row, start : start + valid])

    return main


@tilelang.jit
def simd_cross_entropy_kernel(N, V, dtype, cores, need_grad, stages=2, storage="auto", save_stats=False):
    """Explicit SIMD row owner; raw resident row or stable online statistics.

    API structure adapted from TK softmax_kernel.py's simd_fold_block and
    simd_apply_block; no performance-template claim. Automatic and manual
    representatives share arithmetic, tile size and task mapping.
    """
    narrow = dtype == "bfloat16"
    resident = V <= 32000
    tile = tilelang.cdiv(V, 64) * 64 if resident else (31680 if narrow else 15808)
    full_blocks, tail = divmod(V, tile)
    manual = storage == "manual" and not resident and stages == 2
    active_stages = 1 if resident else stages
    store_dist = "PK_B32" if narrow else "NORM_B32"

    def load(buf, offset, slot):
        if manual:
            raw = S.vld(buf[slot, offset], dist="UNPK_B16" if narrow else "NORM")
        else:
            raw = S.vld(buf[offset], dist="UNPK_B16" if narrow else "NORM")
        return S.vcvt(raw, T.float32, part=0) if narrow else raw

    def store(buf, offset, value, slot, mask):
        converted = S.vcvt(value, T.bfloat16) if narrow else value
        if manual:
            return S.vsts(buf[slot, offset], converted, mask, store_dist)
        else:
            return S.vsts(buf[offset], converted, mask, store_dist)

    @T.macro
    def fold(x, state, target, start, length, slot):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            zero = S.vdup(0.0, T.float32, mask)
            negative = S.vdup(-T.infinity("float32"), T.float32, mask)
            run_max = S.vld(state[0], dist="BRC_B32")
            run_sum = S.vld(state[1], dist="BRC_B32")
            run_target = S.vld(state[2], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            lane_max = S.alloc_var(T.float32)
            lane_max = run_max
            lane_target = S.alloc_var(T.float32)
            lane_target = zero
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                value = S.vsel(load(x, vec * 64, slot), negative, live)
                lane_max = S.vmax(lane_max, value, mask)
                columns = S.vadds(lane, start + vec * 64, mask)
                chosen = S.vcmp(columns, label, live, "eq")
                lane_target = S.vadd(lane_target, S.vsel(value, zero, chosen), mask)
            new_max = S.vdupv(S.vcmax(lane_max, mask), mask)
            lane_sum = S.alloc_var(T.float32)
            lane_sum = zero
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane2 = S.vci(T.int32(0), T.int32)
                live2 = S.vcmps(lane2, length - vec * 64, mask, "lt")
                value2 = S.vsel(load(x, vec * 64, slot), negative, live2)
                lane_sum = S.vadd(lane_sum, S.vexp(S.vsub(value2, new_max, mask), mask), mask)
            block_sum = S.vdupv(S.vcadd(lane_sum, mask), mask)
            merged = S.vadd(S.vmul(S.vexp(S.vsub(run_max, new_max, mask), mask), run_sum, mask), block_sum, mask)
            # The masked target sum has exactly one potentially nonzero lane.
            # Select that known lane in-register instead of a third horizontal
            # reduction per tile. Only max and exp-sum need horizontal reductions.
            target_lane = S.vand(label, S.vdup(63, T.int32, mask), mask)
            merged_target = S.vadd(run_target, S.vselr(lane_target, target_lane), mask)
            S.vsts(state[0], new_max, mask, "ONEPT_B32")
            S.vsts(state[1], merged, mask, "ONEPT_B32")
            S.vsts(state[2], merged_target, mask, "ONEPT_B32")

    @T.macro
    def apply(x, dx, state, target, start, length, slot, ignore_index, denominator):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            max_value = S.vld(state[0], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            zero = S.vdup(0.0, T.float32, mask)
            one = S.vdup(1.0, T.float32, mask)
            if resident:
                # The resident path has one apply call per row; preserve its
                # original in-apply placement and instruction count.
                sum_value = S.vld(state[1], dist="BRC_B32")
                apply_inv_sum = S.vdiv(one, sum_value, mask, precision="exact")
                apply_inv_den = S.vdiv(one, S.vdup(denominator, T.float32, mask), mask, precision="exact")
            else:
                # The exact reciprocal vectors are saved once per row in UB
                # before the pipelined tile loop. Reload them here because
                # TileLang SIMD SSA values cannot cross a T.Pipelined region.
                apply_inv_sum = S.vld(state[3], dist="BRC_B32")
                apply_inv_den = S.vld(state[4], dist="BRC_B32")
            ignored = S.vcmps(label, ignore_index, mask, "eq")
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                columns = S.vadds(lane, start + vec * 64, mask)
                p = S.vmul(S.vexp(S.vsub(load(x, vec * 64, slot), max_value, mask), mask), apply_inv_sum, mask)
                g = S.vmul(
                    S.vsub(p, S.vsel(one, zero, S.vcmp(columns, label, mask, "eq")), mask),
                    apply_inv_den,
                    mask,
                )
                store(dx, vec * 64, S.vsel(zero, g, ignored), slot, live)

    @T.macro
    def row_body(row, x, dx, target, state, loss, zloss, X, Target, Loss, ZLoss, DX, Stats, ignore_index, denominator):
        T.copy(Target[row : row + 1], target[:1])
        with T.SimdVF():
            mask = S.pset(32)
            S.vsts(state[0], S.vdup(-T.infinity("float32"), T.float32, mask), mask, "ONEPT_B32")
            S.vsts(state[1], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
            S.vsts(state[2], S.vdup(0.0, T.float32, mask), mask, "ONEPT_B32")
        if resident:
            T.copy(X[row, :V], x[:V], l2_cache_ctrl="NORMAL_PERS")
            fold(x, state, target, 0, V, 0)
        else:
            for block in T.Pipelined(full_blocks, num_stages=active_stages):
                if manual:
                    if block % 2 == 0:
                        T.copy(X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS")
                        fold(x, state, target, block * tile, tile, 0)
                    else:
                        T.copy(X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS")
                        fold(x, state, target, block * tile, tile, 1)
                else:
                    T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                    fold(x, state, target, block * tile, tile, 0)
            if tail > 0:
                if manual:
                    T.copy(X[row, full_blocks * tile : V], x[0, :tail], l2_cache_ctrl="NORMAL_PERS")
                else:
                    T.copy(X[row, full_blocks * tile : V], x[:tail], l2_cache_ctrl="NORMAL_PERS")
                fold(x, state, target, full_blocks * tile, tail, 0)
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            m = S.vld(state[0], dist="BRC_B32")
            d = S.vld(state[1], dist="BRC_B32")
            chosen = S.vld(state[2], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            zero = S.vdup(0.0, T.float32, mask)
            lse = S.vadd(m, S.vln(d, mask), mask)
            result = S.vdiv(S.vsub(lse, chosen, mask), S.vdup(denominator, T.float32, mask), mask)
            selected = S.vsel(zero, result, S.vcmps(label, ignore_index, mask, "eq"))
            S.vsts(
                loss[0],
                S.vcvt(selected, T.bfloat16) if narrow else selected,
                S.pset(32, "PAT_VL1"),
                store_dist,
                extent=1,
            )
            S.vsts(zloss[0], S.vcvt(zero, T.bfloat16) if narrow else zero, S.pset(32, "PAT_VL1"), store_dist, extent=1)
            if save_stats:
                S.vsts(state[0], lse, S.pset(32, "PAT_VL1"), "ONEPT_B32", extent=1)
        T.copy(loss[:1], Loss[row : row + 1])
        T.copy(zloss[:1], ZLoss[row : row + 1])
        if save_stats:
            T.copy(state[:1], Stats[row, :1])
        if need_grad:
            if resident:
                apply(x, x, state, target, 0, V, 0, ignore_index, denominator)
                T.copy(x[:V], DX[row, :V])
            else:
                # The online fold state and mean denominator are row-invariant.
                # Compute their exact fp32 reciprocals once and reuse them for
                # every tile in the long-V apply pass.
                with T.SimdVF():
                    mask = S.pset(32)
                    one = S.vdup(1.0, T.float32, mask)
                    sum_value = S.vld(state[1], dist="BRC_B32")
                    inv_sum = S.vdiv(one, sum_value, mask, precision="exact")
                    inv_den = S.vdiv(one, S.vdup(denominator, T.float32, mask), mask, precision="exact")
                    S.vsts(state[3], inv_sum, mask, "ONEPT_B32")
                    S.vsts(state[4], inv_den, mask, "ONEPT_B32")
                for block in T.Pipelined(full_blocks, num_stages=active_stages):
                    if manual:
                        if block % 2 == 0:
                            T.copy(X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS")
                            apply(x, dx, state, target, block * tile, tile, 0, ignore_index, denominator)
                            T.copy(dx[0, :], DX[row, block * tile : (block + 1) * tile])
                        else:
                            T.copy(X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS")
                            apply(x, dx, state, target, block * tile, tile, 1, ignore_index, denominator)
                            T.copy(dx[1, :], DX[row, block * tile : (block + 1) * tile])
                    else:
                        T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                        apply(x, dx, state, target, block * tile, tile, 0, ignore_index, denominator)
                        T.copy(dx, DX[row, block * tile : (block + 1) * tile])
                if tail > 0:
                    if manual:
                        T.copy(X[row, full_blocks * tile : V], x[0, :tail], l2_cache_ctrl="NORMAL_PERS")
                    else:
                        T.copy(X[row, full_blocks * tile : V], x[:tail], l2_cache_ctrl="NORMAL_PERS")
                    apply(x, dx, state, target, full_blocks * tile, tail, 0, ignore_index, denominator)
                    if manual:
                        T.copy(dx[0, :tail], DX[row, full_blocks * tile : V])
                    else:
                        T.copy(dx[:tail], DX[row, full_blocks * tile : V])

    @T.prim_func
    def main(
        X: T.Tensor((N, V), dtype),
        Target: T.Tensor((N,), "int32"),
        W: T.Tensor((V,), "float32"),
        Loss: T.Tensor((N,), dtype),
        ZLoss: T.Tensor((N,), dtype),
        DX: T.Tensor((N, V), dtype),
        ignore_index: T.int32,
        denominator: T.float32,
        valid_count: T.float32,
        weight_sum: T.float32,
        Stats: T.Tensor((N, 8), "float32"),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((2, tile) if manual else (tile,), dtype)
            if not resident:
                dx = T.alloc_shared((2, tile) if manual else (tile,), dtype)
            target = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((8,), "float32")
            loss = T.alloc_shared((16 if narrow else 8,), dtype)
            zloss = T.alloc_shared((16 if narrow else 8,), dtype)
            T.annotate_buffer_versions({target: 1, state: 1, loss: 1, zloss: 1})
            if not resident:
                if manual:
                    T.annotate_manual_multi_buffer(x, dx)
                else:
                    T.annotate_buffer_versions({x: active_stages, dx: active_stages})

            for wave in T.serial(N // cores):
                row_body(
                    wave * cores + core,
                    x,
                    x if resident else dx,
                    target,
                    state,
                    loss,
                    zloss,
                    X,
                    Target,
                    Loss,
                    ZLoss,
                    DX,
                    Stats,
                    ignore_index,
                    denominator,
                )
            if core < N % cores:
                row_body(
                    (N // cores) * cores + core,
                    x,
                    x if resident else dx,
                    target,
                    state,
                    loss,
                    zloss,
                    X,
                    Target,
                    Loss,
                    ZLoss,
                    DX,
                    Stats,
                    ignore_index,
                    denominator,
                )

    return main


@tilelang.jit
def simd_scale_gradient_kernel(N, V, dtype, cores, vector_grad, tile=8192, stages=2, storage="auto"):
    """Flat scalar-upstream tasks; vector upstream retains row broadcasts."""
    narrow = dtype == "bfloat16"
    manual = storage == "manual" and stages == 2
    total = N * V
    full_tasks, tail = divmod(total, tile)
    shape = (2, tile) if manual else (tile,)
    store_dist = "PK_B32" if narrow else "NORM_B32"

    @T.macro
    def compute(x, y, grad, length, slot):
        with T.SimdVF():
            mask = S.pset(32)
            # Grad is cast once from its original dtype, then broadcast in fp32.
            raw_grad = S.vld(grad[0], dist="BRC_B16" if narrow else "BRC_B32")
            gr = S.vcvt(raw_grad, T.float32) if narrow else raw_grad
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                if manual:
                    raw = S.vld(x[slot, vec * 64], dist="UNPK_B16" if narrow else "NORM")
                else:
                    raw = S.vld(x[vec * 64], dist="UNPK_B16" if narrow else "NORM")
                value = S.vcvt(raw, T.float32) if narrow else raw
                product = S.vmul(value, gr, mask)
                converted = S.vcvt(product, T.bfloat16) if narrow else product
                if manual:
                    S.vsts(y[slot, vec * 64], converted, live, store_dist)
                else:
                    S.vsts(y[vec * 64], converted, live, store_dist)

    @T.macro
    def task(task_id, length, slot, x, y, grad, DX, Out):
        begin = task_id * tile
        if manual:
            T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + length], x[slot, :length])
        else:
            T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + length], x[:length])
        compute(x, y, grad, length, slot)
        if manual:
            T.copy(y[slot, :length], T.Tensor((N * V,), dtype, Out.data)[begin : begin + length])
        else:
            T.copy(y[:length], T.Tensor((N * V,), dtype, Out.data)[begin : begin + length])

    @T.prim_func
    def main(
        DX: T.Tensor((N, V), dtype), Grad: T.Tensor((N if vector_grad else 1,), dtype), Out: T.Tensor((N, V), dtype)
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared(shape, dtype)
            y = T.alloc_shared(shape, dtype)
            grad = T.alloc_shared((16,), dtype)
            T.annotate_buffer_versions({grad: 1})
            if manual:
                T.annotate_manual_multi_buffer(x, y)
            else:
                T.annotate_buffer_versions({x: stages, y: stages})

            if vector_grad:
                # A vector upstream requires preserving each row's factor; no
                # flattened task may span two rows with different factors.
                for wave in T.serial(tilelang.cdiv(N, cores)):
                    row = wave * cores + core
                    if row < N:
                        T.copy(Grad[row : row + 1], grad[:1])
                        for block in T.serial(tilelang.cdiv(V, tile)):
                            begin = row * V + block * tile
                            valid = T.min(tile, V - block * tile)
                            if manual:
                                T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + valid], x[0, :valid])
                            else:
                                T.copy(T.Tensor((N * V,), dtype, DX.data)[begin : begin + valid], x[:valid])
                            compute(x, y, grad, valid, 0)
                            if manual:
                                T.copy(y[0, :valid], T.Tensor((N * V,), dtype, Out.data)[begin : begin + valid])
                            else:
                                T.copy(y[:valid], T.Tensor((N * V,), dtype, Out.data)[begin : begin + valid])
            else:
                T.copy(Grad[:1], grad[:1])
                for wave in T.Pipelined(full_tasks // cores, num_stages=stages):
                    if manual:
                        if wave % 2 == 0:
                            task(wave * cores + core, tile, 0, x, y, grad, DX, Out)
                        else:
                            task(wave * cores + core, tile, 1, x, y, grad, DX, Out)
                    else:
                        task(wave * cores + core, tile, 0, x, y, grad, DX, Out)
                if core < full_tasks % cores:
                    task((full_tasks // cores) * cores + core, tile, 0, x, y, grad, DX, Out)
                if tail > 0:
                    if core == 0:
                        task(full_tasks, tail, 0, x, y, grad, DX, Out)

    return main


@tilelang.jit
def simd_deferred_gradient_kernel(N, V, dtype, cores, tile=8192, stages=2, storage="auto"):
    """Build the scaled CE gradient from saved logits and per-row logsumexp."""
    narrow = dtype == "bfloat16"
    # Keep both input and output in UB for small rows. The two buffers avoid
    # racing an asynchronous MTE3 writeback against the next row's input copy.
    resident = V <= 32000
    tile = tilelang.cdiv(V, 64) * 64 if resident else (31680 if narrow else 15808)
    full_blocks, tail = divmod(V, tile)
    manual = storage == "manual" and stages == 2 and not resident
    active_stages = 1 if resident else stages
    shape = (2, tile) if manual else (tile,)
    store_dist = "PK_B32" if narrow else "NORM_B32"

    @T.macro
    def apply(x, dx, target, state, grad, length, start, slot, ignore_index):
        with T.SimdVF():
            mask = S.pset(32)
            S.mem_bar("VST_VLD")
            lse = S.vld(state[0], dist="BRC_B32")
            inv_den = S.vld(state[8], dist="BRC_B32")
            label = S.vld(target[0], dist="BRC_B32")
            raw_grad = S.vld(grad[0], dist="BRC_B16" if narrow else "BRC_B32")
            upstream = S.vcvt(raw_grad, T.float32) if narrow else raw_grad
            zero = S.vdup(0.0, T.float32, mask)
            one = S.vdup(1.0, T.float32, mask)
            ignored = S.vcmps(label, ignore_index, mask, "eq")
            for vec in T.serial(tilelang.cdiv(length, 64)):
                lane = S.vci(T.int32(0), T.int32)
                live = S.vcmps(lane, length - vec * 64, mask, "lt")
                columns = S.vadds(lane, start + vec * 64, mask)
                is_target = S.vcmp(columns, label, mask, "eq")
                raw = S.vld(
                    x[slot, vec * 64] if manual else x[vec * 64],
                    dist="UNPK_B16" if narrow else "NORM",
                )
                value = S.vcvt(raw, T.float32) if narrow else raw
                probability = S.vexp(S.vsub(value, lse, mask), mask)
                probability = S.vsub(probability, S.vsel(one, zero, is_target), mask)
                result = S.vmul(S.vmul(probability, upstream, mask), inv_den, mask)
                result = S.vsel(zero, result, ignored)
                converted = S.vcvt(result, T.bfloat16) if narrow else result
                if manual:
                    S.vsts(dx[slot, vec * 64], converted, live, store_dist)
                else:
                    S.vsts(dx[vec * 64], converted, live, store_dist)

    @T.prim_func
    def main(
        X: T.Tensor((N, V), dtype),
        Target: T.Tensor((N,), "int32"),
        Stats: T.Tensor((N, 8), "float32"),
        Grad: T.Tensor((1,), dtype),
        DX: T.Tensor((N, V), dtype),
        ignore_index: T.int32,
        denominator: T.float32,
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared(shape, dtype)
            if not resident:
                dx = T.alloc_shared(shape, dtype)
            target = T.alloc_shared((8,), "int32")
            state = T.alloc_shared((16,), "float32")
            grad = T.alloc_shared((16,), dtype)
            T.annotate_buffer_versions({target: 1, state: 1, grad: 1})
            if manual:
                T.annotate_manual_multi_buffer(x, dx)
            elif not resident:
                T.annotate_buffer_versions({x: active_stages, dx: active_stages})

            for wave in T.serial(tilelang.cdiv(N, cores)):
                row = wave * cores + core
                if row < N:
                    T.copy(Target[row : row + 1], target[:1])
                    T.copy(Stats[row, :1], state[:1])
                    T.copy(Grad[:1], grad[:1])
                    with T.SimdVF():
                        mask = S.pset(32)
                        one = S.vdup(1.0, T.float32, mask)
                        inv_den = S.vdiv(
                            one,
                            S.vdup(denominator, T.float32, mask),
                            mask,
                            precision="exact",
                        )
                        # Keep the scalar reciprocal at a 32-byte aligned slot.
                        S.vsts(state[8], inv_den, mask, "ONEPT_B32")
                    if resident:
                        T.copy(X[row, :V], x[:V], l2_cache_ctrl="NORMAL_PERS")
                        apply(x, x, target, state, grad, V, 0, 0, ignore_index)
                        T.copy(x[:V], DX[row, :V])
                    else:
                        for block in T.Pipelined(full_blocks, num_stages=active_stages):
                            if manual:
                                if block % 2 == 0:
                                    T.copy(
                                        X[row, block * tile : (block + 1) * tile], x[0, :], l2_cache_ctrl="NORMAL_PERS"
                                    )
                                    apply(x, dx, target, state, grad, tile, block * tile, 0, ignore_index)
                                    T.copy(dx[0, :], DX[row, block * tile : (block + 1) * tile])
                                else:
                                    T.copy(
                                        X[row, block * tile : (block + 1) * tile], x[1, :], l2_cache_ctrl="NORMAL_PERS"
                                    )
                                    apply(x, dx, target, state, grad, tile, block * tile, 1, ignore_index)
                                    T.copy(dx[1, :], DX[row, block * tile : (block + 1) * tile])
                            else:
                                T.copy(X[row, block * tile : (block + 1) * tile], x, l2_cache_ctrl="NORMAL_PERS")
                                apply(x, dx, target, state, grad, tile, block * tile, 0, ignore_index)
                                T.copy(dx, DX[row, block * tile : (block + 1) * tile])
                        if tail > 0:
                            T.copy(
                                X[row, full_blocks * tile : V],
                                x[0, :tail] if manual else x[:tail],
                                l2_cache_ctrl="NORMAL_PERS",
                            )
                            apply(x, dx, target, state, grad, tail, full_blocks * tile, 0, ignore_index)
                            if manual:
                                T.copy(dx[0, :tail], DX[row, full_blocks * tile : V])
                            else:
                                T.copy(dx[:tail], DX[row, full_blocks * tile : V])

    return main


def cross_entropy_kernel(
    N,
    V,
    dtype,
    cores,
    tile,
    has_weight,
    smoothing,
    softcap,
    z_scale,
    need_grad,
    pipeline_stages=2,
    pipeline_storage="auto",
    save_stats=False,
):
    if not has_weight and smoothing == 0 and softcap == 0 and z_scale == 0:
        return simd_cross_entropy_kernel(
            N, V, dtype, cores, need_grad, stages=pipeline_stages, storage=pipeline_storage, save_stats=save_stats
        )
    return baseline_cross_entropy_kernel(N, V, dtype, cores, tile, has_weight, smoothing, softcap, z_scale, need_grad)


def scale_gradient_kernel(N, V, dtype, cores, vector_grad, tile=8192, pipeline_stages=2, pipeline_storage="auto"):
    # Same resource-bound tile as the streaming forward representative.
    chosen_tile = 31680 if dtype == "bfloat16" else 15808
    return simd_scale_gradient_kernel(
        N,
        V,
        dtype,
        min(56, tilelang.cdiv(N * V, chosen_tile)),
        vector_grad,
        chosen_tile,
        stages=pipeline_stages,
        storage=pipeline_storage,
    )


@tilelang.jit
def baseline_scale_gradient_kernel(N, V, dtype, cores, vector_grad, tile=8192):
    """Apply the autograd upstream gradient without mutating saved gradients."""

    @T.prim_func
    def main(
        DX: T.Tensor((N, V), dtype),
        Grad: T.Tensor((N if vector_grad else 1,), dtype),
        Out: T.Tensor((N, V), dtype),
    ):
        with T.Kernel(cores) as core:
            x = T.alloc_shared((tile,), dtype)
            y = T.alloc_shared((tile,), dtype)
            grad = T.alloc_shared((8,), dtype)
            for step in T.serial(tilelang.cdiv(N, cores)):
                row = step * cores + core
                if row < N:
                    gr = row if vector_grad else 0
                    T.copy(Grad[gr : gr + 1], grad[:1])
                    for block in T.serial(tilelang.cdiv(V, tile)):
                        start = block * tile
                        valid = T.min(tile, V - start)
                        T.copy(DX[row, start : start + valid], x[:valid])
                        with T.SimtVF(threads=128):
                            for i in T.Parallel(tile):
                                if i < valid:
                                    y[i] = T.cast(T.cast(x[i], "float32") * T.cast(grad[0], "float32"), dtype)
                        T.copy(y[:valid], Out[row, start : start + valid])

    return main
