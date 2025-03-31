const std = @import("std");

// seven bih nodes are packed into a single cache aligned chunk
// and all their children are stored contiguously
// data: upper/lower bound of left/right child or an index and number of children if leaf
//    0
//   / \
//  1   2
// / \ / \
// 3 4 5 6
// children: index of run of 8 Node (left/right per child)
// info: 2 bits pert node (in high->low order, so 0b11 is node 0, 0b1100 is node 1, etc)
//       detailing if it's a leaf (== 3) or the split axis (<3)

// leaves can hold between 0 and some fixed size number of values
// dependent on level in node, we want to fully utilize the node while avoiding creating more
const max_leaf_sizes = [_]u32{ 64, 128, 128, 256, 256, 256, 256 };
const max_leaf_size = blk: {
    var max: u32 = 0;
    for (max_leaf_sizes) |leaf_size| max = @max(max, leaf_size);
    break :blk max;
};

const Node = extern struct {
    data: [7]extern union {
        internal: extern struct { left_max: f32, right_min: f32 },
        leaf: extern struct { index: u32, count: u32 },
    } align(64),
    children: u32,
    info: u32,

    fn isLeaf(node: Node, ix: u32) bool {
        return (node.info >> @intCast(2 * ix)) & 0b11 == 3;
    }

    fn axis(node: Node, ix: u32) u32 {
        const a = (node.info >> @intCast(2 * ix)) & 0b11;
        std.debug.assert(a < 3);
        return a;
    }

    fn set(node: *Node, n: u32, ix: u32) void {
        const mask: u32 = @as(u32, 0b11) << @intCast(ix * 2);
        node.info = (node.info & ~mask) | (n << @intCast(ix * 2));
    }
};

comptime {
    std.debug.assert(@sizeOf(Node) == 64);
    std.debug.assert(@alignOf(Node) == 64);
}

const BoundingBox = struct {
    low: @Vector(3, f32),
    high: @Vector(3, f32),

    fn center(bb: BoundingBox) @Vector(3, f32) {
        return @as(@Vector(3, f32), @splat(0.5)) * (bb.low + bb.high);
    }

    fn intersects(a: BoundingBox, b: BoundingBox) bool {
        const lh = a.low > b.high;
        const rh = a.high < b.low;
        return !(@reduce(.Or, lh) or @reduce(.Or, rh));
    }

    fn raydist(bb: BoundingBox, src: @Vector(3, f32), idir: @Vector(3, f32)) ?f32 {
        var tmin: f32 = 0.0;
        var tmax: f32 = std.math.inf(f32);

        const t1 = (bb.low - src) * idir;
        const t2 = (bb.high - src) * idir;
        for (0..3) |i| {
            tmin = @min(@max(t1[i], tmin), @max(t2[i], tmin));
            tmax = @max(@min(t1[i], tmax), @min(t2[i], tmax));
        }
        return if (tmin <= tmax) tmin else null;
    }
};

const BoundingBox2 = struct {
    low_x: f32,
    low_y: f32,
    low_z: f32,
    high_x: f32,
    high_y: f32,
    high_z: f32,
};

fn raydistSoA(
    index: u32,
    count: u32,
    slices: std.MultiArrayList(BoundingBox2).Slice,
    src: @Vector(3, f32),
    idir: @Vector(3, f32),
    dists: []f32,
) void {
    const lane_count = 8;

    const src0: @Vector(lane_count, f32) = @splat(src[0]);
    const src1: @Vector(lane_count, f32) = @splat(src[1]);
    const src2: @Vector(lane_count, f32) = @splat(src[2]);
    const idir0: @Vector(lane_count, f32) = @splat(idir[0]);
    const idir1: @Vector(lane_count, f32) = @splat(idir[1]);
    const idir2: @Vector(lane_count, f32) = @splat(idir[2]);

    const low_x = slices.items(.low_x)[index .. index + count];
    const low_y = slices.items(.low_y)[index .. index + count];
    const low_z = slices.items(.low_z)[index .. index + count];
    const high_x = slices.items(.high_x)[index .. index + count];
    const high_y = slices.items(.high_y)[index .. index + count];
    const high_z = slices.items(.high_z)[index .. index + count];

    var i: usize = 0;
    while (i + lane_count <= count) : (i += lane_count) {
        const bb_low_x: @Vector(lane_count, f32) = low_x[i..][0..lane_count].*;
        const bb_low_y: @Vector(lane_count, f32) = low_y[i..][0..lane_count].*;
        const bb_low_z: @Vector(lane_count, f32) = low_z[i..][0..lane_count].*;
        const bb_high_x: @Vector(lane_count, f32) = high_x[i..][0..lane_count].*;
        const bb_high_y: @Vector(lane_count, f32) = high_y[i..][0..lane_count].*;
        const bb_high_z: @Vector(lane_count, f32) = high_z[i..][0..lane_count].*;

        const t1_x = (bb_low_x - src0) * idir0;
        const t1_y = (bb_low_y - src1) * idir1;
        const t1_z = (bb_low_z - src2) * idir2;

        const t2_x = (bb_high_x - src0) * idir0;
        const t2_y = (bb_high_y - src1) * idir1;
        const t2_z = (bb_high_z - src2) * idir2;

        const tmin = @max(@max(@min(t1_x, t2_x), @min(t1_y, t2_y)), @min(t1_z, t2_z));
        const tmax = @min(@min(@max(t1_x, t2_x), @max(t1_y, t2_y)), @max(t1_z, t2_z));

        dists[i..][0..lane_count].* = @select(
            f32,
            tmin <= tmax,
            tmin,
            @as(@Vector(lane_count, f32), @splat(-1)),
        );
    }

    while (i < count) : (i += 1) {
        const aabb = BoundingBox{
            .low = .{ low_x[i], low_y[i], low_z[i] },
            .high = .{ high_x[i], high_y[i], high_z[i] },
        };
        dists[i] = aabb.raydist(src, idir) orelse -1;
    }
}

comptime {
    std.debug.assert(@sizeOf(Node) <= 64);
    std.debug.assert(@alignOf(Node) == 64);
}

pub fn Space(comptime T: type) type {
    return struct {
        const _Space = @This();

        alloc: std.mem.Allocator,
        nodes: std.ArrayListUnmanaged(Node),
        aabbs: std.MultiArrayList(BoundingBox2),
        values: std.ArrayListUnmanaged(T),

        pub fn deinit(space: *_Space) void {
            space.nodes.deinit(space.alloc);
            space.aabbs.deinit(space.alloc);
            space.values.deinit(space.alloc);
        }

        pub fn raycastCapacity(
            space: _Space,
            src: [3]f32,
            _dir: [3]f32,
            comptime capacity: comptime_int,
        ) struct { [capacity]T, usize } {
            std.debug.assert(_dir[0] * _dir[0] + _dir[1] * _dir[1] + _dir[2] * _dir[2] > 0);
            if (space.nodes.items.len == 0) return .{ undefined, 0 };
            const inorm = 1 / @sqrt(_dir[0] * _dir[0] + _dir[1] * _dir[1] + _dir[2] * _dir[2]);
            const dir = @as(@Vector(3, f32), _dir) * @as(@Vector(3, f32), @splat(inorm));
            const idir = @as(@Vector(3, f32), @splat(1.0)) / dir;
            var values: [capacity]T = undefined;
            var dists: [capacity]f32 = undefined;
            var count: usize = 0;
            space.raycastCapacityImpl(space.nodes.items[0], 0, src, idir, &values, &dists, &count);
            return .{ values, count };
        }

        pub fn raycastCapacityImpl(
            space: _Space,
            node: Node,
            index: u32,
            src: @Vector(3, f32),
            idir: @Vector(3, f32),
            values: []T,
            dists: []f32,
            count: *usize,
        ) void {
            if (node.isLeaf(index)) {
                const leaf = node.data[index].leaf;
                var results: [max_leaf_size]f32 = undefined;

                const slices = space.aabbs.slice();
                raydistSoA(leaf.index, leaf.count, slices, src, idir, results[0..leaf.count]);
                for (0..leaf.count) |i| {
                    if (results[i] < 0) continue;
                    _ = insertOrdered(
                        space.values.items[leaf.index + i],
                        results[i],
                        values,
                        dists,
                        count,
                    );
                }
            } else {
                const internal = node.data[index].internal;
                const axis = node.axis(index);

                if (index < 3) {
                    if (idir[axis] < 0) {
                        if (src[axis] >= internal.right_min) {
                            space.raycastCapacityImpl(
                                node,
                                2 * index + 2,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                        if (src[axis] <= internal.left_max) {
                            space.raycastCapacityImpl(
                                node,
                                2 * index + 1,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                    } else {
                        if (src[axis] <= internal.left_max) {
                            space.raycastCapacityImpl(
                                node,
                                2 * index + 1,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                        if (src[axis] >= internal.right_min) {
                            space.raycastCapacityImpl(
                                node,
                                2 * index + 2,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                    }
                } else {
                    if (idir[axis] < 0) {
                        if (src[axis] >= internal.right_min) {
                            space.raycastCapacityImpl(
                                space.nodes.items[node.children + 2 * (index - 3) + 1],
                                0,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                        if (src[axis] <= internal.left_max) {
                            space.raycastCapacityImpl(
                                space.nodes.items[node.children + 2 * (index - 3)],
                                0,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                    } else {
                        if (src[axis] <= internal.left_max) {
                            space.raycastCapacityImpl(
                                space.nodes.items[node.children + 2 * (index - 3)],
                                0,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                        if (src[axis] >= internal.right_min) {
                            space.raycastCapacityImpl(
                                space.nodes.items[node.children + 2 * (index - 3) + 1],
                                0,
                                src,
                                idir,
                                values,
                                dists,
                                count,
                            );
                        }
                    }
                }
            }
        }

        fn compareF32(context: f32, item: f32) std.math.Order {
            return std.math.order(context, item);
        }

        fn insertOrdered(value: T, dist: f32, values: []T, dists: []f32, count: *usize) bool {
            const index = std.sort.lowerBound(f32, dists[0..count.*], dist, compareF32);
            if (index >= dists.len) return false;
            if (index < count.*) {
                std.mem.copyBackwards(
                    f32,
                    dists[index + 1 .. dists.len],
                    dists[index .. dists.len - 1],
                );
                std.mem.copyBackwards(
                    T,
                    values[index + 1 .. dists.len],
                    values[index .. dists.len - 1],
                );
            }
            count.* += 1;
            dists[index] = dist;
            values[index] = value;
            return true;
        }

        pub fn depth(space: _Space) usize {
            if (space.nodes.items.len == 0) return 0;
            return space.depthImpl(space.nodes.items[0], 0);
        }

        fn depthImpl(space: _Space, node: Node, index: u32) usize {
            if (node.isLeaf(index)) return 1;
            if (index < 3) {
                return 1 + @max(
                    space.depthImpl(node, 2 * index + 1),
                    space.depthImpl(node, 2 * index + 2),
                );
            } else {
                return 1 + @max(
                    space.depthImpl(space.nodes.items[node.children + 2 * (index - 3)], 0),
                    space.depthImpl(space.nodes.items[node.children + 2 * (index - 3) + 1], 0),
                );
            }
        }
    };
}

pub fn Builder(comptime T: type) type {
    return struct {
        const _Builder = @This();
        const _Space = Space(T);

        space: _Space,
        center_aabb: BoundingBox,

        pub fn init(alloc: std.mem.Allocator) _Builder {
            return .{
                .space = .{
                    .alloc = alloc,
                    .nodes = std.ArrayListUnmanaged(Node){},
                    .aabbs = std.MultiArrayList(BoundingBox2){},
                    .values = std.ArrayListUnmanaged(T){},
                },
                .center_aabb = .{ .low = .{ 0, 0, 0 }, .high = .{ 0, 0, 0 } },
            };
        }

        pub fn ensureUnusedCapacity(builder: *_Builder, n: usize) !void {
            try builder.space.aabbs.ensureUnusedCapacity(builder.space.alloc, n);
            try builder.space.values.ensureUnusedCapacity(builder.space.alloc, n);
        }

        pub fn ensureTotalCapacity(builder: *_Builder, n: usize) !void {
            try builder.space.aabbs.ensureTotalCapacity(builder.space.alloc, n);
            try builder.space.values.ensureTotalCapacity(builder.space.alloc, n);
        }

        pub fn add(builder: *_Builder, low: [3]f32, high: [3]f32, value: T) !void {
            try builder.ensureUnusedCapacity(1);
            builder.addAssumeCapacity(low, high, value);
        }

        pub fn addAssumeCapacity(builder: *_Builder, low: [3]f32, high: [3]f32, value: T) void {
            std.debug.assert(low[0] <= high[0]);
            std.debug.assert(low[1] <= high[1]);
            std.debug.assert(low[2] <= high[2]);
            std.debug.assert(high[0] - low[0] + high[1] - low[1] + high[2] - low[2] > 0);
            const c = (BoundingBox{ .low = low, .high = high }).center();
            if (builder.space.values.items.len == 0) {
                builder.center_aabb.low = c;
                builder.center_aabb.high = c;
            } else {
                builder.center_aabb.low = @min(builder.center_aabb.low, c);
                builder.center_aabb.high = @max(builder.center_aabb.high, c);
            }
            builder.space.aabbs.appendAssumeCapacity(.{
                .low_x = low[0],
                .low_y = low[1],
                .low_z = low[2],
                .high_x = high[0],
                .high_y = high[1],
                .high_z = high[2],
            });
            builder.space.values.appendAssumeCapacity(value);
        }

        pub fn finish(builder: *_Builder) !_Space {
            if (builder.space.values.items.len == 0) {
                return builder.space;
            }

            try builder.space.nodes.ensureTotalCapacity(
                builder.space.alloc,
                1 + 8 * (builder.space.values.items.len / 3 + 1),
            ); // TODO validate somehow that this is the correct upper bound
            try builder.space.nodes.append(builder.space.alloc, .{
                .data = undefined,
                .info = undefined,
                .children = std.math.maxInt(u32),
            });
            finishImpl(
                &builder.space,
                &builder.space.nodes.items[0],
                0,
                builder.center_aabb,
                0,
                @intCast(builder.space.values.items.len),
            );

            return builder.space;
        }

        fn finishImpl(
            space: *_Space,
            node: *Node,
            index: u32,
            aabb: BoundingBox,
            value_index: u32,
            value_count: u32,
        ) void {
            if (value_count <= max_leaf_sizes[index]) {
                node.data[index].leaf = .{ .index = value_index, .count = value_count };
                node.set(3, index);
                return;
            }

            const split, const axis = blk: {
                const d = aabb.high - aabb.low;
                if (d[0] > d[1] and d[0] > d[2]) {
                    break :blk .{ 0.5 * (aabb.low[0] + aabb.high[0]), @as(u32, 0) };
                } else if (d[1] > d[2]) {
                    break :blk .{ 0.5 * (aabb.low[1] + aabb.high[1]), @as(u32, 1) };
                } else {
                    break :blk .{ 0.5 * (aabb.low[2] + aabb.high[2]), @as(u32, 2) };
                }
            };

            var head: u32 = value_index;
            var tail: u32 = value_index + value_count - 1;
            var left_center_min = aabb.high;
            var left_center_max = aabb.low;
            var right_center_min = aabb.high;
            var right_center_max = aabb.low;
            var left_max = aabb.low[axis];
            var right_min = aabb.high[axis];
            const slices = space.aabbs.slice();
            const low_x = slices.items(.low_x);
            const low_y = slices.items(.low_y);
            const low_z = slices.items(.low_z);
            const high_x = slices.items(.high_x);
            const high_y = slices.items(.high_y);
            const high_z = slices.items(.high_z);
            for (0..value_count) |_| {
                const value_aabb = BoundingBox{
                    .low = .{ low_x[head], low_y[head], low_z[head] },
                    .high = .{ high_x[head], high_y[head], high_z[head] },
                };
                const value_center = value_aabb.center();
                if (value_center[axis] < split) {
                    left_center_min = @min(left_center_min, value_center);
                    left_center_max = @max(left_center_max, value_center);
                    left_max = @max(left_max, value_aabb.high[axis]);
                    head +%= 1;
                } else {
                    right_center_min = @min(right_center_min, value_center);
                    right_center_max = @max(right_center_max, value_center);
                    right_min = @min(right_min, value_aabb.low[axis]);
                    std.mem.swap(f32, &low_x[head], &low_x[tail]);
                    std.mem.swap(f32, &low_y[head], &low_y[tail]);
                    std.mem.swap(f32, &low_z[head], &low_z[tail]);
                    std.mem.swap(f32, &high_x[head], &high_x[tail]);
                    std.mem.swap(f32, &high_y[head], &high_y[tail]);
                    std.mem.swap(f32, &high_z[head], &high_z[tail]);
                    std.mem.swap(T, &space.values.items[head], &space.values.items[tail]);
                    tail -%= 1;
                }
            }
            const ix_split = head;

            node.data[index].internal = .{ .left_max = left_max, .right_min = right_min };
            node.set(axis, index);

            if (index < 3) {
                finishImpl(
                    space,
                    node,
                    2 * index + 1,
                    .{ .low = left_center_min, .high = left_center_max },
                    value_index,
                    ix_split - value_index,
                );
                finishImpl(
                    space,
                    node,
                    2 * index + 2,
                    .{ .low = right_center_min, .high = right_center_max },
                    ix_split,
                    value_count + value_index - ix_split,
                );
            } else {
                if (node.children == std.math.maxInt(u32)) {
                    node.children = @intCast(space.nodes.items.len);
                    space.nodes.appendNTimesAssumeCapacity(.{
                        .data = undefined,
                        .info = undefined,
                        .children = std.math.maxInt(u32),
                    }, 8);
                }
                finishImpl(
                    space,
                    &space.nodes.items[node.children + 2 * (index - 3)],
                    0,
                    .{ .low = left_center_min, .high = left_center_max },
                    value_index,
                    ix_split - value_index,
                );
                finishImpl(
                    space,
                    &space.nodes.items[node.children + 2 * (index - 3) + 1],
                    0,
                    .{ .low = right_center_min, .high = right_center_max },
                    ix_split,
                    value_count + value_index - ix_split,
                );
            }
        }
    };
}

test "fuzz raycastCapacity" {
    var rng = std.Random.DefaultPrng.init(@bitCast(std.time.microTimestamp() *% 2197057547));
    const rand = rng.random();

    const N = 100;
    const M = 100;

    for (0..M) |_| {
        var a = std.ArrayList(struct { BoundingBox, u32 }).init(std.testing.allocator);
        defer a.deinit();

        var b = Builder(u32).init(std.testing.allocator);
        for (0..N) |i| {
            const x = (rand.float(f32) - 0.5) * 20;
            const y = (rand.float(f32) - 0.5) * 20;
            const z = (rand.float(f32) - 0.5) * 20;
            const dx = rand.float(f32) * 5;
            const dy = rand.float(f32) * 5;
            const dz = rand.float(f32) * 5;
            const bb = BoundingBox{
                .low = .{ x - dx, y - dy, z - dz },
                .high = .{ x + dx, y + dy, z + dz },
            };
            try a.append(.{ bb, @intCast(i) });
            try b.add(bb.low, bb.high, @intCast(i));
        }
        var s = try b.finish();
        defer s.deinit();

        for (0..N) |_| {
            const x = (rand.float(f32) - 0.5) * 20;
            const y = (rand.float(f32) - 0.5) * 20;
            const z = (rand.float(f32) - 0.5) * 20;
            const dx = rand.float(f32) - 0.5;
            const dy = rand.float(f32) - 0.5;
            const dz = rand.float(f32) - 0.5;
            const src: [3]f32 = .{ x, y, z };
            const inorm = 1 / @sqrt(dx * dx + dy * dy + dz * dz);
            const dir: [3]f32 = .{ dx * inorm, dy * inorm, dz * inorm };

            const isects, const count = s.raycastCapacity(src, dir, 128);
            if (count == 128) continue;

            var count2: usize = 0;
            var min: f32 = std.math.inf(f32);
            var best: u32 = std.math.maxInt(u32);
            for (a.items) |pair| {
                const bb, const val = pair;
                const idir = @as(@Vector(3, f32), @splat(1.0)) / dir;
                const dist = bb.raydist(src, idir) orelse continue;
                if (dist < min) {
                    min = dist;
                    best = val;
                }
                count2 += 1;
            }
            try std.testing.expect(count <= count2);
            if (count > 0 and min > 0) try std.testing.expect(isects[0] == best);
        }
    }
}
