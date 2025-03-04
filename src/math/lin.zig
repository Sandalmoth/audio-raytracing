const std = @import("std");

const pga = @import("pga.zig");

const Point = pga.Point;
const Dir = pga.Dir;
const Line = pga.Line;
const Motor = pga.Motor;

/// linear algebra vector, homogenous coordinates
pub const Vec = struct {
    data: @Vector(4, f32),

    pub const zeros = Vec{ .data = .{ 0, 0, 0, 0 } };
    pub const ones = Vec{ .data = .{ 1, 1, 1, 1 } };
    pub const origin = Vec{ .data = .{ 0, 0, 0, 1 } };
};

/// linear algebra matrix
/// column major
pub const Mat = struct {
    data: [4]@Vector(4, f32),

    pub const identity = Mat{
        .data = .{
            .{ 1, 0, 0, 0 },
            .{ 0, 1, 0, 0 },
            .{ 0, 0, 1, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
};

pub const Vec2f = struct {
    x: f32,
    y: f32,

    pub const zeros = Vec2f{ .x = 0, .y = 0 };
    pub const ones = Vec2f{ .x = 1, .y = 1 };
};

pub const Vec2i = struct {
    x: i32,
    y: i32,

    pub const zeros = Vec2i{ .x = 0, .y = 0 };
    pub const ones = Vec2i{ .x = 1, .y = 1 };
};

pub fn addVecVec(a: Vec, b: Vec) Vec {
    return .{ .data = a.data + b.data };
}

pub fn addVec2fVec2f(a: Vec2f, b: Vec2f) Vec2f {
    return .{ .x = a.x + b.x, .y = a.y + b.y };
}

pub fn addVec2iVec2i(a: Vec2i, b: Vec2i) Vec2i {
    return .{ .x = a.x + b.x, .y = a.y + b.y };
}

pub fn subVecVec(a: Vec, b: Vec) Vec {
    return .{ .data = a.data - b.data };
}

pub fn subVec2fVec2f(a: Vec2f, b: Vec2f) Vec2f {
    return .{ .x = a.x - b.x, .y = a.y - b.y };
}

pub fn subVec2iVec2i(a: Vec2i, b: Vec2i) Vec2i {
    return .{ .x = a.x - b.x, .y = a.y - b.y };
}

pub fn mulFloatVec(a: f32, b: Vec) Vec {
    return .{ .data = @as(@Vector(4, f32), @splat(a)) * b.data };
}

pub fn normalizeVec(a: Vec) Vec {
    const a2 = a.data * a.data;
    const inorm = 1.0 / (a2[0] + a2[1] + a2[2]);
    const b: @Vector(4, f32) = .{ inorm, inorm, inorm, 1.0 };
    return .{ .data = a.data * b };
}
