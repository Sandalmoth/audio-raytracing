const std = @import("std");

const lin = @import("math/lin.zig");
const pga = @import("math/pga.zig");

pub const Vec = lin.Vec;
pub const Mat = lin.Mat;
// pub const Point = pga.Point;
// pub const Line = pga.Line;
// pub const Motor = pga.Motor;

pub const Vec2f = lin.Vec2f;
pub const Vec2i = lin.Vec2i;

fn addRetType(comptime A: type, comptime B: type) type {
    if (A == Vec and B == Vec) return Vec;
    if (A == Vec2i and B == Vec2i) return Vec2i;
    if (A == Vec2f and B == Vec2f) return Vec2f;
    @compileError("add not supported for types " ++ @typeName(A) ++ " " ++ @typeName(B));
}
pub fn add(a: anytype, b: anytype) addRetType(@TypeOf(a), @TypeOf(b)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    if (A == Vec and B == Vec) return lin.addVecVec(a, b);
    if (A == Vec2i and B == Vec2i) return lin.addVec2iVec2i(a, b);
    if (A == Vec2f and B == Vec2f) return lin.addVec2fVec2f(a, b);
    @compileError("add not supported for types " ++ @typeName(A) ++ " " ++ @typeName(B));
}
