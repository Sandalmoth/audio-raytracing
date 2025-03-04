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

fn subRetType(comptime A: type, comptime B: type) type {
    if (A == Vec and B == Vec) return Vec;
    if (A == Vec2i and B == Vec2i) return Vec2i;
    if (A == Vec2f and B == Vec2f) return Vec2f;
    @compileError("sub not supported for types " ++ @typeName(A) ++ " " ++ @typeName(B));
}
pub fn sub(a: anytype, b: anytype) subRetType(@TypeOf(a), @TypeOf(b)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    if (A == Vec and B == Vec) return lin.subVecVec(a, b);
    if (A == Vec2i and B == Vec2i) return lin.subVec2iVec2i(a, b);
    if (A == Vec2f and B == Vec2f) return lin.subVec2fVec2f(a, b);
    @compileError("add not supported for types " ++ @typeName(A) ++ " " ++ @typeName(B));
}

fn normalizeRetType(comptime A: type) type {
    if (A == Vec) return Vec;
    @compileError("normalize not supported for type" ++ @typeName(A));
}
pub fn normalize(a: anytype) normalizeRetType(@TypeOf(a)) {
    const A = @TypeOf(a);
    if (A == Vec) return lin.normalizeVec(a);
    @compileError("normalize not supported for types " ++ @typeName(A));
}
