//! SIMD 3-D math for the render and physics hot paths.
//!
//! Everything is built on [`glam::Vec3A`], a 16-byte SIMD vector: on aarch64 it
//! lowers to NEON, on x86_64 to SSE2. These are the operations the launcher
//! performs per entity and per frame: AABB overlap, swept collision resolution,
//! ray casts and frustum culling.
//!
//! The collision model matches the voxel world: a moving AABB is swept against a
//! set of static solid AABBs and resolved one axis at a time, which is what
//! stops a player from tunnelling through a block corner.

use glam::{Mat4, Vec3A};

/// A SIMD 3-vector. Re-exported so callers do not depend on glam directly.
pub type Vec3 = Vec3A;

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    #[inline]
    pub const fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    /// Builds a box from a centre and half-extents.
    #[inline]
    pub fn from_center_half(center: Vec3A, half: Vec3A) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    #[inline]
    pub fn center(&self) -> Vec3A {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn half_extents(&self) -> Vec3A {
        (self.max - self.min) * 0.5
    }

    #[inline]
    pub fn translated(&self, delta: Vec3A) -> Self {
        Self {
            min: self.min + delta,
            max: self.max + delta,
        }
    }

    /// Grows the box by `by` on every side.
    #[inline]
    pub fn expanded(&self, by: f32) -> Self {
        let d = Vec3A::splat(by);
        Self {
            min: self.min - d,
            max: self.max + d,
        }
    }

    /// True when the two boxes overlap (touching counts as overlapping).
    ///
    /// SIMD: three component compares, ANDed, then a horizontal reduction.
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        (self.min.cmple(other.max) & self.max.cmpge(other.min)).all()
    }

    #[inline]
    pub fn contains_point(&self, p: Vec3A) -> bool {
        (p.cmpge(self.min) & p.cmple(self.max)).all()
    }
}

/// A ray hit: the parameter `t` and the face normal of the entry point.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub t: f32,
    pub normal: Vec3A,
}

/// One axis slab test. `d == 0` is handled explicitly so an origin exactly on a
/// slab plane cannot produce a `0 * inf` NaN and silently reject the hit.
#[inline]
fn slab(o: f32, d: f32, lo: f32, hi: f32) -> (f32, f32) {
    if d.abs() < 1e-12 {
        if o < lo || o > hi {
            (f32::INFINITY, f32::NEG_INFINITY)
        } else {
            (f32::NEG_INFINITY, f32::INFINITY)
        }
    } else {
        let inv = 1.0 / d;
        let t1 = (lo - o) * inv;
        let t2 = (hi - o) * inv;
        if t1 > t2 {
            (t2, t1)
        } else {
            (t1, t2)
        }
    }
}

/// Casts a ray against an AABB. Returns the nearest forward hit, or `None`.
pub fn ray_aabb(origin: Vec3A, dir: Vec3A, b: &Aabb) -> Option<RayHit> {
    let (tx0, tx1) = slab(origin.x, dir.x, b.min.x, b.max.x);
    let (ty0, ty1) = slab(origin.y, dir.y, b.min.y, b.max.y);
    let (tz0, tz1) = slab(origin.z, dir.z, b.min.z, b.max.z);

    let t_enter = tx0.max(ty0).max(tz0);
    let t_exit = tx1.min(ty1).min(tz1);
    if t_enter > t_exit || t_exit < 0.0 {
        return None;
    }

    let t = t_enter.max(0.0);
    let normal = if t_enter == tx0 {
        Vec3A::new(-dir.x.signum(), 0.0, 0.0)
    } else if t_enter == ty0 {
        Vec3A::new(0.0, -dir.y.signum(), 0.0)
    } else {
        Vec3A::new(0.0, 0.0, -dir.z.signum())
    };
    Some(RayHit { t, normal })
}

/// The outcome of [`resolve_movement`].
#[derive(Clone, Copy, Debug)]
pub struct CollisionResult {
    /// The final, non-penetrating box.
    pub aabb: Aabb,
    pub hit_x: bool,
    pub hit_y: bool,
    /// True when the collision happened on the ground (a downward `+Y` stop).
    pub on_ground: bool,
    pub hit_z: bool,
}

/// Time of impact of a swept AABB against a static AABB.
///
/// `disp` is the mover's displacement and `size` its dimensions. Returns the
/// entry time in `[0, 1]` and which axis stopped it, or `None` when the sweep
/// does not overlap the obstacle. Using a time of impact rather than "does the
/// destination overlap" is what stops a fast mover tunnelling through a thin
/// wall: the destination may be clear while the path is not.
#[allow(clippy::too_many_arguments)]
#[inline]
fn swept_axis(
    start_min: f32,
    disp: f32,
    size: f32,
    other_min: f32,
    other_max: f32,
    t_enter: &mut f32,
    t_exit: &mut f32,
    hit_axis: &mut u8,
    axis_bit: u8,
) {
    if disp.abs() < 1e-9 {
        return;
    }
    let inv = 1.0 / disp;
    let mut t1 = (other_min - (start_min + size)) * inv;
    let mut t2 = (other_max - start_min) * inv;
    if t1 > t2 {
        std::mem::swap(&mut t1, &mut t2);
    }
    if t1 > *t_enter {
        *t_enter = t1;
        *hit_axis = axis_bit;
    }
    if t2 < *t_exit {
        *t_exit = t2;
    }
}

/// Sweeps `start` by `delta` against `solids`, resolving one axis at a time.
///
/// Each axis is swept independently with a time of impact, then the mover is
/// advanced to the earliest stop and the remaining displacement is zeroed on that
/// axis. X is resolved first, then Y, then Z, so a diagonal move into a corner
/// stops on both walls instead of squeezing through. `solids` is expected to be
/// the small set of boxes near the mover (broad-phase filtered by the caller).
pub fn resolve_movement(start: Aabb, delta: Vec3A, solids: &[Aabb]) -> CollisionResult {
    let size = start.max - start.min;
    let mut result = CollisionResult {
        aabb: start,
        hit_x: false,
        hit_y: false,
        on_ground: false,
        hit_z: false,
    };

    // Small back-off so the resolved box rests exactly against the surface
    // without being counted as overlapping it next tick.
    const EPS: f32 = 1e-4;

    let mut origin = start;

    // --- X ---
    if delta.x.abs() > 1e-9 {
        let mut t_enter = 0.0f32;
        let mut t_exit = 1.0f32;
        let mut hit_axis = 0u8;
        for s in solids {
            if origin.min.y < s.max.y
                && origin.max.y > s.min.y
                && origin.min.z < s.max.z
                && origin.max.z > s.min.z
            {
                swept_axis(
                    origin.min.x,
                    delta.x,
                    size.x,
                    s.min.x,
                    s.max.x,
                    &mut t_enter,
                    &mut t_exit,
                    &mut hit_axis,
                    1,
                );
            }
        }
        if t_enter < t_exit && hit_axis == 1 {
            let move_x = delta.x * t_enter - delta.x.signum() * EPS;
            origin = origin.translated(Vec3A::new(move_x, 0.0, 0.0));
            result.hit_x = true;
        } else {
            origin = origin.translated(Vec3A::new(delta.x, 0.0, 0.0));
        }
    }

    // --- Y ---
    if delta.y.abs() > 1e-9 {
        let mut t_enter = 0.0f32;
        let mut t_exit = 1.0f32;
        let mut hit_axis = 0u8;
        for s in solids {
            if origin.min.x < s.max.x
                && origin.max.x > s.min.x
                && origin.min.z < s.max.z
                && origin.max.z > s.min.z
            {
                swept_axis(
                    origin.min.y,
                    delta.y,
                    size.y,
                    s.min.y,
                    s.max.y,
                    &mut t_enter,
                    &mut t_exit,
                    &mut hit_axis,
                    2,
                );
            }
        }
        if t_enter < t_exit && hit_axis == 2 {
            let move_y = delta.y * t_enter - delta.y.signum() * EPS;
            origin = origin.translated(Vec3A::new(0.0, move_y, 0.0));
            result.hit_y = true;
            if delta.y < 0.0 {
                result.on_ground = true;
            }
        } else {
            origin = origin.translated(Vec3A::new(0.0, delta.y, 0.0));
        }
    }

    // --- Z ---
    if delta.z.abs() > 1e-9 {
        let mut t_enter = 0.0f32;
        let mut t_exit = 1.0f32;
        let mut hit_axis = 0u8;
        for s in solids {
            if origin.min.x < s.max.x
                && origin.max.x > s.min.x
                && origin.min.y < s.max.y
                && origin.max.y > s.min.y
            {
                swept_axis(
                    origin.min.z,
                    delta.z,
                    size.z,
                    s.min.z,
                    s.max.z,
                    &mut t_enter,
                    &mut t_exit,
                    &mut hit_axis,
                    3,
                );
            }
        }
        if t_enter < t_exit && hit_axis == 3 {
            let move_z = delta.z * t_enter - delta.z.signum() * EPS;
            origin = origin.translated(Vec3A::new(0.0, 0.0, move_z));
            result.hit_z = true;
        } else {
            origin = origin.translated(Vec3A::new(0.0, 0.0, delta.z));
        }
    }

    result.aabb = origin;
    result
}

/// A single frustum plane in the form `dot(normal, p) + d = 0`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub normal: Vec3A,
    pub d: f32,
}

/// The six clipping planes of a camera frustum.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

#[inline]
fn row(m: &Mat4, i: usize) -> [f32; 4] {
    [m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]]
}

impl Frustum {
    /// Extracts the frustum from a view-projection matrix (Gribb-Hartmann).
    ///
    /// Uses the OpenGL clip convention (`z` in `[-1, 1]`), which is what the
    /// client's GLES renderer feeds in.
    pub fn from_view_projection(m: Mat4) -> Self {
        let r0 = row(&m, 0);
        let r1 = row(&m, 1);
        let r2 = row(&m, 2);
        let r3 = row(&m, 3);

        let combine = |a: [f32; 4], b: [f32; 4], sub: bool| -> Plane {
            let s = if sub { -1.0 } else { 1.0 };
            let p = [
                a[0] + s * b[0],
                a[1] + s * b[1],
                a[2] + s * b[2],
                a[3] + s * b[3],
            ];
            let n = Vec3A::new(p[0], p[1], p[2]);
            let len = n.length();
            if len > 1e-12 {
                Plane {
                    normal: n / len,
                    d: p[3] / len,
                }
            } else {
                Plane { normal: n, d: p[3] }
            }
        };

        Self {
            planes: [
                combine(r3, r0, false), // left
                combine(r3, r0, true),  // right
                combine(r3, r1, false), // bottom
                combine(r3, r1, true),  // top
                combine(r3, r2, false), // near
                combine(r3, r2, true),  // far
            ],
        }
    }

    /// True when the AABB is inside or intersects the frustum.
    ///
    /// Uses the positive-vertex test: for each plane, only the box corner most
    /// along the plane normal needs to be checked.
    pub fn contains_aabb(&self, b: &Aabb) -> bool {
        for p in &self.planes {
            let pv = Vec3A::new(
                if p.normal.x >= 0.0 { b.max.x } else { b.min.x },
                if p.normal.y >= 0.0 { b.max.y } else { b.min.y },
                if p.normal.z >= 0.0 { b.max.z } else { b.min.z },
            );
            if p.normal.dot(pv) + p.d < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Interpolates between two sampled entity positions (render-time smoothing).
#[inline]
pub fn lerp_position(prev: Vec3A, curr: Vec3A, alpha: f32) -> Vec3A {
    prev + (curr - prev) * alpha
}

/// Integrates a position forward by a velocity over `dt` seconds.
#[inline]
pub fn integrate_position(pos: Vec3A, vel: Vec3A, dt: f32) -> Vec3A {
    pos + vel * dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn box_at(x: f32, y: f32, z: f32, half: f32) -> Aabb {
        Aabb::from_center_half(Vec3A::new(x, y, z), Vec3A::splat(half))
    }

    #[test]
    fn aabb_overlap_and_separation() {
        let a = box_at(0.0, 0.0, 0.0, 0.5);
        let b = box_at(0.75, 0.0, 0.0, 0.5);
        let c = box_at(2.0, 0.0, 0.0, 0.5);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn touching_boxes_count_as_overlapping() {
        let a = Aabb::new(Vec3A::ZERO, Vec3A::ONE);
        let b = Aabb::new(Vec3A::ONE, Vec3A::splat(2.0));
        assert!(a.intersects(&b));
    }

    #[test]
    fn ray_hits_box_front_face() {
        let b = box_at(5.0, 0.0, 0.0, 0.5);
        let hit = ray_aabb(Vec3A::ZERO, Vec3A::X, &b).expect("hit");
        assert!((hit.t - 4.5).abs() < 1e-5, "t was {}", hit.t);
        assert!((hit.normal - Vec3A::new(-1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn ray_misses_box_behind() {
        let b = box_at(-5.0, 0.0, 0.0, 0.5);
        assert!(ray_aabb(Vec3A::ZERO, Vec3A::X, &b).is_none());
    }

    #[test]
    fn ray_from_inside_box_hits_at_zero() {
        let b = box_at(0.0, 0.0, 0.0, 1.0);
        let hit = ray_aabb(Vec3A::ZERO, Vec3A::Y, &b).expect("hit");
        assert_eq!(hit.t, 0.0);
    }

    #[test]
    fn axis_aligned_zero_component_does_not_nan() {
        // Origin exactly on the min-x plane with a zero x direction must still
        // resolve, not produce 0 * inf.
        let b = box_at(0.0, 0.0, 0.0, 1.0);
        let origin = Vec3A::new(-1.0, 0.0, 0.0);
        let dir = Vec3A::new(1.0, 0.0, 0.0);
        let hit = ray_aabb(origin, dir, &b).expect("hit");
        assert!((hit.t - 0.0).abs() < 1e-6);
    }

    #[test]
    fn falling_box_lands_on_floor() {
        let start = box_at(0.0, 5.0, 0.0, 0.5);
        let floor = Aabb::new(Vec3A::new(-10.0, -1.0, -10.0), Vec3A::new(10.0, 0.0, 10.0));
        let r = resolve_movement(start, Vec3A::new(0.0, -10.0, 0.0), &[floor]);
        assert!(r.on_ground);
        assert!(r.hit_y);
        // Bottom of the box rests on the floor top (y = 0), backed off by the
        // sub-millimetre epsilon that keeps it from re-colliding next tick.
        assert!(
            r.aabb.min.y >= 0.0 && r.aabb.min.y < 1e-3,
            "min.y = {}",
            r.aabb.min.y
        );
    }

    #[test]
    fn walking_into_a_wall_stops() {
        let start = box_at(0.0, 0.0, 0.0, 0.5);
        let wall = Aabb::new(Vec3A::new(2.0, -5.0, -5.0), Vec3A::new(3.0, 5.0, 5.0));
        let r = resolve_movement(start, Vec3A::new(10.0, 0.0, 0.0), &[wall]);
        assert!(r.hit_x);
        // Right face stops against the wall's left face (x = 2), minus epsilon.
        assert!(
            r.aabb.max.x <= 2.0 && r.aabb.max.x > 2.0 - 1e-3,
            "max.x = {}",
            r.aabb.max.x
        );
    }

    #[test]
    fn fast_mover_cannot_tunnel_through_a_thin_wall() {
        // A 20-unit step in one tick with a 0.2-unit-thick wall in the path: the
        // destination is clear, so a destination-overlap check would tunnel. The
        // swept test must stop at the wall.
        let start = box_at(0.0, 0.0, 0.0, 0.5);
        let thin = Aabb::new(Vec3A::new(5.0, -5.0, -5.0), Vec3A::new(5.2, 5.0, 5.0));
        let r = resolve_movement(start, Vec3A::new(20.0, 0.0, 0.0), &[thin]);
        assert!(r.hit_x, "the sweep must detect the wall");
        assert!(r.aabb.max.x <= 5.0 + 1e-3, "tunnelled to {}", r.aabb.max.x);
    }

    #[test]
    fn unobstructed_move_is_exact() {
        let start = box_at(0.0, 0.0, 0.0, 0.5);
        let r = resolve_movement(start, Vec3A::new(1.0, 2.0, 3.0), &[]);
        assert!(!r.hit_x && !r.hit_y && !r.hit_z);
        assert!((r.aabb.min.x - 0.5).abs() < 1e-6);
        assert!((r.aabb.min.y - 1.5).abs() < 1e-6);
        assert!((r.aabb.min.z - 2.5).abs() < 1e-6);
    }

    #[test]
    fn diagonal_into_corner_stops_on_both_axes() {
        let start = box_at(0.0, 0.0, 0.0, 0.5);
        let wall_x = Aabb::new(Vec3A::new(1.5, -5.0, -5.0), Vec3A::new(2.5, 5.0, 5.0));
        let wall_z = Aabb::new(Vec3A::new(-5.0, -5.0, 1.5), Vec3A::new(5.0, 5.0, 2.5));
        let r = resolve_movement(start, Vec3A::new(5.0, 0.0, 5.0), &[wall_x, wall_z]);
        assert!(r.hit_x && r.hit_z);
    }

    #[test]
    fn frustum_keeps_front_box_and_culls_behind() {
        let proj = Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let f = Frustum::from_view_projection(proj * view);

        let in_front = box_at(0.0, 0.0, -10.0, 0.5);
        let behind = box_at(0.0, 0.0, 10.0, 0.5);
        assert!(f.contains_aabb(&in_front));
        assert!(!f.contains_aabb(&behind));
    }

    #[test]
    fn lerp_and_integrate() {
        let p = lerp_position(Vec3A::ZERO, Vec3A::new(10.0, 0.0, 0.0), 0.25);
        assert!((p.x - 2.5).abs() < 1e-6);
        let q = integrate_position(Vec3A::ZERO, Vec3A::new(0.0, 2.0, 0.0), 0.5);
        assert!((q.y - 1.0).abs() < 1e-6);
    }
}
