"""MagWave rung-1 reference formulas -- verified implementations + self-checks.

Companion to docs/research/validation-sources.md. Every closed form here was
numerically verified against an independent discretised Biot-Savart / charge
integration before publication; the self-checks below pin those verification
results so the file cannot silently go stale.

Run:  python3 reference_formulas.py          (needs numpy only)
Exit code 0 = all checks pass.

Elliptic-integral convention: K(m), E(m) use the PARAMETER m = k^2
(the scipy.special.ellipk/ellipe convention, Abramowitz & Stegun).
"""
import numpy as np

MU0 = 4e-7 * np.pi


# ----------------------------------------------------------------------
# Complete elliptic integrals K(m), E(m) (parameter-m convention)
# ----------------------------------------------------------------------
def ellipk_m(m):
    """K(m) = int_0^{pi/2} (1 - m sin^2 t)^{-1/2} dt,  m = k^2."""
    t = np.linspace(0, np.pi / 2, 30001)
    s = np.sin(t)
    y = 1.0 / np.sqrt(1.0 - m * s * s)
    h = t[1] - t[0]
    return h / 3 * (y[0] + y[-1] + 4 * y[1::2].sum() + 2 * y[2:-1:2].sum())


def ellipe_m(m):
    """E(m) = int_0^{pi/2} (1 - m sin^2 t)^{1/2} dt,  m = k^2."""
    t = np.linspace(0, np.pi / 2, 30001)
    s = np.sin(t)
    y = np.sqrt(1.0 - m * s * s)
    h = t[1] - t[0]
    return h / 3 * (y[0] + y[-1] + 4 * y[1::2].sum() + 2 * y[2:-1:2].sum())


# ----------------------------------------------------------------------
# (a) magnetic dipole
# ----------------------------------------------------------------------
def dipole_B(m_vec, p):
    """B at point p from a dipole m_vec at the origin [tesla].
    m_vec [A m^2], p [m]; returns array of 3 components (SI)."""
    r = np.asarray(p, float)
    rabs = np.sqrt(r @ r)
    m = np.asarray(m_vec, float)
    if rabs < 1e-300:
        raise ValueError("dipole field undefined at r=0")
    return MU0 / (4 * np.pi) * (3 * np.dot(m, r) * r / rabs**5 - m / rabs**3)


# ----------------------------------------------------------------------
# (b) circular current loop (radius a, z=0 plane, current I), any point
# ----------------------------------------------------------------------
def loop_B(a, rho, z, I=1.0):
    """Field of a planar circular loop centred at the origin.
    rho >= 0 is the cylindrical radius of the field point.  Exact off-axis
    form in complete elliptic integrals (parameter-m convention)."""
    Q2 = (a + rho) ** 2 + z * z
    p2 = (a - rho) ** 2 + z * z
    m = 4.0 * a * rho / Q2
    if rho < 1e-13:
        return 0.0, MU0 * I * a * a / (2.0 * (a * a + z * z) ** 1.5)
    K, E = ellipk_m(m), ellipe_m(m)
    c = MU0 * I / (2.0 * np.pi * np.sqrt(Q2))
    Brho = c * (z / rho) * (-K + (a * a + rho * rho + z * z) / p2 * E)
    Bz = c * (K + (a * a - rho * rho - z * z) / p2 * E)
    return Brho, Bz


def loop_B_numeric(a, p, I=1.0, nseg=60000):
    """Independent reference: Biot-Savart sum over straight chords."""
    t = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    dt2 = 2 * np.pi / nseg
    dxs, dys = -a * np.sin(t) * dt2, a * np.cos(t) * dt2
    rx, ry, rz = p[0] - a * np.cos(t), p[1] - a * np.sin(t), p[2]
    rr = np.sqrt(rx * rx + ry * ry + rz * rz)
    f = (MU0 * I / (4 * np.pi)) / rr**3
    return np.array([((dys * rz) * f).sum(),
                     ((-dxs * rz) * f).sum(),
                     ((dxs * ry - dys * rx) * f).sum()])


# ----------------------------------------------------------------------
# (c) finite solenoid, on-axis
# ----------------------------------------------------------------------
def thin_solenoid_Bz(R, z1, z2, nI, z):
    """On-axis field of a thin-walled finite solenoid, radius R, winding
    between z1 < z2, ampere-turn density nI [A] (n turns/length times I).
    Exact for a current sheet; elementary closed form."""
    c1 = (z - z1) / np.sqrt(R * R + (z - z1) ** 2)
    c2 = (z - z2) / np.sqrt(R * R + (z - z2) ** 2)
    return 0.5 * MU0 * nI * (c1 - c2)


def thick_solenoid_Bz(R1, R2, z1, z2, J, z):
    """On-axis field of a solenoid with rectangular winding cross-section
    R1<=a<=R2, z1<=z'<=z2, current density J [A/m^2].  Elementary closed
    form (verified against a ring-slice sum); elliptic integrals are only
    needed off axis.  NOTE: a volume winding is NOT a magnet model -- a
    uniformly magnetised cylinder is the thin-wall limit K = J*(R2-R1) -> M
    and uses cylinder_Bz_axis, not this function."""
    s1, s2 = z - z1, z - z2

    def bracket(s):
        # C(a,s) = s*asinh(a/|s|) is the antiderivative of s/sqrt(a^2+s^2) in a;
        # s*asinh(R/s) without |s| is wrong for s < 0.
        if s == 0:
            return 0.0
        return s * (np.arcsinh(R2 / abs(s)) - np.arcsinh(R1 / abs(s)))

    return 0.5 * MU0 * J * (bracket(s1) - bracket(s2))


# ----------------------------------------------------------------------
# (d) uniformly magnetised rectangular cuboid: 8-corner charge formula
# ----------------------------------------------------------------------
def cuboid_B(M, box, p):
    """B of a uniformly magnetised cuboid, magnetisation M [A/m] along +z.
    box = (x1, x2, y1, y2, z1, z2); magnetised from z1 (bottom, charge -M)
    to z2 (top, charge +M).  Valid outside AND inside the body (atan2 branch
    handles the surface jump); do not evaluate exactly on a face plane."""
    x1, x2, y1, y2, z1, z2 = box
    x, y, z = p

    def Gx(u, v, t):
        if t == 0.0:
            t = 1e-12          # tangential B is continuous; see docstring note
        r = np.sqrt(u * u + v * v + t * t)
        return np.log((r - v) / abs(t))
    def Gy(u, v, t):
        if t == 0.0:
            t = 1e-12
        r = np.sqrt(u * u + v * v + t * t)
        return np.log((r - u) / abs(t))
    def Gz(u, v, t):
        r = np.sqrt(u * u + v * v + t * t)
        return np.arctan2(u * v, t * r)

    corners = [(x1, y1, +1.0), (x2, y1, -1.0), (x1, y2, -1.0), (x2, y2, +1.0)]

    def patch(G, t):
        return sum(s * G(x - Xc, y - Yc, t) for (Xc, Yc, s) in corners)

    t2, t1 = z - z2, z - z1
    fac = MU0 * M / (4 * np.pi)
    return np.array([fac * (patch(Gx, t2) - patch(Gx, t1)),
                     fac * (patch(Gy, t2) - patch(Gy, t1)),
                     fac * (patch(Gz, t2) - patch(Gz, t1))])


def cuboid_B_numeric(M, box, p, nq=1200):
    """Independent reference: field of the two charged end faces."""
    x1, x2, y1, y2, z1, z2 = box
    gx = np.linspace(x1, x2, nq + 1); dx = gx[1] - gx[0]; gxc = (gx[:-1] + gx[1:]) / 2
    gy = np.linspace(y1, y2, nq + 1); dy = gy[1] - gy[0]; gyc = (gy[:-1] + gy[1:]) / 2
    X, Y = np.meshgrid(gxc, gyc, indexing="ij")
    B = np.zeros(3)
    for (zf, sig) in ((z2, M), (z1, -M)):
        rx, ry, rz = p[0] - X, p[1] - Y, p[2] - zf
        r3 = (rx * rx + ry * ry + rz * rz) ** 1.5
        B += (MU0 * sig / (4 * np.pi)) * np.array([(rx / r3).sum(),
                                                   (ry / r3).sum(),
                                                   (rz / r3).sum()]) * dx * dy
    return B


# ----------------------------------------------------------------------
# (e) uniformly magnetised cylinder (radius R, z in [z1, z2], M along +z)
# ----------------------------------------------------------------------
def cylinder_Bz_axis(R, z1, z2, M, z):
    """On-axis field; exact elementary form (two charged end discs)."""
    return 0.5 * MU0 * M * ((z - z1) / np.sqrt((z - z1) ** 2 + R * R)
                            - (z - z2) / np.sqrt((z - z2) ** 2 + R * R))


def cylinder_B_loopsuper(R, z1, z2, M, p, nt=2000):
    """Field at an arbitrary point via exact superposition of loop fields:
    a uniformly axially magnetised cylinder is magnetostatically identical to
    an ideal solenoid with n*I = M (surface current K_phi = M on the side)."""
    zps = np.linspace(z1, z2, nt + 1)[:-1] + (z2 - z1) / nt / 2
    dI = M * (z2 - z1) / nt
    Br = Bz = 0.0
    for zp in zps:
        br, bz = loop_B(R, p[0], p[2] - zp, I=dI)
        Br += br
        Bz += bz
    return np.array([Br, 0.0, Bz])


def cylinder_sheet_numeric(R, z1, z2, M, p, nseg=6000, nz=300):
    """Independent reference: Biot-Savart over the side-surface current."""
    zs = np.linspace(z1, z2, nz + 1)
    dz = (z2 - z1) / nz
    B = np.zeros(3)
    t = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    dt2 = 2 * np.pi / nseg
    dxs, dys = -R * np.sin(t) * dt2, R * np.cos(t) * dt2
    for zc in (zs[:-1] + zs[1:]) / 2:
        rz = p[2] - zc
        rx, ry = p[0] - R * np.cos(t), p[1] - R * np.sin(t)
        rr = np.sqrt(rx * rx + ry * ry + rz * rz)
        f = (MU0 * M * dz / (4 * np.pi)) / rr**3
        B += np.array([((dys * rz) * f).sum(), 0.0,
                       ((dxs * ry - dys * rx) * f).sum()])
    return B


# ----------------------------------------------------------------------
# (f) Halbach k=1 cylinder: bore field (infinite length, 2D exact)
# ----------------------------------------------------------------------
def halbach_bore_ideal(Br, Ri, Ro):
    """Bore flux density of an infinite k=1 Halbach cylinder:
    B = Br * ln(Ro/Ri), uniform, directed along the fixed transverse axis.
    Magnetisation pattern (verified): M(phi) = (Br/mu0)*(cos(phi) rhat
    + sin(phi) phihat), which in Cartesian is (Br/mu0)(cos 2phi x + sin 2phi y)."""
    return Br * np.log(Ro / Ri)


def halbach_bore_2d_numeric(Br, Ri, Ro, pts, nr=220, nphi=540):
    """2D verification of the bore field from the bound currents of the
    pattern above: volume Jz = 2*M0*sin(phi)/r, side sheets Kz = -/+M0*sin(phi),
    M0 = Br/mu0.  Returns the in-plane field at each (x, y) sample point."""
    M0 = Br / MU0
    rs = Ri + (Ro - Ri) * (np.arange(nr) + 0.5) / nr
    dr = (Ro - Ri) / nr
    ts = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    dphi = 2 * np.pi / nphi
    Rg, Tg = np.meshgrid(rs, ts, indexing="ij")
    xg, yg = Rg * np.cos(Tg), Rg * np.sin(Tg)
    st = np.sin(Tg)
    Jz = 2.0 * M0 * st / Rg
    Jq = Jz * Rg * dr * dphi                       # charge-per-length weighting
    out = []
    for (px, py) in pts:
        dx, dy = px - xg, py - yg
        d2 = dx * dx + dy * dy
        Bx = (MU0 / (2 * np.pi)) * (-Jq * dy / d2).sum()
        By = (MU0 / (2 * np.pi)) * (Jq * dx / d2).sum()
        for (Rr, ksign) in ((Ri, -1.0), (Ro, +1.0)):
            Kz = ksign * M0 * np.sin(ts)
            xo, yo = Rr * np.cos(ts), Rr * np.sin(ts)
            dx, dy = px - xo, py - yo
            w = Rr * dphi
            d2 = dx * dx + dy * dy
            Bx += (MU0 / (2 * np.pi)) * (-Kz * w * dy / d2).sum()
            By += (MU0 / (2 * np.pi)) * (Kz * w * dx / d2).sum()
        out.append(np.array([Bx, By]))
    return out


# ----------------------------------------------------------------------
# (g) force between two coaxial magnets
# ----------------------------------------------------------------------
def F_dipole_coaxial(Br1, Br2, V1, V2, d):
    """Dipole-dipole force magnitude between two coaxially aligned magnets
    (dipoles along the axis, centres separated by d).  F = 3 mu0 m1 m2/(2 pi d^4)."""
    m1 = (Br1 / MU0) * V1
    m2 = (Br2 / MU0) * V2
    return 3.0 * MU0 * m1 * m2 / (2.0 * np.pi * d**4)


def F_pointpole_cylinders(Br, R, L, g):
    """Point-pole (charged-face-centre) approximation for two identical
    coaxial cylindrical magnets, each radius R, length L, gap g between the
    near faces.  Attraction magnitude.  Exact in the far limit only."""
    c = np.pi * Br**2 * R**4 / (4 * MU0)
    return c * (1.0 / g**2 - 2.0 / (g + L) ** 2 + 1.0 / (g + 2 * L) ** 2)


def F_facepair_numeric(g, Br=1.2, R=0.005, L=0.02, nr=120, nt=540):
    """Exact force on magnet 2 for the coaxial pair above, from the four
    face-pair interactions of the magnetic-charge (pole) model:
    F = sum over face pairs (mu0 s1 s2 / 4 pi) * IntInt dz / r^3 dA1 dA2.
    Returns Fz; NEGATIVE means attraction of magnet 2 toward magnet 1."""
    from numpy.polynomial.legendre import leggauss
    M = Br / MU0
    xg, wg = leggauss(nr)
    a1 = R * (xg + 1) / 2; wa1 = wg * (R / 2) * a1          # r dr weights
    a2 = a1; wa2 = wa1
    phis = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    A1 = a1[:, None, None]; A2 = a2[None, :, None]; PHI = phis[None, None, :]
    W = wa1[:, None, None] * wa2[None, :, None] * (2 * np.pi / nt)

    def kernel(dz):
        RR2 = dz * dz + A1 * A1 + A2 * A2 - 2 * A1 * A2 * np.cos(PHI)
        return (dz / RR2**1.5 * W).sum() * 2 * np.pi

    faces1 = [(0.0, -M), (L, +M)]
    faces2 = [(L + g, -M), (L + g + L, +M)]
    F = 0.0
    for (zf, s1) in faces1:
        for (z2, s2) in faces2:
            F += (MU0 / (4 * np.pi)) * s1 * s2 * kernel(z2 - zf)
    return F


# ======================================================================
# self-checks (each pins a verification result documented in
# validation-sources.md; all tolerances come from that session's runs)
# ======================================================================
failures = []

def check(name, value, expected, tol):
    ok = abs(value - expected) <= tol * max(abs(expected), 1e-30)
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    print(f"  [{status}] {name}: got {value:.10g}, expected {expected:.10g} (tol {tol:.0e})")


print("== elliptic integrals ==")
check("K(0.5)", ellipk_m(0.5), 1.8540746773013719, 1e-13)
check("E(0.5)", ellipe_m(0.5), 1.3506438810476755, 1e-13)
check("K(0)", ellipk_m(0.0), np.pi / 2, 1e-14)

print("== (b) loop vs Biot-Savart (10 points incl. near-wire) ==")
maxe = 0.0
for (rho, z) in [(0.3, 0.4), (0.8, 0.1), (1.2, 0.0), (1.5, 0.0), (0.6, 1.1),
                 (0.05, 2.0), (2.5, 1.0), (0.9, 0.0), (0.2, 0.02), (1.0, 0.005)]:
    Bn = loop_B_numeric(1.0, (rho, 0.0, z))
    Brk, Bzk = loop_B(1.0, rho, z)
    maxe = max(maxe, max(abs(Brk - Bn[0]), abs(Bzk - Bn[2])) / max(abs(Bn).max(), 1e-300))
print(f"  max relative error over the 10 points: {maxe:.2e} (must be < 1e-11)")
if maxe > 1e-11:
    failures.append("loop")

print("== (c) solenoid on-axis ==")
check("thin solenoid centre field", thin_solenoid_Bz(1.0, -2.0, 2.0, 100.0, 0.0),
      MU0 * 100.0 * (2.0 / np.sqrt(5.0)), 1e-12)


def _thick_loopsum(R1, R2, z1, z2, J, z, na=60, nz=300):
    """Independent reference: build the thick winding from thin current rings."""
    az = np.linspace(R1, R2, na + 1)
    da = az[1] - az[0]
    tot = 0.0
    for a in (az[:-1] + az[1:]) / 2:
        tot += thin_solenoid_Bz(a, z1, z2, J * da, z)   # J*da = ampere-turn density of the slice
    return tot

check("thick solenoid vs ring-slice sum", thick_solenoid_Bz(0.5, 1.5, -1.0, 1.0, 1e6, 0.0),
      _thick_loopsum(0.5, 1.5, -1.0, 1.0, 1e6, 0.0), 1e-4)

print("== (d) cuboid 8-corner vs charged-face quadrature ==")
maxe = 0.0
box = (-0.5, 0.5, -0.3, 0.3, 1.0, 1.8)
for p in [(0.0, 0.0, 3.0), (0.6, 0.2, 2.5), (1.0, 0.4, 1.8), (-0.7, -0.3, 2.9),
          (0.35, 0.35, 3.2), (2.0, 0.0, 1.4), (0.0, 1.0, 1.4)]:
    Bc = cuboid_B(8e5, box, p)
    Bn = cuboid_B_numeric(8e5, box, p, nq=800)
    maxe = max(maxe, np.abs(Bc - Bn).max() / np.abs(Bn).max())
print(f"  max relative error over 7 external points: {maxe:.2e} (must be < 5e-6)")
if maxe > 5e-6:
    failures.append("cuboid")
# inside point: the corner formula returns the true B directly (atan2 branch)
Bc = cuboid_B(8e5, box, (0.1, 0.05, 1.4))
Bn = cuboid_B_numeric(8e5, box, (0.1, 0.05, 1.4), nq=800) + np.array([0, 0, MU0 * 8e5])
check("cuboid interior Bz", Bc[2], Bn[2], 1e-5)

print("== (e) cylinder ==")
check("cylinder on-axis (z=0)", cylinder_Bz_axis(1.0, -1.0, 1.0, 8e5, 0.0),
      MU0 * 8e5 * np.sqrt(2) / 2, 1e-12)
Bsup = cylinder_B_loopsuper(1.0, -1.0, 1.0, 8e5, (0.5, 0.0, 0.0))
Bref = cylinder_sheet_numeric(1.0, -1.0, 1.0, 8e5, (0.5, 0.0, 0.0))
check("cylinder off-axis superposition vs sheet", abs(Bsup[2]), abs(Bref[2]), 2e-6)

print("== (f) Halbach k=1 bore field ==")
Br, Ri, Ro = 1.0, 1.0, np.e
ideal = halbach_bore_ideal(Br, Ri, Ro)
pts2d = [(0, 0), (Ri / 2, 0), (Ri / 4, Ri / 4), (0, Ri / 2), (-Ri / 2, 0)]
b2d = halbach_bore_2d_numeric(Br, Ri, Ro, pts2d)
for pp, b in zip(pts2d, b2d):
    mag = float(np.abs(b).max())
    check(f"bore field magnitude at {pp}/(Br) equals ln(Ro/Ri)", mag / Br, 1.0, 3e-3)
mags = np.array([np.abs(b).max() / Br for b in b2d])
print(f"  bore-field spread across the 5 samples: {mags.max() - mags.min():.2e} (uniform)")

print("== (g) two coaxial magnets (Br=1.2 T, R=5 mm, L=20 mm; L=4R) ==")
BrF, RF, LF = 1.2, 0.005, 0.02
V = np.pi * RF * RF * LF
check("exact face-pair force, far limit g=20L", abs(F_facepair_numeric(20 * LF, Br=BrF, R=RF, L=LF)),
      F_dipole_coaxial(BrF, BrF, V, V, 21 * LF), 0.01)
check("exact force g=5L vs point-pole", abs(F_facepair_numeric(5 * LF, Br=BrF, R=RF, L=LF)),
      F_pointpole_cylinders(BrF, RF, LF, 5 * LF), 0.02)

print()
if failures:
    print("FAILURES:", failures)
    raise SystemExit(1)
print("ALL CHECKS PASS")
