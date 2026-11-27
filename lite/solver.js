// top-op lite SIMP solver. app.js runs this function's source in a Web Worker built from a Blob URL.
// Conventions follow CLAUDE.md: C-order element/node ids, DOF = 3*node + axis, Hex8 node order.
function topopWorkerMain() {
  'use strict';
  const HEX = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const EMIN = 1e-9;
  const ELIVE = 1e-7; // elements softer than this are skipped in the matvec

  // BEGIN GENERATED KERNELS: KE(h = 1, nu = 0.3) unrolled; selfTest() checks them against hexKE().
  const k0 = 0.23504273504273501, k1 = 0.080128205128205107, k2 = 0.10683760683760685, k3 = 0.016025641025641031, k4 = 0.085470085470085458, k5 = 0.0080128205128205086, k6 = 0.053418803418803389, k7 = 0.040064102564102547, k8 = 0.058760683760683732, k9 = 0.0053418803418803498;
  function mvKernel(p, out, en, Eh, live, nl) {
    for (let q = 0; q < nl; q++) {
      const k = live[q];
      const b = 8 * k;
      const E = Eh[k];
      const a0 = 3 * en[b + 0];
      const u0 = p[a0], u1 = p[a0 + 1], u2 = p[a0 + 2];
      const a1 = 3 * en[b + 1];
      const u3 = p[a1], u4 = p[a1 + 1], u5 = p[a1 + 2];
      const a2 = 3 * en[b + 2];
      const u6 = p[a2], u7 = p[a2 + 1], u8 = p[a2 + 2];
      const a3 = 3 * en[b + 3];
      const u9 = p[a3], u10 = p[a3 + 1], u11 = p[a3 + 2];
      const a4 = 3 * en[b + 4];
      const u12 = p[a4], u13 = p[a4 + 1], u14 = p[a4 + 2];
      const a5 = 3 * en[b + 5];
      const u15 = p[a5], u16 = p[a5 + 1], u17 = p[a5 + 2];
      const a6 = 3 * en[b + 6];
      const u18 = p[a6], u19 = p[a6 + 1], u20 = p[a6 + 2];
      const a7 = 3 * en[b + 7];
      const u21 = p[a7], u22 = p[a7 + 1], u23 = p[a7 + 2];
      out[a0] += E * (k0*u0+k1*u1+k1*u2-k2*u3+k3*u4+k3*u5-k4*u6-k1*u7+k5*u8+k6*u9-k3*u10+k7*u11+k6*u12+k7*u13-k3*u14-k4*u15+k5*u16-k1*u17-k8*u18-k7*u19-k7*u20-k9*u21-k5*u22-k5*u23);
      out[a0 + 1] += E * (k1*u0+k0*u1+k1*u2-k3*u3+k6*u4+k7*u5-k1*u6-k4*u7+k5*u8+k3*u9-k2*u10+k3*u11+k7*u12+k6*u13-k3*u14-k5*u15-k9*u16-k5*u17-k7*u18-k8*u19-k7*u20+k5*u21-k4*u22-k1*u23);
      out[a0 + 2] += E * (k1*u0+k1*u1+k0*u2-k3*u3+k7*u4+k6*u5-k5*u6-k5*u7-k9*u8+k7*u9-k3*u10+k6*u11+k3*u12+k3*u13-k2*u14-k1*u15+k5*u16-k4*u17-k7*u18-k7*u19-k8*u20+k5*u21-k1*u22-k4*u23);
      out[a1] += E * (-k2*u0-k3*u1-k3*u2+k0*u3-k1*u4-k1*u5+k6*u6+k3*u7-k7*u8-k4*u9+k1*u10-k5*u11-k4*u12-k5*u13+k1*u14+k6*u15-k7*u16+k3*u17-k9*u18+k5*u19+k5*u20-k8*u21+k7*u22+k7*u23);
      out[a1 + 1] += E * (k3*u0+k6*u1+k7*u2-k1*u3+k0*u4+k1*u5-k3*u6-k2*u7+k3*u8+k1*u9-k4*u10+k5*u11+k5*u12-k9*u13-k5*u14-k7*u15+k6*u16-k3*u17-k5*u18-k4*u19-k1*u20+k7*u21-k8*u22-k7*u23);
      out[a1 + 2] += E * (k3*u0+k7*u1+k6*u2-k1*u3+k1*u4+k0*u5-k7*u6-k3*u7+k6*u8+k5*u9-k5*u10-k9*u11+k1*u12+k5*u13-k4*u14-k3*u15+k3*u16-k2*u17-k5*u18-k1*u19-k4*u20+k7*u21-k7*u22-k8*u23);
      out[a2] += E * (-k4*u0-k1*u1-k5*u2+k6*u3-k3*u4-k7*u5+k0*u6+k1*u7-k1*u8-k2*u9+k3*u10-k3*u11-k8*u12-k7*u13+k7*u14-k9*u15-k5*u16+k5*u17+k6*u18+k7*u19+k3*u20-k4*u21+k5*u22+k1*u23);
      out[a2 + 1] += E * (-k1*u0-k4*u1-k5*u2+k3*u3-k2*u4-k3*u5+k1*u6+k0*u7-k1*u8-k3*u9+k6*u10-k7*u11-k7*u12-k8*u13+k7*u14+k5*u15-k4*u16+k1*u17+k7*u18+k6*u19+k3*u20-k5*u21-k9*u22+k5*u23);
      out[a2 + 2] += E * (k5*u0+k5*u1-k9*u2-k7*u3+k3*u4+k6*u5-k1*u6-k1*u7+k0*u8+k3*u9-k7*u10+k6*u11+k7*u12+k7*u13-k8*u14-k5*u15+k1*u16-k4*u17-k3*u18-k3*u19-k2*u20+k1*u21-k5*u22-k4*u23);
      out[a3] += E * (k6*u0+k3*u1+k7*u2-k4*u3+k1*u4+k5*u5-k2*u6-k3*u7+k3*u8+k0*u9-k1*u10+k1*u11-k9*u12+k5*u13-k5*u14-k8*u15+k7*u16-k7*u17-k4*u18-k5*u19-k1*u20+k6*u21-k7*u22-k3*u23);
      out[a3 + 1] += E * (-k3*u0-k2*u1-k3*u2+k1*u3-k4*u4-k5*u5+k3*u6+k6*u7-k7*u8-k1*u9+k0*u10-k1*u11-k5*u12-k4*u13+k1*u14+k7*u15-k8*u16+k7*u17+k5*u18-k9*u19+k5*u20-k7*u21+k6*u22+k3*u23);
      out[a3 + 2] += E * (k7*u0+k3*u1+k6*u2-k5*u3+k5*u4-k9*u5-k3*u6-k7*u7+k6*u8+k1*u9-k1*u10+k0*u11+k5*u12+k1*u13-k4*u14-k7*u15+k7*u16-k8*u17-k1*u18-k5*u19-k4*u20+k3*u21-k3*u22-k2*u23);
      out[a4] += E * (k6*u0+k7*u1+k3*u2-k4*u3+k5*u4+k1*u5-k8*u6-k7*u7+k7*u8-k9*u9-k5*u10+k5*u11+k0*u12+k1*u13-k1*u14-k2*u15+k3*u16-k3*u17-k4*u18-k1*u19-k5*u20+k6*u21-k3*u22-k7*u23);
      out[a4 + 1] += E * (k7*u0+k6*u1+k3*u2-k5*u3-k9*u4+k5*u5-k7*u6-k8*u7+k7*u8+k5*u9-k4*u10+k1*u11+k1*u12+k0*u13-k1*u14-k3*u15+k6*u16-k7*u17-k1*u18-k4*u19-k5*u20+k3*u21-k2*u22-k3*u23);
      out[a4 + 2] += E * (-k3*u0-k3*u1-k2*u2+k1*u3-k5*u4-k4*u5+k7*u6+k7*u7-k8*u8-k5*u9+k1*u10-k4*u11-k1*u12-k1*u13+k0*u14+k3*u15-k7*u16+k6*u17+k5*u18+k5*u19-k9*u20-k7*u21+k3*u22+k6*u23);
      out[a5] += E * (-k4*u0-k5*u1-k1*u2+k6*u3-k7*u4-k3*u5-k9*u6+k5*u7-k5*u8-k8*u9+k7*u10-k7*u11-k2*u12-k3*u13+k3*u14+k0*u15-k1*u16+k1*u17+k6*u18+k3*u19+k7*u20-k4*u21+k1*u22+k5*u23);
      out[a5 + 1] += E * (k5*u0-k9*u1+k5*u2-k7*u3+k6*u4+k3*u5-k5*u6-k4*u7+k1*u8+k7*u9-k8*u10+k7*u11+k3*u12+k6*u13-k7*u14-k1*u15+k0*u16-k1*u17-k3*u18-k2*u19-k3*u20+k1*u21-k4*u22-k5*u23);
      out[a5 + 2] += E * (-k1*u0-k5*u1-k4*u2+k3*u3-k3*u4-k2*u5+k5*u6+k1*u7-k4*u8-k7*u9+k7*u10-k8*u11-k3*u12-k7*u13+k6*u14+k1*u15-k1*u16+k0*u17+k7*u18+k3*u19+k6*u20-k5*u21+k5*u22-k9*u23);
      out[a6] += E * (-k8*u0-k7*u1-k7*u2-k9*u3-k5*u4-k5*u5+k6*u6+k7*u7-k3*u8-k4*u9+k5*u10-k1*u11-k4*u12-k1*u13+k5*u14+k6*u15-k3*u16+k7*u17+k0*u18+k1*u19+k1*u20-k2*u21+k3*u22+k3*u23);
      out[a6 + 1] += E * (-k7*u0-k8*u1-k7*u2+k5*u3-k4*u4-k1*u5+k7*u6+k6*u7-k3*u8-k5*u9-k9*u10-k5*u11-k1*u12-k4*u13+k5*u14+k3*u15-k2*u16+k3*u17+k1*u18+k0*u19+k1*u20-k3*u21+k6*u22+k7*u23);
      out[a6 + 2] += E * (-k7*u0-k7*u1-k8*u2+k5*u3-k1*u4-k4*u5+k3*u6+k3*u7-k2*u8-k1*u9+k5*u10-k4*u11-k5*u12-k5*u13-k9*u14+k7*u15-k3*u16+k6*u17+k1*u18+k1*u19+k0*u20-k3*u21+k7*u22+k6*u23);
      out[a7] += E * (-k9*u0+k5*u1+k5*u2-k8*u3+k7*u4+k7*u5-k4*u6-k5*u7+k1*u8+k6*u9-k7*u10+k3*u11+k6*u12+k3*u13-k7*u14-k4*u15+k1*u16-k5*u17-k2*u18-k3*u19-k3*u20+k0*u21-k1*u22-k1*u23);
      out[a7 + 1] += E * (-k5*u0-k4*u1-k1*u2+k7*u3-k8*u4-k7*u5+k5*u6-k9*u7-k5*u8-k7*u9+k6*u10-k3*u11-k3*u12-k2*u13+k3*u14+k1*u15-k4*u16+k5*u17+k3*u18+k6*u19+k7*u20-k1*u21+k0*u22+k1*u23);
      out[a7 + 2] += E * (-k5*u0-k1*u1-k4*u2+k7*u3-k7*u4-k8*u5+k1*u6+k5*u7-k4*u8-k3*u9+k3*u10-k2*u11-k7*u12-k3*u13+k6*u14+k5*u15-k5*u16-k9*u17+k3*u18+k7*u19+k6*u20-k1*u21+k1*u22+k0*u23);
    }
  }
  function ceKernel(p, en, ne, ce) {
    for (let k = 0; k < ne; k++) {
      const b = 8 * k;
      const a0 = 3 * en[b + 0];
      const u0 = p[a0], u1 = p[a0 + 1], u2 = p[a0 + 2];
      const a1 = 3 * en[b + 1];
      const u3 = p[a1], u4 = p[a1 + 1], u5 = p[a1 + 2];
      const a2 = 3 * en[b + 2];
      const u6 = p[a2], u7 = p[a2 + 1], u8 = p[a2 + 2];
      const a3 = 3 * en[b + 3];
      const u9 = p[a3], u10 = p[a3 + 1], u11 = p[a3 + 2];
      const a4 = 3 * en[b + 4];
      const u12 = p[a4], u13 = p[a4 + 1], u14 = p[a4 + 2];
      const a5 = 3 * en[b + 5];
      const u15 = p[a5], u16 = p[a5 + 1], u17 = p[a5 + 2];
      const a6 = 3 * en[b + 6];
      const u18 = p[a6], u19 = p[a6 + 1], u20 = p[a6 + 2];
      const a7 = 3 * en[b + 7];
      const u21 = p[a7], u22 = p[a7 + 1], u23 = p[a7 + 2];
      ce[k] =
        u0 * (k0*u0+k1*u1+k1*u2-k2*u3+k3*u4+k3*u5-k4*u6-k1*u7+k5*u8+k6*u9-k3*u10+k7*u11+k6*u12+k7*u13-k3*u14-k4*u15+k5*u16-k1*u17-k8*u18-k7*u19-k7*u20-k9*u21-k5*u22-k5*u23) +
        u1 * (k1*u0+k0*u1+k1*u2-k3*u3+k6*u4+k7*u5-k1*u6-k4*u7+k5*u8+k3*u9-k2*u10+k3*u11+k7*u12+k6*u13-k3*u14-k5*u15-k9*u16-k5*u17-k7*u18-k8*u19-k7*u20+k5*u21-k4*u22-k1*u23) +
        u2 * (k1*u0+k1*u1+k0*u2-k3*u3+k7*u4+k6*u5-k5*u6-k5*u7-k9*u8+k7*u9-k3*u10+k6*u11+k3*u12+k3*u13-k2*u14-k1*u15+k5*u16-k4*u17-k7*u18-k7*u19-k8*u20+k5*u21-k1*u22-k4*u23) +
        u3 * (-k2*u0-k3*u1-k3*u2+k0*u3-k1*u4-k1*u5+k6*u6+k3*u7-k7*u8-k4*u9+k1*u10-k5*u11-k4*u12-k5*u13+k1*u14+k6*u15-k7*u16+k3*u17-k9*u18+k5*u19+k5*u20-k8*u21+k7*u22+k7*u23) +
        u4 * (k3*u0+k6*u1+k7*u2-k1*u3+k0*u4+k1*u5-k3*u6-k2*u7+k3*u8+k1*u9-k4*u10+k5*u11+k5*u12-k9*u13-k5*u14-k7*u15+k6*u16-k3*u17-k5*u18-k4*u19-k1*u20+k7*u21-k8*u22-k7*u23) +
        u5 * (k3*u0+k7*u1+k6*u2-k1*u3+k1*u4+k0*u5-k7*u6-k3*u7+k6*u8+k5*u9-k5*u10-k9*u11+k1*u12+k5*u13-k4*u14-k3*u15+k3*u16-k2*u17-k5*u18-k1*u19-k4*u20+k7*u21-k7*u22-k8*u23) +
        u6 * (-k4*u0-k1*u1-k5*u2+k6*u3-k3*u4-k7*u5+k0*u6+k1*u7-k1*u8-k2*u9+k3*u10-k3*u11-k8*u12-k7*u13+k7*u14-k9*u15-k5*u16+k5*u17+k6*u18+k7*u19+k3*u20-k4*u21+k5*u22+k1*u23) +
        u7 * (-k1*u0-k4*u1-k5*u2+k3*u3-k2*u4-k3*u5+k1*u6+k0*u7-k1*u8-k3*u9+k6*u10-k7*u11-k7*u12-k8*u13+k7*u14+k5*u15-k4*u16+k1*u17+k7*u18+k6*u19+k3*u20-k5*u21-k9*u22+k5*u23) +
        u8 * (k5*u0+k5*u1-k9*u2-k7*u3+k3*u4+k6*u5-k1*u6-k1*u7+k0*u8+k3*u9-k7*u10+k6*u11+k7*u12+k7*u13-k8*u14-k5*u15+k1*u16-k4*u17-k3*u18-k3*u19-k2*u20+k1*u21-k5*u22-k4*u23) +
        u9 * (k6*u0+k3*u1+k7*u2-k4*u3+k1*u4+k5*u5-k2*u6-k3*u7+k3*u8+k0*u9-k1*u10+k1*u11-k9*u12+k5*u13-k5*u14-k8*u15+k7*u16-k7*u17-k4*u18-k5*u19-k1*u20+k6*u21-k7*u22-k3*u23) +
        u10 * (-k3*u0-k2*u1-k3*u2+k1*u3-k4*u4-k5*u5+k3*u6+k6*u7-k7*u8-k1*u9+k0*u10-k1*u11-k5*u12-k4*u13+k1*u14+k7*u15-k8*u16+k7*u17+k5*u18-k9*u19+k5*u20-k7*u21+k6*u22+k3*u23) +
        u11 * (k7*u0+k3*u1+k6*u2-k5*u3+k5*u4-k9*u5-k3*u6-k7*u7+k6*u8+k1*u9-k1*u10+k0*u11+k5*u12+k1*u13-k4*u14-k7*u15+k7*u16-k8*u17-k1*u18-k5*u19-k4*u20+k3*u21-k3*u22-k2*u23) +
        u12 * (k6*u0+k7*u1+k3*u2-k4*u3+k5*u4+k1*u5-k8*u6-k7*u7+k7*u8-k9*u9-k5*u10+k5*u11+k0*u12+k1*u13-k1*u14-k2*u15+k3*u16-k3*u17-k4*u18-k1*u19-k5*u20+k6*u21-k3*u22-k7*u23) +
        u13 * (k7*u0+k6*u1+k3*u2-k5*u3-k9*u4+k5*u5-k7*u6-k8*u7+k7*u8+k5*u9-k4*u10+k1*u11+k1*u12+k0*u13-k1*u14-k3*u15+k6*u16-k7*u17-k1*u18-k4*u19-k5*u20+k3*u21-k2*u22-k3*u23) +
        u14 * (-k3*u0-k3*u1-k2*u2+k1*u3-k5*u4-k4*u5+k7*u6+k7*u7-k8*u8-k5*u9+k1*u10-k4*u11-k1*u12-k1*u13+k0*u14+k3*u15-k7*u16+k6*u17+k5*u18+k5*u19-k9*u20-k7*u21+k3*u22+k6*u23) +
        u15 * (-k4*u0-k5*u1-k1*u2+k6*u3-k7*u4-k3*u5-k9*u6+k5*u7-k5*u8-k8*u9+k7*u10-k7*u11-k2*u12-k3*u13+k3*u14+k0*u15-k1*u16+k1*u17+k6*u18+k3*u19+k7*u20-k4*u21+k1*u22+k5*u23) +
        u16 * (k5*u0-k9*u1+k5*u2-k7*u3+k6*u4+k3*u5-k5*u6-k4*u7+k1*u8+k7*u9-k8*u10+k7*u11+k3*u12+k6*u13-k7*u14-k1*u15+k0*u16-k1*u17-k3*u18-k2*u19-k3*u20+k1*u21-k4*u22-k5*u23) +
        u17 * (-k1*u0-k5*u1-k4*u2+k3*u3-k3*u4-k2*u5+k5*u6+k1*u7-k4*u8-k7*u9+k7*u10-k8*u11-k3*u12-k7*u13+k6*u14+k1*u15-k1*u16+k0*u17+k7*u18+k3*u19+k6*u20-k5*u21+k5*u22-k9*u23) +
        u18 * (-k8*u0-k7*u1-k7*u2-k9*u3-k5*u4-k5*u5+k6*u6+k7*u7-k3*u8-k4*u9+k5*u10-k1*u11-k4*u12-k1*u13+k5*u14+k6*u15-k3*u16+k7*u17+k0*u18+k1*u19+k1*u20-k2*u21+k3*u22+k3*u23) +
        u19 * (-k7*u0-k8*u1-k7*u2+k5*u3-k4*u4-k1*u5+k7*u6+k6*u7-k3*u8-k5*u9-k9*u10-k5*u11-k1*u12-k4*u13+k5*u14+k3*u15-k2*u16+k3*u17+k1*u18+k0*u19+k1*u20-k3*u21+k6*u22+k7*u23) +
        u20 * (-k7*u0-k7*u1-k8*u2+k5*u3-k1*u4-k4*u5+k3*u6+k3*u7-k2*u8-k1*u9+k5*u10-k4*u11-k5*u12-k5*u13-k9*u14+k7*u15-k3*u16+k6*u17+k1*u18+k1*u19+k0*u20-k3*u21+k7*u22+k6*u23) +
        u21 * (-k9*u0+k5*u1+k5*u2-k8*u3+k7*u4+k7*u5-k4*u6-k5*u7+k1*u8+k6*u9-k7*u10+k3*u11+k6*u12+k3*u13-k7*u14-k4*u15+k1*u16-k5*u17-k2*u18-k3*u19-k3*u20+k0*u21-k1*u22-k1*u23) +
        u22 * (-k5*u0-k4*u1-k1*u2+k7*u3-k8*u4-k7*u5+k5*u6-k9*u7-k5*u8-k7*u9+k6*u10-k3*u11-k3*u12-k2*u13+k3*u14+k1*u15-k4*u16+k5*u17+k3*u18+k6*u19+k7*u20-k1*u21+k0*u22+k1*u23) +
        u23 * (-k5*u0-k1*u1-k4*u2+k7*u3-k7*u4-k8*u5+k1*u6+k5*u7-k4*u8-k3*u9+k3*u10-k2*u11-k7*u12-k3*u13+k6*u14+k5*u15-k5*u16-k9*u17+k3*u18+k7*u19+k6*u20-k1*u21+k1*u22+k0*u23);
    }
  }
  // END GENERATED KERNELS

  // 24x24 stiffness of a cubic Hex8 of edge h, E = 1, 2x2x2 Gauss.
  function hexKE(h, nu) {
    const c = 1 / ((1 + nu) * (1 - 2 * nu));
    const D = new Float64Array(36);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) D[i * 6 + j] = c * (i === j ? 1 - nu : nu);
    for (let i = 3; i < 6; i++) D[i * 6 + i] = (c * (1 - 2 * nu)) / 2;
    const KE = new Float64Array(576);
    const B = new Float64Array(144);
    const DB = new Float64Array(144);
    const g = 1 / Math.sqrt(3);
    const s = 2 / h;
    const detJ = (h / 2) ** 3;
    for (const xi of [-g, g]) for (const eta of [-g, g]) for (const zeta of [-g, g]) {
      B.fill(0);
      for (let a = 0; a < 8; a++) {
        const sx = 2 * HEX[a][0] - 1, sy = 2 * HEX[a][1] - 1, sz = 2 * HEX[a][2] - 1;
        const dx = (s * sx * (1 + sy * eta) * (1 + sz * zeta)) / 8;
        const dy = (s * sy * (1 + sx * xi) * (1 + sz * zeta)) / 8;
        const dz = (s * sz * (1 + sx * xi) * (1 + sy * eta)) / 8;
        const k = 3 * a;
        B[k] = dx; B[24 + k + 1] = dy; B[48 + k + 2] = dz;
        B[72 + k] = dy; B[72 + k + 1] = dx;
        B[96 + k + 1] = dz; B[96 + k + 2] = dy;
        B[120 + k] = dz; B[120 + k + 2] = dx;
      }
      for (let i = 0; i < 6; i++) for (let j = 0; j < 24; j++) {
        let t = 0;
        for (let k = 0; k < 6; k++) t += D[i * 6 + k] * B[k * 24 + j];
        DB[i * 24 + j] = t;
      }
      for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) {
        let t = 0;
        for (let k = 0; k < 6; k++) t += B[k * 24 + i] * DB[k * 24 + j];
        KE[i * 24 + j] += t * detJ;
      }
    }
    return KE;
  }

  // Cyclic Jacobi eigenvalues of a symmetric n x n matrix.
  function symEig(Ain, n) {
    const A = Float64Array.from(Ain);
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0, tot = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = A[i * n + j] * A[i * n + j];
        tot += v;
        if (i !== j) off += v;
      }
      if (off <= 1e-30 * tot) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = cs * akp - sn * akq;
          A[k * n + q] = sn * akp + cs * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = cs * apk - sn * aqk;
          A[q * n + k] = sn * apk + cs * aqk;
        }
      }
    }
    const ev = [];
    for (let i = 0; i < n; i++) ev.push(A[i * n + i]);
    return ev.sort((a, b) => a - b);
  }

  function selfTest() {
    const KE = hexKE(1, 0.3);
    let asym = 0, kmax = 0;
    for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) {
      asym = Math.max(asym, Math.abs(KE[i * 24 + j] - KE[j * 24 + i]));
      kmax = Math.max(kmax, Math.abs(KE[i * 24 + j]));
    }
    const ev = symEig(KE, 24);
    const lmax = ev[ev.length - 1];
    const zeroModes = ev.filter((v) => Math.abs(v) < 1e-9 * lmax).length;
    const minPositive = ev.find((v) => Math.abs(v) >= 1e-9 * lmax);
    // Unrolled kernels against the dense KE on a random element vector.
    const en = Int32Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
    const uv = new Float64Array(24).map(() => Math.random() - 0.5);
    const ref = new Float64Array(24);
    let eref = 0;
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 24; j++) ref[i] += KE[i * 24 + j] * uv[j];
      eref += uv[i] * ref[i];
    }
    const got = new Float64Array(24);
    mvKernel(uv, got, en, Float64Array.of(1), Int32Array.of(0), 1);
    const ce = new Float64Array(1);
    ceKernel(uv, en, 1, ce);
    let kerr = Math.abs(ce[0] - eref) / Math.abs(eref);
    for (let i = 0; i < 24; i++) kerr = Math.max(kerr, Math.abs(got[i] - ref[i]) / lmax);
    const ok = asym <= 1e-12 * kmax && zeroModes === 6 && minPositive > 0 && kerr < 1e-12;
    return {
      type: 'selftest',
      symmetric: asym <= 1e-12 * kmax,
      asym,
      zeroModes,
      negative: ev.filter((v) => v < -1e-9 * lmax).length,
      lmin: minPositive,
      lmax,
      kernelError: kerr,
      ok,
    };
  }

  let stopFlag = false;
  let job = null;

  self.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'selftest') self.postMessage(selfTest());
    else if (m.type === 'stop') stopFlag = true;
    else if (m.type === 'run') {
      try {
        stopFlag = false;
        job = setup(m);
        setTimeout(step, 0);
      } catch (err) {
        self.postMessage({ type: 'error', message: String((err && err.message) || err) });
      }
    }
  };

  function setup(m) {
    const { nx, ny, nz, h } = m;
    const NE = nx * ny * nz, NZ1 = nz + 1, NYZ1 = (ny + 1) * NZ1;
    const active = m.active, passiveFull = m.passive;
    const eMap = new Int32Array(NE).fill(-1);
    let ne = 0;
    for (let e = 0; e < NE; e++) if (active[e]) eMap[e] = ne++;
    const elems = new Int32Array(ne);
    const eijk = new Int32Array(ne * 3);
    for (let e = 0, k = 0; e < NE; e++) {
      if (!active[e]) continue;
      elems[k] = e;
      const iz = e % nz, t = (e - iz) / nz, iy = t % ny, ix = (t - iy) / ny;
      eijk[3 * k] = ix; eijk[3 * k + 1] = iy; eijk[3 * k + 2] = iz;
      k++;
    }
    const nodeMap = new Int32Array((nx + 1) * NYZ1).fill(-1);
    const off = HEX.map(([a, b, c]) => a * NYZ1 + b * NZ1 + c);
    const en = new Int32Array(ne * 8);
    let nn = 0;
    for (let k = 0; k < ne; k++) {
      const n0 = eijk[3 * k] * NYZ1 + eijk[3 * k + 1] * NZ1 + eijk[3 * k + 2];
      for (let a = 0; a < 8; a++) {
        const g = n0 + off[a];
        if (nodeMap[g] < 0) nodeMap[g] = nn++;
        en[k * 8 + a] = nodeMap[g];
      }
    }
    const ndof = 3 * nn;
    const fixed = new Uint8Array(ndof);
    for (const n of m.fixedNodes) {
      const c = nodeMap[n];
      if (c >= 0) fixed[3 * c] = fixed[3 * c + 1] = fixed[3 * c + 2] = 1;
    }
    const fixedList = [];
    for (let d = 0; d < ndof; d++) if (fixed[d]) fixedList.push(d);
    const f = new Float64Array(ndof);
    for (let i = 0; i < m.loads.length; i += 4) {
      const c = nodeMap[m.loads[i]];
      if (c < 0) continue;
      for (let d = 0; d < 3; d++) if (!fixed[3 * c + d]) f[3 * c + d] += m.loads[i + 1 + d];
    }

    // Density filter: weights max(0, rmin - dist) over active neighbours (dist in voxels).
    const rmin = m.rmin;
    const R = Math.ceil(rmin);
    const offs = [];
    for (let a = -R; a <= R; a++) for (let b = -R; b <= R; b++) for (let c = -R; c <= R; c++) {
      const d = Math.hypot(a, b, c);
      if (d < rmin) offs.push([a, b, c, rmin - d]);
    }
    const fStart = new Int32Array(ne + 1);
    const fIdx = [];
    const fW = [];
    for (let k = 0; k < ne; k++) {
      const ix = eijk[3 * k], iy = eijk[3 * k + 1], iz = eijk[3 * k + 2];
      for (const [a, b, c, w] of offs) {
        const jx = ix + a, jy = iy + b, jz = iz + c;
        if (jx < 0 || jy < 0 || jz < 0 || jx >= nx || jy >= ny || jz >= nz) continue;
        const j = eMap[(jx * ny + jy) * nz + jz];
        if (j < 0) continue;
        fIdx.push(j);
        fW.push(w);
      }
      fStart[k + 1] = fIdx.length;
    }
    const Hs = new Float64Array(ne);
    for (let k = 0; k < ne; k++) for (let q = fStart[k]; q < fStart[k + 1]; q++) Hs[k] += fW[q];

    const pas = new Int8Array(ne);
    let nFree = 0;
    for (let k = 0; k < ne; k++) {
      pas[k] = passiveFull ? passiveFull[elems[k]] : 0;
      if (!pas[k]) nFree++;
    }
    const volfrac = m.volfrac;
    const x = new Float64Array(ne);
    for (let k = 0; k < ne; k++) x[k] = pas[k] === 1 ? 1 : pas[k] === -1 ? 0 : volfrac;

    const J = {
      nx, ny, nz, h, ne, nn, ndof, elems, en, fixed, fixedList: Int32Array.from(fixedList), f,
      fStart, fIdx: Int32Array.from(fIdx), fW: Float64Array.from(fW), Hs, pas, nFree, volfrac,
      penal: m.penal || 3, move: m.move || 0.2, maxIter: m.maxIter, minChange: m.minChange ?? 0.01,
      KE: hexKE(h, 0.3), x, Eh: new Float64Array(ne), live: new Int32Array(ne), nl: 0,
      mask: new Uint8Array(ndof), maskList: new Int32Array(0), xPhys: new Float64Array(ne), xnew: new Float64Array(ne),
      xTmp: new Float64Array(ne), Ee: new Float64Array(ne), ce: new Float64Array(ne),
      dc: new Float64Array(ne), dv: new Float64Array(ne), dcf: new Float64Array(ne),
      dvf: new Float64Array(ne), u: new Float64Array(ndof), r: new Float64Array(ndof),
      z: new Float64Array(ndof), p: new Float64Array(ndof), q: new Float64Array(ndof),
      invD: new Float64Array(ndof), it: 0, change: 1, fnorm: 0,
    };
    let fn = 0;
    for (let d = 0; d < ndof; d++) fn += f[d] * f[d];
    J.fnorm = Math.sqrt(fn);
    if (!(J.fnorm > 0)) throw new Error('No load reaches the active grid: add a load on the part.');
    if (!fixedList.length) throw new Error('No support reaches the active grid: add a support.');
    filter(J, x, J.xPhys);
    applyPassive(J, J.xPhys);
    self.postMessage({ type: 'setup', ne, nn, ndof, nFree, fixedDofs: fixedList.length });
    return J;
  }

  function filter(J, src, dst) {
    const { ne, fStart, fIdx, fW, Hs } = J;
    for (let k = 0; k < ne; k++) {
      let s = 0;
      for (let q = fStart[k]; q < fStart[k + 1]; q++) s += fW[q] * src[fIdx[q]];
      dst[k] = s / Hs[k];
    }
  }

  // Transpose filter (H is symmetric): dst_j = sum_i w_ij src_i / Hs_i.
  function filterT(J, src, dst) {
    const { ne, fStart, fIdx, fW, Hs } = J;
    for (let k = 0; k < ne; k++) {
      let s = 0;
      for (let q = fStart[k]; q < fStart[k + 1]; q++) {
        const i = fIdx[q];
        s += (fW[q] * src[i]) / Hs[i];
      }
      dst[k] = s;
    }
  }

  function applyPassive(J, xp) {
    const pas = J.pas;
    for (let k = 0; k < J.ne; k++) if (pas[k]) xp[k] = pas[k] === 1 ? 1 : 0;
  }

  // K p over live elements; masked DOFs (supports, and nodes touching only skipped elements) -> 0.
  function matvec(J, pin, out) {
    out.fill(0);
    mvKernel(pin, out, J.en, J.Eh, J.live, J.nl);
    const ml = J.maskList;
    for (let i = 0; i < ml.length; i++) out[ml[i]] = 0;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function prepareOperator(J) {
    const { ne, ndof, en, KE, Ee, Eh, live, h, fixed, mask, invD } = J;
    let nl = 0;
    for (let k = 0; k < ne; k++) {
      Eh[k] = Ee[k] * h;
      if (Ee[k] >= ELIVE) live[nl++] = k;
    }
    J.nl = nl;
    invD.fill(0);
    for (let q = 0; q < nl; q++) {
      const k = live[q];
      const E = Ee[k];
      for (let a = 0; a < 8; a++) {
        const n3 = 3 * en[8 * k + a];
        for (let d = 0; d < 3; d++) invD[n3 + d] += E * KE[(3 * a + d) * 25];
      }
    }
    const ml = [];
    for (let d = 0; d < ndof; d++) {
      mask[d] = fixed[d] || invD[d] === 0 ? 1 : 0;
      if (mask[d]) {
        ml.push(d);
        invD[d] = 0;
      } else invD[d] = 1 / invD[d];
    }
    J.maskList = Int32Array.from(ml);
  }

  // Jacobi-preconditioned CG, warm-started from J.u.
  function pcg(J, tol, maxIt) {
    const { ndof, mask, f, u, r, z, p, q, invD } = J;
    for (let d = 0; d < ndof; d++) if (mask[d]) u[d] = 0;
    matvec(J, u, q);
    for (let d = 0; d < ndof; d++) r[d] = mask[d] ? 0 : f[d] - q[d];
    for (let d = 0; d < ndof; d++) z[d] = r[d] * invD[d];
    p.set(z);
    let rz = dot(r, z);
    let res = Math.sqrt(dot(r, r)) / J.fnorm;
    let it = 0;
    while (res > tol && it < maxIt) {
      matvec(J, p, q);
      const alpha = rz / dot(p, q);
      let rr = 0;
      for (let d = 0; d < ndof; d++) {
        u[d] += alpha * p[d];
        const rd = r[d] - alpha * q[d];
        r[d] = rd;
        rr += rd * rd;
      }
      it++;
      res = Math.sqrt(rr) / J.fnorm;
      if (res <= tol) break;
      let rzNew = 0;
      for (let d = 0; d < ndof; d++) {
        const zd = r[d] * invD[d];
        z[d] = zd;
        rzNew += r[d] * zd;
      }
      const beta = rzNew / rz;
      rz = rzNew;
      for (let d = 0; d < ndof; d++) p[d] = z[d] + beta * p[d];
    }
    return { it, res };
  }

  function cgTol(J) {
    if (J.it < 5 || J.change > 0.1) return 1e-3;
    if (J.change > 0.03) return 1e-4;
    return 1e-5;
  }

  function iterate(J) {
    const t0 = performance.now();
    const { ne, xPhys, Ee, ce, dc, dv, dcf, dvf, x, xnew, xTmp, u, pas, penal, move, h } = J;
    J.it++;
    for (let k = 0; k < ne; k++) Ee[k] = EMIN + xPhys[k] ** penal * (1 - EMIN);
    prepareOperator(J);
    const tol = cgTol(J);
    const cg = pcg(J, tol, 2000);
    ceKernel(u, J.en, ne, ce);
    for (let k = 0; k < ne; k++) ce[k] *= h;
    const c = dot(J.f, u);
    for (let k = 0; k < ne; k++) {
      if (pas[k]) {
        dc[k] = 0;
        dv[k] = 0;
      } else {
        dc[k] = -penal * xPhys[k] ** (penal - 1) * (1 - EMIN) * ce[k];
        dv[k] = 1;
      }
    }
    filterT(J, dc, dcf);
    filterT(J, dv, dvf);
    // OC update, bisection on the Lagrange multiplier against the filtered volume.
    let lmax = 1e-30;
    for (let k = 0; k < ne; k++) if (!pas[k] && dvf[k] > 0) lmax = Math.max(lmax, -dcf[k] / dvf[k]);
    let l1 = 0, l2 = 1e3 * lmax;
    const target = J.volfrac * J.nFree;
    let vol = 0;
    for (let b = 0; b < 200 && (l2 - l1) / (l1 + l2) > 1e-6; b++) {
      const lmid = 0.5 * (l1 + l2);
      for (let k = 0; k < ne; k++) {
        if (pas[k]) { xnew[k] = x[k]; continue; }
        const Be = Math.max(1e-30, -dcf[k] / Math.max(1e-30, dvf[k]) / lmid);
        let xn = x[k] * Math.sqrt(Be);
        xn = Math.min(xn, x[k] + move, 1);
        xn = Math.max(xn, x[k] - move, 0);
        xnew[k] = xn;
      }
      filter(J, xnew, xTmp);
      vol = 0;
      for (let k = 0; k < ne; k++) if (!pas[k]) vol += xTmp[k];
      if (vol > target) l1 = lmid;
      else l2 = lmid;
    }
    let change = 0;
    for (let k = 0; k < ne; k++) change = Math.max(change, Math.abs(xnew[k] - x[k]));
    x.set(xnew);
    filter(J, x, xPhys);
    applyPassive(J, xPhys);
    vol = 0;
    for (let k = 0; k < ne; k++) if (!pas[k]) vol += xPhys[k];
    J.change = change;
    return {
      c, vol: vol / J.nFree, change, cgIters: cg.it, cgRes: cg.res, tol, live: J.nl,
      ms: performance.now() - t0,
    };
  }

  function densityFrame(J) {
    const out = new Uint8Array(J.nx * J.ny * J.nz);
    for (let k = 0; k < J.ne; k++) out[J.elems[k]] = Math.round(J.xPhys[k] * 255);
    return out;
  }

  function step() {
    const J = job;
    if (!J) return;
    try {
      if (stopFlag) return finish(J, 'stopped');
      const info = iterate(J);
      const rho = densityFrame(J);
      self.postMessage({ type: 'iter', it: J.it, ...info, rho }, [rho.buffer]);
      if (J.it >= J.maxIter) return finish(J, 'max-iterations');
      if (J.it >= 10 && info.change < J.minChange) return finish(J, 'converged');
      setTimeout(step, 0);
    } catch (err) {
      self.postMessage({ type: 'error', message: String((err && err.message) || err) });
    }
  }

  function finish(J, reason) {
    const rho = new Float32Array(J.nx * J.ny * J.nz);
    for (let k = 0; k < J.ne; k++) rho[J.elems[k]] = J.xPhys[k];
    self.postMessage({ type: 'done', reason, it: J.it, rho }, [rho.buffer]);
    job = null;
  }
}
