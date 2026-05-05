import * as THREE from 'three';

/**
 * Visual sizing aimed at later hand-tracking "catch" + photo expansion:
 * slightly smaller sprites than overlap-heavy layouts — pair with seeded grid gaps.
 */
const PARTICLE_HAND_TARGET = {
  pointScale: 950,
  minDiameterPx: 36,
  maxDiameterPx: 380,
};

/** Radial branches per orb — up to ~10 photo thumbnails per node later. */
const ORB_BRANCH_COUNT = 10;

const VERTEX = /* glsl */ `
  attribute vec3 color;
  attribute float phase;
  attribute float visibility;
  uniform float time;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;

  void main() {
    vVis = visibility;
    vColor = color;
    vTwinkle = 0.72 + 0.28 * sin(time * 0.55 + phase * 6.2831853);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depthScale = uPointScale / max(-mvPosition.z, 1.0);
    gl_PointSize = clamp(depthScale * uPixelRatio, uPointPxMin, uPointPxMax);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;

  void main() {
    if (vVis < 0.001) discard;
    vec2 c = gl_PointCoord - vec2(0.5);
    float len = length(c);
    if (len > 0.5) discard;
    float core = 1.0 - smoothstep(0.0, 0.38, len);
    float glow = exp(-len * 8.5) * 0.22;
    float alpha = (core * 0.88 + glow * 0.38) * vTwinkle * 0.78 * vVis;
    gl_FragColor = vec4(vColor * (core + glow * 0.42), alpha);
  }
`;

const ORB_CENTER_VERT = /* glsl */ `
  attribute vec3 color;
  attribute float phase;
  attribute float visibility;
  uniform float time;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;

  void main() {
    vVis = visibility;
    vColor = color;
    vTwinkle = 0.72 + 0.28 * sin(time * 0.55 + phase * 6.2831853);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depthScale = uPointScale / max(-mvPosition.z, 1.0);
    gl_PointSize = clamp(depthScale * uPixelRatio, uPointPxMin, uPointPxMax);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const ORB_TIP_VERT = /* glsl */ `
  attribute vec3 color;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depthScale = uPointScale / max(-mvPosition.z, 1.0);
    gl_PointSize = clamp(depthScale * uPixelRatio, uPointPxMin, uPointPxMax);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const ORB_TIP_FRAG = /* glsl */ `
  varying vec3 vColor;

  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float r = length(c);
    if (r > 0.49 || r < 0.32) discard;
    gl_FragColor = vec4(vColor, 0.94);
  }
`;

function prefersReducedMotion() {
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function modSafe(a, n) {
  return ((a % n) + n) % n;
}

/** Soft accel/decel — reads less 'pop' than cubic for organic growth. */
function easeInOutSine(t) {
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return 0.5 - 0.5 * Math.cos(Math.PI * x);
}

function hexToRgb01(hex) {
  const r = ((hex >> 16) & 255) / 255;
  const g = ((hex >> 8) & 255) / 255;
  const b = (hex & 255) / 255;
  return [r, g, b];
}

/** Linear in [0,1]: people #CEF9F2, places soft gray‑white, projects #AFC1D6 */
const CATEGORY_BASE_RGB = [
  hexToRgb01(0xcef9f2),
  hexToRgb01(0xe6eaef),
  hexToRgb01(0xafc1d6),
];

/** Per particle: 0 = people, 1 = places, 2 = projects (for future photo-node clusters). */
function assignParticleCategories(count) {
  const categories = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    categories[i] = modSafe(i, 3);
  }
  return categories;
}

function colorsFromCategories(categories) {
  const colors = new Float32Array(categories.length * 3);
  const v = 0.045;
  for (let i = 0; i < categories.length; i++) {
    const base = CATEGORY_BASE_RGB[categories[i]];
    const j = i * 3;
    const jitter = () => v * (Math.random() - 0.5);
    colors[j] = THREE.MathUtils.clamp(base[0] + jitter(), 0.2, 1);
    colors[j + 1] = THREE.MathUtils.clamp(base[1] + jitter(), 0.2, 1);
    colors[j + 2] = THREE.MathUtils.clamp(base[2] + jitter(), 0.2, 1);
  }
  return colors;
}

function verticalFallExtents(cam, zWorld) {
  const dist = -zWorld;
  const halfV = Math.tan(THREE.MathUtils.degToRad(cam.fov * 0.5));
  const halfH = halfV * dist;
  const margin = 0.72;
  const top = halfH * margin;
  const bottom = -halfH * margin;
  return { top, bottom, span: top - bottom, halfH };
}

/**
 * Rows/cols biased by aspect so we don't pile many particles in one screen column:
 * portrait → more rows than cols, landscape → the opposite (still aspect-aware caps).
 */
function seedGridExtents(count, aspect) {
  if (count <= 0) return { cols: 1, rows: 1 };
  const ratio = THREE.MathUtils.clamp(aspect, 0.62, 2.42);
  const squareish = Math.sqrt(count);
  const cols = THREE.MathUtils.clamp(
    Math.ceil(squareish * ratio * 1.12),
    3,
    Math.max(3, count),
  );
  const rows = Math.ceil(count / cols);
  return { cols: Math.min(cols, count), rows };
}

/**
 * Lattice + brick staggering for separation in screen space; depth step tracks lateral cell
 * width so stacked rows don't land on nearly the same pixel column.
 */
function seedParticlesHorizontal(base, cam, count) {
  const z0 = -34;
  const dist = -z0;
  const halfV = Math.tan(THREE.MathUtils.degToRad(cam.fov * 0.5));
  const halfH = halfV * dist;
  const halfW = halfH * cam.aspect;

  const { cols, rows } = seedGridExtents(count, cam.aspect);
  /** Use nearly full lateral frustum → larger centres-to-centres gap. */
  const lateralFrac = 0.95;
  const gridHalfW = halfW * lateralFrac;

  /** Slightly narrower cells when rows are staggered so nothing clips past ±gridHalfW. */
  const colDen = cols + (rows > 1 ? 0.54 : 0);
  const cellW = Math.max((2 * gridHalfW) / colDen, 0.055);

  /**
   * Separate row planes roughly as wide as lateral cells (~isotropic pairwise spacing in XZ).
   */
  const zStep = THREE.MathUtils.clamp(cellW * (rows > 2 ? 1.08 : 1.14), 1.82, cellW * 1.74);
  const depthBand = rows > 1 ? zStep * (rows - 1) : zStep;

  const jitterFrac = 0.03;
  const jxMax = Math.min(cellW * jitterFrac, 0.06);
  const jzMax = Math.min(rows > 1 ? zStep * jitterFrac : depthBand * jitterFrac * 0.55, 0.09);

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);

    const brick = (row & 1) * (cellW * 0.5);
    const cx = -gridHalfW + (col + 0.5) * cellW + brick;
    const cz = rows > 1 ? z0 + (row / (rows - 1) - 0.5) * depthBand : z0;

    const i3 = i * 3;
    base[i3] = cx + (Math.random() - 0.5) * 2 * jxMax;
    base[i3 + 2] = cz + (Math.random() - 0.5) * 2 * jzMax;
  }
}

function syncCameraAspect(cam) {
  const w = Math.max(1, window.innerWidth || 1);
  const h = Math.max(1, window.innerHeight || 1);
  cam.aspect = w / h;
  cam.updateProjectionMatrix();
}

function fillFibonacciDirections(outDirs, n, jitterStrength, scratch, randFn) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    scratch.set(Math.cos(theta) * r, y, Math.sin(theta) * r);
    scratch.x += (randFn() - 0.5) * jitterStrength;
    scratch.y += (randFn() - 0.5) * jitterStrength;
    scratch.z += (randFn() - 0.5) * jitterStrength;
    scratch.normalize();
    const i3 = i * 3;
    outDirs[i3] = scratch.x;
    outDirs[i3 + 1] = scratch.y;
    outDirs[i3 + 2] = scratch.z;
  }
}

const reducedMotion = prefersReducedMotion();

const canvas = document.getElementById('canvas');
if (!canvas) {
  throw new Error('Missing #canvas element.');
}

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 220);
camera.position.z = 0;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0a0818, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

syncCameraAspect(camera);

const COUNT = 20;
const positions = new Float32Array(COUNT * 3);
const basePositions = new Float32Array(COUNT * 3);
const phases = new Float32Array(COUNT);

seedParticlesHorizontal(basePositions, camera, COUNT);

for (let i = 0; i < COUNT; i++) {
  const i3 = i * 3;
  positions[i3] = basePositions[i3];
  positions[i3 + 1] = 0;
  positions[i3 + 2] = basePositions[i3 + 2];
  phases[i] = Math.random();
}

const particleCategories = assignParticleCategories(COUNT);
const particleColors = colorsFromCategories(particleCategories);

const visibilityAttr = new Float32Array(COUNT);
visibilityAttr.fill(1);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
geometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));
geometry.setAttribute('visibility', new THREE.BufferAttribute(visibilityAttr, 1));
geometry.computeBoundingSphere();

const material = new THREE.ShaderMaterial({
  uniforms: {
    time: { value: 0 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uPointScale: { value: PARTICLE_HAND_TARGET.pointScale },
    uPointPxMin: { value: PARTICLE_HAND_TARGET.minDiameterPx },
    uPointPxMax: { value: PARTICLE_HAND_TARGET.maxDiameterPx },
  },
  vertexShader: VERTEX,
  fragmentShader: FRAGMENT,
  transparent: true,
  depthWrite: false,
  depthTest: false,
  blending: THREE.NormalBlending,
});

const points = new THREE.Points(geometry, material);
points.frustumCulled = false;
points.renderOrder = 999;
scene.add(points);

/* ---------- Tap → radial orb (lines + hollow tip nodes; ready for image quads later) ---------- */

const orbGroup = new THREE.Group();
orbGroup.visible = false;
orbGroup.renderOrder = 1002;
scene.add(orbGroup);

const orbDirs = new Float32Array(ORB_BRANCH_COUNT * 3);
const orbLenJit = new Float32Array(ORB_BRANCH_COUNT);
const orbBranchStagger = new Float32Array(ORB_BRANCH_COUNT);
const dirScratch = new THREE.Vector3();
const bendAxisScratch = new THREE.Vector3();
const worldUp = new THREE.Vector3(0, 1, 0);

const linePositions = new Float32Array(ORB_BRANCH_COUNT * 2 * 3);
const lineGeom = new THREE.BufferGeometry();
lineGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
const lineMat = new THREE.LineBasicMaterial({
  transparent: true,
  opacity: 0.76,
  depthWrite: false,
  depthTest: false,
});
const orbLines = new THREE.LineSegments(lineGeom, lineMat);
orbLines.frustumCulled = false;
orbGroup.add(orbLines);

const tipPositions = new Float32Array(ORB_BRANCH_COUNT * 3);
const tipColors = new Float32Array(ORB_BRANCH_COUNT * 3);
const tipGeom = new THREE.BufferGeometry();
tipGeom.setAttribute('position', new THREE.BufferAttribute(tipPositions, 3));
tipGeom.setAttribute('color', new THREE.BufferAttribute(tipColors, 3));

const tipMat = new THREE.ShaderMaterial({
  uniforms: {
    uPixelRatio: { value: renderer.getPixelRatio() },
    uPointScale: { value: PARTICLE_HAND_TARGET.pointScale * 0.38 },
    uPointPxMin: { value: 10 },
    uPointPxMax: { value: 96 },
  },
  vertexShader: ORB_TIP_VERT,
  fragmentShader: ORB_TIP_FRAG,
  transparent: true,
  depthWrite: false,
  depthTest: false,
  blending: THREE.NormalBlending,
});
const orbTips = new THREE.Points(tipGeom, tipMat);
orbTips.frustumCulled = false;
orbGroup.add(orbTips);

const centerPos = new Float32Array(3);
const centerGeom = new THREE.BufferGeometry();
centerGeom.setAttribute('position', new THREE.BufferAttribute(centerPos, 3));
const centerColorArr = new Float32Array(3);
centerGeom.setAttribute('color', new THREE.BufferAttribute(centerColorArr, 3));
centerGeom.setAttribute('phase', new THREE.BufferAttribute(new Float32Array([0.35]), 1));
centerGeom.setAttribute('visibility', new THREE.BufferAttribute(new Float32Array([1]), 1));

const orbCenterMat = new THREE.ShaderMaterial({
  uniforms: {
    time: { value: 0 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uPointScale: { value: PARTICLE_HAND_TARGET.pointScale },
    uPointPxMin: { value: PARTICLE_HAND_TARGET.minDiameterPx },
    uPointPxMax: { value: PARTICLE_HAND_TARGET.maxDiameterPx },
  },
  vertexShader: ORB_CENTER_VERT,
  fragmentShader: FRAGMENT,
  transparent: true,
  depthWrite: false,
  depthTest: false,
  blending: THREE.NormalBlending,
});
const orbCenter = new THREE.Points(centerGeom, orbCenterMat);
orbCenter.frustumCulled = false;
orbGroup.add(orbCenter);

const ORB_EXPAND_DURATION = reducedMotion ? 1.05 : 2.25;
const ORB_BASE_BRANCH_LEN = 5.65;

const orbState = {
  active: false,
  particleIndex: -1,
  animT: 0,
};

const projScratch = new THREE.Vector3();

function pickParticle(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const arr = geometry.attributes.position.array;
  const vis = geometry.attributes.visibility.array;
  const pxTol = PARTICLE_HAND_TARGET.maxDiameterPx * 0.58;
  let best = -1;
  let bestD = Infinity;

  for (let i = 0; i < COUNT; i++) {
    if (vis[i] < 0.05) continue;
    const i3 = i * 3;
    projScratch.set(arr[i3], arr[i3 + 1], arr[i3 + 2]);
    projScratch.project(camera);
    if (projScratch.z <= -1 || projScratch.z >= 1) continue;
    const sx = (projScratch.x * 0.5 + 0.5) * rect.width;
    const sy = (-projScratch.y * 0.5 + 0.5) * rect.height;
    const dx = clientX - rect.left - sx;
    const dy = clientY - rect.top - sy;
    const d = Math.hypot(dx, dy);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }

  if (best < 0 || bestD > pxTol) return -1;
  return best;
}

function restoreOrbParticle() {
  if (orbState.active && orbState.particleIndex >= 0) {
    visibilityAttr[orbState.particleIndex] = 1;
    geometry.attributes.visibility.needsUpdate = true;
  }
}

function closeOrb() {
  restoreOrbParticle();
  orbState.active = false;
  orbState.particleIndex = -1;
  orbState.animT = 0;
  orbGroup.visible = false;
}

function openOrb(particleIndex) {
  restoreOrbParticle();

  visibilityAttr[particleIndex] = 0;
  geometry.attributes.visibility.needsUpdate = true;

  fillFibonacciDirections(orbDirs, ORB_BRANCH_COUNT, 0.14, dirScratch, Math.random);
  for (let i = 0; i < ORB_BRANCH_COUNT; i++) {
    orbLenJit[i] = 0.88 + Math.random() * 0.2;
    const u = Math.random();
    orbBranchStagger[i] = u ** 1.45 * 0.72;
  }

  const ci = particleIndex * 3;
  const r = particleColors[ci];
  const g = particleColors[ci + 1];
  const b = particleColors[ci + 2];
  lineMat.color.setRGB(r, g, b);
  lineMat.needsUpdate = true;

  for (let i = 0; i < ORB_BRANCH_COUNT; i++) {
    const i3 = i * 3;
    tipColors[i3] = r;
    tipColors[i3 + 1] = g;
    tipColors[i3 + 2] = b;
  }
  tipGeom.attributes.color.needsUpdate = true;

  centerColorArr[0] = r;
  centerColorArr[1] = g;
  centerColorArr[2] = b;
  centerGeom.attributes.color.needsUpdate = true;

  orbState.active = true;
  orbState.particleIndex = particleIndex;
  orbState.animT = 0;
  orbGroup.visible = true;
}

canvas.addEventListener(
  'pointerdown',
  (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const picked = pickParticle(e.clientX, e.clientY);
    if (picked >= 0) {
      openOrb(picked);
    } else {
      closeOrb();
    }
  },
  { passive: true },
);

function updateOrb(arr, dt, tShader) {
  if (!orbState.active) return;

  orbState.animT = Math.min(1, orbState.animT + dt / ORB_EXPAND_DURATION);
  const gt = orbState.animT;

  const centerPulse = THREE.MathUtils.lerp(
    1.06,
    1.34,
    easeInOutSine(Math.min(1, gt * 0.88)),
  );

  const ci = orbState.particleIndex * 3;
  const cx = arr[ci];
  const cy = arr[ci + 1];
  const cz = arr[ci + 2];

  centerPos[0] = cx;
  centerPos[1] = cy;
  centerPos[2] = cz;
  centerGeom.attributes.position.needsUpdate = true;

  orbCenterMat.uniforms.time.value = tShader;
  orbCenterMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  orbCenterMat.uniforms.uPointScale.value = PARTICLE_HAND_TARGET.pointScale * centerPulse;

  tipMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();

  const lp = linePositions;
  const tp = tipPositions;

  for (let i = 0; i < ORB_BRANCH_COUNT; i++) {
    const i3 = i * 3;
    const i6 = i * 6;
    const dx = orbDirs[i3];
    const dy = orbDirs[i3 + 1];
    const dz = orbDirs[i3 + 2];

    const start = orbBranchStagger[i];
    const denom = Math.max(1e-5, 1 - start);
    const branchLinear = gt <= start ? 0 : (gt - start) / denom;
    const branchE = easeInOutSine(THREE.MathUtils.clamp(branchLinear, 0, 1));

    const L = ORB_BASE_BRANCH_LEN * orbLenJit[i] * branchE;

    bendAxisScratch.set(dx, dy, dz).cross(worldUp);
    if (bendAxisScratch.lengthSq() < 1e-8) {
      bendAxisScratch.set(1, 0, 0);
    }
    bendAxisScratch.normalize();
    const sag = easeInOutSine(branchE) * 0.19;

    const ox = dx * L + bendAxisScratch.x * sag;
    const oy = dy * L + bendAxisScratch.y * sag;
    const oz = dz * L + bendAxisScratch.z * sag;

    lp[i6] = cx;
    lp[i6 + 1] = cy;
    lp[i6 + 2] = cz;
    lp[i6 + 3] = cx + ox;
    lp[i6 + 4] = cy + oy;
    lp[i6 + 5] = cz + oz;

    tp[i3] = lp[i6 + 3];
    tp[i3 + 1] = lp[i6 + 4];
    tp[i3 + 2] = lp[i6 + 5];
  }

  lineGeom.attributes.position.needsUpdate = true;
  tipGeom.attributes.position.needsUpdate = true;
}

/* ---------- Main loop ---------- */

const clock = new THREE.Clock();

const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
const onMove = (e) => {
  mouse.tx = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.ty = -(e.clientY / window.innerHeight) * 2 + 1;
};
window.addEventListener('pointermove', onMove, { passive: true });

function resize() {
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  orbCenterMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  tipMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  seedParticlesHorizontal(basePositions, camera, COUNT);
}
window.addEventListener('resize', resize);
resize();

clock.getDelta();

const motionScale = reducedMotion ? 0.35 : 1;
const fallSpeed = reducedMotion ? 0.78 : 1.48;

function tick() {
  const dt = THREE.MathUtils.clamp(clock.getDelta(), 0, 0.05);
  const wallT = clock.getElapsedTime();
  const tShader = wallT * motionScale;
  material.uniforms.time.value = tShader;

  mouse.x += (mouse.tx - mouse.x) * 0.014 * motionScale;
  mouse.y += (mouse.ty - mouse.y) * 0.014 * motionScale;

  const posAttr = geometry.attributes.position;
  const arr = posAttr.array;

  const zPlane = -34;
  const { top, span } = verticalFallExtents(camera, zPlane);

  for (let i = 0; i < COUNT; i++) {
    const i3 = i * 3;
    const bx = basePositions[i3];
    const bz = basePositions[i3 + 2];
    const seed = phases[i] * 12.9898;

    const flowX = mouse.x * 0.38 + Math.sin(tShader * 0.055 + seed) * 0.14;

    const progress = modSafe(phases[i] * span + wallT * fallSpeed, span);
    const y = top - progress;

    arr[i3] =
      bx + Math.sin(tShader * 0.07 + seed * 1.7 + flowX * 0.048) * 0.17 + mouse.x * 0.62;
    arr[i3 + 1] = y + Math.cos(tShader * 0.06 + seed * 1.1) * 0.1 + mouse.y * 0.42;
    arr[i3 + 2] =
      bz +
      Math.sin(tShader * 0.045 + phases[i] * 8) * 0.13 +
      Math.cos(tShader * 0.03 + seed) * 0.06;
  }

  posAttr.needsUpdate = true;

  updateOrb(arr, dt, tShader);

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
