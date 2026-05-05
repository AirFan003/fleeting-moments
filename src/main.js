import * as THREE from 'three';

/** MediaPipe hand landmark indices (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker). */
const HAND_LM = {
  THUMB_TIP: 4,
  INDEX_TIP: 8,
};

/** Pinch span in normalized image space — tuned for arm's-length webcam. */
const PINCH_CLOSED = 0.034;
const PINCH_OPEN = 0.19;

/** Bloom when left pinch exceeds this (after mapping); avoids jitter-opens. */
const HAND_BLOOM_OPEN_THRESH = 0.07;

/**
 * Pinch below this (with animT mostly there) ends the hand session — loose "close enough"
 * without pinching all the way shut.
 */
const HAND_PINCH_END_SESSION = 0.14;

/**
 * Visual sizing aimed at later hand-tracking "catch" + photo expansion:
 * slightly smaller sprites than overlap-heavy layouts — pair with seeded grid gaps.
 */
const PARTICLE_HAND_TARGET = {
  pointScale: 1120,
  minDiameterPx: 52,
  maxDiameterPx: 480,
};

/** Radial branches per orb — up to ~10 photo thumbnails per node later. */
const ORB_BRANCH_COUNT = 10;

const VERTEX = /* glsl */ `
  attribute vec3 color;
  attribute float phase;
  attribute float visibility;
  attribute float highlight;
  uniform float time;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;
  varying float vHighlight;

  void main() {
    vVis = visibility;
    vColor = color;
    vHighlight = clamp(highlight, 0.0, 1.0);
    vTwinkle = 0.72 + 0.28 * sin(time * 0.55 + phase * 6.2831853);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depthScale = uPointScale / max(-mvPosition.z, 1.0);
    /** Full highlight = +50% point diameter vs idle (clearer hand selection). */
    float sizeMul = mix(1.0, 1.5, vHighlight);
    float pxMax = mix(uPointPxMax, uPointPxMax * 1.5, vHighlight);
    gl_PointSize = clamp(depthScale * uPixelRatio * sizeMul, uPointPxMin, pxMax);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;
  varying float vHighlight;

  void main() {
    if (vVis < 0.001) discard;
    vec2 c = gl_PointCoord - vec2(0.5);
    float len = length(c);
    if (len > 0.5) discard;
    float h = clamp(vHighlight, 0.0, 1.0);
    float core = 1.0 - smoothstep(0.0, 0.38, len);
    float glow = exp(-len * 8.5) * 0.22;
    vec3 rgb = vColor * (1.0 + 0.62 * h);
    float alpha = (core * 0.88 + glow * 0.38) * vTwinkle * 0.78 * vVis * (1.0 + 0.42 * h);
    gl_FragColor = vec4(rgb * (core + glow * 0.52 * (1.0 + 0.35 * h)), alpha);
  }
`;

const ORB_CENTER_VERT = /* glsl */ `
  attribute vec3 color;
  attribute float phase;
  attribute float visibility;
  attribute float highlight;
  uniform float time;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;
  varying float vTwinkle;
  varying float vVis;
  varying float vHighlight;

  void main() {
    vVis = visibility;
    vColor = color;
    vHighlight = clamp(highlight, 0.0, 1.0);
    vTwinkle = 0.72 + 0.28 * sin(time * 0.55 + phase * 6.2831853);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depthScale = uPointScale / max(-mvPosition.z, 1.0);
    /** Full highlight = +50% point diameter vs idle (clearer hand selection). */
    float sizeMul = mix(1.0, 1.5, vHighlight);
    float pxMax = mix(uPointPxMax, uPointPxMax * 1.5, vHighlight);
    gl_PointSize = clamp(depthScale * uPixelRatio * sizeMul, uPointPxMin, pxMax);
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

const webcam = document.getElementById('webcam');
if (!webcam) {
  throw new Error('Missing #webcam element.');
}

const webcamPreview = document.getElementById('webcam-preview');
if (!webcamPreview) {
  throw new Error('Missing #webcam-preview element.');
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

const highlightAttr = new Float32Array(COUNT);
highlightAttr.fill(0);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
geometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));
geometry.setAttribute('visibility', new THREE.BufferAttribute(visibilityAttr, 1));
geometry.setAttribute('highlight', new THREE.BufferAttribute(highlightAttr, 1));
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
centerGeom.setAttribute('highlight', new THREE.BufferAttribute(new Float32Array([0]), 1));

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
/** Subtle idle pulse while bloom is open (scale + line opacity). */
const ORB_BREATH_AMP = 0.036;
const ORB_BREATH_SPEED = 2.65;

const orbState = {
  active: false,
  closing: false,
  particleIndex: -1,
  animT: 0,
  /** True when the orb was opened via right-hand target + left pinch (not tap). */
  openedViaHand: false,
  /** Last mapped pinch openness when left hand was seen; used to hold bloom if left hand briefly leaves. */
  lastPinchBloomTarget: 0,
};

const projScratch = new THREE.Vector3();

/**
 * @param {boolean} [forFinger] — looser screen radius for index-finger aim (webcam lag / cover crop).
 */
function pickParticle(clientX, clientY, forFinger = false) {
  const rect = canvas.getBoundingClientRect();
  const arr = geometry.attributes.position.array;
  const vis = geometry.attributes.visibility.array;
  const pxTol = PARTICLE_HAND_TARGET.maxDiameterPx * (forFinger ? 0.82 : 0.58);
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

/**
 * Orb bloom hides the field particle; taps use world position of active index (same dot).
 */
function hitBloomAnchor(clientX, clientY, pxTolMultiplier = 1.25) {
  if (!orbState.active || orbState.particleIndex < 0) return false;
  const rect = canvas.getBoundingClientRect();
  const arr = geometry.attributes.position.array;
  const i = orbState.particleIndex;
  const i3 = i * 3;
  projScratch.set(arr[i3], arr[i3 + 1], arr[i3 + 2]);
  projScratch.project(camera);
  if (projScratch.z <= -1 || projScratch.z >= 1) return false;
  const sx = (projScratch.x * 0.5 + 0.5) * rect.width;
  const sy = (-projScratch.y * 0.5 + 0.5) * rect.height;
  const dx = clientX - rect.left - sx;
  const dy = clientY - rect.top - sy;
  const pxTol = PARTICLE_HAND_TARGET.maxDiameterPx * pxTolMultiplier * 1.06;
  return Math.hypot(dx, dy) <= pxTol;
}

function restoreOrbParticle() {
  if (orbState.active && orbState.particleIndex >= 0) {
    visibilityAttr[orbState.particleIndex] = 1;
    geometry.attributes.visibility.needsUpdate = true;
  }
}

function finishCloseOrb() {
  restoreOrbParticle();
  orbState.active = false;
  orbState.closing = false;
  orbState.particleIndex = -1;
  orbState.animT = 0;
  orbState.openedViaHand = false;
  orbState.lastPinchBloomTarget = 0;
  orbGroup.visible = false;
}

function closeOrb() {
  finishCloseOrb();
}

function openOrb(particleIndex, { viaHand = false, initialAnimT = 0 } = {}) {
  orbState.closing = false;
  restoreOrbParticle();
  orbState.openedViaHand = viaHand;

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
  orbState.animT = THREE.MathUtils.clamp(initialAnimT, 0, 1);
  orbGroup.visible = true;
}

function requestCollapseOrb() {
  if (!orbState.active) return;
  if (reducedMotion) {
    finishCloseOrb();
    return;
  }
  orbState.closing = true;
}

canvas.addEventListener(
  'pointerdown',
  (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;

    if (orbState.active && hitBloomAnchor(e.clientX, e.clientY)) {
      requestCollapseOrb();
      return;
    }

    const picked = pickParticle(e.clientX, e.clientY);
    if (picked >= 0) {
      if (orbState.active && orbState.openedViaHand && picked !== orbState.particleIndex) {
        return;
      }
      openOrb(picked);
    } else {
      if (orbState.active && orbState.openedViaHand) {
        return;
      }
      closeOrb();
    }
  },
  { passive: true },
);

function updateOrb(arr, dt, tShader, handBloomDrive) {
  if (!orbState.active) return;

  if (orbState.closing) {
    orbState.animT = Math.max(0, orbState.animT - dt / ORB_EXPAND_DURATION);
    if (orbState.animT <= 0) {
      orbState.animT = 0;
      finishCloseOrb();
      return;
    }
  } else if (handBloomDrive != null) {
    const target = THREE.MathUtils.clamp(handBloomDrive.target, 0, 1);
    orbState.animT = THREE.MathUtils.damp(orbState.animT, target, 14.5, dt);
    if (target < HAND_PINCH_END_SESSION && orbState.animT < HAND_PINCH_END_SESSION + 0.08) {
      orbState.animT = 0;
      finishCloseOrb();
      return;
    }
  } else {
    orbState.animT = Math.min(1, orbState.animT + dt / ORB_EXPAND_DURATION);
  }

  const gt = orbState.animT;

  const pulseEnv = easeInOutSine(THREE.MathUtils.clamp((gt - 0.08) / 0.9, 0, 1));
  const breath =
    1 +
    (reducedMotion ? 0 : ORB_BREATH_AMP * Math.sin(tShader * ORB_BREATH_SPEED)) * pulseEnv;

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
  orbCenterMat.uniforms.uPointScale.value = PARTICLE_HAND_TARGET.pointScale * centerPulse * breath;

  tipMat.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  tipMat.uniforms.uPointScale.value = PARTICLE_HAND_TARGET.pointScale * 0.38 * breath * 0.97;

  if (reducedMotion) {
    lineMat.opacity = 0.76;
  } else {
    lineMat.opacity = THREE.MathUtils.clamp(0.76 * (1 + (breath - 1) * 0.78), 0.63, 0.86);
  }

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

/* ---------- MediaPipe hands → selection + pinch bloom ---------- */

const handPipe = {
  landmarker: null,
  lastVideoTime: -1,
  leftPresent: false,
  rightPresent: false,
  pinchOpen01: 0,
  smoothedPinch: 0,
  /** Stabilized index under right index finger (-1 if none). */
  effectivePick: -1,
};

/** Require this many consecutive matching raw picks before swapping hover (reduces flicker). */
const FINGER_PICK_HOLD_FRAMES = 4;

const fingerPickHyst = {
  stable: -1,
  candidate: -1,
  streak: 0,
  miss: 0,
};

function updateStableFingerPick(rawIndex) {
  if (rawIndex < 0) {
    fingerPickHyst.miss++;
    if (fingerPickHyst.miss > 10) {
      fingerPickHyst.stable = -1;
      fingerPickHyst.candidate = -1;
      fingerPickHyst.streak = 0;
    }
    return fingerPickHyst.stable;
  }
  fingerPickHyst.miss = 0;
  if (rawIndex === fingerPickHyst.stable) {
    fingerPickHyst.candidate = -1;
    fingerPickHyst.streak = 0;
    return fingerPickHyst.stable;
  }
  if (rawIndex === fingerPickHyst.candidate) {
    fingerPickHyst.streak++;
  } else {
    fingerPickHyst.candidate = rawIndex;
    fingerPickHyst.streak = 1;
  }
  if (fingerPickHyst.streak >= FINGER_PICK_HOLD_FRAMES) {
    fingerPickHyst.stable = fingerPickHyst.candidate;
    fingerPickHyst.candidate = -1;
    fingerPickHyst.streak = 0;
  }
  return fingerPickHyst.stable;
}

/**
 * Map a normalized landmark to window client coords for a full-viewport mirrored video
 * using object-fit: cover (matches #webcam and #canvas stacking).
 */
function landmarkToCanvasClient(lm) {
  const rect = canvas.getBoundingClientRect();
  const vw = webcam.videoWidth || 1;
  const vh = webcam.videoHeight || 1;
  const cw = Math.max(1, rect.width);
  const ch = Math.max(1, rect.height);
  const scale = Math.max(cw / vw, ch / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const ox = (cw - dispW) * 0.5;
  const oy = (ch - dispH) * 0.5;
  const nx = 1.0 - lm.x;
  const ny = lm.y;
  return {
    clientX: rect.left + ox + nx * dispW,
    clientY: rect.top + oy + ny * dispH,
  };
}

function pinchSpan(lms) {
  const t = lms[HAND_LM.THUMB_TIP];
  const i = lms[HAND_LM.INDEX_TIP];
  return Math.hypot(t.x - i.x, t.y - i.y);
}

/** Map pinch span to ~0 (pinched) … ~1 (spread). */
function pinchToOpen01(span) {
  const u = THREE.MathUtils.clamp(
    (span - PINCH_CLOSED) / Math.max(1e-5, PINCH_OPEN - PINCH_CLOSED),
    0,
    1,
  );
  return u * u;
}

async function startWebcam(video, previewVideo) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  previewVideo.srcObject = stream;
  await Promise.all([video.play(), previewVideo.play()]);
}

async function initHandTracking() {
  if (!navigator.mediaDevices?.getUserMedia) return;

  try {
    await startWebcam(webcam, webcamPreview);
  } catch (e) {
    console.warn('[hands] Webcam unavailable:', e);
    return;
  }

  try {
    const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
    );
    const baseOptions = {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    };
    try {
      handPipe.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { ...baseOptions, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      });
    } catch {
      handPipe.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions,
        runningMode: 'VIDEO',
        numHands: 2,
      });
    }
  } catch (e) {
    console.warn('[hands] MediaPipe init failed:', e);
  }
}

void initHandTracking();

function runHandLandmarker(dt) {
  const lm = handPipe.landmarker;
  if (!lm || webcam.readyState < 2) {
    handPipe.leftPresent = false;
    handPipe.rightPresent = false;
    handPipe.pinchOpen01 = 0;
    handPipe.smoothedPinch = THREE.MathUtils.damp(handPipe.smoothedPinch, 0, 10, dt);
    return;
  }

  if (webcam.currentTime === handPipe.lastVideoTime) {
    return;
  }
  handPipe.lastVideoTime = webcam.currentTime;

  handPipe.leftPresent = false;
  handPipe.rightPresent = false;
  handPipe.pinchOpen01 = 0;

  let result;
  try {
    result = lm.detectForVideo(webcam, performance.now());
  } catch {
    return;
  }

  const marks = result?.landmarks;
  const handed = result?.handednesses;
  if (!marks?.length) {
    handPipe.smoothedPinch = THREE.MathUtils.damp(handPipe.smoothedPinch, 0, 12, dt);
    return;
  }

  let leftLms = null;
  let rightLms = null;
  for (let i = 0; i < marks.length; i++) {
    const label = handed?.[i]?.[0]?.categoryName ?? '';
    if (label === 'Left') leftLms = marks[i];
    else if (label === 'Right') rightLms = marks[i];
  }

  if (rightLms) {
    handPipe.rightPresent = true;
    const tip = rightLms[HAND_LM.INDEX_TIP];
    const { clientX, clientY } = landmarkToCanvasClient(tip);
    const rawPick = pickParticle(clientX, clientY, true);
    handPipe.effectivePick = updateStableFingerPick(rawPick);
  } else {
    handPipe.effectivePick = updateStableFingerPick(-1);
  }

  if (leftLms) {
    handPipe.leftPresent = true;
    const span = pinchSpan(leftLms);
    handPipe.pinchOpen01 = pinchToOpen01(span);
    handPipe.smoothedPinch = THREE.MathUtils.damp(
      handPipe.smoothedPinch,
      handPipe.pinchOpen01,
      20,
      dt,
    );
  } else {
    handPipe.smoothedPinch = THREE.MathUtils.damp(handPipe.smoothedPinch, 0, 14, dt);
  }
}

/** Left pinch→bloom; `openedViaHand` locks until you ease the pinch closed enough (see HAND_PINCH_END_SESSION). */
function buildHandBloomDrive() {
  const pinch = handPipe.smoothedPinch;

  if (!orbState.active) {
    if (!handPipe.leftPresent || pinch < HAND_BLOOM_OPEN_THRESH) return null;
    if (!handPipe.rightPresent || handPipe.effectivePick < 0) return null;
    openOrb(handPipe.effectivePick, { viaHand: true, initialAnimT: pinch });
    orbState.lastPinchBloomTarget = pinch;
    return { target: pinch };
  }

  if (orbState.openedViaHand) {
    if (handPipe.leftPresent) {
      orbState.lastPinchBloomTarget = pinch;
      return { target: pinch };
    }
    return { target: orbState.lastPinchBloomTarget };
  }

  if (handPipe.leftPresent) {
    orbState.lastPinchBloomTarget = pinch;
    return { target: pinch };
  }

  return null;
}

function updateFingerHoverHighlights(dt) {
  const hi = geometry.attributes.highlight.array;
  let targetIdx = -1;
  if (orbState.active && orbState.openedViaHand) {
    targetIdx = orbState.particleIndex;
  } else if (handPipe.rightPresent) {
    targetIdx = handPipe.effectivePick;
  }
  const λ = 20;
  for (let i = 0; i < COUNT; i++) {
    const t = targetIdx >= 0 && i === targetIdx ? 1 : 0;
    hi[i] = THREE.MathUtils.damp(hi[i], t, λ, dt);
  }
  geometry.attributes.highlight.needsUpdate = true;
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

  runHandLandmarker(dt);
  updateFingerHoverHighlights(dt);
  updateOrb(arr, dt, tShader, buildHandBloomDrive());

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
