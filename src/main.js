import * as THREE from 'three';

/**
 * Visual sizing aimed at later hand-tracking "catch" + photo expansion:
 * keep sprites large enough for forgiving overlap checks in screen space.
 * Pair `pointScale` with particle depth (~34); tune min/max px if grabs feel tight.
 */
const PARTICLE_HAND_TARGET = {
  pointScale: 1050,
  minDiameterPx: 42,
  maxDiameterPx: 450,
};

const VERTEX = /* glsl */ `
  attribute vec3 color;
  attribute float phase;
  uniform float time;
  uniform float uPixelRatio;
  uniform float uPointScale;
  uniform float uPointPxMin;
  uniform float uPointPxMax;
  varying vec3 vColor;
  varying float vTwinkle;

  void main() {
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

  void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float len = length(c);
    if (len > 0.5) discard;
    float core = 1.0 - smoothstep(0.0, 0.38, len);
    float glow = exp(-len * 8.5) * 0.22;
    float alpha = (core * 0.88 + glow * 0.38) * vTwinkle * 0.78;
    gl_FragColor = vec4(vColor * (core + glow * 0.42), alpha);
  }
`;

function prefersReducedMotion() {
  return typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function modSafe(a, n) {
  return ((a % n) + n) % n;
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

/** Seeds horizontal spread + depth anchors; vertical motion is computed each frame. */
function seedParticlesHorizontal(base, cam, count) {
  const z0 = -34;
  const dist = -z0;
  const halfV = Math.tan(THREE.MathUtils.degToRad(cam.fov * 0.5));
  const halfH = halfV * dist;
  const halfW = halfH * cam.aspect;
  const margin = 0.72;
  const COLS = 5;
  const ROWS = 4;

  for (let i = 0; i < count; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const u = (col + 0.5) / COLS;
    const gx = (u - 0.5) * 2 * halfW * margin + (Math.random() - 0.5) * 2;
    const rowSkew = ((row + 0.5) / ROWS - 0.5) * halfW * 0.12;
    const i3 = i * 3;
    base[i3] = gx + rowSkew;
    base[i3 + 2] = z0 + (Math.random() - 0.5) * 2.5;
  }
}

function syncCameraAspect(cam) {
  const w = Math.max(1, window.innerWidth || 1);
  const h = Math.max(1, window.innerHeight || 1);
  cam.aspect = w / h;
  cam.updateProjectionMatrix();
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

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colorsFromCategories(particleCategories), 3));
geometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));
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
  seedParticlesHorizontal(basePositions, camera, COUNT);
}
window.addEventListener('resize', resize);
resize();

const motionScale = reducedMotion ? 0.35 : 1;
const fallSpeed = reducedMotion ? 1.2 : 2.35;

function tick() {
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

    const flowX = mouse.x * 0.55 + Math.sin(tShader * 0.055 + seed) * 0.35;

    const progress = modSafe(phases[i] * span + wallT * fallSpeed, span);
    const y = top - progress;

    arr[i3] = bx + Math.sin(tShader * 0.07 + seed * 1.7 + flowX * 0.08) * 0.55 + mouse.x * 0.9;
    arr[i3 + 1] =
      y +
      Math.cos(tShader * 0.06 + seed * 1.1) * 0.22 +
      mouse.y * 0.55;
    arr[i3 + 2] =
      bz + Math.sin(tShader * 0.045 + phases[i] * 8) * 0.35 + Math.cos(tShader * 0.03 + seed) * 0.12;
  }

  posAttr.needsUpdate = true;

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
