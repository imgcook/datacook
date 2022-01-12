export function loadBetaModule() {
  if (typeof (globalThis as any)?.window === 'object') {
    throw new TypeError('"rand/beta" is unavailable at browser environment.');
  }
  return import('./beta');
}
