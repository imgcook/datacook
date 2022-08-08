export function mix(...mixins: Array<any>): any {
  class Mix {
    constructor() {
      for (const mixin of mixins) {
        copyProperties(this, new mixin()); // 拷贝实例属性
      }
    }
  }
  for (const mixin of mixins) {
    copyProperties(Mix, mixin); // 拷贝静态属性
    copyProperties(Mix.prototype, mixin.prototype); // 拷贝原型属性
  }

  return Mix;
}

// 深拷贝
function copyProperties(target: any, source: any) {
  for (const key of Reflect.ownKeys(source)) {
    if ( key !== 'constructor' && key !== 'prototype' && key !== 'name') {
      const desc = Object.getOwnPropertyDescriptor(source, key);
      Object.defineProperty(target, key, desc);
    }
  }
}
export function applyMixins(derivedCreator: any, baseCreators: any[]): void {
  baseCreators.forEach((baseCreator) => {
    Object.getOwnPropertyNames(baseCreator.prototype).forEach((name) => {
      derivedCreator.prototype[name] = baseCreator.prototype[name];
    });
  });
}
